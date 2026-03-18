# Benchmark Log : Qwen3-0.6B — Procédure détaillée

**Date** : 2026-03-17
**Machine** : beast (192.168.1.39) — AMD Ryzen 9 7900, 96 GB DDR5, Ubuntu 24.04 kernel 6.8.0

## Step 1 — Pré-requis (déjà en place)

### llama.cpp
- Commit : `740a447fc` (vulkan: allow graphics queue only through env var)
- Build : `cmake -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_AVX512=ON -DGGML_AVX512_BF16=ON -DGGML_AVX512_VNNI=ON`
- Path : `~/llama.cpp/llama.cpp/build/bin/`

### Herbert-rs
- Version : 0.1.0
- Build : `cargo build --release` (opt-level=3, lto=true)
- Path : `~/herbert-rs/target/release/herbert-cli`

### Modèle Qwen3-0.6B
- Source : `~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca/`
- GGUF convertis : `~/benchmark-gguf/qwen3-0.6b-{bf16,q8_0,q4_k_m}.gguf`

## Step 2 — Setup des venvs Python

```bash
# Transformers (torch CPU + transformers + accelerate)
python3 -m venv ~/venvs/bench-transformers
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers accelerate
# → torch 2.10.0+cpu, transformers 5.3.0

# ONNX Runtime (onnxruntime + optimum + olive)
python3 -m venv ~/venvs/bench-onnx
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install onnxruntime optimum[onnxruntime] transformers
pip install olive-ai onnxruntime-genai
# → onnxruntime 1.24.3, optimum 2.1.0, olive-ai 0.11.0

# vLLM-CPU (éliminé — pas de vrai backend CPU)
python3 -m venv ~/venvs/bench-vllm-cpu
pip install vllm
# → vllm 0.17.1 — device='cpu' non supporté, éliminé
```

## Step 3 — Export ONNX

### Tentative 1 : optimum-cli
```bash
optimum-cli export onnx --model <snapshot> --task text-generation-with-past ~/benchmark-onnx/qwen3-0.6b/
```
Export réussi mais **runtime bugué** : `InvalidArgument: Got invalid dimensions for input: past_key_values.9.value index: 3 Got: 64 Expected: 128`

Bug GQA dans optimum-onnxruntime : les KV heads (8) sont confondus avec la dimension (128).
Voir : https://github.com/huggingface/optimum/issues/2351

### Tentative 2 : olive auto-opt
```bash
olive auto-opt --model_name_or_path Qwen/Qwen3-0.6B --device cpu --provider CPUExecutionProvider --precision int4
```
Échoue avec `RuntimeError: unordered_map::at` — même bug d'export ONNX sous-jacent.

### Solution : modèle pré-converti onnx-community
```bash
huggingface-cli download onnx-community/Qwen3-0.6B-ONNX --local-dir ~/benchmark-onnx/qwen3-0.6b-community
```
Variantes disponibles : model.onnx (FP32, 301M), model_fp16.onnx (1.2G), model_int8.onnx (590M), model_q4.onnx (877M)

Utilisation directe via `onnxruntime.InferenceSession` avec gestion manuelle du KV cache
(contournement du bug optimum).

## Step 4 — Benchmarks

### 4.1 — llama.cpp via llama-bench (prefill bulk)
```bash
llama-bench -m <gguf> -p 100,500,...,25000 -n 128 -t 12 -r 3 -o json
```
- BF16 : 10:54 → 11:16 (22 min)
- Q8_0 : 11:16 → 11:39 (23 min)
- Q4_K_M : 11:39 → 12:01 (22 min)

**Limitation** : llama-bench ne donne qu'un seul chiffre decode (sans contexte).

### 4.2 — llama.cpp via llama-server (decode par context size)
```bash
llama-server -m <gguf> -t 12 -c 32768 --port 8090 -np 1 [-ctk q8_0 -ctv q8_0]
curl http://127.0.0.1:8090/completion -d '{"prompt":"...","n_predict":128,"temperature":0,"cache_prompt":false}'
```
6 runs (3 quants × 2 KV configs) × 9 prompt sizes = 54 mesures

- 13:22 → 14:09 (47 min total)

### 4.3 — Herbert
```bash
herbert-cli --model <dir> --backend <backend> --prompt <text> --max-tokens 128 \
    --ignore-eos --temperature 0 --nothink --num-threads 12 --verbose
```
3 backends × 9 prompt sizes = 27 runs

- bf16-avx512 : 12:02 → 12:09 (7 min)
- int8-avx512 : 12:10 → 12:17 (7 min)
- q4 : 12:18 → 12:24 (6 min)

### 4.4 — HF Transformers
```python
model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16)
# Prefill timing + decode loop timing séparés
```
9 prompt sizes, modèle rechargé à chaque run.

- 12:25 → 12:29 (4 min)

### 4.5 — vLLM-CPU
**Attention** : le package standard `vllm` (v0.17.1) n'a PAS de backend CPU.
Il faut installer le package séparé `vllm-cpu` (v0.15.0) :
```bash
pip install vllm-cpu --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
```

Pièges rencontrés :
1. `pip install vllm` → pas de `device='cpu'`, CUDA path, crash
2. Multiprocessing spawn → nécessite `if __name__ == "__main__"` guard
3. Heredoc `<< PYEOF` → le spawn re-exécute `<stdin>` comme fichier → crash. Utiliser un `.py`
4. EngineCore meurt après chaque generate (normal en offline mode)

Script final : un seul process Python avec toutes les prompt sizes en boucle.

```python
# ~/vllm_bench.py — avec guard __main__
if __name__ == "__main__":
    llm = LLM(model=MODEL_DIR, dtype="bfloat16", max_model_len=32768, enforce_eager=True)
    for pp in PP_SIZES:
        outputs = llm.generate([prompt], params)
```
9 prompt sizes en un seul run (modèle chargé une fois).

### 4.6 — ONNX Runtime
```python
session = ort.InferenceSession(model_path, sess_options=opts)
# Prefill avec empty KV cache, decode loop avec KV passthrough
```
4 quants (FP32, FP16, Int8, Q4) × 9 prompt sizes = 36 runs

- 15:16 → 15:55 (39 min)

## Step 5 — Durée totale

| Phase | Durée |
|-------|-------|
| llama-bench (3 quants) | 67 min |
| llama-server (6 configs) | 47 min |
| Herbert (3 backends) | 20 min |
| HF Transformers | 4 min |
| ONNX Runtime (4 quants) | 39 min |
| Cooldowns (30s × ~120 runs) | ~60 min |
| **Total** | **~4h** |

## Step 6 — Cleanup

```bash
# GGUF : gardés pour les prochains modèles
# ONNX community : gardé pour les prochains modèles
# ONNX export local (buggé) : supprimé
rm -rf ~/benchmark-onnx/qwen3-0.6b/
# Venvs : gardés pour les prochains modèles
```
