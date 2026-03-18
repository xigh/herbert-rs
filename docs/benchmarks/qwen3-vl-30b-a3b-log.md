# Benchmark Log : Qwen3-VL-30B-A3B — Procédure détaillée

**Date** : 2026-03-17
**Machine** : beast (192.168.1.39) — AMD Ryzen 9 7900, 96 GB DDR5

## Modèle

- **Qwen3-VL-30B-A3B-Instruct** : MoE Vision-Language
- 128 experts, 8 actifs par token (~3B params actifs)
- 48 layers, hidden=2048, 32 heads, 4 KV heads
- Snapshot : `9c4b90e1e4ba969fd3b5378b57d966d725f1b86c`

## Step 1 — Conversion GGUF

```bash
SNAP=$(ls -d ~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/*/)

# BF16 (57 GB, 2min53)
python3 convert_hf_to_gguf.py "$SNAP" --outtype bf16 --outfile ~/benchmark-gguf/qwen3-vl-30b-a3b-bf16.gguf

# Q8_0 (31 GB, 2min38)
python3 convert_hf_to_gguf.py "$SNAP" --outtype q8_0 --outfile ~/benchmark-gguf/qwen3-vl-30b-a3b-q8_0.gguf

# Q4_K_M (18 GB, 2min58 quantize)
llama-quantize ~/benchmark-gguf/qwen3-vl-30b-a3b-bf16.gguf ~/benchmark-gguf/qwen3-vl-30b-a3b-q4_k_m.gguf Q4_K_M
```

**Attention disque** : les 3 GGUF totalisent 105 GB. Le disque était plein à 100% après conversion.
BF16 GGUF supprimé après benchmark llama-server (résultats déjà collectés).

## Step 2 — llama-server (6 configs)

```bash
llama-server -m <gguf> -t 12 -c <ctx> --port 8090 -np 1 [-ctk q8_0 -ctv q8_0]
```

- Q4_K_M kv-f16, kv-q8 (ctx=16384)
- Q8_0 kv-f16, kv-q8 (ctx=16384)
- BF16 kv-f16, kv-q8 (ctx=4096 — RAM limitée)
- 16:50 → 17:15 (~25 min)

## Step 3 — Herbert (3 backends)

```bash
herbert-cli --model <dir> --backend <q4|int8-avx512|bf16-avx512> \
    --prompt <text> --max-tokens 128 --ignore-eos --nothink --temperature 0 \
    --num-threads 12 --verbose
```

- Q4 : ~6 min
- Int8-avx512 : ~7 min
- BF16-avx512 : ~10 min
- 17:15 → 17:35 (~20 min)

## Step 4 — HF Transformers

```python
from transformers import Qwen3VLMoeForConditionalGeneration
model = Qwen3VLMoeForConditionalGeneration.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16)
```

**Pièges** :
- `AutoModelForCausalLM` ne supporte pas `Qwen3VLMoeConfig` — erreur explicite
- Il faut utiliser la classe directe `Qwen3VLMoeForConditionalGeneration`
- Modèle rechargé à chaque prompt size (58 GB, ~15s de chargement)

5 prompt sizes (100-5000), ~15 min total.

## Step 5 — vLLM-CPU

```python
from vllm import LLM
llm = LLM(model=MODEL_DIR, dtype="bfloat16", max_model_len=8192, enforce_eager=True)
```

**Pièges** :
- OOM kill (-9) avec `VLLM_CPU_KVCACHE_SPACE` par défaut (46 GB)
- Solution : `VLLM_CPU_KVCACHE_SPACE=8` (8 GB de KV cache)
- Fonctionne en venv `bench-vllm-cpu` (pas `bench-transformers`)

5 prompt sizes (100-5000), ~7 min total.

## Step 6 — ONNX Runtime

**Éliminé** : pas de modèle pré-converti pour Qwen3-VL-30B-A3B MoE.

## Durée totale

| Phase | Durée |
|-------|-------|
| Conversion GGUF (3 quants) | ~9 min |
| llama-server (6 configs) | ~25 min |
| Herbert (3 backends) | ~20 min |
| HF Transformers | ~15 min |
| vLLM-CPU | ~7 min |
| Cooldowns | ~15 min |
| **Total** | **~1h30** |

## Cleanup

```bash
rm ~/benchmark-gguf/qwen3-vl-30b-a3b-bf16.gguf  # supprimé pendant le bench (disque plein)
# Q8_0 et Q4_K_M gardés pour éventuelle ré-exécution
```
