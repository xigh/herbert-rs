# Benchmark comparatif : Herbert-rs vs llama.cpp vs Transformers vs ONNX Runtime

Benchmark CPU-only sur **beast** (AMD Ryzen 9 7900, 12C/24T, AVX-512, 96 GB DDR5).

## Engines comparés

| Engine | Quantisations | KV cache | Notes |
|--------|---------------|----------|-------|
| **Herbert-rs** | bf16-avx512, int8-avx512, q4 | int8 (défaut) | Kernels SIMD hand-written, charge safetensors |
| **llama.cpp** | BF16, Q8_0, Q4_K_M | f16 (défaut), q8_0 | Compilé avec AVX-512 + VNNI + BF16 |
| **HF Transformers** | BF16 | f32 (PyTorch) | torch 2.10.0+cpu, matmul batché MKL |
| **ONNX Runtime** | FP32, FP16, Int8, Q4 | f32/f16 | Session manuelle (optimum buggé sur Qwen3) |
| **vLLM-CPU** | BF16 | auto | Package séparé `vllm-cpu` (pas `vllm`) |
| ~~SGLang~~ | — | — | Éliminé : pas de backend CPU |

### Résultats

Les résultats détaillés sont dans `docs/benchmarks/` :

#### Qwen3-0.6B (Dense, 0.6B params)
- [`qwen3-0.6b-results.md`](benchmarks/qwen3-0.6b-results.md) — tableaux comparatifs
- [`qwen3-0.6b-log.md`](benchmarks/qwen3-0.6b-log.md) — procédure et log détaillé

#### Qwen3-VL-4B (Dense VL, 4B params)
- [`qwen3-vl-4b-results.md`](benchmarks/qwen3-vl-4b-results.md) — tableaux comparatifs

#### Ministral-3 3B (Dense text, 3B params)
- [`ministral-3-3b-results.md`](benchmarks/ministral-3-3b-results.md) — tableaux comparatifs

#### Qwen3-VL-30B-A3B (MoE, 30B params, 3B actifs)
- [`qwen3-vl-30b-a3b-results.md`](benchmarks/qwen3-vl-30b-a3b-results.md) — tableaux comparatifs
- [`qwen3-vl-30b-a3b-log.md`](benchmarks/qwen3-vl-30b-a3b-log.md) — procédure et log détaillé

## Modèles testés

| # | Modèle | Type | Params | Notes |
|---|--------|------|--------|-------|
| 1 | Qwen3-0.6B | Dense text | 0.6B | Plus petit, baseline |
| 2 | Qwen3-VL-2B-Instruct | Dense VL | 2B | Vision-Language |
| 3 | Qwen3-VL-4B-Instruct | Dense VL | 4B | Vision-Language |
| 4 | Qwen3-VL-8B-Instruct | Dense VL | 8B | Vision-Language |
| 5 | Qwen3-VL-30B-A3B-Instruct | MoE VL | 30B (3B actifs) | Mixture-of-Experts |
| 6 | Qwen3-VL-Embedding-2B | Dense VL | 2B | Embedding model |
| 7 | Ministral-3-3B-Instruct-2512-BF16 | Dense text | 3B | Mistral |
| 8 | Ministral-3-8B-Instruct-2512-BF16 | Dense text | 8B | Mistral |
| 9 | Ministral-3-14B-Instruct-2512 | Dense text | 14B | Mistral |
| 10 | Devstral-Small-2-24B-Instruct-2512-FP8 | Dense text | 24B | Mistral, source FP8 |

## Quantisations comparées

| Herbert backend | Équivalent GGUF (llama.cpp) | Description |
|-----------------|----------------------------|-------------|
| `bf16-avx512` | `BF16` | Poids BFloat16, calcul natif |
| `int8-avx512` | `Q8_0` | Quantisation 8-bit symétrique |
| `q4` | `Q4_K_M` | Quantisation 4-bit (K-quant medium) |

## Tailles de prompt testées

Le prefill est testé avec 9 longueurs de contexte croissantes pour observer
le comportement en fonction de la taille du prompt (linéaire ? sous-linéaire ?) :

| Label | Tokens (approx.) | Objectif |
|-------|-------------------|----------|
| `pp100` | ~100 | Baseline, prompt court |
| `pp500` | ~500 | Usage conversationnel typique |
| `pp1000` | ~1 000 | Prompt moyen |
| `pp2000` | ~2 000 | Prompt long / RAG |
| `pp5000` | ~5 000 | Document court |
| `pp10000` | ~10 000 | Document moyen |
| `pp15000` | ~15 000 | Document long |
| `pp20000` | ~20 000 | Stress test mémoire/bande passante |
| `pp25000` | ~25 000 | Limite haute (attention: certains petits modèles ont un ctx < 32K) |

Pour llama.cpp, cela correspond aux flags `-p 100`, `-p 500`, …, `-p 25000`.

Pour Herbert, on utilise un fichier texte pré-tokenisé ou un prompt suffisamment
long pour atteindre le nombre de tokens voulu (généré une fois, réutilisé pour
les deux engines).

**Contexte maximal par modèle** :

| Modèle | ctx_len | Tailles testables |
|--------|---------|-------------------|
| Qwen3-0.6B | 32 768 | toutes (pp100→pp25000) |
| Qwen3-VL-2B | 32 768 | toutes |
| Qwen3-VL-4B | 32 768 | toutes |
| Qwen3-VL-8B | 32 768 | toutes |
| Qwen3-VL-30B-A3B | 32 768 | toutes |
| Qwen3-VL-Emb-2B | 32 768 | toutes |
| Ministral-3-3B | 131 072 | toutes |
| Ministral-3-8B | 131 072 | toutes |
| Ministral-3-14B | 131 072 | toutes |
| Devstral-24B | 131 072 | toutes |

## Métriques mesurées

- **Prefill** : tokens/s pour chaque taille de prompt (pp100 → pp25000)
- **Decode** : tokens/s (génération de 128 tokens, après chaque taille de prefill)
- **Mémoire** : RSS peak en GB
- **Temps de chargement** : secondes

## Machine cible

```
Host:       beast (192.168.1.39)
CPU:        AMD Ryzen 9 7900 (12C/24T, Zen4)
SIMD:       AVX-512 (BF16, VNNI, VBMI2)
RAM:        96 GB DDR5
OS:         Ubuntu 24.04, kernel 6.8.0
```

## Procédure

### 1. Compiler llama.cpp (dernière version)

```bash
cd /home/zexigh/llama.cpp/llama.cpp
git pull
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DGGML_NATIVE=ON \
         -DGGML_AVX512=ON \
         -DGGML_AVX512_BF16=ON \
         -DGGML_AVX512_VNNI=ON
make -j$(nproc)
```

Vérifier :
```bash
./bin/llama-bench --help
./bin/llama-cli --version
```

### 2. Convertir les modèles HuggingFace en GGUF

Les modèles HF sont dans `~/.cache/huggingface/hub/`. Le script
`convert_hf_to_gguf.py` produit un fichier GGUF à partir du snapshot.

**Important** : convertir un modèle à la fois et supprimer le GGUF après le
benchmark pour ne pas saturer le disque.

```bash
HF_CACHE=~/.cache/huggingface/hub
GGUF_DIR=~/benchmark-gguf
mkdir -p $GGUF_DIR

# Exemple pour Qwen3-0.6B :
SNAP=$(ls -d $HF_CACHE/models--Qwen--Qwen3-0.6B/snapshots/*/.)
python3 convert_hf_to_gguf.py "$SNAP" --outtype bf16 --outfile $GGUF_DIR/qwen3-0.6b-bf16.gguf
```

Pour chaque modèle, produire 3 fichiers GGUF :

| Commande outtype | Fichier produit | Quant llama.cpp |
|------------------|----------------|-----------------|
| `--outtype bf16` | `*-bf16.gguf` | BF16 natif |
| `--outtype q8_0` | `*-q8_0.gguf` | Q8_0 (≈ int8) |
| (quantize après) | `*-q4_k_m.gguf` | Q4_K_M (≈ q4) |

Pour Q4_K_M, d'abord convertir en BF16 puis quantiser :
```bash
./build/bin/llama-quantize $GGUF_DIR/qwen3-0.6b-bf16.gguf $GGUF_DIR/qwen3-0.6b-q4_k_m.gguf Q4_K_M
```

### 3. Benchmarker avec llama.cpp (llama-bench)

`llama-bench` accepte plusieurs valeurs `-p` séparées par des virgules :

```bash
LLAMA=~/llama.cpp/llama.cpp/build/bin
PP_SIZES="100,500,1000,2000,5000,10000,15000,20000,25000"

$LLAMA/llama-bench \
    -m $GGUF_DIR/qwen3-0.6b-bf16.gguf \
    -p $PP_SIZES -n 128 \
    -t $(nproc) \
    -r 3 \
    -o json > results/llama-qwen3-0.6b-bf16.json

# Répéter pour q8_0 et q4_k_m
```

Cela produit 9 mesures de prefill + 9 mesures de decode (un par taille de prompt)
en une seule invocation.

Options clés de `llama-bench` :
- `-p N[,N,…]` : longueur(s) du prompt (prefill), virgule-séparées
- `-n N` : tokens à générer (decode)
- `-t N` : nombre de threads
- `-r N` : nombre de répétitions
- `-o json` : sortie JSON pour post-traitement

### 4. Benchmarker avec Herbert

Herbert n'a pas de flag multi-prompt comme llama-bench, donc on itère sur les
tailles de prompt dans un script. On utilise un fichier texte long pré-généré
et on tronque à la taille voulue.

**Préparer le prompt de référence** (une seule fois) :

```bash
# Générer un fichier texte de ~30K tokens (≈120K caractères anglais)
# On peut utiliser un extrait de Wikipedia ou un texte répétitif
python3 -c "
import json, pathlib
# Répéter un paragraphe réaliste pour atteindre ~120K chars
para = 'The history of computing is a fascinating journey through human ingenuity. ' * 20
text = para * 150  # ~120K chars ≈ 30K tokens
pathlib.Path('benchmark-prompt.txt').write_text(text)
print(f'Generated {len(text)} chars')
"
```

**Boucle de benchmark** :

```bash
HERBERT=~/herbert-rs/target/release/herbert-cli
HF_CACHE=~/.cache/huggingface/hub
PP_SIZES=(100 500 1000 2000 5000 10000 15000 20000 25000)

model_dir() {
    ls -d $HF_CACHE/models--${1}--${2}/snapshots/*/
}

for backend in bf16-avx512 int8-avx512 q4; do
    for pp in "${PP_SIZES[@]}"; do
        # Tronquer le prompt à ~pp tokens (≈ pp*4 caractères)
        PROMPT=$(head -c $((pp * 4)) benchmark-prompt.txt)

        $HERBERT \
            --model $(model_dir Qwen Qwen3-0.6B) \
            --backend $backend \
            --prompt "$PROMPT" \
            --max-tokens 128 \
            --temperature 0 \
            --verbose 2>&1 | tee results/herbert-qwen3-0.6b-${backend}-pp${pp}.txt
    done
done
```

### 5. Libérer les GGUF après chaque modèle

```bash
rm -f $GGUF_DIR/qwen3-0.6b-*.gguf
```

Herbert charge directement les safetensors depuis le cache HF, donc pas de
fichier intermédiaire à nettoyer côté Herbert.

### 6. Automatisation

La matrice complète est : **10 modèles × 3 quantisations × 2 engines × 9 tailles de prompt = 540 runs**.

Un script `benchmark_all.sh` itère sur cette matrice. Pour llama.cpp, les 9
tailles sont passées en une seule invocation (`-p 100,500,...,25000`). Pour
Herbert, on boucle sur chaque taille.

## Résultats attendus (template)

Les résultats sont organisés **par modèle**, avec un tableau prefill (9 tailles)
et un tableau decode pour chaque modèle.

### Exemple : Qwen3-0.6B

#### Prefill (tokens/s)

| Prompt (tokens) | Herbert BF16 | Herbert Int8 | Herbert Q4 | llama.cpp BF16 | llama.cpp Q8_0 | llama.cpp Q4_K_M |
|----------------:|-------------|-------------|-----------|----------------|----------------|------------------|
| 100 | — | — | — | — | — | — |
| 500 | — | — | — | — | — | — |
| 1 000 | — | — | — | — | — | — |
| 2 000 | — | — | — | — | — | — |
| 5 000 | — | — | — | — | — | — |
| 10 000 | — | — | — | — | — | — |
| 15 000 | — | — | — | — | — | — |
| 20 000 | — | — | — | — | — | — |
| 25 000 | — | — | — | — | — | — |

#### Decode (tokens/s) — génération de 128 tokens après prefill

| Prompt (tokens) | Herbert BF16 | Herbert Int8 | Herbert Q4 | llama.cpp BF16 | llama.cpp Q8_0 | llama.cpp Q4_K_M |
|----------------:|-------------|-------------|-----------|----------------|----------------|------------------|
| 100 | — | — | — | — | — | — |
| 500 | — | — | — | — | — | — |
| 1 000 | — | — | — | — | — | — |
| 2 000 | — | — | — | — | — | — |
| 5 000 | — | — | — | — | — | — |
| 10 000 | — | — | — | — | — | — |
| 15 000 | — | — | — | — | — | — |
| 20 000 | — | — | — | — | — | — |
| 25 000 | — | — | — | — | — | — |

#### Mémoire peak (GB)

| Quant | Herbert | llama.cpp |
|-------|---------|-----------|
| BF16 | — | — |
| Int8/Q8_0 | — | — |
| Q4/Q4_K_M | — | — |

*(Répéter ce bloc pour chaque modèle)*

### Synthèse comparative (tous modèles)

#### Prefill moyen (tokens/s) — moyenne sur les 9 tailles de prompt

| Modèle | Herbert BF16 | Herbert Int8 | Herbert Q4 | llama.cpp BF16 | llama.cpp Q8_0 | llama.cpp Q4_K_M |
|--------|-------------|-------------|-----------|----------------|----------------|------------------|
| Qwen3-0.6B | — | — | — | — | — | — |
| Qwen3-VL-2B | — | — | — | — | — | — |
| Qwen3-VL-4B | — | — | — | — | — | — |
| Qwen3-VL-8B | — | — | — | — | — | — |
| Qwen3-VL-30B-A3B | — | — | — | — | — | — |
| Qwen3-VL-Emb-2B | — | — | — | — | — | — |
| Ministral-3-3B | — | — | — | — | — | — |
| Ministral-3-8B | — | — | — | — | — | — |
| Ministral-3-14B | — | — | — | — | — | — |
| Devstral-24B | — | — | — | — | — | — |

#### Decode à pp1000 (tokens/s) — point de référence typique

| Modèle | Herbert BF16 | Herbert Int8 | Herbert Q4 | llama.cpp BF16 | llama.cpp Q8_0 | llama.cpp Q4_K_M |
|--------|-------------|-------------|-----------|----------------|----------------|------------------|
| Qwen3-0.6B | — | — | — | — | — | — |
| Qwen3-VL-2B | — | — | — | — | — | — |
| Qwen3-VL-4B | — | — | — | — | — | — |
| Qwen3-VL-8B | — | — | — | — | — | — |
| Qwen3-VL-30B-A3B | — | — | — | — | — | — |
| Qwen3-VL-Emb-2B | — | — | — | — | — | — |
| Ministral-3-3B | — | — | — | — | — | — |
| Ministral-3-8B | — | — | — | — | — | — |
| Ministral-3-14B | — | — | — | — | — | — |
| Devstral-24B | — | — | — | — | — | — |

## Notes

- Le modèle **Devstral-Small-2-24B-FP8** est distribué en FP8. Pour llama.cpp,
  `convert_hf_to_gguf.py` déquantise en BF16 puis on requantise. Pour Herbert,
  vérifier la compatibilité FP8→BF16 du loader.
- Le modèle **Qwen3-VL-30B-A3B** est un MoE (Mixture-of-Experts). Les benchmarks
  MoE sont particulièrement intéressants car Herbert a des optimisations MoE
  spécifiques (`moe-v6` feature).
- Les modèles **VL** (Vision-Language) : pour le benchmark text-only, seul le
  décodeur texte est exercé. Un benchmark vision séparé pourrait être ajouté.
- **Qwen3-VL-Embedding-2B** est un modèle d'embedding, pas de génération.
  Le benchmark mesure uniquement le prefill (pas de decode).
- Herbert quantise à la volée depuis les safetensors (pas de fichier
  intermédiaire pour int8/q4). Première exécution plus lente si pas de cache.
  Utiliser `--no-cache` pour forcer la re-quantisation.
