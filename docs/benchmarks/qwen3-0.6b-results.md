# Benchmark : Qwen3-0.6B — 5-engine comparison

**Date** : 2026-03-17
**Machine** : beast — AMD Ryzen 9 7900 (12C/24T, Zen4, AVX-512), 96 GB DDR5, Ubuntu 24.04

## Engines testés

| Engine | Version | Quantisations | KV cache | Notes |
|--------|---------|---------------|----------|-------|
| **Herbert-rs** | 0.1.0 | bf16-avx512, int8-avx512, q4 | int8 (défaut) | Kernels SIMD hand-written |
| **llama.cpp** | b8391 (740a447fc) | BF16, Q8_0, Q4_K_M | f16 (défaut) + q8_0 | AVX-512 + VNNI + BF16 |
| **HF Transformers** | 5.3.0 | BF16 (torch.bfloat16) | f32 (PyTorch) | torch 2.10.0+cpu |
| **ONNX Runtime** | 1.24.3 | FP32, FP16, Int8, Q4 | f32 / f16 | Modèle onnx-community |
| **vLLM-CPU** | 0.15.0 (vllm-cpu) | BF16 | auto (46 GB) | Package séparé `pip install vllm-cpu` |

## Configuration

- **Threads** : 12 (P-cores uniquement)
- **Prompt sizes** : 100, 500, 1000, 2000, 5000, 10000, 15000, 20000, 25000 tokens cible
- **Decode** : 128 tokens générés après chaque prefill
- **Cooldown** : 30 secondes entre chaque run
- **Herbert** : KV cache int8 par défaut, `--ignore-eos --nothink --temperature 0`
- **llama.cpp** : testé avec KV f16 (défaut) ET KV q8_0 (pour comparaison fair avec Herbert)
- **Transformers** : modèle rechargé à chaque prompt size (pas de cache persistant)
- **ONNX** : session manuelle avec gestion KV cache (optimum-onnxruntime buggé sur Qwen3)

> **Note** : les token counts réels varient légèrement du cible (ex: pp100 → 64-72 tokens réels)
> car le prompt est tronqué par caractères (~4 chars/token). Tous les tableaux reportent les
> **tokens réels** mesurés par chaque engine.

---

## 1. Decode — tokens/s (génération de 128 tokens après prefill)

C'est la métrique UX principale : la vitesse à laquelle l'utilisateur voit les tokens apparaître.

### Herbert vs llama.cpp (comparaison fair : même KV cache q8)

| Context | Herbert BF16 | llama BF16 kv-q8 | Herbert Int8 | llama Q8 kv-q8 | Herbert Q4 | llama Q4 kv-q8 |
|--------:|-------------:|------------------:|-------------:|---------------:|-----------:|---------------:|
| ~64 | 40.5 | **45.0** | 73.6 | **80.8** | **110.2** | 110.0 |
| ~320 | 40.4 | **41.4** | **71.7** | 66.6 | **105.1** | 83.8 |
| ~640 | **39.0** | 37.6 | **68.2** | 57.8 | **101.0** | 71.2 |
| ~1280 | **37.2** | 32.0 | **63.7** | 45.2 | **95.2** | 53.5 |
| ~3200 | **32.8** | 21.7 | **50.9** | 27.4 | **70.0** | 31.1 |
| ~6400 | **28.4** | 14.3 | **40.4** | 16.0 | **52.3** | 18.0 |
| ~9600 | **25.1** | 10.7 | **34.0** | 11.8 | **41.8** | 12.4 |
| ~12800 | **22.4** | 8.5 | **29.8** | 9.2 | **33.4** | 9.5 |
| ~16000 | **20.1** | 6.9 | **25.8** | 7.2 | **29.2** | 7.4 |

**Herbert gagne dès ~300-600 tokens de context** et l'écart se creuse massivement.
À 16K tokens, Herbert est **2.9× à 3.9× plus rapide** que llama.cpp (à KV cache égal q8).

### Herbert vs llama.cpp (llama.cpp en config par défaut : KV f16)

| Context | Herbert BF16 | llama BF16 kv-f16 | Herbert Int8 | llama Q8 kv-f16 | Herbert Q4 | llama Q4 kv-f16 |
|--------:|-------------:|-------------------:|-------------:|----------------:|-----------:|----------------:|
| ~64 | 40.5 | **45.7** | 73.6 | **78.6** | **110.2** | 109.7 |
| ~320 | 40.4 | **44.1** | 71.7 | **76.9** | 105.1 | **105.1** |
| ~640 | 39.0 | **42.9** | 68.2 | **72.8** | **101.0** | 97.0 |
| ~1280 | 37.2 | **39.5** | **63.7** | 63.1 | **95.2** | 81.0 |
| ~3200 | **32.8** | 32.7 | **50.9** | 46.7 | **70.0** | 56.1 |
| ~6400 | **28.4** | 24.3 | **40.4** | 31.4 | **52.3** | 35.3 |
| ~9600 | **25.1** | 20.2 | **34.0** | 24.9 | **41.8** | 27.6 |
| ~12800 | **22.4** | 16.6 | **29.8** | 19.7 | **33.4** | 21.2 |
| ~16000 | **20.1** | 14.1 | **25.8** | 16.2 | **29.2** | 17.1 |

Même avec llama.cpp en KV f16 (avantage : pas de dequant), Herbert prend le dessus à partir de **~3000 tokens** en BF16 et **~1300 tokens** en Int8/Q4.

### Tous engines — decode à pp1000 (point de référence conversationnel)

| Engine | Quant | Decode (t/s) |
|--------|-------|-------------:|
| **Herbert** | Q4 | **101.0** |
| llama.cpp | Q4_K_M kv-f16 | 97.0 |
| Herbert | Int8 | 68.2 |
| llama.cpp | Q8_0 kv-f16 | 72.8 |
| Herbert | BF16 | 39.0 |
| llama.cpp | BF16 kv-f16 | 42.9 |
| ONNX Runtime | Q4 | 30.5 |
| vLLM-CPU | BF16 | 29.4 |
| HF Transformers | BF16 | 21.8 |
| ONNX Runtime | FP32 | 16.8 |

### Tous engines — decode à pp10000 (contexte long)

| Engine | Quant | Decode (t/s) |
|--------|-------|-------------:|
| **Herbert** | Q4 | **52.3** |
| Herbert | Int8 | 40.4 |
| llama.cpp | Q4_K_M kv-f16 | 35.3 |
| llama.cpp | Q8_0 kv-f16 | 31.4 |
| **Herbert** | BF16 | **28.4** |
| llama.cpp | BF16 kv-f16 | 24.3 |
| vLLM-CPU | BF16 | 21.0 |
| HF Transformers | BF16 | 7.3 |
| ONNX Runtime | Q4 | 6.2 |
| ONNX Runtime | FP32 | 5.6 |

---

## 2. Prefill — tokens/s

### Herbert vs llama.cpp (llama.cpp en config par défaut : KV f16)

| Context | Herbert BF16 | llama BF16 kv-f16 | Herbert Int8 | llama Q8 kv-f16 | Herbert Q4 | llama Q4 kv-f16 |
|--------:|-------------:|-------------------:|-------------:|----------------:|-----------:|----------------:|
| ~64 | 522 | **1098** | 511 | **1054** | 678 | **1119** |
| ~320 | 539 | **1352** | 503 | **1181** | 744 | **1192** |
| ~640 | 512 | **1254** | 463 | **1050** | 751 | **1072** |
| ~1280 | 426 | **1057** | 411 | **924** | 578 | **899** |
| ~3200 | 308 | **704** | 301 | **640** | 328 | **600** |
| ~6400 | 185 | **444** | 196 | **402** | 204 | **376** |
| ~9600 | 134 | **332** | 146 | **304** | 155 | **281** |
| ~12800 | 116 | **261** | 118 | **241** | 126 | **220** |
| ~16000 | 87 | **208** | 90 | **194** | 96 | **176** |

**llama.cpp domine le prefill** (~2× en BF16 sur les courts prompts). Cela s'explique par son
implémentation batched matmul (BLAS-style) vs le matvec séquentiel de Herbert.

### vLLM-CPU et HF Transformers — prefill

| Context | vLLM-CPU BF16 | HF Transfo BF16 |
|--------:|--------------:|----------------:|
| ~64 | 908 | 633 |
| ~320 | 1745 | 1223 |
| ~640 | **2476** | 1427 |
| ~1280 | **1989** | 1292 |
| ~3200 | **1124** | 950 |
| ~6400 | **737** | 685 |
| ~9600 | **717** | 515 |
| ~12800 | **709** | 433 |
| ~16000 | **702** | 377 |

**vLLM-CPU a le meilleur prefill** de tous les engines (2476 t/s à pp640), grâce au
chunked prefill (4096 tokens) + torch matmul batché. Il surpasse même Transformers
et llama.cpp. Mais son decode reste moyen (30 t/s).

### ONNX Runtime — prefill

| Context | ONNX FP32 | ONNX FP16 | ONNX Int8 | ONNX Q4 |
|--------:|----------:|----------:|----------:|--------:|
| ~64 | 596 | 590 | 215 | 733 |
| ~320 | 804 | 643 | 650 | 787 |
| ~640 | 746 | 640 | **784** | 776 |
| ~1280 | 616 | 572 | **785** | 662 |
| ~3200 | 452 | 389 | **537** | 445 |
| ~6400 | 280 | 261 | **315** | 279 |
| ~9600 | 216 | 205 | **228** | 217 |
| ~12800 | 154 | 153 | **169** | 155 |
| ~16000 | 126 | 126 | **136** | 126 |

ONNX Int8 est le plus rapide en prefill pour les longs prompts, mais le decode reste
très lent (~2 t/s à 16K) car le KV cache est en FP32.

---

## 3. Notes techniques

### vLLM-CPU
**Attention** : le package standard `pip install vllm` (v0.17.1) n'a **pas** de backend CPU.
Il faut installer le package séparé `vllm-cpu` (v0.15.0) :
```bash
pip install vllm-cpu --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
```

Résultats vLLM-CPU BF16 (prefill et decode séparés via double-pass) :

| Context | Prefill (t/s) | Decode (t/s) |
|--------:|--------------:|-------------:|
| ~64 | 908 | 30.5 |
| ~320 | 1745 | 29.6 |
| ~640 | **2476** | 29.4 |
| ~1280 | 1989 | 28.4 |
| ~3200 | 1124 | 25.2 |
| ~6400 | 737 | 21.0 |
| ~9600 | 717 | 18.1 |
| ~12800 | 709 | 15.8 |
| ~16000 | 702 | 14.0 |

**Prefill : vLLM-CPU a le meilleur prefill de tous les engines** (2476 t/s à pp640),
grâce au chunked prefill (4096 tokens) + torch matmul batché. Il surpasse même
HF Transformers (1427 t/s) et llama.cpp (1254 t/s) à cette taille.

**Decode** : correct (~30 t/s court contexte), entre Transformers et Herbert BF16.
Le decode se dégrade avec le contexte (30 → 14 t/s) mais moins brutalement que
Transformers et ONNX.

**Notes techniques** :
- `enforce_eager=True` nécessaire (pas de torch.compile CPU)
- Le multiprocessing spawn nécessite un guard `if __name__ == "__main__"`
- Le process EngineCore meurt après chaque génération en mode offline (normal)
- KV cache auto-dimensionné à ~47 GB (toute la RAM libre)

### ONNX Runtime — bugs optimum
L'export ONNX de Qwen3 fonctionne (modèles pré-convertis disponibles sur HuggingFace via
`onnx-community/Qwen3-0.6B-ONNX`), mais le runtime `optimum-onnxruntime` a un
[bug GQA/KV cache](https://github.com/huggingface/optimum/issues/2351) qui empêche l'inférence.
Le même bug affecte `olive auto-opt` (`RuntimeError: unordered_map::at`).

**Solution** : utiliser `onnxruntime` directement avec gestion manuelle du KV cache
(voir script de benchmark). Les résultats ci-dessus utilisent cette méthode.

### KV cache — impact critique sur le decode
Le benchmark révèle que le choix du format KV cache a un **impact majeur** sur le decode
en contexte long, plus important que la quantisation des poids :

- **llama.cpp kv-f16** : decode 14.1 t/s à 16K (BF16)
- **llama.cpp kv-q8** : decode 6.9 t/s à 16K (BF16) — **2× plus lent**
- **Herbert kv-int8** : decode 20.1 t/s à 16K (BF16) — **2.9× plus rapide** que llama kv-q8

L'implémentation KV cache quantifié de llama.cpp sur CPU a un overhead de dequantization
qui dépasse le gain en bande passante. Herbert, avec ses kernels VNNI dédiés, ne souffre
pas de ce problème.

### Herbert — caractéristiques
- KV cache int8 par défaut (configurable via `--kv-quant`)
- Charge directement les safetensors (pas de conversion GGUF)
- Quantise à la volée (int8/q4) au chargement
- Pas de batched prefill (matvec séquentiel) — explique le retard en prefill

---

## 4. Résumé

### Forces de Herbert
- **Decode long context** : domine tous les engines à partir de ~1000-3000 tokens
- **KV cache int8 efficace** : aucune pénalité vs les engines concurrents en mode quantifié
- **Q4 decode** : parité avec llama.cpp sur les courts contextes, nettement supérieur au-delà

### Faiblesses de Herbert
- **Prefill** : ~2× plus lent que llama.cpp (pas de batched matmul)
- **Pas de decode ultra-court optimisé** : llama.cpp est ~10-15% plus rapide sous ~500 tokens

### Classement global (usage typique : chat avec contexte de 1K-10K tokens)

1. **Herbert Q4** — meilleur compromis vitesse/qualité
2. **llama.cpp Q4_K_M kv-f16** — bon prefill, decode correct
3. **Herbert Int8** — decode excellent, prefill moyen
4. **llama.cpp Q8_0 kv-f16** — équilibré
5. **Herbert BF16** — decode supérieur, prefill lent
6. **llama.cpp BF16 kv-f16** — prefill rapide, decode moyen
7. **ONNX Runtime Q4** — decode acceptable sur courts contextes
8. **vLLM-CPU BF16** — overhead scheduler, viable pour batching multi-requêtes
9. **HF Transformers BF16** — bon prefill, decode très lent
10. **ONNX Runtime FP32/FP16/Int8** — lent partout
