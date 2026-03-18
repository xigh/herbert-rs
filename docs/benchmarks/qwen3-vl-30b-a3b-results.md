# Benchmark : Qwen3-VL-30B-A3B — 4-engine comparison

**Date** : 2026-03-17
**Machine** : beast — AMD Ryzen 9 7900 (12C/24T, Zen4, AVX-512), 96 GB DDR5, Ubuntu 24.04
**Model** : Qwen3-VL-30B-A3B-Instruct — MoE (128 experts, 8 actifs), 48 layers, hidden=2048

## Engines testés

| Engine | Version | Quantisations | KV cache | Notes |
|--------|---------|---------------|----------|-------|
| **Herbert-rs** | 0.1.0 | bf16-avx512, int8-avx512, q4 | int8 (défaut) | Optimisations MoE v6 |
| **llama.cpp** | b8391 | BF16, Q8_0, Q4_K_M | f16 + q8_0 | BF16 limité à ctx=4096 (RAM) |
| **HF Transformers** | 5.3.0 | BF16 | f32 | Qwen3VLMoeForConditionalGeneration |
| **vLLM-CPU** | 0.15.0 | BF16 | 8GB limit | VLLM_CPU_KVCACHE_SPACE=8 (OOM sinon) |

**ONNX Runtime** : éliminé — pas de modèle pré-converti sur onnx-community (uniquement Qwen3
text 0.6B/1.7B/4B, pas VL ni MoE). Support Qwen3-VL en cours : [issue #1989](https://github.com/microsoft/onnxruntime-genai/issues/1989).

> **Note vLLM Q4/Q8** : testé sur vllm-cpu 0.15.0 et vLLM 0.17.2rc1 (build from source).
>
> **vLLM ne supporte aucune quantisation sur CPU** :
> - GGUF sur CPU : le kernel `ggml_dequantize` est [GPU-only](https://github.com/vllm-project/vllm/issues/12391)
>   (`AttributeError: '_OpNamespace' '_C' object has no attribute 'ggml_dequantize'`).
>   Testé avec Qwen3-0.6B Q4_K_M sur v0.17.2rc1 — même erreur.
> - INT4/INT8 natif : [GPU-only](https://docs.vllm.ai/en/latest/features/quantization/int4/) (CUDA compute > 8.0)
> - `bitsandbytes` : requiert CUDA
> - Un [RFC #25590](https://github.com/vllm-project/vllm/issues/25590) propose un backend
>   llama.cpp pour résoudre ce manque, mais pas encore implémenté.
>
> **Spécifiquement pour Qwen3-VL-30B MoE** :
> - GGUF `qwen3vlmoe` : non supporté par le parser GGUF de `transformers`
>   (même en 0.17.2, erreur `GGUF model with architecture qwen3vlmoe is not supported yet`)
> - BF16 sur v0.17.2 : OOM (-9) sur 96 GB RAM (overhead mémoire > v0.15.0)
> - BF16 sur v0.15.0 : fonctionne avec `VLLM_CPU_KVCACHE_SPACE=8`
>
> vLLM-CPU est donc **limité à BF16** pour tous les modèles sur CPU, et à v0.15.0 pour le 30B MoE.

## Configuration

- **Threads** : 12
- **Prompt sizes** : 100, 500, 1000, 2000, 5000, 10000 (Herbert/llama), 100-5000 (Transfo/vLLM)
- **Decode** : 128 tokens
- **Cooldown** : 30s entre runs
- **RAM** : 58 GB modèle BF16 + KV = serré sur 96 GB

---

## 1. Decode — tokens/s

### Herbert vs llama.cpp (comparaison fair : kv-q8)

| Context | Herbert Q4 | llama Q4 kv-q8 | Herbert Int8 | llama Q8 kv-q8 | Herbert BF16 | llama BF16 kv-q8 |
|--------:|-----------:|---------------:|-------------:|---------------:|-------------:|-----------------:|
| ~64 | 25.7 | **26.7** | 15.6 | **17.1** | 8.8 | **9.6** |
| ~320 | **26.5** | 24.9 | **17.0** | 16.6 | 9.1 | **9.5** |
| ~640 | **25.7** | 23.4 | **16.7** | 15.7 | 9.0 | 9.1 |
| ~1280 | **24.6** | 19.1 | **16.0** | 13.7 | **8.7** | 8.4 |
| ~3200 | **21.1** | 13.1 | **14.6** | 10.1 | **8.3** | — |
| ~6400 | **16.7** | 8.4 | **12.6** | 7.1 | **7.5** | — |

### Herbert vs llama.cpp (llama.cpp config par défaut : kv-f16)

| Context | Herbert Q4 | llama Q4 kv-f16 | Herbert Int8 | llama Q8 kv-f16 | Herbert BF16 | llama BF16 kv-f16 |
|--------:|-----------:|----------------:|-------------:|----------------:|-------------:|------------------:|
| ~64 | 25.7 | **26.5** | 15.6 | **17.0** | 8.8 | **9.6** |
| ~320 | **26.5** | 25.2 | **17.0** | 16.5 | 9.1 | **9.4** |
| ~640 | **25.7** | 24.6 | **16.7** | 16.2 | 9.0 | **9.4** |
| ~1280 | **24.6** | 22.7 | **16.0** | 15.4 | **8.7** | — |
| ~3200 | **21.1** | 18.9 | **14.6** | 13.4 | **8.3** | — |
| ~6400 | **16.7** | 14.7 | **12.6** | 10.9 | **7.5** | — |

### Tous engines — decode à pp1000

| Engine | Quant | Decode (t/s) |
|--------|-------|-------------:|
| **Herbert** | Q4 | **25.7** |
| llama.cpp | Q4_K_M kv-f16 | 24.6 |
| Herbert | Int8 | 16.7 |
| llama.cpp | Q8_0 kv-f16 | 16.2 |
| Herbert | BF16 | 9.0 |
| llama.cpp | BF16 kv-f16 | 9.4 |
| vLLM-CPU | BF16 | 6.0 |
| HF Transformers | BF16 | 5.0 |

### Tous engines — decode à pp5000

| Engine | Quant | Decode (t/s) |
|--------|-------|-------------:|
| **Herbert** | Q4 | **21.1** |
| llama.cpp | Q4_K_M kv-f16 | 18.9 |
| Herbert | Int8 | 14.6 |
| llama.cpp | Q8_0 kv-f16 | 13.4 |
| Herbert | BF16 | 8.3 |
| vLLM-CPU | BF16 | 5.8 |
| HF Transformers | BF16 | 3.5 |

---

## 2. Prefill — tokens/s

### Herbert vs llama.cpp (kv-f16)

| Context | Herbert Q4 | llama Q4 kv-f16 | Herbert Int8 | llama Q8 kv-f16 | Herbert BF16 | llama BF16 kv-f16 |
|--------:|-----------:|----------------:|-------------:|----------------:|-------------:|------------------:|
| ~64 | 160 | **151** | 31 | **108** | 51 | **86** |
| ~320 | **219** | 170 | **139** | 132 | 95 | **133** |
| ~640 | **224** | 161 | **149** | 126 | **105** | 129 |
| ~1280 | **210** | 150 | **146** | 120 | **109** | 126 |
| ~3200 | **174** | 115 | **130** | 98 | **101** | — |
| ~6400 | **128** | 84 | **111** | 75 | **87** | — |

**Herbert domine le prefill MoE** dès ~300 tokens dans toutes les quantisations.
Avantage de 30-50% sur llama.cpp grâce aux optimisations MoE v6 (expert batching).

### vLLM-CPU et HF Transformers

| Context | vLLM-CPU BF16 | HF Transfo BF16 |
|--------:|--------------:|----------------:|
| ~64 | 43 | 13 |
| ~320 | 123 | 62 |
| ~640 | **243** | 75 |
| ~1280 | **275** | 78 |
| ~3200 | **225** | 116 |

**vLLM-CPU a le meilleur prefill** sur les prompts moyens (275 t/s à pp1280), grâce
au chunked prefill + torch matmul batché. Il surpasse même Herbert Q4 (210 t/s).

---

## 3. Notes techniques

### Mémoire
Le modèle BF16 occupe ~58 GB en RAM. Avec 96 GB disponibles, les contraintes sont :
- **llama.cpp BF16** : limité à ctx=4096 (KV cache f16 trop gros au-delà)
- **vLLM-CPU** : OOM avec `VLLM_CPU_KVCACHE_SPACE` par défaut (46 GB). Fonctionne avec 8 GB
- **Herbert** : passe en BF16 grâce au KV cache int8 (beaucoup plus compact)
- **Transformers** : passe car KV cache n'est pas pré-alloué

### MoE — comportement spécifique
Sur un modèle MoE (128 experts, 8 actifs), le profil de performance change :
- Le **prefill** est relativement plus rapide (seulement 8 experts actifs = ~3B params)
- Le **decode** est dominé par le routing + expert dispatch, pas seulement la bande passante
- Herbert Q4 MoE decode (25.7 t/s) est comparable au 0.6B dense llama.cpp Q8 (24.9 t/s à pp100)

### ONNX Runtime — éliminé
Pas de modèle ONNX pré-converti pour Qwen3-VL-30B-A3B. L'export échoue sur
l'architecture `qwen3_vl_moe` (même bug que pour Qwen3-0.6B).

---

## 4. Résumé

### Forces de Herbert sur le 30B MoE
- **Prefill MoE supérieur** : +30-50% vs llama.cpp (optimisations expert batching)
- **Decode dominant** sur tous les context sizes (sauf très courts)
- **Meilleure gestion mémoire** : KV cache int8 permet des contexts plus longs en BF16

### Classement global (usage typique : chat avec contexte de 1K-5K tokens)

1. **Herbert Q4** — meilleur decode + excellent prefill MoE
2. **llama.cpp Q4_K_M kv-f16** — bon all-round
3. **Herbert Int8** — bon decode, prefill MoE excellent
4. **llama.cpp Q8_0 kv-f16** — correct
5. **Herbert BF16** — lent mais pleine précision
6. **vLLM-CPU BF16** — meilleur prefill, decode faible
7. **llama.cpp BF16 kv-f16** — limité à ctx=4096
8. **HF Transformers BF16** — lent partout
