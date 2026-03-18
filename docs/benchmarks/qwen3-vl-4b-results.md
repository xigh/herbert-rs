# Benchmark : Qwen3-VL-4B — 4-engine comparison

**Date** : 2026-03-17
**Machine** : beast — AMD Ryzen 9 7900 (12C/24T, Zen4, AVX-512), 96 GB DDR5, Ubuntu 24.04
**Model** : Qwen3-VL-4B-Instruct — Dense VL, 36 layers, hidden=2560, 32 heads, 8 KV heads

## Engines testés

| Engine | Version | Quantisations | KV cache | Notes |
|--------|---------|---------------|----------|-------|
| **Herbert-rs** | 0.1.0 | bf16-avx512, int8-avx512, q4 | int8 (défaut) | Kernels SIMD hand-written |
| **llama.cpp** | b8391 | BF16, Q8_0, Q4_K_M | f16 + q8_0 | AVX-512 + VNNI + BF16 |
| **HF Transformers** | 5.3.0 | BF16 | f32 | Qwen3VLForConditionalGeneration |
| **vLLM-CPU** | 0.17.2rc1 | BF16 | 32GB | Build from source |

**ONNX Runtime** : éliminé — pas de modèle ONNX pré-converti pour Qwen3-VL-4B.

---

## 1. Decode — tokens/s

### Herbert vs llama.cpp (comparaison fair : kv-q8)

| Context | Herbert Q4 | llama Q4 kv-q8 | Herbert Int8 | llama Q8 kv-q8 | Herbert BF16 | llama BF16 kv-q8 |
|--------:|-----------:|---------------:|-------------:|---------------:|-------------:|-----------------:|
| ~64 | 20.3 | **22.2** | 13.2 | **13.6** | 7.1 | **7.5** |
| ~320 | **20.7** | 20.6 | **13.5** | 13.0 | 7.2 | **7.3** |
| ~640 | **20.3** | 18.9 | **13.3** | 12.3 | **7.1** | 7.1 |
| ~1280 | **19.4** | 16.2 | **12.9** | 11.2 | **7.0** | 6.7 |
| ~3200 | **17.3** | 11.6 | **11.9** | 8.6 | **6.7** | 5.7 |
| ~6400 | **14.6** | 7.7 | **10.6** | 6.4 | **6.3** | 4.7 |
| ~9600 | **12.6** | 6.0 | **9.5** | 5.2 | **5.8** | 3.9 |
| ~12800 | **10.9** | 4.9 | **8.5** | 4.3 | **5.5** | 3.4 |
| ~16000 | **9.8** | — | **7.8** | — | — | — |

### Herbert vs llama.cpp (llama.cpp config par défaut : kv-f16)

| Context | Herbert Q4 | llama Q4 kv-f16 | Herbert Int8 | llama Q8 kv-f16 | Herbert BF16 | llama BF16 kv-f16 |
|--------:|-----------:|----------------:|-------------:|----------------:|-------------:|------------------:|
| ~64 | 20.3 | **22.5** | 13.2 | **13.7** | 7.1 | **7.5** |
| ~320 | 20.7 | **21.7** | **13.5** | 13.4 | 7.2 | **7.4** |
| ~640 | 20.3 | **21.2** | **13.3** | 13.2 | 7.1 | **7.3** |
| ~1280 | 19.4 | **19.8** | **12.9** | 12.6 | 7.0 | **7.1** |
| ~3200 | **17.3** | 16.6 | **11.9** | 11.3 | **6.7** | 6.7 |
| ~6400 | **14.6** | 12.6 | **10.6** | 9.4 | **6.3** | 6.0 |
| ~9600 | **12.6** | 10.7 | **9.5** | 8.2 | **5.8** | 5.5 |
| ~12800 | **10.9** | 9.0 | **8.5** | 7.1 | **5.5** | 5.0 |
| ~16000 | **9.8** | 7.9 | **7.8** | 6.3 | — | 4.6 |

### Tous engines — decode à pp1000

| Engine | Quant | Decode (t/s) |
|--------|-------|-------------:|
| **Herbert** | Q4 | **20.3** |
| llama.cpp | Q4_K_M kv-f16 | 21.2 |
| Herbert | Int8 | 13.3 |
| llama.cpp | Q8_0 kv-f16 | 13.2 |
| Herbert | BF16 | 7.1 |
| llama.cpp | BF16 kv-f16 | 7.3 |
| vLLM-CPU | BF16 | 6.5 |
| HF Transformers | BF16 | 5.3 |

### Tous engines — decode à pp10000

| Engine | Quant | Decode (t/s) |
|--------|-------|-------------:|
| **Herbert** | Q4 | **12.6** |
| llama.cpp | Q4_K_M kv-f16 | 10.7 |
| Herbert | Int8 | 9.5 |
| llama.cpp | Q8_0 kv-f16 | 8.2 |
| Herbert | BF16 | 5.8 |
| vLLM-CPU | BF16 | 5.5 |
| llama.cpp | BF16 kv-f16 | 5.5 |
| HF Transformers | BF16 | 2.3 |

---

## 2. Prefill — tokens/s

### Herbert vs llama.cpp (kv-f16)

| Context | Herbert Q4 | llama Q4 kv-f16 | Herbert Int8 | llama Q8 kv-f16 | Herbert BF16 | llama BF16 kv-f16 |
|--------:|-----------:|----------------:|-------------:|----------------:|-------------:|------------------:|
| ~64 | 142 | **203** | 115 | **178** | 77 | **177** |
| ~320 | 167 | **206** | 119 | **179** | 81 | **189** |
| ~640 | 161 | **195** | 117 | **170** | 79 | **181** |
| ~1280 | 148 | **184** | 111 | **160** | 76 | **170** |
| ~3200 | 110 | **152** | 88 | **130** | 65 | **137** |
| ~6400 | 77 | **117** | 67 | **96** | 54 | **105** |
| ~9600 | 65 | **96** | 56 | **78** | 46 | **85** |
| ~12800 | 51 | **81** | 47 | **67** | 41 | **72** |
| ~16000 | 43 | **69** | 40 | **57** | — | **61** |

**llama.cpp domine le prefill** sur le 4B dense (~1.3-1.6× Herbert). Même pattern que le 0.6B dense — le batched matmul fait la différence sur les modèles denses.

### vLLM-CPU et HF Transformers — prefill

| Context | vLLM-CPU BF16 | HF Transfo BF16 |
|--------:|--------------:|----------------:|
| ~64 | 249 | 178 |
| ~320 | 325 | 258 |
| ~640 | **484** | 256 |
| ~1280 | **421** | 237 |
| ~3200 | **270** | 204 |
| ~6400 | **206** | 171 |
| ~9600 | **211** | 148 |
| ~12800 | **213** | 133 |
| ~16000 | **214** | 120 |

**vLLM-CPU a le meilleur prefill** (484 t/s à pp640), suivi de Transformers puis llama.cpp.
Intéressant : le prefill vLLM se stabilise à ~213 t/s au-delà de 10K tokens.

---

## 3. Notes techniques

### Comparaison avec les autres modèles

Le 4B dense se comporte comme le 0.6B (Herbert gagne en decode long context, llama.cpp gagne en prefill),
contrairement au 30B MoE où Herbert dominait aussi le prefill. C'est cohérent : sur les modèles denses,
le batched matmul de llama.cpp est avantageux pour le prefill. Sur les MoE, les optimisations
expert batching de Herbert prennent le dessus.

### Crossover point Herbert vs llama.cpp (kv-f16)

| Quant | Herbert gagne en decode à partir de... |
|-------|---------------------------------------|
| Q4 | ~3200 tokens |
| Int8/Q8 | ~320 tokens |
| BF16 | ~3200 tokens |

---

## 4. Résumé

### Classement global (usage typique : chat avec contexte de 1K-10K tokens)

1. **Herbert Q4** — meilleur decode long context
2. **llama.cpp Q4_K_M kv-f16** — meilleur prefill, bon decode court
3. **Herbert Int8** — bon decode, prefill moyen
4. **llama.cpp Q8_0 kv-f16** — équilibré
5. **Herbert BF16** — decode supérieur en long context
6. **llama.cpp BF16 kv-f16** — prefill rapide
7. **vLLM-CPU BF16** — meilleur prefill global, decode faible
8. **HF Transformers BF16** — lent partout
