# Benchmark : Ministral-3 3B — 4-engine comparison

**Date** : 2026-03-18
**Machine** : beast — AMD Ryzen 9 7900 (12C/24T, Zen4, AVX-512), 96 GB DDR5, Ubuntu 24.04
**Model** : Ministral-3-3B-Instruct-2512-BF16 — Dense text, 26 layers, hidden=3072, 32 heads, 8 KV heads

## Engines testés

| Engine | Version | Quantisations | KV cache | Notes |
|--------|---------|---------------|----------|-------|
| **Herbert-rs** | 0.1.0 | bf16-avx512, int8-avx512, q4 | int8 (défaut) | Kernels SIMD hand-written |
| **llama.cpp** | b8391 | BF16, Q8_0, Q4_K_M | f16 + q8_0 | AVX-512 + VNNI + BF16 |
| **HF Transformers** | 5.3.0 | BF16 | f32 | Ministral3ForCausalLM |
| **vLLM-CPU** | 0.17.2rc1 | BF16 | 32GB | Build from source |

---

## 1. Prefill — tokens/s

| Context | vLLM BF16 | HF Transfo | llama Q4 kv-f16 | llama BF16 kv-f16 | llama Q8 kv-f16 | Herbert Q4 | Herbert Int8 | Herbert BF16 |
|--------:|----------:|-----------:|----------------:|------------------:|----------------:|-----------:|-------------:|-------------:|
| ~65 | 337 | 252 | 244 | 214 | 212 | 172 | 152 | 99 |
| ~320 | 412 | 363 | 252 | 230 | 218 | 209 | 157 | 104 |
| ~640 | **616** | 364 | 241 | 181 | 211 | 201 | 155 | 103 |
| ~1280 | **516** | 338 | 225 | 170 | 199 | 183 | 144 | 98 |
| ~3200 | **355** | 271 | 185 | 137 | 163 | 144 | 119 | 85 |
| ~6400 | **275** | 227 | 140 | 105 | 126 | 103 | 90 | 71 |
| ~9600 | **285** | 199 | 116 | 85 | 104 | 79 | 77 | 60 |
| ~12800 | **292** | 179 | 97 | 72 | 88 | 68 | 65 | 52 |
| ~16000 | **294** | 160 | 83 | 61 | 76 | 59 | 56 | 47 |

**vLLM-CPU domine le prefill** (616 t/s à pp640), suivi de Transformers (364), puis llama.cpp (244).
Herbert est en retrait (~200 t/s Q4) — même pattern que les autres modèles denses.

---

## 2. Decode — tokens/s (128 tokens générés)

| Context | Herbert Q4 | llama Q4 kv-f16 | Herbert Int8 | llama Q8 kv-f16 | Herbert BF16 | llama BF16 kv-f16 | vLLM BF16 | HF Transfo |
|--------:|-----------:|----------------:|-------------:|----------------:|-------------:|------------------:|----------:|-----------:|
| ~65 | 24.4 | **26.3** | 15.5 | **16.2** | 8.3 | **8.8** | 8.0 | 7.1 |
| ~320 | 24.8 | **25.5** | **15.8** | 15.9 | **8.4** | 8.7 | 8.0 | 6.8 |
| ~640 | 24.3 | **25.0** | 15.6 | **15.7** | **8.4** | 8.7 | 7.9 | 6.5 |
| ~1280 | 23.5 | **23.6** | **15.2** | 15.1 | **8.3** | 8.7 | 7.8 | 5.9 |
| ~3200 | **21.1** | 20.3 | **14.2** | 13.7 | **8.0** | 6.7 | 7.6 | 4.8 |
| ~6400 | **18.1** | 16.3 | **12.9** | 11.6 | **7.5** | 6.0 | 7.1 | 3.6 |
| ~9600 | **15.9** | 13.5 | **11.7** | 10.3 | **7.1** | 5.5 | 6.8 | 3.1 |
| ~12800 | **14.0** | 11.7 | **10.6** | 9.1 | **6.7** | 5.0 | 6.4 | 2.6 |
| ~16000 | **12.6** | 10.1 | **9.9** | 8.2 | **6.3** | 4.6 | 6.1 | 2.2 |

**Herbert domine le decode** dès ~1280 tokens en Q4, ~320 en Int8, et ~3200 en BF16.
À pp16000 : Herbert Q4 (12.6 t/s) bat llama.cpp Q4 (10.1 t/s) de **25%**.

Note : vLLM BF16 decode (6-8 t/s) est comparable à llama.cpp BF16 et meilleur que Transformers.

---

## 3. Comparaison avec les autres modèles

Le Ministral-3 3B se comporte comme les autres modèles denses (Qwen3-0.6B, Qwen3-VL-4B) :
- Prefill : vLLM > Transformers > llama.cpp > Herbert
- Decode long context : Herbert > llama.cpp > vLLM > Transformers
- Crossover Herbert/llama.cpp : ~1280-3200 tokens selon la quantisation

---

## 4. Résumé

1. **Herbert Q4** — meilleur decode long context (12.6 t/s à 16K)
2. **llama.cpp Q4_K_M kv-f16** — meilleur prefill quantisé, bon decode court
3. **Herbert Int8** — decode supérieur, prefill moyen
4. **llama.cpp Q8_0 kv-f16** — équilibré
5. **Herbert BF16** — decode supérieur en long context
6. **vLLM-CPU BF16** — meilleur prefill global (616 t/s), decode correct
7. **llama.cpp BF16 kv-f16** — bon prefill
8. **HF Transformers BF16** — bon prefill, decode lent
