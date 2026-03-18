# herbert-rs

A local LLM inference engine written from scratch in Rust, with hand-written SIMD kernels and GPU compute shaders. No GGML, no llama.cpp — every matrix multiply, attention kernel, and quantization routine is implemented directly.

## Supported Models

### Qwen3 — Text (dense)

| Model | Params | Tested |
|-------|--------|--------|
| `Qwen/Qwen3-0.6B` | 0.6B | yes |
| `Qwen/Qwen3-1.7B` | 1.7B | yes |
| `Qwen/Qwen3-4B` | 4B | yes |
| `Qwen/Qwen3-8B` | 8B | yes |
| `Qwen/Qwen3-14B` | 14B | yes |
| `Qwen/Qwen3-32B` | 32B | yes |

### Qwen3 — Text (MoE)

| Model | Params | Active | Tested |
|-------|--------|--------|--------|
| `Qwen/Qwen3-30B-A3B` | 30B | 3B | yes |
| `Qwen/Qwen3-235B-A22B` | 235B | 22B | - |

### Qwen3-VL — Vision-Language

| Model | Params | Tested |
|-------|--------|--------|
| `Qwen/Qwen3-VL-2B-Instruct` | 2B | yes |
| `Qwen/Qwen3-VL-4B-Instruct` | 4B | yes |
| `Qwen/Qwen3-VL-8B-Instruct` | 8B | yes |
| `Qwen/Qwen3-VL-32B-Instruct` | 32B | yes |
| `Qwen/Qwen3-VL-30B-A3B-Instruct` | 30B (MoE, 3B active) | yes |
| `Qwen/Qwen3-VL-235B-A22B-Instruct` | 235B (MoE, 22B active) | - |

### Mistral3 / Ministral3

| Model | Params | Tested |
|-------|--------|--------|
| `mistralai/Ministral-3-3B-Instruct-2512-BF16` | 3B | yes |
| `mistralai/Ministral-3-8B-Instruct-2512-BF16` | 8B | yes |
| `mistralai/Ministral-3-14B-Instruct-2512-BF16` | 14B | yes |
| `mistralai/Mistral-Small-3.2-24B-Instruct-2506` | 24B | yes |
| `mistralai/Magistral-Small-2509` | 24B | yes |
| `mistralai/Devstral-Small-2-24B-Instruct-2512` | 24B | yes |

> **Note:** Some Mistral models ship with `tekken.json` instead of `tokenizer.json`. The BF16 variants (recommended) include `tokenizer.json` directly.

## Downloading Models

Models are hosted on HuggingFace. Use the [`hf` CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli) to download:

```bash
# Install the HuggingFace CLI
pip install huggingface_hub

# Download a model (example: Qwen3-4B)
hf download Qwen/Qwen3-4B

# Download a vision-language model
hf download Qwen/Qwen3-VL-8B-Instruct

# Download a Mistral model (use BF16 variant for tokenizer.json)
hf download mistralai/Ministral-3-8B-Instruct-2512-BF16
```

Models are cached in `~/.cache/huggingface/hub/`. Pass the snapshot path to `--model`:

```bash
herbert-cli --model ~/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/<hash>
```

## Features

### Quantization

| Format | Weights | KV Cache | Notes |
|--------|---------|----------|-------|
| BF16 | 2 bytes/param | — | Full precision baseline |
| INT8 | 1 byte/param | 1 byte/elem | Per-channel symmetric, VNNI acceleration |
| Q4 | 0.5 bytes/param | — | Per-group (group_size=32), pre-interleaved nibbles |

Mixed-precision supported: Q4, INT8, and BF16 weights in the same model.

### CPU Backends

- **BF16 scalar** — pure f32, no SIMD (reference/verification)
- **BF16 AVX-512** — `vdpbf16ps` native BF16 dot-product (Zen4, Sapphire Rapids)
- **INT8 AVX-512** — `VPDPBUSD` VNNI with fused QKV and gate+up projections
- **Q4 AVX-512** — pre-interleaved nibble layout, huge pages, fused tile-local scales

All critical inner loops are hand-written x86-64 assembly (`.S` files).

### GPU Backends

- **Metal** (macOS) — 92 compute shaders covering Q4/INT8/BF16 matvec, matmul, flash attention, MoE, vision encoding, KV cache management
- **Vulkan** (Linux) — 31 GLSL compute shaders with cooperative matrix support, portable across AMD/NVIDIA/Intel. Use `--gpu list` to enumerate devices, `--gpu N` to select (0=first discrete, 1000+=global index for iGPU)

### Inference

- Streaming token generation with UTF-8 multi-byte handling
- KV cache quantization (BF16, INT8)
- Thinking mode control for reasoning models (`--nothink`, `--think-budget`)
- Repetition loop detection
- Token sampling: temperature, top-k, top-p, greedy
- Mixture-of-Experts with batched expert dispatch
- Vision-Language support with multi-image input

## Binaries

### `herbert-cli` — CLI

Interactive chat or single-shot inference.

```bash
# Single-shot
herbert-cli --model <path> --prompt "What is 2+2?"

# Interactive chat
herbert-cli --model <path>

# With a system prompt
herbert-cli --model <path> --system "You are a helpful assistant."

# Vision (Qwen3-VL or Pixtral models)
herbert-cli --model <path> --image photo.jpg --prompt "Describe this image."

# Greedy decoding (temperature=0)
herbert-cli --model <path> --temperature 0 --prompt "Hello"

# Show stats after generation
herbert-cli --model <path> --prompt "Hello" --verbose

# Choose a specific backend
herbert-cli --model <path> --backend metal-q4
herbert-cli --model <path> --backend help   # list available backends

# GPU selection (Vulkan)
herbert-cli --gpu list                       # list available GPUs
herbert-cli --gpu 0 --model <path> --backend vulkan-bf16   # first discrete GPU (default)
herbert-cli --gpu 1000 --model <path> --backend vulkan-bf16 # iGPU (global index)

# Tool calling
herbert-cli --model <path> --tools
```

**Sampling options:** `--temperature` (default 0.4), `--top-k` (default 40), `--top-p` (default 0.9), `--max-tokens` (default 2048)

**Chat commands:** `/help`, `/config`, `/temp`, `/topk`, `/topp`, `/think`, `/nothink`, `/tools`, `/image`, `/stats`, `/arch`, `/clear`, `/quit`

**Built-in tools** (with `--tools`): `get_datetime`, `calculate`, `list_directory`, `read_file`

### `herbert-server` — HTTP API

Anthropic Messages API compatible server with SSE streaming.

```bash
herbert-server --model <path> --addr 0.0.0.0:3000
herbert-server --model <path> --addr 0.0.0.0:3000 --api-key mysecretkey
```

Endpoints:
- `POST /v1/messages` — chat completion (streaming SSE or JSON)
- `POST /v1/messages/count_tokens` — token counting
- `POST /v1/tokenize` — tokenization
- `GET /v1/metrics` — performance metrics

### `herbert-desktop` — Desktop App

Native desktop application built with Tauri 2 and Vue 3.

- Multi-conversation chat with sidebar
- Streaming with markdown rendering and syntax highlighting
- Image support with drag & drop and encoding progress
- Per-message performance stats
- Model loading with progress feedback
- Settings panel for sampling parameters and backend selection

## Design

Most inference engines optimize for prefill throughput (batched GEMM). Herbert takes a different approach: it is built around the assumption that inference performance is limited by memory bandwidth, not compute. It prioritizes decode speed, which is what determines the user experience in interactive use.

At decode time, the bottleneck is memory bandwidth — each generated token requires reading the full KV cache. Herbert addresses this with an INT8 KV cache that halves the bandwidth requirement compared to FP16, using hand-written VNNI kernels that avoid the dequantization overhead seen in other implementations.

In practice:
- **Prefill**: not the primary optimization target yet (batched matmul is in progress). Currently 1.5-2x behind llama.cpp on dense models
- **Decode (short context)**: on par with llama.cpp
- **Decode (long context)**: performance improves as context grows, because KV cache bandwidth becomes the dominant cost — and Herbert reduces that cost

On Mixture-of-Experts models, Herbert's expert batching also improves prefill, leading to better performance across the board.

These results are consistent across all four tested models and architectures (dense, VL, MoE).

### Methodology

Every kernel optimization is validated empirically using:
- **Hardware performance counters** (AMD Zen4 PMC via `perf_event_open` + `rdpmc` fast-path) — cycle-precise, core-pinned, multi-pass measurement of L1/L2/L3 cache behavior, retired instructions, and branch mispredictions
- **Memory bandwidth sweeps** — working set sizes from L1 (48KB) through L2 (1.25MB) to DRAM (64MB+) to establish theoretical bandwidth ceilings
- **Wall-clock throughput** — end-to-end prefill and decode measurements with controlled cooldown periods between runs

## Benchmarks

CPU-only benchmarks on an AMD Ryzen 9 7900 (12C/24T, AVX-512, 96 GB DDR5), comparing Herbert with llama.cpp, HF Transformers, vLLM-CPU, and ONNX Runtime.

### Decode throughput (tokens/s) — what the user sees

Herbert's KV cache quantization (INT8 by default) gives it an increasing advantage over llama.cpp as context length grows. On short contexts, llama.cpp is slightly faster; on longer contexts (1K+ tokens), Herbert pulls ahead.

**Qwen3-0.6B — Q4 decode**

| Context | Herbert Q4 | llama.cpp Q4 | vLLM BF16 | HF Transformers |
|--------:|-----------:|-------------:|----------:|----------------:|
| ~100 | 110 | 110 | 30 | 27 |
| ~1000 | 101 | 97 | 29 | 22 |
| ~5000 | 70 | 56 | 25 | 11 |
| ~10000 | 52 | 35 | 18 | 7 |
| ~16000 | 29 | 17 | 14 | 3 |

**Qwen3-VL-30B-A3B (MoE) — Q4 decode**

| Context | Herbert Q4 | llama.cpp Q4 | vLLM BF16 | HF Transformers |
|--------:|-----------:|-------------:|----------:|----------------:|
| ~300 | 27 | 25 | 6 | 5 |
| ~1300 | 25 | 23 | 6 | 5 |
| ~3200 | 21 | 19 | 6 | 4 |
| ~6400 | 17 | 15 | — | — |

On MoE models, Herbert also wins on prefill thanks to its expert batching optimizations (moe-v6).

### Prefill throughput (tokens/s)

llama.cpp has faster prefill on dense models (~1.5-2x) due to its batched GEMM. vLLM-CPU has the best prefill overall thanks to chunked prefill + torch matmul.

**Qwen3-0.6B — prefill at ~1000 tokens**

| Engine | t/s |
|--------|----:|
| vLLM-CPU BF16 | 1989 |
| HF Transformers BF16 | 1292 |
| llama.cpp BF16 | 1057 |
| Herbert Q4 | 751 |

### Detailed results

Full benchmarks with all prompt sizes, quantizations, and KV cache configurations are in [`docs/benchmarks/`](docs/benchmarks/):

- [Qwen3-0.6B](docs/benchmarks/qwen3-0.6b-results.md) — dense 0.6B, 5 engines including ONNX Runtime
- [Qwen3-VL-4B](docs/benchmarks/qwen3-vl-4b-results.md) — dense VL 4B
- [Ministral-3 3B](docs/benchmarks/ministral-3-3b-results.md) — dense text 3B (Mistral)
- [Qwen3-VL-30B-A3B](docs/benchmarks/qwen3-vl-30b-a3b-results.md) — MoE 30B (3B active)

## Building

```bash
cargo build --release
```

The build auto-detects available CPU features (AVX-512, VNNI, AVX-512 BF16) and compiles the appropriate assembly kernels. Metal shaders are compiled on macOS, Vulkan shaders on Linux.

### Platform Support

| Platform | CPU Backends | GPU Backend |
|----------|-------------|-------------|
| macOS (Apple Silicon) | BF16 scalar/Neo | Metal |
| Linux (x86-64) | BF16, BF16-AVX2/512, INT8-AVX2/512, Q4-AVX2/512 | Vulkan |

## License

MIT — see [LICENSE](LICENSE)
