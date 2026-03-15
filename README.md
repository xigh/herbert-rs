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
- **Vulkan** (Linux) — 31 GLSL compute shaders with cooperative matrix support, portable across AMD/NVIDIA/Intel

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
