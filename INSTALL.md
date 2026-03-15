# Installation

## Prerequisites

- **Rust** (stable, latest recommended): [rustup.rs](https://rustup.rs/)

### macOS (Metal backend)

- **Xcode Command Line Tools** — required for pre-compiled Metal shaders (`.metallib`)

```bash
xcode-select --install
```

If Xcode CLI tools are not available, the build falls back to runtime MSL compilation (slower first launch, but works).

### Linux (Vulkan backend)

- **Vulkan SDK** — provides `glslc` (GLSL → SPIR-V compiler) and Vulkan headers

```bash
# Ubuntu/Debian
sudo apt install vulkan-tools libvulkan-dev glslc

# Or install the full LunarG Vulkan SDK:
# https://vulkan.lunarg.com/sdk/home
```

The build script looks for `glslc` in `$VULKAN_SDK/bin/` or `$PATH`.

- **GPU drivers** with Vulkan 1.3 support (NVIDIA, AMD, Intel)

## Building

```bash
git clone https://github.com/xigh/herbert-rs.git
cd herbert-rs
cargo build --release
```

Binaries are in `target/release/`:
- `herbert-cli` — CLI for interactive chat and single-shot inference
- `herbert-server` — HTTP API server (Anthropic Messages API compatible)

### Build options

The build auto-detects platform capabilities:
- **macOS**: compiles Metal shaders (MSL 3.1, MSL 4.0 if SDK supports it)
- **Linux x86-64**: compiles GLSL shaders to SPIR-V, detects AVX-512/VNNI/BF16

## Downloading models

Models are hosted on HuggingFace:

```bash
pip install huggingface_hub
hf download Qwen/Qwen3-0.6B
hf download Qwen/Qwen3-4B
```

Models are cached in `~/.cache/huggingface/hub/`. Pass the snapshot path to `--model`:

```bash
./target/release/herbert-cli \
  --model ~/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/<hash>/ \
  --prompt "Hello"
```

## Quick test

```bash
# Auto-detect best backend
./target/release/herbert-cli --model <path> --prompt "What is 2+2?" --verbose

# List available backends
./target/release/herbert-cli --backend help

# List available GPUs (Vulkan/Linux)
./target/release/herbert-cli --gpu list
```

## Troubleshooting

### Metal: shader compilation error (struct redefinition)

Make sure you're on the latest version. If the error persists, ensure Xcode CLI tools are installed (`xcode-select --install`) — this enables pre-compiled `.metallib` and avoids the runtime concatenation path.

### Vulkan: `glslc` not found

Install the Vulkan SDK or set `VULKAN_SDK` to point to your installation:

```bash
export VULKAN_SDK=/path/to/vulkan-sdk
cargo build --release
```

### Vulkan: wrong GPU selected

Use `--gpu list` to see available devices, then `--gpu <index>` to select:

```bash
./target/release/herbert-cli --gpu list
./target/release/herbert-cli --gpu 1000 --model <path> --prompt "test"
```

Index convention: `0..N` = discrete GPUs, `1000+N` = all devices (for iGPU access).
