//! Metal compute pipeline management.
//!
//! Loads the pre-compiled .metallib (from build.rs) or falls back to
//! runtime compilation from embedded MSL source strings.
//!
//! Adapted for Qwen3: removed conv pipelines (compute_bx, conv_causal, conv_gate_decode),
//! added Q4 pipelines (q4_matvec, q4_matmul).

use herbert_core::error::{HerbertError, Result};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::*;

/// A single Metal compute pipeline.
pub struct ComputePipeline {
    pub state: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

/// All compute pipelines for the Metal backend (Qwen3).
pub struct Pipelines {
    // Normalization
    pub rms_norm: ComputePipeline,
    pub rms_norm_batch: ComputePipeline,
    pub head_rms_norm: ComputePipeline,
    pub rms_norm_residual: ComputePipeline,
    // Activation & utilities
    pub softmax: ComputePipeline,
    pub argmax_stage1: ComputePipeline,
    pub argmax_stage2: ComputePipeline,
    pub swiglu: ComputePipeline,
    pub embedding: ComputePipeline,
    pub residual_add: ComputePipeline,
    pub bias_add_batch: ComputePipeline,
    pub scaled_add: ComputePipeline,
    pub copy_buffer: ComputePipeline,
    pub fused_gate_up_swiglu: ComputePipeline,
    // Matvec (decode, M=1)
    pub bf16_matvec: ComputePipeline,
    pub f32_matvec: ComputePipeline,
    pub int8_matvec: ComputePipeline,
    pub q4_matvec: ComputePipeline,
    pub q4_matvec_residual_add: ComputePipeline,
    pub q4_matvec_residual_add_8row: ComputePipeline,
    pub q4_matvec_qkv: ComputePipeline,
    pub q4_matvec_qkv_normed: ComputePipeline,
    // Matmul (prefill, M>1)
    pub bf16_matmul: ComputePipeline,
    pub f32_matmul: ComputePipeline,
    pub int8_matmul: ComputePipeline,
    pub q4_matmul: ComputePipeline,
    // Attention
    pub attention_decode: ComputePipeline,
    pub attention_decode_gqa: ComputePipeline,
    pub attention_decode_flash_tile: ComputePipeline,
    pub attention_decode_flash_reduce: ComputePipeline,
    pub attention_prefill: ComputePipeline,
    pub attention_prefill_v2_gqa: ComputePipeline,
    pub attention_prefill_v3_tiled: ComputePipeline,
    pub attention_prefill_v4_simdgroup: ComputePipeline,
    pub attention_prefill_v5_half_kv: ComputePipeline,
    // RoPE
    pub rope_single: ComputePipeline,
    pub rope_batch: ComputePipeline,
    pub rope_kv_append: ComputePipeline,
    // KV cache
    pub kv_cache_append: ComputePipeline,
    pub kv_cache_append_batch: ComputePipeline,
    // INT8 KV cache
    pub kv_cache_append_i8: ComputePipeline,
    pub kv_cache_append_batch_i8: ComputePipeline,
    pub kv_cache_quantize_half_to_i8: ComputePipeline,
    pub attention_decode_flash_tile_i8: ComputePipeline,
    pub head_norm_rope_kv_append_i8: ComputePipeline,
    // MoE
    pub moe_gather: ComputePipeline,
    pub moe_scatter_add: ComputePipeline,
    pub moe_softmax_topk: ComputePipeline,
    // Fused MoE expert kernels
    pub moe_fused_gate_up_swiglu_q4: ComputePipeline,
    pub moe_fused_gate_up_swiglu_int8: ComputePipeline,
    pub matvec_scaled_add_q4: ComputePipeline,
    pub matvec_scaled_add_int8: ComputePipeline,
    pub matvec_scaled_add_bf16: ComputePipeline,
    // Fused attention kernels
    pub head_norm_rope: ComputePipeline,
    pub head_norm_rope_batch: ComputePipeline,
    pub head_norm_rope_kv_append: ComputePipeline,
    // Tiled matmul (prefill)
    pub q4_matmul_tiled: ComputePipeline,
    pub bf16_matmul_tiled: ComputePipeline,
    // Batched MoE (contiguous expert weights, no CPU sync)
    pub moe_batched_gate_up_swiglu_q4: ComputePipeline,
    pub moe_batched_gate_up_swiglu_int8: ComputePipeline,
    pub moe_batched_gate_up_swiglu_bf16: ComputePipeline,
    pub moe_batched_down_q4: ComputePipeline,
    pub moe_batched_down_int8: ComputePipeline,
    pub moe_batched_down_bf16: ComputePipeline,
    pub moe_reduce: ComputePipeline,
    pub moe_reduce_residual: ComputePipeline,
    // Prefill MoE (zero-sync, tiled with counting sort)
    pub moe_softmax_topk_batch: ComputePipeline,
    pub moe_prefill_sort: ComputePipeline,
    pub moe_prefill_tiled_gate_up_swiglu_q4: ComputePipeline,
    pub moe_prefill_tiled_gate_up_swiglu_int8: ComputePipeline,
    pub moe_prefill_tiled_gate_up_swiglu_bf16: ComputePipeline,
    pub moe_prefill_tiled_down_q4: ComputePipeline,
    pub moe_prefill_tiled_down_int8: ComputePipeline,
    pub moe_prefill_tiled_down_bf16: ComputePipeline,
    pub moe_prefill_reduce_residual: ComputePipeline,
    // Fused prefill MoE down + reduce + residual (atomic accumulation)
    pub moe_prefill_tiled_down_residual_q4: ComputePipeline,
    pub moe_prefill_tiled_down_residual_int8: ComputePipeline,
    pub moe_prefill_tiled_down_residual_bf16: ComputePipeline,
    // H2O eviction
    pub h2o_score_probe_half: ComputePipeline,
    pub h2o_score_probe_i8: ComputePipeline,
    pub kv_cache_compact_half: ComputePipeline,
    pub kv_cache_compact_i8: ComputePipeline,
    pub kv_cache_compact_scales: ComputePipeline,
    // DeepStack (VL prefill: add vision features to image token positions)
    pub deepstack_add: ComputePipeline,
    // Vision encoder
    pub layer_norm_batch: ComputePipeline,
    pub gelu_batch: ComputePipeline,
    pub vision_split_qkv: ComputePipeline,
    pub vision_spatial_merge: ComputePipeline,
    // FlashDecoding v2: double-buffered K/V loads (async copy + overlap)
    pub attention_decode_flash_tile_v2: Option<ComputePipeline>,
    pub attention_decode_flash_tile_i8_v2: Option<ComputePipeline>,
    // Metal 4 cooperative_tensor 32×32 pipelines (Apple11+ only, None on older GPUs)
    pub q4_matmul_tiled_coop32: Option<ComputePipeline>,
    pub bf16_matmul_tiled_coop32: Option<ComputePipeline>,
    pub attention_prefill_v6_coop32: Option<ComputePipeline>,
    // Metal 4 MetalPerformancePrimitives matmul2d (Neural Accelerators, Apple11+ only)
    pub q4_matmul_tiled_mpp: Option<ComputePipeline>,
    pub bf16_matmul_tiled_mpp: Option<ComputePipeline>,
    pub attention_prefill_v7_mpp: Option<ComputePipeline>,
    // Metal 4 half-precision accumulation (FP16 throughput 2×)
    pub rms_norm_half: Option<ComputePipeline>,
    pub rms_norm_batch_half: Option<ComputePipeline>,
    // Q4 matvec v2: 8-row, arithmetic dequant, uint4 loads (decode optimisation)
    pub q4_matvec_v2: Option<ComputePipeline>,
    pub q4_matvec_residual_add_v2: Option<ComputePipeline>,
}

// Embed all MSL shader sources as strings for runtime compilation fallback
macro_rules! include_msl {
    ($name:expr) => {
        include_str!(concat!("../shaders/", $name, ".metal"))
    };
}

// Q4 tiled matmul: classic tiling (default) or simdgroup_matrix (feature-gated)
#[cfg(feature = "simdgroup-matmul")]
const Q4_MATMUL_TILED_SRC: &str = include_str!("../shaders/q4_matmul_tiled_simdgroup.metal");
#[cfg(not(feature = "simdgroup-matmul"))]
const Q4_MATMUL_TILED_SRC: &str = include_str!("../shaders/q4_matmul_tiled.metal");

impl Pipelines {
    /// Create all compute pipelines from the pre-compiled metallib or MSL sources.
    pub fn new(device: &ProtocolObject<dyn MTLDevice>, metal4: bool) -> Result<Self> {
        let library = Self::load_library(device)?;

        // Load Metal 4 library if available (Metal4 GPU + MSL 4.0 compiled shaders)
        let m4_library = if metal4 {
            Self::load_m4_library(device).ok()
        } else {
            None
        };

        Ok(Self {
            // Normalization
            rms_norm: Self::make_pipeline(device, &library, "rms_norm")?,
            rms_norm_batch: Self::make_pipeline(device, &library, "rms_norm_batch")?,
            head_rms_norm: Self::make_pipeline(device, &library, "head_rms_norm")?,
            rms_norm_residual: Self::make_pipeline(device, &library, "rms_norm_residual")?,
            // Activation & utilities
            softmax: Self::make_pipeline(device, &library, "softmax")?,
            argmax_stage1: Self::make_pipeline(device, &library, "argmax_stage1")?,
            argmax_stage2: Self::make_pipeline(device, &library, "argmax_stage2")?,
            swiglu: Self::make_pipeline(device, &library, "swiglu")?,
            embedding: Self::make_pipeline(device, &library, "embedding")?,
            residual_add: Self::make_pipeline(device, &library, "residual_add")?,
            bias_add_batch: Self::make_pipeline(device, &library, "bias_add_batch")?,
            scaled_add: Self::make_pipeline(device, &library, "scaled_add")?,
            copy_buffer: Self::make_pipeline(device, &library, "copy_buffer")?,
            fused_gate_up_swiglu: Self::make_pipeline(device, &library, "fused_gate_up_swiglu")?,
            // Matvec
            bf16_matvec: Self::make_pipeline(device, &library, "bf16_matvec")?,
            f32_matvec: Self::make_pipeline(device, &library, "f32_matvec")?,
            int8_matvec: Self::make_pipeline(device, &library, "int8_matvec")?,
            q4_matvec: Self::make_pipeline(device, &library, "q4_matvec")?,
            q4_matvec_residual_add: Self::make_pipeline(device, &library, "q4_matvec_residual_add")?,
            q4_matvec_residual_add_8row: Self::make_pipeline(device, &library, "q4_matvec_residual_add_8row")?,
            q4_matvec_qkv: Self::make_pipeline(device, &library, "q4_matvec_qkv")?,
            q4_matvec_qkv_normed: Self::make_pipeline(device, &library, "q4_matvec_qkv_normed")?,
            // Matmul
            bf16_matmul: Self::make_pipeline(device, &library, "bf16_matmul")?,
            f32_matmul: Self::make_pipeline(device, &library, "f32_matmul")?,
            int8_matmul: Self::make_pipeline(device, &library, "int8_matmul")?,
            q4_matmul: Self::make_pipeline(device, &library, "q4_matmul")?,
            // Attention
            attention_decode: Self::make_pipeline(device, &library, "attention_decode")?,
            attention_decode_gqa: Self::make_pipeline(device, &library, "attention_decode_gqa")?,
            attention_decode_flash_tile: Self::make_pipeline(device, &library, "attention_decode_flash_tile")?,
            attention_decode_flash_reduce: Self::make_pipeline(device, &library, "attention_decode_flash_reduce")?,
            attention_prefill: Self::make_pipeline(device, &library, "attention_prefill")?,
            attention_prefill_v2_gqa: Self::make_pipeline(device, &library, "attention_prefill_v2_gqa")?,
            attention_prefill_v3_tiled: Self::make_pipeline(device, &library, "attention_prefill_v3_tiled")?,
            attention_prefill_v4_simdgroup: Self::make_pipeline(device, &library, "attention_prefill_v4_simdgroup")?,
            attention_prefill_v5_half_kv: Self::make_pipeline(device, &library, "attention_prefill_v5_half_kv")?,
            // RoPE
            rope_single: Self::make_pipeline(device, &library, "rope_single")?,
            rope_batch: Self::make_pipeline(device, &library, "rope_batch")?,
            rope_kv_append: Self::make_pipeline(device, &library, "rope_kv_append")?,
            // KV cache
            kv_cache_append: Self::make_pipeline(device, &library, "kv_cache_append")?,
            kv_cache_append_batch: Self::make_pipeline(device, &library, "kv_cache_append_batch")?,
            // INT8 KV cache
            kv_cache_append_i8: Self::make_pipeline(device, &library, "kv_cache_append_i8")?,
            kv_cache_append_batch_i8: Self::make_pipeline(device, &library, "kv_cache_append_batch_i8")?,
            kv_cache_quantize_half_to_i8: Self::make_pipeline(device, &library, "kv_cache_quantize_half_to_i8")?,
            attention_decode_flash_tile_i8: Self::make_pipeline(device, &library, "attention_decode_flash_tile_i8")?,
            head_norm_rope_kv_append_i8: Self::make_pipeline(device, &library, "head_norm_rope_kv_append_i8")?,
            // MoE
            moe_gather: Self::make_pipeline(device, &library, "moe_gather")?,
            moe_scatter_add: Self::make_pipeline(device, &library, "moe_scatter_add")?,
            moe_softmax_topk: Self::make_pipeline(device, &library, "moe_softmax_topk")?,
            // Fused MoE expert kernels
            moe_fused_gate_up_swiglu_q4: Self::make_pipeline(device, &library, "moe_fused_gate_up_swiglu_q4")?,
            moe_fused_gate_up_swiglu_int8: Self::make_pipeline(device, &library, "moe_fused_gate_up_swiglu_int8")?,
            matvec_scaled_add_q4: Self::make_pipeline(device, &library, "matvec_scaled_add_q4")?,
            matvec_scaled_add_int8: Self::make_pipeline(device, &library, "matvec_scaled_add_int8")?,
            matvec_scaled_add_bf16: Self::make_pipeline(device, &library, "matvec_scaled_add_bf16")?,
            // Fused attention kernels
            head_norm_rope: Self::make_pipeline(device, &library, "head_norm_rope")?,
            head_norm_rope_batch: Self::make_pipeline(device, &library, "head_norm_rope_batch")?,
            head_norm_rope_kv_append: Self::make_pipeline(device, &library, "head_norm_rope_kv_append")?,
            // Tiled matmul (prefill)
            q4_matmul_tiled: Self::make_pipeline(device, &library, "q4_matmul_tiled")?,
            bf16_matmul_tiled: Self::make_pipeline(device, &library, "bf16_matmul_tiled")?,
            // Batched MoE (contiguous expert weights, no CPU sync)
            moe_batched_gate_up_swiglu_q4: Self::make_pipeline(device, &library, "moe_batched_gate_up_swiglu_q4")?,
            moe_batched_gate_up_swiglu_int8: Self::make_pipeline(device, &library, "moe_batched_gate_up_swiglu_int8")?,
            moe_batched_gate_up_swiglu_bf16: Self::make_pipeline(device, &library, "moe_batched_gate_up_swiglu_bf16")?,
            moe_batched_down_q4: Self::make_pipeline(device, &library, "moe_batched_down_q4")?,
            moe_batched_down_int8: Self::make_pipeline(device, &library, "moe_batched_down_int8")?,
            moe_batched_down_bf16: Self::make_pipeline(device, &library, "moe_batched_down_bf16")?,
            moe_reduce: Self::make_pipeline(device, &library, "moe_reduce")?,
            moe_reduce_residual: Self::make_pipeline(device, &library, "moe_reduce_residual")?,
            // Prefill MoE (zero-sync, tiled with counting sort)
            moe_softmax_topk_batch: Self::make_pipeline(device, &library, "moe_softmax_topk_batch")?,
            moe_prefill_sort: Self::make_pipeline(device, &library, "moe_prefill_sort")?,
            moe_prefill_tiled_gate_up_swiglu_q4: Self::make_pipeline(device, &library, "moe_prefill_tiled_gate_up_swiglu_q4")?,
            moe_prefill_tiled_gate_up_swiglu_int8: Self::make_pipeline(device, &library, "moe_prefill_tiled_gate_up_swiglu_int8")?,
            moe_prefill_tiled_gate_up_swiglu_bf16: Self::make_pipeline(device, &library, "moe_prefill_tiled_gate_up_swiglu_bf16")?,
            moe_prefill_tiled_down_q4: Self::make_pipeline(device, &library, "moe_prefill_tiled_down_q4")?,
            moe_prefill_tiled_down_int8: Self::make_pipeline(device, &library, "moe_prefill_tiled_down_int8")?,
            moe_prefill_tiled_down_bf16: Self::make_pipeline(device, &library, "moe_prefill_tiled_down_bf16")?,
            moe_prefill_reduce_residual: Self::make_pipeline(device, &library, "moe_prefill_reduce_residual")?,
            // Fused prefill MoE down + reduce + residual
            moe_prefill_tiled_down_residual_q4: Self::make_pipeline(device, &library, "moe_prefill_tiled_down_residual_q4")?,
            moe_prefill_tiled_down_residual_int8: Self::make_pipeline(device, &library, "moe_prefill_tiled_down_residual_int8")?,
            moe_prefill_tiled_down_residual_bf16: Self::make_pipeline(device, &library, "moe_prefill_tiled_down_residual_bf16")?,
            // H2O eviction
            h2o_score_probe_half: Self::make_pipeline(device, &library, "h2o_score_probe_half")?,
            h2o_score_probe_i8: Self::make_pipeline(device, &library, "h2o_score_probe_i8")?,
            kv_cache_compact_half: Self::make_pipeline(device, &library, "kv_cache_compact_half")?,
            kv_cache_compact_i8: Self::make_pipeline(device, &library, "kv_cache_compact_i8")?,
            kv_cache_compact_scales: Self::make_pipeline(device, &library, "kv_cache_compact_scales")?,
            // DeepStack
            deepstack_add: Self::make_pipeline(device, &library, "deepstack_add")?,
            // Vision encoder
            layer_norm_batch: Self::make_pipeline(device, &library, "layer_norm_batch")?,
            gelu_batch: Self::make_pipeline(device, &library, "gelu_batch")?,
            vision_split_qkv: Self::make_pipeline(device, &library, "vision_split_qkv")?,
            vision_spatial_merge: Self::make_pipeline(device, &library, "vision_spatial_merge")?,
            // Metal 4 cooperative_tensor pipelines (optional, Apple11+ only)
            q4_matmul_tiled_coop32: m4_library.as_ref()
                .and_then(|lib| Self::make_pipeline(device, lib, "q4_matmul_tiled_coop32").ok()),
            bf16_matmul_tiled_coop32: m4_library.as_ref()
                .and_then(|lib| Self::make_pipeline(device, lib, "bf16_matmul_tiled_coop32").ok()),
            attention_prefill_v6_coop32: m4_library.as_ref()
                .and_then(|lib| Self::make_pipeline(device, lib, "attention_prefill_v6_coop32").ok()),
            // FlashDecoding v2: double-buffered K/V loads
            attention_decode_flash_tile_v2: Self::make_pipeline(device, &library, "attention_decode_flash_tile_v2").ok(),
            attention_decode_flash_tile_i8_v2: Self::make_pipeline(device, &library, "attention_decode_flash_tile_i8_v2").ok(),
            // MetalPerformancePrimitives matmul2d (Neural Accelerators)
            q4_matmul_tiled_mpp: m4_library.as_ref()
                .and_then(|lib| Self::make_pipeline(device, lib, "q4_matmul_tiled_mpp").ok()),
            bf16_matmul_tiled_mpp: m4_library.as_ref()
                .and_then(|lib| Self::make_pipeline(device, lib, "bf16_matmul_tiled_mpp").ok()),
            attention_prefill_v7_mpp: m4_library.as_ref()
                .and_then(|lib| Self::make_pipeline(device, lib, "attention_prefill_v7_mpp").ok()),
            // Metal 4 half-precision accumulation
            rms_norm_half: m4_library.as_ref()
                .and_then(|lib| Self::make_pipeline(device, lib, "rms_norm_half").ok()),
            rms_norm_batch_half: m4_library.as_ref()
                .and_then(|lib| Self::make_pipeline(device, lib, "rms_norm_batch_half").ok()),
            // Q4 matvec v2 (decode optimisation — works on all GPUs, MSL 3.1)
            q4_matvec_v2: Self::make_pipeline(device, &library, "q4_matvec_v2").ok(),
            q4_matvec_residual_add_v2: Self::make_pipeline(device, &library, "q4_matvec_residual_add_v2").ok(),
        })
    }

    fn load_library(device: &ProtocolObject<dyn MTLDevice>) -> Result<Retained<ProtocolObject<dyn MTLLibrary>>> {
        // Try pre-compiled metallib from build.rs
        if let Some(metallib_path) = option_env!("METAL_LIB_PATH") {
            let url = objc2_foundation::NSURL::fileURLWithPath(&NSString::from_str(metallib_path));
            match device.newLibraryWithURL_error(&url) {
                Ok(lib) => return Ok(lib),
                Err(e) => {
                    tracing::warn!("Failed to load pre-compiled metallib: {:?}, falling back to runtime compilation", e);
                }
            }
        }

        // Fallback: runtime compilation from embedded MSL sources
        Self::compile_library_from_sources(device)
    }

    /// Load the Metal 4 library (MSL 4.0 shaders compiled for Apple11+).
    fn load_m4_library(device: &ProtocolObject<dyn MTLDevice>) -> Result<Retained<ProtocolObject<dyn MTLLibrary>>> {
        if let Some(metallib_path) = option_env!("METAL4_LIB_PATH") {
            let url = objc2_foundation::NSURL::fileURLWithPath(&NSString::from_str(metallib_path));
            match device.newLibraryWithURL_error(&url) {
                Ok(lib) => {
                    eprintln!("[metal] Loaded Metal 4 shader library (cooperative_tensor 32×32)");
                    return Ok(lib);
                }
                Err(e) => {
                    eprintln!("[metal] WARNING: Failed to load Metal 4 metallib: {:?}", e);
                }
            }
        } else {
            eprintln!("[metal] Metal 4 shaders not compiled (SDK < macOS 26); coop32 pipelines unavailable");
        }
        Err(HerbertError::Backend("Metal 4 library not available".into()))
    }

    fn compile_library_from_sources(device: &ProtocolObject<dyn MTLDevice>) -> Result<Retained<ProtocolObject<dyn MTLLibrary>>> {
        let all_sources = [
            // Normalization
            include_msl!("rms_norm"),
            include_msl!("rms_norm_batch"),
            include_msl!("head_rms_norm"),
            include_msl!("rms_norm_residual"),
            // Activation & utilities
            include_msl!("softmax"),
            include_msl!("argmax"),
            include_msl!("swiglu"),
            include_msl!("embedding"),
            include_msl!("residual_add"),
            include_msl!("bias_add_batch"),
            include_msl!("scaled_add"),
            include_msl!("copy_buffer"),
            include_msl!("fused_gate_up_swiglu"),
            // Matvec
            include_msl!("bf16_matvec"),
            include_msl!("f32_matvec"),
            include_msl!("int8_matvec"),
            include_msl!("q4_matvec"),
            include_msl!("q4_matvec_residual_add"),
            include_msl!("q4_matvec_residual_add_8row"),
            include_msl!("q4_matvec_v2"),
            include_msl!("q4_matvec_residual_add_v2"),
            include_msl!("q4_matvec_qkv"),
            include_msl!("q4_matvec_qkv_normed"),
            // Matmul
            include_msl!("bf16_matmul"),
            include_msl!("f32_matmul"),
            include_msl!("int8_matmul"),
            include_msl!("q4_matmul"),
            // Attention
            include_msl!("attention_decode"),
            include_msl!("attention_decode_gqa"),
            include_msl!("attention_decode_flash"),
            include_msl!("attention_prefill"),
            include_msl!("attention_prefill_v2_gqa"),
            include_msl!("attention_prefill_v3_tiled"),
            include_msl!("attention_prefill_v4_simdgroup"),
            include_msl!("attention_prefill_v5_half_kv"),
            // RoPE
            include_msl!("rope_single"),
            include_msl!("rope_batch"),
            include_msl!("rope_kv_append"),
            // KV cache
            include_msl!("kv_cache_append"),
            include_msl!("kv_cache_append_batch"),
            include_msl!("kv_cache_append_i8"),
            include_msl!("kv_cache_quantize_i8"),
            include_msl!("head_norm_rope_kv_append_i8"),
            // MoE
            include_msl!("moe_gather"),
            include_msl!("moe_scatter_add"),
            include_msl!("moe_softmax_topk"),
            // Fused MoE expert kernels
            include_msl!("moe_fused_gate_up_swiglu_q4"),
            include_msl!("moe_fused_gate_up_swiglu_int8"),
            include_msl!("matvec_scaled_add_q4"),
            include_msl!("matvec_scaled_add_int8"),
            include_msl!("matvec_scaled_add_bf16"),
            // Fused attention kernels
            include_msl!("head_norm_rope"),
            include_msl!("head_norm_rope_batch"),
            include_msl!("head_norm_rope_kv_append"),
            // Tiled matmul (prefill) — shader selected by `simdgroup-matmul` feature
            Q4_MATMUL_TILED_SRC,
            include_msl!("bf16_matmul_tiled"),
            // Batched MoE (contiguous expert weights)
            include_msl!("moe_batched_gate_up_swiglu_q4"),
            include_msl!("moe_batched_gate_up_swiglu_int8"),
            include_msl!("moe_batched_gate_up_swiglu_bf16"),
            include_msl!("moe_batched_down_q4"),
            include_msl!("moe_batched_down_int8"),
            include_msl!("moe_batched_down_bf16"),
            include_msl!("moe_reduce"),
            include_msl!("moe_reduce_residual"),
            // Prefill MoE (zero-sync, tiled with counting sort)
            include_msl!("moe_softmax_topk_batch"),
            include_msl!("moe_prefill_sort"),
            include_msl!("moe_prefill_tiled_gate_up_swiglu_q4"),
            include_msl!("moe_prefill_tiled_gate_up_swiglu_int8"),
            include_msl!("moe_prefill_tiled_gate_up_swiglu_bf16"),
            include_msl!("moe_prefill_tiled_down_q4"),
            include_msl!("moe_prefill_tiled_down_int8"),
            include_msl!("moe_prefill_tiled_down_bf16"),
            include_msl!("moe_prefill_reduce_residual"),
            // Fused prefill MoE down + reduce + residual
            include_msl!("moe_prefill_tiled_down_residual_q4"),
            include_msl!("moe_prefill_tiled_down_residual_int8"),
            include_msl!("moe_prefill_tiled_down_residual_bf16"),
            // H2O eviction
            include_msl!("h2o_score_probe"),
            include_msl!("kv_cache_compact"),
            // DeepStack
            include_msl!("deepstack_add"),
            // Vision encoder
            include_msl!("layer_norm_batch"),
            include_msl!("gelu_batch"),
            include_msl!("vision_split_qkv"),
            include_msl!("vision_spatial_merge"),
        ].join("\n");

        let source = NSString::from_str(&all_sources);
        let options = MTLCompileOptions::new();
        options.setLanguageVersion(MTLLanguageVersion::Version3_1);

        device.newLibraryWithSource_options_error(&source, Some(&options))
            .map_err(|e| HerbertError::Backend(
                format!("Failed to compile MSL shaders: {:?}", e),
            ))
    }

    fn make_pipeline(
        device: &ProtocolObject<dyn MTLDevice>,
        library: &ProtocolObject<dyn MTLLibrary>,
        function_name: &str,
    ) -> Result<ComputePipeline> {
        let name = NSString::from_str(function_name);
        let function = library.newFunctionWithName(&name)
            .ok_or_else(|| HerbertError::Backend(
                format!("Metal function '{}' not found in library", function_name),
            ))?;

        let state = device.newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| HerbertError::Backend(
                format!("Failed to create pipeline for '{}': {:?}", function_name, e),
            ))?;

        Ok(ComputePipeline { state })
    }
}
