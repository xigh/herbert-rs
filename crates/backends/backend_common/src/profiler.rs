//! Layer profiler — progressive instrumentation behind feature flags.
//!
//! - No feature: stub (zero-cost, no timing data)
//! - `profile-layer`: per-layer breakdown (norm1, attn, norm2, ffn)
//! - `profile-attn`: attention sub-step detail (implies profile-layer)
//! - `profile-experts`: MoE sub-step detail (implies profile-layer)
//! - `profile-kv`: KV cache operation timing
//! - `profile-prefill-layer`: 11-step per-layer detail, runs 2 layers then exits
//! - `profile-decode-layer`: 4-step per-layer detail for decode, runs 2 layers then exits

use herbert_core::error::Result;

// ============================================================
// Stub LayerProfiler (always available, used by generic_model)
// ============================================================

pub struct LayerProfiler {
    #[cfg(feature = "profile-layer")]
    times: Vec<f64>,
}

impl LayerProfiler {
    #[inline]
    pub fn new(_num_layers: usize) -> Self {
        Self {
            #[cfg(feature = "profile-layer")]
            times: Vec::with_capacity(_num_layers),
        }
    }

    #[inline]
    pub fn measure<T, F>(&mut self, f: F) -> Result<T>
    where
        F: FnOnce() -> Result<T>,
    {
        #[cfg(feature = "profile-layer")]
        {
            let t0 = std::time::Instant::now();
            let result = f();
            self.times.push(t0.elapsed().as_secs_f64() * 1000.0);
            result
        }
        #[cfg(not(feature = "profile-layer"))]
        {
            f()
        }
    }

    #[inline]
    pub fn finish(self) -> Vec<f64> {
        #[cfg(feature = "profile-layer")]
        {
            self.times
        }
        #[cfg(not(feature = "profile-layer"))]
        {
            Vec::new()
        }
    }
}

// ============================================================
// Profile data structures (behind feature flags)
// ============================================================

#[cfg(feature = "profile-layer")]
#[derive(Clone, Debug)]
pub struct LayerStepProfile {
    pub norm1_us: u64,
    pub attention_us: u64,
    pub norm2_us: u64,
    pub ffn_us: u64,
}

#[cfg(feature = "profile-attn")]
#[derive(Clone, Debug)]
pub struct AttnStepProfile {
    pub qkv_matmul_us: u64,
    pub qk_norm_us: u64,
    pub rope_us: u64,
    pub kv_append_us: u64,
    pub kernel_us: u64,
    pub o_proj_us: u64,
}

#[cfg(feature = "profile-experts")]
#[derive(Clone, Debug)]
pub struct MoeStepProfile {
    pub router_us: u64,
    pub routing_us: u64,
    pub dispatch_us: u64,
    pub scatter_us: u64,
    pub num_active_experts: usize,
    pub num_experts_total: usize,
}

#[cfg(feature = "profile-kv")]
#[derive(Clone, Debug)]
pub struct KvAppendProfile {
    pub append_us: u64,
}

// ============================================================
// Global accumulator
// ============================================================

#[cfg(feature = "profile-layer")]
struct PrefillProfileData {
    seq_len: usize,
    num_layers: usize,
    layer_steps: Vec<LayerStepProfile>,
    #[cfg(feature = "profile-attn")]
    attn_steps: Vec<AttnStepProfile>,
    #[cfg(feature = "profile-experts")]
    moe_steps: Vec<MoeStepProfile>,
    #[cfg(feature = "profile-kv")]
    kv_appends: Vec<KvAppendProfile>,
}

#[cfg(feature = "profile-layer")]
static PROFILER: std::sync::Mutex<Option<PrefillProfileData>> = std::sync::Mutex::new(None);

// ============================================================
// Public API
// ============================================================

/// Reset the profiler for a new prefill pass.
#[cfg(feature = "profile-layer")]
pub fn reset(seq_len: usize, num_layers: usize) {
    let mut guard = PROFILER.lock().unwrap_or_else(|e| e.into_inner());
    *guard = Some(PrefillProfileData {
        seq_len,
        num_layers,
        layer_steps: Vec::with_capacity(num_layers),
        #[cfg(feature = "profile-attn")]
        attn_steps: Vec::with_capacity(num_layers),
        #[cfg(feature = "profile-experts")]
        moe_steps: Vec::new(),
        #[cfg(feature = "profile-kv")]
        kv_appends: Vec::new(),
    });
}

/// Record one layer's timing breakdown.
#[cfg(feature = "profile-layer")]
pub fn push_layer(profile: LayerStepProfile) {
    let mut guard = PROFILER.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(ref mut data) = *guard {
        data.layer_steps.push(profile);
    }
}

/// Record one layer's attention detail.
#[cfg(feature = "profile-attn")]
pub fn push_attn(profile: AttnStepProfile) {
    let mut guard = PROFILER.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(ref mut data) = *guard {
        data.attn_steps.push(profile);
    }
}

/// Record one MoE layer's timing breakdown.
#[cfg(feature = "profile-experts")]
pub fn push_moe(profile: MoeStepProfile) {
    let mut guard = PROFILER.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(ref mut data) = *guard {
        data.moe_steps.push(profile);
    }
}

/// Record one KV append operation timing.
#[cfg(feature = "profile-kv")]
pub fn push_kv_append(append_us: u64) {
    let mut guard = PROFILER.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(ref mut data) = *guard {
        data.kv_appends.push(KvAppendProfile { append_us });
    }
}

/// Print the profiling report to stderr and clear the data.
#[cfg(feature = "profile-layer")]
pub fn report() {
    let mut guard = PROFILER.lock().unwrap_or_else(|e| e.into_inner());
    let data = match guard.take() {
        Some(d) => d,
        None => return,
    };

    let seq_len = data.seq_len;
    let num_layers = data.num_layers;

    eprintln!("[PROFILE] Prefill: {} tokens, {} layers", seq_len, num_layers);
    eprintln!("[PROFILE]");

    if data.layer_steps.is_empty() {
        eprintln!("[PROFILE] (no layer data recorded)");
        return;
    }

    // Per-layer breakdown
    eprintln!("[PROFILE] Per-layer breakdown (ms):");
    eprintln!(
        "[PROFILE]   {:>5}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}",
        "Layer", "norm1", "attn", "norm2", "ffn", "total"
    );
    for (i, step) in data.layer_steps.iter().enumerate() {
        let norm1 = step.norm1_us as f64 / 1000.0;
        let attn = step.attention_us as f64 / 1000.0;
        let norm2 = step.norm2_us as f64 / 1000.0;
        let ffn = step.ffn_us as f64 / 1000.0;
        let total = norm1 + attn + norm2 + ffn;
        eprintln!(
            "[PROFILE]   {:>5}  {:>8.1}  {:>8.1}  {:>8.1}  {:>8.1}  {:>8.1}",
            i, norm1, attn, norm2, ffn, total
        );
    }

    // Aggregate
    let total_norm1: u64 = data.layer_steps.iter().map(|s| s.norm1_us).sum();
    let total_attn: u64 = data.layer_steps.iter().map(|s| s.attention_us).sum();
    let total_norm2: u64 = data.layer_steps.iter().map(|s| s.norm2_us).sum();
    let total_ffn: u64 = data.layer_steps.iter().map(|s| s.ffn_us).sum();
    let grand_total = total_norm1 + total_attn + total_norm2 + total_ffn;
    let gt_ms = grand_total as f64 / 1000.0;

    eprintln!("[PROFILE]");
    eprintln!("[PROFILE] Aggregate ({} layers):", num_layers);

    let pct = |v: u64| {
        if grand_total > 0 {
            v as f64 / grand_total as f64 * 100.0
        } else {
            0.0
        }
    };
    eprintln!(
        "[PROFILE]   norm1:     {:>8.1}ms ({:>5.1}%)",
        total_norm1 as f64 / 1000.0,
        pct(total_norm1)
    );
    eprintln!(
        "[PROFILE]   attention: {:>8.1}ms ({:>5.1}%)",
        total_attn as f64 / 1000.0,
        pct(total_attn)
    );
    eprintln!(
        "[PROFILE]   norm2:     {:>8.1}ms ({:>5.1}%)",
        total_norm2 as f64 / 1000.0,
        pct(total_norm2)
    );
    eprintln!(
        "[PROFILE]   ffn:       {:>8.1}ms ({:>5.1}%)",
        total_ffn as f64 / 1000.0,
        pct(total_ffn)
    );
    let toks_per_sec = if gt_ms > 0.0 {
        seq_len as f64 / (gt_ms / 1000.0)
    } else {
        0.0
    };
    eprintln!(
        "[PROFILE]   TOTAL:     {:>8.1}ms ({:.1} tok/s)",
        gt_ms, toks_per_sec
    );

    // Attention detail
    #[cfg(feature = "profile-attn")]
    if !data.attn_steps.is_empty() {
        let total_qkv: u64 = data.attn_steps.iter().map(|s| s.qkv_matmul_us).sum();
        let total_qk_norm: u64 = data.attn_steps.iter().map(|s| s.qk_norm_us).sum();
        let total_rope: u64 = data.attn_steps.iter().map(|s| s.rope_us).sum();
        let total_kv_append: u64 = data.attn_steps.iter().map(|s| s.kv_append_us).sum();
        let total_kernel: u64 = data.attn_steps.iter().map(|s| s.kernel_us).sum();
        let total_o_proj: u64 = data.attn_steps.iter().map(|s| s.o_proj_us).sum();
        let attn_total = total_qkv + total_qk_norm + total_rope + total_kv_append + total_kernel + total_o_proj;

        eprintln!("[PROFILE]");
        eprintln!("[PROFILE] Attention detail (aggregate, ms):");

        let apct = |v: u64| {
            if attn_total > 0 {
                v as f64 / attn_total as f64 * 100.0
            } else {
                0.0
            }
        };
        eprintln!(
            "[PROFILE]   qkv_matmul:  {:>8.1}ms ({:>5.1}%)",
            total_qkv as f64 / 1000.0,
            apct(total_qkv)
        );
        eprintln!(
            "[PROFILE]   qk_norm:     {:>8.1}ms ({:>5.1}%)",
            total_qk_norm as f64 / 1000.0,
            apct(total_qk_norm)
        );
        eprintln!(
            "[PROFILE]   rope:        {:>8.1}ms ({:>5.1}%)",
            total_rope as f64 / 1000.0,
            apct(total_rope)
        );
        eprintln!(
            "[PROFILE]   kv_append:   {:>8.1}ms ({:>5.1}%)",
            total_kv_append as f64 / 1000.0,
            apct(total_kv_append)
        );
        eprintln!(
            "[PROFILE]   attn_kernel: {:>8.1}ms ({:>5.1}%)",
            total_kernel as f64 / 1000.0,
            apct(total_kernel)
        );
        eprintln!(
            "[PROFILE]   o_proj:      {:>8.1}ms ({:>5.1}%)",
            total_o_proj as f64 / 1000.0,
            apct(total_o_proj)
        );
    }

    // MoE detail
    #[cfg(feature = "profile-experts")]
    if !data.moe_steps.is_empty() {
        let total_router: u64 = data.moe_steps.iter().map(|s| s.router_us).sum();
        let total_routing: u64 = data.moe_steps.iter().map(|s| s.routing_us).sum();
        let total_dispatch: u64 = data.moe_steps.iter().map(|s| s.dispatch_us).sum();
        let total_scatter: u64 = data.moe_steps.iter().map(|s| s.scatter_us).sum();
        let total_active: usize = data.moe_steps.iter().map(|s| s.num_active_experts).sum();
        let total_experts: usize = data.moe_steps.iter().map(|s| s.num_experts_total).sum();
        let num_moe_layers = data.moe_steps.len();
        let avg_active = if num_moe_layers > 0 {
            total_active as f64 / num_moe_layers as f64
        } else {
            0.0
        };
        let avg_total = if num_moe_layers > 0 {
            total_experts as f64 / num_moe_layers as f64
        } else {
            0.0
        };

        eprintln!("[PROFILE]");
        eprintln!(
            "[PROFILE] MoE detail (aggregate across {} MoE layers, ms):",
            num_moe_layers
        );
        eprintln!(
            "[PROFILE]   router:     {:>8.1}ms",
            total_router as f64 / 1000.0
        );
        eprintln!(
            "[PROFILE]   routing:    {:>8.1}ms",
            total_routing as f64 / 1000.0
        );
        eprintln!(
            "[PROFILE]   dispatch:   {:>8.1}ms",
            total_dispatch as f64 / 1000.0
        );
        eprintln!(
            "[PROFILE]   scatter:    {:>8.1}ms",
            total_scatter as f64 / 1000.0
        );
        eprintln!(
            "[PROFILE]   avg active experts: {:.1}/{:.0}",
            avg_active, avg_total
        );
    }

    // KV cache detail
    #[cfg(feature = "profile-kv")]
    if !data.kv_appends.is_empty() {
        let total_append: u64 = data.kv_appends.iter().map(|s| s.append_us).sum();
        let num_appends = data.kv_appends.len();

        eprintln!("[PROFILE]");
        eprintln!(
            "[PROFILE] KV cache detail ({} appends):",
            num_appends
        );
        eprintln!(
            "[PROFILE]   total append: {:>8.1}ms",
            total_append as f64 / 1000.0
        );
        eprintln!(
            "[PROFILE]   avg per append: {:.1}us",
            if num_appends > 0 {
                total_append as f64 / num_appends as f64
            } else {
                0.0
            }
        );
    }
}

// ============================================================
// profile-prefill-layer: fine-grained 11-step per-layer detail
// ============================================================

#[cfg(feature = "profile-prefill-layer")]
pub enum BlockSubTimings {
    Attn {
        qkv_proj_us: u64,
        qk_norm_us: u64,
        rope_us: u64,
        kv_append_us: u64,
        attn_kernel_us: u64,
        o_proj_us: u64,
    },
    Conv {
        in_proj_us: u64,
        gating_us: u64,
        conv_kernel_us: u64,
        out_proj_us: u64,
    },
}

#[cfg(feature = "profile-prefill-layer")]
pub struct PrefillLayerDetailProfile {
    pub rmsnorm1_us: u64,
    pub block: BlockSubTimings,
    pub residual1_us: u64,
    pub rmsnorm2_us: u64,
    pub ffn_us: u64,
    pub residual2_us: u64,
}

/// Attention sub-timings passed from generic_attention to generic_layer via side-channel.
#[cfg(feature = "profile-prefill-layer")]
pub struct AttnSubTimings {
    pub qkv_proj_us: u64,
    pub qk_norm_us: u64,
    pub rope_us: u64,
    pub kv_append_us: u64,
    pub attn_kernel_us: u64,
    pub o_proj_us: u64,
}

/// Convolution sub-timings passed from GenericConvolution to generic_layer via side-channel.
#[cfg(feature = "profile-prefill-layer")]
pub struct ConvSubTimings {
    pub in_proj_us: u64,
    pub gating_us: u64,
    pub conv_kernel_us: u64,
    pub out_proj_us: u64,
}

#[cfg(feature = "profile-prefill-layer")]
static PENDING_ATTN_TIMINGS: std::sync::Mutex<Option<AttnSubTimings>> =
    std::sync::Mutex::new(None);

#[cfg(feature = "profile-prefill-layer")]
static PENDING_CONV_TIMINGS: std::sync::Mutex<Option<ConvSubTimings>> =
    std::sync::Mutex::new(None);

#[cfg(feature = "profile-prefill-layer")]
static PREFILL_LAYER_PROFILES: std::sync::Mutex<Vec<PrefillLayerDetailProfile>> =
    std::sync::Mutex::new(Vec::new());

#[cfg(feature = "profile-prefill-layer")]
pub fn set_attn_sub_timings(t: AttnSubTimings) {
    let mut guard = PENDING_ATTN_TIMINGS.lock().unwrap_or_else(|e| e.into_inner());
    *guard = Some(t);
}

#[cfg(feature = "profile-prefill-layer")]
pub fn take_attn_sub_timings() -> Option<AttnSubTimings> {
    let mut guard = PENDING_ATTN_TIMINGS.lock().unwrap_or_else(|e| e.into_inner());
    guard.take()
}

#[cfg(feature = "profile-prefill-layer")]
pub fn set_conv_sub_timings(t: ConvSubTimings) {
    let mut guard = PENDING_CONV_TIMINGS.lock().unwrap_or_else(|e| e.into_inner());
    *guard = Some(t);
}

#[cfg(feature = "profile-prefill-layer")]
pub fn take_conv_sub_timings() -> Option<ConvSubTimings> {
    let mut guard = PENDING_CONV_TIMINGS.lock().unwrap_or_else(|e| e.into_inner());
    guard.take()
}

#[cfg(feature = "profile-prefill-layer")]
pub fn push_prefill_layer_detail(p: PrefillLayerDetailProfile) {
    let mut guard = PREFILL_LAYER_PROFILES.lock().unwrap_or_else(|e| e.into_inner());
    guard.push(p);
}

#[cfg(feature = "profile-prefill-layer")]
pub fn reset_prefill_layer_detail() {
    let mut guard = PREFILL_LAYER_PROFILES.lock().unwrap_or_else(|e| e.into_inner());
    guard.clear();
}

#[cfg(feature = "profile-prefill-layer")]
pub fn report_prefill_layer_detail(seq_len: usize) {
    let mut guard = PREFILL_LAYER_PROFILES.lock().unwrap_or_else(|e| e.into_inner());
    let profiles = std::mem::take(&mut *guard);

    for (layer_idx, p) in profiles.iter().enumerate() {
        let mut steps: Vec<(&str, u64)> = Vec::new();
        steps.push(("RMSNorm1", p.rmsnorm1_us));
        match &p.block {
            BlockSubTimings::Attn {
                qkv_proj_us,
                qk_norm_us,
                rope_us,
                kv_append_us,
                attn_kernel_us,
                o_proj_us,
            } => {
                steps.push(("QKV projection", *qkv_proj_us));
                steps.push(("Q/K norm", *qk_norm_us));
                steps.push(("RoPE", *rope_us));
                steps.push(("KV append", *kv_append_us));
                steps.push(("Attention kernel", *attn_kernel_us));
                steps.push(("O projection", *o_proj_us));
            }
            BlockSubTimings::Conv {
                in_proj_us,
                gating_us,
                conv_kernel_us,
                out_proj_us,
            } => {
                steps.push(("Conv in_proj", *in_proj_us));
                steps.push(("Conv gating", *gating_us));
                steps.push(("Conv kernel", *conv_kernel_us));
                steps.push(("Conv out_proj", *out_proj_us));
            }
        }
        steps.push(("Residual1", p.residual1_us));
        steps.push(("RMSNorm2", p.rmsnorm2_us));
        steps.push(("MoE/FFN", p.ffn_us));
        steps.push(("Residual2", p.residual2_us));

        let total_us: u64 = steps.iter().map(|(_, v)| v).sum();
        let block_label = match &p.block {
            BlockSubTimings::Attn { .. } => "attention",
            BlockSubTimings::Conv { .. } => "convolution",
        };

        eprintln!(
            "\n[PROFILE-PREFILL-LAYER] Layer {} ({}, seq_len={}):",
            layer_idx, block_label, seq_len
        );
        eprintln!("  {:<22} {:>10}  {:>6}", "Step", "Duration", "%");
        eprintln!("  {}", "\u{2500}".repeat(42));
        for (name, us) in &steps {
            let ms = *us as f64 / 1000.0;
            let pct = if total_us > 0 {
                *us as f64 / total_us as f64 * 100.0
            } else {
                0.0
            };
            eprintln!("  {:<22} {:>8.1} ms  {:>5.1}%", name, ms, pct);
        }
        eprintln!("  {}", "\u{2500}".repeat(42));
        eprintln!(
            "  {:<22} {:>8.1} ms",
            "TOTAL",
            total_us as f64 / 1000.0
        );
    }
}

// ============================================================
// profile-decode-layer: per-layer 4-step detail (decode path)
// ============================================================

#[cfg(feature = "profile-decode-layer")]
pub enum DecodeBlockSubTimings {
    Attn {
        qkv_proj_us: u64,
        qk_norm_us: u64,
        rope_us: u64,
        kv_cache_attn_us: u64,
        output_gate_us: u64,
        o_proj_us: u64,
    },
    DeltaNet {
        in_proj_qkv_us: u64,
        in_proj_z_us: u64,
        in_proj_ab_us: u64,
        recurrent_step_us: u64,
        out_proj_us: u64,
    },
}

#[cfg(feature = "profile-decode-layer")]
static PENDING_DECODE_BLOCK_TIMINGS: std::sync::Mutex<Option<DecodeBlockSubTimings>> =
    std::sync::Mutex::new(None);

#[cfg(feature = "profile-decode-layer")]
pub fn set_decode_block_sub_timings(t: DecodeBlockSubTimings) {
    let mut guard = PENDING_DECODE_BLOCK_TIMINGS.lock().unwrap_or_else(|e| e.into_inner());
    *guard = Some(t);
}

#[cfg(feature = "profile-decode-layer")]
pub fn take_decode_block_sub_timings() -> Option<DecodeBlockSubTimings> {
    let mut guard = PENDING_DECODE_BLOCK_TIMINGS.lock().unwrap_or_else(|e| e.into_inner());
    guard.take()
}

#[cfg(feature = "profile-decode-layer")]
pub struct DecodeLayerDetailProfile {
    pub norm1_us: u64,
    pub attention_us: u64,
    pub norm2_us: u64,
    pub ffn_us: u64,
    pub layer_type: &'static str, // "Attn", "DeltaNet", or "Conv"
    pub block_sub: Option<DecodeBlockSubTimings>,
}

#[cfg(feature = "profile-decode-layer")]
static DECODE_LAYER_PROFILES: std::sync::Mutex<Vec<DecodeLayerDetailProfile>> =
    std::sync::Mutex::new(Vec::new());

#[cfg(feature = "profile-decode-layer")]
pub fn reset_decode_layer_detail() {
    let mut guard = DECODE_LAYER_PROFILES.lock().unwrap_or_else(|e| e.into_inner());
    guard.clear();
}

#[cfg(feature = "profile-decode-layer")]
pub fn push_decode_layer_detail(p: DecodeLayerDetailProfile) {
    let mut guard = DECODE_LAYER_PROFILES.lock().unwrap_or_else(|e| e.into_inner());
    guard.push(p);
}

#[cfg(feature = "profile-decode-layer")]
pub fn report_decode_layer_detail() {
    let mut guard = DECODE_LAYER_PROFILES.lock().unwrap_or_else(|e| e.into_inner());
    let profiles = std::mem::take(&mut *guard);

    // Per-layer detail
    for (layer_idx, p) in profiles.iter().enumerate() {
        let steps: [(&str, u64); 4] = [
            ("RMSNorm1", p.norm1_us),
            ("Block", p.attention_us),
            ("RMSNorm2", p.norm2_us),
            ("MoE/FFN", p.ffn_us),
        ];
        let total_us: u64 = steps.iter().map(|(_, v)| v).sum();

        eprintln!("\n[PROFILE-DECODE-LAYER] Layer {} ({}):", layer_idx, p.layer_type);
        eprintln!("  {:<20} {:>10}  {:>6}", "Step", "Duration", "%");
        eprintln!("  {}", "\u{2500}".repeat(40));
        for (name, us) in &steps {
            let ms = *us as f64 / 1000.0;
            let pct = if total_us > 0 {
                *us as f64 / total_us as f64 * 100.0
            } else {
                0.0
            };
            eprintln!("  {:<20} {:>8.1} ms  {:>5.1}%", name, ms, pct);
        }

        // Sub-block detail
        if let Some(ref sub) = p.block_sub {
            let sub_steps: Vec<(&str, u64)> = match sub {
                DecodeBlockSubTimings::Attn {
                    qkv_proj_us, qk_norm_us, rope_us,
                    kv_cache_attn_us, output_gate_us, o_proj_us,
                } => vec![
                    ("  qkv_proj", *qkv_proj_us),
                    ("  qk_norm", *qk_norm_us),
                    ("  rope", *rope_us),
                    ("  kv+attn_kernel", *kv_cache_attn_us),
                    ("  output_gate", *output_gate_us),
                    ("  o_proj", *o_proj_us),
                ],
                DecodeBlockSubTimings::DeltaNet {
                    in_proj_qkv_us, in_proj_z_us, in_proj_ab_us,
                    recurrent_step_us, out_proj_us,
                } => vec![
                    ("  in_proj_qkv", *in_proj_qkv_us),
                    ("  in_proj_z", *in_proj_z_us),
                    ("  in_proj_ab", *in_proj_ab_us),
                    ("  recurrent_step", *recurrent_step_us),
                    ("  out_proj", *out_proj_us),
                ],
            };
            for (name, us) in &sub_steps {
                let ms = *us as f64 / 1000.0;
                let pct = if p.attention_us > 0 {
                    *us as f64 / p.attention_us as f64 * 100.0
                } else { 0.0 };
                eprintln!("  {:<20} {:>8.1} ms  {:>5.1}%", name, ms, pct);
            }
        }

        eprintln!("  {}", "\u{2500}".repeat(40));
        eprintln!(
            "  {:<20} {:>8.1} ms",
            "TOTAL",
            total_us as f64 / 1000.0
        );
    }

    // Summary by layer type
    let mut type_stats: std::collections::BTreeMap<&str, (u64, u64, u64, u64, usize)> =
        std::collections::BTreeMap::new();
    for p in &profiles {
        let e = type_stats.entry(p.layer_type).or_insert((0, 0, 0, 0, 0));
        e.0 += p.norm1_us;
        e.1 += p.attention_us;
        e.2 += p.norm2_us;
        e.3 += p.ffn_us;
        e.4 += 1;
    }
    eprintln!("\n[PROFILE-DECODE-LAYER] === Summary by layer type ===");
    eprintln!("  {:<10} {:>5}  {:>10} {:>10} {:>10} {:>10}  {:>10}",
        "Type", "Count", "Norm1", "Block", "Norm2", "FFN", "Total");
    eprintln!("  {}", "\u{2500}".repeat(72));
    let mut grand_total_us = 0u64;
    for (lt, (n1, blk, n2, ffn, count)) in &type_stats {
        let total = n1 + blk + n2 + ffn;
        grand_total_us += total;
        let avg = total as f64 / *count as f64 / 1000.0;
        eprintln!("  {:<10} {:>5}  {:>8.1}ms {:>8.1}ms {:>8.1}ms {:>8.1}ms  {:>8.1}ms (avg {:.1}ms)",
            lt, count,
            *n1 as f64 / 1000.0, *blk as f64 / 1000.0,
            *n2 as f64 / 1000.0, *ffn as f64 / 1000.0,
            total as f64 / 1000.0, avg);
    }
    eprintln!("  {}", "\u{2500}".repeat(72));
    eprintln!("  {:<10} {:>5}  {:>53.1}ms",
        "ALL", profiles.len(), grand_total_us as f64 / 1000.0);

    // Sub-block aggregate per type
    // DeltaNet sub-block aggregate
    let dn_profiles: Vec<_> = profiles.iter()
        .filter_map(|p| p.block_sub.as_ref().and_then(|s| match s {
            DecodeBlockSubTimings::DeltaNet { .. } => Some(s),
            _ => None,
        }))
        .collect();
    if !dn_profiles.is_empty() {
        let mut tot_qkv = 0u64; let mut tot_z = 0u64;
        let mut tot_ab = 0u64; let mut tot_rec = 0u64; let mut tot_out = 0u64;
        for s in &dn_profiles {
            if let DecodeBlockSubTimings::DeltaNet { in_proj_qkv_us, in_proj_z_us, in_proj_ab_us, recurrent_step_us, out_proj_us } = s {
                tot_qkv += in_proj_qkv_us; tot_z += in_proj_z_us;
                tot_ab += in_proj_ab_us; tot_rec += recurrent_step_us; tot_out += out_proj_us;
            }
        }
        let block_total = tot_qkv + tot_z + tot_ab + tot_rec + tot_out;
        let n = dn_profiles.len() as f64;
        let pct = |v: u64| if block_total > 0 { v as f64 / block_total as f64 * 100.0 } else { 0.0 };
        eprintln!("\n[PROFILE-DECODE-LAYER] === DeltaNet Block breakdown ({} layers) ===", dn_profiles.len());
        eprintln!("  {:<20} {:>10} {:>10}  {:>6}", "Sub-step", "Total", "Avg", "%");
        eprintln!("  {}", "\u{2500}".repeat(52));
        for (name, v) in [("in_proj_qkv", tot_qkv), ("in_proj_z", tot_z), ("in_proj_ab", tot_ab), ("recurrent_step", tot_rec), ("out_proj", tot_out)] {
            eprintln!("  {:<20} {:>8.1}ms {:>8.1}ms  {:>5.1}%", name, v as f64/1000.0, v as f64/n/1000.0, pct(v));
        }
        eprintln!("  {}", "\u{2500}".repeat(52));
        eprintln!("  {:<20} {:>8.1}ms {:>8.1}ms", "TOTAL", block_total as f64/1000.0, block_total as f64/n/1000.0);
    }

    // Attention sub-block aggregate
    let attn_profiles: Vec<_> = profiles.iter()
        .filter_map(|p| p.block_sub.as_ref().and_then(|s| match s {
            DecodeBlockSubTimings::Attn { .. } => Some(s),
            _ => None,
        }))
        .collect();
    if !attn_profiles.is_empty() {
        let mut tot_qkv = 0u64; let mut tot_qkn = 0u64; let mut tot_rope = 0u64;
        let mut tot_kva = 0u64; let mut tot_gate = 0u64; let mut tot_out = 0u64;
        for s in &attn_profiles {
            if let DecodeBlockSubTimings::Attn { qkv_proj_us, qk_norm_us, rope_us, kv_cache_attn_us, output_gate_us, o_proj_us } = s {
                tot_qkv += qkv_proj_us; tot_qkn += qk_norm_us; tot_rope += rope_us;
                tot_kva += kv_cache_attn_us; tot_gate += output_gate_us; tot_out += o_proj_us;
            }
        }
        let block_total = tot_qkv + tot_qkn + tot_rope + tot_kva + tot_gate + tot_out;
        let n = attn_profiles.len() as f64;
        let pct = |v: u64| if block_total > 0 { v as f64 / block_total as f64 * 100.0 } else { 0.0 };
        eprintln!("\n[PROFILE-DECODE-LAYER] === Attention Block breakdown ({} layers) ===", attn_profiles.len());
        eprintln!("  {:<20} {:>10} {:>10}  {:>6}", "Sub-step", "Total", "Avg", "%");
        eprintln!("  {}", "\u{2500}".repeat(52));
        for (name, v) in [("qkv_proj", tot_qkv), ("qk_norm", tot_qkn), ("rope", tot_rope), ("kv+attn_kernel", tot_kva), ("output_gate", tot_gate), ("o_proj", tot_out)] {
            eprintln!("  {:<20} {:>8.1}ms {:>8.1}ms  {:>5.1}%", name, v as f64/1000.0, v as f64/n/1000.0, pct(v));
        }
        eprintln!("  {}", "\u{2500}".repeat(52));
        eprintln!("  {:<20} {:>8.1}ms {:>8.1}ms", "TOTAL", block_total as f64/1000.0, block_total as f64/n/1000.0);
    }
}

// ============================================================================
// int8 KV Cache profiling (feature: profile-kvcache-int8)
// ============================================================================

#[cfg(any(feature = "profile-kvcache-int8", feature = "profile-attn-kernel", feature = "profile-moe-decode"))]
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(feature = "profile-kvcache-int8")]
static KVCACHE_INT8_SCORES_US: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "profile-kvcache-int8")]
static KVCACHE_INT8_SOFTMAX_US: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "profile-kvcache-int8")]
static KVCACHE_INT8_OUTPUT_US: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "profile-kvcache-int8")]
static KVCACHE_INT8_CALLS: AtomicU64 = AtomicU64::new(0);

/// Record int8 scores computation time (Q·K^T with dequantization).
#[cfg(feature = "profile-kvcache-int8")]
pub fn push_kvcache_int8_scores(us: u64) {
    KVCACHE_INT8_SCORES_US.fetch_add(us, Ordering::Relaxed);
    KVCACHE_INT8_CALLS.fetch_add(1, Ordering::Relaxed);
}

/// Record int8 softmax time.
#[cfg(feature = "profile-kvcache-int8")]
pub fn push_kvcache_int8_softmax(us: u64) {
    KVCACHE_INT8_SOFTMAX_US.fetch_add(us, Ordering::Relaxed);
}

/// Record int8 output aggregation time.
#[cfg(feature = "profile-kvcache-int8")]
pub fn push_kvcache_int8_output(us: u64) {
    KVCACHE_INT8_OUTPUT_US.fetch_add(us, Ordering::Relaxed);
}

/// Print int8 KV cache profiling stats.
#[cfg(feature = "profile-kvcache-int8")]
pub fn report_kvcache_int8() {
    let scores_us = KVCACHE_INT8_SCORES_US.load(Ordering::Relaxed);
    let softmax_us = KVCACHE_INT8_SOFTMAX_US.load(Ordering::Relaxed);
    let output_us = KVCACHE_INT8_OUTPUT_US.load(Ordering::Relaxed);
    let calls = KVCACHE_INT8_CALLS.load(Ordering::Relaxed);

    if calls == 0 {
        return;
    }

    let total_us = scores_us + softmax_us + output_us;
    let total_ms = total_us as f64 / 1000.0;

    eprintln!("");
    eprintln!("[PROFILE] int8 KV Cache Decode Breakdown ({} calls):", calls);
    eprintln!("[PROFILE]   Scores (Q·K^T + deq): {:>8.1} ms  ({:>5.1}%)",
        scores_us as f64 / 1000.0,
        if total_us > 0 { scores_us as f64 / total_us as f64 * 100.0 } else { 0.0 }
    );
    eprintln!("[PROFILE]   Softmax:              {:>8.1} ms  ({:>5.1}%)",
        softmax_us as f64 / 1000.0,
        if total_us > 0 { softmax_us as f64 / total_us as f64 * 100.0 } else { 0.0 }
    );
    eprintln!("[PROFILE]   Output (agg + deq):   {:>8.1} ms  ({:>5.1}%)",
        output_us as f64 / 1000.0,
        if total_us > 0 { output_us as f64 / total_us as f64 * 100.0 } else { 0.0 }
    );
    eprintln!("[PROFILE]   {}", "\u{2500}".repeat(44));
    eprintln!("[PROFILE]   TOTAL:                {:>8.1} ms", total_ms);
    eprintln!("[PROFILE]   Per-call avg:         {:>8.3} ms", total_ms / calls as f64);
}

// Report on demand (called from CLI at exit)
#[cfg(feature = "profile-kvcache-int8")]
pub fn flush_kvcache_int8_stats() {
    report_kvcache_int8();
}

// ============================================================================
// Attention kernel breakdown profiling (feature: profile-attn-kernel)
// ============================================================================

#[cfg(feature = "profile-attn-kernel")]
static ATTN_KERNEL_DOT_NS: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "profile-attn-kernel")]
static ATTN_KERNEL_EXP_NS: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "profile-attn-kernel")]
static ATTN_KERNEL_SV_NS: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "profile-attn-kernel")]
static ATTN_KERNEL_NORM_NS: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "profile-attn-kernel")]
static ATTN_KERNEL_OVERHEAD_NS: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "profile-attn-kernel")]
static ATTN_KERNEL_TOTAL_ITERS: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "profile-attn-kernel")]
static ATTN_KERNEL_SAMPLED_ITERS: AtomicU64 = AtomicU64::new(0);

/// Called once per thread chunk to flush thread-local accumulators.
#[cfg(feature = "profile-attn-kernel")]
pub fn push_attn_kernel_breakdown(
    dot_ns: u64,
    exp_ns: u64,
    sv_ns: u64,
    norm_ns: u64,
    overhead_ns: u64,
    total_iters: u64,
    sampled_iters: u64,
) {
    ATTN_KERNEL_DOT_NS.fetch_add(dot_ns, Ordering::Relaxed);
    ATTN_KERNEL_EXP_NS.fetch_add(exp_ns, Ordering::Relaxed);
    ATTN_KERNEL_SV_NS.fetch_add(sv_ns, Ordering::Relaxed);
    ATTN_KERNEL_NORM_NS.fetch_add(norm_ns, Ordering::Relaxed);
    ATTN_KERNEL_OVERHEAD_NS.fetch_add(overhead_ns, Ordering::Relaxed);
    ATTN_KERNEL_TOTAL_ITERS.fetch_add(total_iters, Ordering::Relaxed);
    ATTN_KERNEL_SAMPLED_ITERS.fetch_add(sampled_iters, Ordering::Relaxed);
}

/// Reset accumulators before a new prefill pass.
#[cfg(feature = "profile-attn-kernel")]
pub fn reset_attn_kernel_breakdown() {
    ATTN_KERNEL_DOT_NS.store(0, Ordering::Relaxed);
    ATTN_KERNEL_EXP_NS.store(0, Ordering::Relaxed);
    ATTN_KERNEL_SV_NS.store(0, Ordering::Relaxed);
    ATTN_KERNEL_NORM_NS.store(0, Ordering::Relaxed);
    ATTN_KERNEL_OVERHEAD_NS.store(0, Ordering::Relaxed);
    ATTN_KERNEL_TOTAL_ITERS.store(0, Ordering::Relaxed);
    ATTN_KERNEL_SAMPLED_ITERS.store(0, Ordering::Relaxed);
}

/// Print attention kernel breakdown.
/// On x86_64 with fused kernels: DOT/EXP/SV are rdtsc cycle counts (sampled == total).
/// On aarch64 or fallback: DOT/EXP/SV are ns with 1/64 sampling extrapolation.
#[cfg(feature = "profile-attn-kernel")]
pub fn report_attn_kernel_breakdown() {
    let sampled = ATTN_KERNEL_SAMPLED_ITERS.load(Ordering::Relaxed);
    let total = ATTN_KERNEL_TOTAL_ITERS.load(Ordering::Relaxed);
    if sampled == 0 || total == 0 {
        return;
    }

    let rdtsc_mode = sampled == total;
    let ratio = if rdtsc_mode { 1.0 } else { total as f64 / sampled as f64 };
    let dot_raw = ATTN_KERNEL_DOT_NS.load(Ordering::Relaxed) as f64 * ratio;
    let exp_raw = ATTN_KERNEL_EXP_NS.load(Ordering::Relaxed) as f64 * ratio;
    let sv_raw = ATTN_KERNEL_SV_NS.load(Ordering::Relaxed) as f64 * ratio;
    let norm_ns = ATTN_KERNEL_NORM_NS.load(Ordering::Relaxed) as f64;  // always ns
    let overhead_ns = ATTN_KERNEL_OVERHEAD_NS.load(Ordering::Relaxed) as f64 * ratio;

    eprintln!();
    if rdtsc_mode {
        // rdtsc mode: DOT/EXP/SV are in TSC cycles, report as Mcycles + percentages
        let kernel_total = dot_raw + exp_raw + sv_raw;
        let to_mc = |cy: f64| cy / 1_000_000.0;
        let pct = |cy: f64| {
            if kernel_total > 0.0 { cy / kernel_total * 100.0 } else { 0.0 }
        };

        eprintln!(
            "[PROFILE-ATTN-KERNEL] Attention kernel breakdown (rdtsc, {} calls across all threads):",
            total
        );
        eprintln!(
            "  {:<16} {:>12}  {:>6}",
            "Phase", "Mcycles", "%"
        );
        eprintln!("  {}", "\u{2500}".repeat(38));
        eprintln!(
            "  {:<16} {:>10.1} Mc  {:>5.1}%",
            "DOT+HSUM", to_mc(dot_raw), pct(dot_raw)
        );
        eprintln!(
            "  {:<16} {:>10.1} Mc  {:>5.1}%",
            "EXP (softmax)", to_mc(exp_raw), pct(exp_raw)
        );
        eprintln!(
            "  {:<16} {:>10.1} Mc  {:>5.1}%",
            "SV (score·V)", to_mc(sv_raw), pct(sv_raw)
        );
        eprintln!("  {}", "\u{2500}".repeat(38));
        eprintln!(
            "  {:<16} {:>10.1} Mc",
            "TOTAL", to_mc(kernel_total)
        );
        if norm_ns > 0.0 {
            eprintln!(
                "  {:<16} {:>8.1} ms  (NORM, not included above)",
                "NORM (final)", norm_ns / 1_000_000.0
            );
        }
    } else {
        // Sampling mode: DOT/EXP/SV are in ns, extrapolate and report as ms
        let kernel_total = dot_raw + exp_raw + sv_raw + norm_ns;
        let to_ms = |ns: f64| ns / 1_000_000.0;
        let pct = |ns: f64| {
            if kernel_total > 0.0 { ns / kernel_total * 100.0 } else { 0.0 }
        };

        eprintln!(
            "[PROFILE-ATTN-KERNEL] Attention kernel breakdown (sampled {}/{} iters, extrapolated {:.1}x):",
            sampled, total, ratio
        );
        eprintln!(
            "  {:<16} {:>10}  {:>6}",
            "Phase", "Duration", "%"
        );
        eprintln!("  {}", "\u{2500}".repeat(36));
        eprintln!(
            "  {:<16} {:>8.1} ms  {:>5.1}%",
            "DOT (Q·K^T)", to_ms(dot_raw), pct(dot_raw)
        );
        eprintln!(
            "  {:<16} {:>8.1} ms  {:>5.1}%",
            "EXP (softmax)", to_ms(exp_raw), pct(exp_raw)
        );
        eprintln!(
            "  {:<16} {:>8.1} ms  {:>5.1}%",
            "SV (score·V)", to_ms(sv_raw), pct(sv_raw)
        );
        eprintln!(
            "  {:<16} {:>8.1} ms  {:>5.1}%",
            "NORM (final)", to_ms(norm_ns), pct(norm_ns)
        );
        eprintln!("  {}", "\u{2500}".repeat(36));
        eprintln!(
            "  {:<16} {:>8.1} ms",
            "TOTAL", to_ms(kernel_total)
        );
        eprintln!(
            "  {:<16} {:>8.1} ms  (timing overhead, not included above)",
            "overhead", to_ms(overhead_ns)
        );
    }
}

// ============================================================================
// MoE decode profiling (feature: profile-moe-decode)
// ============================================================================

#[cfg(feature = "profile-moe-decode")]
static MOE_DEC_ROUTER_US: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "profile-moe-decode")]
static MOE_DEC_DISPATCH_US: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "profile-moe-decode")]
static MOE_DEC_REDUCE_US: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "profile-moe-decode")]
static MOE_DEC_TOTAL_US: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "profile-moe-decode")]
static MOE_DEC_CALLS: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "profile-moe-decode")]
static MOE_DEC_EXPERTS: AtomicU64 = AtomicU64::new(0);
// Per-expert kernel breakdown (accumulated across all workers)
#[cfg(feature = "profile-moe-decode")]
static MOE_DEC_GATE_US: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "profile-moe-decode")]
static MOE_DEC_UP_US: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "profile-moe-decode")]
static MOE_DEC_SWIGLU_US: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "profile-moe-decode")]
static MOE_DEC_DOWN_US: AtomicU64 = AtomicU64::new(0);

/// Record one MoE decode call's outer timing.
#[cfg(feature = "profile-moe-decode")]
pub fn push_moe_decode(router_us: u64, dispatch_us: u64, reduce_us: u64, total_us: u64, n_experts: u64) {
    MOE_DEC_ROUTER_US.fetch_add(router_us, Ordering::Relaxed);
    MOE_DEC_DISPATCH_US.fetch_add(dispatch_us, Ordering::Relaxed);
    MOE_DEC_REDUCE_US.fetch_add(reduce_us, Ordering::Relaxed);
    MOE_DEC_TOTAL_US.fetch_add(total_us, Ordering::Relaxed);
    MOE_DEC_CALLS.fetch_add(1, Ordering::Relaxed);
    MOE_DEC_EXPERTS.fetch_add(n_experts, Ordering::Relaxed);
}

/// Record per-expert kernel breakdown (called from parallel workers).
#[cfg(feature = "profile-moe-decode")]
pub fn push_moe_decode_kernel(gate_us: u64, up_us: u64, swiglu_us: u64, down_us: u64) {
    MOE_DEC_GATE_US.fetch_add(gate_us, Ordering::Relaxed);
    MOE_DEC_UP_US.fetch_add(up_us, Ordering::Relaxed);
    MOE_DEC_SWIGLU_US.fetch_add(swiglu_us, Ordering::Relaxed);
    MOE_DEC_DOWN_US.fetch_add(down_us, Ordering::Relaxed);
}

/// Print MoE decode profiling stats and reset.
#[cfg(feature = "profile-moe-decode")]
pub fn report_moe_decode() {
    let calls = MOE_DEC_CALLS.swap(0, Ordering::Relaxed);
    if calls == 0 {
        return;
    }

    let router_us = MOE_DEC_ROUTER_US.swap(0, Ordering::Relaxed);
    let dispatch_us = MOE_DEC_DISPATCH_US.swap(0, Ordering::Relaxed);
    let reduce_us = MOE_DEC_REDUCE_US.swap(0, Ordering::Relaxed);
    let total_us = MOE_DEC_TOTAL_US.swap(0, Ordering::Relaxed);
    let experts = MOE_DEC_EXPERTS.swap(0, Ordering::Relaxed);

    let gate_us = MOE_DEC_GATE_US.swap(0, Ordering::Relaxed);
    let up_us = MOE_DEC_UP_US.swap(0, Ordering::Relaxed);
    let swiglu_us = MOE_DEC_SWIGLU_US.swap(0, Ordering::Relaxed);
    let down_us = MOE_DEC_DOWN_US.swap(0, Ordering::Relaxed);

    let to_ms = |us: u64| us as f64 / 1000.0;
    let pct = |part: u64, whole: u64| {
        if whole > 0 { part as f64 / whole as f64 * 100.0 } else { 0.0 }
    };
    let avg_experts = experts as f64 / calls as f64;
    let avg_total_us = total_us as f64 / calls as f64;

    eprintln!();
    eprintln!(
        "[PROFILE-MOE-DECODE] MoE decode breakdown ({} calls, avg {:.1} experts/call, avg {:.0}us/call):",
        calls, avg_experts, avg_total_us
    );
    eprintln!("  {:<16} {:>10}  {:>6}", "Phase", "Total ms", "%");
    eprintln!("  {}", "\u{2500}".repeat(36));
    eprintln!(
        "  {:<16} {:>8.1} ms  {:>5.1}%",
        "Router", to_ms(router_us), pct(router_us, total_us)
    );
    eprintln!(
        "  {:<16} {:>8.1} ms  {:>5.1}%",
        "Dispatch", to_ms(dispatch_us), pct(dispatch_us, total_us)
    );
    eprintln!(
        "  {:<16} {:>8.1} ms  {:>5.1}%",
        "Reduce", to_ms(reduce_us), pct(reduce_us, total_us)
    );
    let overhead_us = total_us.saturating_sub(router_us + dispatch_us + reduce_us);
    eprintln!(
        "  {:<16} {:>8.1} ms  {:>5.1}%",
        "Overhead", to_ms(overhead_us), pct(overhead_us, total_us)
    );
    eprintln!("  {}", "\u{2500}".repeat(36));
    eprintln!("  {:<16} {:>8.1} ms", "TOTAL", to_ms(total_us));

    // Per-expert kernel breakdown
    let kernel_total = gate_us + up_us + swiglu_us + down_us;
    if kernel_total > 0 {
        eprintln!();
        eprintln!(
            "  Expert kernel breakdown (sum of all worker-time, {} expert calls):",
            experts
        );
        eprintln!("  {:<16} {:>10}  {:>6}", "Kernel", "Total ms", "%");
        eprintln!("  {}", "\u{2500}".repeat(36));
        eprintln!(
            "  {:<16} {:>8.1} ms  {:>5.1}%",
            "gate_proj", to_ms(gate_us), pct(gate_us, kernel_total)
        );
        eprintln!(
            "  {:<16} {:>8.1} ms  {:>5.1}%",
            "up_proj", to_ms(up_us), pct(up_us, kernel_total)
        );
        eprintln!(
            "  {:<16} {:>8.1} ms  {:>5.1}%",
            "SwiGLU", to_ms(swiglu_us), pct(swiglu_us, kernel_total)
        );
        eprintln!(
            "  {:<16} {:>8.1} ms  {:>5.1}%",
            "down_proj", to_ms(down_us), pct(down_us, kernel_total)
        );
        eprintln!("  {}", "\u{2500}".repeat(36));
        eprintln!("  {:<16} {:>8.1} ms", "TOTAL", to_ms(kernel_total));
        let avg_per_expert = kernel_total as f64 / experts as f64;
        eprintln!("  avg/expert:    {:.0} us", avg_per_expert);
    }
}
