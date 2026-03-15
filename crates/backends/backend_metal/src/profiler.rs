//! Per-kernel GPU profiling for Metal decode.
//!
//! - `METAL_PROFILE=1`: coarse (attn / mlp per layer)
//! - `METAL_PROFILE=2`: fine-grained (norm, qkv, rope, attn_k, oproj,
//!                       norm2, router, gateup, down, resid per layer)
//! - `METAL_PROFILE=3`: same detail as =2, but uses GPU counter sampling
//!                       (zero overhead — no command buffer splitting)
//!
//! Usage: `METAL_PROFILE=3 herbert --backend metal-q4 ...`

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Once;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSRange;
use objc2_metal::*;

use herbert_core::error::{HerbertError, Result};

use crate::context::MetalContext;

// ---------------------------------------------------------------------------
// Global enable flags (0=off, 1=coarse, 2=detailed, 3=counter sampling)
// ---------------------------------------------------------------------------

static INIT: Once = Once::new();
static LEVEL: AtomicU8 = AtomicU8::new(0);

fn ensure_init() {
    INIT.call_once(|| {
        let lvl = std::env::var("METAL_PROFILE")
            .ok()
            .and_then(|v| v.parse::<u8>().ok())
            .unwrap_or(0);
        if lvl >= 1 {
            LEVEL.store(lvl.min(3), Ordering::Relaxed);
            let mode = match lvl {
                1 => "coarse",
                2 => "detailed (CB splitting)",
                _ => "detailed (GPU counters)",
            };
            eprintln!("[metal-profile] GPU profiling enabled ({mode})");
        }
    });
}

/// Returns true if any profiling is active (level >= 1).
pub fn is_enabled() -> bool {
    ensure_init();
    LEVEL.load(Ordering::Relaxed) >= 1
}

/// Returns true if detailed sub-phase profiling is active (level >= 2).
pub fn is_detailed() -> bool {
    ensure_init();
    LEVEL.load(Ordering::Relaxed) >= 2
}

// ---------------------------------------------------------------------------
// Raw GPU timing via msg_send (avoids objc2-core-foundation dependency)
// ---------------------------------------------------------------------------

unsafe fn gpu_start_time(cb: &ProtocolObject<dyn MTLCommandBuffer>) -> f64 {
    objc2::msg_send![cb, GPUStartTime]
}

unsafe fn gpu_end_time(cb: &ProtocolObject<dyn MTLCommandBuffer>) -> f64 {
    objc2::msg_send![cb, GPUEndTime]
}

// ---------------------------------------------------------------------------
// ProfiledEncoder
// ---------------------------------------------------------------------------

struct ProfileEntry {
    label: String,
    cb: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
}

/// Wraps a Metal compute encoder with optional per-phase GPU profiling.
///
/// - Levels 1-2: `mark()` commits the current CB and starts a new one.
/// - Level 3: `mark()` inserts an inline GPU counter sample (no CB splitting).
pub struct ProfiledEncoder<'a> {
    ctx: &'a MetalContext,
    cb: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    encoder: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
    // CB-splitting mode (levels 1 and 2)
    entries: Vec<ProfileEntry>,
    // Counter sampling mode (level 3)
    counter_labels: Vec<String>,
    counter_sample_idx: usize,
    // Common
    profiling: bool,
    detailed: bool,
    counter_mode: bool,
    token_idx: u32,
    finished: bool,
}

impl<'a> ProfiledEncoder<'a> {
    pub fn new(ctx: &'a MetalContext, token_idx: u32) -> Result<Self> {
        ensure_init();
        let lvl = LEVEL.load(Ordering::Relaxed);
        let cb = ctx.begin_command_buffer()?;
        let encoder = MetalContext::new_compute_encoder(&cb)?;

        // Level 3 uses counter sampling if GPU counters are available.
        // If counters weren't set up (unsupported), fall back to level 2.
        let counter_mode = lvl >= 3 && ctx.gpu_counters.is_some();
        let effective_detailed = lvl >= 2 || counter_mode;

        let mut pe = Self {
            ctx,
            cb,
            encoder,
            entries: Vec::new(),
            counter_labels: Vec::new(),
            counter_sample_idx: 0,
            profiling: lvl >= 1,
            detailed: effective_detailed,
            counter_mode,
            token_idx,
            finished: false,
        };

        // Insert initial timestamp sample (sample index 0 = start of first phase)
        if counter_mode {
            pe.sample_counter();
        }

        Ok(pe)
    }

    #[inline]
    pub fn enc(&self) -> &ProtocolObject<dyn MTLComputeCommandEncoder> {
        &self.encoder
    }

    #[inline]
    pub fn is_detailed(&self) -> bool {
        self.detailed
    }

    /// Insert a profiling boundary (always, when profiling is enabled).
    pub fn mark(&mut self, label: &str) -> Result<()> {
        if !self.profiling {
            return Ok(());
        }
        if self.counter_mode {
            self.sample_counter();
            self.counter_labels.push(label.to_string());
            return Ok(());
        }
        self.split_cb(label)
    }

    /// Insert a profiling boundary only in detailed mode (METAL_PROFILE >= 2).
    pub fn mark_detail(&mut self, label: &str) -> Result<()> {
        if !self.detailed {
            return Ok(());
        }
        if self.counter_mode {
            self.sample_counter();
            self.counter_labels.push(label.to_string());
            return Ok(());
        }
        self.split_cb(label)
    }

    /// Insert an inline GPU counter sample at the current point in the encoder.
    fn sample_counter(&mut self) {
        if let Some(ref counters) = self.ctx.gpu_counters {
            if self.counter_sample_idx < counters.max_samples {
                unsafe {
                    self.encoder.sampleCountersInBuffer_atSampleIndex_withBarrier(
                        &counters.sample_buffer,
                        self.counter_sample_idx,
                        true, // barrier for repeatable results
                    );
                }
                self.counter_sample_idx += 1;
            }
        }
    }

    fn split_cb(&mut self, label: &str) -> Result<()> {
        self.encoder.endEncoding();
        self.cb.commit();

        let new_cb = self.ctx.begin_command_buffer()?;
        let new_enc = MetalContext::new_compute_encoder(&new_cb)?;

        let old_cb = std::mem::replace(&mut self.cb, new_cb);
        let _old_enc = std::mem::replace(&mut self.encoder, new_enc);

        self.entries.push(ProfileEntry {
            label: label.to_string(),
            cb: old_cb,
        });

        Ok(())
    }

    /// Submit the final CB, wait, and print timing.
    pub fn finish(&mut self, final_label: &str) -> Result<()> {
        assert!(!self.finished, "ProfiledEncoder::finish() called twice");
        self.finished = true;

        if self.counter_mode {
            // Insert final sample + label, then submit
            self.sample_counter();
            self.counter_labels.push(final_label.to_string());

            self.encoder.endEncoding();
            self.cb.commit();
            self.cb.waitUntilCompleted();

            if let Some(error) = self.cb.error() {
                return Err(HerbertError::Backend(
                    format!("Metal command buffer error: {:?}", error),
                ));
            }

            self.finish_counter_profiling();
            return Ok(());
        }

        self.encoder.endEncoding();
        self.cb.commit();
        self.cb.waitUntilCompleted();

        if let Some(error) = self.cb.error() {
            return Err(HerbertError::Backend(
                format!("Metal command buffer error: {:?}", error),
            ));
        }

        if !self.profiling {
            return Ok(());
        }

        // Collect timing from all CBs (all completed — serial queue).
        let mut timings: Vec<(String, f64)> = Vec::with_capacity(self.entries.len() + 1);

        for entry in &self.entries {
            let ms = unsafe {
                let s = gpu_start_time(&entry.cb);
                let e = gpu_end_time(&entry.cb);
                (e - s) * 1000.0
            };
            timings.push((entry.label.clone(), ms));
        }

        let ms = unsafe {
            let s = gpu_start_time(&self.cb);
            let e = gpu_end_time(&self.cb);
            (e - s) * 1000.0
        };
        timings.push((final_label.to_string(), ms));

        if self.detailed {
            Self::print_detailed_report(self.token_idx, &timings);
        } else {
            Self::print_coarse_report(self.token_idx, &timings);
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Counter profiling (METAL_PROFILE=3) — resolve timestamps, compute deltas
    // -----------------------------------------------------------------------

    fn finish_counter_profiling(&self) {
        let counters = match self.ctx.gpu_counters {
            Some(ref c) => c,
            None => return,
        };

        let n_samples = self.counter_sample_idx;
        let n_labels = self.counter_labels.len();
        // We have n_labels phases, needing n_labels + 1 samples
        // (sample 0 = start, samples 1..=n_labels = end of each phase)
        if n_samples < 2 || n_labels == 0 {
            eprintln!("[metal-profile] No counter samples collected");
            return;
        }

        // Resolve all samples
        let data = unsafe {
            counters.sample_buffer.resolveCounterRange(
                NSRange::new(0, n_samples),
            )
        };

        let data = match data {
            Some(d) => d,
            None => {
                eprintln!("[metal-profile] Failed to resolve counter samples");
                return;
            }
        };

        // Parse timestamps. Each sample is an MTLCounterResultTimestamp (8 bytes).
        let expected_len = n_samples * std::mem::size_of::<MTLCounterResultTimestamp>();
        let actual_len = data.len();
        if actual_len < expected_len {
            eprintln!(
                "[metal-profile] Resolved data too short: {} bytes, expected {}",
                actual_len, expected_len,
            );
            return;
        }

        // SAFETY: data is not mutated while we hold this slice (we own the Retained<NSData>)
        let bytes = unsafe { data.as_bytes_unchecked() };
        let timestamps: &[MTLCounterResultTimestamp] = unsafe {
            std::slice::from_raw_parts(
                bytes.as_ptr() as *const MTLCounterResultTimestamp,
                n_samples,
            )
        };

        // Build timings from consecutive timestamp deltas
        let error_value: u64 = !0; // MTLCounterErrorValue
        let mut timings: Vec<(String, f64)> = Vec::with_capacity(n_labels);
        let mut errors = 0u32;

        for i in 0..n_labels {
            let t_start = timestamps[i].timestamp;
            let t_end = timestamps[i + 1].timestamp;

            if t_start == error_value || t_end == error_value {
                timings.push((self.counter_labels[i].clone(), 0.0));
                errors += 1;
                continue;
            }

            // GPU timestamps on Apple Silicon are in nanoseconds
            let delta_ns = if t_end >= t_start {
                t_end - t_start
            } else {
                // Wraparound (shouldn't happen for u64 ns, but be safe)
                0
            };
            let delta_ms = delta_ns as f64 / 1_000_000.0;
            timings.push((self.counter_labels[i].clone(), delta_ms));
        }

        if errors > 0 {
            eprintln!(
                "[metal-profile] WARNING: {} samples returned MTLCounterErrorValue",
                errors,
            );
        }

        // Print using the same detailed report format
        Self::print_detailed_report(self.token_idx, &timings);
    }

    // -----------------------------------------------------------------------
    // Coarse report (METAL_PROFILE=1) — same as before
    // -----------------------------------------------------------------------

    fn print_coarse_report(token_idx: u32, timings: &[(String, f64)]) {
        let total_ms: f64 = timings.iter().map(|(_, ms)| ms).sum();

        let mut attn_times: Vec<f64> = Vec::new();
        let mut mlp_times: Vec<f64> = Vec::new();
        let mut lm_head_ms = 0.0f64;
        let mut other_ms = 0.0f64;

        for (label, ms) in timings {
            if label.ends_with("_attn") {
                attn_times.push(*ms);
            } else if label.ends_with("_mlp") {
                mlp_times.push(*ms);
            } else if label == "lm_head" {
                lm_head_ms = *ms;
            } else {
                other_ms += ms;
            }
        }

        let attn_total: f64 = attn_times.iter().sum();
        let mlp_total: f64 = mlp_times.iter().sum();

        eprintln!(
            "[PROFILE] Token {} — total {:.1}ms GPU",
            token_idx, total_ms,
        );

        for (label, ms) in timings {
            if !label.ends_with("_attn") && !label.ends_with("_mlp") {
                eprintln!("  {:<16} {:>7.2}ms", label, ms);
            }
        }

        if !attn_times.is_empty() {
            let n = attn_times.len();
            let avg = attn_total / n as f64;
            let min = attn_times.iter().copied().fold(f64::INFINITY, f64::min);
            let max = attn_times.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            eprintln!(
                "  attn x{}:       {:>7.2}ms total, avg {:.2}ms [min {:.2} max {:.2}]",
                n, attn_total, avg, min, max,
            );
        }

        if !mlp_times.is_empty() {
            let n = mlp_times.len();
            let avg = mlp_total / n as f64;
            let min = mlp_times.iter().copied().fold(f64::INFINITY, f64::min);
            let max = mlp_times.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            eprintln!(
                "  mlp  x{}:       {:>7.2}ms total, avg {:.2}ms [min {:.2} max {:.2}]",
                n, mlp_total, avg, min, max,
            );
        }

        let pct = |v: f64| if total_ms > 0.0 { v / total_ms * 100.0 } else { 0.0 };
        eprintln!(
            "  SUMMARY: attn={:.1}ms ({:.0}%) mlp={:.1}ms ({:.0}%) lm_head={:.1}ms ({:.0}%) other={:.1}ms ({:.0}%)",
            attn_total, pct(attn_total),
            mlp_total, pct(mlp_total),
            lm_head_ms, pct(lm_head_ms),
            other_ms, pct(other_ms),
        );
    }

    // -----------------------------------------------------------------------
    // Detailed report (METAL_PROFILE=2 and =3) — per-sub-phase aggregation
    // -----------------------------------------------------------------------

    fn print_detailed_report(token_idx: u32, timings: &[(String, f64)]) {
        let total_ms: f64 = timings.iter().map(|(_, ms)| ms).sum();

        // Separate layer entries (L{n}_{suffix}) from global entries
        let mut sub_phases: BTreeMap<String, Vec<f64>> = BTreeMap::new();
        let mut global_entries: Vec<(&str, f64)> = Vec::new();

        for (label, ms) in timings {
            if let Some(pos) = label.find('_') {
                if label.starts_with('L') && label[1..pos].parse::<u32>().is_ok() {
                    let suffix = &label[pos + 1..];
                    sub_phases.entry(suffix.to_string()).or_default().push(*ms);
                    continue;
                }
            }
            global_entries.push((label.as_str(), *ms));
        }

        eprintln!(
            "[PROFILE] Token {} — total {:.1}ms GPU (detailed)",
            token_idx, total_ms,
        );

        // Global entries
        for (label, ms) in &global_entries {
            eprintln!("  {:<16} {:>7.3}ms", label, ms);
        }

        // Sub-phase table
        if !sub_phases.is_empty() {
            eprintln!("  --- per-layer sub-phases ---");

            // Compute totals for percentage
            let layer_total: f64 = sub_phases.values().flatten().sum();

            // Define display order (attention phases first, then MLP phases)
            let order = [
                "norm1", "qkv", "rope", "attn_k", "oproj",
                "norm2", "router", "gate", "up", "swiglu", "gateup", "down", "resid",
            ];

            let print_phase = |suffix: &str, times: &[f64]| {
                let n = times.len();
                let tot: f64 = times.iter().sum();
                let avg = tot / n as f64;
                let min = times.iter().copied().fold(f64::INFINITY, f64::min);
                let max = times.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let pct = if layer_total > 0.0 { tot / layer_total * 100.0 } else { 0.0 };
                eprintln!(
                    "    {:<10} x{:<3} {:>7.2}ms total ({:>4.1}%)  avg {:.3}ms [min {:.3} max {:.3}]",
                    suffix, n, tot, pct, avg, min, max,
                );
            };

            // Print in defined order first
            for &key in &order {
                if let Some(times) = sub_phases.get(key) {
                    print_phase(key, times);
                }
            }
            // Print any remaining phases not in the predefined order
            for (key, times) in &sub_phases {
                if !order.contains(&key.as_str()) {
                    print_phase(key, times);
                }
            }

            // Group into attention vs MLP totals
            let attn_keys = ["norm1", "qkv", "rope", "attn_k", "oproj"];
            let mlp_keys = ["norm2", "router", "gate", "up", "swiglu", "gateup", "down", "resid"];

            let sum_keys = |keys: &[&str]| -> f64 {
                keys.iter()
                    .filter_map(|k| sub_phases.get(*k))
                    .flatten()
                    .sum()
            };

            let attn_total = sum_keys(&attn_keys);
            let mlp_total = sum_keys(&mlp_keys);
            let global_total: f64 = global_entries.iter().map(|(_, ms)| ms).sum();

            let pct = |v: f64| if total_ms > 0.0 { v / total_ms * 100.0 } else { 0.0 };
            eprintln!(
                "  SUMMARY: attn={:.1}ms ({:.0}%) mlp={:.1}ms ({:.0}%) global={:.1}ms ({:.0}%)",
                attn_total, pct(attn_total),
                mlp_total, pct(mlp_total),
                global_total, pct(global_total),
            );
        }
    }
}

impl Drop for ProfiledEncoder<'_> {
    fn drop(&mut self) {
        if !self.finished {
            self.encoder.endEncoding();
        }
    }
}
