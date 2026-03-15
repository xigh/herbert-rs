//! Streaming shard-by-shard model loader with Q4 quantization at load time.
//!
//! Delegates to the generic streaming loader in `loader_common`, passing
//! a closure that quantizes each BF16 projection weight to Q4 (x86_64).
//! MoE routers use INT8 for precision. LM head uses Q4 (DRAM-bound, halving
//! data size improves throughput).
//! Supports quantized weight caching to skip quantization on subsequent loads.

use herbert_backend_common::generic_model::GenericModel;
use herbert_backend_common::loader_common;
use herbert_backend_common::weight_cache;
use herbert_core::backend::LoadOpts;
use herbert_core::config::Config;
use herbert_core::error::Result;
use std::path::Path;

use crate::kernels::KernelWeight;
use crate::ops::Q4Ops;
use crate::weight::bf16_to_f32_weight;

#[cfg(target_arch = "x86_64")]
use crate::weight::{quantize_and_pack_q4, quantize_bf16_to_int8};

fn get_backend_id() -> &'static str {
    "avx512_q4"
}

/// Tensors that should be kept at full precision (BF16→f32) instead of Q4.
fn should_keep_bf16(_name: &str) -> bool {
    false
}

/// Tensors that should use INT8 quantization (better than Q4, smaller than BF16).
/// Includes MoE router gates (small, precision-sensitive).
fn should_use_int8(name: &str) -> bool {
    name.contains(".mlp.gate.weight")           // Qwen MoE router
        || name.contains(".feed_forward.gate.weight")   // Mistral MoE router
        || name.contains(".block_sparse_moe.gate.weight")
}

/// Analyze LM head quantization risk: compare BF16 vs INT8 vs Q4 precision.
/// Triggered by env var ANALYZE_LMHEAD=1. Prints results to stderr and exits.
fn analyze_lmhead_quantization(bf16_data: &[herbert_core::tensor::BF16], n: usize, k: usize) {
    use herbert_core::tensor::bf16_to_f32;

    const GROUP_SIZE: usize = 32;
    const N_TRIALS: usize = 100;
    let n_groups = k.div_ceil(GROUP_SIZE);
    let total_elements = n * k;

    eprintln!("\n========================================================================");
    eprintln!("  LM HEAD QUANTIZATION ANALYSIS");
    eprintln!("  Matrix: {} × {} ({:.1}M elements)", n, k, total_elements as f64 / 1e6);
    eprintln!("========================================================================");

    // Simple deterministic PRNG (xorshift64)
    let mut rng_state: u64 = 0xDEADBEEF_CAFEBABE;
    let mut next_f32 = || -> f32 {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        ((rng_state >> 33) as f32) / (0x7FFF_FFFF_u32 as f32) * 2.0 - 1.0
    };

    // Generate random unit-norm hidden states
    let mut hidden_states: Vec<Vec<f32>> = Vec::with_capacity(N_TRIALS);
    for _ in 0..N_TRIALS {
        let mut h: Vec<f32> = (0..k).map(|_| next_f32()).collect();
        let norm: f32 = h.iter().map(|v| v * v).sum::<f32>().sqrt();
        h.iter_mut().for_each(|v| *v /= norm);
        hidden_states.push(h);
    }

    // Allocate per-trial logit buffers
    let mut trial_logits_bf16 = vec![vec![0.0f32; n]; N_TRIALS];
    let mut trial_logits_i8 = vec![vec![0.0f32; n]; N_TRIALS];
    let mut trial_logits_q4 = vec![vec![0.0f32; n]; N_TRIALS];

    // Error accumulators
    let mut q4_sum_sq_err = 0.0f64;
    let mut i8_sum_sq_err = 0.0f64;
    let mut q4_max_err = 0.0f32;
    let mut i8_max_err = 0.0f32;

    // Per-row error RMS
    let mut q4_row_rms = Vec::with_capacity(n);
    let mut i8_row_rms = Vec::with_capacity(n);

    // Weight statistics
    let mut w_sq_sum = 0.0f64;
    let mut w_abs_max = 0.0f32;
    let mut w_abs_sum = 0.0f64;

    // Per-group scale distributions
    let mut q4_scales_all = Vec::with_capacity(n * n_groups);
    let mut i8_scales_all = Vec::with_capacity(n * n_groups);

    let t_start = std::time::Instant::now();

    // Single pass over all rows
    for row in 0..n {
        let bf16_row = &bf16_data[row * k..(row + 1) * k];

        // Convert BF16 → f32
        let f32_row: Vec<f32> = bf16_row.iter().map(|&b| bf16_to_f32(b)).collect();

        let mut q4_row_sq_err = 0.0f64;
        let mut i8_row_sq_err = 0.0f64;

        // Dequantized row buffers
        let mut q4_dequant = Vec::with_capacity(k);
        let mut i8_dequant = Vec::with_capacity(k);

        for g in 0..n_groups {
            let start = g * GROUP_SIZE;
            let end = (start + GROUP_SIZE).min(k);
            let group = &f32_row[start..end];

            let abs_max = group.iter().map(|v| v.abs()).fold(0.0f32, f32::max);

            let q4_scale = if abs_max > 0.0 { abs_max / 7.0 } else { 1.0 };
            let i8_scale = if abs_max > 0.0 { abs_max / 127.0 } else { 1.0 };
            q4_scales_all.push(q4_scale);
            i8_scales_all.push(i8_scale);

            for &val in group {
                // Weight statistics
                w_sq_sum += (val as f64) * (val as f64);
                w_abs_sum += val.abs() as f64;
                if val.abs() > w_abs_max { w_abs_max = val.abs(); }

                // Q4: quantize to [-8, 7] (16 levels), dequantize
                let q4_q = (val / q4_scale).round().clamp(-8.0, 7.0);
                let q4_d = q4_q * q4_scale;
                q4_dequant.push(q4_d);
                let q4_err = (q4_d - val).abs();
                q4_row_sq_err += (q4_err as f64) * (q4_err as f64);
                q4_sum_sq_err += (q4_err as f64) * (q4_err as f64);
                if q4_err > q4_max_err { q4_max_err = q4_err; }

                // INT8: quantize to [-127, 127] (255 levels), dequantize
                let i8_q = (val / i8_scale).round().clamp(-127.0, 127.0);
                let i8_d = i8_q * i8_scale;
                i8_dequant.push(i8_d);
                let i8_err = (i8_d - val).abs();
                i8_row_sq_err += (i8_err as f64) * (i8_err as f64);
                i8_sum_sq_err += (i8_err as f64) * (i8_err as f64);
                if i8_err > i8_max_err { i8_max_err = i8_err; }
            }
        }

        // Accumulate logits for all trials
        for t in 0..N_TRIALS {
            let h = &hidden_states[t];
            let mut bf16_logit = 0.0f32;
            let mut q4_logit = 0.0f32;
            let mut i8_logit = 0.0f32;
            for j in 0..k {
                bf16_logit += f32_row[j] * h[j];
                q4_logit += q4_dequant[j] * h[j];
                i8_logit += i8_dequant[j] * h[j];
            }
            trial_logits_bf16[t][row] = bf16_logit;
            trial_logits_q4[t][row] = q4_logit;
            trial_logits_i8[t][row] = i8_logit;
        }

        q4_row_rms.push((q4_row_sq_err / k as f64).sqrt());
        i8_row_rms.push((i8_row_sq_err / k as f64).sqrt());

        if row % 10000 == 0 {
            eprint!("\r  Processing row {}/{} ...", row, n);
        }
    }

    let elapsed = t_start.elapsed();
    eprintln!("\r  Processed {} rows in {:.1}s                    ", n, elapsed.as_secs_f64());

    // === Weight Distribution ===
    let w_rms = (w_sq_sum / total_elements as f64).sqrt();
    let w_mean_abs = w_abs_sum / total_elements as f64;

    eprintln!("\n--- Weight Distribution ---");
    eprintln!("  RMS:      {:.6}", w_rms);
    eprintln!("  Mean|w|:  {:.6}", w_mean_abs);
    eprintln!("  Max|w|:   {:.6}", w_abs_max);

    // === Per-Element Error ===
    let q4_rms_err = (q4_sum_sq_err / total_elements as f64).sqrt();
    let i8_rms_err = (i8_sum_sq_err / total_elements as f64).sqrt();
    let q4_snr = 20.0 * (w_rms / q4_rms_err).log10();
    let i8_snr = 20.0 * (w_rms / i8_rms_err).log10();

    eprintln!("\n--- Per-Element Quantization Error ---");
    eprintln!("  {:>6}  {:>14}  {:>14}  {:>10}", "", "RMS error", "Max error", "SNR (dB)");
    eprintln!("  {:>6}  {:>14.8}  {:>14.8}  {:>10.1}", "INT8", i8_rms_err, i8_max_err, i8_snr);
    eprintln!("  {:>6}  {:>14.8}  {:>14.8}  {:>10.1}", "Q4", q4_rms_err, q4_max_err, q4_snr);
    eprintln!("  Q4/INT8 error ratio: {:.1}×", q4_rms_err / i8_rms_err);

    // === Per-Row Error Distribution ===
    let mut q4_sorted = q4_row_rms.clone();
    let mut i8_sorted = i8_row_rms.clone();
    q4_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    i8_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let p = |sorted: &[f64], pct: f64| -> f64 {
        let idx = ((sorted.len() as f64 * pct / 100.0) as usize).min(sorted.len() - 1);
        sorted[idx]
    };

    eprintln!("\n--- Per-Row RMS Error Distribution ---");
    eprintln!("  {:>6}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}", "", "min", "p50", "p95", "p99", "max");
    eprintln!("  {:>6}  {:>10.7}  {:>10.7}  {:>10.7}  {:>10.7}  {:>10.7}",
        "INT8", p(&i8_sorted, 0.0), p(&i8_sorted, 50.0), p(&i8_sorted, 95.0), p(&i8_sorted, 99.0), p(&i8_sorted, 100.0));
    eprintln!("  {:>6}  {:>10.7}  {:>10.7}  {:>10.7}  {:>10.7}  {:>10.7}",
        "Q4", p(&q4_sorted, 0.0), p(&q4_sorted, 50.0), p(&q4_sorted, 95.0), p(&q4_sorted, 99.0), p(&q4_sorted, 100.0));

    // === Scale Distribution ===
    q4_scales_all.sort_by(|a, b| a.partial_cmp(b).unwrap());
    i8_scales_all.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pf = |sorted: &[f32], pct: f64| -> f32 {
        let idx = ((sorted.len() as f64 * pct / 100.0) as usize).min(sorted.len() - 1);
        sorted[idx]
    };

    eprintln!("\n--- Quantization Scale Distribution (per group) ---");
    eprintln!("  Q4 step = scale (range [-8,7], 16 levels)");
    eprintln!("  INT8 step = scale/127 (range [-127,127], 255 levels)");
    eprintln!("  {:>6}  {:>10}  {:>10}  {:>10}  {:>10}", "", "p50", "p95", "p99", "max");
    eprintln!("  {:>6}  {:>10.6}  {:>10.6}  {:>10.6}  {:>10.6}",
        "Q4", pf(&q4_scales_all, 50.0), pf(&q4_scales_all, 95.0), pf(&q4_scales_all, 99.0), pf(&q4_scales_all, 100.0));
    eprintln!("  {:>6}  {:>10.6}  {:>10.6}  {:>10.6}  {:>10.6}",
        "INT8", pf(&i8_scales_all, 50.0), pf(&i8_scales_all, 95.0), pf(&i8_scales_all, 99.0), pf(&i8_scales_all, 100.0));

    // === Logit-Level Analysis ===
    eprintln!("\n--- Logit-Level Impact ({} random unit-norm hidden states) ---", N_TRIALS);

    let mut q4_top1_match = 0usize;
    let mut i8_top1_match = 0usize;
    let mut q4_top5_overlap_sum = 0.0f64;
    let mut i8_top5_overlap_sum = 0.0f64;
    let mut margin_bf16_all = Vec::with_capacity(N_TRIALS);
    let mut q4_logit_err_rms_all = Vec::with_capacity(N_TRIALS);
    let mut i8_logit_err_rms_all = Vec::with_capacity(N_TRIALS);
    let mut q4_logit_err_max_all = Vec::with_capacity(N_TRIALS);
    let mut i8_logit_err_max_all = Vec::with_capacity(N_TRIALS);

    for t in 0..N_TRIALS {
        let bf16_logits = &trial_logits_bf16[t];
        let q4_logits = &trial_logits_q4[t];
        let i8_logits = &trial_logits_i8[t];

        // Logit error statistics
        let mut q4_err_sq_sum = 0.0f64;
        let mut i8_err_sq_sum = 0.0f64;
        let mut q4_err_max = 0.0f32;
        let mut i8_err_max = 0.0f32;
        for i in 0..n {
            let q4_e = (q4_logits[i] - bf16_logits[i]).abs();
            let i8_e = (i8_logits[i] - bf16_logits[i]).abs();
            q4_err_sq_sum += (q4_e as f64) * (q4_e as f64);
            i8_err_sq_sum += (i8_e as f64) * (i8_e as f64);
            if q4_e > q4_err_max { q4_err_max = q4_e; }
            if i8_e > i8_err_max { i8_err_max = i8_e; }
        }
        q4_logit_err_rms_all.push((q4_err_sq_sum / n as f64).sqrt());
        i8_logit_err_rms_all.push((i8_err_sq_sum / n as f64).sqrt());
        q4_logit_err_max_all.push(q4_err_max);
        i8_logit_err_max_all.push(i8_err_max);

        // Top-K analysis: sort by BF16 logit descending
        let mut bf16_ranked: Vec<(usize, f32)> = bf16_logits.iter().copied().enumerate().collect();
        bf16_ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let bf16_top1 = bf16_ranked[0].0;
        let bf16_top5: std::collections::HashSet<usize> = bf16_ranked[..5].iter().map(|x| x.0).collect();

        // BF16 margin: top1 - top2
        let margin = bf16_ranked[0].1 - bf16_ranked[1].1;
        margin_bf16_all.push(margin);

        // Q4 top-1
        let q4_top1 = q4_logits.iter().copied().enumerate()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap().0;
        if q4_top1 == bf16_top1 { q4_top1_match += 1; }

        // INT8 top-1
        let i8_top1 = i8_logits.iter().copied().enumerate()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap().0;
        if i8_top1 == bf16_top1 { i8_top1_match += 1; }

        // Top-5 overlap
        let mut q4_ranked: Vec<(usize, f32)> = q4_logits.iter().copied().enumerate().collect();
        q4_ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let q4_top5: std::collections::HashSet<usize> = q4_ranked[..5].iter().map(|x| x.0).collect();
        q4_top5_overlap_sum += bf16_top5.intersection(&q4_top5).count() as f64 / 5.0;

        let mut i8_ranked: Vec<(usize, f32)> = i8_logits.iter().copied().enumerate().collect();
        i8_ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let i8_top5: std::collections::HashSet<usize> = i8_ranked[..5].iter().map(|x| x.0).collect();
        i8_top5_overlap_sum += bf16_top5.intersection(&i8_top5).count() as f64 / 5.0;
    }

    eprintln!("\n  Token Selection Accuracy:");
    eprintln!("    {:>6}  {:>18}  {:>18}", "", "Top-1 match", "Top-5 overlap");
    eprintln!("    {:>6}  {:>17.1}%  {:>17.1}%",
        "INT8", i8_top1_match as f64 / N_TRIALS as f64 * 100.0,
        i8_top5_overlap_sum / N_TRIALS as f64 * 100.0);
    eprintln!("    {:>6}  {:>17.1}%  {:>17.1}%",
        "Q4", q4_top1_match as f64 / N_TRIALS as f64 * 100.0,
        q4_top5_overlap_sum / N_TRIALS as f64 * 100.0);

    // Margin statistics
    margin_bf16_all.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let margin_median = margin_bf16_all[N_TRIALS / 2];
    let margin_min = margin_bf16_all[0];
    let margin_p10 = margin_bf16_all[N_TRIALS / 10];

    // Logit error statistics (aggregate)
    let q4_lerr_mean: f64 = q4_logit_err_rms_all.iter().sum::<f64>() / N_TRIALS as f64;
    let i8_lerr_mean: f64 = i8_logit_err_rms_all.iter().sum::<f64>() / N_TRIALS as f64;
    let q4_lerr_max = q4_logit_err_max_all.iter().cloned().fold(0.0f32, f32::max);
    let i8_lerr_max = i8_logit_err_max_all.iter().cloned().fold(0.0f32, f32::max);

    eprintln!("\n  Logit Error (per-token, across {} trials):", N_TRIALS);
    eprintln!("    {:>6}  {:>14}  {:>14}", "", "Mean RMS", "Worst max");
    eprintln!("    {:>6}  {:>14.6}  {:>14.6}", "INT8", i8_lerr_mean, i8_lerr_max);
    eprintln!("    {:>6}  {:>14.6}  {:>14.6}", "Q4", q4_lerr_mean, q4_lerr_max);

    eprintln!("\n  BF16 Top-1 vs Top-2 Margin (how much logit separation):");
    eprintln!("    min:    {:.6}", margin_min);
    eprintln!("    p10:    {:.6}", margin_p10);
    eprintln!("    median: {:.6}", margin_median);

    eprintln!("\n  Safety Ratio (margin / logit_error — higher = safer):");
    eprintln!("    {:>6}  {:>14}  {:>14}", "", "median/rms", "min/max");
    eprintln!("    {:>6}  {:>14.1}  {:>14.3}", "INT8",
        margin_median as f64 / i8_lerr_mean,
        margin_min as f64 / i8_lerr_max as f64);
    eprintln!("    {:>6}  {:>14.1}  {:>14.3}", "Q4",
        margin_median as f64 / q4_lerr_mean,
        margin_min as f64 / q4_lerr_max as f64);

    // === Data Size Comparison ===
    let bf16_size_mb = (n * k * 2) as f64 / 1e6;
    let i8_size_mb = (n * k + n * n_groups * 4) as f64 / 1e6;
    let q4_size_mb = (n * k / 2 + n * n_groups * 4) as f64 / 1e6;

    eprintln!("\n--- Data Size ---");
    eprintln!("  BF16:  {:.0} MB", bf16_size_mb);
    eprintln!("  INT8:  {:.0} MB ({:.1}× vs BF16)", i8_size_mb, bf16_size_mb / i8_size_mb);
    eprintln!("  Q4:    {:.0} MB ({:.1}× vs BF16, {:.1}× vs INT8)", q4_size_mb,
        bf16_size_mb / q4_size_mb, i8_size_mb / q4_size_mb);

    eprintln!("\n========================================================================");
    eprintln!("  Analysis complete. Exiting.");
    eprintln!("========================================================================\n");
    std::process::exit(0);
}

pub fn load_model(
    model_dir: &Path,
    opts: LoadOpts,
) -> Result<(Config, GenericModel<Q4Ops>)> {
    let config = Config::from_file(&model_dir.join("config.json"))?;

    let backend_id = get_backend_id();

    // For analysis mode, always skip cache (need raw BF16 weights)
    let analyze_lmhead = std::env::var("ANALYZE_LMHEAD").is_ok();

    if !opts.no_cache && !analyze_lmhead {
        if let Ok(model_hash) = weight_cache::compute_model_hash(model_dir) {
            if let Some(model) = weight_cache::load_model_cache::<Q4Ops>(
                &config,
                model_dir,
                backend_id,
                model_hash,
                opts.show_progress,
            ) {
                herbert_backend_common::numa::first_touch_weights(&model);
                return Ok((config, model));
            }
        }
    }

    let (config, model) = loader_common::load_model_streaming_progress::<Q4Ops>(
        model_dir,
        move |bf16_data, out_f, in_f, name| {
            // Run LM head quantization analysis if requested
            if name == "lm_head.weight" && analyze_lmhead {
                analyze_lmhead_quantization(bf16_data, out_f, in_f);
                // analyze_lmhead_quantization calls process::exit(0)
            }

            // Allow overriding LM head quantization via LMHEAD_QUANT env var
            if name == "lm_head.weight" {
                if let Ok(quant) = std::env::var("LMHEAD_QUANT") {
                    match quant.as_str() {
                        "bf16" => {
                            tracing::info!(tensor = name, n = out_f, k = in_f, "LM head: BF16 (override)");
                            return Ok(KernelWeight::BF16(bf16_to_f32_weight(bf16_data, out_f, in_f)));
                        }
                        "int8" => {
                            #[cfg(target_arch = "x86_64")]
                            {
                                tracing::info!(tensor = name, n = out_f, k = in_f, "LM head: INT8 (override)");
                                let i8w = quantize_bf16_to_int8(bf16_data, out_f, in_f);
                                return Ok(KernelWeight::INT8(i8w));
                            }
                        }
                        _ => {} // "q4" or anything else: fall through to default Q4
                    }
                }
            }

            if should_keep_bf16(name) {
                tracing::info!(tensor = name, n = out_f, k = in_f, "keeping as BF16 (full precision)");
                return Ok(KernelWeight::BF16(bf16_to_f32_weight(bf16_data, out_f, in_f)));
            }
            #[cfg(target_arch = "x86_64")]
            {
                if should_use_int8(name) {
                    tracing::info!(tensor = name, n = out_f, k = in_f, "quantizing to INT8 (per-group symmetric)");
                    let i8w = quantize_bf16_to_int8(bf16_data, out_f, in_f);
                    return Ok(KernelWeight::INT8(i8w));
                }
                let (data, scales) = quantize_and_pack_q4(bf16_data, out_f, in_f);
                let mut w = crate::weight::Q4Weight {
                    data: herbert_backend_common::hugepages::HugeVec::from_vec_no_huge(data),
                    scales: herbert_backend_common::hugepages::HugeVec::from_vec_no_huge(scales),
                    scales_tiled: herbert_backend_common::hugepages::HugeVec::from_vec_no_huge(vec![]),
                    n: out_f,
                    k: in_f,
                };
                w.compute_scales_tiled();
                w.move_to_hugepages();
                Ok(KernelWeight::Q4(w))
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                let _ = (bf16_data, out_f, in_f, name);
                Err(herbert_core::error::HerbertError::Backend(
                    "Q4 backend not supported on this architecture".to_string(),
                ))
            }
        },
        opts.show_progress,
    )?;

    if !opts.no_cache {
        if let Ok(model_hash) = weight_cache::compute_model_hash(model_dir) {
            match weight_cache::save_model_cache(&model, &config, model_dir, backend_id, model_hash)
            {
                Ok(Some(_cache_path)) => {
                    if opts.show_progress {
                        tracing::info!(backend = backend_id, "saved weight cache");
                    }
                }
                Ok(None) => {}
                Err(e) => {
                    tracing::warn!(error = %e, "failed to save weight cache");
                }
            }
        }
    }

    // NUMA first-touch: force page placement near pinned workers (Linux only, no-op elsewhere)
    herbert_backend_common::numa::first_touch_weights(&model);

    Ok((config, model))
}
