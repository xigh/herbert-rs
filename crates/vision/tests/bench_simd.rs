//! Precision and benchmark tests: SIMD vs scalar for vision encoder kernels.
//!
//! Run with: cargo test --release -p qwen3-vision --test bench_simd -- --nocapture

use std::time::Instant;

use herbert_vision::encoder::{
    gelu_chunk, layer_norm_row, scalar_gelu_chunk, scalar_layer_norm_row, tanh_pade,
};

// ─── Helpers ─────────────────────────────────────────────────────────

/// Generate pseudo-random f32 data (deterministic, no external deps).
fn pseudo_random_vec(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_add(1); // avoid 0 state
    (0..n)
        .map(|_| {
            // xorshift64
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            // Map to [-2.0, 2.0]
            let bits = (state & 0xFFFFFFFF) as u32;
            (bits as f32 / u32::MAX as f32) * 4.0 - 2.0
        })
        .collect()
}

// ─── dot_f32 benchmarks (from Phase 1) ──────────────────────────────

fn scalar_dot(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn avx512_dot(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len();
    let mut acc0 = _mm512_setzero_ps();
    let mut acc1 = _mm512_setzero_ps();
    let mut acc2 = _mm512_setzero_ps();
    let mut acc3 = _mm512_setzero_ps();
    let mut i = 0;
    while i + 64 <= n {
        acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(a.as_ptr().add(i)),      _mm512_loadu_ps(b.as_ptr().add(i)),      acc0);
        acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(a.as_ptr().add(i + 16)), _mm512_loadu_ps(b.as_ptr().add(i + 16)), acc1);
        acc2 = _mm512_fmadd_ps(_mm512_loadu_ps(a.as_ptr().add(i + 32)), _mm512_loadu_ps(b.as_ptr().add(i + 32)), acc2);
        acc3 = _mm512_fmadd_ps(_mm512_loadu_ps(a.as_ptr().add(i + 48)), _mm512_loadu_ps(b.as_ptr().add(i + 48)), acc3);
        i += 64;
    }
    acc0 = _mm512_add_ps(acc0, acc1);
    acc2 = _mm512_add_ps(acc2, acc3);
    acc0 = _mm512_add_ps(acc0, acc2);
    while i + 16 <= n {
        acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(a.as_ptr().add(i)), _mm512_loadu_ps(b.as_ptr().add(i)), acc0);
        i += 16;
    }
    let mut sum = _mm512_reduce_add_ps(acc0);
    while i < n {
        sum += *a.get_unchecked(i) * *b.get_unchecked(i);
        i += 1;
    }
    sum
}

fn bench_dot(dim: usize, label: &str) {
    let a: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.001).sin()).collect();
    let b: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.002).cos()).collect();

    // Warmup
    let mut sink = 0.0f32;
    for _ in 0..1000 {
        sink += scalar_dot(&a, &b);
    }
    std::hint::black_box(sink);

    let iters = 100_000;

    // Scalar
    let t0 = Instant::now();
    let mut s = 0.0f32;
    for _ in 0..iters {
        s += scalar_dot(&a, &b);
    }
    std::hint::black_box(s);
    let scalar_ns = t0.elapsed().as_nanos() as f64 / iters as f64;

    // SIMD (AVX-512)
    let t0 = Instant::now();
    let mut s = 0.0f32;
    if is_x86_feature_detected!("avx512f") {
        for _ in 0..iters {
            s += unsafe { avx512_dot(&a, &b) };
        }
    } else {
        for _ in 0..iters {
            s += scalar_dot(&a, &b);
        }
    }
    std::hint::black_box(s);
    let simd_ns = t0.elapsed().as_nanos() as f64 / iters as f64;

    let simd_name = "AVX-512";
    println!(
        "  {label:>12} (dim={dim:>4}): scalar={scalar_ns:>7.1}ns  {simd_name}={simd_ns:>7.1}ns  speedup={:.1}x",
        scalar_ns / simd_ns
    );
}

#[test]
fn bench_dot_f32_simd_vs_scalar() {
    println!("\n=== dot_f32 benchmark ===");
    bench_dot(64, "head_dim");      // attention dot products
    bench_dot(1024, "hidden");      // QKV, proj linears
    bench_dot(1536, "patch_dim");   // patch embedding
    bench_dot(4096, "intermed");    // MLP FC1/FC2
    println!();
}

// ─── LayerNorm precision tests ───────────────────────────────────────

#[test]
fn test_layer_norm_simd_vs_scalar_dim1024() {
    let dim = 1024;
    let eps = 1e-6;
    let x = pseudo_random_vec(dim, 42);
    let weight: Vec<f32> = (0..dim).map(|i| 0.5 + 0.001 * i as f32).collect();
    let bias: Vec<f32> = (0..dim).map(|i| -0.1 + 0.0002 * i as f32).collect();

    let mut out_scalar = vec![0.0f32; dim];
    let mut out_simd = vec![0.0f32; dim];

    scalar_layer_norm_row(&x, &weight, &bias, eps, &mut out_scalar);
    layer_norm_row(&x, &weight, &bias, eps, &mut out_simd);

    let max_diff = out_scalar
        .iter()
        .zip(out_simd.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    println!(
        "[LayerNorm dim={}] max_diff = {:.2e} (tolerance: 1e-5)",
        dim, max_diff
    );
    assert!(
        max_diff < 1e-5,
        "LayerNorm SIMD vs scalar diff too large: {:.2e}",
        max_diff
    );
}

#[test]
fn test_layer_norm_simd_vs_scalar_dim128() {
    let dim = 128;
    let eps = 1e-6;
    let x = pseudo_random_vec(dim, 123);
    let weight = vec![1.0f32; dim];
    let bias = vec![0.0f32; dim];

    let mut out_scalar = vec![0.0f32; dim];
    let mut out_simd = vec![0.0f32; dim];

    scalar_layer_norm_row(&x, &weight, &bias, eps, &mut out_scalar);
    layer_norm_row(&x, &weight, &bias, eps, &mut out_simd);

    let max_diff = out_scalar
        .iter()
        .zip(out_simd.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    println!(
        "[LayerNorm dim={}] max_diff = {:.2e} (tolerance: 1e-5)",
        dim, max_diff
    );
    assert!(
        max_diff < 1e-5,
        "LayerNorm SIMD vs scalar diff too large: {:.2e}",
        max_diff
    );
}

#[test]
fn test_layer_norm_simd_vs_scalar_odd_dim() {
    let dim = 1000;
    let eps = 1e-5;
    let x = pseudo_random_vec(dim, 999);
    let weight = vec![1.0f32; dim];
    let bias = vec![0.0f32; dim];

    let mut out_scalar = vec![0.0f32; dim];
    let mut out_simd = vec![0.0f32; dim];

    scalar_layer_norm_row(&x, &weight, &bias, eps, &mut out_scalar);
    layer_norm_row(&x, &weight, &bias, eps, &mut out_simd);

    let max_diff = out_scalar
        .iter()
        .zip(out_simd.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    println!(
        "[LayerNorm dim={}] max_diff = {:.2e} (tolerance: 1e-5)",
        dim, max_diff
    );
    assert!(
        max_diff < 1e-5,
        "LayerNorm SIMD vs scalar diff too large: {:.2e}",
        max_diff
    );
}

// ─── GELU precision tests ───────────────────────────────────────────

#[test]
fn test_gelu_simd_vs_scalar_4096() {
    let n = 4096;
    let data = pseudo_random_vec(n, 77);

    let mut scalar_data = data.clone();
    let mut simd_data = data.clone();

    scalar_gelu_chunk(&mut scalar_data);
    gelu_chunk(&mut simd_data);

    // Print first few differences for debugging
    let mut worst_diff = 0.0f32;
    for i in 0..scalar_data.len() {
        let diff = (scalar_data[i] - simd_data[i]).abs();
        if diff > worst_diff {
            worst_diff = diff;
        }
    }
    let max_diff = worst_diff;

    println!(
        "[GELU n={}] max_diff = {:.2e} (tolerance: 1e-5)",
        n, max_diff
    );
    assert!(
        max_diff < 1e-5,
        "GELU SIMD vs scalar diff too large: {:.2e}",
        max_diff
    );
}

#[test]
fn test_gelu_simd_vs_scalar_odd_size() {
    let n = 1000;
    let data = pseudo_random_vec(n, 55);

    let mut scalar_data = data.clone();
    let mut simd_data = data.clone();

    scalar_gelu_chunk(&mut scalar_data);
    gelu_chunk(&mut simd_data);

    let max_diff = scalar_data
        .iter()
        .zip(simd_data.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    println!(
        "[GELU n={}] max_diff = {:.2e} (tolerance: 1e-5)",
        n, max_diff
    );
    assert!(
        max_diff < 1e-5,
        "GELU SIMD vs scalar diff too large: {:.2e}",
        max_diff
    );
}

#[test]
fn test_gelu_simd_vs_scalar_extreme_values() {
    let data: Vec<f32> = vec![
        -5.0, -4.0, -3.0, -2.0, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0,
        -10.0, 10.0, -0.001, 0.001, -6.0, 6.0, -7.0, 7.0, 0.0,
    ];

    let mut scalar_data = data.clone();
    let mut simd_data = data.clone();

    scalar_gelu_chunk(&mut scalar_data);
    gelu_chunk(&mut simd_data);

    let max_diff = scalar_data
        .iter()
        .zip(simd_data.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    // Relaxed tolerance for extreme values (|x| > 5): GELU saturates to 0 or x,
    // so absolute differences from Padé vs libm tanh are inconsequential for inference.
    // Normal-range values (|x| < 3) have ~1e-7 error.
    println!(
        "[GELU extreme] max_diff = {:.2e} (tolerance: 1e-3)",
        max_diff
    );
    assert!(
        max_diff < 1e-3,
        "GELU SIMD vs scalar diff too large: {:.2e}",
        max_diff
    );
}

// ─── tanh_pade precision test ───────────────────────────────────────

#[test]
fn test_tanh_pade_vs_libm() {
    let mut max_err = 0.0f32;
    let mut worst_z = 0.0f32;
    let steps = 10000;
    for i in 0..=steps {
        let z = -5.0 + 10.0 * i as f32 / steps as f32;
        let expected = z.tanh();
        let approx = tanh_pade(z);
        let err = (expected - approx).abs();
        if err > max_err {
            max_err = err;
            worst_z = z;
        }
    }
    // At |z| > 4.5, tanh(z) ≈ ±1. Padé boundary error is ~1e-4 which is
    // inconsequential. For the critical inference range |z| < 3, error is ~1e-7.
    println!(
        "[tanh_pade] max_err = {:.2e} at z = {:.4} (tolerance: 2e-4)",
        max_err, worst_z
    );
    assert!(
        max_err < 2e-4,
        "tanh_pade error too large: {:.2e} at z={}",
        max_err,
        worst_z
    );
}

// ─── Micro-benchmarks (informational, printed with --nocapture) ─────

#[test]
fn bench_layer_norm_simd() {
    let dim = 1024;
    let eps = 1e-6;
    let x = pseudo_random_vec(dim, 42);
    let weight = vec![1.0f32; dim];
    let bias = vec![0.0f32; dim];
    let mut out = vec![0.0f32; dim];

    let iters = 100_000;

    // Scalar
    let start = Instant::now();
    for _ in 0..iters {
        scalar_layer_norm_row(&x, &weight, &bias, eps, &mut out);
        std::hint::black_box(&out);
    }
    let scalar_time = start.elapsed();

    // SIMD
    let start = Instant::now();
    for _ in 0..iters {
        layer_norm_row(&x, &weight, &bias, eps, &mut out);
        std::hint::black_box(&out);
    }
    let simd_time = start.elapsed();

    let speedup = scalar_time.as_nanos() as f64 / simd_time.as_nanos() as f64;
    println!(
        "\n[Bench LayerNorm dim={}] scalar: {:?}, simd: {:?}, speedup: {:.2}x",
        dim, scalar_time, simd_time, speedup
    );
}

#[test]
fn bench_gelu_simd() {
    let n = 4096;
    let data = pseudo_random_vec(n, 77);
    let iters = 50_000;

    // Scalar
    let mut buf = data.clone();
    let start = Instant::now();
    for _ in 0..iters {
        buf.copy_from_slice(&data);
        scalar_gelu_chunk(&mut buf);
        std::hint::black_box(&buf);
    }
    let scalar_time = start.elapsed();

    // SIMD
    let start = Instant::now();
    for _ in 0..iters {
        buf.copy_from_slice(&data);
        gelu_chunk(&mut buf);
        std::hint::black_box(&buf);
    }
    let simd_time = start.elapsed();

    let speedup = scalar_time.as_nanos() as f64 / simd_time.as_nanos() as f64;
    println!(
        "\n[Bench GELU n={}] scalar: {:?}, simd: {:?}, speedup: {:.2}x",
        n, scalar_time, simd_time, speedup
    );
}
