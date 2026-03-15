//! Generic decoder layer module parameterized over LinearOps.

use herbert_core::config::NormType;
use herbert_core::error::Result;
use herbert_core::tensor::{bf16_to_f32, BF16};

use crate::generic_attention::GenericAttention;
use crate::generic_moe::GenericFFN;
use crate::kernels as common_kernels;
use crate::kv_cache::CpuKvCache;
use crate::linear_ops::LinearOps;
use crate::thread_pool::{global_pool, SendMutPtr, SendPtr};

// ============================================================================
// AVX-512 SIMD recurrence for DeltaNet (x86_64 only)
// ============================================================================

#[cfg(target_arch = "x86_64")]
use crate::kernels::has_avx512f;


/// AVX-512 SIMD recurrence for one V-head: decay+read, delta, write+query, gated RMS norm.
///
/// Requires: hvd == 128 (8 full ZMM vectors kept in registers across kd loop).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn recurrent_head_avx512(
    s: &mut [f32],       // [hkd * hvd] recurrent state
    out: &mut [f32],     // [hvd] output
    q: &[f32],           // [hkd] query (L2-normalized)
    k: &[f32],           // [hkd] key (L2-normalized)
    v: &[f32],           // [hvd] value
    z_h: &[f32],         // [hvd] gate input
    _kv_mem: &mut [f32], // [hvd] scratch (unused — kept in registers)
    delta: &mut [f32],   // [hvd] scratch
    b_h: f32,            // beta (update gate)
    decay: f32,          // exp(g) (decay factor)
    scale: f32,          // 1/sqrt(hkd)
    hkd: usize,
    hvd: usize,
    norm_w: *const u16,  // BF16 norm weights [hvd]
    eps: f32,
) {
    use std::arch::x86_64::*;
    debug_assert_eq!(hvd, 128, "AVX-512 recurrence requires hvd=128");

    let decay_v = _mm512_set1_ps(decay);

    // ── Pass 1: Decay + Read ───────────────────────────────────
    // Keep kv_mem accumulator in 8 ZMM registers (128 f32 = 8×16).
    // Eliminates 128 kd × 8 loads + 8 stores = 2048 memory ops.
    let mut kv0 = _mm512_setzero_ps();
    let mut kv1 = _mm512_setzero_ps();
    let mut kv2 = _mm512_setzero_ps();
    let mut kv3 = _mm512_setzero_ps();
    let mut kv4 = _mm512_setzero_ps();
    let mut kv5 = _mm512_setzero_ps();
    let mut kv6 = _mm512_setzero_ps();
    let mut kv7 = _mm512_setzero_ps();

    for kd in 0..hkd {
        let k_val = _mm512_set1_ps(*k.get_unchecked(kd));
        let rp = s.as_mut_ptr().add(kd * 128);

        macro_rules! decay_read {
            ($acc:ident, $off:expr) => {{
                let mut rv = _mm512_loadu_ps(rp.add($off));
                rv = _mm512_mul_ps(rv, decay_v);
                _mm512_storeu_ps(rp.add($off), rv);
                $acc = _mm512_fmadd_ps(rv, k_val, $acc);
            }};
        }
        decay_read!(kv0, 0);
        decay_read!(kv1, 16);
        decay_read!(kv2, 32);
        decay_read!(kv3, 48);
        decay_read!(kv4, 64);
        decay_read!(kv5, 80);
        decay_read!(kv6, 96);
        decay_read!(kv7, 112);
    }

    // ── Delta: delta = (v - kv_mem) * beta ─────────────────────
    let beta_v = _mm512_set1_ps(b_h);
    macro_rules! compute_delta {
        ($kv:expr, $off:expr) => {{
            let vv = _mm512_loadu_ps(v.as_ptr().add($off));
            _mm512_storeu_ps(delta.as_mut_ptr().add($off),
                _mm512_mul_ps(_mm512_sub_ps(vv, $kv), beta_v));
        }};
    }
    compute_delta!(kv0, 0);
    compute_delta!(kv1, 16);
    compute_delta!(kv2, 32);
    compute_delta!(kv3, 48);
    compute_delta!(kv4, 64);
    compute_delta!(kv5, 80);
    compute_delta!(kv6, 96);
    compute_delta!(kv7, 112);

    // ── Pass 2: Write + Query ──────────────────────────────────
    // Keep out accumulator in 8 ZMM registers.
    let mut o0 = _mm512_setzero_ps();
    let mut o1 = _mm512_setzero_ps();
    let mut o2 = _mm512_setzero_ps();
    let mut o3 = _mm512_setzero_ps();
    let mut o4 = _mm512_setzero_ps();
    let mut o5 = _mm512_setzero_ps();
    let mut o6 = _mm512_setzero_ps();
    let mut o7 = _mm512_setzero_ps();

    // Pre-load delta into registers (stays constant across kd loop)
    let d0 = _mm512_loadu_ps(delta.as_ptr().add(0));
    let d1 = _mm512_loadu_ps(delta.as_ptr().add(16));
    let d2 = _mm512_loadu_ps(delta.as_ptr().add(32));
    let d3 = _mm512_loadu_ps(delta.as_ptr().add(48));
    let d4 = _mm512_loadu_ps(delta.as_ptr().add(64));
    let d5 = _mm512_loadu_ps(delta.as_ptr().add(80));
    let d6 = _mm512_loadu_ps(delta.as_ptr().add(96));
    let d7 = _mm512_loadu_ps(delta.as_ptr().add(112));

    for kd in 0..hkd {
        let k_val = _mm512_set1_ps(*k.get_unchecked(kd));
        let q_val = _mm512_set1_ps(*q.get_unchecked(kd) * scale);
        let rp = s.as_mut_ptr().add(kd * 128);

        macro_rules! write_query {
            ($out:ident, $delt:expr, $off:expr) => {{
                let mut rv = _mm512_loadu_ps(rp.add($off));
                rv = _mm512_fmadd_ps(k_val, $delt, rv);
                _mm512_storeu_ps(rp.add($off), rv);
                $out = _mm512_fmadd_ps(rv, q_val, $out);
            }};
        }
        write_query!(o0, d0, 0);
        write_query!(o1, d1, 16);
        write_query!(o2, d2, 32);
        write_query!(o3, d3, 48);
        write_query!(o4, d4, 64);
        write_query!(o5, d5, 80);
        write_query!(o6, d6, 96);
        write_query!(o7, d7, 112);
    }

    // Store out from registers
    _mm512_storeu_ps(out.as_mut_ptr().add(0), o0);
    _mm512_storeu_ps(out.as_mut_ptr().add(16), o1);
    _mm512_storeu_ps(out.as_mut_ptr().add(32), o2);
    _mm512_storeu_ps(out.as_mut_ptr().add(48), o3);
    _mm512_storeu_ps(out.as_mut_ptr().add(64), o4);
    _mm512_storeu_ps(out.as_mut_ptr().add(80), o5);
    _mm512_storeu_ps(out.as_mut_ptr().add(96), o6);
    _mm512_storeu_ps(out.as_mut_ptr().add(112), o7);

    // ── Gated RMS Norm ─────────────────────────────────────────
    let sq_acc = _mm512_fmadd_ps(o0, o0, _mm512_setzero_ps());
    let sq_acc = _mm512_fmadd_ps(o1, o1, sq_acc);
    let sq_acc = _mm512_fmadd_ps(o2, o2, sq_acc);
    let sq_acc = _mm512_fmadd_ps(o3, o3, sq_acc);
    let sq_acc = _mm512_fmadd_ps(o4, o4, sq_acc);
    let sq_acc = _mm512_fmadd_ps(o5, o5, sq_acc);
    let sq_acc = _mm512_fmadd_ps(o6, o6, sq_acc);
    let sq_acc = _mm512_fmadd_ps(o7, o7, sq_acc);
    let sq_sum = _mm512_reduce_add_ps(sq_acc);
    let inv_rms = 1.0 / (sq_sum / hvd as f32 + eps).sqrt();

    // SiLU(z) = z / (1 + exp(-z)). Scalar for SiLU + BF16 weight.
    for d in 0..hvd {
        let w = bf16_to_f32(*norm_w.add(d));
        let z_val = *z_h.get_unchecked(d);
        let silu_z = z_val / (1.0 + (-z_val).exp());
        *out.get_unchecked_mut(d) *= inv_rms * w * silu_z;
    }
}

/// Depthwise causal 1D convolution block (LFM2 hybrid layers).
pub struct GenericConvolution<L: LinearOps> {
    pub in_proj: L::Weight,
    /// Depthwise causal kernel, row-major [hidden_size, conv_l_cache].
    pub conv_kernel: Vec<f32>,
    pub out_proj: L::Weight,
    pub hidden_size: usize,
    pub conv_l_cache: usize,
}

impl<L: LinearOps> GenericConvolution<L> {
    pub fn forward_decode(
        &self,
        x: &[f32],
        kv_cache: &mut CpuKvCache,
        layer_idx: usize,
        output: &mut [f32],
    ) -> Result<()> {
        let mut in_proj_out = vec![0.0f32; self.hidden_size * 3];
        L::matvec_float_dequant(x, &self.in_proj, &mut in_proj_out)?;

        let (b, c_x) = in_proj_out.split_at(self.hidden_size);
        let (c, inner_x) = c_x.split_at(self.hidden_size);

        let mut gated = vec![0.0f32; self.hidden_size];
        for i in 0..self.hidden_size {
            gated[i] = b[i] * inner_x[i];
        }

        let mut conv_out = vec![0.0f32; self.hidden_size];
        let l_cache = self.conv_l_cache;
        let state = &mut kv_cache.kv.conv_cache[layer_idx];

        for h in 0..self.hidden_size {
            let base = h * l_cache;
            // Roll left by one and append latest gated value at the end.
            for j in 0..(l_cache - 1) {
                state[base + j] = state[base + j + 1];
            }
            state[base + l_cache - 1] = gated[h];

            let mut sum = 0.0f32;
            for j in 0..l_cache {
                sum += state[base + j] * self.conv_kernel[base + j];
            }
            conv_out[h] = sum;
        }

        let mut y = vec![0.0f32; self.hidden_size];
        for i in 0..self.hidden_size {
            y[i] = c[i] * conv_out[i];
        }

        L::matvec_float_dequant(&y, &self.out_proj, output)?;

        Ok(())
    }

    pub fn forward_prefill(
        &self,
        x: &[f32],
        seq_len: usize,
        kv_cache: &mut CpuKvCache,
        layer_idx: usize,
        output: &mut [f32],
    ) -> Result<()> {
        let l_cache = self.conv_l_cache;

        #[cfg(feature = "profile-prefill-layer")]
        let t_ppl = std::time::Instant::now();
        let mut in_proj_out = vec![0.0f32; seq_len * self.hidden_size * 3];
        L::matmul(x, &self.in_proj, &mut in_proj_out, seq_len)?;
        #[cfg(feature = "profile-prefill-layer")]
        let in_proj_us = t_ppl.elapsed().as_micros() as u64;

        #[cfg(feature = "profile-prefill-layer")]
        let t_ppl = std::time::Instant::now();
        let mut bx = vec![0.0f32; seq_len * self.hidden_size];
        let mut c_buf = vec![0.0f32; seq_len * self.hidden_size];
        for t in 0..seq_len {
            let row = t * self.hidden_size * 3;
            let b_off = row;
            let c_off = row + self.hidden_size;
            let x_off = row + 2 * self.hidden_size;
            for h in 0..self.hidden_size {
                let b = in_proj_out[b_off + h];
                let c = in_proj_out[c_off + h];
                let xv = in_proj_out[x_off + h];
                bx[t * self.hidden_size + h] = b * xv;
                c_buf[t * self.hidden_size + h] = c;
            }
        }
        #[cfg(feature = "profile-prefill-layer")]
        let gating_us = t_ppl.elapsed().as_micros() as u64;

        #[cfg(feature = "profile-prefill-layer")]
        let t_ppl = std::time::Instant::now();
        let mut y_before_out = vec![0.0f32; seq_len * self.hidden_size];
        for t in 0..seq_len {
            for h in 0..self.hidden_size {
                let base = h * l_cache;
                let mut sum = 0.0f32;
                let t_isize = t as isize;
                let l_isize = l_cache as isize;
                for j in 0..l_cache {
                    let src_t = t_isize - (l_isize - 1) + j as isize;
                    if src_t >= 0 {
                        let src_t = src_t as usize;
                        if src_t < seq_len {
                            sum += bx[src_t * self.hidden_size + h] * self.conv_kernel[base + j];
                        }
                    }
                }
                let idx = t * self.hidden_size + h;
                y_before_out[idx] = c_buf[idx] * sum;
            }
        }

        // Update decode cache with the last L_cache gated activations.
        let state = &mut kv_cache.kv.conv_cache[layer_idx];
        state.fill(0.0);
        for h in 0..self.hidden_size {
            let base = h * l_cache;
            for j in 0..l_cache {
                let src_t = seq_len as isize - l_cache as isize + j as isize;
                state[base + j] = if src_t >= 0 {
                    bx[src_t as usize * self.hidden_size + h]
                } else {
                    0.0
                };
            }
        }
        #[cfg(feature = "profile-prefill-layer")]
        let conv_kernel_us = t_ppl.elapsed().as_micros() as u64;

        #[cfg(feature = "profile-prefill-layer")]
        let t_ppl = std::time::Instant::now();
        L::matmul(&y_before_out, &self.out_proj, output, seq_len)?;
        #[cfg(feature = "profile-prefill-layer")]
        let out_proj_us = t_ppl.elapsed().as_micros() as u64;

        #[cfg(feature = "profile-prefill-layer")]
        crate::profiler::set_conv_sub_timings(crate::profiler::ConvSubTimings {
            in_proj_us,
            gating_us,
            conv_kernel_us,
            out_proj_us,
        });

        Ok(())
    }
}

/// Gated DeltaNet linear attention block (Qwen3.5).
/// Implements recurrent mode for decode (v1: also used for prefill).
pub struct GenericGatedDeltaNet<L: LinearOps> {
    /// Fused Q+K+V projection [conv_dim, hidden_size].
    pub in_proj_qkv: L::Weight,
    /// Output gate projection [value_dim, hidden_size].
    pub in_proj_z: L::Weight,
    /// Decay gate projection [num_v_heads, hidden_size] — f32.
    pub in_proj_a: Vec<f32>,
    /// Update gate projection [num_v_heads, hidden_size] — f32.
    pub in_proj_b: Vec<f32>,
    /// Depthwise causal conv1d kernel [conv_dim, kernel_size] — f32.
    pub conv1d_weight: Vec<f32>,
    /// Log-space decay parameter [num_v_heads] — f32.
    pub a_log: Vec<f32>,
    /// Timestep bias [num_v_heads] — f32.
    pub dt_bias: Vec<f32>,
    /// RMSNormGated weight [head_v_dim] — BF16.
    pub norm_weight: Vec<BF16>,
    /// Output projection [hidden_size, value_dim].
    pub out_proj: L::Weight,
    // Config
    pub num_k_heads: usize,
    pub num_v_heads: usize,
    pub head_k_dim: usize,
    pub head_v_dim: usize,
    pub conv_kernel_size: usize,
    pub hidden_size: usize,
    pub rms_norm_eps: f32,
}

impl<L: LinearOps> GenericGatedDeltaNet<L> {
    #[inline]
    fn key_dim(&self) -> usize { self.num_k_heads * self.head_k_dim }

    #[inline]
    fn value_dim(&self) -> usize { self.num_v_heads * self.head_v_dim }

    #[inline]
    fn conv_dim(&self) -> usize { self.key_dim() * 2 + self.value_dim() }

    #[inline]
    fn heads_per_group(&self) -> usize { self.num_v_heads / self.num_k_heads }

    /// F32 matvec for tiny projections (in_proj_a, in_proj_b).
    fn matvec_f32(x: &[f32], weight: &[f32], out_dim: usize, output: &mut [f32]) {
        let in_dim = x.len();
        for o in 0..out_dim {
            let mut sum = 0.0f32;
            let row = o * in_dim;
            for i in 0..in_dim {
                sum += weight[row + i] * x[i];
            }
            output[o] = sum;
        }
    }

    #[inline]
    fn l2_normalize(v: &mut [f32]) {
        let sq_sum: f32 = v.iter().map(|&x| x * x).sum();
        let inv_norm = 1.0 / (sq_sum + 1e-6).sqrt();
        for x in v.iter_mut() {
            *x *= inv_norm;
        }
    }

    #[inline]
    fn silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    #[inline]
    fn softplus(x: f32) -> f32 {
        if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
    }

    /// Steps 2-9 of DeltaNet: conv1d, split, normalize, gates, recurrence, gated norm.
    /// Takes pre-computed projections as input.
    fn recurrent_step(
        &self,
        qkv: &mut [f32],      // [conv_dim], modified in-place by conv1d+SiLU
        z: &[f32],            // [value_dim]
        a: &[f32],            // [num_v_heads]
        b: &[f32],            // [num_v_heads]
        kv_cache: &mut CpuKvCache,
        layer_idx: usize,
        attn_out: &mut [f32], // [value_dim], output (zeroed by caller)
    ) {
        let key_dim = self.key_dim();
        let conv_dim = self.conv_dim();
        let hpg = self.heads_per_group();

        // 2. Causal conv1d update + SiLU
        let conv_state = &mut kv_cache.kv.deltanet_conv[layer_idx];
        let ks = self.conv_kernel_size;
        for ch in 0..conv_dim {
            let base = ch * ks;
            for j in 0..(ks - 1) {
                conv_state[base + j] = conv_state[base + j + 1];
            }
            conv_state[base + ks - 1] = qkv[ch];
            let mut sum = 0.0f32;
            for j in 0..ks {
                sum += conv_state[base + j] * self.conv1d_weight[base + j];
            }
            qkv[ch] = Self::silu(sum);
        }

        // 3. Split into q, k, v
        let q_data = &qkv[..key_dim];
        let k_data = &qkv[key_dim..key_dim * 2];
        let v_data = &qkv[key_dim * 2..];

        // 4. L2 normalize q, k (per K head)
        let mut q_norm = q_data.to_vec();
        let mut k_norm = k_data.to_vec();
        for h in 0..self.num_k_heads {
            let off = h * self.head_k_dim;
            Self::l2_normalize(&mut q_norm[off..off + self.head_k_dim]);
            Self::l2_normalize(&mut k_norm[off..off + self.head_k_dim]);
        }

        // 5. Compute gates
        let mut beta = vec![0.0f32; self.num_v_heads];
        let mut g = vec![0.0f32; self.num_v_heads];
        for h in 0..self.num_v_heads {
            beta[h] = 1.0 / (1.0 + (-b[h]).exp()); // sigmoid
            g[h] = -self.a_log[h].exp() * Self::softplus(a[h] + self.dt_bias[h]);
        }

        // 8+9. Fused recurrence + gated RMS norm (parallel per V head)
        let hkd = self.head_k_dim;
        let hvd = self.head_v_dim;
        let s_size = hkd * hvd;
        let num_v_heads = self.num_v_heads;
        let eps = self.rms_norm_eps;

        let recurrent_state = &mut kv_cache.kv.deltanet_recurrent[layer_idx];

        let pool = global_pool();
        let workers = pool.num_workers().min(num_v_heads).max(1);

        let state_ptr = SendMutPtr::new(recurrent_state.as_mut_ptr());
        let attn_ptr = SendMutPtr::new(attn_out.as_mut_ptr());
        let q_ptr = SendPtr::new(q_norm.as_ptr());
        let k_ptr = SendPtr::new(k_norm.as_ptr());
        let v_ptr = SendPtr::new(v_data.as_ptr());
        let z_ptr = SendPtr::new(z.as_ptr());
        let beta_ptr = SendPtr::new(beta.as_ptr());
        let g_ptr = SendPtr::new(g.as_ptr());
        let norm_ptr = SendPtr::new(self.norm_weight.as_ptr());

        #[cfg(target_arch = "x86_64")]
        let use_avx512 = has_avx512f() && hvd % 16 == 0;

        pool.parallel_for_with_max_workers(
            num_v_heads,
            workers,
            move |_worker_id, vh_start, vh_end| {
                let mut kv_mem = vec![0.0f32; hvd];
                let mut delta = vec![0.0f32; hvd];

                for vh in vh_start..vh_end {
                    let kh = vh / hpg;
                    unsafe {
                        let s = std::slice::from_raw_parts_mut(
                            state_ptr.ptr().add(vh * s_size),
                            s_size,
                        );
                        let out = std::slice::from_raw_parts_mut(
                            attn_ptr.ptr().add(vh * hvd),
                            hvd,
                        );
                        let q = std::slice::from_raw_parts(q_ptr.ptr().add(kh * hkd), hkd);
                        let k = std::slice::from_raw_parts(k_ptr.ptr().add(kh * hkd), hkd);
                        let v = std::slice::from_raw_parts(v_ptr.ptr().add(vh * hvd), hvd);
                        let z_h = std::slice::from_raw_parts(z_ptr.ptr().add(vh * hvd), hvd);
                        let b_h = *beta_ptr.ptr().add(vh);
                        let g_h = *g_ptr.ptr().add(vh);
                        let decay = g_h.exp();
                        let scale = 1.0 / (hkd as f32).sqrt();

                        #[cfg(target_arch = "x86_64")]
                        if use_avx512 {
                            recurrent_head_avx512(
                                s, out, q, k, v, z_h,
                                &mut kv_mem, &mut delta,
                                b_h, decay, scale, hkd, hvd,
                                norm_ptr.ptr(), eps,
                            );
                            continue;
                        }

                        // Scalar fallback
                        // Fused Pass 1: Decay + Read
                        kv_mem.fill(0.0);
                        for kd in 0..hkd {
                            let k_val = k[kd];
                            let row = &mut s[kd * hvd..(kd + 1) * hvd];
                            for vd in 0..hvd {
                                row[vd] *= decay;
                                kv_mem[vd] += row[vd] * k_val;
                            }
                        }

                        // Delta (no S access)
                        for vd in 0..hvd {
                            delta[vd] = (v[vd] - kv_mem[vd]) * b_h;
                        }

                        // Fused Pass 2: Write + Query
                        for kd in 0..hkd {
                            let k_val = k[kd];
                            let q_val = q[kd] * scale;
                            let row = &mut s[kd * hvd..(kd + 1) * hvd];
                            for vd in 0..hvd {
                                row[vd] += k_val * delta[vd];
                                out[vd] += row[vd] * q_val;
                            }
                        }

                        // 9. Gated RMS Norm (inline per V head)
                        let sq_sum: f32 = out.iter().map(|&x| x * x).sum();
                        let inv_rms = 1.0 / (sq_sum / hvd as f32 + eps).sqrt();
                        for d in 0..hvd {
                            let w = bf16_to_f32(*norm_ptr.ptr().add(d));
                            let z_val = z_h[d];
                            let silu_z = z_val / (1.0 + (-z_val).exp());
                            out[d] = out[d] * inv_rms * w * silu_z;
                        }
                    }
                }
            },
        )
        .expect("deltanet parallel_for");
    }

    pub fn forward_decode(
        &self,
        x: &[f32],
        kv_cache: &mut CpuKvCache,
        layer_idx: usize,
        output: &mut [f32],
    ) -> Result<()> {
        let conv_dim = self.conv_dim();
        let value_dim = self.value_dim();

        // 1. Projections (qkv + z fused: quantize x once)
        #[cfg(feature = "profile-decode-layer")]
        let t0 = std::time::Instant::now();
        let mut qkv = vec![0.0f32; conv_dim];
        let mut z = vec![0.0f32; value_dim];
        L::matvec_float_dequant(x, &self.in_proj_qkv, &mut qkv)?;
        L::matvec_float_dequant(x, &self.in_proj_z, &mut z)?;
        #[cfg(feature = "profile-decode-layer")]
        let in_proj_qkv_us = t0.elapsed().as_micros() as u64;
        #[cfg(feature = "profile-decode-layer")]
        let in_proj_z_us = 0u64; // included in qkv timing above

        #[cfg(feature = "bench-decode-snapshot")]
        if crate::snapshot::is_snapshot_active() && crate::snapshot::should_snapshot_layer(layer_idx) {
            crate::snapshot::dump_f32(&format!("layer{}_qkv_output", layer_idx), &qkv);
            crate::snapshot::dump_f32(&format!("layer{}_z_output", layer_idx), &z);
        }

        #[cfg(feature = "profile-decode-layer")]
        let t0 = std::time::Instant::now();
        let mut a = vec![0.0f32; self.num_v_heads];
        Self::matvec_f32(x, &self.in_proj_a, self.num_v_heads, &mut a);
        let mut b = vec![0.0f32; self.num_v_heads];
        Self::matvec_f32(x, &self.in_proj_b, self.num_v_heads, &mut b);
        #[cfg(feature = "profile-decode-layer")]
        let in_proj_ab_us = t0.elapsed().as_micros() as u64;

        #[cfg(feature = "bench-decode-snapshot")]
        if crate::snapshot::is_snapshot_active() && crate::snapshot::should_snapshot_layer(layer_idx) {
            crate::snapshot::dump_f32(&format!("layer{}_a_output", layer_idx), &a);
            crate::snapshot::dump_f32(&format!("layer{}_b_output", layer_idx), &b);
        }

        // 2-9. Recurrent step
        #[cfg(feature = "profile-decode-layer")]
        let t0 = std::time::Instant::now();
        let mut attn_out = vec![0.0f32; value_dim];
        self.recurrent_step(&mut qkv, &z, &a, &b, kv_cache, layer_idx, &mut attn_out);
        #[cfg(feature = "profile-decode-layer")]
        let recurrent_step_us = t0.elapsed().as_micros() as u64;

        #[cfg(feature = "bench-decode-snapshot")]
        if crate::snapshot::is_snapshot_active() && crate::snapshot::should_snapshot_layer(layer_idx) {
            crate::snapshot::dump_f32(&format!("layer{}_out_proj_input", layer_idx), &attn_out);
        }

        // 10. Output projection (float dequant to avoid error accumulation)
        #[cfg(feature = "profile-decode-layer")]
        let t0 = std::time::Instant::now();
        L::matvec_float_dequant(&attn_out, &self.out_proj, output)?;
        #[cfg(feature = "profile-decode-layer")]
        let out_proj_us = t0.elapsed().as_micros() as u64;

        #[cfg(feature = "profile-decode-layer")]
        crate::profiler::set_decode_block_sub_timings(
            crate::profiler::DecodeBlockSubTimings::DeltaNet {
                in_proj_qkv_us,
                in_proj_z_us,
                in_proj_ab_us,
                recurrent_step_us,
                out_proj_us,
            },
        );

        Ok(())
    }

    /// Batched prefill: matmul projections, sequential recurrence, matmul output.
    pub fn forward_prefill(
        &self,
        x: &[f32],
        seq_len: usize,
        kv_cache: &mut CpuKvCache,
        layer_idx: usize,
        output: &mut [f32],
    ) -> Result<()> {
        let conv_dim = self.conv_dim();
        let value_dim = self.value_dim();

        // Phase 1: Batched projections (matmul)
        let mut qkv_all = std::mem::take(&mut kv_cache.ctx.prefill_deltanet_qkv);
        qkv_all.resize(seq_len * conv_dim, 0.0);
        L::matmul(x, &self.in_proj_qkv, &mut qkv_all, seq_len)?;

        let mut z_all = std::mem::take(&mut kv_cache.ctx.prefill_deltanet_z);
        z_all.resize(seq_len * value_dim, 0.0);
        L::matmul(x, &self.in_proj_z, &mut z_all, seq_len)?;

        // a/b projections: tiny (hidden_size→num_v_heads), keep as matvec per token
        let mut a_t = vec![0.0f32; self.num_v_heads];
        let mut b_t = vec![0.0f32; self.num_v_heads];

        // Phase 2: Sequential recurrence
        let mut attn_out_all = std::mem::take(&mut kv_cache.ctx.prefill_deltanet_attn_out);
        attn_out_all.resize(seq_len * value_dim, 0.0);

        for t in 0..seq_len {
            let x_t = &x[t * self.hidden_size..(t + 1) * self.hidden_size];
            let qkv_t = &mut qkv_all[t * conv_dim..(t + 1) * conv_dim];
            let z_t = &z_all[t * value_dim..(t + 1) * value_dim];
            let out_t = &mut attn_out_all[t * value_dim..(t + 1) * value_dim];
            out_t.fill(0.0);

            Self::matvec_f32(x_t, &self.in_proj_a, self.num_v_heads, &mut a_t);
            Self::matvec_f32(x_t, &self.in_proj_b, self.num_v_heads, &mut b_t);

            self.recurrent_step(qkv_t, z_t, &a_t, &b_t, kv_cache, layer_idx, out_t);
        }

        // Phase 3: Batched output projection (matmul)
        L::matmul(&attn_out_all, &self.out_proj, output, seq_len)?;

        // Return buffers to cache
        kv_cache.ctx.prefill_deltanet_qkv = qkv_all;
        kv_cache.ctx.prefill_deltanet_z = z_all;
        kv_cache.ctx.prefill_deltanet_attn_out = attn_out_all;

        Ok(())
    }
}

/// Layer block: attention, convolution (LFM2), or GatedDeltaNet (Qwen3.5).
pub enum LayerBlock<L: LinearOps> {
    Attention(GenericAttention<L>),
    Convolution(GenericConvolution<L>),
    GatedDeltaNet(GenericGatedDeltaNet<L>),
}

/// Decoder layer generic over the linear ops / weight type.
pub struct GenericDecoderLayer<L: LinearOps> {
    pub input_layernorm: Vec<BF16>,
    pub input_layernorm_bias: Option<Vec<f32>>,
    pub block: LayerBlock<L>,
    pub post_attention_layernorm: Vec<BF16>,
    pub post_attention_layernorm_bias: Option<Vec<f32>>,
    /// Gemma3: pre-feedforward layernorm (applied before FFN input).
    pub pre_feedforward_layernorm: Option<Vec<BF16>>,
    /// Gemma3: post-feedforward layernorm (applied after FFN output, before residual).
    pub post_feedforward_layernorm: Option<Vec<BF16>>,
    pub ffn: GenericFFN<L>,
    pub hidden_size: usize,
    pub rms_norm_eps: f32,
    pub norm_type: herbert_core::config::NormType,
}

/// Intermediate state from layer Phase A (norm + attention Phase A).
/// Used by decomposed prefill for distributed inference.
pub struct LayerPhaseAOutput {
    /// Normalized input (result of RMSNorm)
    pub norm1: Vec<f32>,
    /// Attention Phase A output (Q, K, V after proj + QK norm + RoPE)
    pub attn_phase_a: crate::generic_attention::prefill::PrefillPhaseAOutput,
}

impl<L: LinearOps> GenericDecoderLayer<L> {
    #[cfg(feature = "profile-decode-layer")]
    fn block_type_name(&self) -> &'static str {
        match &self.block {
            LayerBlock::Attention(_) => "Attn",
            LayerBlock::Convolution(_) => "Conv",
            LayerBlock::GatedDeltaNet(_) => "DeltaNet",
        }
    }

    /// Apply normalization (dispatch based on norm_type).
    #[inline]
    fn apply_norm(&self, input: &[f32], weight: &[BF16], bias: Option<&[f32]>, output: &mut [f32]) {
        let _ = bias; // unused after LayerNorm removal
        match self.norm_type {
            NormType::GemmaRMSNorm => {
                common_kernels::rms_norm_gemma_bf16(input, weight, output, self.rms_norm_eps);
            }
            NormType::RMSNorm => L::rms_norm(input, weight, output, self.rms_norm_eps),
        }
    }

    /// Dump snapshot data for this layer (weights, state, norms).
    #[cfg(feature = "bench-decode-snapshot")]
    pub fn dump_snapshot(
        &self,
        layer_idx: usize,
        x: &[f32],
        kv_cache: &CpuKvCache,
    ) {
        use crate::snapshot;
        let prefix = format!("layer{}", layer_idx);

        // Layer input
        snapshot::dump_f32(&format!("{}_input", prefix), x);

        // Norm weights
        snapshot::dump_bf16(&format!("{}_norm1_weight", prefix), &self.input_layernorm);
        snapshot::dump_bf16(
            &format!("{}_norm2_weight", prefix),
            &self.post_attention_layernorm,
        );

        // Block weights + state
        match &self.block {
            LayerBlock::GatedDeltaNet(gdn) => {
                eprintln!("[SNAPSHOT]   Layer {} block: GatedDeltaNet", layer_idx);
                snapshot::dump_weight::<L>(
                    &format!("{}_in_proj_qkv", prefix),
                    &gdn.in_proj_qkv,
                );
                snapshot::dump_weight::<L>(
                    &format!("{}_in_proj_z", prefix),
                    &gdn.in_proj_z,
                );
                snapshot::dump_f32(&format!("{}_in_proj_a", prefix), &gdn.in_proj_a);
                snapshot::dump_f32(&format!("{}_in_proj_b", prefix), &gdn.in_proj_b);
                snapshot::dump_f32(&format!("{}_conv1d_weight", prefix), &gdn.conv1d_weight);
                snapshot::dump_f32(&format!("{}_a_log", prefix), &gdn.a_log);
                snapshot::dump_f32(&format!("{}_dt_bias", prefix), &gdn.dt_bias);
                snapshot::dump_bf16(&format!("{}_gated_norm_weight", prefix), &gdn.norm_weight);
                snapshot::dump_weight::<L>(
                    &format!("{}_out_proj", prefix),
                    &gdn.out_proj,
                );

                // DeltaNet state
                snapshot::dump_f32(
                    &format!("{}_recurrent_state", prefix),
                    &kv_cache.kv.deltanet_recurrent[layer_idx],
                );
                snapshot::dump_f32(
                    &format!("{}_conv_state", prefix),
                    &kv_cache.kv.deltanet_conv[layer_idx],
                );

                // Config scalars
                snapshot::dump_usize(&format!("{}_num_k_heads", prefix), gdn.num_k_heads);
                snapshot::dump_usize(&format!("{}_num_v_heads", prefix), gdn.num_v_heads);
                snapshot::dump_usize(&format!("{}_head_k_dim", prefix), gdn.head_k_dim);
                snapshot::dump_usize(&format!("{}_head_v_dim", prefix), gdn.head_v_dim);
                snapshot::dump_usize(
                    &format!("{}_conv_kernel_size", prefix),
                    gdn.conv_kernel_size,
                );
            }
            LayerBlock::Attention(attn) => {
                eprintln!("[SNAPSHOT]   Layer {} block: Attention", layer_idx);
                snapshot::dump_weight::<L>(
                    &format!("{}_q_proj", prefix),
                    &attn.q_proj,
                );
                snapshot::dump_weight::<L>(
                    &format!("{}_k_proj", prefix),
                    &attn.k_proj,
                );
                snapshot::dump_weight::<L>(
                    &format!("{}_v_proj", prefix),
                    &attn.v_proj,
                );
                snapshot::dump_weight::<L>(
                    &format!("{}_o_proj", prefix),
                    &attn.o_proj,
                );
                if let Some(ref qn) = attn.q_norm {
                    snapshot::dump_bf16(&format!("{}_q_norm", prefix), qn);
                }
                if let Some(ref kn) = attn.k_norm {
                    snapshot::dump_bf16(&format!("{}_k_norm", prefix), kn);
                }

                // Attention config
                snapshot::dump_usize(&format!("{}_num_heads", prefix), attn.num_heads);
                snapshot::dump_usize(&format!("{}_num_kv_heads", prefix), attn.num_kv_heads);
                snapshot::dump_usize(&format!("{}_head_dim", prefix), attn.head_dim);
            }
            LayerBlock::Convolution(_) => {
                eprintln!("[SNAPSHOT]   Layer {} block: Convolution (not dumped)", layer_idx);
            }
        }

        // MoE / FFN weights
        self.ffn.dump_snapshot(&prefix);
    }

    pub fn forward_decode(
        &self,
        x: &mut [f32],
        cos: &[f32],
        sin: &[f32],
        kv_cache: &mut CpuKvCache,
        layer_idx: usize,
        pos: usize,
    ) -> Result<()> {
        let mut norm1 = std::mem::take(&mut kv_cache.ctx.decode_norm1[layer_idx]);
        norm1.resize(self.hidden_size, 0.0);
        #[cfg(feature = "profile-decode-layer")]
        let t0 = std::time::Instant::now();
        self.apply_norm(x, &self.input_layernorm, self.input_layernorm_bias.as_deref(), &mut norm1);
        #[cfg(feature = "profile-decode-layer")]
        let norm1_us = t0.elapsed().as_micros() as u64;

        #[cfg(feature = "bench-decode-snapshot")]
        if crate::snapshot::is_snapshot_active() && crate::snapshot::should_snapshot_layer(layer_idx) {
            crate::snapshot::dump_f32(&format!("layer{}_normed1", layer_idx), &norm1);
        }

        let mut attn_proj_out = std::mem::take(&mut kv_cache.ctx.decode_attn_proj_out[layer_idx]);
        attn_proj_out.resize(self.hidden_size, 0.0);
        #[cfg(feature = "profile-decode-layer")]
        let t0 = std::time::Instant::now();
        match &self.block {
            LayerBlock::Attention(attn) => {
                attn.forward_decode(
                    &norm1,
                    cos,
                    sin,
                    kv_cache,
                    layer_idx,
                    pos,
                    &mut attn_proj_out,
                )?;
            }
            LayerBlock::Convolution(conv) => {
                conv.forward_decode(&norm1, kv_cache, layer_idx, &mut attn_proj_out)?;
            }
            LayerBlock::GatedDeltaNet(gdn) => {
                gdn.forward_decode(&norm1, kv_cache, layer_idx, &mut attn_proj_out)?;
            }
        }
        #[cfg(feature = "profile-decode-layer")]
        let attention_us = t0.elapsed().as_micros() as u64;
        kv_cache.ctx.decode_norm1[layer_idx] = norm1;

        // Gemma3 4-norm: apply post_attention_layernorm to attn output before residual
        if let Some(ref post_attn_norm) = self.post_feedforward_layernorm {
            // 4-norm path: post_attention_layernorm wraps attn output
            let mut normed = vec![0.0f32; self.hidden_size];
            self.apply_norm(&attn_proj_out, &self.post_attention_layernorm, None, &mut normed);
            for i in 0..self.hidden_size {
                x[i] += normed[i];
            }
            // pre_feedforward_layernorm for FFN input
            let mut norm2 = std::mem::take(&mut kv_cache.ctx.decode_norm2[layer_idx]);
            norm2.resize(self.hidden_size, 0.0);
            #[cfg(feature = "profile-decode-layer")]
            let t0 = std::time::Instant::now();
            self.apply_norm(x, self.pre_feedforward_layernorm.as_ref().unwrap(), None, &mut norm2);
            #[cfg(feature = "profile-decode-layer")]
            let norm2_us = t0.elapsed().as_micros() as u64;

            let mut mlp_gate = std::mem::take(&mut kv_cache.ctx.decode_mlp_gate[layer_idx]);
            let mut mlp_up = std::mem::take(&mut kv_cache.ctx.decode_mlp_up[layer_idx]);
            let mut mlp_out = std::mem::take(&mut kv_cache.ctx.decode_mlp_out[layer_idx]);
            mlp_gate.resize(self.ffn.intermediate_size(), 0.0);
            mlp_up.resize(self.ffn.intermediate_size(), 0.0);
            mlp_out.resize(self.hidden_size, 0.0);
            let mut moe_router = std::mem::take(&mut kv_cache.ctx.decode_moe_router);
            let mut moe_expert_out = std::mem::take(&mut kv_cache.ctx.decode_moe_expert_out);
            #[cfg(feature = "profile-decode-layer")]
            let t0 = std::time::Instant::now();
            self.ffn
                .forward_decode(&norm2, &mut mlp_gate, &mut mlp_up, &mut mlp_out, &mut moe_router, &mut moe_expert_out)?;
            #[cfg(feature = "profile-decode-layer")]
            let ffn_us = t0.elapsed().as_micros() as u64;

            // post_feedforward_layernorm on FFN output
            let mut ffn_normed = vec![0.0f32; self.hidden_size];
            self.apply_norm(&mlp_out, post_attn_norm, None, &mut ffn_normed);
            for i in 0..self.hidden_size {
                x[i] += ffn_normed[i];
            }

            kv_cache.ctx.decode_norm2[layer_idx] = norm2;
            kv_cache.ctx.decode_moe_router = moe_router;
            kv_cache.ctx.decode_moe_expert_out = moe_expert_out;
            kv_cache.ctx.decode_attn_proj_out[layer_idx] = attn_proj_out;
            kv_cache.ctx.decode_mlp_gate[layer_idx] = mlp_gate;
            kv_cache.ctx.decode_mlp_up[layer_idx] = mlp_up;
            kv_cache.ctx.decode_mlp_out[layer_idx] = mlp_out;

            #[cfg(feature = "profile-decode-layer")]
            crate::profiler::push_decode_layer_detail(crate::profiler::DecodeLayerDetailProfile {
                norm1_us,
                attention_us,
                norm2_us,
                ffn_us,
                layer_type: self.block_type_name(),
                block_sub: crate::profiler::take_decode_block_sub_timings(),
            });

            return Ok(());
        }

        // Standard 2-norm path
        #[cfg(feature = "bench-decode-snapshot")]
        if crate::snapshot::is_snapshot_active() && crate::snapshot::should_snapshot_layer(layer_idx) {
            crate::snapshot::dump_f32(&format!("layer{}_block_output", layer_idx), &attn_proj_out);
        }

        let mut norm2 = std::mem::take(&mut kv_cache.ctx.decode_norm2[layer_idx]);
        norm2.resize(self.hidden_size, 0.0);
        #[cfg(feature = "profile-decode-layer")]
        let t0 = std::time::Instant::now();
        // Fused residual + RMSNorm + prequantize: residual add + norm + quantize
        // in minimal passes, eliminating one full load+store of x vs separate ops.
        let pq = if self.norm_type == NormType::RMSNorm
            && self.post_attention_layernorm_bias.is_none()
        {
            L::fused_residual_rmsnorm_prequantize(x, &attn_proj_out, &self.post_attention_layernorm, &mut norm2, self.rms_norm_eps)
        } else {
            for i in 0..self.hidden_size {
                x[i] += attn_proj_out[i];
            }
            self.apply_norm(x, &self.post_attention_layernorm, self.post_attention_layernorm_bias.as_deref(), &mut norm2);
            None
        };
        #[cfg(feature = "profile-decode-layer")]
        let norm2_us = t0.elapsed().as_micros() as u64;

        #[cfg(feature = "bench-decode-snapshot")]
        if crate::snapshot::is_snapshot_active() && crate::snapshot::should_snapshot_layer(layer_idx) {
            crate::snapshot::dump_f32(&format!("layer{}_norm2_output", layer_idx), &norm2);
        }

        let mut mlp_gate = std::mem::take(&mut kv_cache.ctx.decode_mlp_gate[layer_idx]);
        let mut mlp_up = std::mem::take(&mut kv_cache.ctx.decode_mlp_up[layer_idx]);
        let mut mlp_out = std::mem::take(&mut kv_cache.ctx.decode_mlp_out[layer_idx]);
        mlp_gate.resize(self.ffn.intermediate_size(), 0.0);
        mlp_up.resize(self.ffn.intermediate_size(), 0.0);
        mlp_out.resize(self.hidden_size, 0.0);
        let mut moe_router = std::mem::take(&mut kv_cache.ctx.decode_moe_router);
        let mut moe_expert_out = std::mem::take(&mut kv_cache.ctx.decode_moe_expert_out);
        #[cfg(feature = "profile-decode-layer")]
        let t0 = std::time::Instant::now();
        self.ffn
            .forward_decode_with_pq(&norm2, pq, &mut mlp_gate, &mut mlp_up, &mut mlp_out, &mut moe_router, &mut moe_expert_out)?;
        #[cfg(feature = "profile-decode-layer")]
        let ffn_us = t0.elapsed().as_micros() as u64;
        kv_cache.ctx.decode_norm2[layer_idx] = norm2;
        kv_cache.ctx.decode_moe_router = moe_router;
        kv_cache.ctx.decode_moe_expert_out = moe_expert_out;

        #[cfg(feature = "bench-decode-snapshot")]
        if crate::snapshot::is_snapshot_active() && crate::snapshot::should_snapshot_layer(layer_idx) {
            crate::snapshot::dump_f32(&format!("layer{}_ffn_output", layer_idx), &mlp_out);
        }

        for i in 0..self.hidden_size {
            x[i] += mlp_out[i];
        }

        kv_cache.ctx.decode_attn_proj_out[layer_idx] = attn_proj_out;
        kv_cache.ctx.decode_mlp_gate[layer_idx] = mlp_gate;
        kv_cache.ctx.decode_mlp_up[layer_idx] = mlp_up;
        kv_cache.ctx.decode_mlp_out[layer_idx] = mlp_out;

        #[cfg(feature = "profile-decode-layer")]
        crate::profiler::push_decode_layer_detail(crate::profiler::DecodeLayerDetailProfile {
            norm1_us,
            attention_us,
            norm2_us,
            ffn_us,
            layer_type: self.block_type_name(),
            block_sub: crate::profiler::take_decode_block_sub_timings(),
        });

        Ok(())
    }

    /// Run norm1 -> attention -> residual1 -> norm2.
    /// Returns the post-norm2 hidden state (input for MoE routing + experts).
    /// Used by distributed decode to split attention from FFN/MoE.
    ///
    /// `x` is mutated in-place (residual1 applied).
    pub fn forward_decode_through_norm2(
        &self,
        x: &mut [f32],
        cos: &[f32],
        sin: &[f32],
        kv_cache: &mut CpuKvCache,
        layer_idx: usize,
        pos: usize,
    ) -> Result<Vec<f32>> {
        // norm1
        let mut norm1 = std::mem::take(&mut kv_cache.ctx.decode_norm1[layer_idx]);
        norm1.resize(self.hidden_size, 0.0);
        self.apply_norm(x, &self.input_layernorm, self.input_layernorm_bias.as_deref(), &mut norm1);

        // attention
        let mut attn_proj_out = std::mem::take(&mut kv_cache.ctx.decode_attn_proj_out[layer_idx]);
        attn_proj_out.resize(self.hidden_size, 0.0);
        match &self.block {
            LayerBlock::Attention(attn) => {
                attn.forward_decode(&norm1, cos, sin, kv_cache, layer_idx, pos, &mut attn_proj_out)?;
            }
            LayerBlock::Convolution(conv) => {
                conv.forward_decode(&norm1, kv_cache, layer_idx, &mut attn_proj_out)?;
            }
            LayerBlock::GatedDeltaNet(gdn) => {
                gdn.forward_decode(&norm1, kv_cache, layer_idx, &mut attn_proj_out)?;
            }
        }
        kv_cache.ctx.decode_norm1[layer_idx] = norm1;

        kv_cache.ctx.decode_attn_proj_out[layer_idx] = attn_proj_out;

        // Fused residual1 + norm2: saves one load+store of x
        let mut norm2 = std::mem::take(&mut kv_cache.ctx.decode_norm2[layer_idx]);
        norm2.resize(self.hidden_size, 0.0);
        if self.norm_type == NormType::RMSNorm && self.post_attention_layernorm_bias.is_none() {
            let attn_out = &kv_cache.ctx.decode_attn_proj_out[layer_idx];
            L::fused_rms_norm_residual(x, attn_out, &self.post_attention_layernorm, &mut norm2, self.rms_norm_eps);
        } else {
            let attn_out = &kv_cache.ctx.decode_attn_proj_out[layer_idx];
            for i in 0..self.hidden_size {
                x[i] += attn_out[i];
            }
            self.apply_norm(x, &self.post_attention_layernorm, self.post_attention_layernorm_bias.as_deref(), &mut norm2);
        }
        let norm2_out = norm2.clone();
        kv_cache.ctx.decode_norm2[layer_idx] = norm2;

        Ok(norm2_out)
    }

    /// Apply FFN/MoE result + residual2.
    /// Called after receiving aggregated expert output from all machines.
    #[inline]
    pub fn apply_ffn_residual(
        &self,
        x: &mut [f32],
        ffn_output: &[f32],
    ) {
        for i in 0..self.hidden_size {
            x[i] += ffn_output[i];
        }
    }

    /// Apply normalization to a batch of vectors.
    fn apply_norm_batch(&self, input: &[f32], weight: &[BF16], bias: Option<&[f32]>, output: &mut [f32], seq_len: usize) -> Result<()> {
        let _ = bias; // unused after LayerNorm removal
        match self.norm_type {
            NormType::GemmaRMSNorm => {
                for i in 0..seq_len {
                    let offset = i * self.hidden_size;
                    common_kernels::rms_norm_gemma_bf16(
                        &input[offset..offset + self.hidden_size],
                        weight,
                        &mut output[offset..offset + self.hidden_size],
                        self.rms_norm_eps,
                    );
                }
                Ok(())
            }
            NormType::RMSNorm => {
                L::rms_norm_batch(input, weight, output, seq_len, self.hidden_size, self.rms_norm_eps)
            }
        }
    }

    pub fn forward_prefill(
        &self,
        x: &[f32],
        cos_cache: &[f32],
        sin_cache: &[f32],
        kv_cache: &mut CpuKvCache,
        layer_idx: usize,
        seq_len: usize,
        start_pos: usize,
        output: &mut [f32],
    ) -> Result<()> {
        let n = seq_len * self.hidden_size;

        // norm1 (reuse buffer across layers)
        #[cfg(feature = "profile-layer")]
        let t0 = std::time::Instant::now();
        #[cfg(feature = "profile-prefill-layer")]
        let t_ppl = std::time::Instant::now();
        let mut norm1 = std::mem::take(&mut kv_cache.ctx.prefill_norm1);
        norm1.resize(n, 0.0);
        self.apply_norm_batch(x, &self.input_layernorm, self.input_layernorm_bias.as_deref(), &mut norm1, seq_len)?;
        #[cfg(feature = "profile-layer")]
        let norm1_us = t0.elapsed().as_micros() as u64;
        #[cfg(feature = "profile-prefill-layer")]
        let rmsnorm1_us = t_ppl.elapsed().as_micros() as u64;

        // block (attention or convolution) -> writes to o_proj scratch
        #[cfg(feature = "profile-layer")]
        let t0 = std::time::Instant::now();
        let mut o_proj = std::mem::take(&mut kv_cache.ctx.prefill_o_proj);
        o_proj.resize(n, 0.0);
        match &self.block {
            LayerBlock::Attention(attn) => {
                attn.forward_prefill(
                    &norm1, cos_cache, sin_cache, kv_cache, layer_idx, seq_len, start_pos, &mut o_proj,
                )?;
            }
            LayerBlock::Convolution(conv) => {
                conv.forward_prefill(&norm1, seq_len, kv_cache, layer_idx, &mut o_proj)?;
            }
            LayerBlock::GatedDeltaNet(gdn) => {
                gdn.forward_prefill(&norm1, seq_len, kv_cache, layer_idx, &mut o_proj)?;
            }
        }
        #[cfg(feature = "profile-layer")]
        let attention_us = t0.elapsed().as_micros() as u64;

        kv_cache.ctx.prefill_norm1 = norm1;

        // Gemma3 4-norm path for prefill
        if self.post_feedforward_layernorm.is_some() {
            // post_attention_layernorm on attn output
            let mut attn_normed = std::mem::take(&mut kv_cache.ctx.prefill_norm2);
            attn_normed.resize(n, 0.0);
            self.apply_norm_batch(&o_proj, &self.post_attention_layernorm, None, &mut attn_normed, seq_len)?;
            // residual1: output = x + post_attn_norm(attn_out)
            for i in 0..n {
                output[i] = x[i] + attn_normed[i];
            }
            kv_cache.ctx.prefill_o_proj = o_proj;

            // pre_feedforward_layernorm
            let mut norm_ffn_in = attn_normed; // reuse buffer
            norm_ffn_in.resize(n, 0.0);
            self.apply_norm_batch(output, self.pre_feedforward_layernorm.as_ref().unwrap(), None, &mut norm_ffn_in, seq_len)?;

            #[cfg(feature = "profile-layer")]
            let t0 = std::time::Instant::now();
            let mut ffn_out = std::mem::take(&mut kv_cache.ctx.prefill_ffn_out);
            ffn_out.resize(n, 0.0);
            self.ffn.forward_prefill(&norm_ffn_in, seq_len, kv_cache, &mut ffn_out)?;
            #[cfg(feature = "profile-layer")]
            let ffn_us = t0.elapsed().as_micros() as u64;

            // post_feedforward_layernorm on FFN output
            let mut ffn_normed = norm_ffn_in;
            ffn_normed.resize(n, 0.0);
            self.apply_norm_batch(&ffn_out, self.post_feedforward_layernorm.as_ref().unwrap(), None, &mut ffn_normed, seq_len)?;
            for i in 0..n {
                output[i] += ffn_normed[i];
            }

            kv_cache.ctx.prefill_norm2 = ffn_normed;
            kv_cache.ctx.prefill_ffn_out = ffn_out;

            #[cfg(feature = "profile-layer")]
            {
                let norm2_us = 0u64;
                crate::profiler::push_layer(crate::profiler::LayerStepProfile {
                    norm1_us,
                    attention_us,
                    norm2_us,
                    ffn_us,
                });
            }

            return Ok(());
        }

        // Standard 2-norm prefill path
        // residual1: output = x + o_proj (writes into ping-pong buffer)
        #[cfg(feature = "profile-prefill-layer")]
        let t_ppl = std::time::Instant::now();
        for i in 0..n {
            output[i] = x[i] + o_proj[i];
        }
        #[cfg(feature = "profile-prefill-layer")]
        let residual1_us = t_ppl.elapsed().as_micros() as u64;
        kv_cache.ctx.prefill_o_proj = o_proj;

        // norm2 (reuse buffer across layers)
        #[cfg(feature = "profile-layer")]
        let t0 = std::time::Instant::now();
        #[cfg(feature = "profile-prefill-layer")]
        let t_ppl = std::time::Instant::now();
        let mut norm2 = std::mem::take(&mut kv_cache.ctx.prefill_norm2);
        norm2.resize(n, 0.0);
        self.apply_norm_batch(output, &self.post_attention_layernorm, self.post_attention_layernorm_bias.as_deref(), &mut norm2, seq_len)?;
        #[cfg(feature = "profile-layer")]
        let norm2_us = t0.elapsed().as_micros() as u64;
        #[cfg(feature = "profile-prefill-layer")]
        let rmsnorm2_us = t_ppl.elapsed().as_micros() as u64;

        // ffn → writes to ffn_out scratch
        #[cfg(feature = "profile-layer")]
        let t0 = std::time::Instant::now();
        #[cfg(feature = "profile-prefill-layer")]
        let t_ppl = std::time::Instant::now();
        let mut ffn_out = std::mem::take(&mut kv_cache.ctx.prefill_ffn_out);
        ffn_out.resize(n, 0.0);
        self.ffn.forward_prefill(&norm2, seq_len, kv_cache, &mut ffn_out)?;
        #[cfg(feature = "profile-layer")]
        let ffn_us = t0.elapsed().as_micros() as u64;
        #[cfg(feature = "profile-prefill-layer")]
        let ffn_us_ppl = t_ppl.elapsed().as_micros() as u64;

        kv_cache.ctx.prefill_norm2 = norm2;

        // residual2: output += ffn_out
        #[cfg(feature = "profile-prefill-layer")]
        let t_ppl = std::time::Instant::now();
        for i in 0..n {
            output[i] += ffn_out[i];
        }
        #[cfg(feature = "profile-prefill-layer")]
        let residual2_us = t_ppl.elapsed().as_micros() as u64;
        kv_cache.ctx.prefill_ffn_out = ffn_out;

        #[cfg(feature = "profile-layer")]
        crate::profiler::push_layer(crate::profiler::LayerStepProfile {
            norm1_us,
            attention_us,
            norm2_us,
            ffn_us,
        });

        #[cfg(feature = "profile-prefill-layer")]
        {
            let block = if let Some(attn) = crate::profiler::take_attn_sub_timings() {
                crate::profiler::BlockSubTimings::Attn {
                    qkv_proj_us: attn.qkv_proj_us,
                    qk_norm_us: attn.qk_norm_us,
                    rope_us: attn.rope_us,
                    kv_append_us: attn.kv_append_us,
                    attn_kernel_us: attn.attn_kernel_us,
                    o_proj_us: attn.o_proj_us,
                }
            } else if let Some(conv) = crate::profiler::take_conv_sub_timings() {
                crate::profiler::BlockSubTimings::Conv {
                    in_proj_us: conv.in_proj_us,
                    gating_us: conv.gating_us,
                    conv_kernel_us: conv.conv_kernel_us,
                    out_proj_us: conv.out_proj_us,
                }
            } else {
                crate::profiler::BlockSubTimings::Attn {
                    qkv_proj_us: 0,
                    qk_norm_us: 0,
                    rope_us: 0,
                    kv_append_us: 0,
                    attn_kernel_us: 0,
                    o_proj_us: 0,
                }
            };
            crate::profiler::push_prefill_layer_detail(
                crate::profiler::PrefillLayerDetailProfile {
                    rmsnorm1_us,
                    block,
                    residual1_us,
                    rmsnorm2_us,
                    ffn_us: ffn_us_ppl,
                    residual2_us,
                },
            );
        }

        Ok(())
    }

    /// Phase A of a layer: input normalization + QKV projections + QK norm + RoPE.
    /// Independent of remote KV data.
    ///
    /// Only supported for Attention layers (returns error for Conv/GDN).
    pub fn forward_prefill_phase_a(
        &self,
        x: &[f32],
        cos_cache: &[f32],
        sin_cache: &[f32],
        kv_cache: &mut CpuKvCache,
        seq_len: usize,
        start_pos: usize,
    ) -> Result<LayerPhaseAOutput> {
        let n = seq_len * self.hidden_size;

        // norm1
        let mut norm1 = std::mem::take(&mut kv_cache.ctx.prefill_norm1);
        norm1.resize(n, 0.0);
        self.apply_norm_batch(x, &self.input_layernorm, self.input_layernorm_bias.as_deref(), &mut norm1, seq_len)?;

        // attention phase A
        let attn = match &self.block {
            LayerBlock::Attention(attn) => attn,
            _ => return Err(herbert_core::error::HerbertError::Backend(
                "decomposed prefill only supports Attention layers".to_string(),
            )),
        };

        let attn_phase_a = attn.forward_prefill_phase_a(
            &norm1, cos_cache, sin_cache, kv_cache, seq_len, start_pos,
        )?;

        Ok(LayerPhaseAOutput { norm1, attn_phase_a })
    }

    /// Phase B of a layer: KV append + attention kernel + O proj + residual + FFN.
    /// Requires remote KV to be already inserted into kv_cache.
    pub fn forward_prefill_phase_b(
        &self,
        x: &[f32],
        phase_a: LayerPhaseAOutput,
        kv_cache: &mut CpuKvCache,
        layer_idx: usize,
        seq_len: usize,
        start_pos: usize,
        output: &mut [f32],
    ) -> Result<()> {
        let n = seq_len * self.hidden_size;
        let LayerPhaseAOutput { norm1, attn_phase_a } = phase_a;

        // attention phase B → writes to o_proj scratch
        let mut o_proj = std::mem::take(&mut kv_cache.ctx.prefill_o_proj);
        o_proj.resize(n, 0.0);

        let attn = match &self.block {
            LayerBlock::Attention(attn) => attn,
            _ => unreachable!(), // Phase A already checked
        };
        attn.forward_prefill_phase_b(attn_phase_a, kv_cache, layer_idx, start_pos, &mut o_proj)?;

        kv_cache.ctx.prefill_norm1 = norm1;

        // Handle Gemma3 4-norm path
        if self.post_feedforward_layernorm.is_some() {
            let mut attn_normed = std::mem::take(&mut kv_cache.ctx.prefill_norm2);
            attn_normed.resize(n, 0.0);
            self.apply_norm_batch(&o_proj, &self.post_attention_layernorm, None, &mut attn_normed, seq_len)?;
            for i in 0..n {
                output[i] = x[i] + attn_normed[i];
            }
            kv_cache.ctx.prefill_o_proj = o_proj;

            let mut norm_ffn_in = attn_normed;
            norm_ffn_in.resize(n, 0.0);
            self.apply_norm_batch(output, self.pre_feedforward_layernorm.as_ref().unwrap(), None, &mut norm_ffn_in, seq_len)?;

            let mut ffn_out = std::mem::take(&mut kv_cache.ctx.prefill_ffn_out);
            ffn_out.resize(n, 0.0);
            self.ffn.forward_prefill(&norm_ffn_in, seq_len, kv_cache, &mut ffn_out)?;

            let mut ffn_normed = norm_ffn_in;
            ffn_normed.resize(n, 0.0);
            self.apply_norm_batch(&ffn_out, self.post_feedforward_layernorm.as_ref().unwrap(), None, &mut ffn_normed, seq_len)?;
            for i in 0..n {
                output[i] += ffn_normed[i];
            }

            kv_cache.ctx.prefill_norm2 = ffn_normed;
            kv_cache.ctx.prefill_ffn_out = ffn_out;
            return Ok(());
        }

        // Standard 2-norm path
        // residual1
        for i in 0..n {
            output[i] = x[i] + o_proj[i];
        }
        kv_cache.ctx.prefill_o_proj = o_proj;

        // norm2
        let mut norm2 = std::mem::take(&mut kv_cache.ctx.prefill_norm2);
        norm2.resize(n, 0.0);
        self.apply_norm_batch(output, &self.post_attention_layernorm, self.post_attention_layernorm_bias.as_deref(), &mut norm2, seq_len)?;

        // FFN
        let mut ffn_out = std::mem::take(&mut kv_cache.ctx.prefill_ffn_out);
        ffn_out.resize(n, 0.0);
        self.ffn.forward_prefill(&norm2, seq_len, kv_cache, &mut ffn_out)?;

        kv_cache.ctx.prefill_norm2 = norm2;

        // residual2
        for i in 0..n {
            output[i] += ffn_out[i];
        }
        kv_cache.ctx.prefill_ffn_out = ffn_out;

        Ok(())
    }
}
