//! MRoPE (Multimodal Rotary Position Embedding) support for VL models.
//!
//! Implements 3D interleaved position encoding for (T, H, W) coordinates.
//! The key insight: the same rotation kernel (`apply_rope_position_unchecked`)
//! is reused — only the cos/sin computation changes (3D interleaved vs 1D sequential).

/// Build the interleaved MRoPE pattern mapping each frequency pair to a coordinate axis.
///
/// For `section = [24, 20, 20]` (head_dim=128, half_dim=64):
/// - `num_triplets = min(h_section, w_section) = 20`
/// - First 60 entries: `[T, H, W, T, H, W, ...]` (20 complete triplets)
/// - Last 4 entries: `[T, T, T, T]` (remaining T dimensions)
///
/// Returns a Vec of length `half_dim` where each value is 0 (T), 1 (H), or 2 (W).
pub fn build_mrope_pattern(section: [usize; 3]) -> Vec<u8> {
    let half_dim: usize = section.iter().sum();
    let mut pattern = vec![0u8; half_dim];
    let num_triplets = section[1].min(section[2]);

    for i in 0..num_triplets {
        pattern[i * 3] = 0;     // T
        pattern[i * 3 + 1] = 1; // H
        pattern[i * 3 + 2] = 2; // W
    }
    // Remaining positions are already 0 (T) from vec initialization

    pattern
}

/// Compute inverse frequencies for RoPE: `inv_freq[i] = 1.0 / theta^(2i/head_dim)`.
///
/// Returns a Vec of length `head_dim / 2`.
pub fn compute_inv_freq(head_dim: usize, rope_theta: f32) -> Vec<f32> {
    let half_dim = head_dim / 2;
    let mut inv_freq = vec![0.0f32; half_dim];
    for i in 0..half_dim {
        inv_freq[i] = 1.0 / rope_theta.powf((2 * i) as f32 / head_dim as f32);
    }
    inv_freq
}

/// Compute MRoPE cos/sin for a single token with 3D position.
///
/// `pos_3d`: `[t, h, w]` coordinates for this token.
/// `inv_freq`: precomputed inverse frequencies (length = half_dim).
/// `pattern`: interleaved pattern (length = half_dim), values 0/1/2.
/// `cos_out`, `sin_out`: output slices (length = half_dim).
pub fn compute_mrope_cos_sin(
    pos_3d: [u32; 3],
    inv_freq: &[f32],
    pattern: &[u8],
    cos_out: &mut [f32],
    sin_out: &mut [f32],
) {
    let half_dim = inv_freq.len();
    debug_assert_eq!(pattern.len(), half_dim);
    debug_assert!(cos_out.len() >= half_dim);
    debug_assert!(sin_out.len() >= half_dim);

    for j in 0..half_dim {
        let coord = pattern[j] as usize; // 0=T, 1=H, 2=W
        let freq = pos_3d[coord] as f32 * inv_freq[j];
        cos_out[j] = freq.cos();
        sin_out[j] = freq.sin();
    }
}

