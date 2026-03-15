//! H2O (Heavy Hitter Oracle) KV cache eviction for CPU backends.
//!
//! When the KV cache exceeds a budget, evicts low-importance positions
//! based on cumulative Q·K attention scores from probe layers.
//!
//! Flow: score_probe → select_kept → compact → update seq_len
//!
//! Three protected zones:
//! - Sinks: first 8 positions (always kept, attention sinks)
//! - Recent window: last 128 positions (always kept, active context)
//! - Middle: scored and top-K kept by importance

use crate::kv_cache::{CpuKvCache, KvLayerData};
use herbert_core::config::Config;
use herbert_core::tensor::bf16_to_f32;

/// Number of initial positions always kept (attention sinks).
const H2O_SINK_COUNT: usize = 8;

/// Number of recent positions always kept (active context window).
const H2O_RECENT_WINDOW: usize = 128;

/// Select 3 late probe layers for score computation.
fn probe_layers(num_layers: usize) -> [usize; 3] {
    [
        num_layers.saturating_sub(8),
        num_layers.saturating_sub(4),
        num_layers.saturating_sub(1),
    ]
}

/// Compute H2O eviction hysteresis: don't evict until budget + hysteresis exceeded.
/// Prevents eviction thrashing (re-evicting every single decode step).
pub fn hysteresis(budget: usize) -> usize {
    (budget / 8).min(512)
}

/// Compute H2O importance scores by probing Q·K dot products on late layers.
///
/// Accumulates scores across `probe_layers`: for each cached position,
/// computes max(0, Q·K * scale) summed over all Q heads and probe layers.
/// Higher score = more important position.
fn score_probe(
    kv_cache: &CpuKvCache,
    config: &Config,
    scores: &mut [f32],
) {
    let cached_len = scores.len();
    let head_dim = config.head_dim;
    let num_heads = config.num_attention_heads;
    let num_kv_heads = config.num_key_value_heads;
    let heads_per_kv = num_heads / num_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();

    scores.fill(0.0);

    let layers = probe_layers(config.num_layers);

    for &layer_idx in &layers {
        // Q from the last decode step (still in decode_q buffer)
        let q = &kv_cache.ctx.decode_q[layer_idx];
        let q_dim = config.q_dim();

        match &kv_cache.kv.layer_data[layer_idx] {
            KvLayerData::F32 { keys_per_head, .. } => {
                for kv_h in 0..num_kv_heads {
                    let k_head = &keys_per_head[kv_h];
                    for t in 0..cached_len {
                        let k_offset = t * head_dim;
                        let mut max_score = 0.0f32;
                        for qh_local in 0..heads_per_kv {
                            let h = kv_h * heads_per_kv + qh_local;
                            let q_offset = h * head_dim;
                            let mut dot = 0.0f32;
                            for d in 0..head_dim {
                                dot += q[q_offset + d] * k_head[k_offset + d];
                            }
                            max_score = max_score.max(dot * scale);
                        }
                        scores[t] += max_score.max(0.0);
                    }
                }
            }
            KvLayerData::BF16 { keys_per_head, .. } => {
                for kv_h in 0..num_kv_heads {
                    let k_head = &keys_per_head[kv_h];
                    for t in 0..cached_len {
                        let k_offset = t * head_dim;
                        let mut max_score = 0.0f32;
                        for qh_local in 0..heads_per_kv {
                            let h = kv_h * heads_per_kv + qh_local;
                            let q_offset = h * head_dim;
                            let mut dot = 0.0f32;
                            for d in 0..head_dim {
                                dot += q[q_offset + d] * bf16_to_f32(k_head[k_offset + d]);
                            }
                            max_score = max_score.max(dot * scale);
                        }
                        scores[t] += max_score.max(0.0);
                    }
                }
            }
            KvLayerData::INT8 { keys_per_head, keys_scales, .. } => {
                for kv_h in 0..num_kv_heads {
                    let k_head = &keys_per_head[kv_h];
                    let k_scales = &keys_scales[kv_h];
                    for t in 0..cached_len {
                        let k_offset = t * head_dim;
                        let k_scale = k_scales[t];
                        let mut max_score = 0.0f32;
                        for qh_local in 0..heads_per_kv {
                            let h = kv_h * heads_per_kv + qh_local;
                            let q_offset = h * head_dim;
                            let mut dot = 0.0f32;
                            for d in 0..head_dim {
                                dot += q[q_offset + d] * (k_head[k_offset + d] as f32);
                            }
                            max_score = max_score.max(dot * k_scale * scale);
                        }
                        scores[t] += max_score.max(0.0);
                    }
                }
            }
            KvLayerData::INT4 { keys_per_head, keys_scales, .. } => {
                let packed_per_pos = head_dim / 2;
                let num_groups = head_dim / 32;
                for kv_h in 0..num_kv_heads {
                    let k_head = &keys_per_head[kv_h];
                    let k_scales_flat = &keys_scales[kv_h];
                    for t in 0..cached_len {
                        let k_offset = t * packed_per_pos;
                        let k_gs_base = t * num_groups;
                        let mut max_score = 0.0f32;
                        for qh_local in 0..heads_per_kv {
                            let h = kv_h * heads_per_kv + qh_local;
                            let q_offset = h * head_dim;
                            let mut dot = 0.0f32;
                            for g in 0..num_groups {
                                let gs = k_scales_flat[k_gs_base + g];
                                let packed = &k_head[k_offset + g * 16..k_offset + (g + 1) * 16];
                                let q_base = q_offset + g * 32;
                                let mut group_dot = 0.0f32;
                                for i in 0..16 {
                                    let byte = packed[i];
                                    let lo = ((byte & 0x0F) as i32 - 8) as f32;
                                    let hi = ((byte >> 4) as i32 - 8) as f32;
                                    group_dot += q[q_base + i] * lo;
                                    group_dot += q[q_base + i + 16] * hi;
                                }
                                dot += group_dot * gs;
                            }
                            max_score = max_score.max(dot * scale);
                        }
                        scores[t] += max_score.max(0.0);
                    }
                }
            }
        }
    }
}

/// Select which positions to keep based on scores.
/// Returns sorted (ascending) list of kept position indices.
fn select_kept(scores: &[f32], budget: usize, cached_len: usize) -> Vec<u32> {
    let sink_count = H2O_SINK_COUNT.min(cached_len);
    let recent_start = cached_len.saturating_sub(H2O_RECENT_WINDOW);
    let evictable_start = sink_count;
    let evictable_end = recent_start.max(sink_count);

    let mut kept: Vec<u32> = Vec::with_capacity(budget);

    // Always keep sinks
    for i in 0..sink_count {
        kept.push(i as u32);
    }

    // Score-based selection of middle region
    if evictable_start < evictable_end {
        let middle_budget = budget.saturating_sub(sink_count).saturating_sub(H2O_RECENT_WINDOW.min(cached_len - recent_start));
        let mut scored: Vec<(u32, f32)> = (evictable_start..evictable_end)
            .map(|i| (i as u32, scores[i]))
            .collect();
        // Sort by score descending, keep top-N
        scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(middle_budget);
        // Re-sort ascending by position (critical for in-place compaction safety)
        scored.sort_unstable_by_key(|&(pos, _)| pos);
        for (pos, _) in scored {
            kept.push(pos);
        }
    }

    // Always keep recent window
    for i in evictable_end.max(sink_count)..cached_len {
        kept.push(i as u32);
    }

    kept
}

/// Compact KV cache in-place: gather kept positions into contiguous layout.
///
/// SAFETY: `kept_positions` must be sorted ascending. This guarantees
/// new_pos <= old_pos for all entries, making in-place copy safe.
fn compact(kv_cache: &mut CpuKvCache, kept_positions: &[u32], head_dim: usize) {
    let num_kept = kept_positions.len();

    for layer_data in &mut kv_cache.kv.layer_data {
        match layer_data {
            KvLayerData::F32 { keys_per_head, values_per_head } => {
                for head_vecs in [keys_per_head.as_mut_slice(), values_per_head.as_mut_slice()] {
                    for hv in head_vecs.iter_mut() {
                        for (new_pos, &old_pos) in kept_positions.iter().enumerate() {
                            let old_pos = old_pos as usize;
                            if old_pos == new_pos { continue; }
                            let src = old_pos * head_dim;
                            let dst = new_pos * head_dim;
                            hv.copy_within(src..src + head_dim, dst);
                        }
                        hv.truncate(num_kept * head_dim);
                    }
                }
            }
            KvLayerData::BF16 { keys_per_head, values_per_head } => {
                for head_vecs in [keys_per_head.as_mut_slice(), values_per_head.as_mut_slice()] {
                    for hv in head_vecs.iter_mut() {
                        for (new_pos, &old_pos) in kept_positions.iter().enumerate() {
                            let old_pos = old_pos as usize;
                            if old_pos == new_pos { continue; }
                            let src = old_pos * head_dim;
                            let dst = new_pos * head_dim;
                            hv.copy_within(src..src + head_dim, dst);
                        }
                        hv.truncate(num_kept * head_dim);
                    }
                }
            }
            KvLayerData::INT8 { keys_per_head, values_per_head, keys_scales, values_scales } => {
                for head_vecs in [keys_per_head.as_mut_slice(), values_per_head.as_mut_slice()] {
                    for hv in head_vecs.iter_mut() {
                        for (new_pos, &old_pos) in kept_positions.iter().enumerate() {
                            let old_pos = old_pos as usize;
                            if old_pos == new_pos { continue; }
                            let src = old_pos * head_dim;
                            let dst = new_pos * head_dim;
                            hv.copy_within(src..src + head_dim, dst);
                        }
                        hv.truncate(num_kept * head_dim);
                    }
                }
                for scale_vecs in [keys_scales.as_mut_slice(), values_scales.as_mut_slice()] {
                    for sv in scale_vecs.iter_mut() {
                        for (new_pos, &old_pos) in kept_positions.iter().enumerate() {
                            let old_pos = old_pos as usize;
                            if old_pos == new_pos { continue; }
                            sv[new_pos] = sv[old_pos];
                        }
                        sv.truncate(num_kept);
                    }
                }
            }
            KvLayerData::INT4 { keys_per_head, values_per_head, keys_scales, values_scales } => {
                let packed_per_pos = head_dim / 2;
                let num_groups = head_dim / 32;
                for head_vecs in [keys_per_head.as_mut_slice(), values_per_head.as_mut_slice()] {
                    for hv in head_vecs.iter_mut() {
                        for (new_pos, &old_pos) in kept_positions.iter().enumerate() {
                            let old_pos = old_pos as usize;
                            if old_pos == new_pos { continue; }
                            let src = old_pos * packed_per_pos;
                            let dst = new_pos * packed_per_pos;
                            hv.copy_within(src..src + packed_per_pos, dst);
                        }
                        hv.truncate(num_kept * packed_per_pos);
                    }
                }
                for scale_vecs in [keys_scales.as_mut_slice(), values_scales.as_mut_slice()] {
                    for sv in scale_vecs.iter_mut() {
                        for (new_pos, &old_pos) in kept_positions.iter().enumerate() {
                            let old_pos = old_pos as usize;
                            if old_pos == new_pos { continue; }
                            let src = old_pos * num_groups;
                            let dst = new_pos * num_groups;
                            sv.copy_within(src..src + num_groups, dst);
                        }
                        sv.truncate(num_kept * num_groups);
                    }
                }
            }
        }
    }

    // Also compact flat BF16 arrays (prefix cache serialization)
    if kv_cache.kv.kv_quant == herbert_core::config::KvQuantType::BF16 {
        let kv_dim = kv_cache.kv.kv_dim;
        for flat in [&mut kv_cache.kv.keys, &mut kv_cache.kv.values] {
            for layer_vec in flat.iter_mut() {
                for (new_pos, &old_pos) in kept_positions.iter().enumerate() {
                    let old_pos = old_pos as usize;
                    if old_pos == new_pos { continue; }
                    let src = old_pos * kv_dim;
                    let dst = new_pos * kv_dim;
                    layer_vec.copy_within(src..src + kv_dim, dst);
                }
                layer_vec.truncate(num_kept * kv_dim);
            }
        }
    }
}

/// Run H2O eviction on a CPU KV cache.
///
/// Called after each decode step when `seq_len > budget + hysteresis`.
/// Returns the number of evicted positions.
pub fn h2o_evict(
    kv_cache: &mut CpuKvCache,
    config: &Config,
    budget: usize,
) -> usize {
    let cached_len = kv_cache.kv.seq_len;
    if cached_len <= budget {
        return 0;
    }

    // 1. Score probe
    let mut scores = vec![0.0f32; cached_len];
    score_probe(kv_cache, config, &mut scores);

    // 2. Select kept positions
    let kept = select_kept(&scores, budget, cached_len);
    let num_kept = kept.len();
    let num_evicted = cached_len - num_kept;

    // 3. Compact KV cache in-place
    compact(kv_cache, &kept, config.head_dim);

    // 4. Update seq_len (rope_pos stays unchanged — positions are absolute)
    kv_cache.kv.seq_len = num_kept;

    num_evicted
}
