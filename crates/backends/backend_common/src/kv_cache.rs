//! Shared KV cache for CPU inference backends.
//!
//! Split into two sub-structs:
//! - `KvStore`: persistent KV data (grows during prefill/decode)
//! - `InferenceContext`: scratch buffers (one per active request)

use herbert_core::config::{Config, KvQuantType};
use herbert_core::tensor::f32_to_bf16;

use crate::hugepages::{hint_hugepages, HugeVec};
use crate::thread_pool::global_pool;

// ============================================================================
// Quantization utilities (always compiled — needed by runtime dispatch)
// ============================================================================

#[inline]
pub fn compute_int8_scale(head: &[f32]) -> f32 {
    let max_abs = head.iter().copied().fold(0.0f32, |a, x| a.max(x.abs()));
    if max_abs == 0.0 {
        1.0
    } else {
        max_abs / 127.0
    }
}

#[inline]
pub fn quantize_i8(x: f32, scale: f32) -> i8 {
    (x / scale).round().clamp(-127.0, 127.0) as i8
}

// ============================================================================
// INT4 quantization utilities (Q4_0 group quantization, group_size=32)
// ============================================================================

/// Group size for INT4 quantization (number of elements per scale).
pub const INT4_GROUP_SIZE: usize = 32;

/// Compute scale for a group of elements: symmetric, scale = max_abs / 7.0.
#[inline]
pub fn compute_int4_scale(group: &[f32]) -> f32 {
    let max_abs = group.iter().copied().fold(0.0f32, |a, x| a.max(x.abs()));
    if max_abs == 0.0 {
        1.0
    } else {
        max_abs / 7.0
    }
}

/// Number of groups for a given head_dim.
#[inline]
pub fn int4_num_groups(head_dim: usize) -> usize {
    head_dim / INT4_GROUP_SIZE
}

/// Quantize a single f32 value to 4-bit signed [-8, +7], returned as u8 [0, 15].
#[inline]
pub fn quantize_i4(x: f32, scale: f32) -> u8 {
    let q = (x / scale).round().clamp(-8.0, 7.0) as i8;
    (q + 8) as u8 // unsigned [0, 15]
}

/// Pack a group of 32 f32 elements into 16 bytes using half-split layout.
/// byte[i] = elem[i] | (elem[i+16] << 4) for i in 0..16.
#[inline]
pub fn pack_i4_group(src: &[f32], scale: f32, dst: &mut [u8]) {
    debug_assert!(src.len() >= 32);
    debug_assert!(dst.len() >= 16);
    for i in 0..16 {
        let lo = quantize_i4(src[i], scale);
        let hi = quantize_i4(src[i + 16], scale);
        dst[i] = lo | (hi << 4);
    }
}

/// Unpack a group of 16 packed bytes into 32 f32 values (half-split layout).
#[inline]
pub fn unpack_i4_group(packed: &[u8], scale: f32, dst: &mut [f32]) {
    debug_assert!(packed.len() >= 16);
    debug_assert!(dst.len() >= 32);
    for i in 0..16 {
        let byte = packed[i];
        let lo = (byte & 0x0F) as i32 - 8;
        let hi = (byte >> 4) as i32 - 8;
        dst[i] = lo as f32 * scale;
        dst[i + 16] = hi as f32 * scale;
    }
}

// ============================================================================
// KvLayerData: runtime-dispatched per-layer KV storage
// ============================================================================

/// Per-layer KV data in a format selected at runtime.
///
/// KV vectors use `HugeVec<T>` for MAP_HUGETLB backing when the `hugepages`
/// feature is enabled. This eliminates TLB page-walk overhead for attention
/// dot products which access KV cache semi-randomly across sequence positions.
#[derive(Debug, Clone)]
pub enum KvLayerData {
    /// Full precision f32 KV cache.
    F32 {
        keys_per_head: Vec<HugeVec<f32>>,
        values_per_head: Vec<HugeVec<f32>>,
    },
    /// BF16 KV cache (stored as u16).
    BF16 {
        keys_per_head: Vec<HugeVec<u16>>,
        values_per_head: Vec<HugeVec<u16>>,
    },
    /// INT8 symmetric quantized KV cache with per-position per-head scales.
    INT8 {
        keys_per_head: Vec<HugeVec<i8>>,
        values_per_head: Vec<HugeVec<i8>>,
        keys_scales: Vec<Vec<f32>>,
        values_scales: Vec<Vec<f32>>,
    },
    /// INT4 symmetric quantized KV cache with Q4_0 group quantization (group_size=32).
    /// Half-split nibble packing within each group. Per-group scales (head_dim/32 per position).
    /// For head_dim=128: 64 packed bytes + 4×f32 scales = 80 bytes per position per head.
    /// Scales layout: [pos0_group0, pos0_group1, ..., pos0_groupN, pos1_group0, ...].
    INT4 {
        keys_per_head: Vec<HugeVec<u8>>,
        values_per_head: Vec<HugeVec<u8>>,
        /// Per-group scales: num_groups f32 per position, flattened.
        keys_scales: Vec<Vec<f32>>,
        values_scales: Vec<Vec<f32>>,
    },
}

impl KvLayerData {
    /// Number of cached positions in this layer.
    pub fn cached_len(&self, head_dim: usize) -> usize {
        match self {
            Self::F32 { keys_per_head, .. } => {
                keys_per_head.first().map_or(0, |v| v.len() / head_dim)
            }
            Self::BF16 { keys_per_head, .. } => {
                keys_per_head.first().map_or(0, |v| v.len() / head_dim)
            }
            Self::INT8 { keys_per_head, .. } => {
                keys_per_head.first().map_or(0, |v| v.len() / head_dim)
            }
            Self::INT4 { keys_scales, .. } => {
                // Per-group scales: num_groups f32 per position
                let num_groups = int4_num_groups(head_dim);
                keys_scales.first().map_or(0, |v| {
                    if num_groups == 0 { 0 } else { v.len() / num_groups }
                })
            }
        }
    }

    /// Truncate all heads to `trim_to` positions.
    pub fn truncate_to(&mut self, trim_to: usize, head_dim: usize) {
        let elems = trim_to * head_dim;
        match self {
            Self::F32 { keys_per_head, values_per_head } => {
                for v in keys_per_head.iter_mut().chain(values_per_head.iter_mut()) {
                    v.truncate(elems);
                }
            }
            Self::BF16 { keys_per_head, values_per_head } => {
                for v in keys_per_head.iter_mut().chain(values_per_head.iter_mut()) {
                    v.truncate(elems);
                }
            }
            Self::INT8 { keys_per_head, values_per_head, keys_scales, values_scales } => {
                for v in keys_per_head.iter_mut().chain(values_per_head.iter_mut()) {
                    v.truncate(elems);
                }
                for v in keys_scales.iter_mut().chain(values_scales.iter_mut()) {
                    v.truncate(trim_to);
                }
            }
            Self::INT4 { keys_per_head, values_per_head, keys_scales, values_scales } => {
                let packed_elems = trim_to * (head_dim / 2);
                for v in keys_per_head.iter_mut().chain(values_per_head.iter_mut()) {
                    v.truncate(packed_elems);
                }
                let num_groups = int4_num_groups(head_dim);
                for v in keys_scales.iter_mut().chain(values_scales.iter_mut()) {
                    v.truncate(trim_to * num_groups);
                }
            }
        }
    }
}

// ============================================================================
// KvStoreSnapshot: deep-copyable snapshot of KV data for prefix caching
// ============================================================================

/// A deep-copyable snapshot of all KV data in a `KvStore`.
///
/// Used by `SharedPrefixCache` to store in-memory snapshots of KV state
/// at known prefix boundaries. Can be cloned into a fresh `KvStore` to
/// resume inference from the snapshotted position.
#[derive(Clone)]
pub struct KvStoreSnapshot {
    pub kv_quant: KvQuantType,
    pub layer_data: Vec<KvLayerData>,
    /// Flat BF16 keys (only populated when kv_quant == BF16).
    pub keys: Vec<Vec<u16>>,
    /// Flat BF16 values (only populated when kv_quant == BF16).
    pub values: Vec<Vec<u16>>,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub seq_len: usize,
    pub kv_dim: usize,
}

impl KvStoreSnapshot {
    /// Estimated memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        let mut total = 0usize;
        for ld in &self.layer_data {
            match ld {
                KvLayerData::F32 { keys_per_head, values_per_head } => {
                    for v in keys_per_head.iter().chain(values_per_head.iter()) {
                        total += v.len() * std::mem::size_of::<f32>();
                    }
                }
                KvLayerData::BF16 { keys_per_head, values_per_head } => {
                    for v in keys_per_head.iter().chain(values_per_head.iter()) {
                        total += v.len() * std::mem::size_of::<u16>();
                    }
                }
                KvLayerData::INT8 { keys_per_head, values_per_head, keys_scales, values_scales } => {
                    for v in keys_per_head.iter().chain(values_per_head.iter()) {
                        total += v.len() * std::mem::size_of::<i8>();
                    }
                    for v in keys_scales.iter().chain(values_scales.iter()) {
                        total += v.len() * std::mem::size_of::<f32>();
                    }
                }
                KvLayerData::INT4 { keys_per_head, values_per_head, keys_scales, values_scales } => {
                    for v in keys_per_head.iter().chain(values_per_head.iter()) {
                        total += v.len(); // packed u8 bytes
                    }
                    for v in keys_scales.iter().chain(values_scales.iter()) {
                        total += v.len() * std::mem::size_of::<f32>();
                    }
                }
            }
        }
        for v in self.keys.iter().chain(self.values.iter()) {
            total += v.len() * std::mem::size_of::<u16>();
        }
        total
    }
}

// ============================================================================
// KvStore: persistent KV data (append-only during prefill/decode)
// ============================================================================

/// Persistent KV cache data that grows during inference.
#[derive(Debug)]
pub struct KvStore {
    /// Runtime KV quantization type.
    pub kv_quant: KvQuantType,
    /// Per-layer KV data in the runtime-selected format.
    pub layer_data: Vec<KvLayerData>,
    /// Flat BF16 keys/values for prefix cache serialization (only populated when kv_quant == BF16).
    pub keys: Vec<Vec<u16>>,
    pub values: Vec<Vec<u16>>,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub seq_len: usize,
    pub kv_dim: usize,
    /// Current text position for VL decode (incremented after each text token).
    pub text_pos: u32,
    /// Per-layer convolution state: [hidden_size * conv_l_cache] per layer.
    /// Empty vecs for pure-attention models (zero cost).
    pub conv_cache: Vec<Vec<f32>>,
    /// Per-layer DeltaNet recurrent state: [num_kv_heads * key_dim * value_dim] per layer.
    /// Empty vecs for non-DeltaNet layers.
    pub deltanet_recurrent: Vec<Vec<f32>>,
    /// Per-layer DeltaNet conv1d state: [conv_dim * conv_kernel_dim] per layer.
    /// Empty vecs for non-DeltaNet layers.
    pub deltanet_conv: Vec<Vec<f32>>,
    /// When true, prefill attention is bidirectional (no causal mask).
    /// Used for encoder models like ColBERT.
    pub bidirectional: bool,
}

impl KvStore {
    pub fn append(&mut self, layer_idx: usize, k: &[f32], v: &[f32]) {
        #[cfg(feature = "profile-kv")]
        let t0 = std::time::Instant::now();

        // Write to flat BF16 arrays for prefix cache serialization (BF16 mode only)
        if self.kv_quant == KvQuantType::BF16 {
            self.keys[layer_idx].extend(k.iter().map(|&x| f32_to_bf16(x)));
            self.values[layer_idx].extend(v.iter().map(|&x| f32_to_bf16(x)));
        }

        // Write to layer_data (primary storage)
        self.append_layer_data(layer_idx, k, v);

        #[cfg(feature = "profile-kv")]
        crate::profiler::push_kv_append(t0.elapsed().as_micros() as u64);
    }

    /// Append to layer_data based on runtime kv_quant.
    fn append_layer_data(&mut self, layer_idx: usize, k: &[f32], v: &[f32]) {
        let kv_dim = self.num_kv_heads * self.head_dim;
        let num_positions = k.len() / kv_dim;

        match &mut self.layer_data[layer_idx] {
            KvLayerData::F32 { keys_per_head, values_per_head } => {
                for pos in 0..num_positions {
                    for kv_h in 0..self.num_kv_heads {
                        let src = pos * kv_dim + kv_h * self.head_dim;
                        keys_per_head[kv_h].extend_from_slice(&k[src..src + self.head_dim]);
                        values_per_head[kv_h].extend_from_slice(&v[src..src + self.head_dim]);
                    }
                }
            }
            KvLayerData::BF16 { keys_per_head, values_per_head } => {
                for pos in 0..num_positions {
                    for kv_h in 0..self.num_kv_heads {
                        let src = pos * kv_dim + kv_h * self.head_dim;
                        for d in 0..self.head_dim {
                            keys_per_head[kv_h].push(herbert_core::tensor::f32_to_bf16(k[src + d]));
                            values_per_head[kv_h].push(herbert_core::tensor::f32_to_bf16(v[src + d]));
                        }
                    }
                }
            }
            KvLayerData::INT8 { keys_per_head, values_per_head, keys_scales, values_scales } => {
                for pos in 0..num_positions {
                    for kv_h in 0..self.num_kv_heads {
                        let src = pos * kv_dim + kv_h * self.head_dim;
                        let k_head = &k[src..src + self.head_dim];
                        let v_head = &v[src..src + self.head_dim];

                        let k_scale = compute_int8_scale(k_head);
                        keys_scales[kv_h].push(k_scale);
                        for &kval in k_head {
                            keys_per_head[kv_h].push(quantize_i8(kval, k_scale));
                        }

                        let v_scale = compute_int8_scale(v_head);
                        values_scales[kv_h].push(v_scale);
                        for &vval in v_head {
                            values_per_head[kv_h].push(quantize_i8(vval, v_scale));
                        }
                    }
                }
            }
            KvLayerData::INT4 { keys_per_head, values_per_head, keys_scales, values_scales } => {
                let num_groups = int4_num_groups(self.head_dim);
                let packed_per_pos = self.head_dim / 2;
                let mut pack_buf = vec![0u8; packed_per_pos];
                for pos in 0..num_positions {
                    for kv_h in 0..self.num_kv_heads {
                        let src = pos * kv_dim + kv_h * self.head_dim;
                        let k_head = &k[src..src + self.head_dim];
                        let v_head = &v[src..src + self.head_dim];

                        // Per-group scales for keys
                        for g in 0..num_groups {
                            let group_start = g * INT4_GROUP_SIZE;
                            let k_group = &k_head[group_start..group_start + INT4_GROUP_SIZE];
                            let k_scale = compute_int4_scale(k_group);
                            keys_scales[kv_h].push(k_scale);
                            pack_i4_group(k_group, k_scale, &mut pack_buf[g * 16..]);
                        }
                        keys_per_head[kv_h].extend_from_slice(&pack_buf);

                        // Per-group scales for values
                        for g in 0..num_groups {
                            let group_start = g * INT4_GROUP_SIZE;
                            let v_group = &v_head[group_start..group_start + INT4_GROUP_SIZE];
                            let v_scale = compute_int4_scale(v_group);
                            values_scales[kv_h].push(v_scale);
                            pack_i4_group(v_group, v_scale, &mut pack_buf[g * 16..]);
                        }
                        values_per_head[kv_h].extend_from_slice(&pack_buf);
                    }
                }
            }
        }
    }

    /// Truncate both flat BF16 arrays and layer_data for a layer.
    pub fn truncate_to(&mut self, layer_idx: usize, trim_to: usize) {
        // Truncate flat BF16 arrays (prefix cache)
        if self.kv_quant == KvQuantType::BF16 {
            self.keys[layer_idx].truncate(trim_to * self.kv_dim);
            self.values[layer_idx].truncate(trim_to * self.kv_dim);
        }
        // Truncate layer_data
        self.layer_data[layer_idx].truncate_to(trim_to, self.head_dim);
    }

    /// Number of cached positions for a given layer (from layer_data).
    pub fn cached_len(&self, layer_idx: usize) -> usize {
        self.layer_data[layer_idx].cached_len(self.head_dim)
    }

    pub fn advance_seq_len(&mut self, count: usize) {
        self.seq_len += count;
    }

    /// Estimated memory usage in bytes for the KV data.
    pub fn memory_bytes(&self) -> usize {
        let mut total = 0usize;
        for ld in &self.layer_data {
            match ld {
                KvLayerData::F32 { keys_per_head, values_per_head } => {
                    for v in keys_per_head.iter().chain(values_per_head.iter()) {
                        total += v.len() * std::mem::size_of::<f32>();
                    }
                }
                KvLayerData::BF16 { keys_per_head, values_per_head } => {
                    for v in keys_per_head.iter().chain(values_per_head.iter()) {
                        total += v.len() * std::mem::size_of::<u16>();
                    }
                }
                KvLayerData::INT8 { keys_per_head, values_per_head, keys_scales, values_scales } => {
                    for v in keys_per_head.iter().chain(values_per_head.iter()) {
                        total += v.len() * std::mem::size_of::<i8>();
                    }
                    for v in keys_scales.iter().chain(values_scales.iter()) {
                        total += v.len() * std::mem::size_of::<f32>();
                    }
                }
                KvLayerData::INT4 { keys_per_head, values_per_head, keys_scales, values_scales } => {
                    for v in keys_per_head.iter().chain(values_per_head.iter()) {
                        total += v.len(); // packed u8 bytes
                    }
                    for v in keys_scales.iter().chain(values_scales.iter()) {
                        total += v.len() * std::mem::size_of::<f32>();
                    }
                }
            }
        }
        // Add flat BF16 arrays
        for v in self.keys.iter().chain(self.values.iter()) {
            total += v.len() * std::mem::size_of::<u16>();
        }
        total
    }

    /// Create a deep-copy snapshot of the current KV data.
    ///
    /// The snapshot captures `layer_data`, flat BF16 keys/values, and all
    /// dimension metadata. It does NOT capture `text_pos`, `conv_cache`,
    /// `deltanet_recurrent`, `deltanet_conv`, or `bidirectional` — those
    /// are either zero at prefix boundaries or model-structural.
    pub fn snapshot(&self) -> KvStoreSnapshot {
        KvStoreSnapshot {
            kv_quant: self.kv_quant,
            layer_data: self.layer_data.clone(),
            keys: self.keys.clone(),
            values: self.values.clone(),
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
            seq_len: self.seq_len,
            kv_dim: self.kv_dim,
        }
    }

    /// Create a new `KvStore` by deep-copying data from a snapshot.
    ///
    /// The returned `KvStore` has freshly allocated vectors with the snapshot's
    /// data copied in. Conv/DeltaNet state is initialized from `config` (zeroed),
    /// matching the behavior of a fresh `CpuKvCache::new()`.
    pub fn restore_from_snapshot(snapshot: &KvStoreSnapshot, config: &Config) -> KvStore {
        let num_layers = config.num_layers;
        let hidden_size = config.hidden_size;
        let conv_l_cache = config.conv_l_cache;

        KvStore {
            kv_quant: snapshot.kv_quant,
            layer_data: snapshot.layer_data.clone(),
            keys: snapshot.keys.clone(),
            values: snapshot.values.clone(),
            num_kv_heads: snapshot.num_kv_heads,
            head_dim: snapshot.head_dim,
            seq_len: snapshot.seq_len,
            kv_dim: snapshot.kv_dim,
            text_pos: 0,
            conv_cache: (0..num_layers)
                .map(|_| {
                    if conv_l_cache > 0 {
                        vec![0.0; hidden_size * conv_l_cache]
                    } else {
                        Vec::new()
                    }
                })
                .collect(),
            deltanet_recurrent: (0..num_layers)
                .map(|i| {
                    if config.is_deltanet_layer(i) {
                        vec![0.0; config.deltanet_recurrent_size()]
                    } else {
                        Vec::new()
                    }
                })
                .collect(),
            deltanet_conv: (0..num_layers)
                .map(|i| {
                    if config.is_deltanet_layer(i) {
                        let conv_dim = config.deltanet_conv_dim();
                        let ks = config.linear_conv_kernel_dim.unwrap_or(4);
                        vec![0.0; conv_dim * ks]
                    } else {
                        Vec::new()
                    }
                })
                .collect(),
            bidirectional: false,
        }
    }
}

// ============================================================================
// InferenceContext: scratch buffers (one per active request)
// ============================================================================

/// Temporary scratch buffers used during inference. One instance per active request.
#[derive(Debug)]
pub struct InferenceContext {
    // --- Decode per-layer scratch ---
    pub decode_q: Vec<Vec<f32>>,
    pub decode_k: Vec<Vec<f32>>,
    pub decode_v: Vec<Vec<f32>>,
    pub decode_attn_out: Vec<Vec<f32>>,
    pub decode_attn_proj_out: Vec<Vec<f32>>,
    pub decode_scores: Vec<Vec<f32>>,
    pub decode_scores_workers: Vec<Vec<Vec<f32>>>,
    pub decode_norm1: Vec<Vec<f32>>,
    pub decode_norm2: Vec<Vec<f32>>,
    pub decode_mlp_gate: Vec<Vec<f32>>,
    pub decode_mlp_up: Vec<Vec<f32>>,
    pub decode_mlp_out: Vec<Vec<f32>>,
    // --- Decode global scratch ---
    pub decode_embed: Vec<f32>,
    pub decode_final_norm: Vec<f32>,
    pub decode_logits: Vec<f32>,
    // --- Decode MoE scratch ---
    pub decode_moe_router: Vec<f32>,
    pub decode_moe_expert_out: Vec<f32>,
    // --- Q BF16 scratch (for VDPBF16PS optimization) ---
    pub decode_q_bf16: Vec<Vec<u16>>,
    pub prefill_q_bf16: Vec<u16>,
    // --- Prefill attention scratch ---
    pub prefill_q: Vec<f32>,
    pub prefill_k: Vec<f32>,
    pub prefill_v: Vec<f32>,
    pub prefill_attn_out: Vec<f32>,
    // --- Prefill layer scratch ---
    pub prefill_norm1: Vec<f32>,
    pub prefill_norm2: Vec<f32>,
    pub prefill_buf_a: Vec<f32>,
    pub prefill_buf_b: Vec<f32>,
    pub prefill_o_proj: Vec<f32>,
    pub prefill_ffn_out: Vec<f32>,
    pub prefill_mlp_gate: Vec<f32>,
    pub prefill_mlp_up: Vec<f32>,
    // --- MoE prefill scratch ---
    pub prefill_moe_logits: Vec<f32>,
    pub prefill_moe_expert_out: Vec<f32>,
    pub prefill_moe_batch_x: Vec<f32>,
    pub prefill_moe_flat_tok: Vec<usize>,
    pub prefill_moe_flat_wt: Vec<f32>,
    pub prefill_moe_active: Vec<[usize; 4]>,
    pub prefill_moe_flat_assigns: Vec<usize>,
    pub prefill_moe_worker_ranges: Vec<[usize; 2]>,
    pub prefill_moe_expert_assignments: Vec<Vec<(usize, f32)>>,
    // --- MRoPE scratch ---
    pub mrope_cos: Vec<f32>,
    pub mrope_sin: Vec<f32>,
    // --- DeltaNet prefill scratch ---
    pub prefill_deltanet_qkv: Vec<f32>,
    pub prefill_deltanet_z: Vec<f32>,
    pub prefill_deltanet_attn_out: Vec<f32>,
    // --- EAGLE-3 hidden state extraction ---
    pub eagle3_active: bool,
    pub eagle3_extract_layers: [usize; 3],
    pub eagle3_hidden_states: [Vec<f32>; 3],
}

// ============================================================================
// CpuKvCache: composite wrapper
// ============================================================================

/// KV cache with runtime-selected quantization format (F32 / BF16 / INT8).
/// Composite of persistent KV data and scratch buffers.
#[derive(Debug)]
pub struct CpuKvCache {
    /// Persistent KV data (grows during inference).
    pub kv: KvStore,
    /// Scratch buffers for inference (one per active request).
    pub ctx: InferenceContext,
}

impl CpuKvCache {
    pub fn new_with_quant(config: &Config, reserve_tokens: Option<usize>, kv_quant: KvQuantType) -> Self {
        let mut cache = Self::new(config, reserve_tokens);
        cache.kv.kv_quant = kv_quant;
        // Initialize layer_data with the proper type
        let num_layers = config.num_layers;
        let num_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;
        let reserve = reserve_tokens.unwrap_or(0);
        cache.kv.layer_data = (0..num_layers).map(|_| {
            match kv_quant {
                KvQuantType::F32 => KvLayerData::F32 {
                    keys_per_head: (0..num_kv_heads).map(|_| HugeVec::with_capacity(reserve * head_dim)).collect(),
                    values_per_head: (0..num_kv_heads).map(|_| HugeVec::with_capacity(reserve * head_dim)).collect(),
                },
                KvQuantType::BF16 => KvLayerData::BF16 {
                    keys_per_head: (0..num_kv_heads).map(|_| HugeVec::with_capacity(reserve * head_dim)).collect(),
                    values_per_head: (0..num_kv_heads).map(|_| HugeVec::with_capacity(reserve * head_dim)).collect(),
                },
                KvQuantType::INT8 => KvLayerData::INT8 {
                    keys_per_head: (0..num_kv_heads).map(|_| HugeVec::with_capacity(reserve * head_dim)).collect(),
                    values_per_head: (0..num_kv_heads).map(|_| HugeVec::with_capacity(reserve * head_dim)).collect(),
                    keys_scales: (0..num_kv_heads).map(|_| Vec::with_capacity(reserve)).collect(),
                    values_scales: (0..num_kv_heads).map(|_| Vec::with_capacity(reserve)).collect(),
                },
                KvQuantType::INT4 => {
                    let packed_per_pos = head_dim / 2;
                    KvLayerData::INT4 {
                        keys_per_head: (0..num_kv_heads).map(|_| HugeVec::with_capacity(reserve * packed_per_pos)).collect(),
                        values_per_head: (0..num_kv_heads).map(|_| HugeVec::with_capacity(reserve * packed_per_pos)).collect(),
                        keys_scales: (0..num_kv_heads).map(|_| Vec::with_capacity(reserve)).collect(),
                        values_scales: (0..num_kv_heads).map(|_| Vec::with_capacity(reserve)).collect(),
                    }
                },
            }
        }).collect();
        cache
    }

    pub fn new(config: &Config, reserve_tokens: Option<usize>) -> Self {
        let num_layers = config.num_layers;
        let hidden_size = config.hidden_size;
        let q_dim = config.q_dim();
        let kv_dim = config.kv_dim();
        let intermediate_size = if config.is_moe() {
            config.moe_intermediate_size.unwrap_or(config.intermediate_size)
        } else {
            config.intermediate_size
        };
        let vocab_size = config.vocab_size;
        let reserve_tokens = reserve_tokens.unwrap_or(0);
        let reserve_per_layer = reserve_tokens.saturating_mul(kv_dim);
        let scores_capacity = reserve_tokens.max(1);
        let pool_workers = global_pool().num_workers();
        let num_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;
        let conv_l_cache = config.conv_l_cache;

        // Flat BF16 keys/values for prefix cache serialization (always allocated for BF16 default)
        let keys: Vec<Vec<u16>> = (0..num_layers)
            .map(|_| {
                let v = Vec::with_capacity(reserve_per_layer);
                hint_hugepages(&v);
                v
            })
            .collect();
        let values: Vec<Vec<u16>> = (0..num_layers)
            .map(|_| {
                let v = Vec::with_capacity(reserve_per_layer);
                hint_hugepages(&v);
                v
            })
            .collect();

        // Default layer_data: BF16 per-head (hugepage-backed when feature enabled)
        let layer_data: Vec<KvLayerData> = (0..num_layers).map(|_| {
            KvLayerData::BF16 {
                keys_per_head: (0..num_kv_heads).map(|_| HugeVec::with_capacity(reserve_tokens * head_dim)).collect(),
                values_per_head: (0..num_kv_heads).map(|_| HugeVec::with_capacity(reserve_tokens * head_dim)).collect(),
            }
        }).collect();

        let kv = KvStore {
            kv_quant: KvQuantType::BF16,
            layer_data,
            keys,
            values,
            num_kv_heads,
            head_dim,
            seq_len: 0,
            kv_dim,
            text_pos: 0,
            conv_cache: (0..num_layers)
                .map(|_| {
                    if conv_l_cache > 0 {
                        vec![0.0; hidden_size * conv_l_cache]
                    } else {
                        Vec::new()
                    }
                })
                .collect(),
            deltanet_recurrent: (0..num_layers)
                .map(|i| {
                    if config.is_deltanet_layer(i) {
                        vec![0.0; config.deltanet_recurrent_size()]
                    } else {
                        Vec::new()
                    }
                })
                .collect(),
            deltanet_conv: (0..num_layers)
                .map(|i| {
                    if config.is_deltanet_layer(i) {
                        let conv_dim = config.deltanet_conv_dim();
                        let ks = config.linear_conv_kernel_dim.unwrap_or(4);
                        vec![0.0; conv_dim * ks]
                    } else {
                        Vec::new()
                    }
                })
                .collect(),
            bidirectional: false,
        };

        let ctx = InferenceContext {
            decode_q: (0..num_layers).map(|_| vec![0.0; q_dim]).collect(),
            decode_k: (0..num_layers).map(|_| vec![0.0; kv_dim]).collect(),
            decode_v: (0..num_layers).map(|_| vec![0.0; kv_dim]).collect(),
            decode_attn_out: (0..num_layers).map(|_| vec![0.0; q_dim]).collect(),
            decode_attn_proj_out: (0..num_layers).map(|_| vec![0.0; hidden_size]).collect(),
            decode_scores: (0..num_layers)
                .map(|_| Vec::with_capacity(scores_capacity))
                .collect(),
            decode_scores_workers: (0..num_layers)
                .map(|_| {
                    (0..pool_workers)
                        .map(|_| Vec::with_capacity(scores_capacity))
                        .collect()
                })
                .collect(),
            decode_norm1: (0..num_layers).map(|_| vec![0.0; hidden_size]).collect(),
            decode_norm2: (0..num_layers).map(|_| vec![0.0; hidden_size]).collect(),
            decode_mlp_gate: (0..num_layers)
                .map(|_| vec![0.0; intermediate_size])
                .collect(),
            decode_mlp_up: (0..num_layers)
                .map(|_| vec![0.0; intermediate_size])
                .collect(),
            decode_mlp_out: (0..num_layers).map(|_| vec![0.0; hidden_size]).collect(),
            decode_embed: vec![0.0; hidden_size],
            decode_final_norm: vec![0.0; hidden_size],
            decode_logits: vec![0.0; vocab_size],
            decode_q_bf16: (0..num_layers).map(|_| Vec::new()).collect(),
            prefill_q_bf16: Vec::new(),
            prefill_q: Vec::new(),
            prefill_k: Vec::new(),
            prefill_v: Vec::new(),
            prefill_attn_out: Vec::new(),
            prefill_norm1: Vec::new(),
            prefill_norm2: Vec::new(),
            prefill_buf_a: Vec::new(),
            prefill_buf_b: Vec::new(),
            prefill_o_proj: Vec::new(),
            prefill_ffn_out: Vec::new(),
            prefill_mlp_gate: Vec::new(),
            prefill_mlp_up: Vec::new(),
            prefill_moe_logits: Vec::new(),
            prefill_moe_expert_out: Vec::new(),
            prefill_moe_batch_x: Vec::new(),
            prefill_moe_flat_tok: Vec::new(),
            prefill_moe_flat_wt: Vec::new(),
            prefill_moe_active: Vec::new(),
            prefill_moe_flat_assigns: Vec::new(),
            prefill_moe_worker_ranges: Vec::new(),
            prefill_moe_expert_assignments: Vec::new(),
            decode_moe_router: Vec::new(),
            decode_moe_expert_out: Vec::new(),
            mrope_cos: Vec::new(),
            mrope_sin: Vec::new(),
            prefill_deltanet_qkv: Vec::new(),
            prefill_deltanet_z: Vec::new(),
            prefill_deltanet_attn_out: Vec::new(),
            eagle3_active: false,
            eagle3_extract_layers: [0; 3],
            eagle3_hidden_states: [Vec::new(), Vec::new(), Vec::new()],
        };

        Self { kv, ctx }
    }

    // === Delegation methods for backwards compatibility ===

    pub fn append(&mut self, layer_idx: usize, k: &[f32], v: &[f32]) {
        self.kv.append(layer_idx, k, v);
    }

    pub fn truncate_to(&mut self, layer_idx: usize, trim_to: usize) {
        self.kv.truncate_to(layer_idx, trim_to);
    }

    pub fn cached_len(&self, layer_idx: usize) -> usize {
        self.kv.cached_len(layer_idx)
    }

    pub fn advance_seq_len(&mut self, count: usize) {
        self.kv.advance_seq_len(count);
    }
}
