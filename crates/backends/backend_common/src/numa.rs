//! NUMA first-touch: force page placement near pinned worker threads.
//!
//! On multi-CCD/multi-socket systems, all weight buffers are initially allocated by the main
//! thread, so all pages land on the main thread's NUMA node. Workers pinned to other
//! CCDs/sockets then pay a remote DRAM penalty (20-40% on dual-socket EPYC).
//!
//! This module provides `first_touch_weights`, which runs after model load and before the
//! first inference. Each worker reads (touches) its proportional share of every weight
//! buffer, triggering the kernel's first-touch NUMA page migration policy.
//!
//! On non-Linux platforms this is a no-op (macOS has no NUMA topology).

use crate::generic_model::GenericModel;
use crate::linear_ops::LinearOps;

/// First-touch all weight pages from the thread pool's pinned workers.
///
/// Each worker reads one byte per OS page (4 KiB) in its assigned chunk of every weight
/// buffer. This triggers page faults that place pages on the worker's local NUMA node.
///
/// No-op on non-Linux or when the model has no layers.
pub fn first_touch_weights<L: LinearOps>(model: &GenericModel<L>) {
    // Only meaningful on Linux where NUMA first-touch policy applies.
    if !cfg!(target_os = "linux") {
        return;
    }

    let t0 = std::time::Instant::now();

    // Collect all raw memory ranges from the model.
    let mut ranges: Vec<(*const u8, usize)> = Vec::new();

    // embed_tokens (BF16 = u16)
    let embed_bytes = model.embed_tokens.len() * std::mem::size_of::<u16>();
    if embed_bytes > 0 {
        ranges.push((model.embed_tokens.as_ptr() as *const u8, embed_bytes));
    }

    // lm_head
    for r in L::weight_byte_ranges(&model.lm_head) {
        ranges.push(r);
    }

    // Per-layer weights
    for layer in &model.decoder_layers {
        // Attention projections (via LayerBlock)
        match &layer.block {
            crate::generic_layer::LayerBlock::Attention(attn) => {
                for w in [&attn.q_proj, &attn.k_proj, &attn.v_proj, &attn.o_proj] {
                    for r in L::weight_byte_ranges(w) {
                        ranges.push(r);
                    }
                }
            }
            crate::generic_layer::LayerBlock::Convolution(_) => {}
            crate::generic_layer::LayerBlock::GatedDeltaNet(_) => {}
        }

        // FFN (dense MLP or MoE experts)
        match &layer.ffn {
            crate::generic_moe::GenericFFN::Dense(mlp) => {
                for w in [&mlp.gate_proj, &mlp.up_proj, &mlp.down_proj] {
                    for r in L::weight_byte_ranges(w) {
                        ranges.push(r);
                    }
                }
            }
            crate::generic_moe::GenericFFN::MoE(moe) => {
                // Router gate
                for r in L::weight_byte_ranges(&moe.gate) {
                    ranges.push(r);
                }
                // All experts
                if let Ok(pool) = moe.expert_pool.lock() {
                    for expert in pool.experts() {
                        for w in [&expert.gate_proj, &expert.up_proj, &expert.down_proj] {
                            for r in L::weight_byte_ranges(w) {
                                ranges.push(r);
                            }
                        }
                    }
                }
                // Shared expert
                if let Some(shared) = &moe.shared_expert {
                    for w in [&shared.gate_proj, &shared.up_proj, &shared.down_proj] {
                        for r in L::weight_byte_ranges(w) {
                            ranges.push(r);
                        }
                    }
                }
            }
        }
    }

    if ranges.is_empty() {
        return;
    }

    // Compute total bytes
    let total_bytes: usize = ranges.iter().map(|&(_, len)| len).sum();

    // Touch pages from worker threads
    let pool = crate::thread_pool::global_pool();
    let num_workers = pool.num_workers();

    // SAFETY: We only read one byte per page via volatile read. The model is immutable
    // during this phase (called before first inference). The pointers are valid because
    // they come from live Vec allocations on the model.
    for &(ptr, len) in &ranges {
        if len == 0 {
            continue;
        }
        let chunk_size = (len + num_workers - 1) / num_workers;
        let send_ptr = crate::thread_pool::SendPtr::new(ptr);

        let _ = pool.parallel_for(num_workers, move |worker_id, _start, _end| {
            let base = send_ptr.ptr();
            let my_start = (worker_id * chunk_size).min(len);
            let my_end = ((worker_id + 1) * chunk_size).min(len);
            let mut offset = my_start;
            while offset < my_end {
                unsafe {
                    let _ = std::ptr::read_volatile(base.add(offset));
                }
                offset += 4096; // page size
            }
        });
    }

    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let total_mb = total_bytes as f64 / (1024.0 * 1024.0);
    tracing::info!(
        total_mb = format!("{:.0}", total_mb),
        elapsed_ms = format!("{:.1}", elapsed_ms),
        ranges = ranges.len(),
        workers = num_workers,
        "NUMA first-touch complete"
    );
}
