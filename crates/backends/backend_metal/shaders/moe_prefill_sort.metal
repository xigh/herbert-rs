#include <metal_stdlib>
using namespace metal;

struct MoePrefillSortParams {
    uint total;        // M * top_k
    uint num_experts;  // ne
    uint top_k;
    uint N_gate;       // moe_intermediate_size (for indirect grid)
    uint N_down;       // hidden_size (for indirect grid)
    uint tile_n_gate;  // output tile width for gate_up kernel
    uint tile_m_gate;  // token tile height for gate_up kernel
    uint tile_n_down;  // output tile width for down kernel
    uint tile_m_down;  // token tile height for down kernel
};

// GPU counting sort: group M*top_k assignments by expert_id.
//
// Single threadgroup of 256 threads. Four phases:
// 1. Zero histogram
// 2. Count expert occurrences (atomic)
// 3. Thread 0: exclusive prefix sum -> offsets, copy counts, compute max_count,
//    write indirect dispatch grids for tiled gate_up and down kernels
// 4. Re-zero hist as cursors, scatter sorted_src_idx
//
// indirect_args layout (6 x uint32, 24 bytes):
//   [0..2] gate_up grid: (ceil(N_gate/tile_n_gate), ceil(max_count/tile_m_gate), num_experts)
//   [3..5] down grid:    (ceil(N_down/tile_n_down), ceil(max_count/tile_m_down), num_experts)
//
// Dispatch: (1, 1, 1) threadgroups, (256, 1, 1) threads
[[kernel]]
void moe_prefill_sort(
    device const uint*     expert_ids      [[buffer(0)]],
    device uint*           expert_counts   [[buffer(1)]],
    device uint*           expert_offsets  [[buffer(2)]],
    device uint*           sorted_src_idx  [[buffer(3)]],
    device uint*           indirect_args   [[buffer(4)]],
    constant MoePrefillSortParams& p      [[buffer(5)]],
    uint                   tid             [[thread_index_in_threadgroup]]
) {
    uint total = p.total;
    uint ne = p.num_experts;

    // Phase 1: Zero histogram (max 256 experts, one per thread)
    threadgroup atomic_uint hist[256];
    if (tid < ne) {
        atomic_store_explicit(&hist[tid], 0, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: Count occurrences
    for (uint i = tid; i < total; i += 256) {
        uint eid = expert_ids[i];
        atomic_fetch_add_explicit(&hist[eid], 1, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: Thread 0 does exclusive prefix sum, computes max_count,
    //          writes counts/offsets and indirect dispatch grids
    if (tid == 0) {
        uint running = 0;
        uint max_c = 0;
        for (uint e = 0; e < ne; e++) {
            uint c = atomic_load_explicit(&hist[e], memory_order_relaxed);
            expert_counts[e] = c;
            expert_offsets[e] = running;
            running += c;
            if (c > max_c) max_c = c;
        }
        // Write indirect dispatch args for gate_up kernel
        indirect_args[0] = (p.N_gate + p.tile_n_gate - 1) / p.tile_n_gate;
        indirect_args[1] = (max_c + p.tile_m_gate - 1) / p.tile_m_gate;
        indirect_args[2] = ne;
        // Write indirect dispatch args for down kernel
        indirect_args[3] = (p.N_down + p.tile_n_down - 1) / p.tile_n_down;
        indirect_args[4] = (max_c + p.tile_m_down - 1) / p.tile_m_down;
        indirect_args[5] = ne;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 4: Re-zero hist as per-expert cursors, then scatter
    if (tid < ne) {
        atomic_store_explicit(&hist[tid], 0, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < total; i += 256) {
        uint eid = expert_ids[i];
        uint slot = atomic_fetch_add_explicit(&hist[eid], 1, memory_order_relaxed);
        sorted_src_idx[expert_offsets[eid] + slot] = i;
    }
}
