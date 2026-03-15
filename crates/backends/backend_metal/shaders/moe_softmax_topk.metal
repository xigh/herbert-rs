#include <metal_stdlib>
using namespace metal;

struct MoeSoftmaxTopkParams {
    uint num_experts;
    uint top_k;
    uint norm_topk_prob;
};

// GPU-side MoE routing: softmax over expert logits + top-k selection.
//
// Replaces CPU softmax+top-k to avoid GPU→CPU sync overhead.
// Single threadgroup of 32 threads handles up to 256 experts.
//
// Algorithm:
//   1. Cooperative load logits into threadgroup memory
//   2. Stable softmax: max → subtract → exp → sum → normalize
//   3. Repeated argmax for top-k: find max, record, mask to -inf, repeat
//   4. Optional weight renormalization (norm_topk_prob)
//
// Dispatch: (1, 1, 1) threadgroups of 32 threads.
//
// Buffers:
//   router_logits  : [num_experts]  - f32 input logits
//   expert_ids     : [top_k]        - u32 output expert indices
//   expert_weights : [top_k]        - f32 output expert weights
[[kernel]]
void moe_softmax_topk(
    device const float*              router_logits  [[buffer(0)]],
    device uint*                     expert_ids     [[buffer(1)]],
    device float*                    expert_weights [[buffer(2)]],
    constant MoeSoftmaxTopkParams&   p              [[buffer(3)]],
    uint                             lane           [[thread_position_in_threadgroup]]
) {
    uint ne = p.num_experts;

    // Threadgroup memory for probabilities
    threadgroup float probs[256];

    // Step 1: Load logits into shared memory and find max (for numerical stability)
    float local_max = -INFINITY;
    for (uint i = lane; i < ne; i += 32) {
        float v = router_logits[i];
        probs[i] = v;
        local_max = max(local_max, v);
    }
    float global_max = simd_max(local_max);

    // Step 2: Compute exp(x - max) and partial sum
    float local_sum = 0.0f;
    for (uint i = lane; i < ne; i += 32) {
        float e = exp(probs[i] - global_max);
        probs[i] = e;
        local_sum += e;
    }
    float total_sum = simd_sum(local_sum);

    // Step 3: Normalize to probabilities
    float inv_sum = (total_sum > 0.0f) ? (1.0f / total_sum) : 0.0f;
    for (uint i = lane; i < ne; i += 32) {
        probs[i] *= inv_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 4: Repeated argmax for top-k selection
    // Uses a simple cooperative approach: each lane finds its local max,
    // then we use simd_max to get global max, and a single-lane scan to
    // identify the winning index.
    for (uint sel = 0; sel < p.top_k; sel++) {
        // Each lane finds its local max
        float lane_max_val = -INFINITY;
        uint lane_max_idx = 0;
        for (uint i = lane; i < ne; i += 32) {
            if (probs[i] > lane_max_val) {
                lane_max_val = probs[i];
                lane_max_idx = i;
            }
        }

        // Reduce to find global max value across all lanes
        float best_val = simd_max(lane_max_val);

        // Find which lane has the winner using shuffle
        // Each lane checks if it has the max; broadcast the index from the winner
        bool is_winner = (lane_max_val == best_val);
        // Use simd_min to find the first winning lane
        uint winner_lane = is_winner ? lane : 32u;
        winner_lane = simd_min(winner_lane);

        // Broadcast the winner's index
        uint best_idx = simd_broadcast(lane_max_idx, winner_lane);

        // Lane 0 writes the result
        if (lane == 0) {
            expert_ids[sel] = best_idx;
            expert_weights[sel] = best_val;
            // Mask out this expert for next iteration
            probs[best_idx] = -INFINITY;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Step 5: Optional weight renormalization
    if (p.norm_topk_prob != 0 && lane == 0) {
        float weight_sum = 0.0f;
        for (uint i = 0; i < p.top_k; i++) {
            weight_sum += expert_weights[i];
        }
        if (weight_sum > 0.0f) {
            float inv = 1.0f / weight_sum;
            for (uint i = 0; i < p.top_k; i++) {
                expert_weights[i] *= inv;
            }
        }
    }
}
