// AVX-512 BF16 batched dot product: 8 Q_bf16 · K_bf16 → 8 f32 scores
// Both Q and K are already in BF16 format — no per-head conversion needed.
// Uses VDPBF16PS to process 32 BF16 elements per instruction.
//
// Compared to avx512_attn_dot8_qf32_kbf16_dpbf16 (Q F32 → BF16 inline):
//   Old: 3 insn/head (2× F32 load + vcvtne2ps2bf16 + vdpbf16ps)
//   New: 2 insn/head (1× BF16 load + vdpbf16ps)
//   = 2.1× fewer instructions per element per head

.intel_syntax noprefix
#include "avx512_common.inc"
#include "avx512_hsum8.inc"

// avx512_attn_dot8_qbf16_kbf16(q_ptrs: **u16, k: *u16, dim: u64, scores: *mut f32)
    .globl SYMBOL(avx512_attn_dot8_qbf16_kbf16)
    .p2align 4
SYMBOL(avx512_attn_dot8_qbf16_kbf16):
    push        rbx
    push        r12
    push        r13
    push        r14
    push        r15

    mov         rbx, rcx            // scores output pointer

    // Load 8 Q pointers (BF16)
    mov         r8,  [rdi]
    mov         r9,  [rdi + 8]
    mov         r10, [rdi + 16]
    mov         r11, [rdi + 24]
    mov         r12, [rdi + 32]
    mov         r13, [rdi + 40]
    mov         r14, [rdi + 48]
    mov         r15, [rdi + 56]

    // Zero 8 accumulators
    vxorps      zmm4, zmm4, zmm4
    vxorps      zmm5, zmm5, zmm5
    vxorps      zmm6, zmm6, zmm6
    vxorps      zmm7, zmm7, zmm7
    vxorps      zmm8, zmm8, zmm8
    vxorps      zmm9, zmm9, zmm9
    vxorps      zmm10, zmm10, zmm10
    vxorps      zmm11, zmm11, zmm11

    xor         rcx, rcx

.Lqbf16_kbf16_dot8_loop:
    // Load 32× K BF16 (shared across all 8 heads)
    vmovups     zmm0, [rsi + rcx*2]

    // Head 0: load 32× Q BF16, dot product → F32 accumulator
    vmovups     zmm1, [r8 + rcx*2]
    vdpbf16ps   zmm4, zmm1, zmm0

    // Head 1
    vmovups     zmm1, [r9 + rcx*2]
    vdpbf16ps   zmm5, zmm1, zmm0

    // Head 2
    vmovups     zmm1, [r10 + rcx*2]
    vdpbf16ps   zmm6, zmm1, zmm0

    // Head 3
    vmovups     zmm1, [r11 + rcx*2]
    vdpbf16ps   zmm7, zmm1, zmm0

    // Head 4
    vmovups     zmm1, [r12 + rcx*2]
    vdpbf16ps   zmm8, zmm1, zmm0

    // Head 5
    vmovups     zmm1, [r13 + rcx*2]
    vdpbf16ps   zmm9, zmm1, zmm0

    // Head 6
    vmovups     zmm1, [r14 + rcx*2]
    vdpbf16ps   zmm10, zmm1, zmm0

    // Head 7
    vmovups     zmm1, [r15 + rcx*2]
    vdpbf16ps   zmm11, zmm1, zmm0

    add         rcx, 32
    cmp         rcx, rdx
    jb          .Lqbf16_kbf16_dot8_loop

    HSUM8_STORE rbx, 0

    vzeroupper
    pop         r15
    pop         r14
    pop         r13
    pop         r12
    pop         rbx
    ret
