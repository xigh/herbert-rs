// AVX-512 BF16 batched dot product using VDPBF16PS
// 8 Q_f32 · K_bf16 → 8 f32 scores
// Each head's Q is converted F32→BF16 inline, K is loaded as BF16 directly.
// Uses VDPBF16PS to process 32 BF16 elements per instruction.

.intel_syntax noprefix
#include "avx512_common.inc"
#include "avx512_hsum8.inc"

// avx512_attn_dot8_qf32_kbf16_dpbf16(q_ptrs: **f32, k: *u16, dim: u64, scores: *mut f32)
    .globl SYMBOL(avx512_attn_dot8_qf32_kbf16_dpbf16)
    .p2align 4
SYMBOL(avx512_attn_dot8_qf32_kbf16_dpbf16):
    push        rbx
    push        r12
    push        r13
    push        r14
    push        r15

    mov         rbx, rcx            // scores output pointer

    // Load 8 Q pointers
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

.Ldpbf16_dot8_loop:
    // Load 32× K BF16 (shared across all 8 heads)
    vmovups     zmm0, [rsi + rcx*2]

    // Head 0
    vmovups     zmm2, [r8 + rcx*4]
    vmovups     zmm3, [r8 + rcx*4 + 64]
    vcvtne2ps2bf16 zmm1, zmm3, zmm2
    vdpbf16ps   zmm4, zmm1, zmm0

    // Head 1
    vmovups     zmm2, [r9 + rcx*4]
    vmovups     zmm3, [r9 + rcx*4 + 64]
    vcvtne2ps2bf16 zmm1, zmm3, zmm2
    vdpbf16ps   zmm5, zmm1, zmm0

    // Head 2
    vmovups     zmm2, [r10 + rcx*4]
    vmovups     zmm3, [r10 + rcx*4 + 64]
    vcvtne2ps2bf16 zmm1, zmm3, zmm2
    vdpbf16ps   zmm6, zmm1, zmm0

    // Head 3
    vmovups     zmm2, [r11 + rcx*4]
    vmovups     zmm3, [r11 + rcx*4 + 64]
    vcvtne2ps2bf16 zmm1, zmm3, zmm2
    vdpbf16ps   zmm7, zmm1, zmm0

    // Head 4
    vmovups     zmm2, [r12 + rcx*4]
    vmovups     zmm3, [r12 + rcx*4 + 64]
    vcvtne2ps2bf16 zmm1, zmm3, zmm2
    vdpbf16ps   zmm8, zmm1, zmm0

    // Head 5
    vmovups     zmm2, [r13 + rcx*4]
    vmovups     zmm3, [r13 + rcx*4 + 64]
    vcvtne2ps2bf16 zmm1, zmm3, zmm2
    vdpbf16ps   zmm9, zmm1, zmm0

    // Head 6
    vmovups     zmm2, [r14 + rcx*4]
    vmovups     zmm3, [r14 + rcx*4 + 64]
    vcvtne2ps2bf16 zmm1, zmm3, zmm2
    vdpbf16ps   zmm10, zmm1, zmm0

    // Head 7
    vmovups     zmm2, [r15 + rcx*4]
    vmovups     zmm3, [r15 + rcx*4 + 64]
    vcvtne2ps2bf16 zmm1, zmm3, zmm2
    vdpbf16ps   zmm11, zmm1, zmm0

    add         rcx, 32
    cmp         rcx, rdx
    jb          .Ldpbf16_dot8_loop

    HSUM8_STORE rbx, 0

    vzeroupper
    pop         r15
    pop         r14
    pop         r13
    pop         r12
    pop         rbx
    ret
