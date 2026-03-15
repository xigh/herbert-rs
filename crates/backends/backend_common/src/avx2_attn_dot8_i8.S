// AVX2+FMA batched dot product: 8 Q_f32 · K_i8 → 8 f32 scores
// K converted i8→f32 once, reused across all 8 Q heads.

.intel_syntax noprefix
#include "avx2_common.inc"
#include "avx2_hsum8.inc"

// avx2_attn_dot8_qf32_ki8(q_ptrs: **f32, k: *i8, dim: u64, scores: *mut f32)
    .globl SYMBOL(avx2_attn_dot8_qf32_ki8)
    .p2align 4
SYMBOL(avx2_attn_dot8_qf32_ki8):
    push        rbx
    push        r12
    push        r13
    push        r14
    push        r15

    // Save scores pointer (rcx) in rbx
    mov         rbx, rcx

    // Load 8 Q pointers from array at [rdi]
    mov         r8,  [rdi]
    mov         r9,  [rdi + 8]
    mov         r10, [rdi + 16]
    mov         r11, [rdi + 24]
    mov         r12, [rdi + 32]
    mov         r13, [rdi + 40]
    mov         r14, [rdi + 48]
    mov         r15, [rdi + 56]

    // Zero 8 YMM accumulators
    vxorps      ymm4, ymm4, ymm4
    vxorps      ymm5, ymm5, ymm5
    vxorps      ymm6, ymm6, ymm6
    vxorps      ymm7, ymm7, ymm7
    vxorps      ymm8, ymm8, ymm8
    vxorps      ymm9, ymm9, ymm9
    vxorps      ymm10, ymm10, ymm10
    vxorps      ymm11, ymm11, ymm11

    xor         rcx, rcx

.Lavx2_dot8_i8_loop:
    // Load K once: i8→f32 (8 elements per iteration)
    vpmovsxbd   ymm0, qword ptr [rsi + rcx]
    vcvtdq2ps   ymm0, ymm0

    // 8× FMA: each Q head dot with shared K
    vmovups     ymm1, [r8 + rcx*4]
    vfmadd231ps ymm4, ymm0, ymm1
    vmovups     ymm1, [r9 + rcx*4]
    vfmadd231ps ymm5, ymm0, ymm1
    vmovups     ymm1, [r10 + rcx*4]
    vfmadd231ps ymm6, ymm0, ymm1
    vmovups     ymm1, [r11 + rcx*4]
    vfmadd231ps ymm7, ymm0, ymm1
    vmovups     ymm1, [r12 + rcx*4]
    vfmadd231ps ymm8, ymm0, ymm1
    vmovups     ymm1, [r13 + rcx*4]
    vfmadd231ps ymm9, ymm0, ymm1
    vmovups     ymm1, [r14 + rcx*4]
    vfmadd231ps ymm10, ymm0, ymm1
    vmovups     ymm1, [r15 + rcx*4]
    vfmadd231ps ymm11, ymm0, ymm1

    add         rcx, 8
    cmp         rcx, rdx
    jb          .Lavx2_dot8_i8_loop

    // Horizontal sum of all 8 accumulators → 8 f32 scores
    HSUM8_YMM_STORE rbx, 0

    vzeroupper
    pop         r15
    pop         r14
    pop         r13
    pop         r12
    pop         rbx
    ret
