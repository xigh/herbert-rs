// AVX-512 batched dot product: 8 Q_f32 · K_bf16 → 8 f32 scores
// K converted BF16→f32 once, reused across all 8 Q heads.

.intel_syntax noprefix
#include "avx512_common.inc"
#include "avx512_hsum8.inc"

// avx512_attn_dot8_qf32_kbf16(q_ptrs: **f32, k: *u16, dim: u64, scores: *mut f32)
    .globl SYMBOL(avx512_attn_dot8_qf32_kbf16)
    .p2align 4
SYMBOL(avx512_attn_dot8_qf32_kbf16):
    push        rbx
    push        r12
    push        r13
    push        r14
    push        r15

    mov         rbx, rcx

    mov         r8,  [rdi]
    mov         r9,  [rdi + 8]
    mov         r10, [rdi + 16]
    mov         r11, [rdi + 24]
    mov         r12, [rdi + 32]
    mov         r13, [rdi + 40]
    mov         r14, [rdi + 48]
    mov         r15, [rdi + 56]

    vxorps      zmm4, zmm4, zmm4
    vxorps      zmm5, zmm5, zmm5
    vxorps      zmm6, zmm6, zmm6
    vxorps      zmm7, zmm7, zmm7
    vxorps      zmm8, zmm8, zmm8
    vxorps      zmm9, zmm9, zmm9
    vxorps      zmm10, zmm10, zmm10
    vxorps      zmm11, zmm11, zmm11

    xor         rcx, rcx

.Ldot8_bf16_loop:
    vpmovzxwd   zmm0, ymmword ptr [rsi + rcx*2]
    vpslld      zmm0, zmm0, 16

    vmovups     zmm1, [r8 + rcx*4]
    vfmadd231ps zmm4, zmm0, zmm1
    vmovups     zmm1, [r9 + rcx*4]
    vfmadd231ps zmm5, zmm0, zmm1
    vmovups     zmm1, [r10 + rcx*4]
    vfmadd231ps zmm6, zmm0, zmm1
    vmovups     zmm1, [r11 + rcx*4]
    vfmadd231ps zmm7, zmm0, zmm1
    vmovups     zmm1, [r12 + rcx*4]
    vfmadd231ps zmm8, zmm0, zmm1
    vmovups     zmm1, [r13 + rcx*4]
    vfmadd231ps zmm9, zmm0, zmm1
    vmovups     zmm1, [r14 + rcx*4]
    vfmadd231ps zmm10, zmm0, zmm1
    vmovups     zmm1, [r15 + rcx*4]
    vfmadd231ps zmm11, zmm0, zmm1

    add         rcx, 16
    cmp         rcx, rdx
    jb          .Ldot8_bf16_loop

    HSUM8_STORE rbx, 0

    vzeroupper
    pop         r15
    pop         r14
    pop         r13
    pop         r12
    pop         rbx
    ret
