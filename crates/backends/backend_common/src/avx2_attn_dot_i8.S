// AVX2+FMA single dot product: Q_f32 · K_i8 → f32 scalar
// Plus SV accumulation for i8 V. Used by decode path.

.intel_syntax noprefix
#include "avx2_common.inc"

// avx2_attn_dot_qf32_ki8(q: *const f32, k: *const i8, dim: u64) → xmm0
    .globl SYMBOL(avx2_attn_dot_qf32_ki8)
    .p2align 4
SYMBOL(avx2_attn_dot_qf32_ki8):
    vxorps      ymm0, ymm0, ymm0

    cmp         rdx, 128
    jb          .Lavx2_dot_i8_loop_init

    // Fully unrolled path for dim=128 (16 iterations × 8, 4× unrolled)
    vxorps      ymm12, ymm12, ymm12
    vxorps      ymm13, ymm13, ymm13
    vxorps      ymm14, ymm14, ymm14

    .irp I, 0, 4, 8, 12
    // Block \I: 4 iterations of 8 elements = 32 elements

    // Iter \I+0
    vpmovsxbd   ymm1, qword ptr [rsi + (\I * 8)]
    vcvtdq2ps   ymm1, ymm1
    vmovups     ymm2, [rdi + (\I * 32)]
    vfmadd231ps ymm0, ymm1, ymm2

    // Iter \I+1
    vpmovsxbd   ymm1, qword ptr [rsi + (\I * 8 + 8)]
    vcvtdq2ps   ymm1, ymm1
    vmovups     ymm2, [rdi + (\I * 32 + 32)]
    vfmadd231ps ymm12, ymm1, ymm2

    // Iter \I+2
    vpmovsxbd   ymm1, qword ptr [rsi + (\I * 8 + 16)]
    vcvtdq2ps   ymm1, ymm1
    vmovups     ymm2, [rdi + (\I * 32 + 64)]
    vfmadd231ps ymm13, ymm1, ymm2

    // Iter \I+3
    vpmovsxbd   ymm1, qword ptr [rsi + (\I * 8 + 24)]
    vcvtdq2ps   ymm1, ymm1
    vmovups     ymm2, [rdi + (\I * 32 + 96)]
    vfmadd231ps ymm14, ymm1, ymm2
    .endr

    // Merge 4 accumulators
    vaddps      ymm0, ymm0, ymm12
    vaddps      ymm13, ymm13, ymm14
    vaddps      ymm0, ymm0, ymm13
    jmp         .Lavx2_dot_i8_hsum

.Lavx2_dot_i8_loop_init:
    test        rdx, rdx
    jz          .Lavx2_dot_i8_hsum
    xor         rcx, rcx

.Lavx2_dot_i8_loop:
    // i8→f32: sign-extend i8→i32, convert to f32
    vpmovsxbd   ymm1, qword ptr [rsi + rcx]
    vcvtdq2ps   ymm1, ymm1
    vmovups     ymm2, [rdi + rcx*4]
    vfmadd231ps ymm0, ymm1, ymm2
    add         rcx, 8
    cmp         rcx, rdx
    jb          .Lavx2_dot_i8_loop

.Lavx2_dot_i8_hsum:
    // Horizontal sum of ymm0 → xmm0 scalar
    vextractf128  xmm1, ymm0, 1
    vaddps      xmm0, xmm0, xmm1
    vhaddps     xmm0, xmm0, xmm0
    vhaddps     xmm0, xmm0, xmm0
    vzeroupper
    ret


// avx2_attn_sv_accum_i8(out: *mut f32, v: *const i8, xmm0=score, dim: u64)
    .globl SYMBOL(avx2_attn_sv_accum_i8)
    .p2align 4
SYMBOL(avx2_attn_sv_accum_i8):
    vbroadcastss ymm3, xmm0

    cmp         rdx, 128
    jb          .Lavx2_sv_i8_loop_init

    // Fully unrolled for dim=128 (16 iterations × 8 elements)
    .irp OFF, 0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120
    vpmovsxbd   ymm1, qword ptr [rsi + \OFF]
    vcvtdq2ps   ymm1, ymm1
    vmovups     ymm2, [rdi + \OFF * 4]
    vfmadd231ps ymm2, ymm1, ymm3
    vmovups     [rdi + \OFF * 4], ymm2
    .endr

    vzeroupper
    ret

.Lavx2_sv_i8_loop_init:
    test        rdx, rdx
    jz          .Lavx2_sv_i8_done
    xor         rcx, rcx

.Lavx2_sv_i8_loop:
    vpmovsxbd   ymm1, qword ptr [rsi + rcx]
    vcvtdq2ps   ymm1, ymm1
    vmovups     ymm2, [rdi + rcx*4]
    vfmadd231ps ymm2, ymm1, ymm3
    vmovups     [rdi + rcx*4], ymm2
    add         rcx, 8
    cmp         rcx, rdx
    jb          .Lavx2_sv_i8_loop

.Lavx2_sv_i8_done:
    vzeroupper
    ret
