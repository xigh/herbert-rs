// AVX2+FMA single dot product: Q_f32 · K_bf16 → f32 scalar
// Plus SV accumulation for BF16 V. Used by decode path.

.intel_syntax noprefix
#include "avx2_common.inc"

// avx2_attn_dot_qf32_kbf16(q: *const f32, k: *const u16, dim: u64) → xmm0
    .globl SYMBOL(avx2_attn_dot_qf32_kbf16)
    .p2align 4
SYMBOL(avx2_attn_dot_qf32_kbf16):
    vxorps      ymm0, ymm0, ymm0

    cmp         rdx, 128
    jb          .Lavx2_dot_bf16_loop_init

    // Fully unrolled path for dim=128 (16 iterations × 8 elements, 4× unrolled)
    // Use 4 accumulators (ymm0, ymm12, ymm13, ymm14) for ILP
    vxorps      ymm12, ymm12, ymm12
    vxorps      ymm13, ymm13, ymm13
    vxorps      ymm14, ymm14, ymm14

    .irp I, 0, 4, 8, 12
    // Block \I: 4 iterations of 8 elements each = 32 elements

    // Iter \I+0
    vpmovzxwd   ymm1, xmmword ptr [rsi + (\I * 16)]
    vpslld      ymm1, ymm1, 16
    vmovups     ymm2, [rdi + (\I * 32)]
    vfmadd231ps ymm0, ymm1, ymm2

    // Iter \I+1
    vpmovzxwd   ymm1, xmmword ptr [rsi + (\I * 16 + 16)]
    vpslld      ymm1, ymm1, 16
    vmovups     ymm2, [rdi + (\I * 32 + 32)]
    vfmadd231ps ymm12, ymm1, ymm2

    // Iter \I+2
    vpmovzxwd   ymm1, xmmword ptr [rsi + (\I * 16 + 32)]
    vpslld      ymm1, ymm1, 16
    vmovups     ymm2, [rdi + (\I * 32 + 64)]
    vfmadd231ps ymm13, ymm1, ymm2

    // Iter \I+3
    vpmovzxwd   ymm1, xmmword ptr [rsi + (\I * 16 + 48)]
    vpslld      ymm1, ymm1, 16
    vmovups     ymm2, [rdi + (\I * 32 + 96)]
    vfmadd231ps ymm14, ymm1, ymm2
    .endr

    // Merge 4 accumulators
    vaddps      ymm0, ymm0, ymm12
    vaddps      ymm13, ymm13, ymm14
    vaddps      ymm0, ymm0, ymm13
    jmp         .Lavx2_dot_bf16_hsum

.Lavx2_dot_bf16_loop_init:
    test        rdx, rdx
    jz          .Lavx2_dot_bf16_hsum
    xor         rcx, rcx

.Lavx2_dot_bf16_loop:
    // BF16→f32: zero-extend u16→u32, shift left 16
    vpmovzxwd   ymm1, xmmword ptr [rsi + rcx*2]
    vpslld      ymm1, ymm1, 16
    vmovups     ymm2, [rdi + rcx*4]
    vfmadd231ps ymm0, ymm1, ymm2
    add         rcx, 8
    cmp         rcx, rdx
    jb          .Lavx2_dot_bf16_loop

.Lavx2_dot_bf16_hsum:
    // Horizontal sum of ymm0 → xmm0 scalar
    vextractf128  xmm1, ymm0, 1
    vaddps      xmm0, xmm0, xmm1
    vhaddps     xmm0, xmm0, xmm0
    vhaddps     xmm0, xmm0, xmm0
    vzeroupper
    ret


// avx2_attn_sv_accum_bf16(out: *mut f32, v: *const u16, xmm0=score, dim: u64)
    .globl SYMBOL(avx2_attn_sv_accum_bf16)
    .p2align 4
SYMBOL(avx2_attn_sv_accum_bf16):
    vbroadcastss ymm3, xmm0

    cmp         rdx, 128
    jb          .Lavx2_sv_bf16_loop_init

    // Fully unrolled for dim=128 (16 iterations × 8 elements)
    .irp OFF, 0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240
    vpmovzxwd   ymm1, xmmword ptr [rsi + \OFF]
    vpslld      ymm1, ymm1, 16
    vmovups     ymm2, [rdi + \OFF * 2]
    vfmadd231ps ymm2, ymm1, ymm3
    vmovups     [rdi + \OFF * 2], ymm2
    .endr

    vzeroupper
    ret

.Lavx2_sv_bf16_loop_init:
    test        rdx, rdx
    jz          .Lavx2_sv_bf16_done
    xor         rcx, rcx

.Lavx2_sv_bf16_loop:
    vpmovzxwd   ymm1, xmmword ptr [rsi + rcx*2]
    vpslld      ymm1, ymm1, 16
    vmovups     ymm2, [rdi + rcx*4]
    vfmadd231ps ymm2, ymm1, ymm3
    vmovups     [rdi + rcx*4], ymm2
    add         rcx, 8
    cmp         rcx, rdx
    jb          .Lavx2_sv_bf16_loop

.Lavx2_sv_bf16_done:
    vzeroupper
    ret
