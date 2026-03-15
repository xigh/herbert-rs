// AVX-512 single dot product: Q_f32 · K_i8 → f32 scalar
// Plus SV accumulation for i8 V. Used by decode path.

.intel_syntax noprefix
#include "avx512_common.inc"

// avx512_attn_dot_qf32_ki8(q: *const f32, k: *const i8, dim: u64) → xmm0
    .globl SYMBOL(avx512_attn_dot_qf32_ki8)
    .p2align 4
SYMBOL(avx512_attn_dot_qf32_ki8):
    vxorps      zmm0, zmm0, zmm0

    cmp         rdx, 128
    jb          .Ldot_i8_loop_init

    // Fully unrolled for dim=128
    vpmovsxbd   zmm1, xmmword ptr [rsi]
    vcvtdq2ps   zmm1, zmm1
    vmovups     zmm2, [rdi]
    vfmadd231ps zmm0, zmm1, zmm2

    vpmovsxbd   zmm1, xmmword ptr [rsi + 16]
    vcvtdq2ps   zmm1, zmm1
    vmovups     zmm2, [rdi + 64]
    vfmadd231ps zmm0, zmm1, zmm2

    vpmovsxbd   zmm1, xmmword ptr [rsi + 32]
    vcvtdq2ps   zmm1, zmm1
    vmovups     zmm2, [rdi + 128]
    vfmadd231ps zmm0, zmm1, zmm2

    vpmovsxbd   zmm1, xmmword ptr [rsi + 48]
    vcvtdq2ps   zmm1, zmm1
    vmovups     zmm2, [rdi + 192]
    vfmadd231ps zmm0, zmm1, zmm2

    vpmovsxbd   zmm1, xmmword ptr [rsi + 64]
    vcvtdq2ps   zmm1, zmm1
    vmovups     zmm2, [rdi + 256]
    vfmadd231ps zmm0, zmm1, zmm2

    vpmovsxbd   zmm1, xmmword ptr [rsi + 80]
    vcvtdq2ps   zmm1, zmm1
    vmovups     zmm2, [rdi + 320]
    vfmadd231ps zmm0, zmm1, zmm2

    vpmovsxbd   zmm1, xmmword ptr [rsi + 96]
    vcvtdq2ps   zmm1, zmm1
    vmovups     zmm2, [rdi + 384]
    vfmadd231ps zmm0, zmm1, zmm2

    vpmovsxbd   zmm1, xmmword ptr [rsi + 112]
    vcvtdq2ps   zmm1, zmm1
    vmovups     zmm2, [rdi + 448]
    vfmadd231ps zmm0, zmm1, zmm2

    jmp         .Ldot_i8_hsum

.Ldot_i8_loop_init:
    test        rdx, rdx
    jz          .Ldot_i8_hsum
    xor         rcx, rcx

.Ldot_i8_loop:
    vpmovsxbd   zmm1, xmmword ptr [rsi + rcx]
    vcvtdq2ps   zmm1, zmm1
    vmovups     zmm2, [rdi + rcx*4]
    vfmadd231ps zmm0, zmm1, zmm2
    add         rcx, 16
    cmp         rcx, rdx
    jb          .Ldot_i8_loop

.Ldot_i8_hsum:
    vextractf64x4 ymm1, zmm0, 1
    vaddps      ymm0, ymm0, ymm1
    vextractf128  xmm1, ymm0, 1
    vaddps      xmm0, xmm0, xmm1
    vhaddps     xmm0, xmm0, xmm0
    vhaddps     xmm0, xmm0, xmm0
    vzeroupper
    ret


// avx512_attn_sv_accum_i8(out: *mut f32, v: *const i8, xmm0=score, dim: u64)
    .globl SYMBOL(avx512_attn_sv_accum_i8)
    .p2align 4
SYMBOL(avx512_attn_sv_accum_i8):
    vbroadcastss zmm3, xmm0

    cmp         rdx, 128
    jb          .Lsv_i8_loop_init

    // Fully unrolled for dim=128
    vpmovsxbd   zmm1, xmmword ptr [rsi]
    vcvtdq2ps   zmm1, zmm1
    vmovups     zmm2, [rdi]
    vfmadd231ps zmm2, zmm1, zmm3
    vmovups     [rdi], zmm2

    vpmovsxbd   zmm1, xmmword ptr [rsi + 16]
    vcvtdq2ps   zmm1, zmm1
    vmovups     zmm2, [rdi + 64]
    vfmadd231ps zmm2, zmm1, zmm3
    vmovups     [rdi + 64], zmm2

    vpmovsxbd   zmm1, xmmword ptr [rsi + 32]
    vcvtdq2ps   zmm1, zmm1
    vmovups     zmm2, [rdi + 128]
    vfmadd231ps zmm2, zmm1, zmm3
    vmovups     [rdi + 128], zmm2

    vpmovsxbd   zmm1, xmmword ptr [rsi + 48]
    vcvtdq2ps   zmm1, zmm1
    vmovups     zmm2, [rdi + 192]
    vfmadd231ps zmm2, zmm1, zmm3
    vmovups     [rdi + 192], zmm2

    vpmovsxbd   zmm1, xmmword ptr [rsi + 64]
    vcvtdq2ps   zmm1, zmm1
    vmovups     zmm2, [rdi + 256]
    vfmadd231ps zmm2, zmm1, zmm3
    vmovups     [rdi + 256], zmm2

    vpmovsxbd   zmm1, xmmword ptr [rsi + 80]
    vcvtdq2ps   zmm1, zmm1
    vmovups     zmm2, [rdi + 320]
    vfmadd231ps zmm2, zmm1, zmm3
    vmovups     [rdi + 320], zmm2

    vpmovsxbd   zmm1, xmmword ptr [rsi + 96]
    vcvtdq2ps   zmm1, zmm1
    vmovups     zmm2, [rdi + 384]
    vfmadd231ps zmm2, zmm1, zmm3
    vmovups     [rdi + 384], zmm2

    vpmovsxbd   zmm1, xmmword ptr [rsi + 112]
    vcvtdq2ps   zmm1, zmm1
    vmovups     zmm2, [rdi + 448]
    vfmadd231ps zmm2, zmm1, zmm3
    vmovups     [rdi + 448], zmm2

    vzeroupper
    ret

.Lsv_i8_loop_init:
    test        rdx, rdx
    jz          .Lsv_i8_done
    xor         rcx, rcx

.Lsv_i8_loop:
    vpmovsxbd   zmm1, xmmword ptr [rsi + rcx]
    vcvtdq2ps   zmm1, zmm1
    vmovups     zmm2, [rdi + rcx*4]
    vfmadd231ps zmm2, zmm1, zmm3
    vmovups     [rdi + rcx*4], zmm2
    add         rcx, 16
    cmp         rcx, rdx
    jb          .Lsv_i8_loop

.Lsv_i8_done:
    vzeroupper
    ret
