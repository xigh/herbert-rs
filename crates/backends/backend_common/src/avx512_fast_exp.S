// AVX-512 vectorized exp(x) for 16 f32 values.
// Standalone version with its own constants (used by non-fused paths).

.intel_syntax noprefix
#include "avx512_common.inc"

// Constants
#ifdef __APPLE__
.section __TEXT,__const
#else
.section .rodata
#endif

.p2align 6
.Lfexp_log2e:
    .rept 16
    .float 1.4426950408889634
    .endr
.Lfexp_c4:
    .rept 16
    .float 0.009672479704022407
    .endr
.Lfexp_c3:
    .rept 16
    .float 0.05550410866482422
    .endr
.Lfexp_c2:
    .rept 16
    .float 0.2402265069591985
    .endr
.Lfexp_c1:
    .rept 16
    .float 0.6931471824645996
    .endr
.Lfexp_c0:
    .rept 16
    .float 1.0
    .endr
.Lfexp_one_bits:
    .rept 16
    .long 0x3F800000
    .endr
.Lfexp_neg87:
    .rept 16
    .float -87.33654
    .endr

.text

// avx512_fast_exp16(input: *const f32, output: *mut f32)
    .globl SYMBOL(avx512_fast_exp16)
    .p2align 4
SYMBOL(avx512_fast_exp16):
    vmovups     zmm0, [rdi]
    vmaxps      zmm0, zmm0, [rip + .Lfexp_neg87]
    vmulps      zmm1, zmm0, [rip + .Lfexp_log2e]
    vrndscaleps zmm2, zmm1, 0
    vsubps      zmm1, zmm1, zmm2
    vmovaps     zmm3, [rip + .Lfexp_c4]
    vfmadd213ps zmm3, zmm1, [rip + .Lfexp_c3]
    vfmadd213ps zmm3, zmm1, [rip + .Lfexp_c2]
    vfmadd213ps zmm3, zmm1, [rip + .Lfexp_c1]
    vfmadd213ps zmm3, zmm1, [rip + .Lfexp_c0]
    vcvtps2dq   zmm2, zmm2
    vpslld      zmm2, zmm2, 23
    vpaddd      zmm2, zmm2, [rip + .Lfexp_one_bits]
    vmulps      zmm0, zmm3, zmm2
    vmovups     [rsi], zmm0
    vzeroupper
    ret
