// AVX-512 BF16 single dot product using VDPBF16PS
// Q_f32 · K_bf16 → f32 scalar
// Converts Q from F32→BF16 on the fly, then uses VDPBF16PS for the dot product.
// Processes 32 BF16 elements per VDPBF16PS (2× current throughput).

.intel_syntax noprefix
#include "avx512_common.inc"

// avx512_attn_dot_qf32_kbf16_dpbf16(q: *const f32, k: *const u16, dim: u64) → xmm0
    .globl SYMBOL(avx512_attn_dot_qf32_kbf16_dpbf16)
    .p2align 4
SYMBOL(avx512_attn_dot_qf32_kbf16_dpbf16):
    // rdi = q (f32*), rsi = k (u16* bf16), rdx = dim
    vxorps      zmm0, zmm0, zmm0       // accumulator

    cmp         rdx, 128
    jb          .Ldpbf16_dot_loop_init

    // Fully unrolled path for dim=128 (4 iterations of 32 elements)
    // Iteration 0: elements [0..31]
    vmovups     zmm2, [rdi]             // 16× Q F32 (low half)
    vmovups     zmm3, [rdi + 64]        // 16× Q F32 (high half)
    vcvtne2ps2bf16 zmm1, zmm3, zmm2    // 32× Q → BF16
    vmovups     zmm2, [rsi]             // 32× K BF16
    vdpbf16ps   zmm0, zmm1, zmm2       // dot product

    // Iteration 1: elements [32..63]
    vmovups     zmm2, [rdi + 128]
    vmovups     zmm3, [rdi + 192]
    vcvtne2ps2bf16 zmm1, zmm3, zmm2
    vmovups     zmm2, [rsi + 64]
    vdpbf16ps   zmm0, zmm1, zmm2

    // Iteration 2: elements [64..95]
    vmovups     zmm2, [rdi + 256]
    vmovups     zmm3, [rdi + 320]
    vcvtne2ps2bf16 zmm1, zmm3, zmm2
    vmovups     zmm2, [rsi + 128]
    vdpbf16ps   zmm0, zmm1, zmm2

    // Iteration 3: elements [96..127]
    vmovups     zmm2, [rdi + 384]
    vmovups     zmm3, [rdi + 448]
    vcvtne2ps2bf16 zmm1, zmm3, zmm2
    vmovups     zmm2, [rsi + 192]
    vdpbf16ps   zmm0, zmm1, zmm2

    jmp         .Ldpbf16_dot_hsum

.Ldpbf16_dot_loop_init:
    test        rdx, rdx
    jz          .Ldpbf16_dot_hsum
    xor         rcx, rcx

.Ldpbf16_dot_loop:
    // Generic loop: 32 elements per iteration
    vmovups     zmm2, [rdi + rcx*4]         // 16× Q F32 low
    vmovups     zmm3, [rdi + rcx*4 + 64]    // 16× Q F32 high
    vcvtne2ps2bf16 zmm1, zmm3, zmm2         // 32× Q → BF16
    vmovups     zmm2, [rsi + rcx*2]          // 32× K BF16
    vdpbf16ps   zmm0, zmm1, zmm2
    add         rcx, 32
    cmp         rcx, rdx
    jb          .Ldpbf16_dot_loop

.Ldpbf16_dot_hsum:
    // Horizontal sum zmm0 → scalar in xmm0
    vextractf64x4 ymm1, zmm0, 1
    vaddps      ymm0, ymm0, ymm1
    vextractf128  xmm1, ymm0, 1
    vaddps      xmm0, xmm0, xmm1
    vhaddps     xmm0, xmm0, xmm0
    vhaddps     xmm0, xmm0, xmm0
    vzeroupper
    ret
