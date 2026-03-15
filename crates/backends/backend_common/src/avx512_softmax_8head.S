// AVX-512 3-pass softmax for 8 independent heads.
//
// scores layout: [valid_len * 8] row-major (j*8 + h)
//   j = 0..valid_len (positions), h = 0..8 (heads)
//
// Pass 1: max per head → 8 maxes
// Pass 2: exp(x - max), accumulate sum per head, store exp back
// Pass 3: normalize by 1/sum (in-place)
//
// Processes 2 positions per zmm iteration (16 f32 = 2 x 8 heads).
// Handles odd valid_len with ymm tail + zmm-padded exp.
//
// Args (System V ABI):
//   rdi = scores    : *mut f32   [valid_len * 8]
//   rsi = valid_len : u64

.intel_syntax noprefix
#include "avx512_common.inc"
#include "avx512_exp_constants.inc"
#include "avx512_exp_inline.inc"

// Local constants
#ifdef __APPLE__
.section __TEXT,__const
#else
.section .rodata
#endif
.p2align 4
.Lsm_neg_inf:
    .long 0xFF800000

.Lsm_one:
    .float 1.0

.text

    .globl SYMBOL(avx512_softmax_8head_inplace)
    .p2align 4
SYMBOL(avx512_softmax_8head_inplace):
    push        rbx
    push        r12

    mov         rbx, rsi            // valid_len (callee-saved)
    test        rbx, rbx
    jz          .Lsm_done

    mov         r12, rbx
    shr         r12, 1              // pair_count = valid_len / 2

    // ════════════════════════════════════════════════
    // Pass 1: find max per head
    // ════════════════════════════════════════════════
    vbroadcastss zmm15, dword ptr [rip + .Lsm_neg_inf]

    mov         rax, rdi
    mov         rcx, r12
    test        rcx, rcx
    jz          .Lsm_max_tail

.Lsm_max_pair:
    vmaxps      zmm15, zmm15, [rax]
    add         rax, 64             // 2 pos x 8 heads x 4 bytes
    dec         rcx
    jnz         .Lsm_max_pair

.Lsm_max_tail:
    test        rbx, 1
    jz          .Lsm_max_reduce

    // Load 8 floats, pad upper 8 with -inf
    vmovups         ymm0, [rax]
    vbroadcastss    zmm1, dword ptr [rip + .Lsm_neg_inf]
    vinsertf64x4    zmm0, zmm1, ymm0, 0    // lower=data, upper=-inf
    vmaxps          zmm15, zmm15, zmm0

.Lsm_max_reduce:
    // Reduce zmm15: max of lower/upper 256-bit halves
    vextractf64x4   ymm0, zmm15, 1
    vmaxps          ymm15, ymm15, ymm0
    // ymm15 = 8 per-head maxes; duplicate to zmm14 = [max, max]
    vinserti64x4    zmm14, zmm15, ymm15, 1

    // ════════════════════════════════════════════════
    // Pass 2: exp(x - max) and sum
    // ════════════════════════════════════════════════
    vxorps      zmm13, zmm13, zmm13 // sum = 0

    mov         rax, rdi
    mov         rcx, r12
    test        rcx, rcx
    jz          .Lsm_exp_tail

.Lsm_exp_pair:
    vmovups     zmm0, [rax]
    vsubps      zmm0, zmm0, zmm14
    INLINE_EXP16 zmm0, zmm1, zmm2, zmm3
    vmovups     [rax], zmm0
    vaddps      zmm13, zmm13, zmm0
    add         rax, 64
    dec         rcx
    jnz         .Lsm_exp_pair

.Lsm_exp_tail:
    test        rbx, 1
    jz          .Lsm_exp_reduce

    // Load 8 floats, pad upper with -87 (exp(-87) ≈ 0)
    vmovups         ymm0, [rax]
    vsubps          ymm0, ymm0, ymm14      // ymm14 = max (lower half of zmm14)
    vbroadcastss    zmm1, dword ptr [rip + .Lexp_neg87]
    vinsertf64x4    zmm0, zmm1, ymm0, 0    // lower=data-max, upper=-87
    INLINE_EXP16 zmm0, zmm1, zmm2, zmm3
    vmovups         [rax], ymm0             // store only lower 8 (real data)
    vaddps          zmm13, zmm13, zmm0      // upper adds exp(-87) ≈ 0

.Lsm_exp_reduce:
    // Reduce sum: lower + upper halves
    vextractf64x4   ymm0, zmm13, 1
    vaddps          ymm13, ymm13, ymm0
    // ymm13 = 8 per-head sums; compute inv_sum = 1/sum
    vbroadcastss    ymm0, dword ptr [rip + .Lsm_one]
    vdivps          ymm13, ymm0, ymm13
    // Duplicate to zmm12 = [inv_sum, inv_sum]
    vinserti64x4    zmm12, zmm13, ymm13, 1

    // ════════════════════════════════════════════════
    // Pass 3: normalize
    // ════════════════════════════════════════════════
    mov         rax, rdi
    mov         rcx, r12
    test        rcx, rcx
    jz          .Lsm_norm_tail

.Lsm_norm_pair:
    vmovups     zmm0, [rax]
    vmulps      zmm0, zmm0, zmm12
    vmovups     [rax], zmm0
    add         rax, 64
    dec         rcx
    jnz         .Lsm_norm_pair

.Lsm_norm_tail:
    test        rbx, 1
    jz          .Lsm_done

    vmovups     ymm0, [rax]
    vmulps      ymm0, ymm0, ymm13  // ymm13 still has inv_sum
    vmovups     [rax], ymm0

.Lsm_done:
    vzeroupper
    pop         r12
    pop         rbx
    ret
