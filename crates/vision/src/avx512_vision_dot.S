// AVX-512 f32 dot product for vision encoder
//
// 4-way unrolled vfmadd231ps (64 elements/iter) with 16-element tail loop,
// horizontal sum, and scalar tail.
//
// Target: x86_64 with AVX-512F
// Convention: System V AMD64 ABI

.intel_syntax noprefix

#ifdef __APPLE__
#define SYMBOL(name) _##name
#else
#define SYMBOL(name) name
#endif

.text

// ============================================================================
// avx512_vision_dot_f32
//
// Compute dot product: sum(a[i] * b[i]) for i in 0..n
//
// Args (System V ABI):
//   rdi = a   : *const f32
//   rsi = b   : *const f32
//   rdx = n   : u64
//
// Returns:
//   xmm0 = f32 scalar (dot product result)
// ============================================================================
    .globl SYMBOL(avx512_vision_dot_f32)
    .p2align 4
SYMBOL(avx512_vision_dot_f32):
    vxorps      zmm0, zmm0, zmm0       // acc0 = 0
    vxorps      zmm1, zmm1, zmm1       // acc1 = 0
    vxorps      zmm2, zmm2, zmm2       // acc2 = 0
    vxorps      zmm3, zmm3, zmm3       // acc3 = 0

    xor         rcx, rcx                // i = 0

    // ─── Main loop: 64 elements per iteration (4 x 16) ───
.Lavx_dot_loop64:
    lea         rax, [rcx + 64]
    cmp         rax, rdx
    ja          .Lavx_dot_reduce

    vmovups     zmm4, [rdi + rcx*4]
    vmovups     zmm5, [rsi + rcx*4]
    vfmadd231ps zmm0, zmm4, zmm5

    vmovups     zmm6, [rdi + rcx*4 + 64]
    vmovups     zmm7, [rsi + rcx*4 + 64]
    vfmadd231ps zmm1, zmm6, zmm7

    vmovups     zmm8, [rdi + rcx*4 + 128]
    vmovups     zmm9, [rsi + rcx*4 + 128]
    vfmadd231ps zmm2, zmm8, zmm9

    vmovups     zmm10, [rdi + rcx*4 + 192]
    vmovups     zmm11, [rsi + rcx*4 + 192]
    vfmadd231ps zmm3, zmm10, zmm11

    add         rcx, 64
    jmp         .Lavx_dot_loop64

.Lavx_dot_reduce:
    // Reduce 4 accumulators to 1
    vaddps      zmm0, zmm0, zmm1
    vaddps      zmm2, zmm2, zmm3
    vaddps      zmm0, zmm0, zmm2

    // ─── 16-element tail loop ───
.Lavx_dot_loop16:
    lea         rax, [rcx + 16]
    cmp         rax, rdx
    ja          .Lavx_dot_hsum

    vmovups     zmm4, [rdi + rcx*4]
    vmovups     zmm5, [rsi + rcx*4]
    vfmadd231ps zmm0, zmm4, zmm5

    add         rcx, 16
    jmp         .Lavx_dot_loop16

    // ─── Horizontal sum: zmm0 → xmm0 scalar ───
.Lavx_dot_hsum:
    // zmm0 (512b) → ymm0 + ymm4
    vextractf32x8 ymm4, zmm0, 1
    vaddps      ymm0, ymm0, ymm4

    // ymm0 (256b) → xmm0 + xmm4
    vextractf128 xmm4, ymm0, 1
    vaddps      xmm0, xmm0, xmm4

    // xmm0 (128b) → scalar via haddps x2
    vhaddps     xmm0, xmm0, xmm0
    vhaddps     xmm0, xmm0, xmm0

    // ─── Scalar tail ───
.Lavx_dot_scalar:
    cmp         rcx, rdx
    jge         .Lavx_dot_done

    vmovss      xmm4, [rdi + rcx*4]
    vmovss      xmm5, [rsi + rcx*4]
    vfmadd231ss xmm0, xmm4, xmm5

    inc         rcx
    jmp         .Lavx_dot_scalar

.Lavx_dot_done:
    vzeroupper
    ret
