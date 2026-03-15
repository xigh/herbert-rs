// AVX-512 LayerNorm for vision encoder
//
// 3 passes: mean, variance, normalize.
// Saves rbx, r12, r13 (callee-saved) to preserve pointers across passes.
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
// avx512_vision_layer_norm_row
//
// Args (System V ABI):
//   rdi = x       : *const f32
//   rsi = weight  : *const f32
//   rdx = bias    : *const f32
//   xmm0 = eps    : f32
//   rcx = out     : *mut f32
//   r8  = dim     : u64
// ============================================================================
    .globl SYMBOL(avx512_vision_layer_norm_row)
    .p2align 4
SYMBOL(avx512_vision_layer_norm_row):
    // Save callee-saved registers
    push        rbx
    push        r12
    push        r13

    // Preserve arguments across passes
    mov         rbx, rdi                // rbx = x
    mov         r12, rsi                // r12 = weight
    mov         r13, rdx                // r13 = bias
    // rcx = out (preserved — not clobbered in our loops)
    // r8  = dim (preserved — not clobbered in our loops)
    // xmm0 = eps — save to xmm15
    vmovss      xmm15, xmm0, xmm0      // xmm15 = eps

    // ═══════════════════════════════════════════════════════════════════════
    // Pass 1: Compute mean
    // ═══════════════════════════════════════════════════════════════════════
    vxorps      zmm0, zmm0, zmm0       // acc0
    vxorps      zmm1, zmm1, zmm1       // acc1
    vxorps      zmm2, zmm2, zmm2       // acc2
    vxorps      zmm3, zmm3, zmm3       // acc3

    xor         rax, rax                // i = 0

    // ─── 4-way unrolled: 64 elements/iter ───
.Lln_mean_loop64:
    lea         r9, [rax + 64]
    cmp         r9, r8
    ja          .Lln_mean_reduce

    vaddps      zmm0, zmm0, [rbx + rax*4]
    vaddps      zmm1, zmm1, [rbx + rax*4 + 64]
    vaddps      zmm2, zmm2, [rbx + rax*4 + 128]
    vaddps      zmm3, zmm3, [rbx + rax*4 + 192]

    add         rax, 64
    jmp         .Lln_mean_loop64

.Lln_mean_reduce:
    vaddps      zmm0, zmm0, zmm1
    vaddps      zmm2, zmm2, zmm3
    vaddps      zmm0, zmm0, zmm2

    // ─── 16-element tail loop ───
.Lln_mean_loop16:
    lea         r9, [rax + 16]
    cmp         r9, r8
    ja          .Lln_mean_hsum

    vaddps      zmm0, zmm0, [rbx + rax*4]
    add         rax, 16
    jmp         .Lln_mean_loop16

    // ─── Horizontal sum ───
.Lln_mean_hsum:
    vextractf32x8 ymm4, zmm0, 1
    vaddps      ymm0, ymm0, ymm4
    vextractf128 xmm4, ymm0, 1
    vaddps      xmm0, xmm0, xmm4
    vhaddps     xmm0, xmm0, xmm0
    vhaddps     xmm0, xmm0, xmm0       // xmm0[0] = sum

    // ─── Scalar tail for mean ───
.Lln_mean_scalar:
    cmp         rax, r8
    jge         .Lln_mean_done

    vaddss      xmm0, xmm0, [rbx + rax*4]
    inc         rax
    jmp         .Lln_mean_scalar

.Lln_mean_done:
    // mean = sum / dim
    vcvtsi2ss   xmm4, xmm4, r8         // dim as f32
    vdivss      xmm13, xmm0, xmm4      // xmm13 = mean
    vbroadcastss zmm13, xmm13           // zmm13 = broadcast mean

    // ═══════════════════════════════════════════════════════════════════════
    // Pass 2: Compute variance
    // ═══════════════════════════════════════════════════════════════════════
    vxorps      zmm0, zmm0, zmm0       // acc0
    vxorps      zmm1, zmm1, zmm1       // acc1
    vxorps      zmm2, zmm2, zmm2       // acc2
    vxorps      zmm3, zmm3, zmm3       // acc3

    xor         rax, rax

    // ─── 4-way unrolled: 64 elements/iter ───
.Lln_var_loop64:
    lea         r9, [rax + 64]
    cmp         r9, r8
    ja          .Lln_var_reduce

    vmovups     zmm4, [rbx + rax*4]
    vsubps      zmm4, zmm4, zmm13
    vfmadd231ps zmm0, zmm4, zmm4

    vmovups     zmm5, [rbx + rax*4 + 64]
    vsubps      zmm5, zmm5, zmm13
    vfmadd231ps zmm1, zmm5, zmm5

    vmovups     zmm6, [rbx + rax*4 + 128]
    vsubps      zmm6, zmm6, zmm13
    vfmadd231ps zmm2, zmm6, zmm6

    vmovups     zmm7, [rbx + rax*4 + 192]
    vsubps      zmm7, zmm7, zmm13
    vfmadd231ps zmm3, zmm7, zmm7

    add         rax, 64
    jmp         .Lln_var_loop64

.Lln_var_reduce:
    vaddps      zmm0, zmm0, zmm1
    vaddps      zmm2, zmm2, zmm3
    vaddps      zmm0, zmm0, zmm2

    // ─── 16-element tail loop ───
.Lln_var_loop16:
    lea         r9, [rax + 16]
    cmp         r9, r8
    ja          .Lln_var_hsum

    vmovups     zmm4, [rbx + rax*4]
    vsubps      zmm4, zmm4, zmm13
    vfmadd231ps zmm0, zmm4, zmm4

    add         rax, 16
    jmp         .Lln_var_loop16

    // ─── Horizontal sum ───
.Lln_var_hsum:
    vextractf32x8 ymm4, zmm0, 1
    vaddps      ymm0, ymm0, ymm4
    vextractf128 xmm4, ymm0, 1
    vaddps      xmm0, xmm0, xmm4
    vhaddps     xmm0, xmm0, xmm0
    vhaddps     xmm0, xmm0, xmm0       // xmm0[0] = var_sum

    // ─── Scalar tail for variance ───
.Lln_var_scalar:
    cmp         rax, r8
    jge         .Lln_var_done

    vmovss      xmm4, [rbx + rax*4]
    vsubss      xmm4, xmm4, xmm13      // d = x - mean
    vfmadd231ss xmm0, xmm4, xmm4       // var += d*d
    inc         rax
    jmp         .Lln_var_scalar

.Lln_var_done:
    // inv_std = 1.0 / sqrt(var/dim + eps)
    vcvtsi2ss   xmm4, xmm4, r8         // dim as f32
    vdivss      xmm0, xmm0, xmm4       // var / dim
    vaddss      xmm0, xmm0, xmm15      // + eps
    vsqrtss     xmm0, xmm0, xmm0       // sqrt(var/dim + eps)
    mov         eax, 0x3F800000         // 1.0f
    vmovd       xmm4, eax
    vdivss      xmm14, xmm4, xmm0      // xmm14 = inv_std
    vbroadcastss zmm14, xmm14           // zmm14 = broadcast inv_std

    // ═══════════════════════════════════════════════════════════════════════
    // Pass 3: Normalize — out[i] = (x[i] - mean) * inv_std * weight[i] + bias[i]
    // ═══════════════════════════════════════════════════════════════════════
    xor         rax, rax

    // ─── 16-element loop ───
.Lln_norm_loop16:
    lea         r9, [rax + 16]
    cmp         r9, r8
    ja          .Lln_norm_scalar

    vmovups     zmm0, [rbx + rax*4]     // x
    vsubps      zmm0, zmm0, zmm13       // x - mean
    vmulps      zmm0, zmm0, zmm14       // * inv_std

    vmovups     zmm1, [r12 + rax*4]     // weight
    vmovups     zmm2, [r13 + rax*4]     // bias
    vfmadd231ps zmm2, zmm0, zmm1        // bias + normed * weight
    vmovups     [rcx + rax*4], zmm2     // store to out

    add         rax, 16
    jmp         .Lln_norm_loop16

    // ─── Scalar tail ───
.Lln_norm_scalar:
    cmp         rax, r8
    jge         .Lln_done

    vmovss      xmm0, [rbx + rax*4]     // x[i]
    vsubss      xmm0, xmm0, xmm13       // - mean
    vmulss      xmm0, xmm0, xmm14       // * inv_std

    vmovss      xmm1, [r12 + rax*4]     // weight[i]
    vmovss      xmm2, [r13 + rax*4]     // bias[i]
    vfmadd231ss xmm2, xmm0, xmm1        // bias + normed * weight
    vmovss      [rcx + rax*4], xmm2     // store

    inc         rax
    jmp         .Lln_norm_scalar

.Lln_done:
    vzeroupper

    // Restore callee-saved registers
    pop         r13
    pop         r12
    pop         rbx
    ret
