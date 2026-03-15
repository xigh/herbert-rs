// AVX-512 BF16 matvec/matmul kernels using vdpbf16ps (Zen4 / SPR)
//
// Weight layout: row-major BF16 [N, K], 2 bytes/element
// Input x: f32 [K] — converted to BF16 inside the kernel
// Output y: f32 [N]
//
// vdpbf16ps zmm_acc, zmm_a, zmm_b:
//   For each pair i in 0..15:
//     acc[i] += a[2*i] * b[2*i] + a[2*i+1] * b[2*i+1]
//   32 BF16 multiplies → 16 f32 partial sums per instruction.
//   Throughput: 1/cycle on Zen4.
//
// Design: all loops (rows × K) inside the .S — one FFI call per tile.
// Inner loops use byte offsets (+64 per iteration) since x86 SIB only allows scale 1,2,4,8.

.intel_syntax noprefix

#ifdef __APPLE__
#define SYMBOL(name) _##name
#else
#define SYMBOL(name) name
#endif

// ============================================================================
// avx512_bf16_matvec_tile
//
// Compute y[0..num_rows] = W @ x
// Converts x from f32 to BF16 internally, then loops over rows × K.
// 4-row unrolling to saturate vdpbf16ps pipeline.
//
// Args (System V ABI):
//   rdi = x_f32     : pointer to f32 input vector [K]
//   rsi = w_bf16    : pointer to BF16 weight matrix at row_start (row-major)
//   rdx = y_f32     : pointer to f32 output at row_start
//   rcx = num_rows  : number of rows to process
//   r8  = k         : input dimension (BF16 elements per row)
// ============================================================================
    .globl SYMBOL(avx512_bf16_matvec_tile)
    .p2align 4
SYMBOL(avx512_bf16_matvec_tile):
    push        rbp
    mov         rbp, rsp
    push        rbx
    push        r12
    push        r13
    push        r14
    push        r15

    // Save args
    mov         r12, rdi            // x_f32 (phase 1 only)
    mov         r13, rsi            // w_bf16
    mov         r14, rdx            // y_f32
    mov         r15, rcx            // num_rows
    mov         rbx, r8             // k

    // Allocate stack for x_bf16: k * 2 bytes, 64-byte aligned
    lea         rax, [rbx*2 + 63]
    and         rax, -64
    sub         rsp, rax
    mov         r9, rsp             // r9 = x_bf16 buffer

    // ── Phase 1: Convert x f32 → BF16 ──────────────────────────────────
    xor         rcx, rcx            // element offset
    mov         rax, rbx
    and         rax, -32            // k_aligned = k & ~31

    cmp         rax, 0
    je          .Lcvt_tail

.Lcvt_loop:
    vmovups     zmm0, [r12 + rcx*4]
    vmovups     zmm1, [r12 + rcx*4 + 64]
    vcvtne2ps2bf16 zmm2, zmm1, zmm0
    vmovdqu16   [r9 + rcx*2], zmm2
    add         rcx, 32
    cmp         rcx, rax
    jb          .Lcvt_loop

.Lcvt_tail:
    mov         rax, rbx
    sub         rax, rcx
    cmp         rax, 0
    je          .Lcvt_done

    cmp         rax, 16
    jbe         .Lcvt_tail_le16

    // 17..31 remaining
    mov         rdx, rax
    sub         rdx, 16
    vmovups     zmm0, [r12 + rcx*4]
    vcvtneps2bf16 ymm0, zmm0
    vmovdqu     [r9 + rcx*2], ymm0
    add         rcx, 16

    mov         eax, 1
    shlx        eax, eax, edx
    dec         eax
    kmovw       k1, eax
    vmovups     zmm0{k1}{z}, [r12 + rcx*4]
    vcvtneps2bf16 ymm0, zmm0
    vmovdqu16   [r9 + rcx*2]{k1}, ymm0
    jmp         .Lcvt_done

.Lcvt_tail_le16:
    mov         edx, eax
    mov         eax, 1
    shlx        eax, eax, edx
    dec         eax
    kmovw       k1, eax
    vmovups     zmm0{k1}{z}, [r12 + rcx*4]
    vcvtneps2bf16 ymm0, zmm0
    vmovdqu16   [r9 + rcx*2]{k1}, ymm0

.Lcvt_done:
    // r12 is now free

    // ── Phase 2: Dot products ───────────────────────────────────────────
    // k_groups_bytes = (k / 32) * 64   (byte count for full groups)
    mov         r10, rbx
    shr         r10, 5              // k_groups
    shl         r10, 6              // k_groups_bytes = k_groups * 64

    // k_tail = k % 32
    mov         r11, rbx
    and         r11, 31

    // Tail mask in k2
    cmp         r11, 0
    je          .Lno_tail_mask
    mov         eax, 1
    mov         ecx, r11d
    shlx        eax, eax, ecx
    dec         eax
    kmovd       k2, eax
    jmp         .Ltail_mask_done
.Lno_tail_mask:
    xor         eax, eax
    kmovd       k2, eax
.Ltail_mask_done:

    // Row stride in bytes = k * 2
    lea         r8, [rbx*2]

    // num_rows_aligned = num_rows & ~3
    mov         rax, r15
    and         rax, -4

    xor         rcx, rcx            // current row index

    cmp         rax, 0
    je          .Lrow_single_check

    // ════════════════════════════════════════════════════════════════════
    // 4-row unrolled loop
    // ════════════════════════════════════════════════════════════════════
.Lrow4_loop:
    // 4 weight row pointers
    mov         rdi, r13                    // w_row0
    lea         rsi, [r13 + r8]             // w_row1
    lea         rdx, [r13 + r8*2]           // w_row2
    lea         r12, [rsi + r8*2]           // w_row3

    // Zero 4 accumulators
    vxorps      zmm0, zmm0, zmm0
    vxorps      zmm1, zmm1, zmm1
    vxorps      zmm2, zmm2, zmm2
    vxorps      zmm3, zmm3, zmm3

    // Save rax (num_rows_aligned) — frees rax for byte offset
    push        rax

    // Inner loop: byte offset in rax, limit in r10 (k_groups_bytes)
    xor         rax, rax
    cmp         r10, 0
    je          .Lrow4_tail

.Lrow4_inner:
    vmovdqu16   zmm4, [r9 + rax]
    vmovdqu16   zmm5, [rdi + rax]
    vmovdqu16   zmm6, [rsi + rax]
    vmovdqu16   zmm7, [rdx + rax]
    vmovdqu16   zmm8, [r12 + rax]

    vdpbf16ps   zmm0, zmm5, zmm4
    vdpbf16ps   zmm1, zmm6, zmm4
    vdpbf16ps   zmm2, zmm7, zmm4
    vdpbf16ps   zmm3, zmm8, zmm4

    add         rax, 64
    cmp         rax, r10
    jb          .Lrow4_inner

.Lrow4_tail:
    cmp         r11, 0
    je          .Lrow4_reduce

    // rax = k_groups_bytes = offset to tail
    vmovdqu16   zmm4{k2}{z}, [r9 + rax]
    vmovdqu16   zmm5{k2}{z}, [rdi + rax]
    vmovdqu16   zmm6{k2}{z}, [rsi + rax]
    vmovdqu16   zmm7{k2}{z}, [rdx + rax]
    vmovdqu16   zmm8{k2}{z}, [r12 + rax]

    vdpbf16ps   zmm0, zmm5, zmm4
    vdpbf16ps   zmm1, zmm6, zmm4
    vdpbf16ps   zmm2, zmm7, zmm4
    vdpbf16ps   zmm3, zmm8, zmm4

.Lrow4_reduce:
    pop         rax                         // restore num_rows_aligned

    // Horizontal sum zmm0 → y[row+0]
    vextractf32x8 ymm10, zmm0, 1
    vaddps      ymm0, ymm0, ymm10
    vextractf128 xmm10, ymm0, 1
    vaddps      xmm0, xmm0, xmm10
    vhaddps     xmm0, xmm0, xmm0
    vhaddps     xmm0, xmm0, xmm0
    vmovss      [r14 + rcx*4], xmm0

    // zmm1 → y[row+1]
    vextractf32x8 ymm10, zmm1, 1
    vaddps      ymm1, ymm1, ymm10
    vextractf128 xmm10, ymm1, 1
    vaddps      xmm1, xmm1, xmm10
    vhaddps     xmm1, xmm1, xmm1
    vhaddps     xmm1, xmm1, xmm1
    vmovss      [r14 + rcx*4 + 4], xmm1

    // zmm2 → y[row+2]
    vextractf32x8 ymm10, zmm2, 1
    vaddps      ymm2, ymm2, ymm10
    vextractf128 xmm10, ymm2, 1
    vaddps      xmm2, xmm2, xmm10
    vhaddps     xmm2, xmm2, xmm2
    vhaddps     xmm2, xmm2, xmm2
    vmovss      [r14 + rcx*4 + 8], xmm2

    // zmm3 → y[row+3]
    vextractf32x8 ymm10, zmm3, 1
    vaddps      ymm3, ymm3, ymm10
    vextractf128 xmm10, ymm3, 1
    vaddps      xmm3, xmm3, xmm10
    vhaddps     xmm3, xmm3, xmm3
    vhaddps     xmm3, xmm3, xmm3
    vmovss      [r14 + rcx*4 + 12], xmm3

    // Advance to next 4 rows
    lea         r13, [r13 + r8*4]
    add         rcx, 4
    cmp         rcx, rax
    jb          .Lrow4_loop

    // ════════════════════════════════════════════════════════════════════
    // Single-row loop for remaining 0-3 rows
    // ════════════════════════════════════════════════════════════════════
.Lrow_single_check:
    cmp         rcx, r15
    jae         .Ldone

.Lrow1_loop:
    mov         rdi, r13

    vxorps      zmm0, zmm0, zmm0

    xor         rax, rax            // byte offset
    cmp         r10, 0
    je          .Lrow1_tail

.Lrow1_inner:
    vmovdqu16   zmm4, [r9 + rax]
    vmovdqu16   zmm5, [rdi + rax]
    vdpbf16ps   zmm0, zmm5, zmm4
    add         rax, 64
    cmp         rax, r10
    jb          .Lrow1_inner

.Lrow1_tail:
    cmp         r11, 0
    je          .Lrow1_reduce

    vmovdqu16   zmm4{k2}{z}, [r9 + rax]
    vmovdqu16   zmm5{k2}{z}, [rdi + rax]
    vdpbf16ps   zmm0, zmm5, zmm4

.Lrow1_reduce:
    vextractf32x8 ymm10, zmm0, 1
    vaddps      ymm0, ymm0, ymm10
    vextractf128 xmm10, ymm0, 1
    vaddps      xmm0, xmm0, xmm10
    vhaddps     xmm0, xmm0, xmm0
    vhaddps     xmm0, xmm0, xmm0
    vmovss      [r14 + rcx*4], xmm0

    add         r13, r8
    inc         rcx
    cmp         rcx, r15
    jb          .Lrow1_loop

.Ldone:
    lea         rsp, [rbp - 40]
    pop         r15
    pop         r14
    pop         r13
    pop         r12
    pop         rbx
    pop         rbp
    vzeroupper
    ret


// ============================================================================
// avx512_bf16_matmul_tile
//
// Compute C[mi, row..row+num_rows] = A[mi, :] @ W[row..]^T  for mi in 0..m
// Converts each A[mi,:] from f32 → BF16 once per mi.
//
// Args (System V ABI):
//   rdi = a_f32     : *const f32, A [m, k] row-major
//   rsi = w_bf16    : *const u16, weight tile start
//   rdx = c_f32     : *mut f32, C [m, n] row-major, offset to tile start
//   rcx = num_rows  : rows in this tile
//   r8  = k         : input dimension
//   r9  = m         : number of input vectors
//   [rbp+16] = n    : total output dim (C row stride)
// ============================================================================
    .globl SYMBOL(avx512_bf16_matmul_tile)
    .p2align 4
SYMBOL(avx512_bf16_matmul_tile):
    push        rbp
    mov         rbp, rsp
    push        rbx
    push        r12
    push        r13
    push        r14
    push        r15

    // Save args
    mov         r12, rdi            // a_f32
    mov         r13, rsi            // w_bf16 (tile start)
    mov         r14, rdx            // c_f32 (tile start)
    mov         r15, rcx            // num_rows
    mov         rbx, r8             // k
    // r9 = m

    // Load n from stack (7th arg)
    mov         rax, [rbp + 16]
    push        rax                 // [rbp-48] = n

    // Allocate stack for x_bf16: k * 2 bytes, 64-byte aligned
    lea         rax, [rbx*2 + 63]
    and         rax, -64
    sub         rsp, rax

    // k_groups_bytes = (k / 32) * 64
    mov         r10, rbx
    shr         r10, 5
    shl         r10, 6              // r10 = k_groups_bytes

    // k_tail = k % 32
    mov         r11, rbx
    and         r11, 31

    // Tail mask in k2
    cmp         r11, 0
    je          .Lmm_no_tail
    mov         eax, 1
    mov         ecx, r11d
    shlx        eax, eax, ecx
    dec         eax
    kmovd       k2, eax
    jmp         .Lmm_tail_done
.Lmm_no_tail:
    xor         eax, eax
    kmovd       k2, eax
.Lmm_tail_done:

    // Row stride bytes = k * 2
    lea         r8, [rbx*2]

    // ── Loop over m input vectors ───────────────────────────────────────
    xor         rcx, rcx            // mi = 0

.Lmm_mi_loop:
    cmp         rcx, r9
    jae         .Lmm_done

    push        rcx                 // save mi
    push        r9                  // save m

    // x_bf16 buffer = rsp + 16 (2 pushes above)
    lea         rdi, [rsp + 16]

    // a_row_ptr = a_f32 + mi * k * 4
    mov         rax, rcx
    imul        rax, rbx
    lea         rsi, [r12 + rax*4]

    // ── Convert A[mi, :] f32 → BF16 ────────────────────────────────────
    xor         rcx, rcx
    mov         rax, rbx
    and         rax, -32

    cmp         rax, 0
    je          .Lmm_cvt_tail

.Lmm_cvt_loop:
    vmovups     zmm0, [rsi + rcx*4]
    vmovups     zmm1, [rsi + rcx*4 + 64]
    vcvtne2ps2bf16 zmm2, zmm1, zmm0
    vmovdqu16   [rdi + rcx*2], zmm2
    add         rcx, 32
    cmp         rcx, rax
    jb          .Lmm_cvt_loop

.Lmm_cvt_tail:
    mov         rax, rbx
    sub         rax, rcx
    cmp         rax, 0
    je          .Lmm_cvt_done

    cmp         rax, 16
    jbe         .Lmm_cvt_le16

    mov         rdx, rax
    sub         rdx, 16
    vmovups     zmm0, [rsi + rcx*4]
    vcvtneps2bf16 ymm0, zmm0
    vmovdqu     [rdi + rcx*2], ymm0
    add         rcx, 16

    mov         eax, 1
    shlx        eax, eax, edx
    dec         eax
    kmovw       k1, eax
    vmovups     zmm0{k1}{z}, [rsi + rcx*4]
    vcvtneps2bf16 ymm0, zmm0
    vmovdqu16   [rdi + rcx*2]{k1}, ymm0
    jmp         .Lmm_cvt_done

.Lmm_cvt_le16:
    mov         edx, eax
    mov         eax, 1
    shlx        eax, eax, edx
    dec         eax
    kmovw       k1, eax
    vmovups     zmm0{k1}{z}, [rsi + rcx*4]
    vcvtneps2bf16 ymm0, zmm0
    vmovdqu16   [rdi + rcx*2]{k1}, ymm0

.Lmm_cvt_done:
    // rdi = x_bf16 buffer

    // Recover mi (peek)
    mov         rcx, [rsp + 8]

    // c_row_ptr = c_f32 + mi * n * 4
    mov         rax, [rbp - 48]     // n
    imul        rax, rcx            // mi * n
    lea         rsi, [r14 + rax*4]  // &C[mi, 0]

    // w_ptr = tile start
    mov         rdx, r13

    xor         rcx, rcx            // row index

.Lmm_row_loop:
    cmp         rcx, r15
    jae         .Lmm_row_done

    vxorps      zmm0, zmm0, zmm0

    xor         rax, rax            // byte offset
    cmp         r10, 0
    je          .Lmm_row_tail

.Lmm_row_inner:
    vmovdqu16   zmm4, [rdi + rax]
    vmovdqu16   zmm5, [rdx + rax]
    vdpbf16ps   zmm0, zmm5, zmm4
    add         rax, 64
    cmp         rax, r10
    jb          .Lmm_row_inner

.Lmm_row_tail:
    cmp         r11, 0
    je          .Lmm_row_reduce

    vmovdqu16   zmm4{k2}{z}, [rdi + rax]
    vmovdqu16   zmm5{k2}{z}, [rdx + rax]
    vdpbf16ps   zmm0, zmm5, zmm4

.Lmm_row_reduce:
    vextractf32x8 ymm10, zmm0, 1
    vaddps      ymm0, ymm0, ymm10
    vextractf128 xmm10, ymm0, 1
    vaddps      xmm0, xmm0, xmm10
    vhaddps     xmm0, xmm0, xmm0
    vhaddps     xmm0, xmm0, xmm0
    vmovss      [rsi + rcx*4], xmm0

    add         rdx, r8
    inc         rcx
    cmp         rcx, r15
    jb          .Lmm_row_loop

.Lmm_row_done:
    pop         r9                  // m
    pop         rcx                 // mi
    inc         rcx
    jmp         .Lmm_mi_loop

.Lmm_done:
    lea         rsp, [rbp - 40]
    pop         r15
    pop         r14
    pop         r13
    pop         r12
    pop         rbx
    pop         rbp
    vzeroupper
    ret
