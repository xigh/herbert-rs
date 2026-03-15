// AVX-512 VNNI INT8 matvec kernel using VPDPBUSD (Zen4 / ICL / SPR)
//
// Per-channel weights (u8) + per-token activations (s8).
// Outputs raw i32[num_rows] = sum(w_u8[i] * x_i8[i]) over K.
// Caller does bias correction + scaling in Rust:
//   y[row] = (raw_dot - 128 * x_sum) * w_scale[row] * x_scale
//
// VPDPBUSD zmm_acc, zmm_a(u8), zmm_b(i8):
//   64 u8*i8 multiplies + 16 i32 accumulations per instruction.
//   Throughput: 1/cycle Zen4, potentially 2/cycle EPYC Turin.
//
// 4-row unrolling to saturate VPDPBUSD pipeline (5-cycle latency Zen4).
//
// The stride parameters allow the same kernel for matvec (weights=matrix,
// x=vector) and future matmul (both could be matrices).

.intel_syntax noprefix

#ifdef __APPLE__
#define SYMBOL(name) _##name
#else
#define SYMBOL(name) name
#endif

// ============================================================================
// avx512_int8_raw_dot_tile
//
// Args (System V ABI):
//   rdi = src_u8     : *const u8 (weights, row-major)
//   rsi = src_s8     : *const i8 (activations)
//   rdx = y_i32      : *mut i32, output [num_rows]
//   rcx = num_rows
//   r8  = k          : elements per row
//   r9  = u8_stride  : row stride for u8 source (K for weights, 0 for shared vector)
//   [rbp+16] = s8_stride : row stride for s8 source (0 for shared vector, K for weights)
// ============================================================================
    .globl SYMBOL(avx512_int8_raw_dot_tile)
    .p2align 4
SYMBOL(avx512_int8_raw_dot_tile):
    push        rbp
    mov         rbp, rsp
    push        rbx
    push        r12
    push        r13
    push        r14
    push        r15

    mov         r12, rdi            // src_u8
    mov         r13, rsi            // src_s8
    mov         r14, rdx            // y_i32
    mov         r15, rcx            // num_rows
    mov         rbx, r8             // k
    mov         rax, [rbp + 16]     // s8_stride
    push        rax                 // [rbp-48] = s8_stride
    push        r9                  // [rbp-56] = u8_stride

    // k_full = k & ~63 (full 64-byte chunks)
    mov         r10, rbx
    and         r10, -64

    // k_tail = k & 63
    mov         r11, rbx
    and         r11, 63

    // Tail mask in k2
    cmp         r11, 0
    je          .Lno_tail
    mov         rax, -1
    mov         ecx, 64
    sub         ecx, r11d
    shr         rax, cl
    kmovq       k2, rax
    jmp         .Ltail_done
.Lno_tail:
    xor         eax, eax
    kmovq       k2, rax
.Ltail_done:

    // num_rows_4 = num_rows & ~3
    mov         rax, r15
    and         rax, -4

    xor         rcx, rcx            // row = 0

    cmp         rax, 0
    je          .Lsingle_check

    // ════════════════════════════════════════════════════════════════════
    // 4-row unrolled loop
    // ════════════════════════════════════════════════════════════════════
.Lrow4:
    mov         rdi, [rbp - 56]     // u8_stride
    mov         r8,  r12            // u8_row0
    lea         r9,  [r12 + rdi]    // u8_row1
    lea         rdx, [r12 + rdi*2]  // u8_row2
    lea         rsi, [r9 + rdi*2]   // u8_row3

    mov         rdi, [rbp - 48]     // s8_stride
    push        rax                 // save num_rows_4
    push        rcx                 // save row

    // s8 row pointers
    mov         rax, r13
    push        rax                 // s8_row0
    add         rax, rdi
    push        rax                 // s8_row1
    add         rax, rdi
    push        rax                 // s8_row2
    add         rax, rdi
    push        rax                 // s8_row3
    // [rsp]=s8_row3, [rsp+8]=s8_row2, [rsp+16]=s8_row1, [rsp+24]=s8_row0

    vpxord      zmm0, zmm0, zmm0
    vpxord      zmm1, zmm1, zmm1
    vpxord      zmm2, zmm2, zmm2
    vpxord      zmm3, zmm3, zmm3

    xor         rax, rax
    cmp         r10, 0
    je          .Lrow4_tail

.Lrow4_inner:
    vmovdqu8    zmm4, [r8 + rax]
    vmovdqu8    zmm5, [r9 + rax]
    vmovdqu8    zmm8, [rdx + rax]
    vmovdqu8    zmm9, [rsi + rax]

    mov         rdi, [rsp + 24]
    vmovdqu8    zmm6, [rdi + rax]
    mov         rdi, [rsp + 16]
    vmovdqu8    zmm7, [rdi + rax]
    mov         rdi, [rsp + 8]
    vmovdqu8    zmm10, [rdi + rax]
    mov         rdi, [rsp]
    vmovdqu8    zmm11, [rdi + rax]

    vpdpbusd    zmm0, zmm4, zmm6
    vpdpbusd    zmm1, zmm5, zmm7
    vpdpbusd    zmm2, zmm8, zmm10
    vpdpbusd    zmm3, zmm9, zmm11

    add         rax, 64
    cmp         rax, r10
    jb          .Lrow4_inner

.Lrow4_tail:
    cmp         r11, 0
    je          .Lrow4_reduce

    vmovdqu8    zmm4{k2}{z}, [r8 + rax]
    vmovdqu8    zmm5{k2}{z}, [r9 + rax]
    vmovdqu8    zmm8{k2}{z}, [rdx + rax]
    vmovdqu8    zmm9{k2}{z}, [rsi + rax]

    mov         rdi, [rsp + 24]
    vmovdqu8    zmm6{k2}{z}, [rdi + rax]
    mov         rdi, [rsp + 16]
    vmovdqu8    zmm7{k2}{z}, [rdi + rax]
    mov         rdi, [rsp + 8]
    vmovdqu8    zmm10{k2}{z}, [rdi + rax]
    mov         rdi, [rsp]
    vmovdqu8    zmm11{k2}{z}, [rdi + rax]

    vpdpbusd    zmm0, zmm4, zmm6
    vpdpbusd    zmm1, zmm5, zmm7
    vpdpbusd    zmm2, zmm8, zmm10
    vpdpbusd    zmm3, zmm9, zmm11

.Lrow4_reduce:
    add         rsp, 32             // pop s8 row pointers
    pop         rcx                 // row
    pop         rax                 // num_rows_4

    // hsum zmm0 → i32
    vextracti32x8 ymm10, zmm0, 1
    vpaddd      ymm0, ymm0, ymm10
    vextracti128 xmm10, ymm0, 1
    vpaddd      xmm0, xmm0, xmm10
    vpsrldq     xmm10, xmm0, 8
    vpaddd      xmm0, xmm0, xmm10
    vpsrldq     xmm10, xmm0, 4
    vpaddd      xmm0, xmm0, xmm10
    vmovd       [r14 + rcx*4], xmm0

    // hsum zmm1
    vextracti32x8 ymm10, zmm1, 1
    vpaddd      ymm1, ymm1, ymm10
    vextracti128 xmm10, ymm1, 1
    vpaddd      xmm1, xmm1, xmm10
    vpsrldq     xmm10, xmm1, 8
    vpaddd      xmm1, xmm1, xmm10
    vpsrldq     xmm10, xmm1, 4
    vpaddd      xmm1, xmm1, xmm10
    vmovd       [r14 + rcx*4 + 4], xmm1

    // hsum zmm2
    vextracti32x8 ymm10, zmm2, 1
    vpaddd      ymm2, ymm2, ymm10
    vextracti128 xmm10, ymm2, 1
    vpaddd      xmm2, xmm2, xmm10
    vpsrldq     xmm10, xmm2, 8
    vpaddd      xmm2, xmm2, xmm10
    vpsrldq     xmm10, xmm2, 4
    vpaddd      xmm2, xmm2, xmm10
    vmovd       [r14 + rcx*4 + 8], xmm2

    // hsum zmm3
    vextracti32x8 ymm10, zmm3, 1
    vpaddd      ymm3, ymm3, ymm10
    vextracti128 xmm10, ymm3, 1
    vpaddd      xmm3, xmm3, xmm10
    vpsrldq     xmm10, xmm3, 8
    vpaddd      xmm3, xmm3, xmm10
    vpsrldq     xmm10, xmm3, 4
    vpaddd      xmm3, xmm3, xmm10
    vmovd       [r14 + rcx*4 + 12], xmm3

    // Advance by 4 rows
    mov         rdi, [rbp - 56]
    lea         r12, [r12 + rdi*4]
    mov         rdi, [rbp - 48]
    lea         r13, [r13 + rdi*4]

    add         rcx, 4
    cmp         rcx, rax
    jb          .Lrow4

    // ════════════════════════════════════════════════════════════════════
    // Single-row remainder (0-3 rows)
    // ════════════════════════════════════════════════════════════════════
.Lsingle_check:
    cmp         rcx, r15
    jae         .Ldone

.Lrow1:
    vpxord      zmm0, zmm0, zmm0

    xor         rax, rax
    cmp         r10, 0
    je          .Lrow1_tail

.Lrow1_inner:
    vmovdqu8    zmm4, [r12 + rax]
    vmovdqu8    zmm5, [r13 + rax]
    vpdpbusd    zmm0, zmm4, zmm5
    add         rax, 64
    cmp         rax, r10
    jb          .Lrow1_inner

.Lrow1_tail:
    cmp         r11, 0
    je          .Lrow1_reduce

    vmovdqu8    zmm4{k2}{z}, [r12 + rax]
    vmovdqu8    zmm5{k2}{z}, [r13 + rax]
    vpdpbusd    zmm0, zmm4, zmm5

.Lrow1_reduce:
    vextracti32x8 ymm10, zmm0, 1
    vpaddd      ymm0, ymm0, ymm10
    vextracti128 xmm10, ymm0, 1
    vpaddd      xmm0, xmm0, xmm10
    vpsrldq     xmm10, xmm0, 8
    vpaddd      xmm0, xmm0, xmm10
    vpsrldq     xmm10, xmm0, 4
    vpaddd      xmm0, xmm0, xmm10
    vmovd       [r14 + rcx*4], xmm0

    mov         rdi, [rbp - 56]
    add         r12, rdi
    mov         rdi, [rbp - 48]
    add         r13, rdi

    inc         rcx
    cmp         rcx, r15
    jb          .Lrow1

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
