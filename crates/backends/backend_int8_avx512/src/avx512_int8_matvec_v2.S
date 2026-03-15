// AVX-512 VNNI INT8 matvec kernel — OPTIMIZED variants for benchmarking
//
// V2: Matvec-specialized (x loaded once, shared across 4 rows)
//     12 loads/iter → 5 loads/iter  (−58% LDQ pressure)
//
// V3: 8-row unroll with shared x load
//     9 loads / 8 VPDPBUSD = 1.125 loads/op
//
// These are benchmark-only kernels. The production kernel remains in
// backend_int8_avx512/src/avx512_int8_matvec.S unchanged.

.intel_syntax noprefix

#ifdef __APPLE__
#define SYMBOL(name) _##name
#else
#define SYMBOL(name) name
#endif

// ============================================================================
// V2: avx512_int8_matvec_v2
//
// Matvec-specialized: x is a single shared vector (no s8_stride parameter).
// Loads x once per 64-byte chunk, reuses across 4 weight rows.
//
// Args (System V ABI):
//   rdi = w_u8      : *const u8, weights [N, K] row-major
//   rsi = x_i8      : *const i8, input vector [K]
//   rdx = y_i32     : *mut i32, output [num_rows]
//   rcx = num_rows
//   r8  = k         : elements per row (= row stride for weights)
// ============================================================================
    .globl SYMBOL(avx512_int8_matvec_v2)
    .p2align 4
SYMBOL(avx512_int8_matvec_v2):
    push        rbp
    mov         rbp, rsp
    push        rbx
    push        r12
    push        r13
    push        r14
    push        r15

    mov         r12, rdi            // w_u8
    mov         r13, rsi            // x_i8
    mov         r14, rdx            // y_i32
    mov         r15, rcx            // num_rows
    mov         rbx, r8             // k (= u8 row stride)

    // k_full = k & ~63
    mov         r10, rbx
    and         r10, -64

    // k_tail = k & 63
    mov         r11, rbx
    and         r11, 63

    // Tail mask in k2
    cmp         r11, 0
    je          .Lv2_no_tail
    mov         rax, -1
    mov         ecx, 64
    sub         ecx, r11d
    shr         rax, cl
    kmovq       k2, rax
    jmp         .Lv2_tail_done
.Lv2_no_tail:
    xor         eax, eax
    kmovq       k2, rax
.Lv2_tail_done:

    // num_rows_4 = num_rows & ~3
    mov         rax, r15
    and         rax, -4

    xor         rcx, rcx            // row = 0

    cmp         rax, 0
    je          .Lv2_single_check

    // ════════════════════════════════════════════════════════════════════
    // 4-row unrolled loop — x loaded ONCE per chunk, shared across 4 rows
    // Registers: zmm0-3=accumulators, zmm4=x, zmm5-8=weights
    // ════════════════════════════════════════════════════════════════════
.Lv2_row4:
    // Compute weight row pointers
    mov         r8,  r12                // w_row0
    lea         r9,  [r12 + rbx]        // w_row1
    lea         rdx, [r12 + rbx*2]      // w_row2
    lea         rsi, [r9 + rbx*2]       // w_row3

    vpxord      zmm0, zmm0, zmm0
    vpxord      zmm1, zmm1, zmm1
    vpxord      zmm2, zmm2, zmm2
    vpxord      zmm3, zmm3, zmm3

    xor         rdi, rdi                // offset = 0
    cmp         r10, 0
    je          .Lv2_row4_tail

.Lv2_row4_inner:
    vmovdqu8    zmm4, [r13 + rdi]       // x — ONE load, shared
    vmovdqu8    zmm5, [r8 + rdi]        // w_row0
    vmovdqu8    zmm6, [r9 + rdi]        // w_row1
    vmovdqu8    zmm7, [rdx + rdi]       // w_row2
    vmovdqu8    zmm8, [rsi + rdi]       // w_row3
    vpdpbusd    zmm0, zmm5, zmm4       // acc0 += w_row0 * x
    vpdpbusd    zmm1, zmm6, zmm4       // acc1 += w_row1 * x
    vpdpbusd    zmm2, zmm7, zmm4       // acc2 += w_row2 * x
    vpdpbusd    zmm3, zmm8, zmm4       // acc3 += w_row3 * x

    add         rdi, 64
    cmp         rdi, r10
    jb          .Lv2_row4_inner

.Lv2_row4_tail:
    cmp         r11, 0
    je          .Lv2_row4_reduce

    vmovdqu8    zmm4{k2}{z}, [r13 + rdi]
    vmovdqu8    zmm5{k2}{z}, [r8 + rdi]
    vmovdqu8    zmm6{k2}{z}, [r9 + rdi]
    vmovdqu8    zmm7{k2}{z}, [rdx + rdi]
    vmovdqu8    zmm8{k2}{z}, [rsi + rdi]
    vpdpbusd    zmm0, zmm5, zmm4
    vpdpbusd    zmm1, zmm6, zmm4
    vpdpbusd    zmm2, zmm7, zmm4
    vpdpbusd    zmm3, zmm8, zmm4

.Lv2_row4_reduce:
    // hsum zmm0 → i32
    vextracti32x8 ymm10, zmm0, 1
    vpaddd      ymm0, ymm0, ymm10
    vextracti32x4 xmm10, ymm0, 1
    vpaddd      xmm0, xmm0, xmm10
    vpsrldq     xmm10, xmm0, 8
    vpaddd      xmm0, xmm0, xmm10
    vpsrldq     xmm10, xmm0, 4
    vpaddd      xmm0, xmm0, xmm10
    vmovd       [r14 + rcx*4], xmm0

    // hsum zmm1
    vextracti32x8 ymm10, zmm1, 1
    vpaddd      ymm1, ymm1, ymm10
    vextracti32x4 xmm10, ymm1, 1
    vpaddd      xmm1, xmm1, xmm10
    vpsrldq     xmm10, xmm1, 8
    vpaddd      xmm1, xmm1, xmm10
    vpsrldq     xmm10, xmm1, 4
    vpaddd      xmm1, xmm1, xmm10
    vmovd       [r14 + rcx*4 + 4], xmm1

    // hsum zmm2
    vextracti32x8 ymm10, zmm2, 1
    vpaddd      ymm2, ymm2, ymm10
    vextracti32x4 xmm10, ymm2, 1
    vpaddd      xmm2, xmm2, xmm10
    vpsrldq     xmm10, xmm2, 8
    vpaddd      xmm2, xmm2, xmm10
    vpsrldq     xmm10, xmm2, 4
    vpaddd      xmm2, xmm2, xmm10
    vmovd       [r14 + rcx*4 + 8], xmm2

    // hsum zmm3
    vextracti32x8 ymm10, zmm3, 1
    vpaddd      ymm3, ymm3, ymm10
    vextracti32x4 xmm10, ymm3, 1
    vpaddd      xmm3, xmm3, xmm10
    vpsrldq     xmm10, xmm3, 8
    vpaddd      xmm3, xmm3, xmm10
    vpsrldq     xmm10, xmm3, 4
    vpaddd      xmm3, xmm3, xmm10
    vmovd       [r14 + rcx*4 + 12], xmm3

    // Advance by 4 rows
    lea         r12, [r12 + rbx*4]
    add         rcx, 4
    cmp         rcx, rax
    jb          .Lv2_row4

    // ════════════════════════════════════════════════════════════════════
    // Single-row remainder (0-3 rows)
    // ════════════════════════════════════════════════════════════════════
.Lv2_single_check:
    cmp         rcx, r15
    jae         .Lv2_done

.Lv2_row1:
    vpxord      zmm0, zmm0, zmm0
    xor         rdi, rdi
    cmp         r10, 0
    je          .Lv2_row1_tail

.Lv2_row1_inner:
    vmovdqu8    zmm4, [r13 + rdi]       // x
    vmovdqu8    zmm5, [r12 + rdi]       // w
    vpdpbusd    zmm0, zmm5, zmm4
    add         rdi, 64
    cmp         rdi, r10
    jb          .Lv2_row1_inner

.Lv2_row1_tail:
    cmp         r11, 0
    je          .Lv2_row1_reduce
    vmovdqu8    zmm4{k2}{z}, [r13 + rdi]
    vmovdqu8    zmm5{k2}{z}, [r12 + rdi]
    vpdpbusd    zmm0, zmm5, zmm4

.Lv2_row1_reduce:
    vextracti32x8 ymm10, zmm0, 1
    vpaddd      ymm0, ymm0, ymm10
    vextracti32x4 xmm10, ymm0, 1
    vpaddd      xmm0, xmm0, xmm10
    vpsrldq     xmm10, xmm0, 8
    vpaddd      xmm0, xmm0, xmm10
    vpsrldq     xmm10, xmm0, 4
    vpaddd      xmm0, xmm0, xmm10
    vmovd       [r14 + rcx*4], xmm0

    add         r12, rbx
    inc         rcx
    cmp         rcx, r15
    jb          .Lv2_row1

.Lv2_done:
    pop         r15
    pop         r14
    pop         r13
    pop         r12
    pop         rbx
    pop         rbp
    vzeroupper
    ret


// ============================================================================
// V3: avx512_int8_matvec_v3
//
// 8-row unroll with shared x load.
// 9 loads / 8 VPDPBUSD = 1.125 loads/op
//
// Uses zmm0-7 as accumulators, zmm8 for x, zmm9-16 for weights.
// Row pointers: precomputed on stack to free up GPRs.
//
// Args (System V ABI):
//   rdi = w_u8      : *const u8, weights [N, K] row-major
//   rsi = x_i8      : *const i8, input vector [K]
//   rdx = y_i32     : *mut i32, output [num_rows]
//   rcx = num_rows
//   r8  = k         : elements per row (= row stride for weights)
// ============================================================================
    .globl SYMBOL(avx512_int8_matvec_v3)
    .p2align 4
SYMBOL(avx512_int8_matvec_v3):
    push        rbp
    mov         rbp, rsp
    push        rbx
    push        r12
    push        r13
    push        r14
    push        r15

    mov         r12, rdi            // w_u8
    mov         r13, rsi            // x_i8
    mov         r14, rdx            // y_i32
    mov         r15, rcx            // num_rows
    mov         rbx, r8             // k (= row stride)

    // k_full = k & ~63
    mov         r10, rbx
    and         r10, -64

    // k_tail = k & 63
    mov         r11, rbx
    and         r11, 63

    // Tail mask in k2
    cmp         r11, 0
    je          .Lv3_no_tail
    mov         rax, -1
    mov         ecx, 64
    sub         ecx, r11d
    shr         rax, cl
    kmovq       k2, rax
    jmp         .Lv3_tail_done
.Lv3_no_tail:
    xor         eax, eax
    kmovq       k2, rax
.Lv3_tail_done:

    // num_rows_8 = num_rows & ~7
    mov         rax, r15
    and         rax, -8

    xor         rcx, rcx            // row = 0

    cmp         rax, 0
    je          .Lv3_row4_check

    // ════════════════════════════════════════════════════════════════════
    // 8-row unrolled loop
    // zmm0-7 = accumulators, zmm8 = x (shared), zmm9-16 = 8 weight rows
    // Row pointers stored on stack [rsp+0..+56]
    // ════════════════════════════════════════════════════════════════════
.Lv3_row8:
    // Compute 8 row pointers and push to stack
    mov         r8, r12                 // row0
    lea         r9, [r12 + rbx]         // row1
    lea         rdx, [r12 + rbx*2]      // row2
    lea         rsi, [r9 + rbx*2]       // row3
    lea         rdi, [r12 + rbx*4]      // row4
    push        rdi                     // [rsp+32] after all pushes → row4
    lea         rdi, [rdi + rbx]        // row5
    push        rdi                     // [rsp+24] → row5
    lea         rdi, [rdi + rbx]        // row6
    push        rdi                     // [rsp+16] → row6
    lea         rdi, [rdi + rbx]        // row7
    push        rdi                     // [rsp+8] → row7

    push        rax                     // [rsp+0] save num_rows_8
    push        rcx                     // [rsp-8..] save row counter
    // Stack: [rsp]=row, [rsp+8]=num_rows_8, [rsp+16]=row7, [rsp+24]=row6, [rsp+32]=row5, [rsp+40]=row4

    vpxord      zmm0, zmm0, zmm0
    vpxord      zmm1, zmm1, zmm1
    vpxord      zmm2, zmm2, zmm2
    vpxord      zmm3, zmm3, zmm3
    vpxord      zmm4, zmm4, zmm4
    vpxord      zmm5, zmm5, zmm5
    vpxord      zmm6, zmm6, zmm6
    vpxord      zmm7, zmm7, zmm7

    xor         rdi, rdi                // offset = 0
    cmp         r10, 0
    je          .Lv3_row8_tail

.Lv3_row8_inner:
    vmovdqu8    zmm8,  [r13 + rdi]      // x — ONE load for 8 ops

    vmovdqu8    zmm9,  [r8 + rdi]       // w_row0
    vmovdqu8    zmm10, [r9 + rdi]       // w_row1
    vmovdqu8    zmm11, [rdx + rdi]      // w_row2
    vmovdqu8    zmm12, [rsi + rdi]      // w_row3

    vpdpbusd    zmm0, zmm9,  zmm8
    vpdpbusd    zmm1, zmm10, zmm8
    vpdpbusd    zmm2, zmm11, zmm8
    vpdpbusd    zmm3, zmm12, zmm8

    // rows 4-7: load pointers from stack
    mov         rax, [rsp + 40]         // row4
    vmovdqu8    zmm13, [rax + rdi]
    mov         rax, [rsp + 32]         // row5
    vmovdqu8    zmm14, [rax + rdi]
    mov         rax, [rsp + 24]         // row6
    vmovdqu8    zmm15, [rax + rdi]
    mov         rax, [rsp + 16]         // row7
    vmovdqu8    zmm16, [rax + rdi]

    vpdpbusd    zmm4, zmm13, zmm8
    vpdpbusd    zmm5, zmm14, zmm8
    vpdpbusd    zmm6, zmm15, zmm8
    vpdpbusd    zmm7, zmm16, zmm8

    add         rdi, 64
    cmp         rdi, r10
    jb          .Lv3_row8_inner

.Lv3_row8_tail:
    cmp         r11, 0
    je          .Lv3_row8_reduce

    vmovdqu8    zmm8{k2}{z}, [r13 + rdi]

    vmovdqu8    zmm9{k2}{z},  [r8 + rdi]
    vmovdqu8    zmm10{k2}{z}, [r9 + rdi]
    vmovdqu8    zmm11{k2}{z}, [rdx + rdi]
    vmovdqu8    zmm12{k2}{z}, [rsi + rdi]
    vpdpbusd    zmm0, zmm9,  zmm8
    vpdpbusd    zmm1, zmm10, zmm8
    vpdpbusd    zmm2, zmm11, zmm8
    vpdpbusd    zmm3, zmm12, zmm8

    mov         rax, [rsp + 40]
    vmovdqu8    zmm13{k2}{z}, [rax + rdi]
    mov         rax, [rsp + 32]
    vmovdqu8    zmm14{k2}{z}, [rax + rdi]
    mov         rax, [rsp + 24]
    vmovdqu8    zmm15{k2}{z}, [rax + rdi]
    mov         rax, [rsp + 16]
    vmovdqu8    zmm16{k2}{z}, [rax + rdi]
    vpdpbusd    zmm4, zmm13, zmm8
    vpdpbusd    zmm5, zmm14, zmm8
    vpdpbusd    zmm6, zmm15, zmm8
    vpdpbusd    zmm7, zmm16, zmm8

.Lv3_row8_reduce:
    pop         rcx                     // row counter
    pop         rax                     // num_rows_8
    add         rsp, 32                 // pop row4-7 pointers

    // Macro: hsum zmm → scalar i32, store at [r14 + rcx*4 + offset]
    // zmm0
    vextracti32x8 ymm17, zmm0, 1
    vpaddd      ymm0, ymm0, ymm17
    vextracti32x4 xmm17, ymm0, 1
    vpaddd      xmm0, xmm0, xmm17
    vpsrldq     xmm17, xmm0, 8
    vpaddd      xmm0, xmm0, xmm17
    vpsrldq     xmm17, xmm0, 4
    vpaddd      xmm0, xmm0, xmm17
    vmovd       [r14 + rcx*4], xmm0

    // zmm1
    vextracti32x8 ymm17, zmm1, 1
    vpaddd      ymm1, ymm1, ymm17
    vextracti32x4 xmm17, ymm1, 1
    vpaddd      xmm1, xmm1, xmm17
    vpsrldq     xmm17, xmm1, 8
    vpaddd      xmm1, xmm1, xmm17
    vpsrldq     xmm17, xmm1, 4
    vpaddd      xmm1, xmm1, xmm17
    vmovd       [r14 + rcx*4 + 4], xmm1

    // zmm2
    vextracti32x8 ymm17, zmm2, 1
    vpaddd      ymm2, ymm2, ymm17
    vextracti32x4 xmm17, ymm2, 1
    vpaddd      xmm2, xmm2, xmm17
    vpsrldq     xmm17, xmm2, 8
    vpaddd      xmm2, xmm2, xmm17
    vpsrldq     xmm17, xmm2, 4
    vpaddd      xmm2, xmm2, xmm17
    vmovd       [r14 + rcx*4 + 8], xmm2

    // zmm3
    vextracti32x8 ymm17, zmm3, 1
    vpaddd      ymm3, ymm3, ymm17
    vextracti32x4 xmm17, ymm3, 1
    vpaddd      xmm3, xmm3, xmm17
    vpsrldq     xmm17, xmm3, 8
    vpaddd      xmm3, xmm3, xmm17
    vpsrldq     xmm17, xmm3, 4
    vpaddd      xmm3, xmm3, xmm17
    vmovd       [r14 + rcx*4 + 12], xmm3

    // zmm4
    vextracti32x8 ymm17, zmm4, 1
    vpaddd      ymm4, ymm4, ymm17
    vextracti32x4 xmm17, ymm4, 1
    vpaddd      xmm4, xmm4, xmm17
    vpsrldq     xmm17, xmm4, 8
    vpaddd      xmm4, xmm4, xmm17
    vpsrldq     xmm17, xmm4, 4
    vpaddd      xmm4, xmm4, xmm17
    vmovd       [r14 + rcx*4 + 16], xmm4

    // zmm5
    vextracti32x8 ymm17, zmm5, 1
    vpaddd      ymm5, ymm5, ymm17
    vextracti32x4 xmm17, ymm5, 1
    vpaddd      xmm5, xmm5, xmm17
    vpsrldq     xmm17, xmm5, 8
    vpaddd      xmm5, xmm5, xmm17
    vpsrldq     xmm17, xmm5, 4
    vpaddd      xmm5, xmm5, xmm17
    vmovd       [r14 + rcx*4 + 20], xmm5

    // zmm6
    vextracti32x8 ymm17, zmm6, 1
    vpaddd      ymm6, ymm6, ymm17
    vextracti32x4 xmm17, ymm6, 1
    vpaddd      xmm6, xmm6, xmm17
    vpsrldq     xmm17, xmm6, 8
    vpaddd      xmm6, xmm6, xmm17
    vpsrldq     xmm17, xmm6, 4
    vpaddd      xmm6, xmm6, xmm17
    vmovd       [r14 + rcx*4 + 24], xmm6

    // zmm7
    vextracti32x8 ymm17, zmm7, 1
    vpaddd      ymm7, ymm7, ymm17
    vextracti32x4 xmm17, ymm7, 1
    vpaddd      xmm7, xmm7, xmm17
    vpsrldq     xmm17, xmm7, 8
    vpaddd      xmm7, xmm7, xmm17
    vpsrldq     xmm17, xmm7, 4
    vpaddd      xmm7, xmm7, xmm17
    vmovd       [r14 + rcx*4 + 28], xmm7

    // Advance by 8 rows
    lea         r12, [r12 + rbx*8]
    add         rcx, 8
    cmp         rcx, rax
    jb          .Lv3_row8

    // ════════════════════════════════════════════════════════════════════
    // 4-row remainder (0-7 rows left, handle in groups of 4 then 1)
    // ════════════════════════════════════════════════════════════════════
.Lv3_row4_check:
    // How many rows left?
    mov         rax, r15
    sub         rax, rcx
    cmp         rax, 4
    jb          .Lv3_single_check

    // 4-row block (same as v2)
    mov         r8,  r12
    lea         r9,  [r12 + rbx]
    lea         rdx, [r12 + rbx*2]
    lea         rsi, [r9 + rbx*2]

    vpxord      zmm0, zmm0, zmm0
    vpxord      zmm1, zmm1, zmm1
    vpxord      zmm2, zmm2, zmm2
    vpxord      zmm3, zmm3, zmm3

    xor         rdi, rdi
    cmp         r10, 0
    je          .Lv3_rem4_tail

.Lv3_rem4_inner:
    vmovdqu8    zmm8,  [r13 + rdi]
    vmovdqu8    zmm9,  [r8 + rdi]
    vmovdqu8    zmm10, [r9 + rdi]
    vmovdqu8    zmm11, [rdx + rdi]
    vmovdqu8    zmm12, [rsi + rdi]
    vpdpbusd    zmm0, zmm9,  zmm8
    vpdpbusd    zmm1, zmm10, zmm8
    vpdpbusd    zmm2, zmm11, zmm8
    vpdpbusd    zmm3, zmm12, zmm8
    add         rdi, 64
    cmp         rdi, r10
    jb          .Lv3_rem4_inner

.Lv3_rem4_tail:
    cmp         r11, 0
    je          .Lv3_rem4_reduce
    vmovdqu8    zmm8{k2}{z},  [r13 + rdi]
    vmovdqu8    zmm9{k2}{z},  [r8 + rdi]
    vmovdqu8    zmm10{k2}{z}, [r9 + rdi]
    vmovdqu8    zmm11{k2}{z}, [rdx + rdi]
    vmovdqu8    zmm12{k2}{z}, [rsi + rdi]
    vpdpbusd    zmm0, zmm9,  zmm8
    vpdpbusd    zmm1, zmm10, zmm8
    vpdpbusd    zmm2, zmm11, zmm8
    vpdpbusd    zmm3, zmm12, zmm8

.Lv3_rem4_reduce:
    vextracti32x8 ymm17, zmm0, 1
    vpaddd      ymm0, ymm0, ymm17
    vextracti32x4 xmm17, ymm0, 1
    vpaddd      xmm0, xmm0, xmm17
    vpsrldq     xmm17, xmm0, 8
    vpaddd      xmm0, xmm0, xmm17
    vpsrldq     xmm17, xmm0, 4
    vpaddd      xmm0, xmm0, xmm17
    vmovd       [r14 + rcx*4], xmm0

    vextracti32x8 ymm17, zmm1, 1
    vpaddd      ymm1, ymm1, ymm17
    vextracti32x4 xmm17, ymm1, 1
    vpaddd      xmm1, xmm1, xmm17
    vpsrldq     xmm17, xmm1, 8
    vpaddd      xmm1, xmm1, xmm17
    vpsrldq     xmm17, xmm1, 4
    vpaddd      xmm1, xmm1, xmm17
    vmovd       [r14 + rcx*4 + 4], xmm1

    vextracti32x8 ymm17, zmm2, 1
    vpaddd      ymm2, ymm2, ymm17
    vextracti32x4 xmm17, ymm2, 1
    vpaddd      xmm2, xmm2, xmm17
    vpsrldq     xmm17, xmm2, 8
    vpaddd      xmm2, xmm2, xmm17
    vpsrldq     xmm17, xmm2, 4
    vpaddd      xmm2, xmm2, xmm17
    vmovd       [r14 + rcx*4 + 8], xmm2

    vextracti32x8 ymm17, zmm3, 1
    vpaddd      ymm3, ymm3, ymm17
    vextracti32x4 xmm17, ymm3, 1
    vpaddd      xmm3, xmm3, xmm17
    vpsrldq     xmm17, xmm3, 8
    vpaddd      xmm3, xmm3, xmm17
    vpsrldq     xmm17, xmm3, 4
    vpaddd      xmm3, xmm3, xmm17
    vmovd       [r14 + rcx*4 + 12], xmm3

    lea         r12, [r12 + rbx*4]
    add         rcx, 4

    // ════════════════════════════════════════════════════════════════════
    // Single-row remainder (0-3 rows)
    // ════════════════════════════════════════════════════════════════════
.Lv3_single_check:
    cmp         rcx, r15
    jae         .Lv3_done

.Lv3_row1:
    vpxord      zmm0, zmm0, zmm0
    xor         rdi, rdi
    cmp         r10, 0
    je          .Lv3_row1_tail

.Lv3_row1_inner:
    vmovdqu8    zmm8, [r13 + rdi]
    vmovdqu8    zmm9, [r12 + rdi]
    vpdpbusd    zmm0, zmm9, zmm8
    add         rdi, 64
    cmp         rdi, r10
    jb          .Lv3_row1_inner

.Lv3_row1_tail:
    cmp         r11, 0
    je          .Lv3_row1_reduce
    vmovdqu8    zmm8{k2}{z}, [r13 + rdi]
    vmovdqu8    zmm9{k2}{z}, [r12 + rdi]
    vpdpbusd    zmm0, zmm9, zmm8

.Lv3_row1_reduce:
    vextracti32x8 ymm17, zmm0, 1
    vpaddd      ymm0, ymm0, ymm17
    vextracti32x4 xmm17, ymm0, 1
    vpaddd      xmm0, xmm0, xmm17
    vpsrldq     xmm17, xmm0, 8
    vpaddd      xmm0, xmm0, xmm17
    vpsrldq     xmm17, xmm0, 4
    vpaddd      xmm0, xmm0, xmm17
    vmovd       [r14 + rcx*4], xmm0

    add         r12, rbx
    inc         rcx
    cmp         rcx, r15
    jb          .Lv3_row1

.Lv3_done:
    pop         r15
    pop         r14
    pop         r13
    pop         r12
    pop         rbx
    pop         rbp
    vzeroupper
    ret
