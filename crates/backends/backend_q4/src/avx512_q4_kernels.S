// AVX-512 Q4 assembly kernels for Qwen3 inference
//
// Uses VPDPBUSD (AVX-512 VNNI) with 4-bit symmetric quantization.
//
// Weight layout: [n_tiles, K/4, 16, 4] (pre-interleaved)
//   Per K-group, 16 half-lanes × 4 k-values:
//     byte[4*hl + j] = nib(lane_hl, kj) | (nib(lane_hl+16, kj) << 4)
//   Low nibble = lanes 0-15, high nibble = lanes 16-31.
//   16 × 4 = 64 bytes per K-group.
//
// Unpacking (3 ALU ops, no vpunpck):
//   vpandd  zmm_lo, zmm_packed, zmm_mask_0F   → lanes 0-15 ready
//   vpsrld  zmm_tmp, zmm_packed, 4
//   vpandd  zmm_hi, zmm_tmp, zmm_mask_0F      → lanes 16-31 ready
//
// Nibble values are unsigned [0,15] representing symmetric Q4 (q+8).
// VPDPBUSD treats first operand as unsigned, second as signed.
// So we put unpacked nibbles (u8) as the unsigned operand and x_i8 as signed.

.intel_syntax noprefix

#ifdef __APPLE__
#define SYMBOL(name) _##name
#else
#define SYMBOL(name) name
#endif

// ============================================================================
// avx512_q4_matvec_tile
//
// Compute y[0..31] = W_q4[tile] @ x_i8 for a single 32-lane tile.
// Uses dual accumulator sets to hide VPDPBUSD latency (5 cycles on Zen 4).
//
// Args (System V ABI):
//   rdi = x_i8      : pointer to i8 input vector (K elements, quantized)
//   rsi = w_packed   : pointer to u8 packed nibble weights (pre-interleaved)
//   rdx = k_groups   : number of K-groups (K/4)
//   rcx = acc_out    : pointer to i32[32] output accumulator
//
// Even K-groups → accumulate into zmm0/zmm1  (set A)
// Odd  K-groups → accumulate into zmm10/zmm11 (set B)
// After loop: zmm0 += zmm10, zmm1 += zmm11
//
// ============================================================================
    .globl SYMBOL(avx512_q4_matvec_tile)
    .p2align 4
SYMBOL(avx512_q4_matvec_tile):
    // Zero both accumulator sets
    vpxord      zmm0, zmm0, zmm0          // set A lo (lanes 0-15)
    vpxord      zmm1, zmm1, zmm1          // set A hi (lanes 16-31)
    vpxord      zmm10, zmm10, zmm10       // set B lo
    vpxord      zmm11, zmm11, zmm11       // set B hi

    test        rdx, rdx
    jz          .Lq4_mv_store

    mov         r8, rdx

    // Prepare mask constant: 0x0F0F0F0F... in zmm24
    mov         eax, 0x0F0F0F0F
    vpbroadcastd zmm24, eax

    // Main loop: 2 K-groups per iteration, each to a different acc set
    mov         r9, r8
    shr         r9, 1                      // r9 = k_groups / 2 (pairs)
    test        r9, r9
    jz          .Lq4_mv_tail

    .p2align 4
.Lq4_mv_kgroup_loop_2x:
    // --- KG0 → set A (zmm0, zmm1) ---
    vmovdqu8    zmm16, [rsi]
    vpandd      zmm2, zmm16, zmm24        // lo lanes 0-15
    vpsrld      zmm18, zmm16, 4
    vpandd      zmm3, zmm18, zmm24        // hi lanes 16-31
    vpbroadcastd zmm28, dword ptr [rdi]
    vpdpbusd    zmm0, zmm2, zmm28         // set A
    vpdpbusd    zmm1, zmm3, zmm28         // set A

    // --- KG1 → set B (zmm10, zmm11) ---
    vmovdqu8    zmm19, [rsi + 64]
    vpandd      zmm4, zmm19, zmm24        // lo lanes 0-15
    vpsrld      zmm20, zmm19, 4
    vpandd      zmm5, zmm20, zmm24        // hi lanes 16-31
    vpbroadcastd zmm29, dword ptr [rdi + 4]
    vpdpbusd    zmm10, zmm4, zmm29        // set B
    vpdpbusd    zmm11, zmm5, zmm29        // set B

    add         rsi, 128                   // 2 K-groups = 128 packed bytes
    add         rdi, 8                     // 2 K-groups = 8 x values
    dec         r9
    jnz         .Lq4_mv_kgroup_loop_2x

.Lq4_mv_tail:
    // Handle odd K-group remainder → set A
    test        r8, 1
    jz          .Lq4_mv_merge

    vmovdqu8    zmm16, [rsi]
    vpandd      zmm2, zmm16, zmm24
    vpsrld      zmm18, zmm16, 4
    vpandd      zmm3, zmm18, zmm24
    vpbroadcastd zmm28, dword ptr [rdi]
    vpdpbusd    zmm0, zmm2, zmm28
    vpdpbusd    zmm1, zmm3, zmm28

.Lq4_mv_merge:
    // Merge set B into set A
    vpaddd      zmm0, zmm0, zmm10
    vpaddd      zmm1, zmm1, zmm11

.Lq4_mv_store:
    // Store i32 accumulators
    vmovdqu32   [rcx], zmm0
    vmovdqu32   [rcx + 64], zmm1

    vzeroupper
    ret


// ============================================================================
// avx512_q4_matvec_tile_3way
//
// Compute three independent 32-lane tiles against the same x_i8 input.
// Pre-interleaved layout: no vpunpck needed.
//
// Args (System V ABI):
//   rdi = x_i8
//   rsi = w1_packed
//   rdx = w2_packed
//   rcx = w3_packed
//   r8  = k_groups
//   r9  = acc_out_all : i32[96] laid out as [acc1(32), acc2(32), acc3(32)]
// ============================================================================
    .globl SYMBOL(avx512_q4_matvec_tile_3way)
    .p2align 4
SYMBOL(avx512_q4_matvec_tile_3way):
    vpxord      zmm0, zmm0, zmm0
    vpxord      zmm1, zmm1, zmm1
    vpxord      zmm2, zmm2, zmm2
    vpxord      zmm3, zmm3, zmm3
    vpxord      zmm4, zmm4, zmm4
    vpxord      zmm5, zmm5, zmm5

    test        r8, r8
    jz          .Lq4_mv_3way_store

    mov         eax, 0x0F0F0F0F
    vpbroadcastd zmm24, eax

    mov         r10, r8
    shr         r10, 1
    test        r10, r10
    jz          .Lq4_mv_3way_tail

    .p2align 4
.Lq4_mv_3way_loop_2x:
    // --- K-group 0 ---
    vpbroadcastd zmm25, dword ptr [rdi]

    vmovdqu8    zmm16, [rsi]
    vpandd      zmm6, zmm16, zmm24
    vpsrld      zmm18, zmm16, 4
    vpandd      zmm7, zmm18, zmm24
    vpdpbusd    zmm0, zmm6, zmm25
    vpdpbusd    zmm1, zmm7, zmm25

    vmovdqu8    zmm19, [rdx]
    vpandd      zmm8, zmm19, zmm24
    vpsrld      zmm20, zmm19, 4
    vpandd      zmm9, zmm20, zmm24
    vpdpbusd    zmm2, zmm8, zmm25
    vpdpbusd    zmm3, zmm9, zmm25

    vmovdqu8    zmm22, [rcx]
    vpandd      zmm10, zmm22, zmm24
    vpsrld      zmm23, zmm22, 4
    vpandd      zmm11, zmm23, zmm24
    vpdpbusd    zmm4, zmm10, zmm25
    vpdpbusd    zmm5, zmm11, zmm25

    // --- K-group 1 ---
    vpbroadcastd zmm26, dword ptr [rdi + 4]

    vmovdqu8    zmm16, [rsi + 64]
    vpandd      zmm6, zmm16, zmm24
    vpsrld      zmm18, zmm16, 4
    vpandd      zmm7, zmm18, zmm24
    vpdpbusd    zmm0, zmm6, zmm26
    vpdpbusd    zmm1, zmm7, zmm26

    vmovdqu8    zmm19, [rdx + 64]
    vpandd      zmm8, zmm19, zmm24
    vpsrld      zmm20, zmm19, 4
    vpandd      zmm9, zmm20, zmm24
    vpdpbusd    zmm2, zmm8, zmm26
    vpdpbusd    zmm3, zmm9, zmm26

    vmovdqu8    zmm22, [rcx + 64]
    vpandd      zmm10, zmm22, zmm24
    vpsrld      zmm23, zmm22, 4
    vpandd      zmm11, zmm23, zmm24
    vpdpbusd    zmm4, zmm10, zmm26
    vpdpbusd    zmm5, zmm11, zmm26

    add         rsi, 128
    add         rdx, 128
    add         rcx, 128
    add         rdi, 8
    dec         r10
    jnz         .Lq4_mv_3way_loop_2x

.Lq4_mv_3way_tail:
    test        r8, 1
    jz          .Lq4_mv_3way_store

    vpbroadcastd zmm25, dword ptr [rdi]

    vmovdqu8    zmm16, [rsi]
    vpandd      zmm6, zmm16, zmm24
    vpsrld      zmm18, zmm16, 4
    vpandd      zmm7, zmm18, zmm24
    vpdpbusd    zmm0, zmm6, zmm25
    vpdpbusd    zmm1, zmm7, zmm25

    vmovdqu8    zmm19, [rdx]
    vpandd      zmm8, zmm19, zmm24
    vpsrld      zmm20, zmm19, 4
    vpandd      zmm9, zmm20, zmm24
    vpdpbusd    zmm2, zmm8, zmm25
    vpdpbusd    zmm3, zmm9, zmm25

    vmovdqu8    zmm22, [rcx]
    vpandd      zmm10, zmm22, zmm24
    vpsrld      zmm23, zmm22, 4
    vpandd      zmm11, zmm23, zmm24
    vpdpbusd    zmm4, zmm10, zmm25
    vpdpbusd    zmm5, zmm11, zmm25

.Lq4_mv_3way_store:
    vmovdqu32   [r9], zmm0
    vmovdqu32   [r9 + 64], zmm1
    vmovdqu32   [r9 + 128], zmm2
    vmovdqu32   [r9 + 192], zmm3
    vmovdqu32   [r9 + 256], zmm4
    vmovdqu32   [r9 + 320], zmm5

    vzeroupper
    ret


// ============================================================================
// avx512_q4_matmul_tile
//
// Compute C[m_rows, 32] = W_q4[tile] @ X[m_rows, K] using VPDPBUSD.
// Processes up to 8 rows, amortizing weight loads + nibble unpacking.
// Pre-interleaved layout: no vpunpck needed.
//
// Args:
//   rdi = x_i8_base  : pointer to i8 input matrix, row-major [m_rows, K]
//   rsi = w_packed    : pointer to packed nibble tile weights (pre-interleaved)
//   rdx = k_groups    : number of K-groups (K/4)
//   rcx = acc_out     : pointer to i32 output, row-major [m_rows, 32]
//   r8  = m_rows      : number of rows (1..8)
//   r9  = x_stride    : stride between x rows in bytes (K bytes for i8)
//
// Uses 16 ZMM accumulators (8 rows × 2 halves)
// ============================================================================
    .globl SYMBOL(avx512_q4_matmul_tile)
    .p2align 4
SYMBOL(avx512_q4_matmul_tile):
    push        rbx
    push        r12
    push        r13
    push        r14
    push        r15

    // Zero all 16 accumulators
    vpxord      zmm0, zmm0, zmm0
    vpxord      zmm1, zmm1, zmm1
    vpxord      zmm2, zmm2, zmm2
    vpxord      zmm3, zmm3, zmm3
    vpxord      zmm4, zmm4, zmm4
    vpxord      zmm5, zmm5, zmm5
    vpxord      zmm6, zmm6, zmm6
    vpxord      zmm7, zmm7, zmm7
    vpxord      zmm8, zmm8, zmm8
    vpxord      zmm9, zmm9, zmm9
    vpxord      zmm10, zmm10, zmm10
    vpxord      zmm11, zmm11, zmm11
    vpxord      zmm12, zmm12, zmm12
    vpxord      zmm13, zmm13, zmm13
    vpxord      zmm14, zmm14, zmm14
    vpxord      zmm15, zmm15, zmm15

    test        rdx, rdx
    jz          .Lq4_mm_store

    // Prepare mask constant
    mov         eax, 0x0F0F0F0F
    vpbroadcastd zmm24, eax

    // Precompute row pointers
    mov         r10, rdi
    add         r10, r9                    // row 1
    lea         r11, [r10 + r9]            // row 2
    lea         r12, [r11 + r9]            // row 3
    lea         r13, [r12 + r9]            // row 4
    lea         r14, [r13 + r9]            // row 5
    lea         r15, [r14 + r9]            // row 6
    lea         rbx, [r15 + r9]            // row 7

    mov         rax, rdx

    .p2align 4
.Lq4_mm_kgroup_loop:
    // Load and unpack shared weights (pre-interleaved: no vpunpck)
    vmovdqu8    zmm16, [rsi]               // 64 packed bytes

    vpandd      zmm25, zmm16, zmm24        // lo lanes 0-15
    vpsrld      zmm18, zmm16, 4
    vpandd      zmm26, zmm18, zmm24        // hi lanes 16-31

    // Row 0
    vpbroadcastd zmm30, dword ptr [rdi]
    vpdpbusd    zmm0, zmm25, zmm30
    vpdpbusd    zmm1, zmm26, zmm30

    // Row 1
    cmp         r8, 2
    jb          .Lq4_mm_advance
    vpbroadcastd zmm30, dword ptr [r10]
    vpdpbusd    zmm2, zmm25, zmm30
    vpdpbusd    zmm3, zmm26, zmm30

    cmp         r8, 3
    jb          .Lq4_mm_advance
    vpbroadcastd zmm30, dword ptr [r11]
    vpdpbusd    zmm4, zmm25, zmm30
    vpdpbusd    zmm5, zmm26, zmm30

    cmp         r8, 4
    jb          .Lq4_mm_advance
    vpbroadcastd zmm30, dword ptr [r12]
    vpdpbusd    zmm6, zmm25, zmm30
    vpdpbusd    zmm7, zmm26, zmm30

    cmp         r8, 5
    jb          .Lq4_mm_advance
    vpbroadcastd zmm30, dword ptr [r13]
    vpdpbusd    zmm8, zmm25, zmm30
    vpdpbusd    zmm9, zmm26, zmm30

    cmp         r8, 6
    jb          .Lq4_mm_advance
    vpbroadcastd zmm30, dword ptr [r14]
    vpdpbusd    zmm10, zmm25, zmm30
    vpdpbusd    zmm11, zmm26, zmm30

    cmp         r8, 7
    jb          .Lq4_mm_advance
    vpbroadcastd zmm30, dword ptr [r15]
    vpdpbusd    zmm12, zmm25, zmm30
    vpdpbusd    zmm13, zmm26, zmm30

    cmp         r8, 8
    jb          .Lq4_mm_advance
    vpbroadcastd zmm30, dword ptr [rbx]
    vpdpbusd    zmm14, zmm25, zmm30
    vpdpbusd    zmm15, zmm26, zmm30

.Lq4_mm_advance:
    add         rsi, 64                    // 1 K-group = 64 packed bytes
    add         rdi, 4                     // 1 K-group = 4 bytes x (i8)
    add         r10, 4
    add         r11, 4
    add         r12, 4
    add         r13, 4
    add         r14, 4
    add         r15, 4
    add         rbx, 4

    dec         rax
    jnz         .Lq4_mm_kgroup_loop

.Lq4_mm_store:
    // Store i32 accumulators (128 bytes per row = 32 i32)
    vmovdqu32   [rcx], zmm0
    vmovdqu32   [rcx + 64], zmm1

    cmp         r8, 2
    jb          .Lq4_mm_done
    vmovdqu32   [rcx + 128], zmm2
    vmovdqu32   [rcx + 192], zmm3

    cmp         r8, 3
    jb          .Lq4_mm_done
    vmovdqu32   [rcx + 256], zmm4
    vmovdqu32   [rcx + 320], zmm5

    cmp         r8, 4
    jb          .Lq4_mm_done
    vmovdqu32   [rcx + 384], zmm6
    vmovdqu32   [rcx + 448], zmm7

    cmp         r8, 5
    jb          .Lq4_mm_done
    vmovdqu32   [rcx + 512], zmm8
    vmovdqu32   [rcx + 576], zmm9

    cmp         r8, 6
    jb          .Lq4_mm_done
    vmovdqu32   [rcx + 640], zmm10
    vmovdqu32   [rcx + 704], zmm11

    cmp         r8, 7
    jb          .Lq4_mm_done
    vmovdqu32   [rcx + 768], zmm12
    vmovdqu32   [rcx + 832], zmm13

    cmp         r8, 8
    jb          .Lq4_mm_done
    vmovdqu32   [rcx + 896], zmm14
    vmovdqu32   [rcx + 960], zmm15

.Lq4_mm_done:
    vzeroupper
    pop         r15
    pop         r14
    pop         r13
    pop         r12
    pop         rbx
    ret


// ============================================================================
// avx512_q4_tile_fused
//
// Fused tile kernel + dequant for all scale groups of one 32-lane tile.
// Single call per tile — eliminates per-SG call/ret overhead and keeps
// f32_acc in zmm30/zmm31 across all SG iterations.
//
// Inner loop per SG:
//   1. VPDPBUSD dot product (dual-acc, 2x unrolled) → i32 acc in zmm0/zmm1
//   2. Zero-point correction: acc -= 8 * col_sum
//   3. Convert i32 → f32
//   4. MUL by w_scale, FMA accumulate: zmm30/31 += temp * x_scale
//
// Tile kernel advances rdi (x_i8) and rsi (w_packed) naturally through the
// data, so they're positioned for the next SG after each iteration.
// The odd-KG tail does NOT advance pointers, so we fix up after the kernel.
//
// Args (System V ABI):
//   rdi = x_i8            : *const i8 (quantized input, advanced per SG)
//   rsi = w_packed         : *const u8 (packed weights for this tile)
//   rdx = kgs_per_sg       : u64 (k-groups per scale group, e.g. 8)
//   rcx = n_sg             : u64 (number of scale groups)
//   r8  = col_sums         : *const i64 (array of n_sg elements)
//   r9  = w_scales         : *const f32 (scales[0 * n + n_start])
// Stack args (after 6 callee-saved pushes + return addr):
//   [rsp+56] = x_scales            : *const f32 (array of n_sg elements)
//   [rsp+64] = f32_acc             : *mut f32 (32 elements, zeroed by caller)
//   [rsp+72] = w_scales_sg_stride  : u64 (stride between SGs in f32 elements = N)
//
// Register allocation:
//   rbx  = kgs_per_sg (persistent)
//   rbp  = f32_acc pointer (persistent, for final store)
//   r12  = x_scales pointer (advances +4 per SG)
//   r13  = w_scales pointer (advances +stride per SG)
//   r14  = w_scales_sg_stride in bytes (persistent)
//   r15  = SG loop counter
//   r9   = w_packed bytes per SG (precomputed = kgs_per_sg * 64)
//   zmm24 = 0x0F nibble mask (constant)
//   zmm30 = f32_acc lo lanes 0-15 (persistent across SGs)
//   zmm31 = f32_acc hi lanes 16-31 (persistent across SGs)
//
// ============================================================================
    .globl SYMBOL(avx512_q4_tile_fused)
    .p2align 4
SYMBOL(avx512_q4_tile_fused):
    push        rbx
    push        rbp
    push        r12
    push        r13
    push        r14
    push        r15

    // Move register args to callee-saved
    mov         rbx, rdx                // kgs_per_sg
    mov         r15, rcx                // n_sg (loop counter)
    mov         r13, r9                 // w_scales pointer

    // Load stack args (6 pushes = 48 bytes + 8 return addr = offset 56)
    mov         r12, [rsp + 56]         // x_scales
    mov         rbp, [rsp + 64]         // f32_acc
    mov         r14, [rsp + 72]         // w_scales_sg_stride (elements)
    shl         r14, 2                  // convert to bytes (×4 for f32)

    // Precompute w_packed stride per SG
    mov         r9, rbx
    shl         r9, 6                   // r9 = kgs_per_sg * 64

    // Prepare constants
    mov         eax, 0x0F0F0F0F
    vpbroadcastd zmm24, eax             // nibble mask

    // Initialize f32_acc in ZMM (zero)
    vpxord      zmm30, zmm30, zmm30     // f32_acc lo
    vpxord      zmm31, zmm31, zmm31     // f32_acc hi

    test        r15, r15
    jz          .Lfused_store

    // ═══════════════════════════════════════════════════════════════
    // SINGLE-SG LOOP (reverted from 2× unroll which regressed)
    // Each iteration: tile kernel → dequant → accumulate into zmm30/31
    // ═══════════════════════════════════════════════════════════════

    .p2align 4
.Lfused_sg_loop:
    // ─── TILE KERNEL ─────────────────────────────────────────
    vpxord      zmm0, zmm0, zmm0       // set A lo (lanes 0-15)
    vpxord      zmm1, zmm1, zmm1       // set A hi (lanes 16-31)
    vpxord      zmm10, zmm10, zmm10    // set B lo
    vpxord      zmm11, zmm11, zmm11    // set B hi

    mov         rax, rbx                // kgs_per_sg
    mov         rcx, rax
    shr         rcx, 1                  // pairs of K-groups
    test        rcx, rcx
    jz          .Lfused_sg_kg_tail

    .p2align 4
.Lfused_sg_kgroup_2x:
    vmovdqu8    zmm16, [rsi]
    vpandd      zmm2, zmm16, zmm24
    vpsrld      zmm18, zmm16, 4
    vpandd      zmm3, zmm18, zmm24
    vpbroadcastd zmm28, dword ptr [rdi]
    vpdpbusd    zmm0, zmm2, zmm28
    vpdpbusd    zmm1, zmm3, zmm28

    vmovdqu8    zmm19, [rsi + 64]
    vpandd      zmm4, zmm19, zmm24
    vpsrld      zmm20, zmm19, 4
    vpandd      zmm5, zmm20, zmm24
    vpbroadcastd zmm29, dword ptr [rdi + 4]
    vpdpbusd    zmm10, zmm4, zmm29
    vpdpbusd    zmm11, zmm5, zmm29

    add         rsi, 128
    add         rdi, 8
    dec         rcx
    jnz         .Lfused_sg_kgroup_2x

.Lfused_sg_kg_tail:
    test        rax, 1
    jz          .Lfused_sg_merge

    vmovdqu8    zmm16, [rsi]
    vpandd      zmm2, zmm16, zmm24
    vpsrld      zmm18, zmm16, 4
    vpandd      zmm3, zmm18, zmm24
    vpbroadcastd zmm28, dword ptr [rdi]
    vpdpbusd    zmm0, zmm2, zmm28
    vpdpbusd    zmm1, zmm3, zmm28

    // Odd KG tail: advance pointers past this K-group
    add         rsi, 64
    add         rdi, 4

.Lfused_sg_merge:
    vpaddd      zmm0, zmm0, zmm10
    vpaddd      zmm1, zmm1, zmm11

    // ─── DEQUANT ─────────────────────────────────────────────
    // zmm0/zmm1 = i32 acc → zmm30/zmm31 += corrected * w_scale * x_scale

    mov         rax, [r8]              // col_sum[sg] (i64)
    imul        rax, 8                 // × Q4_ZERO_BIAS
    mov         ecx, eax
    vpbroadcastd zmm3, ecx

    vpsubd      zmm0, zmm0, zmm3      // corrected_lo
    vpsubd      zmm1, zmm1, zmm3      // corrected_hi

    vcvtdq2ps   zmm0, zmm0
    vcvtdq2ps   zmm1, zmm1

    vmovups     zmm4, [r13]            // w_scales lo (16 f32)
    vmovups     zmm5, [r13 + 64]       // w_scales hi (16 f32)

    vmulps      zmm0, zmm0, zmm4      // temp_lo = corrected * w_scale
    vmulps      zmm1, zmm1, zmm5      // temp_hi = corrected * w_scale

    vbroadcastss zmm6, dword ptr [r12] // x_scale

    vfmadd231ps zmm30, zmm0, zmm6     // f32_acc_lo += temp_lo * x_scale
    vfmadd231ps zmm31, zmm1, zmm6     // f32_acc_hi += temp_hi * x_scale

    // ─── ADVANCE POINTERS ────────────────────────────────────
    // rdi/rsi already advanced by tile kernel (kgs_per_sg * {4,64} bytes)
    add         r8, 8                  // col_sums += 1 (i64 = 8 bytes)
    add         r13, r14               // w_scales += stride (in bytes)
    add         r12, 4                 // x_scales += 1 (f32 = 4 bytes)

    dec         r15
    jnz         .Lfused_sg_loop

.Lfused_store:
    // Store f32_acc from ZMM to memory
    vmovups     [rbp], zmm30
    vmovups     [rbp + 64], zmm31

    vzeroupper
    pop         r15
    pop         r14
    pop         r13
    pop         r12
    pop         rbp
    pop         rbx
    ret


// ============================================================================
// avx512_q4_tile_fused_batch4
//
// Batched 4-SG dequant variant of avx512_q4_tile_fused.
// Processes SGs in batches of 4 with independent f32 temp accumulators
// to break the zmm30/31 RAW dependency chain (13c → ~4c effective).
//
// Same signature and register conventions as avx512_q4_tile_fused.
// 4 independent temp accumulator pairs:
//   Batch slot 0: zmm7/zmm8
//   Batch slot 1: zmm9/zmm12
//   Batch slot 2: zmm13/zmm14
//   Batch slot 3: zmm15/zmm17
//
// After each batch of 4, merge all temps into zmm30/zmm31 (8× vaddps).
// Tail (0-3 remaining SGs) handled with single-SG loop into zmm30/31.
//
// ============================================================================
    .globl SYMBOL(avx512_q4_tile_fused_batch4)
    .p2align 4
SYMBOL(avx512_q4_tile_fused_batch4):
    push        rbx
    push        rbp
    push        r12
    push        r13
    push        r14
    push        r15

    // Move register args to callee-saved
    mov         rbx, rdx                // kgs_per_sg
    mov         r15, rcx                // n_sg (will be modified)
    mov         r13, r9                 // w_scales pointer

    // Load stack args (6 pushes = 48 bytes + 8 return addr = offset 56)
    mov         r12, [rsp + 56]         // x_scales
    mov         rbp, [rsp + 64]         // f32_acc
    mov         r14, [rsp + 72]         // w_scales_sg_stride (elements)
    shl         r14, 2                  // convert to bytes (×4 for f32)

    // Precompute w_packed stride per SG
    mov         r9, rbx
    shl         r9, 6                   // r9 = kgs_per_sg * 64

    // Prepare constants
    mov         eax, 0x0F0F0F0F
    vpbroadcastd zmm24, eax             // nibble mask

    // Initialize f32_acc in ZMM (zero)
    vpxord      zmm30, zmm30, zmm30     // f32_acc lo
    vpxord      zmm31, zmm31, zmm31     // f32_acc hi

    test        r15, r15
    jz          .Lb4_store

    // Compute n_batches = n_sg / 4, tail = n_sg % 4
    mov         rax, r15
    shr         r15, 2                  // r15 = n_sg / 4 (batches)
    and         eax, 3                  // eax = n_sg % 4 (tail count)
    push        rax                     // save tail count on stack

    test        r15, r15
    jz          .Lb4_tail

    // ═══════════════════════════════════════════════════════════════
    // BATCH-4 MAIN LOOP
    // Process 4 SGs per iteration with independent temp accumulators
    // ═══════════════════════════════════════════════════════════════

    .p2align 4
.Lb4_loop:
    // Zero 4 temp accumulator pairs
    vpxord      zmm7,  zmm7,  zmm7     // batch slot 0 lo
    vpxord      zmm8,  zmm8,  zmm8     // batch slot 0 hi
    vpxord      zmm9,  zmm9,  zmm9     // batch slot 1 lo
    vpxord      zmm12, zmm12, zmm12    // batch slot 1 hi
    vpxord      zmm13, zmm13, zmm13    // batch slot 2 lo
    vpxord      zmm14, zmm14, zmm14    // batch slot 2 hi
    vpxord      zmm15, zmm15, zmm15    // batch slot 3 lo
    vpxord      zmm17, zmm17, zmm17    // batch slot 3 hi

    // ─── SG+0: tile kernel → zmm0/1, dequant → zmm7/8 ──────

    vpxord      zmm0, zmm0, zmm0
    vpxord      zmm1, zmm1, zmm1
    vpxord      zmm10, zmm10, zmm10
    vpxord      zmm11, zmm11, zmm11

    mov         rax, rbx
    mov         rcx, rax
    shr         rcx, 1
    test        rcx, rcx
    jz          .Lb4_s0_kg_tail

    .p2align 4
.Lb4_s0_kg_2x:
    vmovdqu8    zmm16, [rsi]
    vpandd      zmm2, zmm16, zmm24
    vpsrld      zmm18, zmm16, 4
    vpandd      zmm3, zmm18, zmm24
    vpbroadcastd zmm28, dword ptr [rdi]
    vpdpbusd    zmm0, zmm2, zmm28
    vpdpbusd    zmm1, zmm3, zmm28

    vmovdqu8    zmm19, [rsi + 64]
    vpandd      zmm4, zmm19, zmm24
    vpsrld      zmm20, zmm19, 4
    vpandd      zmm5, zmm20, zmm24
    vpbroadcastd zmm29, dword ptr [rdi + 4]
    vpdpbusd    zmm10, zmm4, zmm29
    vpdpbusd    zmm11, zmm5, zmm29

    add         rsi, 128
    add         rdi, 8
    dec         rcx
    jnz         .Lb4_s0_kg_2x

.Lb4_s0_kg_tail:
    test        rax, 1
    jz          .Lb4_s0_merge
    vmovdqu8    zmm16, [rsi]
    vpandd      zmm2, zmm16, zmm24
    vpsrld      zmm18, zmm16, 4
    vpandd      zmm3, zmm18, zmm24
    vpbroadcastd zmm28, dword ptr [rdi]
    vpdpbusd    zmm0, zmm2, zmm28
    vpdpbusd    zmm1, zmm3, zmm28
    add         rsi, 64
    add         rdi, 4

.Lb4_s0_merge:
    vpaddd      zmm0, zmm0, zmm10
    vpaddd      zmm1, zmm1, zmm11

    // SG+0 dequant → zmm7/zmm8
    mov         rax, [r8]
    imul        rax, 8
    mov         ecx, eax
    vpbroadcastd zmm3, ecx
    vpsubd      zmm0, zmm0, zmm3
    vpsubd      zmm1, zmm1, zmm3
    vcvtdq2ps   zmm0, zmm0
    vcvtdq2ps   zmm1, zmm1
    vmovups     zmm4, [r13]
    vmovups     zmm5, [r13 + 64]
    vmulps      zmm0, zmm0, zmm4
    vmulps      zmm1, zmm1, zmm5
    vbroadcastss zmm6, dword ptr [r12]
    vfmadd231ps zmm7, zmm0, zmm6       // temp0_lo += corr * x_scale
    vfmadd231ps zmm8, zmm1, zmm6       // temp0_hi += corr * x_scale

    // ─── SG+1: tile kernel → zmm0/1, dequant → zmm9/12 ─────

    vpxord      zmm0, zmm0, zmm0
    vpxord      zmm1, zmm1, zmm1
    vpxord      zmm10, zmm10, zmm10
    vpxord      zmm11, zmm11, zmm11

    mov         rax, rbx
    mov         rcx, rax
    shr         rcx, 1
    test        rcx, rcx
    jz          .Lb4_s1_kg_tail

    .p2align 4
.Lb4_s1_kg_2x:
    vmovdqu8    zmm16, [rsi]
    vpandd      zmm2, zmm16, zmm24
    vpsrld      zmm18, zmm16, 4
    vpandd      zmm3, zmm18, zmm24
    vpbroadcastd zmm28, dword ptr [rdi]
    vpdpbusd    zmm0, zmm2, zmm28
    vpdpbusd    zmm1, zmm3, zmm28

    vmovdqu8    zmm19, [rsi + 64]
    vpandd      zmm4, zmm19, zmm24
    vpsrld      zmm20, zmm19, 4
    vpandd      zmm5, zmm20, zmm24
    vpbroadcastd zmm29, dword ptr [rdi + 4]
    vpdpbusd    zmm10, zmm4, zmm29
    vpdpbusd    zmm11, zmm5, zmm29

    add         rsi, 128
    add         rdi, 8
    dec         rcx
    jnz         .Lb4_s1_kg_2x

.Lb4_s1_kg_tail:
    test        rax, 1
    jz          .Lb4_s1_merge
    vmovdqu8    zmm16, [rsi]
    vpandd      zmm2, zmm16, zmm24
    vpsrld      zmm18, zmm16, 4
    vpandd      zmm3, zmm18, zmm24
    vpbroadcastd zmm28, dword ptr [rdi]
    vpdpbusd    zmm0, zmm2, zmm28
    vpdpbusd    zmm1, zmm3, zmm28
    add         rsi, 64
    add         rdi, 4

.Lb4_s1_merge:
    vpaddd      zmm0, zmm0, zmm10
    vpaddd      zmm1, zmm1, zmm11

    // SG+1 dequant → zmm9/zmm12
    mov         rax, [r8 + 8]
    imul        rax, 8
    mov         ecx, eax
    vpbroadcastd zmm3, ecx
    vpsubd      zmm0, zmm0, zmm3
    vpsubd      zmm1, zmm1, zmm3
    vcvtdq2ps   zmm0, zmm0
    vcvtdq2ps   zmm1, zmm1
    vmovups     zmm4, [r13 + r14]
    vmovups     zmm5, [r13 + r14 + 64]
    vmulps      zmm0, zmm0, zmm4
    vmulps      zmm1, zmm1, zmm5
    vbroadcastss zmm6, dword ptr [r12 + 4]
    vfmadd231ps zmm9,  zmm0, zmm6      // temp1_lo
    vfmadd231ps zmm12, zmm1, zmm6      // temp1_hi

    // ─── SG+2: tile kernel → zmm0/1, dequant → zmm13/14 ────

    vpxord      zmm0, zmm0, zmm0
    vpxord      zmm1, zmm1, zmm1
    vpxord      zmm10, zmm10, zmm10
    vpxord      zmm11, zmm11, zmm11

    mov         rax, rbx
    mov         rcx, rax
    shr         rcx, 1
    test        rcx, rcx
    jz          .Lb4_s2_kg_tail

    .p2align 4
.Lb4_s2_kg_2x:
    vmovdqu8    zmm16, [rsi]
    vpandd      zmm2, zmm16, zmm24
    vpsrld      zmm18, zmm16, 4
    vpandd      zmm3, zmm18, zmm24
    vpbroadcastd zmm28, dword ptr [rdi]
    vpdpbusd    zmm0, zmm2, zmm28
    vpdpbusd    zmm1, zmm3, zmm28

    vmovdqu8    zmm19, [rsi + 64]
    vpandd      zmm4, zmm19, zmm24
    vpsrld      zmm20, zmm19, 4
    vpandd      zmm5, zmm20, zmm24
    vpbroadcastd zmm29, dword ptr [rdi + 4]
    vpdpbusd    zmm10, zmm4, zmm29
    vpdpbusd    zmm11, zmm5, zmm29

    add         rsi, 128
    add         rdi, 8
    dec         rcx
    jnz         .Lb4_s2_kg_2x

.Lb4_s2_kg_tail:
    test        rax, 1
    jz          .Lb4_s2_merge
    vmovdqu8    zmm16, [rsi]
    vpandd      zmm2, zmm16, zmm24
    vpsrld      zmm18, zmm16, 4
    vpandd      zmm3, zmm18, zmm24
    vpbroadcastd zmm28, dword ptr [rdi]
    vpdpbusd    zmm0, zmm2, zmm28
    vpdpbusd    zmm1, zmm3, zmm28
    add         rsi, 64
    add         rdi, 4

.Lb4_s2_merge:
    vpaddd      zmm0, zmm0, zmm10
    vpaddd      zmm1, zmm1, zmm11

    // SG+2 dequant → zmm13/zmm14
    mov         rax, [r8 + 16]
    imul        rax, 8
    mov         ecx, eax
    vpbroadcastd zmm3, ecx
    vpsubd      zmm0, zmm0, zmm3
    vpsubd      zmm1, zmm1, zmm3
    vcvtdq2ps   zmm0, zmm0
    vcvtdq2ps   zmm1, zmm1
    // w_scales for SG+2: r13 + 2*stride
    lea         r10, [r13 + r14*2]
    vmovups     zmm4, [r10]
    vmovups     zmm5, [r10 + 64]
    vmulps      zmm0, zmm0, zmm4
    vmulps      zmm1, zmm1, zmm5
    vbroadcastss zmm6, dword ptr [r12 + 8]
    vfmadd231ps zmm13, zmm0, zmm6      // temp2_lo
    vfmadd231ps zmm14, zmm1, zmm6      // temp2_hi

    // ─── SG+3: tile kernel → zmm0/1, dequant → zmm15/17 ────

    vpxord      zmm0, zmm0, zmm0
    vpxord      zmm1, zmm1, zmm1
    vpxord      zmm10, zmm10, zmm10
    vpxord      zmm11, zmm11, zmm11

    mov         rax, rbx
    mov         rcx, rax
    shr         rcx, 1
    test        rcx, rcx
    jz          .Lb4_s3_kg_tail

    .p2align 4
.Lb4_s3_kg_2x:
    vmovdqu8    zmm16, [rsi]
    vpandd      zmm2, zmm16, zmm24
    vpsrld      zmm18, zmm16, 4
    vpandd      zmm3, zmm18, zmm24
    vpbroadcastd zmm28, dword ptr [rdi]
    vpdpbusd    zmm0, zmm2, zmm28
    vpdpbusd    zmm1, zmm3, zmm28

    vmovdqu8    zmm19, [rsi + 64]
    vpandd      zmm4, zmm19, zmm24
    vpsrld      zmm20, zmm19, 4
    vpandd      zmm5, zmm20, zmm24
    vpbroadcastd zmm29, dword ptr [rdi + 4]
    vpdpbusd    zmm10, zmm4, zmm29
    vpdpbusd    zmm11, zmm5, zmm29

    add         rsi, 128
    add         rdi, 8
    dec         rcx
    jnz         .Lb4_s3_kg_2x

.Lb4_s3_kg_tail:
    test        rax, 1
    jz          .Lb4_s3_merge
    vmovdqu8    zmm16, [rsi]
    vpandd      zmm2, zmm16, zmm24
    vpsrld      zmm18, zmm16, 4
    vpandd      zmm3, zmm18, zmm24
    vpbroadcastd zmm28, dword ptr [rdi]
    vpdpbusd    zmm0, zmm2, zmm28
    vpdpbusd    zmm1, zmm3, zmm28
    add         rsi, 64
    add         rdi, 4

.Lb4_s3_merge:
    vpaddd      zmm0, zmm0, zmm10
    vpaddd      zmm1, zmm1, zmm11

    // SG+3 dequant → zmm15/zmm17
    mov         rax, [r8 + 24]
    imul        rax, 8
    mov         ecx, eax
    vpbroadcastd zmm3, ecx
    vpsubd      zmm0, zmm0, zmm3
    vpsubd      zmm1, zmm1, zmm3
    vcvtdq2ps   zmm0, zmm0
    vcvtdq2ps   zmm1, zmm1
    // w_scales for SG+3: r13 + 3*stride
    lea         r10, [r13 + r14*2]
    add         r10, r14
    vmovups     zmm4, [r10]
    vmovups     zmm5, [r10 + 64]
    vmulps      zmm0, zmm0, zmm4
    vmulps      zmm1, zmm1, zmm5
    vbroadcastss zmm6, dword ptr [r12 + 12]
    vfmadd231ps zmm15, zmm0, zmm6      // temp3_lo
    vfmadd231ps zmm17, zmm1, zmm6      // temp3_hi

    // ─── MERGE 4 TEMPS INTO zmm30/zmm31 ─────────────────────
    vaddps      zmm30, zmm30, zmm7
    vaddps      zmm30, zmm30, zmm9
    vaddps      zmm30, zmm30, zmm13
    vaddps      zmm30, zmm30, zmm15
    vaddps      zmm31, zmm31, zmm8
    vaddps      zmm31, zmm31, zmm12
    vaddps      zmm31, zmm31, zmm14
    vaddps      zmm31, zmm31, zmm17

    // ─── ADVANCE ALL POINTERS BY 4 SGs ──────────────────────
    add         r8, 32                 // col_sums += 4 (4 × i64 = 32 bytes)
    add         r13, r14               // w_scales += 4 × stride
    add         r13, r14
    add         r13, r14
    add         r13, r14
    add         r12, 16                // x_scales += 4 (4 × f32 = 16 bytes)

    dec         r15
    jnz         .Lb4_loop

    // ═══════════════════════════════════════════════════════════════
    // TAIL: handle remaining 0-3 SGs with single-SG loop into zmm30/31
    // ═══════════════════════════════════════════════════════════════
.Lb4_tail:
    pop         rax                     // tail count (0..3)
    test        rax, rax
    jz          .Lb4_store
    mov         r15, rax                // use r15 as tail loop counter

    .p2align 4
.Lb4_tail_loop:
    vpxord      zmm0, zmm0, zmm0
    vpxord      zmm1, zmm1, zmm1
    vpxord      zmm10, zmm10, zmm10
    vpxord      zmm11, zmm11, zmm11

    mov         rax, rbx
    mov         rcx, rax
    shr         rcx, 1
    test        rcx, rcx
    jz          .Lb4_tail_kg_tail

    .p2align 4
.Lb4_tail_kg_2x:
    vmovdqu8    zmm16, [rsi]
    vpandd      zmm2, zmm16, zmm24
    vpsrld      zmm18, zmm16, 4
    vpandd      zmm3, zmm18, zmm24
    vpbroadcastd zmm28, dword ptr [rdi]
    vpdpbusd    zmm0, zmm2, zmm28
    vpdpbusd    zmm1, zmm3, zmm28

    vmovdqu8    zmm19, [rsi + 64]
    vpandd      zmm4, zmm19, zmm24
    vpsrld      zmm20, zmm19, 4
    vpandd      zmm5, zmm20, zmm24
    vpbroadcastd zmm29, dword ptr [rdi + 4]
    vpdpbusd    zmm10, zmm4, zmm29
    vpdpbusd    zmm11, zmm5, zmm29

    add         rsi, 128
    add         rdi, 8
    dec         rcx
    jnz         .Lb4_tail_kg_2x

.Lb4_tail_kg_tail:
    test        rax, 1
    jz          .Lb4_tail_merge
    vmovdqu8    zmm16, [rsi]
    vpandd      zmm2, zmm16, zmm24
    vpsrld      zmm18, zmm16, 4
    vpandd      zmm3, zmm18, zmm24
    vpbroadcastd zmm28, dword ptr [rdi]
    vpdpbusd    zmm0, zmm2, zmm28
    vpdpbusd    zmm1, zmm3, zmm28
    add         rsi, 64
    add         rdi, 4

.Lb4_tail_merge:
    vpaddd      zmm0, zmm0, zmm10
    vpaddd      zmm1, zmm1, zmm11

    // Dequant directly into zmm30/31
    mov         rax, [r8]
    imul        rax, 8
    mov         ecx, eax
    vpbroadcastd zmm3, ecx
    vpsubd      zmm0, zmm0, zmm3
    vpsubd      zmm1, zmm1, zmm3
    vcvtdq2ps   zmm0, zmm0
    vcvtdq2ps   zmm1, zmm1
    vmovups     zmm4, [r13]
    vmovups     zmm5, [r13 + 64]
    vmulps      zmm0, zmm0, zmm4
    vmulps      zmm1, zmm1, zmm5
    vbroadcastss zmm6, dword ptr [r12]
    vfmadd231ps zmm30, zmm0, zmm6
    vfmadd231ps zmm31, zmm1, zmm6

    add         r8, 8
    add         r13, r14
    add         r12, 4

    dec         r15
    jnz         .Lb4_tail_loop

.Lb4_store:
    vmovups     [rbp], zmm30
    vmovups     [rbp + 64], zmm31

    vzeroupper
    pop         r15
    pop         r14
    pop         r13
    pop         r12
    pop         rbp
    pop         rbx
    ret


// ============================================================================
// avx512_q4_dequant_tile_fma
//
// Fused dequantization for one 32-lane tile with FMA accumulation.
// Replaces the Rust intrinsics version with hand-written assembly to
// guarantee vfmadd231ps emission (LLVM often decomposes intrinsics
// into separate vmulps + vaddps).
//
// Computes for each lane n in [0..31]:
//   corrected = acc_i32[n] - Q4_ZERO_BIAS(8) * col_sum
//   f32_acc[n] += corrected_f32 * w_scale[n] * x_scale
//
// The second multiply + add is fused into a single FMA:
//   temp = cvt(corrected) * w_scale
//   f32_acc = vfmadd231ps(f32_acc, temp, broadcast(x_scale))
//   i.e. f32_acc += temp * x_scale
//
// Args (System V ABI):
//   rdi = acc_i32         : pointer to i32[32] from dot-product kernel
//   rsi = col_sum         : i64 (x_group_col_sums[sg])
//   rdx = w_scales_base   : pointer to f32 scales
//   ecx = w_scales_stride : i32 (1 = contiguous, else stride in elements)
//   xmm0 = x_scale        : f32 (x_group_scales[sg])
//   r8  = f32_acc         : pointer to f32[32] accumulator (read-modify-write)
//
// ============================================================================
    .globl SYMBOL(avx512_q4_dequant_tile_fma)
    .p2align 4
SYMBOL(avx512_q4_dequant_tile_fma):
    // 1. Load acc_i32[0..15] and acc_i32[16..31]
    vmovdqu32   zmm1, [rdi]            // i32 lanes 0-15
    vmovdqu32   zmm2, [rdi + 64]       // i32 lanes 16-31

    // 2. Compute bias_i32 = (i32)(8 * col_sum), broadcast
    imul        rsi, 8
    mov         eax, esi               // truncate to 32-bit
    vpbroadcastd zmm3, eax             // broadcast bias_i32

    // 3. Subtract bias: corrected_i32 = acc_i32 - bias
    vpsubd      zmm1, zmm1, zmm3      // corrected lo
    vpsubd      zmm2, zmm2, zmm3      // corrected hi

    // 4. Convert i32 → f32
    vcvtdq2ps   zmm1, zmm1            // f_lo
    vcvtdq2ps   zmm2, zmm2            // f_hi

    // 5. Load w_scales: branch on stride
    cmp         ecx, 1
    jne         .Lavx512_dq_gather

    // Contiguous path (stride == 1): direct loads
    vmovups     zmm4, [rdx]           // ws_lo (16 floats)
    vmovups     zmm5, [rdx + 64]      // ws_hi (16 floats)
    jmp         .Lavx512_dq_fma

.Lavx512_dq_gather:
    // Strided path: gather w_scales using AVX-512 masked gather
    vpbroadcastd zmm6, ecx             // stride
    vmovdqu32   zmm7, [rip + .Lavx512_seq_0_15]
    vmovdqu32   zmm8, [rip + .Lavx512_seq_16_31]
    vpmulld     zmm7, zmm7, zmm6      // idx_lo = [0..15] * stride
    vpmulld     zmm8, zmm8, zmm6      // idx_hi = [16..31] * stride
    // AVX-512 gather: kN mask register, cleared after gather
    kxnorw      k1, k1, k1            // k1 = all-ones mask
    vgatherdps  zmm4 {k1}, [rdx + zmm7 * 4]
    kxnorw      k1, k1, k1            // reset mask
    vgatherdps  zmm5 {k1}, [rdx + zmm8 * 4]

.Lavx512_dq_fma:
    // 6. temp = corrected_f32 * w_scale
    vmulps      zmm1, zmm1, zmm4      // temp_lo
    vmulps      zmm2, zmm2, zmm5      // temp_hi

    // 7. Broadcast x_scale
    vbroadcastss zmm6, xmm0           // x_scale in all lanes

    // 8. FMA: f32_acc += temp * x_scale
    //    vfmadd231ps dst, src1, src2  →  dst = src1 * src2 + dst
    vmovups     zmm7, [r8]            // load f32_acc lo
    vmovups     zmm8, [r8 + 64]       // load f32_acc hi
    vfmadd231ps zmm7, zmm1, zmm6      // f32_acc_lo += temp_lo * x_scale
    vfmadd231ps zmm8, zmm2, zmm6      // f32_acc_hi += temp_hi * x_scale
    vmovups     [r8], zmm7             // store f32_acc lo
    vmovups     [r8 + 64], zmm8        // store f32_acc hi

    ret

    // Sequence constants for gather (aligned)
    .p2align 6
.Lavx512_seq_0_15:
    .long 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
.Lavx512_seq_16_31:
    .long 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31


// ============================================================================
// avx512_silu_gate_tile
//
// Fused SiLU×Gate for one tile (32 f32 lanes):
//   gate[i] = (gate[i] / (1 + exp(-gate[i]))) * up[i]
//
// Args (System V ABI):
//   rdi = gate_ptr : *mut f32  — 32 f32 values, overwritten with result
//   rsi = up_ptr   : *const f32 — 32 f32 values, read-only
//
// SiLU via minimax exp approximation:
//   1. negate: neg = -gate
//   2. clamp:  neg = max(neg, -88.0)  (exp underflow guard)
//   3. range reduce: n = round(neg / ln2), r = neg - n * ln2
//   4. polynomial: p(r) = ((((c6*r + c5)*r + c4)*r + c3)*r + c2)*r + c1)*r + c0
//   5. reconstruct: exp_neg = vscalefps(p, n)
//   6. sigmoid: sig = vrcp14ps(1 + exp_neg) with Newton-Raphson
//   7. result: gate = gate * sig * up
// ============================================================================
    .p2align 4
    .globl SYMBOL(avx512_silu_gate_tile)
SYMBOL(avx512_silu_gate_tile):

    // Load gate[0..15] and gate[16..31]
    vmovups     zmm0, [rdi]            // gate_lo
    vmovups     zmm1, [rdi + 64]       // gate_hi

    // Load up[0..15] and up[16..31]
    vmovups     zmm8, [rsi]            // up_lo
    vmovups     zmm9, [rsi + 64]       // up_hi

    // Negate gate for exp(-gate)
    vxorps      zmm2, zmm0, [rip + .Lavx512_sign_mask]{1to16}  // neg_lo = -gate_lo
    vxorps      zmm3, zmm1, [rip + .Lavx512_sign_mask]{1to16}  // neg_hi = -gate_hi

    // Clamp to -88.0 to avoid exp underflow (exp(-88) ≈ 0)
    vmaxps      zmm2, zmm2, [rip + .Lavx512_neg88]{1to16}
    vmaxps      zmm3, zmm3, [rip + .Lavx512_neg88]{1to16}

    // Range reduction: n = round(neg / ln2)
    vmulps      zmm4, zmm2, [rip + .Lavx512_log2e]{1to16}  // neg / ln2 = neg * log2(e)
    vmulps      zmm5, zmm3, [rip + .Lavx512_log2e]{1to16}
    vrndscaleps zmm4, zmm4, 0          // n_lo = round to nearest
    vrndscaleps zmm5, zmm5, 0          // n_hi = round to nearest

    // r = neg - n * ln2  (using FMA for precision: r = neg - n * ln2 = -(n * ln2 - neg))
    vmovaps     zmm6, zmm2             // r_lo = neg_lo
    vmovaps     zmm7, zmm3             // r_hi = neg_hi
    vfnmadd231ps zmm6, zmm4, [rip + .Lavx512_ln2]{1to16}  // r_lo = neg_lo - n_lo * ln2
    vfnmadd231ps zmm7, zmm5, [rip + .Lavx512_ln2]{1to16}  // r_hi = neg_hi - n_hi * ln2

    // Polynomial: p(r) = ((((c6*r + c5)*r + c4)*r + c3)*r + c2)*r + c1)*r + c0
    // Horner's method
    vbroadcastss zmm10, [rip + .Lavx512_exp_c6]
    vmovaps     zmm2, zmm10            // p_lo = c6
    vmovaps     zmm3, zmm10            // p_hi = c6

    vfmadd213ps zmm2, zmm6, [rip + .Lavx512_exp_c5]{1to16}  // p_lo = c6*r_lo + c5
    vfmadd213ps zmm3, zmm7, [rip + .Lavx512_exp_c5]{1to16}
    vfmadd213ps zmm2, zmm6, [rip + .Lavx512_exp_c4]{1to16}  // p_lo = p*r_lo + c4
    vfmadd213ps zmm3, zmm7, [rip + .Lavx512_exp_c4]{1to16}
    vfmadd213ps zmm2, zmm6, [rip + .Lavx512_exp_c3]{1to16}  // p_lo = p*r_lo + c3
    vfmadd213ps zmm3, zmm7, [rip + .Lavx512_exp_c3]{1to16}
    vfmadd213ps zmm2, zmm6, [rip + .Lavx512_exp_c2]{1to16}  // p_lo = p*r_lo + c2
    vfmadd213ps zmm3, zmm7, [rip + .Lavx512_exp_c2]{1to16}
    vfmadd213ps zmm2, zmm6, [rip + .Lavx512_exp_c1]{1to16}  // p_lo = p*r_lo + c1
    vfmadd213ps zmm3, zmm7, [rip + .Lavx512_exp_c1]{1to16}
    vfmadd213ps zmm2, zmm6, [rip + .Lavx512_one]{1to16}     // p_lo = p*r_lo + 1.0 (c0)
    vfmadd213ps zmm3, zmm7, [rip + .Lavx512_one]{1to16}

    // Reconstruct: exp_neg = p * 2^n = vscalefps(p, n)
    vscalefps   zmm2, zmm2, zmm4       // exp_neg_lo
    vscalefps   zmm3, zmm3, zmm5       // exp_neg_hi

    // sigmoid = 1 / (1 + exp_neg)
    // denom = 1 + exp_neg
    vaddps      zmm4, zmm2, [rip + .Lavx512_one]{1to16}  // denom_lo
    vaddps      zmm5, zmm3, [rip + .Lavx512_one]{1to16}  // denom_hi

    // Approximate reciprocal + 1 Newton-Raphson step
    // x0 = vrcp14ps(denom), then x1 = x0 * (2 - denom * x0)
    vrcp14ps    zmm6, zmm4             // x0_lo ≈ 1/denom_lo
    vrcp14ps    zmm7, zmm5             // x0_hi ≈ 1/denom_hi

    // NR: x1 = x0 * (2 - denom * x0)
    // = 2*x0 - denom*x0*x0
    // Use FMA: tmp = -denom*x0 + 2, then x1 = tmp * x0
    vbroadcastss zmm10, [rip + .Lavx512_two]
    vfnmadd213ps zmm4, zmm6, zmm10    // zmm4 = 2 - denom_lo * x0_lo
    vfnmadd213ps zmm5, zmm7, zmm10    // zmm5 = 2 - denom_hi * x0_hi
    vmulps      zmm6, zmm6, zmm4      // sig_lo = x0_lo * (2 - denom_lo * x0_lo)
    vmulps      zmm7, zmm7, zmm5      // sig_hi = x0_hi * (2 - denom_hi * x0_hi)

    // result = gate * sigmoid * up
    vmulps      zmm0, zmm0, zmm6      // gate_lo * sig_lo
    vmulps      zmm1, zmm1, zmm7      // gate_hi * sig_hi
    vmulps      zmm0, zmm0, zmm8      // * up_lo
    vmulps      zmm1, zmm1, zmm9      // * up_hi

    // Store result
    vmovups     [rdi], zmm0
    vmovups     [rdi + 64], zmm1

    ret

    // Constants for SiLU (aligned)
    .p2align 6
.Lavx512_sign_mask:
    .long 0x80000000                    // sign bit
.Lavx512_neg88:
    .float -88.0
.Lavx512_log2e:
    .float 1.4426950408889634           // log2(e) = 1/ln(2)
.Lavx512_ln2:
    .float 0.6931471805599453           // ln(2)
.Lavx512_one:
    .float 1.0
.Lavx512_two:
    .float 2.0
.Lavx512_exp_c1:
    .float 1.0                          // 1/1!
.Lavx512_exp_c2:
    .float 0.5                          // 1/2!
.Lavx512_exp_c3:
    .float 0.16666666666666666          // 1/3!
.Lavx512_exp_c4:
    .float 0.041666666666666664         // 1/4!
.Lavx512_exp_c5:
    .float 0.008333333333333333         // 1/5!
.Lavx512_exp_c6:
    .float 0.001388888888888889         // 1/6!


#if defined(__linux__) || defined(__FreeBSD__)
    .section .note.GNU-stack,"",@progbits
#endif
