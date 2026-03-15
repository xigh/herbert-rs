// AVX2 Q4 assembly kernels for Qwen3 inference — pre-interleaved layout
//
// Emulates VPDPBUSD (AVX-512 VNNI) using AVX2 instructions with 4-bit
// symmetric quantization.
//
// Weight layout: [n_tiles, K/4, 16, 4] (pre-interleaved)
//   Per K-group, 16 half-lanes × 4 k-values:
//     byte[4*hl + j] = nib(lane_hl, kj) | (nib(lane_hl+16, kj) << 4)
//   Low nibble = lanes 0-15, high nibble = lanes 16-31.
//   16 × 4 = 64 bytes per K-group.
//
// Unpacking (no vpunpck needed):
//   First 32 bytes (hl 0-7):
//     vpand 0x0F → lanes 0-7 (ymm0)
//     vpsrld+vpand → lanes 16-23 (ymm1)
//   Second 32 bytes (hl 8-15):
//     vpand 0x0F → lanes 8-15 (ymm2)
//     vpsrld+vpand → lanes 24-31 (ymm3)
//
// VPDPBUSD emulation (3 instructions):
//   vpmaddubsw ymm_tmp, ymm_u8, ymm_i8  — u8 * i8 → i16 pairs
//   vpmaddwd   ymm_tmp, ymm_tmp, ymm_ones — i16 pairs → i32
//   vpaddd     ymm_acc, ymm_acc, ymm_tmp  — accumulate
//
// No saturation risk: max |value| = 15 * 127 * 2 = 3810 < 32767.
//
// Output lane mapping (sequential, no scramble):
//   ymm0: lanes [0..7]
//   ymm1: lanes [16..23]
//   ymm2: lanes [8..15]
//   ymm3: lanes [24..31]

.intel_syntax noprefix

#ifdef __APPLE__
#define SYMBOL(name) _##name
#else
#define SYMBOL(name) name
#endif

// ============================================================================
// avx2_q4_matvec_tile
//
// Compute y[0..31] = W_q4[tile] @ x_i8 for a single 32-lane tile.
//
// Args (System V ABI):
//   rdi = x_i8      : pointer to i8 input vector (K elements, quantized)
//   rsi = w_packed   : pointer to u8 packed nibble weights (pre-interleaved)
//   rdx = k_groups   : number of K-groups (K/4)
//   rcx = acc_out    : pointer to i32[32] output accumulator
//
// Register allocation (16 YMM registers):
//   ymm0-ymm3:  4 i32 accumulators (4 × 8 = 32 lanes)
//   ymm4:       0x0F0F0F0F mask constant
//   ymm5:       0x00010001 ones constant for vpmaddwd
//   ymm6:       packed data load (temp)
//   ymm7:       low nibbles (temp)
//   ymm8:       high nibbles (temp)
//   ymm11:      broadcast x_i8 dword (temp)
//   ymm12:      vpmaddubsw/vpmaddwd result (temp)
//
// Output lane order:
//   ymm0: lanes [ 0, 1, 2, 3,  4,  5,  6,  7]
//   ymm1: lanes [16,17,18,19, 20, 21, 22, 23]
//   ymm2: lanes [ 8, 9,10,11, 12, 13, 14, 15]
//   ymm3: lanes [24,25,26,27, 28, 29, 30, 31]
//
// ============================================================================
    .globl SYMBOL(avx2_q4_matvec_tile)
    .p2align 4
SYMBOL(avx2_q4_matvec_tile):
    // Zero accumulators
    vpxor       ymm0, ymm0, ymm0
    vpxor       ymm1, ymm1, ymm1
    vpxor       ymm2, ymm2, ymm2
    vpxor       ymm3, ymm3, ymm3

    test        rdx, rdx
    jz          .Lavx2_mv_store

    mov         r8, rdx

    // Prepare mask constant: 0x0F0F0F0F in ymm4
    mov         eax, 0x0F0F0F0F
    vmovd       xmm4, eax
    vpbroadcastd ymm4, xmm4

    // Prepare ones constant: 0x00010001 in ymm5 for vpmaddwd
    mov         eax, 0x00010001
    vmovd       xmm5, eax
    vpbroadcastd ymm5, xmm5

    // 2x unrolled loop: process 2 K-groups (128 bytes weights) per iteration.
    mov         r9, r8
    shr         r9, 1                       // r9 = k_groups / 2
    test        r9, r9
    jz          .Lavx2_mv_tail

    .p2align 4
.Lavx2_mv_kgroup_loop_2x:
    // --- K-group 0 ---
    vpbroadcastd ymm11, dword ptr [rdi]

    // First 32 bytes: low nibbles = lanes 0-7, high nibbles = lanes 16-23
    vmovdqu     ymm6, [rsi]
    vpand       ymm7, ymm6, ymm4           // lanes 0-7, ready
    vpsrld      ymm8, ymm6, 4
    vpand       ymm8, ymm8, ymm4           // lanes 16-23, ready
    vpmaddubsw  ymm12, ymm7, ymm11
    vpmaddwd    ymm12, ymm12, ymm5
    vpaddd      ymm0, ymm0, ymm12          // acc lanes 0-7
    vpmaddubsw  ymm12, ymm8, ymm11
    vpmaddwd    ymm12, ymm12, ymm5
    vpaddd      ymm1, ymm1, ymm12          // acc lanes 16-23

    // Second 32 bytes: low nibbles = lanes 8-15, high nibbles = lanes 24-31
    vmovdqu     ymm6, [rsi + 32]
    vpand       ymm7, ymm6, ymm4           // lanes 8-15, ready
    vpsrld      ymm8, ymm6, 4
    vpand       ymm8, ymm8, ymm4           // lanes 24-31, ready
    vpmaddubsw  ymm12, ymm7, ymm11
    vpmaddwd    ymm12, ymm12, ymm5
    vpaddd      ymm2, ymm2, ymm12          // acc lanes 8-15
    vpmaddubsw  ymm12, ymm8, ymm11
    vpmaddwd    ymm12, ymm12, ymm5
    vpaddd      ymm3, ymm3, ymm12          // acc lanes 24-31

    // --- K-group 1 ---
    vpbroadcastd ymm11, dword ptr [rdi + 4]

    vmovdqu     ymm6, [rsi + 64]
    vpand       ymm7, ymm6, ymm4
    vpsrld      ymm8, ymm6, 4
    vpand       ymm8, ymm8, ymm4
    vpmaddubsw  ymm12, ymm7, ymm11
    vpmaddwd    ymm12, ymm12, ymm5
    vpaddd      ymm0, ymm0, ymm12
    vpmaddubsw  ymm12, ymm8, ymm11
    vpmaddwd    ymm12, ymm12, ymm5
    vpaddd      ymm1, ymm1, ymm12

    vmovdqu     ymm6, [rsi + 96]
    vpand       ymm7, ymm6, ymm4
    vpsrld      ymm8, ymm6, 4
    vpand       ymm8, ymm8, ymm4
    vpmaddubsw  ymm12, ymm7, ymm11
    vpmaddwd    ymm12, ymm12, ymm5
    vpaddd      ymm2, ymm2, ymm12
    vpmaddubsw  ymm12, ymm8, ymm11
    vpmaddwd    ymm12, ymm12, ymm5
    vpaddd      ymm3, ymm3, ymm12

    // Advance pointers
    add         rsi, 128                    // 2 K-groups = 128 packed bytes
    add         rdi, 8                      // 2 K-groups = 8 x values
    dec         r9
    jnz         .Lavx2_mv_kgroup_loop_2x

.Lavx2_mv_tail:
    // Handle odd K-group remainder
    test        r8, 1
    jz          .Lavx2_mv_store

    vpbroadcastd ymm11, dword ptr [rdi]

    // First 32 bytes
    vmovdqu     ymm6, [rsi]
    vpand       ymm7, ymm6, ymm4
    vpsrld      ymm8, ymm6, 4
    vpand       ymm8, ymm8, ymm4
    vpmaddubsw  ymm12, ymm7, ymm11
    vpmaddwd    ymm12, ymm12, ymm5
    vpaddd      ymm0, ymm0, ymm12
    vpmaddubsw  ymm12, ymm8, ymm11
    vpmaddwd    ymm12, ymm12, ymm5
    vpaddd      ymm1, ymm1, ymm12

    // Second 32 bytes
    vmovdqu     ymm6, [rsi + 32]
    vpand       ymm7, ymm6, ymm4
    vpsrld      ymm8, ymm6, 4
    vpand       ymm8, ymm8, ymm4
    vpmaddubsw  ymm12, ymm7, ymm11
    vpmaddwd    ymm12, ymm12, ymm5
    vpaddd      ymm2, ymm2, ymm12
    vpmaddubsw  ymm12, ymm8, ymm11
    vpmaddwd    ymm12, ymm12, ymm5
    vpaddd      ymm3, ymm3, ymm12

.Lavx2_mv_store:
    // Store i32 accumulators (32 bytes per YMM = 8 i32)
    vmovdqu     [rcx], ymm0
    vmovdqu     [rcx + 32], ymm1
    vmovdqu     [rcx + 64], ymm2
    vmovdqu     [rcx + 96], ymm3

    vzeroupper
    ret


// ============================================================================
// avx2_q4_dequant_tile_fma
//
// Fused dequantization for one 32-lane tile with FMA accumulation (AVX2).
// The 32 lanes are split into 4 YMM registers (8 lanes each).
//
// Input acc_i32 layout (from avx2_q4_matvec_tile, pre-interleaved):
//   acc[0..7]   = lanes 0-7
//   acc[8..15]  = lanes 16-23  (needs swap with acc[16..23])
//   acc[16..23] = lanes 8-15   (needs swap with acc[8..15])
//   acc[24..31] = lanes 24-31
//
// Computes for each lane n in [0..31]:
//   corrected = acc_i32[n] - 8 * col_sum
//   f32_acc[n] += corrected_f32 * w_scale[n] * x_scale
//
// Args (System V ABI):
//   rdi = acc_i32         : pointer to i32[32]
//   rsi = col_sum         : i64
//   rdx = w_scales_base   : pointer to f32 scales
//   ecx = w_scales_stride : i32 (1 = contiguous)
//   xmm0 = x_scale        : f32
//   r8  = f32_acc         : pointer to f32[32] accumulator
//
// ============================================================================
    .globl SYMBOL(avx2_q4_dequant_tile_fma)
    .p2align 4
SYMBOL(avx2_q4_dequant_tile_fma):
    // 1. Load acc_i32[32] as 4 YMM, then swap v1/v2 for sequential order
    vmovdqu     ymm1, [rdi]           // v0 = lanes 0-7
    vmovdqu     ymm3, [rdi + 32]      // v1 = lanes 16-23 (will become ordered_2)
    vmovdqu     ymm2, [rdi + 64]      // v2 = lanes 8-15  (will become ordered_1)
    vmovdqu     ymm4, [rdi + 96]      // v3 = lanes 24-31
    // After: ymm1=lanes0-7, ymm2=lanes8-15, ymm3=lanes16-23, ymm4=lanes24-31

    // 2. Compute bias = (i32)(8 * col_sum), broadcast
    imul        rsi, 8
    mov         eax, esi               // truncate to 32-bit
    vmovd       xmm5, eax
    vpbroadcastd ymm5, xmm5           // bias in all lanes

    // 3. Subtract bias
    vpsubd      ymm1, ymm1, ymm5
    vpsubd      ymm2, ymm2, ymm5
    vpsubd      ymm3, ymm3, ymm5
    vpsubd      ymm4, ymm4, ymm5

    // 4. Convert i32 → f32
    vcvtdq2ps   ymm1, ymm1            // f_0 (lanes 0-7)
    vcvtdq2ps   ymm2, ymm2            // f_1 (lanes 8-15)
    vcvtdq2ps   ymm3, ymm3            // f_2 (lanes 16-23)
    vcvtdq2ps   ymm4, ymm4            // f_3 (lanes 24-31)

    // 5. Load w_scales: branch on stride
    cmp         ecx, 1
    jne         .Lavx2_dq_gather

    // Contiguous path
    vmovups     ymm5, [rdx]           // ws_0
    vmovups     ymm6, [rdx + 32]      // ws_1
    vmovups     ymm7, [rdx + 64]      // ws_2
    vmovups     ymm8, [rdx + 96]      // ws_3
    jmp         .Lavx2_dq_fma

.Lavx2_dq_gather:
    // Strided path: gather w_scales with non-unit stride
    // AVX2 vgatherdps requires all-ones mask, cleared after gather
    vmovd       xmm9, ecx
    vpbroadcastd ymm9, xmm9           // stride broadcast

    vpcmpeqd    ymm11, ymm11, ymm11   // all-ones mask
    vmovdqu     ymm10, [rip + .Lavx2_seq_0_7]
    vpmulld     ymm10, ymm10, ymm9
    vgatherdps  ymm5, [rdx + ymm10 * 4], ymm11

    vpcmpeqd    ymm11, ymm11, ymm11   // reset mask
    vmovdqu     ymm10, [rip + .Lavx2_seq_8_15]
    vpmulld     ymm10, ymm10, ymm9
    vgatherdps  ymm6, [rdx + ymm10 * 4], ymm11

    vpcmpeqd    ymm11, ymm11, ymm11
    vmovdqu     ymm10, [rip + .Lavx2_seq_16_23]
    vpmulld     ymm10, ymm10, ymm9
    vgatherdps  ymm7, [rdx + ymm10 * 4], ymm11

    vpcmpeqd    ymm11, ymm11, ymm11
    vmovdqu     ymm10, [rip + .Lavx2_seq_24_31]
    vpmulld     ymm10, ymm10, ymm9
    vgatherdps  ymm8, [rdx + ymm10 * 4], ymm11

.Lavx2_dq_fma:
    // 6. temp = corrected_f32 * w_scale
    vmulps      ymm1, ymm1, ymm5      // temp_0
    vmulps      ymm2, ymm2, ymm6      // temp_1
    vmulps      ymm3, ymm3, ymm7      // temp_2
    vmulps      ymm4, ymm4, ymm8      // temp_3

    // 7. Broadcast x_scale
    vbroadcastss ymm9, xmm0           // x_scale in all lanes

    // 8. FMA: f32_acc += temp * x_scale
    vmovups     ymm5, [r8]
    vmovups     ymm6, [r8 + 32]
    vmovups     ymm7, [r8 + 64]
    vmovups     ymm8, [r8 + 96]
    vfmadd231ps ymm5, ymm1, ymm9      // f32_acc[0..7]  += temp_0 * x_scale
    vfmadd231ps ymm6, ymm2, ymm9      // f32_acc[8..15] += temp_1 * x_scale
    vfmadd231ps ymm7, ymm3, ymm9      // f32_acc[16..23]+= temp_2 * x_scale
    vfmadd231ps ymm8, ymm4, ymm9      // f32_acc[24..31]+= temp_3 * x_scale
    vmovups     [r8], ymm5
    vmovups     [r8 + 32], ymm6
    vmovups     [r8 + 64], ymm7
    vmovups     [r8 + 96], ymm8

    vzeroupper
    ret

    // Sequence constants for gather
    .p2align 5
.Lavx2_seq_0_7:
    .long 0, 1, 2, 3, 4, 5, 6, 7
.Lavx2_seq_8_15:
    .long 8, 9, 10, 11, 12, 13, 14, 15
.Lavx2_seq_16_23:
    .long 16, 17, 18, 19, 20, 21, 22, 23
.Lavx2_seq_24_31:
    .long 24, 25, 26, 27, 28, 29, 30, 31


// ============================================================================
// avx2_silu_gate_tile
//
// Fused SiLU×Gate for one tile (32 f32 lanes) using AVX2:
//   gate[i] = (gate[i] / (1 + exp(-gate[i]))) * up[i]
//
// Args (System V ABI):
//   rdi = gate_ptr : *mut f32  — 32 f32 values, overwritten with result
//   rsi = up_ptr   : *const f32 — 32 f32 values, read-only
//
// Processes 4 blocks of 8 lanes (ymm registers).
// ============================================================================
    .p2align 4
    .globl SYMBOL(avx2_silu_gate_tile)
SYMBOL(avx2_silu_gate_tile):

    // Process 4 blocks of 8 lanes each.
    // We loop over offsets 0, 32, 64, 96 (bytes) = 0, 8, 16, 24 (floats).
    xor     ecx, ecx                   // offset = 0

.Lavx2_silu_loop:
    // Load gate[offset..offset+7] and up[offset..offset+7]
    vmovups     ymm0, [rdi + rcx]      // gate
    vmovups     ymm7, [rsi + rcx]      // up

    // Negate gate for exp(-gate)
    vxorps      ymm1, ymm0, [rip + .Lavx2_sign_mask]  // neg = -gate

    // Clamp to [-88.0, +88.0] to avoid exp overflow/underflow
    // (AVX2 ldexp trick overflows IEEE754 exponent for |n| > 127)
    vmaxps      ymm1, ymm1, [rip + .Lavx2_neg88]
    vminps      ymm1, ymm1, [rip + .Lavx2_pos88]

    // Range reduction: n = round(neg / ln2)
    vmulps      ymm2, ymm1, [rip + .Lavx2_log2e]   // neg * log2(e)
    vroundps    ymm2, ymm2, 0          // n = round to nearest

    // r = neg - n * ln2
    vmovaps     ymm3, ymm1             // r = neg
    vfnmadd231ps ymm3, ymm2, [rip + .Lavx2_ln2]  // r = neg - n * ln2

    // Polynomial: p(r) = ((((c6*r + c5)*r + c4)*r + c3)*r + c2)*r + c1)*r + 1.0
    vbroadcastss ymm4, [rip + .Lavx2_exp_c6]
    vfmadd213ps ymm4, ymm3, [rip + .Lavx2_exp_c5]    // c6*r + c5
    vfmadd213ps ymm4, ymm3, [rip + .Lavx2_exp_c4]    // p*r + c4
    vfmadd213ps ymm4, ymm3, [rip + .Lavx2_exp_c3]    // p*r + c3
    vfmadd213ps ymm4, ymm3, [rip + .Lavx2_exp_c2]    // p*r + c2
    vfmadd213ps ymm4, ymm3, [rip + .Lavx2_exp_c1]    // p*r + c1
    vfmadd213ps ymm4, ymm3, [rip + .Lavx2_one]        // p*r + 1.0

    // Reconstruct: exp_neg = p * 2^n
    // AVX2 has no vscalefps, use manual ldexp: exp_neg = p * exp2(n)
    // exp2(n) via integer trick: convert n to int, shift into exponent field
    vcvtps2dq   ymm5, ymm2            // n as int32
    vpslld      ymm5, ymm5, 23        // shift into IEEE754 exponent position
    vpaddd      ymm5, ymm5, [rip + .Lavx2_one_int]  // add 1.0 exponent (127 << 23)
    vmulps      ymm4, ymm4, ymm5      // exp_neg = p * 2^n

    // sigmoid = 1 / (1 + exp_neg)
    vaddps      ymm5, ymm4, [rip + .Lavx2_one]       // denom = 1 + exp_neg

    // Approximate reciprocal + 1 Newton-Raphson step
    vrcpps      ymm6, ymm5            // x0 ≈ 1/denom (~12-bit)
    // NR: x1 = x0 * (2 - denom * x0)
    vmovaps     ymm4, [rip + .Lavx2_two]
    vfnmadd213ps ymm5, ymm6, ymm4     // ymm5 = 2 - denom * x0
    vmulps      ymm6, ymm6, ymm5      // sig = x0 * (2 - denom * x0)

    // result = gate * sigmoid * up
    vmulps      ymm0, ymm0, ymm6      // gate * sig
    vmulps      ymm0, ymm0, ymm7      // * up

    // Store result
    vmovups     [rdi + rcx], ymm0

    add         ecx, 32               // advance 8 floats = 32 bytes
    cmp         ecx, 128              // 32 floats * 4 bytes = 128
    jb          .Lavx2_silu_loop

    vzeroupper
    ret

    // Constants for AVX2 SiLU (aligned, 256-bit)
    .p2align 5
.Lavx2_sign_mask:
    .long 0x80000000, 0x80000000, 0x80000000, 0x80000000
    .long 0x80000000, 0x80000000, 0x80000000, 0x80000000
.Lavx2_neg88:
    .float -88.0, -88.0, -88.0, -88.0, -88.0, -88.0, -88.0, -88.0
.Lavx2_pos88:
    .float 88.0, 88.0, 88.0, 88.0, 88.0, 88.0, 88.0, 88.0
.Lavx2_log2e:
    .float 1.4426950408889634, 1.4426950408889634, 1.4426950408889634, 1.4426950408889634
    .float 1.4426950408889634, 1.4426950408889634, 1.4426950408889634, 1.4426950408889634
.Lavx2_ln2:
    .float 0.6931471805599453, 0.6931471805599453, 0.6931471805599453, 0.6931471805599453
    .float 0.6931471805599453, 0.6931471805599453, 0.6931471805599453, 0.6931471805599453
.Lavx2_one:
    .float 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
.Lavx2_two:
    .float 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0
.Lavx2_one_int:
    .long 0x3F800000, 0x3F800000, 0x3F800000, 0x3F800000
    .long 0x3F800000, 0x3F800000, 0x3F800000, 0x3F800000
.Lavx2_exp_c1:
    .float 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
.Lavx2_exp_c2:
    .float 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5
.Lavx2_exp_c3:
    .float 0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666
    .float 0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666
.Lavx2_exp_c4:
    .float 0.041666666666666664, 0.041666666666666664, 0.041666666666666664, 0.041666666666666664
    .float 0.041666666666666664, 0.041666666666666664, 0.041666666666666664, 0.041666666666666664
.Lavx2_exp_c5:
    .float 0.008333333333333333, 0.008333333333333333, 0.008333333333333333, 0.008333333333333333
    .float 0.008333333333333333, 0.008333333333333333, 0.008333333333333333, 0.008333333333333333
.Lavx2_exp_c6:
    .float 0.001388888888888889, 0.001388888888888889, 0.001388888888888889, 0.001388888888888889
    .float 0.001388888888888889, 0.001388888888888889, 0.001388888888888889, 0.001388888888888889


#if defined(__linux__) || defined(__FreeBSD__)
    .section .note.GNU-stack,"",@progbits
#endif
