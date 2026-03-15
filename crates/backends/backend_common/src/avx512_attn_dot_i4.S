// AVX-512 dot product: Q_f32 · K_i4_packed with per-group scales → f32 scalar
// Q4_0 group quantization: group_size=32, half-split nibble packing per group.
// SV accumulation for packed i4 V with per-group scales.

.intel_syntax noprefix
#include "avx512_common.inc"

// avx512_attn_dot_qf32_ki4(q: *const f32, k_packed: *const u8, k_scales: *const f32, dim: u64) → xmm0
// rdi = q_ptr, rsi = k_packed_ptr, rdx = k_scales_ptr, rcx = dim (head_dim)
    .globl SYMBOL(avx512_attn_dot_qf32_ki4)
    .p2align 4
SYMBOL(avx512_attn_dot_qf32_ki4):
    // Constants
    vpbroadcastd zmm13, dword ptr [rip + .Li4_mask_0f]
    vpbroadcastd zmm14, dword ptr [rip + .Li4_bias_8]

    vxorps      zmm0, zmm0, zmm0

    cmp         rcx, 128
    jb          .Ldot_i4_loop_init

    // Fully unrolled for dim=128 (4 groups of 32 elements, each with own scale)

    // Group 0: scale[0], 16 packed bytes → 32 elements
    vbroadcastss zmm5, dword ptr [rdx]
    vpmovzxbd   zmm1, xmmword ptr [rsi]
    vpandd      zmm2, zmm1, zmm13
    vpsrld      zmm1, zmm1, 4
    vpandd      zmm3, zmm1, zmm13
    vpsubd      zmm2, zmm2, zmm14
    vpsubd      zmm3, zmm3, zmm14
    vcvtdq2ps   zmm2, zmm2
    vcvtdq2ps   zmm3, zmm3
    vmulps      zmm2, zmm2, zmm5      // dequant: int_val * scale[0]
    vmulps      zmm3, zmm3, zmm5
    vmovups     zmm4, [rdi]
    vfmadd231ps zmm0, zmm2, zmm4
    vmovups     zmm4, [rdi + 64]
    vfmadd231ps zmm0, zmm3, zmm4

    // Group 1: scale[1]
    vbroadcastss zmm5, dword ptr [rdx + 4]
    vpmovzxbd   zmm1, xmmword ptr [rsi + 16]
    vpandd      zmm2, zmm1, zmm13
    vpsrld      zmm1, zmm1, 4
    vpandd      zmm3, zmm1, zmm13
    vpsubd      zmm2, zmm2, zmm14
    vpsubd      zmm3, zmm3, zmm14
    vcvtdq2ps   zmm2, zmm2
    vcvtdq2ps   zmm3, zmm3
    vmulps      zmm2, zmm2, zmm5
    vmulps      zmm3, zmm3, zmm5
    vmovups     zmm4, [rdi + 128]
    vfmadd231ps zmm0, zmm2, zmm4
    vmovups     zmm4, [rdi + 192]
    vfmadd231ps zmm0, zmm3, zmm4

    // Group 2: scale[2]
    vbroadcastss zmm5, dword ptr [rdx + 8]
    vpmovzxbd   zmm1, xmmword ptr [rsi + 32]
    vpandd      zmm2, zmm1, zmm13
    vpsrld      zmm1, zmm1, 4
    vpandd      zmm3, zmm1, zmm13
    vpsubd      zmm2, zmm2, zmm14
    vpsubd      zmm3, zmm3, zmm14
    vcvtdq2ps   zmm2, zmm2
    vcvtdq2ps   zmm3, zmm3
    vmulps      zmm2, zmm2, zmm5
    vmulps      zmm3, zmm3, zmm5
    vmovups     zmm4, [rdi + 256]
    vfmadd231ps zmm0, zmm2, zmm4
    vmovups     zmm4, [rdi + 320]
    vfmadd231ps zmm0, zmm3, zmm4

    // Group 3: scale[3]
    vbroadcastss zmm5, dword ptr [rdx + 12]
    vpmovzxbd   zmm1, xmmword ptr [rsi + 48]
    vpandd      zmm2, zmm1, zmm13
    vpsrld      zmm1, zmm1, 4
    vpandd      zmm3, zmm1, zmm13
    vpsubd      zmm2, zmm2, zmm14
    vpsubd      zmm3, zmm3, zmm14
    vcvtdq2ps   zmm2, zmm2
    vcvtdq2ps   zmm3, zmm3
    vmulps      zmm2, zmm2, zmm5
    vmulps      zmm3, zmm3, zmm5
    vmovups     zmm4, [rdi + 384]
    vfmadd231ps zmm0, zmm2, zmm4
    vmovups     zmm4, [rdi + 448]
    vfmadd231ps zmm0, zmm3, zmm4

    jmp         .Ldot_i4_hsum

.Ldot_i4_loop_init:
    // Generic loop: process 32 elements (16 packed bytes) per iteration with per-group scale
    test        rcx, rcx
    jz          .Ldot_i4_hsum
    // Convert dim to byte limit for Q array (dim * 4)
    mov         r8, rcx
    shl         r8, 2          // r8 = dim * 4 (byte limit for Q)
    xor         r9, r9         // packed byte offset
    xor         r10, r10       // q f32 offset (in bytes)
    xor         r11, r11       // scale index (in bytes, f32 = 4 bytes each)

.Ldot_i4_loop:
    vbroadcastss zmm5, dword ptr [rdx + r11]
    vpmovzxbd   zmm1, xmmword ptr [rsi + r9]
    vpandd      zmm2, zmm1, zmm13
    vpsrld      zmm1, zmm1, 4
    vpandd      zmm3, zmm1, zmm13
    vpsubd      zmm2, zmm2, zmm14
    vpsubd      zmm3, zmm3, zmm14
    vcvtdq2ps   zmm2, zmm2
    vcvtdq2ps   zmm3, zmm3
    vmulps      zmm2, zmm2, zmm5
    vmulps      zmm3, zmm3, zmm5
    vmovups     zmm4, [rdi + r10]
    vfmadd231ps zmm0, zmm2, zmm4
    vmovups     zmm4, [rdi + r10 + 64]
    vfmadd231ps zmm0, zmm3, zmm4
    add         r9, 16         // 16 packed bytes per group
    add         r10, 128       // 32 f32 elements = 128 bytes
    add         r11, 4         // next scale (f32)
    cmp         r10, r8
    jb          .Ldot_i4_loop

.Ldot_i4_hsum:
    // Horizontal sum zmm0 → scalar in xmm0
    vextractf64x4 ymm1, zmm0, 1
    vaddps      ymm0, ymm0, ymm1
    vextractf128  xmm1, ymm0, 1
    vaddps      xmm0, xmm0, xmm1
    vhaddps     xmm0, xmm0, xmm0
    vhaddps     xmm0, xmm0, xmm0
    vzeroupper
    ret


// avx512_attn_sv_accum_i4(out: *mut f32, v_packed: *const u8, v_scales: *const f32, score: f32, dim: u64)
// rdi = out_ptr, rsi = v_packed_ptr, rdx = v_scales_ptr, xmm0 = score, rcx = dim (head_dim)
    .globl SYMBOL(avx512_attn_sv_accum_i4)
    .p2align 4
SYMBOL(avx512_attn_sv_accum_i4):
    // Save score broadcast
    vbroadcastss zmm6, xmm0

    // Constants
    vpbroadcastd zmm13, dword ptr [rip + .Li4_mask_0f]
    vpbroadcastd zmm14, dword ptr [rip + .Li4_bias_8]

    cmp         rcx, 128
    jb          .Lsv_i4_loop_init

    // Fully unrolled for dim=128 (4 groups, each with own scale)
    // weighted = score * v_scale[g]

    // Group 0
    vbroadcastss zmm5, dword ptr [rdx]
    vmulps      zmm5, zmm5, zmm6       // weighted = score * v_scale[0]
    vpmovzxbd   zmm1, xmmword ptr [rsi]
    vpandd      zmm2, zmm1, zmm13
    vpsrld      zmm1, zmm1, 4
    vpandd      zmm3, zmm1, zmm13
    vpsubd      zmm2, zmm2, zmm14
    vpsubd      zmm3, zmm3, zmm14
    vcvtdq2ps   zmm2, zmm2
    vcvtdq2ps   zmm3, zmm3
    vmovups     zmm4, [rdi]
    vfmadd231ps zmm4, zmm2, zmm5
    vmovups     [rdi], zmm4
    vmovups     zmm4, [rdi + 64]
    vfmadd231ps zmm4, zmm3, zmm5
    vmovups     [rdi + 64], zmm4

    // Group 1
    vbroadcastss zmm5, dword ptr [rdx + 4]
    vmulps      zmm5, zmm5, zmm6
    vpmovzxbd   zmm1, xmmword ptr [rsi + 16]
    vpandd      zmm2, zmm1, zmm13
    vpsrld      zmm1, zmm1, 4
    vpandd      zmm3, zmm1, zmm13
    vpsubd      zmm2, zmm2, zmm14
    vpsubd      zmm3, zmm3, zmm14
    vcvtdq2ps   zmm2, zmm2
    vcvtdq2ps   zmm3, zmm3
    vmovups     zmm4, [rdi + 128]
    vfmadd231ps zmm4, zmm2, zmm5
    vmovups     [rdi + 128], zmm4
    vmovups     zmm4, [rdi + 192]
    vfmadd231ps zmm4, zmm3, zmm5
    vmovups     [rdi + 192], zmm4

    // Group 2
    vbroadcastss zmm5, dword ptr [rdx + 8]
    vmulps      zmm5, zmm5, zmm6
    vpmovzxbd   zmm1, xmmword ptr [rsi + 32]
    vpandd      zmm2, zmm1, zmm13
    vpsrld      zmm1, zmm1, 4
    vpandd      zmm3, zmm1, zmm13
    vpsubd      zmm2, zmm2, zmm14
    vpsubd      zmm3, zmm3, zmm14
    vcvtdq2ps   zmm2, zmm2
    vcvtdq2ps   zmm3, zmm3
    vmovups     zmm4, [rdi + 256]
    vfmadd231ps zmm4, zmm2, zmm5
    vmovups     [rdi + 256], zmm4
    vmovups     zmm4, [rdi + 320]
    vfmadd231ps zmm4, zmm3, zmm5
    vmovups     [rdi + 320], zmm4

    // Group 3
    vbroadcastss zmm5, dword ptr [rdx + 12]
    vmulps      zmm5, zmm5, zmm6
    vpmovzxbd   zmm1, xmmword ptr [rsi + 48]
    vpandd      zmm2, zmm1, zmm13
    vpsrld      zmm1, zmm1, 4
    vpandd      zmm3, zmm1, zmm13
    vpsubd      zmm2, zmm2, zmm14
    vpsubd      zmm3, zmm3, zmm14
    vcvtdq2ps   zmm2, zmm2
    vcvtdq2ps   zmm3, zmm3
    vmovups     zmm4, [rdi + 384]
    vfmadd231ps zmm4, zmm2, zmm5
    vmovups     [rdi + 384], zmm4
    vmovups     zmm4, [rdi + 448]
    vfmadd231ps zmm4, zmm3, zmm5
    vmovups     [rdi + 448], zmm4

    vzeroupper
    ret

.Lsv_i4_loop_init:
    test        rcx, rcx
    jz          .Lsv_i4_done
    mov         r8, rcx
    shl         r8, 2          // byte limit
    xor         r9, r9         // packed byte offset
    xor         r10, r10       // out f32 offset (in bytes)
    xor         r11, r11       // scale index (in bytes)

.Lsv_i4_loop:
    vbroadcastss zmm5, dword ptr [rdx + r11]
    vmulps      zmm5, zmm5, zmm6
    vpmovzxbd   zmm1, xmmword ptr [rsi + r9]
    vpandd      zmm2, zmm1, zmm13
    vpsrld      zmm1, zmm1, 4
    vpandd      zmm3, zmm1, zmm13
    vpsubd      zmm2, zmm2, zmm14
    vpsubd      zmm3, zmm3, zmm14
    vcvtdq2ps   zmm2, zmm2
    vcvtdq2ps   zmm3, zmm3
    vmovups     zmm4, [rdi + r10]
    vfmadd231ps zmm4, zmm2, zmm5
    vmovups     [rdi + r10], zmm4
    vmovups     zmm4, [rdi + r10 + 64]
    vfmadd231ps zmm4, zmm3, zmm5
    vmovups     [rdi + r10 + 64], zmm4
    add         r9, 16
    add         r10, 128
    add         r11, 4
    cmp         r10, r8
    jb          .Lsv_i4_loop

.Lsv_i4_done:
    vzeroupper
    ret


// Constants
    .section .rodata
    .p2align 4
.Li4_mask_0f:
    .long 0x0000000F
.Li4_bias_8:
    .long 0x00000008
