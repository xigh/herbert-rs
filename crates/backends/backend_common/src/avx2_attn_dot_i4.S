// AVX2+FMA dot product: Q_f32 · K_i4_packed with per-group scales → f32 scalar
// Q4_0 group quantization: group_size=32, half-split nibble packing per group.
// SV accumulation for packed i4 V with per-group scales.

.intel_syntax noprefix
#include "avx2_common.inc"

// avx2_attn_dot_qf32_ki4(q: *const f32, k_packed: *const u8, k_scales: *const f32, dim: u64) → xmm0
// rdi = q_ptr, rsi = k_packed_ptr, rdx = k_scales_ptr, rcx = dim (head_dim)
    .globl SYMBOL(avx2_attn_dot_qf32_ki4)
    .p2align 4
SYMBOL(avx2_attn_dot_qf32_ki4):
    vxorps      ymm0, ymm0, ymm0

    vpbroadcastd ymm13, dword ptr [rip + .Lavx2_i4_mask_0f]
    vpbroadcastd ymm14, dword ptr [rip + .Lavx2_i4_bias_8]

    cmp         rcx, 128
    jb          .Lavx2_dot_i4_loop_init

    // Fully unrolled for dim=128 (4 groups of 32 elements, each with own scale)
    .irp G, 0, 1, 2, 3
    vbroadcastss ymm5, dword ptr [rdx + \G * 4]

    // Low nibbles: elements [\G*32 .. \G*32+7]
    vpmovzxbd   ymm1, qword ptr [rsi + \G * 16]
    vpand       ymm2, ymm1, ymm13
    vpsubd      ymm2, ymm2, ymm14
    vcvtdq2ps   ymm2, ymm2
    vmulps      ymm2, ymm2, ymm5
    vmovups     ymm3, [rdi + \G * 128]
    vfmadd231ps ymm0, ymm2, ymm3

    // Low nibbles: elements [\G*32+8 .. \G*32+15]
    vpmovzxbd   ymm1, qword ptr [rsi + \G * 16 + 8]
    vpand       ymm2, ymm1, ymm13
    vpsubd      ymm2, ymm2, ymm14
    vcvtdq2ps   ymm2, ymm2
    vmulps      ymm2, ymm2, ymm5
    vmovups     ymm3, [rdi + \G * 128 + 32]
    vfmadd231ps ymm0, ymm2, ymm3

    // High nibbles: elements [\G*32+16 .. \G*32+23]
    vpmovzxbd   ymm1, qword ptr [rsi + \G * 16]
    vpsrld      ymm1, ymm1, 4
    vpand       ymm2, ymm1, ymm13
    vpsubd      ymm2, ymm2, ymm14
    vcvtdq2ps   ymm2, ymm2
    vmulps      ymm2, ymm2, ymm5
    vmovups     ymm3, [rdi + \G * 128 + 64]
    vfmadd231ps ymm0, ymm2, ymm3

    // High nibbles: elements [\G*32+24 .. \G*32+31]
    vpmovzxbd   ymm1, qword ptr [rsi + \G * 16 + 8]
    vpsrld      ymm1, ymm1, 4
    vpand       ymm2, ymm1, ymm13
    vpsubd      ymm2, ymm2, ymm14
    vcvtdq2ps   ymm2, ymm2
    vmulps      ymm2, ymm2, ymm5
    vmovups     ymm3, [rdi + \G * 128 + 96]
    vfmadd231ps ymm0, ymm2, ymm3
    .endr

    jmp         .Lavx2_dot_i4_hsum

.Lavx2_dot_i4_loop_init:
    test        rcx, rcx
    jz          .Lavx2_dot_i4_hsum
    mov         r8, rcx
    shl         r8, 2          // byte limit
    xor         r9, r9         // packed offset
    xor         r10, r10       // q byte offset
    xor         r11, r11       // scale byte offset

.Lavx2_dot_i4_loop:
    vbroadcastss ymm5, dword ptr [rdx + r11]

    vpmovzxbd   ymm1, qword ptr [rsi + r9]
    vpand       ymm2, ymm1, ymm13
    vpsubd      ymm2, ymm2, ymm14
    vcvtdq2ps   ymm2, ymm2
    vmulps      ymm2, ymm2, ymm5
    vmovups     ymm3, [rdi + r10]
    vfmadd231ps ymm0, ymm2, ymm3

    vpmovzxbd   ymm1, qword ptr [rsi + r9 + 8]
    vpand       ymm2, ymm1, ymm13
    vpsubd      ymm2, ymm2, ymm14
    vcvtdq2ps   ymm2, ymm2
    vmulps      ymm2, ymm2, ymm5
    vmovups     ymm3, [rdi + r10 + 32]
    vfmadd231ps ymm0, ymm2, ymm3

    vpmovzxbd   ymm1, qword ptr [rsi + r9]
    vpsrld      ymm1, ymm1, 4
    vpand       ymm2, ymm1, ymm13
    vpsubd      ymm2, ymm2, ymm14
    vcvtdq2ps   ymm2, ymm2
    vmulps      ymm2, ymm2, ymm5
    vmovups     ymm3, [rdi + r10 + 64]
    vfmadd231ps ymm0, ymm2, ymm3

    vpmovzxbd   ymm1, qword ptr [rsi + r9 + 8]
    vpsrld      ymm1, ymm1, 4
    vpand       ymm2, ymm1, ymm13
    vpsubd      ymm2, ymm2, ymm14
    vcvtdq2ps   ymm2, ymm2
    vmulps      ymm2, ymm2, ymm5
    vmovups     ymm3, [rdi + r10 + 96]
    vfmadd231ps ymm0, ymm2, ymm3

    add         r9, 16
    add         r10, 128
    add         r11, 4
    cmp         r10, r8
    jb          .Lavx2_dot_i4_loop

.Lavx2_dot_i4_hsum:
    vextractf128  xmm1, ymm0, 1
    vaddps      xmm0, xmm0, xmm1
    vhaddps     xmm0, xmm0, xmm0
    vhaddps     xmm0, xmm0, xmm0
    vzeroupper
    ret


// avx2_attn_sv_accum_i4(out: *mut f32, v_packed: *const u8, v_scales: *const f32, score: f32, dim: u64)
// rdi = out_ptr, rsi = v_packed_ptr, rdx = v_scales_ptr, xmm0 = score, rcx = dim (head_dim)
    .globl SYMBOL(avx2_attn_sv_accum_i4)
    .p2align 4
SYMBOL(avx2_attn_sv_accum_i4):
    vbroadcastss ymm6, xmm0   // score broadcast

    vpbroadcastd ymm13, dword ptr [rip + .Lavx2_i4_mask_0f]
    vpbroadcastd ymm14, dword ptr [rip + .Lavx2_i4_bias_8]

    cmp         rcx, 128
    jb          .Lavx2_sv_i4_loop_init

    // Fully unrolled for dim=128
    .irp G, 0, 1, 2, 3
    vbroadcastss ymm5, dword ptr [rdx + \G * 4]
    vmulps      ymm5, ymm5, ymm6      // weighted = score * v_scale[G]

    vpmovzxbd   ymm1, qword ptr [rsi + \G * 16]
    vpand       ymm2, ymm1, ymm13
    vpsubd      ymm2, ymm2, ymm14
    vcvtdq2ps   ymm2, ymm2
    vmovups     ymm3, [rdi + \G * 128]
    vfmadd231ps ymm3, ymm2, ymm5
    vmovups     [rdi + \G * 128], ymm3

    vpmovzxbd   ymm1, qword ptr [rsi + \G * 16 + 8]
    vpand       ymm2, ymm1, ymm13
    vpsubd      ymm2, ymm2, ymm14
    vcvtdq2ps   ymm2, ymm2
    vmovups     ymm3, [rdi + \G * 128 + 32]
    vfmadd231ps ymm3, ymm2, ymm5
    vmovups     [rdi + \G * 128 + 32], ymm3

    vpmovzxbd   ymm1, qword ptr [rsi + \G * 16]
    vpsrld      ymm1, ymm1, 4
    vpand       ymm2, ymm1, ymm13
    vpsubd      ymm2, ymm2, ymm14
    vcvtdq2ps   ymm2, ymm2
    vmovups     ymm3, [rdi + \G * 128 + 64]
    vfmadd231ps ymm3, ymm2, ymm5
    vmovups     [rdi + \G * 128 + 64], ymm3

    vpmovzxbd   ymm1, qword ptr [rsi + \G * 16 + 8]
    vpsrld      ymm1, ymm1, 4
    vpand       ymm2, ymm1, ymm13
    vpsubd      ymm2, ymm2, ymm14
    vcvtdq2ps   ymm2, ymm2
    vmovups     ymm3, [rdi + \G * 128 + 96]
    vfmadd231ps ymm3, ymm2, ymm5
    vmovups     [rdi + \G * 128 + 96], ymm3
    .endr

    vzeroupper
    ret

.Lavx2_sv_i4_loop_init:
    test        rcx, rcx
    jz          .Lavx2_sv_i4_done
    mov         r8, rcx
    shl         r8, 2
    xor         r9, r9
    xor         r10, r10
    xor         r11, r11

.Lavx2_sv_i4_loop:
    vbroadcastss ymm5, dword ptr [rdx + r11]
    vmulps      ymm5, ymm5, ymm6

    vpmovzxbd   ymm1, qword ptr [rsi + r9]
    vpand       ymm2, ymm1, ymm13
    vpsubd      ymm2, ymm2, ymm14
    vcvtdq2ps   ymm2, ymm2
    vmovups     ymm3, [rdi + r10]
    vfmadd231ps ymm3, ymm2, ymm5
    vmovups     [rdi + r10], ymm3

    vpmovzxbd   ymm1, qword ptr [rsi + r9 + 8]
    vpand       ymm2, ymm1, ymm13
    vpsubd      ymm2, ymm2, ymm14
    vcvtdq2ps   ymm2, ymm2
    vmovups     ymm3, [rdi + r10 + 32]
    vfmadd231ps ymm3, ymm2, ymm5
    vmovups     [rdi + r10 + 32], ymm3

    vpmovzxbd   ymm1, qword ptr [rsi + r9]
    vpsrld      ymm1, ymm1, 4
    vpand       ymm2, ymm1, ymm13
    vpsubd      ymm2, ymm2, ymm14
    vcvtdq2ps   ymm2, ymm2
    vmovups     ymm3, [rdi + r10 + 64]
    vfmadd231ps ymm3, ymm2, ymm5
    vmovups     [rdi + r10 + 64], ymm3

    vpmovzxbd   ymm1, qword ptr [rsi + r9 + 8]
    vpsrld      ymm1, ymm1, 4
    vpand       ymm2, ymm1, ymm13
    vpsubd      ymm2, ymm2, ymm14
    vcvtdq2ps   ymm2, ymm2
    vmovups     ymm3, [rdi + r10 + 96]
    vfmadd231ps ymm3, ymm2, ymm5
    vmovups     [rdi + r10 + 96], ymm3

    add         r9, 16
    add         r10, 128
    add         r11, 4
    cmp         r10, r8
    jb          .Lavx2_sv_i4_loop

.Lavx2_sv_i4_done:
    vzeroupper
    ret


// Constants
    .section .rodata
    .p2align 4
.Lavx2_i4_mask_0f:
    .long 0x0000000F
.Lavx2_i4_bias_8:
    .long 0x00000008
