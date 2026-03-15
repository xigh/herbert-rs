// 2-pass SV kernel for int8 V cache: dim-major outer loop, 2 d-chunks.
//
// Same structure as BF16 variant but:
//   - V is int8 (vpmovsxbd + vcvtdq2ps for i8->f32)
//   - v_stride = head_dim * 1 (not * 2)
//   - v_scale is already folded into scores by Rust caller
//
// 2-chunks: process 32 dims per outer iteration (2 × 16).
//   Port 5: 8 vbroadcastss = 8 cycles
//   Ports 0+1: 16 FMA + 2 vcvtdq2ps = 9 cycles
//   Port 0: 2 vpmovsxbd (shuffle port) = 2 cycles
//   Bottleneck: ports 0+1 at 9 cycles → 1024 FLOPs / 9 = ~113 FLOP/cycle
//
// Constraint: head_dim must be multiple of 32 (128 is).
//
// Args (System V ABI):
//   rdi = scores    : *const f32  [valid_len * 8], row-major (j*8+h)
//   rsi = v_base    : *const i8   [valid_len * head_dim]
//   rdx = o_base    : *mut f32    [8 * head_dim]
//   rcx = head_dim  : u64         (128, must be multiple of 32)
//   r8  = valid_len : u64

.intel_syntax noprefix
#include "avx512_common.inc"

.text

    .globl SYMBOL(avx512_2pass_sv8_i8)
    .p2align 4
SYMBOL(avx512_2pass_sv8_i8):
    push        rbx
    push        r12
    push        r13
    push        r14
    push        r15
    sub         rsp, 48

    // Save args
    mov         [rsp],    rdi      // scores_base
    mov         [rsp+8],  rsi      // v_base
    mov         [rsp+16], rdx      // o_base
    mov         rbx, rcx           // head_dim (callee-saved)
    mov         r12, r8            // valid_len (callee-saved)

    // Precompute strides
    mov         r13, rbx           // v_stride = head_dim * 1 (i8 bytes)
    mov         r14, rbx
    shl         r14, 2             // o_stride = head_dim * 4 (f32 bytes)

    // Outer loop: d = 0..head_dim step 32
    xor         r15, r15           // d = 0

.L2p_i8_d_loop:
    // Zero 16 accumulators (8 per chunk × 2 chunks)
    // Chunk 1: zmm8-15
    vxorps      zmm8, zmm8, zmm8
    vxorps      zmm9, zmm9, zmm9
    vxorps      zmm10, zmm10, zmm10
    vxorps      zmm11, zmm11, zmm11
    vxorps      zmm12, zmm12, zmm12
    vxorps      zmm13, zmm13, zmm13
    vxorps      zmm14, zmm14, zmm14
    vxorps      zmm15, zmm15, zmm15
    // Chunk 2: zmm16-23
    vxorps      zmm16, zmm16, zmm16
    vxorps      zmm17, zmm17, zmm17
    vxorps      zmm18, zmm18, zmm18
    vxorps      zmm19, zmm19, zmm19
    vxorps      zmm20, zmm20, zmm20
    vxorps      zmm21, zmm21, zmm21
    vxorps      zmm22, zmm22, zmm22
    vxorps      zmm23, zmm23, zmm23

    // Inner loop setup
    mov         rdi, [rsp]         // scores ptr (reset to base)
    mov         rsi, [rsp+8]       // v_base
    // Offset v_base to current d: v_ptr = v_base + d (i8 bytes)
    add         rsi, r15
    xor         rcx, rcx           // j = 0

    test        r12, r12
    jz          .L2p_i8_store

.L2p_i8_j_loop:
    // Prefetch next V row (both chunks within 32 bytes = same cache line)
    lea         rax, [rsi + r13]
    prefetcht0  [rax]

    // Load V[j][d..d+16]: i8 -> i32 -> f32 (chunk 1)
    vpmovsxbd   zmm0, xmmword ptr [rsi]
    vcvtdq2ps   zmm0, zmm0

    // Load V[j][d+16..d+32]: i8 -> i32 -> f32 (chunk 2)
    vpmovsxbd   zmm2, xmmword ptr [rsi + 16]
    vcvtdq2ps   zmm2, zmm2

    // 8x broadcast score[j*8+h] + 2 FMAs per head
    vbroadcastss zmm1, dword ptr [rdi]
    vfmadd231ps zmm8,  zmm0, zmm1
    vfmadd231ps zmm16, zmm2, zmm1

    vbroadcastss zmm1, dword ptr [rdi + 4]
    vfmadd231ps zmm9,  zmm0, zmm1
    vfmadd231ps zmm17, zmm2, zmm1

    vbroadcastss zmm1, dword ptr [rdi + 8]
    vfmadd231ps zmm10, zmm0, zmm1
    vfmadd231ps zmm18, zmm2, zmm1

    vbroadcastss zmm1, dword ptr [rdi + 12]
    vfmadd231ps zmm11, zmm0, zmm1
    vfmadd231ps zmm19, zmm2, zmm1

    vbroadcastss zmm1, dword ptr [rdi + 16]
    vfmadd231ps zmm12, zmm0, zmm1
    vfmadd231ps zmm20, zmm2, zmm1

    vbroadcastss zmm1, dword ptr [rdi + 20]
    vfmadd231ps zmm13, zmm0, zmm1
    vfmadd231ps zmm21, zmm2, zmm1

    vbroadcastss zmm1, dword ptr [rdi + 24]
    vfmadd231ps zmm14, zmm0, zmm1
    vfmadd231ps zmm22, zmm2, zmm1

    vbroadcastss zmm1, dword ptr [rdi + 28]
    vfmadd231ps zmm15, zmm0, zmm1
    vfmadd231ps zmm23, zmm2, zmm1

    // Advance
    add         rdi, 32            // scores += 8 floats (32 bytes)
    add         rsi, r13           // v_ptr += v_stride
    inc         rcx
    cmp         rcx, r12
    jb          .L2p_i8_j_loop

.L2p_i8_store:
    // Store 16 accumulators to o[h*head_dim + d]
    // chunk 1 at [o + h*o_stride + d*4], chunk 2 at [o + h*o_stride + d*4 + 64]
    mov         rdx, [rsp+16]      // o_base
    lea         rax, [r15*4]        // d * 4 (f32 byte offset)

    vmovups     [rdx + rax],      zmm8     // head 0, chunk 1
    vmovups     [rdx + rax + 64], zmm16    // head 0, chunk 2
    add         rdx, r14
    vmovups     [rdx + rax],      zmm9     // head 1, chunk 1
    vmovups     [rdx + rax + 64], zmm17    // head 1, chunk 2
    add         rdx, r14
    vmovups     [rdx + rax],      zmm10    // head 2, chunk 1
    vmovups     [rdx + rax + 64], zmm18    // head 2, chunk 2
    add         rdx, r14
    vmovups     [rdx + rax],      zmm11    // head 3, chunk 1
    vmovups     [rdx + rax + 64], zmm19    // head 3, chunk 2
    add         rdx, r14
    vmovups     [rdx + rax],      zmm12    // head 4, chunk 1
    vmovups     [rdx + rax + 64], zmm20    // head 4, chunk 2
    add         rdx, r14
    vmovups     [rdx + rax],      zmm13    // head 5, chunk 1
    vmovups     [rdx + rax + 64], zmm21    // head 5, chunk 2
    add         rdx, r14
    vmovups     [rdx + rax],      zmm14    // head 6, chunk 1
    vmovups     [rdx + rax + 64], zmm22    // head 6, chunk 2
    add         rdx, r14
    vmovups     [rdx + rax],      zmm15    // head 7, chunk 1
    vmovups     [rdx + rax + 64], zmm23    // head 7, chunk 2

    // Next d chunk pair: d += 32
    add         r15, 32
    cmp         r15, rbx
    jb          .L2p_i8_d_loop

    vzeroupper
    add         rsp, 48
    pop         r15
    pop         r14
    pop         r13
    pop         r12
    pop         rbx
    ret
