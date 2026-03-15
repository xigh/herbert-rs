// 2-pass SV kernel for BF16 V cache: dim-major outer loop, 2 d-chunks.
//
// After pass 1 (DOT + softmax in Rust), this kernel performs:
//   for d in 0..head_dim step 32:           // outer: 2 chunks of 16 f32
//     acc1[0..7] = 0                        // 8 head accus for d..d+16 (zmm8-15)
//     acc2[0..7] = 0                        // 8 head accus for d+16..d+32 (zmm16-23)
//     for j in 0..valid_len:                // inner: positions
//       v1 = bf16_to_f32(V[j][d..d+16])    // load + convert chunk 1
//       v2 = bf16_to_f32(V[j][d+16..d+32]) // load + convert chunk 2
//       8x: acc1[h] += score[j*8+h] * v1   // 8 broadcasts + 16 FMAs
//            acc2[h] += score[j*8+h] * v2
//     store acc1/acc2
//
// Port analysis (BF16, inner loop per j):
//   Port 5: 8 vbroadcastss + 2 vpmovzxwd = 10 cycles (bottleneck)
//   Ports 0+1: 16 vfmadd231ps / 2 + 2 vpslld / 2 = 9 cycles
//   → 1024 FLOPs / 10 cycles = 102.4 FLOP/cycle = ~80% peak (vs 72% 1-chunk)
//
// Constraint: head_dim must be multiple of 32 (128 is).
//
// Args (System V ABI):
//   rdi = scores    : *const f32  [valid_len * 8], row-major (j*8+h)
//   rsi = v_base    : *const u16  [valid_len * head_dim], BF16
//   rdx = o_base    : *mut f32    [8 * head_dim]
//   rcx = head_dim  : u64         (128, must be multiple of 32)
//   r8  = valid_len : u64

.intel_syntax noprefix
#include "avx512_common.inc"

.text

    .globl SYMBOL(avx512_2pass_sv8_bf16)
    .p2align 4
SYMBOL(avx512_2pass_sv8_bf16):
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
    mov         r13, rbx
    shl         r13, 1             // v_stride = head_dim * 2 (BF16 bytes)
    mov         r14, rbx
    shl         r14, 2             // o_stride = head_dim * 4 (f32 bytes)

    // Outer loop: d = 0..head_dim step 32
    xor         r15, r15           // d = 0

.L2p_bf16_d_loop:
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
    // Offset v_base to current d: v_ptr = v_base + d * 2 (BF16)
    lea         rsi, [rsi + r15*2]
    xor         rcx, rcx           // j = 0

    test        r12, r12
    jz          .L2p_bf16_store

.L2p_bf16_j_loop:
    // Prefetch next V row (both chunks in same cache line for aligned data)
    lea         rax, [rsi + r13]
    prefetcht0  [rax]
    prefetcht0  [rax + 32]

    // Load V[j][d..d+16]: BF16 → f32 (chunk 1)
    vpmovzxwd   zmm0, ymmword ptr [rsi]
    vpslld      zmm0, zmm0, 16

    // Load V[j][d+16..d+32]: BF16 → f32 (chunk 2)
    vpmovzxwd   zmm2, ymmword ptr [rsi + 32]
    vpslld      zmm2, zmm2, 16

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
    jb          .L2p_bf16_j_loop

.L2p_bf16_store:
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
    jb          .L2p_bf16_d_loop

    vzeroupper
    add         rsp, 48
    pop         r15
    pop         r14
    pop         r13
    pop         r12
    pop         rbx
    ret
