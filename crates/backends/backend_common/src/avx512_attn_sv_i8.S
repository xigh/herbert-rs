// AVX-512 online softmax SV kernels for int8 V.
// Contains both single-head and 8-head fused versions.

.intel_syntax noprefix
#include "avx512_common.inc"

// avx512_attn_online_sv_i8(out: *mut f32, v: *const i8, xmm0=correction, xmm1=alpha, dim: u64)
    .globl SYMBOL(avx512_attn_online_sv_i8)
    .p2align 4
SYMBOL(avx512_attn_online_sv_i8):
    vbroadcastss zmm4, xmm0
    vbroadcastss zmm5, xmm1

    test        rdx, rdx
    jz          .Losv_i8_done
    xor         rcx, rcx

.Losv_i8_loop:
    vpmovsxbd   zmm2, xmmword ptr [rsi + rcx]
    vcvtdq2ps   zmm2, zmm2
    vmovups     zmm3, [rdi + rcx*4]
    vmulps      zmm3, zmm3, zmm4
    vfmadd231ps zmm3, zmm2, zmm5
    vmovups     [rdi + rcx*4], zmm3
    add         rcx, 16
    cmp         rcx, rdx
    jb          .Losv_i8_loop

.Losv_i8_done:
    vzeroupper
    ret


// avx512_attn_online_sv8_i8(out_base: *mut f32, v: *const i8,
//     corrections: *const f32, alphas: *const f32, dim: u64)
    .globl SYMBOL(avx512_attn_online_sv8_i8)
    .p2align 4
SYMBOL(avx512_attn_online_sv8_i8):
    push        r12
    push        r13
    push        r14
    push        r15

    // Load 8 correction broadcasts
    vbroadcastss zmm16, dword ptr [rdx]
    vbroadcastss zmm17, dword ptr [rdx + 4]
    vbroadcastss zmm18, dword ptr [rdx + 8]
    vbroadcastss zmm19, dword ptr [rdx + 12]
    vbroadcastss zmm20, dword ptr [rdx + 16]
    vbroadcastss zmm21, dword ptr [rdx + 20]
    vbroadcastss zmm22, dword ptr [rdx + 24]
    vbroadcastss zmm23, dword ptr [rdx + 28]

    // Load 8 alpha broadcasts
    vbroadcastss zmm24, dword ptr [rcx]
    vbroadcastss zmm25, dword ptr [rcx + 4]
    vbroadcastss zmm26, dword ptr [rcx + 8]
    vbroadcastss zmm27, dword ptr [rcx + 12]
    vbroadcastss zmm28, dword ptr [rcx + 16]
    vbroadcastss zmm29, dword ptr [rcx + 20]
    vbroadcastss zmm30, dword ptr [rcx + 24]
    vbroadcastss zmm31, dword ptr [rcx + 28]

    // Head stride
    mov         rax, r8
    shl         rax, 2

    // Head base addresses
    lea         r9,  [rdi + rax]
    lea         r10, [rdi + rax*2]
    lea         r11, [r10 + rax]
    lea         r12, [rdi + rax*4]
    lea         r13, [r12 + rax]
    lea         r14, [r12 + rax*2]
    lea         r15, [r14 + rax]

    // i8 byte offset = element offset
    xor         rcx, rcx
    xor         rax, rax

.Lsv8_i8_loop:
    vpmovsxbd   zmm0, xmmword ptr [rsi + rcx]
    vcvtdq2ps   zmm0, zmm0

    // Head 0
    vmovups     zmm1, [rdi + rax]
    vmulps      zmm1, zmm1, zmm16
    vfmadd231ps zmm1, zmm0, zmm24
    vmovups     [rdi + rax], zmm1

    vmovups     zmm1, [r9 + rax]
    vmulps      zmm1, zmm1, zmm17
    vfmadd231ps zmm1, zmm0, zmm25
    vmovups     [r9 + rax], zmm1

    vmovups     zmm1, [r10 + rax]
    vmulps      zmm1, zmm1, zmm18
    vfmadd231ps zmm1, zmm0, zmm26
    vmovups     [r10 + rax], zmm1

    vmovups     zmm1, [r11 + rax]
    vmulps      zmm1, zmm1, zmm19
    vfmadd231ps zmm1, zmm0, zmm27
    vmovups     [r11 + rax], zmm1

    vmovups     zmm1, [r12 + rax]
    vmulps      zmm1, zmm1, zmm20
    vfmadd231ps zmm1, zmm0, zmm28
    vmovups     [r12 + rax], zmm1

    vmovups     zmm1, [r13 + rax]
    vmulps      zmm1, zmm1, zmm21
    vfmadd231ps zmm1, zmm0, zmm29
    vmovups     [r13 + rax], zmm1

    vmovups     zmm1, [r14 + rax]
    vmulps      zmm1, zmm1, zmm22
    vfmadd231ps zmm1, zmm0, zmm30
    vmovups     [r14 + rax], zmm1

    vmovups     zmm1, [r15 + rax]
    vmulps      zmm1, zmm1, zmm23
    vfmadd231ps zmm1, zmm0, zmm31
    vmovups     [r15 + rax], zmm1

    add         rcx, 16
    add         rax, 64
    cmp         rcx, r8
    jb          .Lsv8_i8_loop

    vzeroupper
    pop         r15
    pop         r14
    pop         r13
    pop         r12
    ret
