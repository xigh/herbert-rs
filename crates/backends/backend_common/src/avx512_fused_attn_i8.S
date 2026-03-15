// Fused DOT+EXP+SV attention kernel for 8 Q heads, int8 KV cache.
//
// Processes one KV position (j-step) for 8 Q heads simultaneously:
//   1. DOT: compute 8 dot products Q[h] · K_i8[j]
//   2. EXP: online softmax (m_new, corrections, alphas via inline exp)
//   3. SV:  out[h] = correction[h] * out[h] + alpha[h] * v_scale * V_i8[j]
//
// Args (System V ABI):
//   rdi = q_ptrs     : *const *const f32  (8 Q pointers)
//   rsi = k_ptr      : *const i8          (K at kv_offset, int8)
//   rdx = v_ptr      : *const i8          (V at kv_offset, int8)
//   rcx = o_base     : *mut f32           (8 * head_dim output buffer)
//   r8  = ml_ptr     : *mut f32           ([m[0..8], l[0..8]] read+write)
//   r9  = dim        : u64                (head_dim, must be multiple of 16)
//   xmm0 = dot_scale : f32               (scale * k_scale, precomputed)
//   xmm1 = v_scale   : f32               (V dequant scale)

.intel_syntax noprefix
#include "avx512_common.inc"
#include "avx512_hsum8.inc"
#include "avx512_exp_constants.inc"
#include "avx512_exp_inline.inc"

.text

    .globl SYMBOL(avx512_fused_attn_step8_i8)
    .p2align 4
SYMBOL(avx512_fused_attn_step8_i8):
    push        rbx
    push        r12
    push        r13
    push        r14
    push        r15
#ifdef PROFILE_RDTSC
    sub         rsp, 128            // +32 bytes for TSC slots at [rsp+96..127]
#else
    sub         rsp, 96
#endif

    // Save args
    mov         [rsp], r8           // ml_ptr
    mov         [rsp+8], rdx        // v_ptr
    mov         [rsp+16], rcx       // o_base
    vmovss      [rsp+24], xmm0     // dot_scale
    vmovss      [rsp+28], xmm1     // v_scale
    mov         rbx, r9             // dim

    // Load 8 Q pointers
    mov         r8,  [rdi]
    mov         r9,  [rdi + 8]
    mov         r10, [rdi + 16]
    mov         r11, [rdi + 24]
    mov         r12, [rdi + 32]
    mov         r13, [rdi + 40]
    mov         r14, [rdi + 48]
    mov         r15, [rdi + 56]

    // Zero 8 accumulators
    vxorps      zmm4, zmm4, zmm4
    vxorps      zmm5, zmm5, zmm5
    vxorps      zmm6, zmm6, zmm6
    vxorps      zmm7, zmm7, zmm7
    vxorps      zmm8, zmm8, zmm8
    vxorps      zmm9, zmm9, zmm9
    vxorps      zmm10, zmm10, zmm10
    vxorps      zmm11, zmm11, zmm11

    // ---- DOT8 loop (i8 K) ----
#ifdef PROFILE_RDTSC
    rdtsc
    shl         rdx, 32
    or          rax, rdx
    mov         [rsp + 96], rax     // T0: DOT start
#endif
    xor         rcx, rcx

.Lfused_i8_dot_loop:
    vpmovsxbd   zmm0, xmmword ptr [rsi + rcx]
    vcvtdq2ps   zmm0, zmm0

    vmovups     zmm1, [r8 + rcx*4]
    vfmadd231ps zmm4, zmm0, zmm1
    vmovups     zmm1, [r9 + rcx*4]
    vfmadd231ps zmm5, zmm0, zmm1
    vmovups     zmm1, [r10 + rcx*4]
    vfmadd231ps zmm6, zmm0, zmm1
    vmovups     zmm1, [r11 + rcx*4]
    vfmadd231ps zmm7, zmm0, zmm1
    vmovups     zmm1, [r12 + rcx*4]
    vfmadd231ps zmm8, zmm0, zmm1
    vmovups     zmm1, [r13 + rcx*4]
    vfmadd231ps zmm9, zmm0, zmm1
    vmovups     zmm1, [r14 + rcx*4]
    vfmadd231ps zmm10, zmm0, zmm1
    vmovups     zmm1, [r15 + rcx*4]
    vfmadd231ps zmm11, zmm0, zmm1

    add         rcx, 16
    cmp         rcx, rbx
    jb          .Lfused_i8_dot_loop

    // ---- HSUM + dot_scale → scores in ymm4 ----
    HSUM8_STORE rsp, 32

    vmovups     ymm4, [rsp + 32]
    vbroadcastss ymm5, dword ptr [rsp + 24]  // dot_scale
    vmulps      ymm4, ymm4, ymm5

#ifdef PROFILE_RDTSC
    rdtsc
    shl         rdx, 32
    or          rax, rdx
    mov         [rsp + 104], rax    // T1: DOT+HSUM end
#endif

    // ---- Online softmax EXP ----
    mov         rdi, [rsp]          // ml_ptr
    vmovups     ymm5, [rdi]         // m_old
    vmaxps      ymm6, ymm4, ymm5   // m_new
    vmovups     [rdi], ymm6

    vsubps      ymm7, ymm5, ymm6   // corrections_in
    vsubps      ymm8, ymm4, ymm6   // alphas_in

    vinsertf64x4 zmm0, zmm7, ymm8, 1
    INLINE_EXP16 zmm0, zmm1, zmm2, zmm3

    vextractf64x4 ymm1, zmm0, 1    // alphas
    // ymm0 = corrections

    // Update l
    vmovups     ymm2, [rdi + 32]
    vmulps      ymm2, ymm2, ymm0
    vaddps      ymm2, ymm2, ymm1
    vmovups     [rdi + 32], ymm2

    // Multiply alphas by v_scale before SV
    vbroadcastss ymm3, dword ptr [rsp + 28]  // v_scale
    vmulps      ymm1, ymm1, ymm3

    // Save for broadcast
    vmovups     [rsp + 32], ymm0    // corrections
    vmovups     [rsp + 64], ymm1    // alphas * v_scale

#ifdef PROFILE_RDTSC
    rdtsc
    shl         rdx, 32
    or          rax, rdx
    mov         [rsp + 112], rax    // T2: EXP end
#endif

    // ---- SV8 setup ----
    mov         rdi, [rsp + 16]     // o_base
    mov         rsi, [rsp + 8]      // v_ptr

    // Broadcast corrections → zmm16-23
    vbroadcastss zmm16, dword ptr [rsp + 32]
    vbroadcastss zmm17, dword ptr [rsp + 36]
    vbroadcastss zmm18, dword ptr [rsp + 40]
    vbroadcastss zmm19, dword ptr [rsp + 44]
    vbroadcastss zmm20, dword ptr [rsp + 48]
    vbroadcastss zmm21, dword ptr [rsp + 52]
    vbroadcastss zmm22, dword ptr [rsp + 56]
    vbroadcastss zmm23, dword ptr [rsp + 60]

    // Broadcast alphas*v_scale → zmm24-31
    vbroadcastss zmm24, dword ptr [rsp + 64]
    vbroadcastss zmm25, dword ptr [rsp + 68]
    vbroadcastss zmm26, dword ptr [rsp + 72]
    vbroadcastss zmm27, dword ptr [rsp + 76]
    vbroadcastss zmm28, dword ptr [rsp + 80]
    vbroadcastss zmm29, dword ptr [rsp + 84]
    vbroadcastss zmm30, dword ptr [rsp + 88]
    vbroadcastss zmm31, dword ptr [rsp + 92]

    // Head stride and addresses
    mov         rax, rbx
    shl         rax, 2
    lea         r9,  [rdi + rax]
    lea         r10, [rdi + rax*2]
    lea         r11, [r10 + rax]
    lea         r12, [rdi + rax*4]
    lea         r13, [r12 + rax]
    lea         r14, [r12 + rax*2]
    lea         r15, [r14 + rax]

    // ---- SV8 loop (i8 V) ----
    xor         rcx, rcx
    xor         rax, rax

.Lfused_i8_sv_loop:
    vpmovsxbd   zmm0, xmmword ptr [rsi + rcx]
    vcvtdq2ps   zmm0, zmm0

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
    cmp         rcx, rbx
    jb          .Lfused_i8_sv_loop

    // ---- Epilogue ----
#ifdef PROFILE_RDTSC
    // Write TSC deltas to ml_ptr + 64 (3 u64: DOT, EXP, SV cycles)
    rdtsc
    shl         rdx, 32
    or          rax, rdx            // T3: SV end
    mov         rcx, [rsp]          // ml_ptr
    mov         r8, [rsp + 104]
    sub         r8, [rsp + 96]
    mov         [rcx + 64], r8      // DOT+HSUM cycles
    mov         r8, [rsp + 112]
    sub         r8, [rsp + 104]
    mov         [rcx + 72], r8      // EXP cycles
    sub         rax, [rsp + 112]
    mov         [rcx + 80], rax     // SV cycles
#endif
    vzeroupper
#ifdef PROFILE_RDTSC
    add         rsp, 128
#else
    add         rsp, 96
#endif
    pop         r15
    pop         r14
    pop         r13
    pop         r12
    pop         rbx
    ret
