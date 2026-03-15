// AVX-512 single dot product: Q_f32 · K_bf16 → f32 scalar
// Used by decode path (single head).

.intel_syntax noprefix
#include "avx512_common.inc"

// avx512_attn_dot_qf32_kbf16(q: *const f32, k: *const u16, dim: u64) → xmm0
    .globl SYMBOL(avx512_attn_dot_qf32_kbf16)
    .p2align 4
SYMBOL(avx512_attn_dot_qf32_kbf16):
    vxorps      zmm0, zmm0, zmm0

    cmp         rdx, 128
    jb          .Ldot_bf16_loop_init

    // Fully unrolled path for dim=128
    vpmovzxwd   zmm1, ymmword ptr [rsi]
    vpslld      zmm1, zmm1, 16
    vmovups     zmm2, [rdi]
    vfmadd231ps zmm0, zmm1, zmm2

    vpmovzxwd   zmm1, ymmword ptr [rsi + 32]
    vpslld      zmm1, zmm1, 16
    vmovups     zmm2, [rdi + 64]
    vfmadd231ps zmm0, zmm1, zmm2

    vpmovzxwd   zmm1, ymmword ptr [rsi + 64]
    vpslld      zmm1, zmm1, 16
    vmovups     zmm2, [rdi + 128]
    vfmadd231ps zmm0, zmm1, zmm2

    vpmovzxwd   zmm1, ymmword ptr [rsi + 96]
    vpslld      zmm1, zmm1, 16
    vmovups     zmm2, [rdi + 192]
    vfmadd231ps zmm0, zmm1, zmm2

    vpmovzxwd   zmm1, ymmword ptr [rsi + 128]
    vpslld      zmm1, zmm1, 16
    vmovups     zmm2, [rdi + 256]
    vfmadd231ps zmm0, zmm1, zmm2

    vpmovzxwd   zmm1, ymmword ptr [rsi + 160]
    vpslld      zmm1, zmm1, 16
    vmovups     zmm2, [rdi + 320]
    vfmadd231ps zmm0, zmm1, zmm2

    vpmovzxwd   zmm1, ymmword ptr [rsi + 192]
    vpslld      zmm1, zmm1, 16
    vmovups     zmm2, [rdi + 384]
    vfmadd231ps zmm0, zmm1, zmm2

    vpmovzxwd   zmm1, ymmword ptr [rsi + 224]
    vpslld      zmm1, zmm1, 16
    vmovups     zmm2, [rdi + 448]
    vfmadd231ps zmm0, zmm1, zmm2

    jmp         .Ldot_bf16_hsum

.Ldot_bf16_loop_init:
    test        rdx, rdx
    jz          .Ldot_bf16_hsum
    xor         rcx, rcx

.Ldot_bf16_loop:
    vpmovzxwd   zmm1, ymmword ptr [rsi + rcx*2]
    vpslld      zmm1, zmm1, 16
    vmovups     zmm2, [rdi + rcx*4]
    vfmadd231ps zmm0, zmm1, zmm2
    add         rcx, 16
    cmp         rcx, rdx
    jb          .Ldot_bf16_loop

.Ldot_bf16_hsum:
    vextractf64x4 ymm1, zmm0, 1
    vaddps      ymm0, ymm0, ymm1
    vextractf128  xmm1, ymm0, 1
    vaddps      xmm0, xmm0, xmm1
    vhaddps     xmm0, xmm0, xmm0
    vhaddps     xmm0, xmm0, xmm0
    vzeroupper
    ret


// avx512_attn_sv_accum_bf16(out: *mut f32, v: *const u16, xmm0=score, dim: u64)
    .globl SYMBOL(avx512_attn_sv_accum_bf16)
    .p2align 4
SYMBOL(avx512_attn_sv_accum_bf16):
    vbroadcastss zmm3, xmm0

    cmp         rdx, 128
    jb          .Lsv_bf16_loop_init

    // Fully unrolled for dim=128
    vpmovzxwd   zmm1, ymmword ptr [rsi]
    vpslld      zmm1, zmm1, 16
    vmovups     zmm2, [rdi]
    vfmadd231ps zmm2, zmm1, zmm3
    vmovups     [rdi], zmm2

    vpmovzxwd   zmm1, ymmword ptr [rsi + 32]
    vpslld      zmm1, zmm1, 16
    vmovups     zmm2, [rdi + 64]
    vfmadd231ps zmm2, zmm1, zmm3
    vmovups     [rdi + 64], zmm2

    vpmovzxwd   zmm1, ymmword ptr [rsi + 64]
    vpslld      zmm1, zmm1, 16
    vmovups     zmm2, [rdi + 128]
    vfmadd231ps zmm2, zmm1, zmm3
    vmovups     [rdi + 128], zmm2

    vpmovzxwd   zmm1, ymmword ptr [rsi + 96]
    vpslld      zmm1, zmm1, 16
    vmovups     zmm2, [rdi + 192]
    vfmadd231ps zmm2, zmm1, zmm3
    vmovups     [rdi + 192], zmm2

    vpmovzxwd   zmm1, ymmword ptr [rsi + 128]
    vpslld      zmm1, zmm1, 16
    vmovups     zmm2, [rdi + 256]
    vfmadd231ps zmm2, zmm1, zmm3
    vmovups     [rdi + 256], zmm2

    vpmovzxwd   zmm1, ymmword ptr [rsi + 160]
    vpslld      zmm1, zmm1, 16
    vmovups     zmm2, [rdi + 320]
    vfmadd231ps zmm2, zmm1, zmm3
    vmovups     [rdi + 320], zmm2

    vpmovzxwd   zmm1, ymmword ptr [rsi + 192]
    vpslld      zmm1, zmm1, 16
    vmovups     zmm2, [rdi + 384]
    vfmadd231ps zmm2, zmm1, zmm3
    vmovups     [rdi + 384], zmm2

    vpmovzxwd   zmm1, ymmword ptr [rsi + 224]
    vpslld      zmm1, zmm1, 16
    vmovups     zmm2, [rdi + 448]
    vfmadd231ps zmm2, zmm1, zmm3
    vmovups     [rdi + 448], zmm2

    vzeroupper
    ret

.Lsv_bf16_loop_init:
    test        rdx, rdx
    jz          .Lsv_bf16_done
    xor         rcx, rcx

.Lsv_bf16_loop:
    vpmovzxwd   zmm1, ymmword ptr [rsi + rcx*2]
    vpslld      zmm1, zmm1, 16
    vmovups     zmm2, [rdi + rcx*4]
    vfmadd231ps zmm2, zmm1, zmm3
    vmovups     [rdi + rcx*4], zmm2
    add         rcx, 16
    cmp         rcx, rdx
    jb          .Lsv_bf16_loop

.Lsv_bf16_done:
    vzeroupper
    ret
