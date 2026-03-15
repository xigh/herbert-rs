// AVX-512 f32 scale-accumulate for vision encoder
//
// out[i] += scale * v[i], 16-element loop with scalar tail.
//
// Target: x86_64 with AVX-512F
// Convention: System V AMD64 ABI

.intel_syntax noprefix

#ifdef __APPLE__
#define SYMBOL(name) _##name
#else
#define SYMBOL(name) name
#endif

.text

// ============================================================================
// avx512_vision_sv_accum_f32
//
// Args (System V ABI):
//   rdi = out   : *mut f32
//   rsi = v     : *const f32
//   xmm0 = scale : f32
//   rdx = n     : u64
// ============================================================================
    .globl SYMBOL(avx512_vision_sv_accum_f32)
    .p2align 4
SYMBOL(avx512_vision_sv_accum_f32):
    // Broadcast scale from xmm0 to zmm4
    vbroadcastss zmm4, xmm0

    xor         rcx, rcx                // i = 0

    // ─── Main loop: 16 elements per iteration ───
.Lavx_sv_loop16:
    lea         rax, [rcx + 16]
    cmp         rax, rdx
    ja          .Lavx_sv_scalar

    vmovups     zmm0, [rdi + rcx*4]     // load out[i..i+16]
    vmovups     zmm1, [rsi + rcx*4]     // load v[i..i+16]
    vfmadd231ps zmm0, zmm1, zmm4       // out += v * scale
    vmovups     [rdi + rcx*4], zmm0     // store

    add         rcx, 16
    jmp         .Lavx_sv_loop16

    // ─── Scalar tail ───
.Lavx_sv_scalar:
    cmp         rcx, rdx
    jge         .Lavx_sv_done

    vmovss      xmm0, [rdi + rcx*4]    // load out[i]
    vmovss      xmm1, [rsi + rcx*4]    // load v[i]
    vfmadd231ss xmm0, xmm1, xmm4      // out[i] += v[i] * scale
    vmovss      [rdi + rcx*4], xmm0    // store

    inc         rcx
    jmp         .Lavx_sv_scalar

.Lavx_sv_done:
    vzeroupper
    ret
