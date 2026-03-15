// AVX-512 GELU with Pade tanh approximation for vision encoder
//
// gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// tanh via Pade [3,3] rational approximation, clamped to |z| < 4.97.
//
// Constants loaded from .rodata via rip-relative vbroadcastss into zmm16-zmm27.
//
// Target: x86_64 with AVX-512F
// Convention: System V AMD64 ABI

.intel_syntax noprefix

#ifdef __APPLE__
#define SYMBOL(name) _##name
#else
#define SYMBOL(name) name
#endif

// ─── Read-only constants ───
#ifdef __APPLE__
.section __TEXT,__const
#else
.section .rodata
#endif
.p2align 6
gelu_sqrt_2_over_pi:    .float 0.79788456      // sqrt(2/pi)
gelu_half:              .float 0.5
gelu_one:               .float 1.0
gelu_coeff:             .float 0.044715
gelu_c135135:           .float 135135.0
gelu_c17325:            .float 17325.0
gelu_c378:              .float 378.0
gelu_c62370:            .float 62370.0
gelu_c3150:             .float 3150.0
gelu_c28:               .float 28.0
gelu_clamp_hi:          .float 4.97
gelu_clamp_lo:          .float -4.97

.text

// ============================================================================
// avx512_vision_gelu_f32
//
// Args (System V ABI):
//   rdi = data  : *mut f32
//   rsi = n     : u64
//
// In-place GELU activation.
//
// Constants in zmm16-zmm27 (all caller-saved on SysV):
//   zmm16 = sqrt(2/pi)    zmm17 = 0.5         zmm18 = 1.0
//   zmm19 = 0.044715      zmm20 = 135135.0    zmm21 = 17325.0
//   zmm22 = 378.0         zmm23 = 62370.0     zmm24 = 3150.0
//   zmm25 = 28.0          zmm26 = 4.97        zmm27 = -4.97
//
// Scratch: zmm0-zmm8, xmm0-xmm8
// ============================================================================
    .globl SYMBOL(avx512_vision_gelu_f32)
    .p2align 4
SYMBOL(avx512_vision_gelu_f32):
    // ─── Preload 12 constants into zmm16-zmm27 ───
    vbroadcastss zmm16, dword ptr [rip + gelu_sqrt_2_over_pi]
    vbroadcastss zmm17, dword ptr [rip + gelu_half]
    vbroadcastss zmm18, dword ptr [rip + gelu_one]
    vbroadcastss zmm19, dword ptr [rip + gelu_coeff]
    vbroadcastss zmm20, dword ptr [rip + gelu_c135135]
    vbroadcastss zmm21, dword ptr [rip + gelu_c17325]
    vbroadcastss zmm22, dword ptr [rip + gelu_c378]
    vbroadcastss zmm23, dword ptr [rip + gelu_c62370]
    vbroadcastss zmm24, dword ptr [rip + gelu_c3150]
    vbroadcastss zmm25, dword ptr [rip + gelu_c28]
    vbroadcastss zmm26, dword ptr [rip + gelu_clamp_hi]
    vbroadcastss zmm27, dword ptr [rip + gelu_clamp_lo]

    xor         rcx, rcx                // i = 0

    // ─── SIMD loop: 16 elements per iteration ───
.Lgelu_loop16:
    lea         rax, [rcx + 16]
    cmp         rax, rsi
    ja          .Lgelu_scalar

    vmovups     zmm0, [rdi + rcx*4]     // zmm0 = x (original)
    vmovaps     zmm4, zmm0              // zmm4 = x (saved copy)

    // x^2, x^3
    vmulps      zmm1, zmm0, zmm0       // zmm1 = x2
    vmulps      zmm2, zmm1, zmm0       // zmm2 = x3

    // inner_arg = x + coeff * x^3
    vmovaps     zmm3, zmm0              // zmm3 = x
    vfmadd231ps zmm3, zmm19, zmm2      // zmm3 = x + 0.044715 * x3

    // inner = sqrt(2/pi) * inner_arg
    vmulps      zmm3, zmm16, zmm3      // zmm3 = inner

    // Clamp z for Pade: z = clamp(inner, -4.97, 4.97)
    vmaxps      zmm3, zmm3, zmm27
    vminps      zmm3, zmm3, zmm26      // zmm3 = z

    // z2
    vmulps      zmm5, zmm3, zmm3       // zmm5 = z2

    // ─── Numerator: z * (135135 + z2 * (17325 + z2 * (378 + z2))) ───
    vaddps      zmm6, zmm22, zmm5              // zmm6 = 378 + z2
    vmovaps     zmm7, zmm21                    // zmm7 = 17325
    vfmadd231ps zmm7, zmm5, zmm6              // zmm7 = 17325 + z2*(378+z2)
    vmovaps     zmm6, zmm20                    // zmm6 = 135135
    vfmadd231ps zmm6, zmm5, zmm7              // zmm6 = 135135 + z2*(17325+z2*(378+z2))
    vmulps      zmm6, zmm3, zmm6              // zmm6 = num = z * (...)

    // ─── Denominator: 135135 + z2 * (62370 + z2 * (3150 + z2 * 28)) ───
    vmovaps     zmm7, zmm24                    // zmm7 = 3150
    vfmadd231ps zmm7, zmm5, zmm25             // zmm7 = 3150 + z2*28
    vmovaps     zmm8, zmm23                    // zmm8 = 62370
    vfmadd231ps zmm8, zmm5, zmm7              // zmm8 = 62370 + z2*(3150+z2*28)
    vmovaps     zmm7, zmm20                    // zmm7 = 135135
    vfmadd231ps zmm7, zmm5, zmm8              // zmm7 = 135135 + z2*(...)

    // tanh approx = num / den
    vdivps      zmm6, zmm6, zmm7              // zmm6 = tanh

    // gelu = 0.5 * x * (1 + tanh)
    vaddps      zmm6, zmm18, zmm6             // zmm6 = 1 + tanh
    vmulps      zmm6, zmm4, zmm6              // zmm6 = x * (1 + tanh)
    vmulps      zmm6, zmm17, zmm6             // zmm6 = 0.5 * x * (1 + tanh)

    vmovups     [rdi + rcx*4], zmm6

    add         rcx, 16
    jmp         .Lgelu_loop16

    // ─── Scalar tail ───
.Lgelu_scalar:
    cmp         rcx, rsi
    jge         .Lgelu_done

    vmovss      xmm0, [rdi + rcx*4]    // x

    // x^2, x^3
    vmulss      xmm1, xmm0, xmm0      // x2
    vmulss      xmm2, xmm1, xmm0      // x3

    // inner = sqrt(2/pi) * (x + 0.044715 * x^3)
    vmulss      xmm2, xmm2, xmm19     // 0.044715 * x3
    vaddss      xmm2, xmm0, xmm2      // x + coeff*x3
    vmulss      xmm2, xmm16, xmm2     // inner

    // Clamp
    vmaxss      xmm2, xmm2, xmm27
    vminss      xmm2, xmm2, xmm26     // z

    // z2
    vmulss      xmm3, xmm2, xmm2

    // Numerator: z * (135135 + z2*(17325 + z2*(378 + z2)))
    vaddss      xmm4, xmm22, xmm3             // 378 + z2
    vmovaps     xmm5, xmm21                    // scratch = 17325
    vfmadd231ss xmm5, xmm3, xmm4              // 17325 + z2*(378+z2)
    vmovaps     xmm4, xmm20                    // scratch = 135135
    vfmadd231ss xmm4, xmm3, xmm5              // 135135 + z2*(...)
    vmulss      xmm4, xmm2, xmm4              // num = z * (...)

    // Denominator: 135135 + z2*(62370 + z2*(3150 + z2*28))
    vmovaps     xmm5, xmm24                    // scratch = 3150
    vfmadd231ss xmm5, xmm3, xmm25             // 3150 + z2*28
    vmovaps     xmm6, xmm23                    // scratch = 62370
    vfmadd231ss xmm6, xmm3, xmm5              // 62370 + z2*(...)
    vmovaps     xmm5, xmm20                    // scratch = 135135
    vfmadd231ss xmm5, xmm3, xmm6              // 135135 + z2*(...)

    // tanh = num / den
    vdivss      xmm4, xmm4, xmm5

    // gelu = 0.5 * x * (1 + tanh)
    vaddss      xmm4, xmm18, xmm4             // 1 + tanh
    vmulss      xmm4, xmm0, xmm4              // x * (1+tanh)
    vmulss      xmm4, xmm17, xmm4             // 0.5 * x * (1+tanh)

    vmovss      [rdi + rcx*4], xmm4

    inc         rcx
    jmp         .Lgelu_scalar

.Lgelu_done:
    vzeroupper
    ret
