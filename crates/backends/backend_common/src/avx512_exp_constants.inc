// Exp constants for inline fast exp(x) polynomial approximation.
// Shared between avx512_fast_exp.S and fused attention kernels.
//
// Uses Horner degree-4 minimax polynomial: exp(x) = 2^n * p(f)
// where n = round(x * log2e), f = x * log2e - n
// Relative error < 2e-7 (within f32 precision).

#ifdef __APPLE__
.section __TEXT,__const
#else
.section .rodata
#endif

.p2align 6
.Lexp_log2e:
    .rept 16
    .float 1.4426950408889634
    .endr

.Lexp_c4:
    .rept 16
    .float 0.009672479704022407
    .endr

.Lexp_c3:
    .rept 16
    .float 0.05550410866482422
    .endr

.Lexp_c2:
    .rept 16
    .float 0.2402265069591985
    .endr

.Lexp_c1:
    .rept 16
    .float 0.6931471824645996
    .endr

.Lexp_c0:
    .rept 16
    .float 1.0
    .endr

.Lexp_one_bits:
    .rept 16
    .long 0x3F800000
    .endr

.Lexp_neg87:
    .rept 16
    .float -87.33654
    .endr
