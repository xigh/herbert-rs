// Horizontal sum macro for 8 YMM accumulators (ymm4-ymm11)
// Reduces each to a scalar f32 and stores to memory.
// Clobbers: xmm1, and the lower halves of the input registers.
// AVX2 version: no vextractf64x4 (AVX-512 only), starts from YMM.

.macro HSUM_YMM_ONE yr, xr, base, off
    vextractf128  xmm1, \yr, 1
    vaddps      \xr, \xr, xmm1
    vhaddps     \xr, \xr, \xr
    vhaddps     \xr, \xr, \xr
    vmovss      [\base + \off], \xr
.endm

.macro HSUM8_YMM_STORE base, off
    HSUM_YMM_ONE ymm4,  xmm4,  \base, \off
    HSUM_YMM_ONE ymm5,  xmm5,  \base, \off+4
    HSUM_YMM_ONE ymm6,  xmm6,  \base, \off+8
    HSUM_YMM_ONE ymm7,  xmm7,  \base, \off+12
    HSUM_YMM_ONE ymm8,  xmm8,  \base, \off+16
    HSUM_YMM_ONE ymm9,  xmm9,  \base, \off+20
    HSUM_YMM_ONE ymm10, xmm10, \base, \off+24
    HSUM_YMM_ONE ymm11, xmm11, \base, \off+28
.endm
