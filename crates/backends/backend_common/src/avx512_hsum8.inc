// Horizontal sum macro for 8 ZMM accumulators (zmm4-zmm11)
// Reduces each to a scalar f32 and stores to memory.
// Clobbers: ymm1, and the lower halves of the input registers.

.macro HSUM_ONE zr, yr, xr, base, off
    vextractf64x4 ymm1, \zr, 1
    vaddps      \yr, \yr, ymm1
    vextractf128  xmm1, \yr, 1
    vaddps      \xr, \xr, xmm1
    vhaddps     \xr, \xr, \xr
    vhaddps     \xr, \xr, \xr
    vmovss      [\base + \off], \xr
.endm

.macro HSUM8_STORE base, off
    HSUM_ONE zmm4,  ymm4,  xmm4,  \base, \off
    HSUM_ONE zmm5,  ymm5,  xmm5,  \base, \off+4
    HSUM_ONE zmm6,  ymm6,  xmm6,  \base, \off+8
    HSUM_ONE zmm7,  ymm7,  xmm7,  \base, \off+12
    HSUM_ONE zmm8,  ymm8,  xmm8,  \base, \off+16
    HSUM_ONE zmm9,  ymm9,  xmm9,  \base, \off+20
    HSUM_ONE zmm10, ymm10, xmm10, \base, \off+24
    HSUM_ONE zmm11, ymm11, xmm11, \base, \off+28
.endm
