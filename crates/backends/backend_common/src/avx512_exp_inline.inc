// Inline fast exp(x) for 16 f32 values in a ZMM register.
// zmm_io = input/output register
// t1, t2, t3 = scratch ZMM registers
// Requires constants from avx512_exp_constants.inc in scope.

.macro INLINE_EXP16 zmm_io, t1, t2, t3
    vmaxps      \zmm_io, \zmm_io, [rip + .Lexp_neg87]
    vmulps      \t1, \zmm_io, [rip + .Lexp_log2e]
    vrndscaleps \t2, \t1, 0
    vsubps      \t1, \t1, \t2
    vmovaps     \t3, [rip + .Lexp_c4]
    vfmadd213ps \t3, \t1, [rip + .Lexp_c3]
    vfmadd213ps \t3, \t1, [rip + .Lexp_c2]
    vfmadd213ps \t3, \t1, [rip + .Lexp_c1]
    vfmadd213ps \t3, \t1, [rip + .Lexp_c0]
    vcvtps2dq   \t2, \t2
    vpslld      \t2, \t2, 23
    vpaddd      \t2, \t2, [rip + .Lexp_one_bits]
    vmulps      \zmm_io, \t3, \t2
.endm
