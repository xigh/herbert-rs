//! FFI declarations for attention kernels.

#[cfg(target_arch = "x86_64")]
extern "C" {
    // ──── AVX-512 BF16 kernels ────
    pub(super) fn avx512_attn_dot8_qf32_kbf16(
        q_ptrs: *const *const f32,
        k: *const u16,
        dim: u64,
        scores: *mut f32,
    );
    pub(super) fn avx512_attn_online_sv_bf16(
        out: *mut f32,
        v: *const u16,
        correction: f32,
        alpha: f32,
        dim: u64,
    );
    pub(super) fn avx512_fused_attn_step8_bf16(
        q_ptrs: *const *const f32,
        k_ptr: *const u16,
        v_ptr: *const u16,
        o_base: *mut f32,
        ml_ptr: *mut f32,
        dim: u64,
        scale: f32,
    );

    // ──── AVX-512 INT8 kernels ────
    pub(super) fn avx512_attn_dot8_qf32_ki8(
        q_ptrs: *const *const f32,
        k: *const i8,
        dim: u64,
        scores: *mut f32,
    );
    pub(super) fn avx512_attn_online_sv_i8(
        out: *mut f32,
        v: *const i8,
        correction: f32,
        alpha: f32,
        dim: u64,
    );
    pub(super) fn avx512_fused_attn_step8_i8(
        q_ptrs: *const *const f32,
        k_ptr: *const i8,
        v_ptr: *const i8,
        o_base: *mut f32,
        ml_ptr: *mut f32,
        dim: u64,
        dot_scale: f32,
        v_scale: f32,
    );

    // ──── AVX2+FMA BF16 kernels ────
    pub(super) fn avx2_attn_dot8_qf32_kbf16(
        q_ptrs: *const *const f32,
        k: *const u16,
        dim: u64,
        scores: *mut f32,
    );

    // ──── AVX2+FMA INT8 kernels ────
    pub(super) fn avx2_attn_dot8_qf32_ki8(
        q_ptrs: *const *const f32,
        k: *const i8,
        dim: u64,
        scores: *mut f32,
    );
}

// ──── AVX-512 BF16 DPBF16 kernels (Q BF16 + K BF16, VDPBF16PS) ────
#[cfg(target_arch = "x86_64")]
extern "C" {
    pub(super) fn avx512_attn_dot8_qbf16_kbf16(
        q_ptrs: *const *const u16,
        k: *const u16,
        dim: u64,
        scores: *mut f32,
    );
    pub(super) fn avx512_fused_attn_step8_bf16_dpbf16(
        q_ptrs: *const *const u16,
        k_ptr: *const u16,
        v_ptr: *const u16,
        o_base: *mut f32,
        ml_ptr: *mut f32,
        dim: u64,
        scale: f32,
    );
}

// 2-pass attention kernel extern declarations (x86_64 only)
#[cfg(all(target_arch = "x86_64", feature = "attn-2pass"))]
extern "C" {
    pub(super) fn avx512_2pass_sv8_bf16(
        scores: *const f32,
        v_base: *const u16,
        o_base: *mut f32,
        head_dim: u64,
        valid_len: u64,
    );

    pub(super) fn avx512_2pass_sv8_i8(
        scores: *const f32,
        v_base: *const i8,
        o_base: *mut f32,
        head_dim: u64,
        valid_len: u64,
    );

    pub(super) fn avx512_softmax_8head_inplace(
        scores: *mut f32,
        valid_len: u64,
    );
}

