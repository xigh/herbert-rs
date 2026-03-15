fn main() {
    #[cfg(target_arch = "x86_64")]
    {
        // AVX-512 attention kernels
        let mut build = cc::Build::new();
        build
            .file("src/avx512_attn_dot_bf16.S")
            .file("src/avx512_attn_dot_i8.S")
            .file("src/avx512_attn_dot_i4.S")
            .file("src/avx512_attn_dot8_bf16.S")
            .file("src/avx512_attn_dot8_i8.S")
            .file("src/avx512_attn_sv_bf16.S")
            .file("src/avx512_attn_sv_i8.S")
            .file("src/avx512_fast_exp.S")
            .file("src/avx512_fused_attn_bf16.S")
            .file("src/avx512_fused_attn_i8.S")
            .file("src/avx512_2pass_sv_bf16.S")
            .file("src/avx512_2pass_sv_i8.S")
            .file("src/avx512_softmax_8head.S")
            .flag("-mavx512f")
            .flag("-Isrc");
        if std::env::var("CARGO_FEATURE_PROFILE_ATTN_KERNEL").is_ok() {
            build.flag("-DPROFILE_RDTSC");
        }
        build.compile("avx512_attn_kernels");

        // AVX-512 BF16 DPBF16 attention kernels (Q BF16 + K BF16)
        let mut dpbf16_build = cc::Build::new();
        dpbf16_build
            .file("src/avx512_attn_dot8_qbf16_kbf16.S")
            .file("src/avx512_fused_attn_bf16_dpbf16.S")
            .flag("-mavx512f")
            .flag("-mavx512bf16")
            .flag("-Isrc");
        if std::env::var("CARGO_FEATURE_PROFILE_ATTN_KERNEL").is_ok() {
            dpbf16_build.flag("-DPROFILE_RDTSC");
        }
        dpbf16_build.compile("avx512_dpbf16_fused_attn_kernels");

        // AVX2+FMA attention kernels
        let mut avx2_build = cc::Build::new();
        avx2_build
            .file("src/avx2_attn_dot_bf16.S")
            .file("src/avx2_attn_dot_i8.S")
            .file("src/avx2_attn_dot_i4.S")
            .file("src/avx2_attn_dot8_bf16.S")
            .file("src/avx2_attn_dot8_i8.S")
            .flag("-mavx2")
            .flag("-mfma")
            .flag("-Isrc");
        avx2_build.compile("avx2_attn_kernels");
    }

}
