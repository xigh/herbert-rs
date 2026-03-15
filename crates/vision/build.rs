fn main() {
    #[cfg(target_arch = "x86_64")]
    {
        cc::Build::new()
            .file("src/avx512_vision_dot.S")
            .file("src/avx512_vision_sv.S")
            .file("src/avx512_vision_layernorm.S")
            .file("src/avx512_vision_gelu.S")
            .flag("-mavx512f")
            .compile("avx512_vision_kernels");
    }
}
