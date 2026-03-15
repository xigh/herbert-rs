fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/avx512_bf16_matvec.S");

    #[cfg(target_arch = "x86_64")]
    {
        cc::Build::new()
            .file("src/avx512_bf16_matvec.S")
            .flag("-mavx512f")
            .flag("-mavx512bw")
            .flag("-mavx512bf16")
            .flag("-mfma")
            .compile("avx512_bf16_kernels");
    }
}
