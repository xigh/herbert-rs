fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/avx512_q4_kernels.S");
    println!("cargo:rerun-if-changed=src/avx2_q4_kernels.S");

    #[cfg(target_arch = "x86_64")]
    {
        cc::Build::new()
            .file("src/avx512_q4_kernels.S")
            .flag("-mavx512f")
            .flag("-mavx512bw")
            .flag("-mavx512vnni")
            .flag("-mfma")
            .compile("avx512_q4_kernels");

        cc::Build::new()
            .file("src/avx2_q4_kernels.S")
            .flag("-mavx2")
            .flag("-mfma")
            .compile("avx2_q4_kernels");
    }
}
