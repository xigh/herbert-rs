fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/avx512_int8_matvec.S");
    println!("cargo:rerun-if-changed=src/avx512_int8_matvec_v2.S");

    #[cfg(target_arch = "x86_64")]
    {
        cc::Build::new()
            .file("src/avx512_int8_matvec.S")
            .flag("-mavx512f")
            .flag("-mavx512bw")
            .flag("-mavx512vnni")
            .flag("-mfma")
            .compile("avx512_int8_kernels");

        cc::Build::new()
            .file("src/avx512_int8_matvec_v2.S")
            .flag("-mavx512f")
            .flag("-mavx512bw")
            .flag("-mavx512vnni")
            .flag("-mfma")
            .compile("avx512_int8_kernels_v2");
    }
}
