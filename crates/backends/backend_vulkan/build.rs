//! Compile GLSL compute shaders to SPIR-V at build time using glslc (from Vulkan SDK).
//! No library dependency -- just calls the glslc binary.
//! On non-Linux, shaders are not compiled (stub backend used instead).
//!
//! Subgroup-dependent shaders are compiled twice:
//!   - `{name}_w32.spv` with SUBGROUP_SIZE=32 (NVIDIA warp32)
//!   - `{name}_w64.spv` with SUBGROUP_SIZE=64 (AMD wave64)
//! Subgroup-independent shaders are compiled once as `{name}.spv`.

fn main() {
    // Only compile shaders on Linux (where Vulkan SDK + glslc are available)
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("linux") {
        return;
    }

    let shader_dir = std::path::Path::new("shaders");
    let out_dir = std::env::var("OUT_DIR").unwrap();

    // Subgroup-dependent shaders: compile twice (w32/w64)
    let subgroup_shaders: &[&str] = &[
        "rms_norm",
        "rms_norm_batch",
        "head_rms_norm",
        "softmax",
        "argmax",
        "f32_matvec",
        "f32_matmul",
        "attention_decode",
        "attention_prefill",
        "int8_matvec",
        "int8_matmul",
        "bf16_matvec",
        "bf16_matmul",
        "q4_matvec",
        "q4_matmul",
        "q4_matvec_v5",
        "q4_matmul_v5",
    ];

    // Subgroup-independent shaders: compile once
    let fixed_shaders = [
        "residual_add",
        "embedding",
        "swiglu",
        "rope_single",
        "rope_batch",
        "copy_buffer",
        "kv_cache_append",
        "kv_cache_append_batch",
        "bias_add_batch",
        "scaled_add",
        "moe_gather",
        "moe_scatter_add",
    ];

    // Find glslc: check VULKAN_SDK, then PATH
    let glslc = std::env::var("VULKAN_SDK")
        .map(|sdk| format!("{}/bin/glslc", sdk))
        .unwrap_or_else(|_| "glslc".to_string());

    // Compile subgroup-dependent shaders twice (w32/w64)
    for name in subgroup_shaders {
        let src_path = shader_dir.join(format!("{}.comp", name));
        for &sg in &[32u32, 64] {
            let define_arg = format!("-DSUBGROUP_SIZE={}", sg);
            let spv_path = format!("{}/{}_w{}.spv", out_dir, name, sg);

            let status = std::process::Command::new(&glslc)
                .args([
                    "--target-env=vulkan1.3",
                    "--target-spv=spv1.5",
                    &define_arg,
                    "-O",
                    "-o",
                    &spv_path,
                    src_path.to_str().unwrap(),
                ])
                .status()
                .unwrap_or_else(|e| {
                    panic!(
                        "Failed to run glslc ({}). Is the Vulkan SDK installed? Error: {}",
                        glslc, e
                    )
                });

            if !status.success() {
                panic!(
                    "glslc failed to compile {:?} (sg={}, exit code: {:?})",
                    src_path,
                    sg,
                    status.code()
                );
            }
        }
        println!("cargo:rerun-if-changed=shaders/{}.comp", name);
    }

    // Compile subgroup-independent shaders once
    for name in &fixed_shaders {
        let src_path = shader_dir.join(format!("{}.comp", name));
        let spv_path = format!("{}/{}.spv", out_dir, name);

        let status = std::process::Command::new(&glslc)
            .args([
                "--target-env=vulkan1.3",
                "--target-spv=spv1.5",
                "-O",
                "-o",
                &spv_path,
                src_path.to_str().unwrap(),
            ])
            .status()
            .unwrap_or_else(|e| {
                panic!(
                    "Failed to run glslc ({}). Is the Vulkan SDK installed? Error: {}",
                    glslc, e
                )
            });

        if !status.success() {
            panic!(
                "glslc failed to compile {:?} (exit code: {:?})",
                src_path,
                status.code()
            );
        }
        println!("cargo:rerun-if-changed=shaders/{}.comp", name);
    }

    // Cooperative matrix shaders: need spv1.6, may fail if glslc is too old
    let coopmat_shaders: &[&str] = &["q4_matmul_coopmat", "q4_matmul_coopmat_i8"];
    for name in coopmat_shaders {
        let src_path = shader_dir.join(format!("{}.comp", name));
        if !src_path.exists() {
            continue;
        }
        for &sg in &[32u32, 64] {
            let define_arg = format!("-DSUBGROUP_SIZE={}", sg);
            let spv_path = format!("{}/{}_w{}.spv", out_dir, name, sg);

            let result = std::process::Command::new(&glslc)
                .args([
                    "--target-env=vulkan1.3",
                    "--target-spv=spv1.6",
                    &define_arg,
                    "-O",
                    "-o",
                    &spv_path,
                    src_path.to_str().unwrap(),
                ])
                .status();

            match result {
                Ok(status) if status.success() => {
                    eprintln!("  {}_w{}.spv compiled OK", name, sg);
                }
                _ => {
                    eprintln!(
                        "  WARNING: {}_w{}.spv failed to compile (cooperative_matrix may not be supported by glslc)",
                        name, sg
                    );
                    // Write empty file so include_bytes doesn't fail
                    std::fs::write(&spv_path, &[]).ok();
                }
            }
        }
        println!("cargo:rerun-if-changed=shaders/{}.comp", name);
    }

    // Rerun if any shader source or build script changes
    println!("cargo:rerun-if-changed=shaders/");
    println!("cargo:rerun-if-changed=build.rs");
}
