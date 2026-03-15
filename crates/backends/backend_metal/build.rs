//! Build script: compile MSL shaders into .metallib files at build time.
//!
//! Pipeline: .metal → xcrun metal -c → .air → xcrun metallib → .metallib
//!
//! Produces two libraries:
//!   - herbert.metallib     — MSL 3.1 (Apple7+, all shaders)
//!   - herbert_m4.metallib  — MSL 4.0 (Apple11+, *_coop32.metal only)
//!
//! If Xcode CLI tools are not available, we embed the MSL sources as strings
//! and compile them at runtime.

fn main() {
    #[cfg(target_os = "macos")]
    compile_metal_shaders();

    #[cfg(not(target_os = "macos"))]
    {
        // Nothing to do on non-macOS platforms
    }
}

#[cfg(target_os = "macos")]
fn compile_metal_shaders() {
    use std::path::{Path, PathBuf};
    use std::process::Command;

    let shader_dir = Path::new("shaders");
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());

    // Collect all .metal files, respecting feature-gated shader variants.
    // simdgroup-matmul feature: use q4_matmul_tiled_simdgroup.metal, skip q4_matmul_tiled.metal
    // default: use q4_matmul_tiled.metal, skip q4_matmul_tiled_simdgroup.metal
    let use_simdgroup = std::env::var("CARGO_FEATURE_SIMDGROUP_MATMUL").is_ok();
    let skip_shader = if use_simdgroup {
        "q4_matmul_tiled.metal"
    } else {
        "q4_matmul_tiled_simdgroup.metal"
    };

    let mut metal_files: Vec<PathBuf> = Vec::new();
    let mut metal4_files: Vec<PathBuf> = Vec::new();

    if let Ok(entries) = std::fs::read_dir(shader_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map_or(false, |e| e == "metal") {
                let name = path.file_name().unwrap().to_str().unwrap_or("");
                if name == skip_shader {
                    continue;
                }
                // Metal 4 shaders (*_coop32.metal, *_mpp.metal) go into the M4 library only
                if name.contains("_coop32") || name.contains("_mpp") {
                    metal4_files.push(path);
                } else {
                    metal_files.push(path);
                }
            }
        }
    }

    if metal_files.is_empty() {
        eprintln!("cargo:warning=No .metal shader files found in shaders/");
        return;
    }

    // Tell cargo to rerun if any shader changes
    println!("cargo:rerun-if-changed=shaders/");
    for f in metal_files.iter().chain(metal4_files.iter()) {
        println!("cargo:rerun-if-changed={}", f.display());
    }

    // Check if xcrun is available
    let xcrun_check = Command::new("xcrun")
        .arg("--find")
        .arg("metal")
        .output();

    if xcrun_check.is_err() || !xcrun_check.unwrap().status.success() {
        eprintln!("cargo:warning=Xcode CLI tools not found; will use runtime MSL compilation");
        // Write a marker so shaders.rs knows to use runtime compilation
        std::fs::write(out_dir.join("use_runtime_compilation"), "1").unwrap();
        return;
    }

    // ---- Compile MSL 3.1 shaders → herbert.metallib ----
    let air_files = compile_shaders_to_air(
        &metal_files, &out_dir,
        "air64-apple-macos14.0", "metal3.1", "",
    );
    link_metallib(&air_files, &out_dir, "herbert.metallib", "METAL_LIB_PATH");

    // ---- Compile MSL 4.0 shaders → herbert_m4.metallib (if any exist) ----
    if !metal4_files.is_empty() {
        if has_metal4_sdk() {
            let m4_air_files = compile_shaders_to_air(
                &metal4_files, &out_dir,
                "air64-apple-macos26.0", "metal4.0", "_m4",
            );
            link_metallib(&m4_air_files, &out_dir, "herbert_m4.metallib", "METAL4_LIB_PATH");
        } else {
            eprintln!("cargo:warning=Metal 4 shaders found but Xcode SDK < 26; skipping MSL 4.0 compilation");
        }
    }
}

/// Compile a set of .metal files to .air files.
#[cfg(target_os = "macos")]
fn compile_shaders_to_air(
    metal_files: &[std::path::PathBuf],
    out_dir: &std::path::Path,
    target: &str,
    std_version: &str,
    suffix: &str,
) -> Vec<std::path::PathBuf> {
    use std::process::Command;

    let mut air_files = Vec::new();
    for metal_file in metal_files {
        let stem = metal_file.file_stem().unwrap().to_str().unwrap();
        let air_path = out_dir.join(format!("{}{}.air", stem, suffix));

        let status = Command::new("xcrun")
            .args(["-sdk", "macosx", "metal"])
            .args(["-c", "-target", target])
            .arg(format!("-std={}", std_version))
            .arg("-O2")
            .arg(metal_file)
            .arg("-o")
            .arg(&air_path)
            .status()
            .expect("Failed to run xcrun metal");

        if !status.success() {
            panic!("Metal shader compilation failed for {} ({})", metal_file.display(), std_version);
        }
        air_files.push(air_path);
    }
    air_files
}

/// Link .air files into a .metallib and set a cargo env var.
#[cfg(target_os = "macos")]
fn link_metallib(
    air_files: &[std::path::PathBuf],
    out_dir: &std::path::Path,
    lib_name: &str,
    env_var: &str,
) {
    use std::process::Command;

    let metallib_path = out_dir.join(lib_name);
    let mut cmd = Command::new("xcrun");
    cmd.args(["-sdk", "macosx", "metallib"]);
    for air in air_files {
        cmd.arg(air);
    }
    cmd.arg("-o").arg(&metallib_path);

    let status = cmd.status().expect("Failed to run xcrun metallib");
    if !status.success() {
        panic!("metallib linking failed for {}", lib_name);
    }

    println!("cargo:rustc-env={}={}", env_var, metallib_path.display());
}

/// Check if the installed Xcode SDK supports Metal 4 (macOS 26+).
#[cfg(target_os = "macos")]
fn has_metal4_sdk() -> bool {
    use std::process::Command;

    let output = Command::new("xcrun")
        .args(["--sdk", "macosx", "--show-sdk-version"])
        .output();

    match output {
        Ok(o) if o.status.success() => {
            let version = String::from_utf8_lossy(&o.stdout);
            let version = version.trim();
            // macOS 26.0+ → Metal 4 / MSL 4.0
            if let Some(major) = version.split('.').next() {
                major.parse::<u32>().map_or(false, |v| v >= 26)
            } else {
                false
            }
        }
        _ => false,
    }
}
