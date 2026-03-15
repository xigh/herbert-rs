//! Hardware detection and cache bandwidth microbenchmark for `/arch` command.

use std::time::Instant;

pub fn print_arch() {
    #[cfg(target_os = "linux")]
    print_arch_linux();
    #[cfg(target_os = "macos")]
    print_arch_macos();
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    eprintln!("/arch not supported on this platform");

    print_bandwidth();
}

// ── Linux ───────────────────────────────────────────────────────────────────

#[cfg(target_os = "linux")]
fn print_arch_linux() {
    // CPU
    if let Ok(lscpu) = run_cmd("lscpu", &[]) {
        let get = |key: &str| -> Option<String> {
            lscpu.lines()
                .find(|l| l.starts_with(key))
                .map(|l| l.split_once(':').unwrap_or(("", "")).1.trim().to_string())
        };
        let model = get("Model name").unwrap_or_default();
        let cpus: usize = get("CPU(s)").and_then(|v| v.parse().ok()).unwrap_or(0);
        let tpc: usize = get("Thread(s) per core").and_then(|v| v.parse().ok()).unwrap_or(1);
        let sockets: usize = get("Socket(s)").and_then(|v| v.parse().ok()).unwrap_or(1);
        let phys = cpus / tpc / sockets;
        let freq = get("CPU max MHz")
            .or_else(|| get("CPU MHz"))
            .and_then(|v| v.parse::<f64>().ok())
            .map(|v| format!("{:.2}", v / 1000.0))
            .unwrap_or_default();
        eprintln!("CPU:    {} — {}C/{}T @ {} GHz", model, phys, cpus, freq);

        // SIMD
        if let Some(flags) = get("Flags") {
            let has = |f: &str| flags.split_whitespace().any(|w| w == f);
            if has("avx512f") {
                let mut exts = String::from("F");
                for (flag, tag) in &[
                    ("avx512bw", "BW"), ("avx512vl", "VL"),
                    ("avx512vnni", "VNNI"), ("avx512_bf16", "BF16"),
                ] {
                    if has(flag) { exts.push(' '); exts.push_str(tag); }
                }
                eprintln!("SIMD:   AVX-512 ({})", exts);
            } else if has("avx2") {
                eprintln!("SIMD:   AVX2");
            } else if has("sse4_2") {
                eprintln!("SIMD:   SSE4.2");
            }
        }
    }

    // Cache
    let mut line = String::new();
    let cpus: usize = std::fs::read_to_string("/sys/devices/system/cpu/online")
        .ok()
        .and_then(|s| count_cpu_list(s.trim()))
        .unwrap_or(1);
    for idx in 0..10 {
        let base = format!("/sys/devices/system/cpu/cpu0/cache/index{}", idx);
        let level = match read_sysfs(&format!("{}/level", base)) { Some(v) => v, None => break };
        let ctype = match read_sysfs(&format!("{}/type", base)) { Some(v) => v, None => continue };
        let size = match read_sysfs(&format!("{}/size", base)) { Some(v) => v, None => continue };
        if ctype != "Data" && ctype != "Unified" { continue; }
        let shared = read_sysfs(&format!("{}/shared_cpu_list", base))
            .and_then(|s| count_cpu_list(&s))
            .unwrap_or(1);
        let count = (cpus / shared).max(1);
        let tag = if ctype == "Unified" {
            format!("L{}", level)
        } else {
            format!("L{}d", level)
        };
        line.push_str(&format!("  {} {}×{}", tag, fmt_size(&size), count));
    }
    eprintln!("Cache: {}", line);

    // RAM
    if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
        if let Some(kb) = meminfo.lines()
            .find(|l| l.starts_with("MemTotal"))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|v| v.parse::<u64>().ok())
        {
            let gb = (kb + 524288) / 1048576; // round
            eprintln!("RAM:    {} GiB", gb);
        }
    }
}

#[cfg(target_os = "linux")]
fn read_sysfs(path: &str) -> Option<String> {
    std::fs::read_to_string(path).ok().map(|s| s.trim().to_string())
}

#[cfg(target_os = "linux")]
fn count_cpu_list(s: &str) -> Option<usize> {
    let mut count = 0usize;
    for part in s.split(',') {
        let part = part.trim();
        if let Some((lo, hi)) = part.split_once('-') {
            let lo: usize = lo.parse().ok()?;
            let hi: usize = hi.parse().ok()?;
            count += hi - lo + 1;
        } else {
            let _: usize = part.parse().ok()?;
            count += 1;
        }
    }
    Some(count)
}

// ── macOS ───────────────────────────────────────────────────────────────────

#[cfg(target_os = "macos")]
fn print_arch_macos() {
    // CPU
    let chip = sysctl_str("machdep.cpu.brand_string")
        .or_else(|| {
            run_cmd("system_profiler", &["SPHardwareDataType"]).ok()
                .and_then(|out| out.lines()
                    .find(|l| l.contains("Chip"))
                    .map(|l| l.split_once(':').unwrap_or(("", "")).1.trim().to_string()))
        })
        .unwrap_or_else(|| "Unknown".into());
    let pcores = sysctl_u64("hw.perflevel0.physicalcpu").unwrap_or(0);
    let ecores = sysctl_u64("hw.perflevel1.physicalcpu").unwrap_or(0);
    let total = sysctl_u64("hw.physicalcpu").unwrap_or(pcores + ecores);
    eprintln!("CPU:    {} — {}C ({}P+{}E)", chip, total, pcores, ecores);

    // SIMD
    let mut feats = String::from("NEON");
    for feat in &["FEAT_DotProd", "FEAT_BF16", "FEAT_I8MM"] {
        let key = format!("hw.optional.arm.{}", feat);
        if sysctl_u64(&key).unwrap_or(0) == 1 {
            if feats == "NEON" { feats.push_str(" + "); } else { feats.push(' '); }
            feats.push_str(feat);
        }
    }
    eprintln!("SIMD:   {}", feats);

    // Cache
    let l1d = sysctl_u64("hw.l1dcachesize").unwrap_or(0);
    let l2 = sysctl_u64("hw.l2cachesize").unwrap_or(0);
    let l3 = sysctl_u64("hw.l3cachesize").unwrap_or(0);
    let mut line = format!("L1d {}  L2 {}", fmt_bytes(l1d), fmt_bytes(l2));
    if l3 > 0 {
        line.push_str(&format!("  L3 {}", fmt_bytes(l3)));
    } else {
        line.push_str("  (shared)");
    }
    eprintln!("Cache:  {}", line);

    // RAM
    let mem = sysctl_u64("hw.memsize").unwrap_or(0);
    let mem_gb = mem / (1024 * 1024 * 1024);
    let ram_type = run_cmd("system_profiler", &["SPMemoryDataType"]).ok()
        .and_then(|out| out.lines()
            .find(|l| l.contains("Type:"))
            .map(|l| l.split_once(':').unwrap_or(("", "")).1.trim().to_string()))
        .unwrap_or_default();
    eprintln!("RAM:    {} GiB {}", mem_gb, ram_type);
}

#[cfg(target_os = "macos")]
fn sysctl_str(key: &str) -> Option<String> {
    run_cmd("sysctl", &["-n", key]).ok().map(|s| s.trim().to_string()).filter(|s| !s.is_empty())
}

#[cfg(target_os = "macos")]
fn sysctl_u64(key: &str) -> Option<u64> {
    sysctl_str(key).and_then(|s| s.parse().ok())
}

#[cfg(target_os = "macos")]
fn fmt_bytes(b: u64) -> String {
    if b >= 1024 * 1024 {
        format!("{}M", b / (1024 * 1024))
    } else {
        format!("{}K", b / 1024)
    }
}

// ── Bandwidth microbench ────────────────────────────────────────────────────

fn print_bandwidth() {
    const SIZES: &[(usize, &str)] = &[
        (16 * 1024, "L1"),
        (256 * 1024, "L2"),
        (4 * 1024 * 1024, "L3"),
        (128 * 1024 * 1024, "DRAM"),
    ];

    let mut parts = Vec::new();
    for &(size, name) in SIZES {
        let bw = measure_bw(size);
        parts.push(format!("{} {:.1}", name, bw));
    }
    eprintln!("BW:     {} GB/s", parts.join("  "));
}

fn measure_bw(size: usize) -> f64 {
    let mut buf = vec![1u8; size];
    // Touch all pages
    for i in (0..size).step_by(4096) {
        buf[i] = 1;
    }

    // Calibrate: target ~0.3s worth of iterations
    let elems = size / 64;
    let mut trips: usize = 1;
    while (trips as u64) * (elems as u64) * 64 < 300_000_000 {
        trips *= 2;
    }

    let mut sink: u64 = 0;
    let ptr = buf.as_ptr();
    let start = Instant::now();
    for _ in 0..trips {
        let mut i = 0;
        while i < size {
            unsafe {
                sink = sink.wrapping_add(std::ptr::read_volatile(ptr.add(i)) as u64);
            }
            i += 64;
        }
    }
    let elapsed = start.elapsed().as_secs_f64();

    // Prevent dead-code elimination
    std::hint::black_box(sink);
    let _ = buf;

    let bytes = trips as f64 * size as f64;
    bytes / elapsed / 1e9
}

// ── Helpers ─────────────────────────────────────────────────────────────────

fn run_cmd(cmd: &str, args: &[&str]) -> Result<String, std::io::Error> {
    let output = std::process::Command::new(cmd).args(args).output()?;
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

#[cfg(target_os = "linux")]
fn fmt_size(s: &str) -> String {
    // Convert "1024K" → "1M", "32768K" → "32M", keep "32K" as-is
    if let Some(num_str) = s.strip_suffix('K') {
        if let Ok(num) = num_str.parse::<u64>() {
            if num >= 1024 {
                return format!("{}M", num / 1024);
            }
        }
    }
    s.to_string()
}
