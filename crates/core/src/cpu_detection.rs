//! CPU detection and thread count utilities

/// Get the number of performance CPUs (P-cores on Apple Silicon, physical cores on Linux)
///
/// On macOS, this detects only the performance cores (P-cores) using sysctl.
/// On Linux, this returns the number of physical cores (excluding SMT siblings)
/// by parsing sysfs topology. This gives each thread a full L2 cache.
/// On other platforms, returns the total number of CPUs.
/// Result is cached after the first call.
pub fn get_performance_cpu_count() -> usize {
    use std::sync::OnceLock;
    static COUNT: OnceLock<usize> = OnceLock::new();
    *COUNT.get_or_init(|| {
        #[cfg(target_os = "macos")]
        {
            use std::process::Command;

            // On Apple Silicon, count only P-cores via sysctl
            // hw.perflevel0.logicalcpu = P-cores
            // hw.perflevel1.logicalcpu = E-cores
            let output = Command::new("sysctl")
                .arg("-n")
                .arg("hw.perflevel0.logicalcpu")
                .output()
                .ok();

            if let Some(output) = output {
                if output.status.success() {
                    if let Ok(count_str) = String::from_utf8(output.stdout) {
                        if let Ok(count) = count_str.trim().parse::<usize>() {
                            return count.max(1); // at least 1 thread
                        }
                    }
                }
            }

            // Fallback: use all CPUs
            num_cpus::get()
        }

        #[cfg(target_os = "linux")]
        {
            // Count physical cores by parsing sysfs topology
            let count = count_physical_cores_linux();
            if count > 0 {
                count
            } else {
                num_cpus::get()
            }
        }

        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        {
            num_cpus::get()
        }
    })
}

/// Count physical cores on Linux by parsing thread_siblings_list.
#[cfg(target_os = "linux")]
fn count_physical_cores_linux() -> usize {
    use std::collections::BTreeSet;

    let cpu_dir = std::path::Path::new("/sys/devices/system/cpu");
    let Ok(entries) = std::fs::read_dir(cpu_dir) else {
        return 0;
    };

    let mut seen_cores = BTreeSet::new();

    let cpus: Vec<usize> = entries
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let name = e.file_name();
            let name = name.to_string_lossy();
            name.strip_prefix("cpu")
                .and_then(|n| n.parse::<usize>().ok())
        })
        .collect();

    for cpu_id in cpus {
        let siblings_path = cpu_dir
            .join(format!("cpu{}", cpu_id))
            .join("topology/thread_siblings_list");
        let Ok(siblings_str) = std::fs::read_to_string(&siblings_path) else {
            continue;
        };

        // Parse siblings list and use the smallest as canonical
        let mut min_sibling = usize::MAX;
        for part in siblings_str.trim().split(',') {
            let part = part.trim();
            if let Some((start, _end)) = part.split_once('-') {
                if let Ok(s) = start.parse::<usize>() {
                    min_sibling = min_sibling.min(s);
                }
            } else if let Ok(n) = part.parse::<usize>() {
                min_sibling = min_sibling.min(n);
            }
        }

        if min_sibling != usize::MAX {
            seen_cores.insert(min_sibling);
        }
    }

    seen_cores.len()
}
