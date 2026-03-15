//! CPU topology detection for NUMA-aware thread scheduling.
//!
//! On Linux, parses sysfs to discover NUMA nodes, cores per node,
//! and physical core IDs for thread affinity pinning.
//! On macOS (and other platforms), assumes UMA with a single node.

use std::sync::OnceLock;

/// Describes the CPU topology relevant to inference scheduling.
#[derive(Debug, Clone)]
pub struct Topology {
    /// Number of NUMA nodes (1 for UMA systems like macOS).
    pub num_numa_nodes: usize,
    /// Physical cores per NUMA node.
    pub cores_per_node: usize,
}

/// Detect the CPU topology of the current system.
pub fn detect_topology() -> Topology {
    #[cfg(target_os = "linux")]
    {
        detect_topology_linux()
    }
    #[cfg(not(target_os = "linux"))]
    {
        detect_topology_fallback()
    }
}

/// Returns sorted list of logical CPU IDs — one per physical core, skipping SMT siblings.
///
/// On Linux: parses `/sys/devices/system/cpu/cpu*/topology/thread_siblings_list`
/// to pick the first logical CPU per physical core.
/// On non-Linux: returns empty vec (no affinity support).
pub fn physical_core_ids() -> &'static [usize] {
    static IDS: OnceLock<Vec<usize>> = OnceLock::new();
    IDS.get_or_init(|| {
        #[cfg(target_os = "linux")]
        {
            physical_core_ids_linux()
        }
        #[cfg(not(target_os = "linux"))]
        {
            Vec::new()
        }
    })
}

#[cfg(target_os = "linux")]
fn physical_core_ids_linux() -> Vec<usize> {
    // Try CCD-aware distribution first, fall back to sequential
    let ccd_result = physical_core_ids_ccd_aware();
    if !ccd_result.is_empty() {
        return ccd_result;
    }

    physical_core_ids_sequential()
}

/// CCD-aware core distribution: detects L3 cache groups (CCDs) and distributes
/// cores round-robin across them for maximum aggregate DRAM bandwidth.
///
/// On EPYC 7443P (3 CCDs × 8 cores): instead of [0,1,...,11] (CCD0×8, CCD1×4),
/// produces [0,8,16,1,9,17,...] (CCD0×4, CCD1×4, CCD2×4).
#[cfg(target_os = "linux")]
fn physical_core_ids_ccd_aware() -> Vec<usize> {
    use std::collections::{BTreeMap, BTreeSet};

    let cpu_dir = std::path::Path::new("/sys/devices/system/cpu");
    let Ok(entries) = std::fs::read_dir(cpu_dir) else {
        return Vec::new();
    };

    let mut cpus: Vec<usize> = entries
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let name = e.file_name();
            let name = name.to_string_lossy();
            name.strip_prefix("cpu")
                .and_then(|n| n.parse::<usize>().ok())
        })
        .collect();
    cpus.sort_unstable();

    // Step 1: Get physical cores (skip SMT siblings)
    let mut seen_cores = BTreeSet::new();
    let mut physical_cores = Vec::new();

    for &cpu_id in &cpus {
        let siblings_path = cpu_dir
            .join(format!("cpu{}", cpu_id))
            .join("topology/thread_siblings_list");
        let Ok(siblings_str) = std::fs::read_to_string(&siblings_path) else {
            continue;
        };
        let siblings = parse_cpu_list(siblings_str.trim());
        if siblings.is_empty() {
            continue;
        }
        let canonical = *siblings.iter().min().expect("siblings non-empty");
        if seen_cores.insert(canonical) {
            physical_cores.push(cpu_id);
        }
    }

    if physical_cores.is_empty() {
        return Vec::new();
    }

    // Step 2: Detect CCD groups via L3 cache (index3/shared_cpu_list)
    // Each CCD has its own L3 cache; cores sharing an L3 are on the same CCD.
    let mut ccd_groups: BTreeMap<Vec<usize>, Vec<usize>> = BTreeMap::new();

    for &cpu_id in &physical_cores {
        let l3_path = cpu_dir
            .join(format!("cpu{}", cpu_id))
            .join("cache/index3/shared_cpu_list");
        let l3_key = if let Ok(l3_str) = std::fs::read_to_string(&l3_path) {
            let mut key = parse_cpu_list(l3_str.trim());
            key.sort_unstable();
            key
        } else {
            // No L3 info — treat each core as its own group (degrades to sequential)
            vec![cpu_id]
        };

        ccd_groups.entry(l3_key).or_default().push(cpu_id);
    }

    let num_ccds = ccd_groups.len();
    if num_ccds <= 1 {
        // Single CCD (or couldn't detect multiple): sequential is fine
        return Vec::new();
    }

    // Step 3: Round-robin distribution across CCDs
    // Sort CCD groups by their first core ID for deterministic ordering
    let mut ccds: Vec<Vec<usize>> = ccd_groups.into_values().collect();
    ccds.sort_by_key(|group| group.first().copied().unwrap_or(0));

    let max_cores_per_ccd = ccds.iter().map(|g| g.len()).max().unwrap_or(0);
    let mut result = Vec::new();

    for slot in 0..max_cores_per_ccd {
        for ccd in &ccds {
            if slot < ccd.len() {
                result.push(ccd[slot]);
            }
        }
    }

    tracing::info!(
        num_ccds,
        cores = result.len(),
        distribution = ?result.iter().take(12).collect::<Vec<_>>(),
        "CCD-aware thread distribution"
    );

    result
}

/// Sequential physical core IDs (original behavior, used as fallback).
#[cfg(target_os = "linux")]
fn physical_core_ids_sequential() -> Vec<usize> {
    use std::collections::BTreeSet;

    let cpu_dir = std::path::Path::new("/sys/devices/system/cpu");
    let mut seen_cores = BTreeSet::new();
    let mut result = Vec::new();

    let Ok(entries) = std::fs::read_dir(cpu_dir) else {
        return Vec::new();
    };

    let mut cpus: Vec<usize> = entries
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let name = e.file_name();
            let name = name.to_string_lossy();
            name.strip_prefix("cpu")
                .and_then(|n| n.parse::<usize>().ok())
        })
        .collect();
    cpus.sort_unstable();

    for cpu_id in cpus {
        let siblings_path = cpu_dir
            .join(format!("cpu{}", cpu_id))
            .join("topology/thread_siblings_list");
        let Ok(siblings_str) = std::fs::read_to_string(&siblings_path) else {
            continue;
        };
        let siblings = parse_cpu_list(siblings_str.trim());
        if siblings.is_empty() {
            continue;
        }
        let canonical = *siblings.iter().min().expect("siblings non-empty");
        if seen_cores.insert(canonical) {
            result.push(cpu_id);
        }
    }

    result.sort_unstable();
    result
}

/// Parse a CPU list string like "0,8" or "0-3" or "0-3,8-11" into a Vec of CPU IDs.
#[cfg(target_os = "linux")]
fn parse_cpu_list(s: &str) -> Vec<usize> {
    let mut result = Vec::new();
    for part in s.split(',') {
        let part = part.trim();
        if let Some((start, end)) = part.split_once('-') {
            if let (Ok(s), Ok(e)) = (start.parse::<usize>(), end.parse::<usize>()) {
                for i in s..=e {
                    result.push(i);
                }
            }
        } else if let Ok(n) = part.parse::<usize>() {
            result.push(n);
        }
    }
    result
}

#[cfg(target_os = "linux")]
fn detect_topology_linux() -> Topology {
    // Try to enumerate NUMA nodes from sysfs
    let numa_dir = std::path::Path::new("/sys/devices/system/node");
    if numa_dir.exists() {
        if let Ok(entries) = std::fs::read_dir(numa_dir) {
            let node_count = entries
                .filter_map(|e| e.ok())
                .filter(|e| {
                    e.file_name()
                        .to_string_lossy()
                        .starts_with("node")
                })
                .filter(|e| {
                    // Only count directories named "node0", "node1", etc.
                    let name = e.file_name();
                    let name = name.to_string_lossy();
                    name.strip_prefix("node")
                        .map(|n| n.parse::<usize>().is_ok())
                        .unwrap_or(false)
                })
                .count();

            if node_count > 0 {
                let total_cores = std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(1);
                return Topology {
                    num_numa_nodes: node_count,
                    cores_per_node: total_cores / node_count,
                };
            }
        }
    }

    // Fallback if sysfs is unavailable
    detect_topology_fallback()
}

fn detect_topology_fallback() -> Topology {
    let total_cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    Topology {
        num_numa_nodes: 1,
        cores_per_node: total_cores,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_topology_returns_sane_values() {
        let topo = detect_topology();
        assert!(topo.num_numa_nodes >= 1, "must have at least 1 NUMA node");
        assert!(topo.cores_per_node >= 1, "must have at least 1 core per node");
        // Total cores should be reasonable
        let total = topo.num_numa_nodes * topo.cores_per_node;
        assert!(total >= 1 && total <= 4096, "total cores {} out of range", total);
    }

    #[test]
    fn fallback_is_uma() {
        let topo = detect_topology_fallback();
        assert_eq!(topo.num_numa_nodes, 1);
        assert!(topo.cores_per_node >= 1);
    }

    #[test]
    fn physical_core_ids_not_empty_or_sane() {
        let ids = physical_core_ids();
        // On macOS this will be empty, on Linux it should have entries
        if !ids.is_empty() {
            // Should be sorted
            for w in ids.windows(2) {
                assert!(w[0] < w[1], "physical_core_ids should be sorted and unique");
            }
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn parse_cpu_list_works() {
        assert_eq!(parse_cpu_list("0,8"), vec![0, 8]);
        assert_eq!(parse_cpu_list("0-3"), vec![0, 1, 2, 3]);
        assert_eq!(parse_cpu_list("0-1,8-9"), vec![0, 1, 8, 9]);
        assert_eq!(parse_cpu_list("5"), vec![5]);
    }
}
