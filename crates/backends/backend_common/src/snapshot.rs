//! Decode snapshot helpers for offline benchmarking.
//!
//! The snapshot feature captures a single decode step to `bench/snapshots/` by
//! default. The output is raw little-endian binary to keep the dump path simple.

use crate::linear_ops::LinearOps;
use std::collections::BTreeSet;
use std::fs::{self, File};
use std::io::{self, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::OnceLock;

static SNAPSHOT_STEP_COUNTER: AtomicUsize = AtomicUsize::new(0);
static SNAPSHOT_ACTIVE: AtomicBool = AtomicBool::new(false);
static SNAPSHOT_DIR: OnceLock<PathBuf> = OnceLock::new();
static SNAPSHOT_LAYERS: OnceLock<BTreeSet<usize>> = OnceLock::new();
static SNAPSHOT_CAPTURE_STEP: OnceLock<usize> = OnceLock::new();

fn snapshot_dir() -> &'static PathBuf {
    SNAPSHOT_DIR.get_or_init(|| {
        std::env::var("BENCH_SNAPSHOT_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("bench").join("snapshots"))
    })
}

fn snapshot_layers() -> &'static BTreeSet<usize> {
    SNAPSHOT_LAYERS.get_or_init(|| {
        let raw = std::env::var("BENCH_SNAPSHOT_LAYERS").unwrap_or_else(|_| "0,3".to_string());
        let mut layers = BTreeSet::new();
        for part in raw.split(',') {
            let trimmed = part.trim();
            if trimmed.is_empty() {
                continue;
            }
            if let Ok(idx) = trimmed.parse::<usize>() {
                layers.insert(idx);
            }
        }
        if layers.is_empty() {
            layers.insert(0);
            layers.insert(3);
        }
        layers
    })
}

fn capture_step() -> usize {
    *SNAPSHOT_CAPTURE_STEP.get_or_init(|| {
        std::env::var("BENCH_SNAPSHOT_STEP")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0)
    })
}

fn prepare_snapshot_dir() -> io::Result<()> {
    let dir = snapshot_dir();
    if dir.exists() {
        fs::remove_dir_all(dir)?;
    }
    fs::create_dir_all(dir)
}

fn path_for(name: &str) -> PathBuf {
    snapshot_dir().join(format!("{}.bin", name))
}

fn write_bytes(path: &Path, bytes: &[u8]) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut w = BufWriter::new(File::create(path)?);
    w.write_all(bytes)?;
    w.flush()
}

fn read_all_bytes(path: &Path) -> io::Result<Vec<u8>> {
    let mut data = Vec::new();
    File::open(path)?.read_to_end(&mut data)?;
    Ok(data)
}

pub fn begin_decode_step() -> bool {
    let step = SNAPSHOT_STEP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let active = step == capture_step();
    if active {
        if let Err(err) = prepare_snapshot_dir() {
            eprintln!("[SNAPSHOT] failed to prepare {:?}: {}", snapshot_dir(), err);
            SNAPSHOT_ACTIVE.store(false, Ordering::Relaxed);
            return false;
        }
        eprintln!(
            "[SNAPSHOT] capturing decode step {} to {:?} (layers {:?})",
            step,
            snapshot_dir(),
            snapshot_layers(),
        );
    }
    SNAPSHOT_ACTIVE.store(active, Ordering::Relaxed);
    active
}

pub fn end_decode_step() {
    SNAPSHOT_ACTIVE.store(false, Ordering::Relaxed);
}

#[inline]
pub fn is_snapshot_active() -> bool {
    SNAPSHOT_ACTIVE.load(Ordering::Relaxed)
}

#[inline]
pub fn should_snapshot_layer(layer_idx: usize) -> bool {
    snapshot_layers().contains(&layer_idx)
}

pub fn dump_f32(name: &str, data: &[f32]) {
    if !is_snapshot_active() {
        return;
    }
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    if let Err(err) = write_bytes(&path_for(name), bytes) {
        eprintln!("[SNAPSHOT] dump_f32 {} failed: {}", name, err);
    }
}

pub fn dump_bf16(name: &str, data: &[u16]) {
    if !is_snapshot_active() {
        return;
    }
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 2) };
    if let Err(err) = write_bytes(&path_for(name), bytes) {
        eprintln!("[SNAPSHOT] dump_bf16 {} failed: {}", name, err);
    }
}

pub fn dump_u32(name: &str, value: u32) {
    if !is_snapshot_active() {
        return;
    }
    if let Err(err) = write_bytes(&path_for(name), &value.to_le_bytes()) {
        eprintln!("[SNAPSHOT] dump_u32 {} failed: {}", name, err);
    }
}

pub fn dump_usize(name: &str, value: usize) {
    if !is_snapshot_active() {
        return;
    }
    let raw = (value as u64).to_le_bytes();
    if let Err(err) = write_bytes(&path_for(name), &raw) {
        eprintln!("[SNAPSHOT] dump_usize {} failed: {}", name, err);
    }
}

pub fn dump_weight<L: LinearOps>(name: &str, weight: &L::Weight) {
    if !is_snapshot_active() {
        return;
    }
    let path = path_for(name);
    if let Err(err) = L::dump_weight(weight, &path) {
        eprintln!("[SNAPSHOT] dump_weight {} failed: {}", name, err);
    }
}

pub fn load_f32_vec(path: &Path) -> io::Result<Vec<f32>> {
    let bytes = read_all_bytes(path)?;
    if bytes.len() % 4 != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{} is not a multiple of 4 bytes", path.display()),
        ));
    }
    let mut out = vec![0.0f32; bytes.len() / 4];
    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            out.as_mut_ptr() as *mut u8,
            bytes.len(),
        );
    }
    Ok(out)
}

pub fn load_bf16_vec(path: &Path) -> io::Result<Vec<u16>> {
    let bytes = read_all_bytes(path)?;
    if bytes.len() % 2 != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{} is not a multiple of 2 bytes", path.display()),
        ));
    }
    let mut out = vec![0u16; bytes.len() / 2];
    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            out.as_mut_ptr() as *mut u8,
            bytes.len(),
        );
    }
    Ok(out)
}

pub fn load_u32(path: &Path) -> io::Result<u32> {
    let bytes = read_all_bytes(path)?;
    if bytes.len() != 4 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{} does not contain exactly 4 bytes", path.display()),
        ));
    }
    let mut raw = [0u8; 4];
    raw.copy_from_slice(&bytes);
    Ok(u32::from_le_bytes(raw))
}

pub fn load_usize(path: &Path) -> io::Result<usize> {
    let bytes = read_all_bytes(path)?;
    if bytes.len() != 8 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{} does not contain exactly 8 bytes", path.display()),
        ));
    }
    let mut raw = [0u8; 8];
    raw.copy_from_slice(&bytes);
    Ok(u64::from_le_bytes(raw) as usize)
}
