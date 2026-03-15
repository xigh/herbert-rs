//! Prefix KV cache: save/load pre-computed KV states to skip redundant prefills.
//!
//! When the same prefix tokens (system prompt, few-shot examples) are reused across
//! inference runs, the KV cache for those tokens can be loaded from disk in ~15ms
//! instead of recomputing via prefill (~500-2000ms).
//!
//! Cache files are stored in `<cache_dir>/<model_hash_hex>/prefix_<token_hash>_<len>.kvcache`.

use crate::kv_cache::CpuKvCache;
use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};

// ============================================================================
// Constants
// ============================================================================

const MAGIC: &[u8; 8] = b"QW3PKCCH";
const HEADER_SIZE: usize = 80;
const DEFAULT_MAX_SIZE_BYTES: u64 = 10 * 1024 * 1024 * 1024; // 10 GB
const DEFAULT_MIN_PREFIX_LEN: usize = 16;
const DEFAULT_CHECKPOINT_INTERVAL: usize = 16;

// Format version: v1 = BF16 (only BF16 supports prefix cache currently)
const FORMAT_VERSION: u32 = 1;

// ============================================================================
// Config
// ============================================================================

/// Configuration for the prefix KV cache.
#[derive(Debug, Clone)]
pub struct PrefixCacheConfig {
    /// Directory to store cache files.
    pub cache_dir: PathBuf,
    /// Maximum total size of all cache files in bytes.
    pub max_size_bytes: u64,
    /// Minimum prefix length (in tokens) to cache.
    pub min_prefix_len: usize,
    /// Round prefix lengths down to multiples of this for cache keys.
    pub checkpoint_interval: usize,
}

impl PrefixCacheConfig {
    pub fn new(cache_dir: PathBuf) -> Self {
        Self {
            cache_dir,
            max_size_bytes: DEFAULT_MAX_SIZE_BYTES,
            min_prefix_len: DEFAULT_MIN_PREFIX_LEN,
            checkpoint_interval: DEFAULT_CHECKPOINT_INTERVAL,
        }
    }
}

// ============================================================================
// Index entry
// ============================================================================

#[derive(Debug, Clone)]
struct IndexEntry {
    prefix_len: usize,
    token_hash: u64,
    file_name: String,
    file_size: u64,
    last_access: u64,
}

// ============================================================================
// Token hashing
// ============================================================================

fn hash_tokens(tokens: &[u32]) -> u64 {
    let mut hasher = DefaultHasher::new();
    tokens.hash(&mut hasher);
    hasher.finish()
}

// ============================================================================
// File header
// ============================================================================

#[derive(Debug)]
struct PrefixCacheHeader {
    kv_dim: u32,
    num_layers: u32,
    seq_len: u32,
    model_hash: [u64; 2],
    token_hash: u64,
}

impl PrefixCacheHeader {
    fn write(&self, w: &mut impl Write) -> io::Result<()> {
        let mut buf = [0u8; HEADER_SIZE];
        buf[0..8].copy_from_slice(MAGIC);
        buf[8..12].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
        buf[12..16].copy_from_slice(&self.kv_dim.to_le_bytes());
        buf[16..20].copy_from_slice(&self.num_layers.to_le_bytes());
        buf[20..24].copy_from_slice(&self.seq_len.to_le_bytes());
        buf[24..32].copy_from_slice(&self.model_hash[0].to_le_bytes());
        buf[32..40].copy_from_slice(&self.model_hash[1].to_le_bytes());
        buf[40..48].copy_from_slice(&self.token_hash.to_le_bytes());
        // 48..80 reserved (zeros)
        w.write_all(&buf)
    }

    fn read(r: &mut impl Read) -> io::Result<Self> {
        let mut buf = [0u8; HEADER_SIZE];
        r.read_exact(&mut buf)?;
        if &buf[0..8] != MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "bad prefix cache magic"));
        }
        let version = u32::from_le_bytes(buf[8..12].try_into().expect("4-byte slice"));
        if version != FORMAT_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported prefix cache version {} (expected {})", version, FORMAT_VERSION),
            ));
        }
        Ok(Self {
            kv_dim: u32::from_le_bytes(buf[12..16].try_into().expect("4-byte slice")),
            num_layers: u32::from_le_bytes(buf[16..20].try_into().expect("4-byte slice")),
            seq_len: u32::from_le_bytes(buf[20..24].try_into().expect("4-byte slice")),
            model_hash: [
                u64::from_le_bytes(buf[24..32].try_into().expect("8-byte slice")),
                u64::from_le_bytes(buf[32..40].try_into().expect("8-byte slice")),
            ],
            token_hash: u64::from_le_bytes(buf[40..48].try_into().expect("8-byte slice")),
        })
    }
}

// ============================================================================
// PrefixCache
// ============================================================================

/// Prefix KV cache manager.
pub struct PrefixCache {
    config: PrefixCacheConfig,
    model_hash: [u64; 2],
    model_dir: PathBuf,
    index: Vec<IndexEntry>,
    num_layers: usize,
    kv_dim: usize,
    access_clock: u64,
}

impl PrefixCache {
    /// Create a new prefix cache for the given model.
    ///
    /// `model_hash` identifies the model (from `weight_cache::compute_model_hash`).
    /// `num_layers` and `kv_dim` must match the model config.
    pub fn new(
        config: PrefixCacheConfig,
        model_hash: [u64; 2],
        num_layers: usize,
        kv_dim: usize,
    ) -> io::Result<Self> {
        let hash_hex = format!("{:016x}{:016x}", model_hash[0], model_hash[1]);
        let model_dir = config.cache_dir.join(hash_hex);
        fs::create_dir_all(&model_dir)?;

        // Scan existing cache files to build the index
        let mut index = Vec::new();
        if let Ok(entries) = fs::read_dir(&model_dir) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if !name.starts_with("prefix_") || !name.ends_with(".kvcache") {
                    continue;
                }
                // Parse: prefix_<token_hash>_<len>.kvcache
                let stem = name.trim_start_matches("prefix_").trim_end_matches(".kvcache");
                let parts: Vec<&str> = stem.rsplitn(2, '_').collect();
                if parts.len() != 2 {
                    continue;
                }
                let prefix_len = match parts[0].parse::<usize>() {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                let token_hash = match u64::from_str_radix(parts[1], 16) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                let file_size = entry.metadata().map(|m| m.len()).unwrap_or(0);
                index.push(IndexEntry {
                    prefix_len,
                    token_hash,
                    file_name: name,
                    file_size,
                    last_access: 0,
                });
            }
        }
        // Sort by prefix_len descending (for longest-match-first lookup)
        index.sort_by(|a, b| b.prefix_len.cmp(&a.prefix_len));

        let count = index.len();
        if count > 0 {
            info!(entries = count, dir = %model_dir.display(), "Prefix cache loaded");
        } else {
            debug!(dir = %model_dir.display(), "Prefix cache initialized (empty)");
        }

        Ok(Self {
            config,
            model_hash,
            model_dir,
            index,
            num_layers,
            kv_dim,
            access_clock: 0,
        })
    }

    /// Find the longest cached prefix that matches the start of `tokens`.
    ///
    /// Returns `(prefix_len, file_path)` on hit, or `None` on miss.
    pub fn lookup(&mut self, tokens: &[u32]) -> Option<(usize, PathBuf)> {
        if tokens.len() < self.config.min_prefix_len {
            return None;
        }

        self.access_clock += 1;

        // Index is sorted by prefix_len descending, so first match is longest
        for entry in &mut self.index {
            if entry.prefix_len > tokens.len() {
                continue;
            }
            let candidate_hash = hash_tokens(&tokens[..entry.prefix_len]);
            if candidate_hash == entry.token_hash {
                entry.last_access = self.access_clock;
                let path = self.model_dir.join(&entry.file_name);
                if path.exists() {
                    info!(
                        prefix_len = entry.prefix_len,
                        total_tokens = tokens.len(),
                        "Prefix cache HIT"
                    );
                    return Some((entry.prefix_len, path));
                }
            }
        }
        debug!(total_tokens = tokens.len(), "Prefix cache MISS");
        None
    }

    /// Load KV cache data from a prefix cache file into an existing `CpuKvCache`.
    ///
    /// Returns the number of tokens loaded (= seq_len from the file).
    pub fn load_kv_data(&self, path: &Path, kv_cache: &mut CpuKvCache) -> io::Result<usize> {
        let file = fs::File::open(path)?;
        let mut r = io::BufReader::with_capacity(4 * 1024 * 1024, file);

        let header = PrefixCacheHeader::read(&mut r)?;

        // Validate against model params
        if header.model_hash != self.model_hash {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "model hash mismatch"));
        }
        if header.num_layers as usize != self.num_layers {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "num_layers mismatch: file={}, expected={}",
                    header.num_layers, self.num_layers
                ),
            ));
        }
        if header.kv_dim as usize != self.kv_dim {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "kv_dim mismatch: file={}, expected={}",
                    header.kv_dim, self.kv_dim
                ),
            ));
        }

        let seq_len = header.seq_len as usize;
        let elems_per_layer = seq_len * self.kv_dim;

        for layer_idx in 0..self.num_layers {
            // Read keys
            kv_cache.kv.keys[layer_idx].resize(elems_per_layer, 0);
            let key_bytes = unsafe {
                std::slice::from_raw_parts_mut(
                    kv_cache.kv.keys[layer_idx].as_mut_ptr() as *mut u8,
                    elems_per_layer * 2,
                )
            };
            r.read_exact(key_bytes)?;

            // Read values
            kv_cache.kv.values[layer_idx].resize(elems_per_layer, 0);
            let val_bytes = unsafe {
                std::slice::from_raw_parts_mut(
                    kv_cache.kv.values[layer_idx].as_mut_ptr() as *mut u8,
                    elems_per_layer * 2,
                )
            };
            r.read_exact(val_bytes)?;
        }

        // Reconstruct per-head BF16 layout in layer_data from flat data
        let num_kv_heads = kv_cache.kv.num_kv_heads;
        let head_dim = kv_cache.kv.head_dim;
        for layer_idx in 0..self.num_layers {
            if let crate::kv_cache::KvLayerData::BF16 { keys_per_head, values_per_head } = &mut kv_cache.kv.layer_data[layer_idx] {
                for kv_h in 0..num_kv_heads {
                    keys_per_head[kv_h].clear();
                    values_per_head[kv_h].clear();
                }
                for pos in 0..seq_len {
                    for kv_h in 0..num_kv_heads {
                        let src = pos * self.kv_dim + kv_h * head_dim;
                        keys_per_head[kv_h].extend_from_slice(&kv_cache.kv.keys[layer_idx][src..src + head_dim]);
                        values_per_head[kv_h].extend_from_slice(&kv_cache.kv.values[layer_idx][src..src + head_dim]);
                    }
                }
            }
        }

        kv_cache.kv.seq_len = seq_len;

        info!(seq_len, kv_dim = self.kv_dim, num_layers = self.num_layers, "Prefix cache loaded");
        Ok(seq_len)
    }

    /// Save the current KV cache as a prefix cache entry.
    ///
    /// The prefix is rounded down to the nearest `checkpoint_interval`.
    pub fn save(&mut self, tokens: &[u32], kv_cache: &CpuKvCache) -> io::Result<()> {
        let interval = self.config.checkpoint_interval.max(1);
        let save_len = (tokens.len() / interval) * interval;

        if save_len < self.config.min_prefix_len {
            debug!(
                save_len,
                min = self.config.min_prefix_len,
                "Prefix too short to cache"
            );
            return Ok(());
        }

        self.save_inner(&tokens[..save_len], kv_cache)
    }

    /// Save the system prefix KV cache (exact length, no rounding or minimum check).
    ///
    /// Used for double prefill: saves the KV state for the system prompt so that
    /// subsequent runs with the same system prompt can skip recomputing it.
    pub fn save_system(&mut self, tokens: &[u32], kv_cache: &CpuKvCache) -> io::Result<()> {
        if tokens.is_empty() {
            return Ok(());
        }
        self.save_inner(tokens, kv_cache)
    }

    /// Internal: save the first `tokens.len()` positions of the KV cache to disk.
    fn save_inner(&mut self, tokens: &[u32], kv_cache: &CpuKvCache) -> io::Result<()> {
        let save_len = tokens.len();
        let token_hash = hash_tokens(tokens);

        // Check if already cached
        if self.index.iter().any(|e| e.prefix_len == save_len && e.token_hash == token_hash) {
            debug!(save_len, "Prefix already cached");
            return Ok(());
        }

        let file_name = format!("prefix_{:016x}_{}.kvcache", token_hash, save_len);
        let file_path = self.model_dir.join(&file_name);
        let tmp_path = file_path.with_extension("tmp");

        let file = fs::File::create(&tmp_path)?;
        let mut w = io::BufWriter::with_capacity(4 * 1024 * 1024, file);

        let header = PrefixCacheHeader {
            kv_dim: self.kv_dim as u32,
            num_layers: self.num_layers as u32,
            seq_len: save_len as u32,
            model_hash: self.model_hash,
            token_hash,
        };
        header.write(&mut w)?;

        let elems_per_layer = save_len * self.kv_dim;

        let zeros = vec![0u8; elems_per_layer * 2];
        for layer_idx in 0..self.num_layers {
            // DeltaNet layers have empty KV vectors (they use recurrent state instead).
            // Write zeros for those layers to keep the file format consistent.
            let has_kv = kv_cache.kv.keys[layer_idx].len() >= elems_per_layer;

            // Write keys (only first save_len * kv_dim elements)
            if has_kv {
                let key_data = &kv_cache.kv.keys[layer_idx][..elems_per_layer];
                let key_bytes = unsafe {
                    std::slice::from_raw_parts(key_data.as_ptr() as *const u8, elems_per_layer * 2)
                };
                w.write_all(key_bytes)?;
            } else {
                w.write_all(&zeros)?;
            }

            // Write values
            if has_kv {
                let val_data = &kv_cache.kv.values[layer_idx][..elems_per_layer];
                let val_bytes = unsafe {
                    std::slice::from_raw_parts(val_data.as_ptr() as *const u8, elems_per_layer * 2)
                };
                w.write_all(val_bytes)?;
            } else {
                w.write_all(&zeros)?;
            }
        }

        w.flush()?;
        drop(w);

        // Atomic rename
        fs::rename(&tmp_path, &file_path)?;

        let file_size = fs::metadata(&file_path)?.len();
        info!(
            save_len,
            file = %file_name,
            size_mb = file_size / (1024 * 1024),
            "Prefix cache saved"
        );

        self.index.push(IndexEntry {
            prefix_len: save_len,
            token_hash,
            file_name,
            file_size,
            last_access: self.access_clock,
        });
        // Re-sort: longest first
        self.index.sort_by(|a, b| b.prefix_len.cmp(&a.prefix_len));

        // Evict if over budget
        self.evict_if_needed()?;

        Ok(())
    }

    /// Evict oldest entries if total cache size exceeds the limit.
    fn evict_if_needed(&mut self) -> io::Result<()> {
        let mut total: u64 = self.index.iter().map(|e| e.file_size).sum();
        if total <= self.config.max_size_bytes {
            return Ok(());
        }

        // Sort by last_access ascending (oldest first) for eviction
        // We'll pick from the end of the sorted-by-access list
        let mut evict_order: Vec<usize> = (0..self.index.len()).collect();
        evict_order.sort_by_key(|&i| self.index[i].last_access);

        let mut evicted = 0;
        for &idx in &evict_order {
            if total <= self.config.max_size_bytes {
                break;
            }
            let entry = &self.index[idx];
            let path = self.model_dir.join(&entry.file_name);
            if let Err(e) = fs::remove_file(&path) {
                warn!(file = %entry.file_name, error = %e, "Failed to evict prefix cache file");
            } else {
                total -= entry.file_size;
                evicted += 1;
                debug!(file = %entry.file_name, "Evicted prefix cache entry");
            }
        }

        if evicted > 0 {
            // Remove evicted entries from index
            self.index.retain(|e| self.model_dir.join(&e.file_name).exists());
            self.index.sort_by(|a, b| b.prefix_len.cmp(&a.prefix_len));
            info!(evicted, "Prefix cache eviction complete");
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_tokens_deterministic() {
        let tokens = vec![1u32, 2, 3, 4, 5];
        let h1 = hash_tokens(&tokens);
        let h2 = hash_tokens(&tokens);
        assert_eq!(h1, h2);
    }

    #[test]
    fn hash_tokens_different_for_different_inputs() {
        let h1 = hash_tokens(&[1, 2, 3]);
        let h2 = hash_tokens(&[1, 2, 4]);
        assert_ne!(h1, h2);
    }

    #[test]
    fn hash_tokens_prefix_differs() {
        let h1 = hash_tokens(&[1, 2, 3]);
        let h2 = hash_tokens(&[1, 2, 3, 4]);
        assert_ne!(h1, h2);
    }

    #[test]
    fn header_round_trip() {
        let header = PrefixCacheHeader {
            kv_dim: 1024,
            num_layers: 36,
            seq_len: 512,
            model_hash: [0xDEADBEEF, 0xCAFEBABE],
            token_hash: 0x123456789ABCDEF0,
        };

        let mut buf = Vec::new();
        header.write(&mut buf).unwrap();
        assert_eq!(buf.len(), HEADER_SIZE);

        let mut cursor = io::Cursor::new(&buf);
        let read_back = PrefixCacheHeader::read(&mut cursor).unwrap();
        assert_eq!(read_back.kv_dim, 1024);
        assert_eq!(read_back.num_layers, 36);
        assert_eq!(read_back.seq_len, 512);
        assert_eq!(read_back.model_hash, [0xDEADBEEF, 0xCAFEBABE]);
        assert_eq!(read_back.token_hash, 0x123456789ABCDEF0);
    }
}
