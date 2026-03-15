//! BF16 weight storage — no quantization.
//!
//! Weights stored as row-major [N, K] Vec<BF16> (u16).
//! No scales, no packing, no layout transform.

use herbert_core::tensor::BF16;

/// BF16 weight matrix — raw row-major storage.
pub struct Bf16Weight {
    /// Row-major BF16 data [N, K], 2 bytes/element.
    pub data: Vec<BF16>,
    /// Number of output features (rows).
    pub n: usize,
    /// Number of input features (columns).
    pub k: usize,
}

// ============================================================================
// CacheWeight implementation
// ============================================================================

impl herbert_backend_common::weight_cache::CacheWeight for Bf16Weight {
    // v2: embed_tokens stored as BF16 (was F32)
    const CACHE_VERSION: u32 = 2;

    fn cache_write(&self, w: &mut impl std::io::Write) -> std::io::Result<()> {
        w.write_all(&(self.n as u64).to_le_bytes())?;
        w.write_all(&(self.k as u64).to_le_bytes())?;
        let data_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(self.data.as_ptr() as *const u8, self.data.len() * 2)
        };
        w.write_all(data_bytes)
    }

    fn cache_read(r: &mut impl std::io::Read) -> std::io::Result<Self> {
        let mut buf8 = [0u8; 8];
        r.read_exact(&mut buf8)?;
        let n = u64::from_le_bytes(buf8) as usize;
        r.read_exact(&mut buf8)?;
        let k = u64::from_le_bytes(buf8) as usize;
        let data_len = n * k;
        let mut data = vec![0u16; data_len];
        let data_bytes: &mut [u8] = unsafe {
            std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, data_len * 2)
        };
        r.read_exact(data_bytes)?;
        Ok(Bf16Weight { data, n, k })
    }
}
