//! Pool of MoE experts (all held in memory) with frequency tracking for L3 caching.

use crate::generic_mlp::GenericMLP;
use crate::linear_ops::LinearOps;
use herbert_core::error::Result;

/// Pool of MoE experts. All experts are always resident in memory.
/// Tracks per-expert usage frequency to enable hot-expert-first dispatch ordering,
/// which improves L3 cache residency for frequently-used experts.
pub struct ExpertPool<L: LinearOps> {
    experts: Vec<Option<GenericMLP<L>>>,
    layer_id: usize,
    /// Per-expert usage count (wraps on overflow — relative ordering is what matters).
    usage_counts: Vec<u32>,
}

impl<L: LinearOps> ExpertPool<L> {
    /// Create a pool with ALL experts loaded.
    pub fn new_all_loaded(
        experts: Vec<GenericMLP<L>>,
        layer_id: usize,
    ) -> Self {
        let n = experts.len();
        Self {
            experts: experts.into_iter().map(Some).collect(),
            layer_id,
            usage_counts: vec![0u32; n],
        }
    }

    /// Number of experts (total).
    #[inline]
    pub fn num_experts(&self) -> usize {
        self.experts.len()
    }

    /// Layer ID.
    #[inline]
    pub fn layer_id(&self) -> usize {
        self.layer_id
    }

    /// Get a reference to an expert. Panics if not loaded.
    #[inline]
    pub fn get(&self, expert_id: usize) -> &GenericMLP<L> {
        self.experts[expert_id]
            .as_ref()
            .expect("expert not loaded")
    }

    /// Iterate over all loaded experts.
    pub fn experts(&self) -> impl Iterator<Item = &GenericMLP<L>> {
        self.experts.iter().filter_map(|e| e.as_ref())
    }

    /// Get a pointer to the underlying Option array (for SendPtr in parallel dispatch).
    #[inline]
    pub fn experts_option_ptr(&self) -> *const Option<GenericMLP<L>> {
        self.experts.as_ptr()
    }

    /// No-op: all experts are always loaded.
    #[inline]
    pub fn ensure_loaded(&mut self, _needed_ids: &[usize]) -> Result<()> {
        Ok(())
    }

    /// Record usage of the given expert IDs and return them sorted by frequency
    /// (most frequently used first). This enables hot-expert-first dispatch ordering
    /// for better L3 cache residency.
    ///
    /// `selected` contains (expert_id, weight) pairs. Returns the same pairs
    /// reordered by descending usage frequency.
    pub fn record_and_sort_by_frequency(
        &mut self,
        selected: &mut [(usize, f32)],
        n_sel: usize,
    ) {
        // Update usage counts
        for i in 0..n_sel {
            let eid = selected[i].0;
            self.usage_counts[eid] = self.usage_counts[eid].wrapping_add(1);
        }

        // Sort by descending frequency (stable sort preserves order for equal counts)
        selected[..n_sel].sort_by(|a, b| {
            self.usage_counts[b.0].cmp(&self.usage_counts[a.0])
        });
    }

    /// Iterate over all loaded experts (for serialization in weight_cache).
    pub fn iter_loaded(&self) -> impl Iterator<Item = (usize, &GenericMLP<L>)> {
        self.experts
            .iter()
            .enumerate()
            .filter_map(|(i, e)| e.as_ref().map(|expert| (i, expert)))
    }

    /// Get a reference to the full experts slice (for snapshot dump).
    #[cfg(feature = "bench-decode-snapshot")]
    pub fn all_experts(&self) -> &[Option<GenericMLP<L>>] {
        &self.experts
    }
}
