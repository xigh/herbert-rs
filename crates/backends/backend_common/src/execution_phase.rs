//! Phase-aware execution control for prefill vs decode.
//!
//! During decode, matvec operations are bandwidth-bound and using all cores
//! can be counter-productive due to NUMA/cache contention. This module
//! provides a global phase indicator and a configurable thread cap for decode.

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::OnceLock;

/// Current execution phase of the inference engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ExecutionPhase {
    Prefill = 0,
    Decode = 1,
}

/// Configuration for phase-aware thread management.
pub struct PhaseConfig {
    /// Max worker threads during decode (0 = use all workers, no cap).
    pub decode_max_workers: usize,
    /// Chunk size for prefill chunking (0 = no chunking).
    pub prefill_chunk_size: usize,
}

static CURRENT_PHASE: AtomicU8 = AtomicU8::new(0);
static PHASE_CONFIG: OnceLock<PhaseConfig> = OnceLock::new();

/// Set the current execution phase (called at prefill/decode boundaries).
#[inline]
pub fn set_phase(phase: ExecutionPhase) {
    CURRENT_PHASE.store(phase as u8, Ordering::Release);
}

/// Get the current execution phase.
#[inline]
pub fn current_phase() -> ExecutionPhase {
    match CURRENT_PHASE.load(Ordering::Acquire) {
        1 => ExecutionPhase::Decode,
        _ => ExecutionPhase::Prefill,
    }
}

/// Get the phase configuration (lazily initialized with defaults).
pub fn phase_config() -> &'static PhaseConfig {
    PHASE_CONFIG.get_or_init(|| PhaseConfig {
        decode_max_workers: 0,
        prefill_chunk_size: 0,
    })
}

/// Return the effective max workers for the current phase.
///
/// During prefill, returns `pool_size` (use all workers).
/// During decode, returns `min(decode_max_workers, pool_size)` if configured,
/// otherwise `pool_size`.
#[inline]
pub fn effective_max_workers(pool_size: usize) -> usize {
    match current_phase() {
        ExecutionPhase::Prefill => pool_size,
        ExecutionPhase::Decode => {
            let cap = phase_config().decode_max_workers;
            if cap == 0 {
                pool_size
            } else {
                cap.min(pool_size)
            }
        }
    }
}

/// Configure the phase system. Must be called before inference starts.
///
/// - `decode_max_workers`: max threads during decode (0 = no cap)
/// - `prefill_chunk_size`: split prefill into chunks of this size (0 = no chunking)
///
/// Subsequent calls are ignored (first-write wins).
pub fn configure_phases(decode_max_workers: usize, prefill_chunk_size: usize) {
    let _ = PHASE_CONFIG.set(PhaseConfig {
        decode_max_workers,
        prefill_chunk_size,
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phase_round_trip() {
        // Default is Prefill
        set_phase(ExecutionPhase::Prefill);
        assert_eq!(current_phase(), ExecutionPhase::Prefill);

        set_phase(ExecutionPhase::Decode);
        assert_eq!(current_phase(), ExecutionPhase::Decode);

        set_phase(ExecutionPhase::Prefill);
        assert_eq!(current_phase(), ExecutionPhase::Prefill);
    }

    #[test]
    fn prefill_uses_all_workers() {
        set_phase(ExecutionPhase::Prefill);
        // During prefill, always returns pool_size regardless of config
        assert_eq!(effective_max_workers(16), 16);
        assert_eq!(effective_max_workers(1), 1);
    }

    #[test]
    fn decode_unconfigured_uses_all_workers() {
        // With default config (decode_max_workers=0), decode also uses all workers
        set_phase(ExecutionPhase::Decode);
        let config = phase_config();
        if config.decode_max_workers == 0 {
            assert_eq!(effective_max_workers(16), 16);
        }
        // Restore
        set_phase(ExecutionPhase::Prefill);
    }

    #[test]
    fn phase_config_defaults() {
        let config = phase_config();
        // Default or whatever was configured first by another test —
        // just verify it doesn't panic and returns valid values
        assert!(config.decode_max_workers <= 1024);
        assert!(config.prefill_chunk_size <= 1_000_000);
    }
}
