//! Runtime auto-tuning helpers for kernel scheduling.
//!
//! The tuner decides between sequential and parallel execution based on
//! workload units and continuously nudges thresholds using sampled timings.
//!
//! **Exploration**: when all observed calls go parallel (or all sequential),
//! the tuner periodically forces the opposite mode on a small workload so
//! it can collect a baseline and start comparing.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

const GLOBAL_AUTOTUNE_KEY: &str = "HERBERT_AUTOTUNE";
const GLOBAL_SAMPLE_RATE_KEY: &str = "HERBERT_AUTOTUNE_SAMPLE_RATE";
const GLOBAL_LOG_KEY: &str = "HERBERT_AUTOTUNE_LOG";
const GLOBAL_LOG_EVERY_KEY: &str = "HERBERT_AUTOTUNE_LOG_EVERY";
const GLOBAL_PAR_THRESHOLD_KEY: &str = "HERBERT_AUTOTUNE_PAR_THRESHOLD";
const GLOBAL_TARGET_PER_WORKER_KEY: &str = "HERBERT_AUTOTUNE_TARGET_WORK_PER_WORKER";
const GLOBAL_EXPLORE_KEY: &str = "HERBERT_AUTOTUNE_EXPLORE";

/// Minimum number of samples on each side before exploration stops.
const EXPLORE_MIN_SAMPLES: u64 = 4;

#[derive(Clone, Copy, Debug)]
pub struct RuntimeAutoTuneDefaults {
    pub threshold_units: usize,
    pub threshold_min: usize,
    pub threshold_max: usize,
    pub target_units_per_worker: usize,
    pub target_min: usize,
    pub target_max: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct ParallelDecision {
    pub parallel: bool,
    pub max_workers: usize,
    pub active_workers: usize,
    pub threshold_units: usize,
    /// True when this call was forced into the opposite mode for data collection.
    pub exploring: bool,
}

#[derive(Default)]
struct PerfStats {
    sampled_calls: u64,
    seq_samples: u64,
    par_samples: u64,
    seq_ns_per_unit: Option<f64>,
    par_ns_per_unit: Option<f64>,
}

#[derive(Clone, Copy)]
struct PerfSnapshot {
    sampled_calls: u64,
    seq_samples: u64,
    par_samples: u64,
    seq_ns_per_unit: Option<f64>,
    par_ns_per_unit: Option<f64>,
}

/// Runtime tuner used by Q4/Q8/BF16 kernels.
pub struct RuntimeAutoTuner {
    name: &'static str,
    threshold_units: AtomicUsize,
    target_units_per_worker: AtomicUsize,
    threshold_min: usize,
    threshold_max: usize,
    target_min: usize,
    target_max: usize,
    autotune_enabled: bool,
    sample_rate: u64,
    log_enabled: bool,
    log_every: u64,
    call_counter: AtomicU64,
    /// Separate counter for rate-limiting exploration probes.
    explore_counter: AtomicU64,
    /// How often (in eligible decide() calls) to force an exploration probe.
    explore_rate: u64,
    /// Lock-free counters mirroring PerfStats for use in decide() without locking.
    seq_sample_count: AtomicU64,
    par_sample_count: AtomicU64,
    stats: Mutex<PerfStats>,
}

impl RuntimeAutoTuner {
    pub fn from_env(
        name: &'static str,
        env_prefix: &'static str,
        defaults: RuntimeAutoTuneDefaults,
    ) -> Self {
        let threshold_units = env_usize_prefixed(
            env_prefix,
            "PAR_THRESHOLD",
            GLOBAL_PAR_THRESHOLD_KEY,
            defaults.threshold_units,
        )
        .clamp(
            defaults.threshold_min,
            defaults.threshold_max.max(defaults.threshold_min),
        );

        let target_units_per_worker = env_usize_prefixed(
            env_prefix,
            "TARGET_WORK_PER_WORKER",
            GLOBAL_TARGET_PER_WORKER_KEY,
            defaults.target_units_per_worker,
        )
        .clamp(
            defaults.target_min,
            defaults.target_max.max(defaults.target_min),
        );

        let autotune_enabled = env_bool_prefixed(env_prefix, "AUTOTUNE", GLOBAL_AUTOTUNE_KEY, true);
        let sample_rate = env_usize_prefixed(
            env_prefix,
            "AUTOTUNE_SAMPLE_RATE",
            GLOBAL_SAMPLE_RATE_KEY,
            8,
        )
        .max(1) as u64;
        let log_enabled = env_bool_prefixed(env_prefix, "AUTOTUNE_LOG", GLOBAL_LOG_KEY, false);
        let log_every =
            env_usize_prefixed(env_prefix, "AUTOTUNE_LOG_EVERY", GLOBAL_LOG_EVERY_KEY, 128).max(1)
                as u64;

        let explore_rate = env_usize_prefixed(
            env_prefix,
            "AUTOTUNE_EXPLORE",
            GLOBAL_EXPLORE_KEY,
            64,
        )
        .max(1) as u64;

        Self {
            name,
            threshold_units: AtomicUsize::new(threshold_units),
            target_units_per_worker: AtomicUsize::new(target_units_per_worker),
            threshold_min: defaults.threshold_min,
            threshold_max: defaults.threshold_max.max(defaults.threshold_min),
            target_min: defaults.target_min,
            target_max: defaults.target_max.max(defaults.target_min),
            autotune_enabled,
            sample_rate,
            log_enabled,
            log_every,
            call_counter: AtomicU64::new(0),
            explore_counter: AtomicU64::new(0),
            explore_rate,
            seq_sample_count: AtomicU64::new(0),
            par_sample_count: AtomicU64::new(0),
            stats: Mutex::new(PerfStats::default()),
        }
    }

    #[inline]
    pub fn decide(
        &self,
        work_units: usize,
        work_items: usize,
        pool_workers: usize,
    ) -> ParallelDecision {
        let threshold_units = self
            .threshold_units
            .load(Ordering::Relaxed)
            .clamp(self.threshold_min, self.threshold_max);

        if pool_workers <= 1 || work_items <= 1 || work_units < threshold_units {
            return ParallelDecision {
                parallel: false,
                max_workers: 1,
                active_workers: 1,
                threshold_units,
                exploring: false,
            };
        }

        let target_units_per_worker = self
            .target_units_per_worker
            .load(Ordering::Relaxed)
            .clamp(self.target_min, self.target_max)
            .max(1);
        let desired_workers = work_units.div_ceil(target_units_per_worker).max(2);
        let max_workers = desired_workers.min(pool_workers).min(work_items).max(1);
        let active_workers = max_workers.min(pool_workers).min(work_items).max(1);

        if active_workers <= 1 {
            return ParallelDecision {
                parallel: false,
                max_workers: 1,
                active_workers: 1,
                threshold_units,
                exploring: false,
            };
        }

        // --- Exploration: force sequential to collect baseline ---
        // Only when autotune is on, we lack sequential data, and the workload
        // is small enough that a single sequential probe won't be catastrophic.
        // Cap at 256× the threshold so we never force-seq the LM head.
        if self.autotune_enabled {
            let seq = self.seq_sample_count.load(Ordering::Relaxed);
            let par = self.par_sample_count.load(Ordering::Relaxed);
            let explore_limit = threshold_units.saturating_mul(256);
            if seq < EXPLORE_MIN_SAMPLES && par >= EXPLORE_MIN_SAMPLES && work_units <= explore_limit {
                let probe = self.explore_counter.fetch_add(1, Ordering::Relaxed);
                if probe.is_multiple_of(self.explore_rate) {
                    return ParallelDecision {
                        parallel: false,
                        max_workers: 1,
                        active_workers: 1,
                        threshold_units,
                        exploring: true,
                    };
                }
            }
        }

        ParallelDecision {
            parallel: true,
            max_workers,
            active_workers,
            threshold_units,
            exploring: false,
        }
    }

    /// Start timing a kernel call.  Pass the decision so exploration probes
    /// are always sampled (otherwise the forced-sequential run would be wasted).
    #[inline]
    pub fn begin_sample(&self, decision: &ParallelDecision) -> Option<Instant> {
        // Exploration probes are always measured.
        if decision.exploring {
            // Still bump the counter so the regular cadence isn't disrupted.
            self.call_counter.fetch_add(1, Ordering::Relaxed);
            return Some(Instant::now());
        }

        if !self.autotune_enabled && !self.log_enabled {
            return None;
        }

        let call = self.call_counter.fetch_add(1, Ordering::Relaxed) + 1;
        if call <= 4 || call.is_multiple_of(self.sample_rate) {
            Some(Instant::now())
        } else {
            None
        }
    }

    #[inline]
    pub fn observe(&self, started: Option<Instant>, decision: ParallelDecision, work_units: usize) {
        let Some(started) = started else {
            return;
        };
        if work_units == 0 {
            return;
        }

        let elapsed_ns = started.elapsed().as_nanos() as f64;
        if elapsed_ns <= 0.0 {
            return;
        }
        let ns_per_unit = elapsed_ns / work_units as f64;
        let snapshot = self.record_sample(decision.parallel, ns_per_unit);

        if self.autotune_enabled {
            self.retune(decision, work_units, snapshot);
        }
        if self.log_enabled {
            self.maybe_log(decision, work_units, elapsed_ns, snapshot);
        }
    }

    fn record_sample(&self, parallel: bool, ns_per_unit: f64) -> PerfSnapshot {
        let mut stats = self.stats.lock().unwrap_or_else(|e| e.into_inner());
        stats.sampled_calls += 1;
        if parallel {
            stats.par_samples += 1;
            self.par_sample_count.fetch_add(1, Ordering::Relaxed);
            ewma_update(&mut stats.par_ns_per_unit, ns_per_unit);
        } else {
            stats.seq_samples += 1;
            self.seq_sample_count.fetch_add(1, Ordering::Relaxed);
            ewma_update(&mut stats.seq_ns_per_unit, ns_per_unit);
        }
        PerfSnapshot {
            sampled_calls: stats.sampled_calls,
            seq_samples: stats.seq_samples,
            par_samples: stats.par_samples,
            seq_ns_per_unit: stats.seq_ns_per_unit,
            par_ns_per_unit: stats.par_ns_per_unit,
        }
    }

    fn retune(&self, decision: ParallelDecision, work_units: usize, snapshot: PerfSnapshot) {
        if snapshot.seq_samples < 2 || snapshot.par_samples < 2 {
            return;
        }
        let (Some(seq_ns_per_unit), Some(par_ns_per_unit)) =
            (snapshot.seq_ns_per_unit, snapshot.par_ns_per_unit)
        else {
            return;
        };
        if seq_ns_per_unit <= 0.0 || par_ns_per_unit <= 0.0 {
            return;
        }

        let speedup = seq_ns_per_unit / par_ns_per_unit;
        let threshold = self
            .threshold_units
            .load(Ordering::Relaxed)
            .clamp(self.threshold_min, self.threshold_max);

        // Parallel slower than sequential estimate: push threshold up.
        if decision.parallel && speedup < 0.95 {
            let target_threshold = work_units.saturating_mul(11).div_ceil(10);
            let raised = blend_usize(threshold, target_threshold, 1, 4)
                .clamp(self.threshold_min, self.threshold_max);
            self.threshold_units.store(raised, Ordering::Relaxed);

            let target = self
                .target_units_per_worker
                .load(Ordering::Relaxed)
                .clamp(self.target_min, self.target_max);
            let raised_target = target.saturating_mul(11).div_ceil(10);
            self.target_units_per_worker.store(
                raised_target.clamp(self.target_min, self.target_max),
                Ordering::Relaxed,
            );
        }

        // Sequential slower than parallel estimate: pull threshold down.
        if !decision.parallel && speedup > 1.05 {
            let target_threshold = work_units.saturating_mul(9) / 10;
            let lowered = blend_usize(threshold, target_threshold, 1, 4)
                .clamp(self.threshold_min, self.threshold_max);
            self.threshold_units.store(lowered, Ordering::Relaxed);

            let target = self
                .target_units_per_worker
                .load(Ordering::Relaxed)
                .clamp(self.target_min, self.target_max);
            let lowered_target = target.saturating_mul(9) / 10;
            self.target_units_per_worker.store(
                lowered_target.clamp(self.target_min, self.target_max),
                Ordering::Relaxed,
            );
        }
    }

    fn maybe_log(
        &self,
        decision: ParallelDecision,
        work_units: usize,
        elapsed_ns: f64,
        snapshot: PerfSnapshot,
    ) {
        if snapshot.sampled_calls != 1 && !snapshot.sampled_calls.is_multiple_of(self.log_every) {
            return;
        }

        let threshold_units = self
            .threshold_units
            .load(Ordering::Relaxed)
            .clamp(self.threshold_min, self.threshold_max);
        let target_units_per_worker = self
            .target_units_per_worker
            .load(Ordering::Relaxed)
            .clamp(self.target_min, self.target_max);
        let seq_ns = snapshot.seq_ns_per_unit.unwrap_or(0.0);
        let par_ns = snapshot.par_ns_per_unit.unwrap_or(0.0);

        tracing::debug!(
            name = self.name,
            mode = if decision.parallel { "par" } else { "seq" },
            exploring = decision.exploring,
            units = work_units,
            workers = decision.active_workers,
            elapsed_ms = elapsed_ns / 1_000_000.0,
            threshold_units,
            target_units_per_worker,
            seq_ns_per_unit = seq_ns,
            par_ns_per_unit = par_ns,
            seq_samples = snapshot.seq_samples,
            par_samples = snapshot.par_samples,
            "autotune decision"
        );
    }
}

#[inline]
fn ewma_update(slot: &mut Option<f64>, sample: f64) {
    *slot = match slot {
        Some(prev) => Some(*prev * 0.8 + sample * 0.2),
        None => Some(sample),
    };
}

#[inline]
fn blend_usize(current: usize, target: usize, num: u128, den: u128) -> usize {
    if den == 0 {
        return current;
    }
    let keep = den.saturating_sub(num);
    let blended = (current as u128)
        .saturating_mul(keep)
        .saturating_add((target as u128).saturating_mul(num))
        / den;
    blended.min(usize::MAX as u128) as usize
}

fn env_usize_prefixed(prefix: &str, suffix: &str, global_key: &str, default: usize) -> usize {
    let key = format!("{}_{}", prefix, suffix);
    env_usize(&key)
        .or_else(|| env_usize(global_key))
        .unwrap_or(default)
}

fn env_bool_prefixed(prefix: &str, suffix: &str, global_key: &str, default: bool) -> bool {
    let key = format!("{}_{}", prefix, suffix);
    env_bool(&key)
        .or_else(|| env_bool(global_key))
        .unwrap_or(default)
}

fn env_usize(key: &str) -> Option<usize> {
    std::env::var(key).ok()?.trim().parse::<usize>().ok()
}

fn env_bool(key: &str) -> Option<bool> {
    let raw = std::env::var(key).ok()?;
    let normalized = raw.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}
