//! Global MoE expert utilization stats collector.
//!
//! Enable with `moe_stats::global().enable()` before prefill.
//! Each MoE layer records its expert assignments via `record()`.
//! Call `print_summary()` after inference to display the table.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

/// One record per MoE layer prefill call.
pub struct MoeCallRecord {
    pub num_experts_total: usize,
    pub num_active: usize,
    /// (expert_id, token_count) for each active expert.
    pub tokens_per_expert: Vec<(usize, usize)>,
}

pub struct MoeStatsCollector {
    enabled: AtomicBool,
    records: Mutex<Vec<MoeCallRecord>>,
}

static INSTANCE: MoeStatsCollector = MoeStatsCollector {
    enabled: AtomicBool::new(false),
    records: Mutex::new(Vec::new()),
};

/// Access the global MoE stats collector.
pub fn global() -> &'static MoeStatsCollector {
    &INSTANCE
}

impl MoeStatsCollector {
    pub fn enable(&self) {
        self.enabled.store(true, Ordering::Release);
    }

    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Acquire)
    }

    pub fn record(&self, rec: MoeCallRecord) {
        if !self.is_enabled() {
            return;
        }
        self.records.lock().unwrap_or_else(|e| e.into_inner()).push(rec);
    }

    /// Print the summary table to stdout. Returns true if anything was printed.
    pub fn print_summary(&self) -> bool {
        let records = self.records.lock().unwrap_or_else(|e| e.into_inner());
        if records.is_empty() {
            return false;
        }

        let num_calls = records.len();
        let ne = records[0].num_experts_total;

        // --- Active experts per layer ---
        let actives: Vec<usize> = records.iter().map(|r| r.num_active).collect();
        let active_min = *actives.iter().min().expect("records non-empty");
        let active_max = *actives.iter().max().expect("records non-empty");
        let active_avg = actives.iter().sum::<usize>() as f64 / num_calls as f64;

        // --- Tokens per active expert (across all layer calls) ---
        let all_counts: Vec<usize> = records
            .iter()
            .flat_map(|r| r.tokens_per_expert.iter().map(|&(_, c)| c))
            .collect();
        let tok_min = *all_counts.iter().min().unwrap_or(&0);
        let tok_max = *all_counts.iter().max().unwrap_or(&0);
        let tok_avg = if all_counts.is_empty() {
            0.0
        } else {
            all_counts.iter().sum::<usize>() as f64 / all_counts.len() as f64
        };

        // --- Load imbalance: average of (max/avg) per layer ---
        let mut imbalance_sum = 0.0f64;
        for r in records.iter() {
            if r.tokens_per_expert.is_empty() {
                continue;
            }
            let counts: Vec<usize> = r.tokens_per_expert.iter().map(|&(_, c)| c).collect();
            let max_c = *counts.iter().max().expect("counts non-empty") as f64;
            let avg_c = counts.iter().sum::<usize>() as f64 / counts.len() as f64;
            if avg_c > 0.0 {
                imbalance_sum += max_c / avg_c;
            }
        }
        let imbalance_avg = imbalance_sum / num_calls as f64;

        // --- Expert frequency: expert_id -> (appearances, total_tokens) ---
        let mut freq: Vec<(usize, usize)> = vec![(0, 0); ne];
        for r in records.iter() {
            for &(eid, count) in &r.tokens_per_expert {
                if eid < ne {
                    freq[eid].0 += 1;
                    freq[eid].1 += count;
                }
            }
        }

        // Sort by frequency descending, then by total tokens
        let mut freq_sorted: Vec<(usize, usize, usize)> = freq
            .iter()
            .enumerate()
            .map(|(eid, &(app, tok))| (eid, app, tok))
            .collect();
        freq_sorted.sort_by(|a, b| b.1.cmp(&a.1).then(b.2.cmp(&a.2)));

        // Count never-used experts
        let never_used = freq_sorted.iter().filter(|e| e.1 == 0).count();

        // --- Print ---
        tracing::info!(
            num_calls,
            num_experts = ne,
            active_min,
            active_max,
            active_avg = format_args!("{:.1}", active_avg),
            active_pct = format_args!("{:.1}", active_avg / ne as f64 * 100.0),
            tok_min,
            tok_max,
            tok_avg = format_args!("{:.2}", tok_avg),
            imbalance = format_args!("{:.2}x", imbalance_avg),
            never_used,
            "MoE expert stats"
        );

        let top_n = 5.min(ne);

        for &(eid, app, tok) in freq_sorted.iter().take(top_n) {
            let avg_tok = if app > 0 { tok as f64 / app as f64 } else { 0.0 };
            tracing::info!(
                expert = eid,
                appearances = app,
                total_calls = num_calls,
                pct = format_args!("{:.1}", app as f64 / num_calls as f64 * 100.0),
                avg_tok = format_args!("{:.1}", avg_tok),
                "MoE top expert"
            );
        }

        if ne > top_n * 2 {
            let start = ne.saturating_sub(top_n);
            for &(eid, app, tok) in &freq_sorted[start..] {
                let avg_tok = if app > 0 { tok as f64 / app as f64 } else { 0.0 };
                tracing::info!(
                    expert = eid,
                    appearances = app,
                    total_calls = num_calls,
                    pct = format_args!("{:.1}", app as f64 / num_calls as f64 * 100.0),
                    avg_tok = format_args!("{:.1}", avg_tok),
                    "MoE bottom expert"
                );
            }
        }

        true
    }
}
