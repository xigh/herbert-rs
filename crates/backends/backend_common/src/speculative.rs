//! Speculative decoding orchestrator.
//!
//! Uses a small draft model to generate K candidate tokens, then verifies
//! them all at once with the larger target model. Rejected tokens are
//! resampled; accepted ones are free. The output distribution is
//! mathematically identical to target-only generation (lossless).

use herbert_core::backend::{Backend, RunOpts};
use herbert_core::error::{HerbertError, Result};
use herbert_core::kv_cache::KvHandle;

/// Statistics from one speculative decoding step.
#[derive(Debug, Clone)]
pub struct SpecStepStats {
    /// Number of draft tokens accepted (0..=K).
    pub accepted: usize,
    /// Total tokens produced this step (accepted + 1 resampled or bonus).
    pub produced: usize,
    /// Draft token IDs that were proposed.
    pub draft_tokens: Vec<u32>,
    /// Final token IDs that were accepted/produced.
    pub final_tokens: Vec<u32>,
}

/// Speculative decoding orchestrator.
///
/// Holds references to draft and target backends, and manages the
/// accept/reject loop with KV cache rollback.
pub struct SpeculativeDecoder {
    /// Number of draft tokens to propose per step (K).
    pub draft_k: usize,
    /// xorshift64 PRNG state for rejection sampling.
    rng_state: u64,
    // Running statistics
    /// Total tokens accepted across all steps.
    pub total_accepted: usize,
    /// Total steps (draft+verify cycles).
    pub total_steps: usize,
    /// Total tokens produced.
    pub total_produced: usize,
}

impl SpeculativeDecoder {
    pub fn new(draft_k: usize) -> Self {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        let rng_state = if seed == 0 { 0xdeadbeef } else { seed };
        Self {
            draft_k,
            rng_state,
            total_accepted: 0,
            total_steps: 0,
            total_produced: 0,
        }
    }

    /// xorshift64 PRNG — returns a uniform f32 in [0, 1).
    fn next_f32(&mut self) -> f32 {
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;
        (x >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Average acceptance rate so far.
    pub fn acceptance_rate(&self) -> f64 {
        if self.total_steps == 0 {
            0.0
        } else {
            self.total_accepted as f64 / (self.total_steps as f64 * self.draft_k as f64)
        }
    }

    /// Average tokens produced per step.
    pub fn tokens_per_step(&self) -> f64 {
        if self.total_steps == 0 {
            0.0
        } else {
            self.total_produced as f64 / self.total_steps as f64
        }
    }

    /// Run one speculative decoding step:
    /// 1. Draft K tokens with the draft model (auto-regressive)
    /// 2. Verify all K+1 positions with the target model in one forward pass
    /// 3. Accept/reject using modified rejection sampling
    /// 4. Rollback KV caches to keep only accepted tokens
    ///
    /// Returns the accepted+resampled tokens and step statistics.
    ///
    /// `current_token` is the last accepted token that neither model has
    /// processed yet. Both KV caches must be at the same sequence position
    /// (neither has current_token in its KV).
    pub fn step(
        &mut self,
        draft_backend: &mut Box<dyn Backend>,
        target_backend: &mut Box<dyn Backend>,
        draft_kv: &mut KvHandle,
        target_kv: &mut KvHandle,
        current_token: u32,
    ) -> Result<SpecStepStats> {
        let k = self.draft_k;
        let draft_start_pos = draft_backend.kv_seq_len(draft_kv)?;
        let target_start_pos = target_backend.kv_seq_len(target_kv)?;

        // Phase 1: Draft K tokens with the draft model (auto-regressive)
        // decode_next(tok) processes tok, adds to KV, returns next token + logits.
        //
        // Call 0: feed current_token → KV gets current_token → draft_tokens[0], draft_logits[0]
        //   draft_logits[0] = q(x | context + current_token)
        // Call i: feed draft_tokens[i-1] → draft_tokens[i], draft_logits[i]
        //   draft_logits[i] = q(x | context + current_token + draft_tokens[0..i-1])
        //
        // After K calls, draft KV has K new entries.
        let mut draft_tokens = Vec::with_capacity(k);
        let mut draft_logits = Vec::with_capacity(k);
        let mut tok = current_token;

        let draft_opts = RunOpts {
            return_logits: true,
            ..Default::default()
        };

        for _ in 0..k {
            let output = draft_backend.decode_next(draft_kv, tok, draft_opts.clone())?;
            let logits = output.logits.ok_or_else(|| {
                HerbertError::Backend("draft model must return logits".to_string())
            })?;
            tok = output.token;
            draft_tokens.push(tok);
            draft_logits.push(logits);
        }

        // Phase 2: Verify with the target model in one forward pass.
        // Feed [current_token, draft_tokens[0], ..., draft_tokens[K-1]] (K+1 tokens).
        // This produces K+1 logit vectors:
        //   target_logits[0] = p(x | context + current_token) → verify draft_tokens[0]
        //   target_logits[i] = p(x | context + current_token + draft_tokens[0..i-1]) → verify draft_tokens[i]
        //   target_logits[K] = p(x | context + current_token + all draft tokens) → bonus
        let mut verify_tokens = Vec::with_capacity(k + 1);
        verify_tokens.push(current_token);
        verify_tokens.extend_from_slice(&draft_tokens);

        let verify_output = target_backend.verify_draft(target_kv, &verify_tokens)?;
        let target_logits = &verify_output.logits_per_position;

        if target_logits.len() < k + 1 {
            return Err(HerbertError::Backend(format!(
                "verify_draft returned {} positions, expected at least {}",
                target_logits.len(), k + 1
            )));
        }

        // Phase 3: Accept/reject using modified rejection sampling
        // For each draft token i (0..K):
        //   p(x) = softmax(target_logits[i])  ← logits at position matching draft_logits[i]
        //   q(x) = softmax(draft_logits[i])
        //   Accept draft_tokens[i] if rand() < min(1, p(draft_tokens[i]) / q(draft_tokens[i]))
        //   If rejected: resample from norm(max(0, p - q)), stop accepting further tokens
        let mut accepted_count = 0;
        let mut final_tokens = Vec::with_capacity(k + 1);

        for i in 0..k {
            let p = softmax(&target_logits[i]);
            let q = softmax(&draft_logits[i]);
            let x_i = draft_tokens[i] as usize;

            let p_xi = if x_i < p.len() { p[x_i] } else { 0.0 };
            let q_xi = if x_i < q.len() { q[x_i] } else { 1e-10 };

            let accept_prob = (p_xi / q_xi).min(1.0);
            let r = self.next_f32();

            if r < accept_prob {
                // Accept draft token
                accepted_count += 1;
                final_tokens.push(draft_tokens[i]);
            } else {
                // Reject: resample from norm(max(0, p - q))
                let resampled = self.resample_from_residual_inner(&p, &q);
                final_tokens.push(resampled);
                break;
            }
        }

        // If all K were accepted, sample a bonus token from target_logits[K]
        if accepted_count == k {
            let bonus_probs = softmax(&target_logits[k]);
            let bonus_token = self.sample_from_probs_inner(&bonus_probs);
            final_tokens.push(bonus_token);
        }

        let produced = final_tokens.len();

        // Phase 4: Rollback KV caches to aligned state.
        //
        // Draft KV: added K entries (current_token + K-1 draft tokens).
        // We want to keep: current_token + accepted_count draft tokens.
        // = draft_start_pos + 1 + accepted_count entries total.
        // But we only keep up to the last accepted token, not the last processed token.
        let draft_keep = draft_start_pos + 1 + accepted_count;
        draft_backend.truncate_kv(draft_kv, draft_keep)?;

        // Target KV: added K+1 entries (current_token + K draft tokens).
        // We want to keep: current_token + accepted_count draft tokens.
        let target_keep = target_start_pos + 1 + accepted_count;
        target_backend.truncate_kv(target_kv, target_keep)?;

        // Update stats
        self.total_accepted += accepted_count;
        self.total_steps += 1;
        self.total_produced += produced;

        Ok(SpecStepStats {
            accepted: accepted_count,
            produced,
            draft_tokens,
            final_tokens,
        })
    }

    /// Sample from the residual distribution: norm(max(0, p - q)).
    fn resample_from_residual_inner(&mut self, p: &[f32], q: &[f32]) -> u32 {
        resample_from_residual(p, q, &mut self.rng_state)
    }

    /// Sample a token from a probability distribution.
    fn sample_from_probs_inner(&mut self, probs: &[f32]) -> u32 {
        sample_from_probs(probs, &mut self.rng_state)
    }
}

/// xorshift64 PRNG step — returns a uniform f32 in [0, 1).
fn xorshift64_next_f32(state: &mut u64) -> f32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    (x >> 40) as f32 / (1u64 << 24) as f32
}

/// Sample from the residual distribution: norm(max(0, p - q)).
pub fn resample_from_residual(p: &[f32], q: &[f32], rng_state: &mut u64) -> u32 {
    let n = p.len().min(q.len());
    let mut residual = vec![0.0f32; n];
    let mut sum = 0.0f32;
    for i in 0..n {
        let r = (p[i] - q[i]).max(0.0);
        residual[i] = r;
        sum += r;
    }
    if sum <= 0.0 {
        return sample_from_probs(p, rng_state);
    }
    let inv_sum = 1.0 / sum;
    let r = xorshift64_next_f32(rng_state);
    let mut cumsum = 0.0f32;
    for (i, &v) in residual.iter().enumerate() {
        cumsum += v * inv_sum;
        if r < cumsum {
            return i as u32;
        }
    }
    (n - 1) as u32
}

/// Sample a token from a probability distribution.
pub fn sample_from_probs(probs: &[f32], rng_state: &mut u64) -> u32 {
    let r = xorshift64_next_f32(rng_state);
    let mut cumsum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            return i as u32;
        }
    }
    (probs.len() - 1) as u32
}

/// Compute softmax of a logit vector, returning probabilities.
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits.iter().map(|&v| (v - max_val).exp()).collect();
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 {
        let inv_sum = 1.0 / sum;
        for p in probs.iter_mut() {
            *p *= inv_sum;
        }
    }
    probs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_basic() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax(&logits);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_softmax_uniform() {
        let logits = vec![0.0; 4];
        let probs = softmax(&logits);
        for &p in &probs {
            assert!((p - 0.25).abs() < 1e-5);
        }
    }

    #[test]
    fn test_speculative_decoder_new() {
        let dec = SpeculativeDecoder::new(5);
        assert_eq!(dec.draft_k, 5);
        assert_eq!(dec.total_steps, 0);
        assert_eq!(dec.total_accepted, 0);
    }

    #[test]
    fn test_resample_from_residual() {
        let mut dec = SpeculativeDecoder::new(5);
        let p = vec![0.5, 0.3, 0.2];
        let q = vec![0.1, 0.8, 0.1];
        // residual = [0.4, 0.0, 0.1], normalized = [0.8, 0.0, 0.2]
        let token = dec.resample_from_residual_inner(&p, &q);
        assert!(token < 3);
    }
}
