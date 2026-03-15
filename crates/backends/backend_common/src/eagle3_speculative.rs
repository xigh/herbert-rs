//! EAGLE-3 speculative decoding orchestrator.
//!
//! Uses a pre-trained EAGLE-3 head to draft tokens from the target model's
//! own hidden states, then verifies them with the target model in one batch.
//! Output distribution is mathematically identical to target-only generation.

use herbert_core::backend::{Backend, RunOpts};
use herbert_core::error::{HerbertError, Result};
use herbert_core::kv_cache::KvHandle;
use herbert_core::tensor::BF16;

use crate::eagle3::Eagle3Head;
use crate::speculative::{softmax, sample_from_probs, resample_from_residual};

/// Statistics from one EAGLE-3 speculative decoding step.
#[derive(Debug, Clone)]
pub struct Eagle3StepStats {
    pub accepted: usize,
    pub produced: usize,
    pub draft_tokens: Vec<u32>,
    pub final_tokens: Vec<u32>,
}

/// EAGLE-3 speculative decoding orchestrator.
pub struct Eagle3SpecDecoder {
    pub draft_k: usize,
    rng_state: u64,
    pub total_accepted: usize,
    pub total_steps: usize,
    pub total_produced: usize,
}

impl Eagle3SpecDecoder {
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

    fn next_f32(&mut self) -> f32 {
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;
        (x >> 40) as f32 / (1u64 << 24) as f32
    }

    pub fn acceptance_rate(&self) -> f64 {
        if self.total_steps == 0 {
            0.0
        } else {
            self.total_accepted as f64 / (self.total_steps as f64 * self.draft_k as f64)
        }
    }

    pub fn tokens_per_step(&self) -> f64 {
        if self.total_steps == 0 {
            0.0
        } else {
            self.total_produced as f64 / self.total_steps as f64
        }
    }

    /// Run one EAGLE-3 speculative decoding step:
    ///
    /// 1. Target model processes `current_token` (decode_next), capturing hidden states
    /// 2. EAGLE-3 head drafts K tokens using the captured hidden states
    /// 3. Target model verifies K draft tokens in one forward pass
    /// 4. Accept/reject with vocab-mapped rejection sampling
    /// 5. Rollback target + EAGLE KV caches
    ///
    /// Returns the accepted + resampled tokens.
    pub fn step(
        &mut self,
        eagle3: &mut Eagle3Head,
        target: &mut Box<dyn Backend>,
        target_kv: &mut KvHandle,
        current_token: u32,
        target_embed_tokens: &[BF16],
    ) -> Result<Eagle3StepStats> {
        let k = self.draft_k;
        let target_start_pos = target.kv_seq_len(target_kv)?;

        // Reset EAGLE KV for each round (no cross-round context, avoids position gaps)
        eagle3.reset_kv();

        // Phase 1: Target processes current_token with logits (hidden states captured)
        let target_opts = RunOpts {
            return_logits: true,
            ..Default::default()
        };
        let target_output = target.decode_next(target_kv, current_token, target_opts)?;
        let current_logits = target_output.logits.ok_or_else(|| {
            HerbertError::Backend("target must return logits for EAGLE-3".into())
        })?;

        // Phase 2: Read captured hidden states from InferenceContext
        let hidden_states = target.take_eagle3_hidden_states(target_kv)?;

        // Phase 3: EAGLE-3 drafts K tokens
        let mut draft_tokens = Vec::with_capacity(k);
        let mut draft_logits = Vec::with_capacity(k);
        let mut prev_output: Option<Vec<f32>> = None;
        let mut tok = current_token;

        // Position for EAGLE RoPE: aligned with target's absolute positions
        let eagle_base_pos = target_start_pos + 1; // after current_token

        for step_i in 0..k {
            let (draft_target_tok, logits, decoder_out) = if step_i == 0 {
                eagle3.draft_step(
                    tok,
                    Some(&hidden_states),
                    None,
                    target_embed_tokens,
                    eagle_base_pos + step_i,
                )?
            } else {
                eagle3.draft_step(
                    tok,
                    None,
                    prev_output.as_deref(),
                    target_embed_tokens,
                    eagle_base_pos + step_i,
                )?
            };

            tok = draft_target_tok;
            draft_tokens.push(draft_target_tok);
            draft_logits.push(logits);
            prev_output = Some(decoder_out);
        }

        // Phase 4: Target verifies K draft tokens in one forward pass
        // verify_draft adds these K tokens to KV and returns K logit vectors.
        // Combined with current_logits from decode_next, we have K+1 logit positions:
        //   current_logits  = p(x | ctx + current_token)     → verify draft_tokens[0]
        //   verify_logits[0] = p(x | ctx + cur + d[0])       → verify draft_tokens[1]
        //   verify_logits[i] = p(x | ctx + cur + d[0..i])    → verify draft_tokens[i+1]
        //   verify_logits[K-1] = p(x | ctx + cur + d[0..K-1]) → bonus token
        let verify_output = target.verify_draft(target_kv, &draft_tokens)?;
        let verify_logits = &verify_output.logits_per_position;

        if verify_logits.len() < k {
            return Err(HerbertError::Backend(format!(
                "eagle3: verify_draft returned {} positions, expected {}",
                verify_logits.len(), k
            )));
        }

        // Build combined target logits: [current_logits, verify_logits[0..K-1]]
        // target_all_logits[i] is used to verify draft_tokens[i] (i = 0..K)
        // target_all_logits[K] (= verify_logits[K-1]) is the bonus position

        // Phase 5: Accept/reject with vocab mapping
        let debug = std::env::var("EAGLE3_DEBUG").is_ok();
        let mut accepted_count = 0;
        let mut final_tokens = Vec::with_capacity(k + 1);

        for i in 0..k {
            // target_all_logits[i] verifies draft_tokens[i]
            let target_logits_i = if i == 0 { &current_logits } else { &verify_logits[i - 1] };
            let target_p = softmax(target_logits_i);
            let draft_q_raw = softmax(&draft_logits[i]);

            let draft_token = draft_tokens[i];
            let draft_token_draft_idx = eagle3.target_to_draft(draft_token);

            // Target argmax for comparison
            let target_argmax = target_p.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx as u32)
                .unwrap_or(0);

            // p(x) from target at the draft token
            let p_x = if (draft_token as usize) < target_p.len() {
                target_p[draft_token as usize]
            } else {
                0.0
            };

            // q(x) from draft at the draft token
            let q_x = if let Some(d_idx) = draft_token_draft_idx {
                if (d_idx as usize) < draft_q_raw.len() {
                    draft_q_raw[d_idx as usize]
                } else {
                    1e-10
                }
            } else {
                1e-10
            };

            let accept_prob = (p_x / q_x).min(1.0);
            let r = self.next_f32();

            if debug {
                let t2d_argmax = eagle3.target_to_draft(target_argmax);
                eprintln!("[eagle3-dbg] step={} i={}: draft_tok={} target_argmax={} t2d[argmax]={:?} p_x={:.6} q_x={:.6} accept_prob={:.4} r={:.4}",
                    self.total_steps, i, draft_token, target_argmax, t2d_argmax, p_x, q_x, accept_prob, r);
            }

            if r < accept_prob {
                accepted_count += 1;
                final_tokens.push(draft_token);
            } else {
                // Reject: resample from norm(max(0, p - q_mapped))
                let mut q_target = vec![0.0f32; target_p.len()];
                for (d_idx, &q_val) in draft_q_raw.iter().enumerate() {
                    let t_idx = eagle3.draft_to_target(d_idx as u32) as usize;
                    if t_idx < q_target.len() {
                        q_target[t_idx] = q_val;
                    }
                }
                let resampled = resample_from_residual(&target_p, &q_target, &mut self.rng_state);
                final_tokens.push(resampled);
                break;
            }
        }

        // If all K accepted, sample bonus from verify_logits[K-1]
        if accepted_count == k {
            let bonus_probs = softmax(&verify_logits[k - 1]);
            let bonus = sample_from_probs(&bonus_probs, &mut self.rng_state);
            final_tokens.push(bonus);
        }

        let produced = final_tokens.len();

        // Phase 6: Rollback KV caches
        // Target KV: decode_next added 1 (current_token), verify_draft added K = total K+1
        // Keep: current_token + accepted draft tokens
        let target_keep = target_start_pos + 1 + accepted_count;
        target.truncate_kv(target_kv, target_keep)?;

        // EAGLE KV: has K entries (one per draft step)
        // Keep: accepted_count entries
        eagle3.truncate_kv(accepted_count);

        // Update stats
        self.total_accepted += accepted_count;
        self.total_steps += 1;
        self.total_produced += produced;

        Ok(Eagle3StepStats {
            accepted: accepted_count,
            produced,
            draft_tokens,
            final_tokens,
        })
    }
}
