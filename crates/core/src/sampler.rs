//! Token sampler with temperature, top-k, and top-p (nucleus) sampling.

/// Configuration for the token sampler.
pub struct SamplerConfig {
    /// Temperature for logit scaling. 0.0 = greedy (argmax).
    pub temperature: f32,
    /// Top-k filtering. 0 = disabled.
    pub top_k: usize,
    /// Top-p (nucleus) sampling threshold. 1.0 = disabled.
    pub top_p: f32,
}

/// Token sampler with built-in xorshift64 PRNG (no external dependencies).
pub struct Sampler {
    config: SamplerConfig,
    rng_state: u64,
}

impl Sampler {
    /// Create a new sampler from config, seeded by current timestamp.
    pub fn new(config: SamplerConfig) -> Self {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        Self::new_with_seed(config, seed)
    }

    /// Create a new sampler with an explicit seed for reproducible output.
    pub fn new_with_seed(config: SamplerConfig, seed: u64) -> Self {
        // Ensure non-zero seed for xorshift
        let rng_state = if seed == 0 { 0xdeadbeef } else { seed };
        Self { config, rng_state }
    }

    /// Returns true if this sampler will behave as greedy (argmax).
    pub fn is_greedy(&self) -> bool {
        self.config.temperature == 0.0
    }

    pub fn temperature(&self) -> f32 { self.config.temperature }
    pub fn top_k(&self) -> usize { self.config.top_k }
    pub fn top_p(&self) -> f32 { self.config.top_p }

    pub fn set_temperature(&mut self, v: f32) { self.config.temperature = v; }
    pub fn set_top_k(&mut self, v: usize) { self.config.top_k = v; }
    pub fn set_top_p(&mut self, v: f32) { self.config.top_p = v; }

    /// xorshift64 PRNG — returns a uniform u64.
    fn next_u64(&mut self) -> u64 {
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;
        x
    }

    /// Returns a uniform f32 in [0, 1).
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Sample a token ID from raw logits.
    ///
    /// Algorithm:
    /// 1. Divide logits by temperature
    /// 2. If top_k > 0: keep only the top_k highest, set rest to -inf
    /// 3. Softmax
    /// 4. If top_p < 1.0: sort by prob desc, find cumsum cutoff, zero the rest
    /// 5. Re-normalize and sample from the CDF
    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        let n = logits.len();
        if n == 0 {
            return 0;
        }

        // Greedy fallback
        if self.config.temperature == 0.0 {
            return argmax(logits);
        }

        // Work buffer: (index, scaled_logit)
        let mut work: Vec<(u32, f32)> = logits
            .iter()
            .enumerate()
            .map(|(i, &v)| (i as u32, v / self.config.temperature))
            .collect();

        // Top-k filtering: partial sort, keep top_k
        let top_k = self.config.top_k;
        if top_k > 0 && top_k < work.len() {
            work.select_nth_unstable_by(top_k, |a, b| b.1.partial_cmp(&a.1).unwrap());
            work.truncate(top_k);
        }

        // Softmax
        let max_val = work.iter().map(|x| x.1).fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for item in &mut work {
            item.1 = (item.1 - max_val).exp();
            sum += item.1;
        }
        let inv_sum = 1.0 / sum;
        for item in &mut work {
            item.1 *= inv_sum;
        }

        // Top-p (nucleus) filtering
        if self.config.top_p < 1.0 {
            // Sort by probability descending
            work.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            let mut cumsum = 0.0f32;
            let mut cutoff = work.len();
            for (i, item) in work.iter().enumerate() {
                cumsum += item.1;
                if cumsum >= self.config.top_p {
                    cutoff = i + 1;
                    break;
                }
            }
            work.truncate(cutoff);

            // Re-normalize
            let sum: f32 = work.iter().map(|x| x.1).sum();
            let inv_sum = 1.0 / sum;
            for item in &mut work {
                item.1 *= inv_sum;
            }
        }

        // Sample from CDF
        let r = self.next_f32();
        let mut cumsum = 0.0f32;
        for &(idx, prob) in &work {
            cumsum += prob;
            if r < cumsum {
                return idx;
            }
        }

        // Fallback: return last token (rounding edge case)
        work.last().map(|x| x.0).unwrap_or(0)
    }
}

/// Controls how many tokens the model can spend inside `<think>...</think>`.
///
/// Uses logits manipulation (à la HuggingFace LogitsProcessor):
/// - At budget-1: force a `\n` by zeroing all other logits
/// - At budget:   force `</think>` by zeroing all other logits
///   This gives the model a clean line break before the close tag.
pub struct ThinkBudget {
    /// Max tokens inside a think block. 0 = unlimited.
    max_tokens: usize,
    /// Token ID for `<think>`.
    think_open_id: u32,
    /// Token ID for `</think>`.
    think_close_id: u32,
    /// Token ID for `\n`.
    newline_id: u32,
    /// Are we currently inside a think block?
    in_think: bool,
    /// Tokens generated since the last `<think>`.
    think_count: usize,
}

impl ThinkBudget {
    pub fn new(max_tokens: usize, think_open_id: u32, think_close_id: u32, newline_id: u32) -> Self {
        Self {
            max_tokens,
            think_open_id,
            think_close_id,
            newline_id,
            in_think: false,
            think_count: 0,
        }
    }

    /// Mark as already inside a think block (for models like Qwen3.5 where
    /// `<think>\n` is part of the prompt, not generated by the model).
    pub fn set_in_think(&mut self) {
        self.in_think = true;
        self.think_count = 0;
    }

    pub fn max_tokens(&self) -> usize { self.max_tokens }
    pub fn set_max_tokens(&mut self, v: usize) { self.max_tokens = v; }

    /// Returns true when think budget is active (non-zero limit).
    pub fn is_active(&self) -> bool { self.max_tokens > 0 }

    /// Reset state for a new turn.
    pub fn reset(&mut self) {
        self.in_think = false;
        self.think_count = 0;
    }

    /// Track a generated token to update think state.
    /// Call this AFTER sampling, with the final token that will be emitted.
    pub fn track(&mut self, token: u32) {
        if token == self.think_open_id {
            self.in_think = true;
            self.think_count = 0;
        } else if token == self.think_close_id {
            self.in_think = false;
            self.think_count = 0;
        } else if self.in_think {
            self.think_count += 1;
        }
    }

    /// Modify logits in-place to enforce the think budget.
    /// Call this BEFORE sampling.
    ///
    /// - At `budget - 1` tokens: force `\n` (clean line break)
    /// - At `budget` tokens:     force `</think>`
    pub fn apply_to_logits(&self, logits: &mut [f32]) {
        if !self.in_think || self.max_tokens == 0 {
            return;
        }
        let budget = self.max_tokens;
        if self.think_count + 1 == budget {
            // One token before hard limit: force \n
            force_single_token(logits, self.newline_id as usize);
        } else if self.think_count + 1 > budget {
            // At or past limit: force </think>
            force_single_token(logits, self.think_close_id as usize);
        }
    }
}

/// Set all logits to -inf except the one at `token_id`.
fn force_single_token(logits: &mut [f32], token_id: usize) {
    for (i, v) in logits.iter_mut().enumerate() {
        if i != token_id {
            *v = f32::NEG_INFINITY;
        }
    }
}

/// Simple argmax over a slice.
pub fn argmax(logits: &[f32]) -> u32 {
    let mut best_idx = 0u32;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i as u32;
        }
    }
    best_idx
}
