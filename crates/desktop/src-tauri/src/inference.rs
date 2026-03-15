//! Inference utilities: sampler, prompt building, text delta, repetition detection.
//! Adapted from crates/cli/src/sampler.rs and crates/cli/src/chat.rs.

/// Configuration for the token sampler.
pub struct SamplerConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
}

/// Token sampler with built-in xorshift64 PRNG.
pub struct Sampler {
    config: SamplerConfig,
    rng_state: u64,
}

impl Sampler {
    pub fn new(config: SamplerConfig) -> Self {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        let rng_state = if seed == 0 { 0xdeadbeef } else { seed };
        Self { config, rng_state }
    }

    pub fn is_greedy(&self) -> bool {
        self.config.temperature == 0.0
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;
        x
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        let n = logits.len();
        if n == 0 {
            return 0;
        }
        if self.config.temperature == 0.0 {
            return argmax(logits);
        }

        let mut work: Vec<(u32, f32)> = logits
            .iter()
            .enumerate()
            .map(|(i, &v)| (i as u32, v / self.config.temperature))
            .collect();

        let top_k = self.config.top_k;
        if top_k > 0 && top_k < work.len() {
            work.select_nth_unstable_by(top_k, |a, b| b.1.partial_cmp(&a.1).unwrap());
            work.truncate(top_k);
        }

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

        if self.config.top_p < 1.0 {
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
            let sum: f32 = work.iter().map(|x| x.1).sum();
            let inv_sum = 1.0 / sum;
            for item in &mut work {
                item.1 *= inv_sum;
            }
        }

        let r = self.next_f32();
        let mut cumsum = 0.0f32;
        for &(idx, prob) in &work {
            cumsum += prob;
            if r < cumsum {
                return idx;
            }
        }
        work.last().map(|x| x.0).unwrap_or(0)
    }
}

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

/// Byte length of `s` excluding trailing U+FFFD replacement characters.
pub fn safe_len(s: &str) -> usize {
    s.trim_end_matches('\u{FFFD}').len()
}

/// Check if the last tokens form a repeating loop (same pattern × 3).
pub fn detect_repetition_loop(tokens: &[u32]) -> Option<usize> {
    const MIN_PAT: usize = 6;
    const MAX_PAT: usize = 100;
    const REPS: usize = 3;
    let len = tokens.len();
    for pat_len in MIN_PAT..=MAX_PAT {
        let need = pat_len * REPS;
        if len < need {
            continue;
        }
        let base = len - need;
        let pattern = &tokens[base..base + pat_len];
        if (1..REPS)
            .all(|r| tokens[base + r * pat_len..base + (r + 1) * pat_len] == *pattern)
        {
            return Some(pat_len);
        }
    }
    None
}

/// Build a ChatML prompt from multi-turn messages.
pub fn build_multi_turn_prompt(
    system_prompt: &str,
    messages: &[(String, String)], // (role, content) pairs
) -> String {
    let mut out = String::new();
    out.push_str("<|startoftext|>");
    if !system_prompt.is_empty() {
        out.push_str("<|im_start|>system\n");
        out.push_str(system_prompt);
        out.push_str("<|im_end|>\n");
    }
    for (role, content) in messages {
        out.push_str("<|im_start|>");
        out.push_str(role);
        out.push('\n');
        out.push_str(content);
        out.push_str("<|im_end|>\n");
    }
    out.push_str("<|im_start|>assistant\n");
    out
}

/// Build a ChatML prompt with vision placeholders.
/// Inserts `<|vision_start|><|image_pad|><|vision_end|>` once per image
/// before the last user message content.
pub fn build_multi_turn_prompt_vl(
    system_prompt: &str,
    messages: &[(String, String)],
    num_images: usize,
) -> String {
    let mut out = String::new();
    out.push_str("<|startoftext|>");
    if !system_prompt.is_empty() {
        out.push_str("<|im_start|>system\n");
        out.push_str(system_prompt);
        out.push_str("<|im_end|>\n");
    }
    let len = messages.len();
    for (i, (role, content)) in messages.iter().enumerate() {
        out.push_str("<|im_start|>");
        out.push_str(role);
        out.push('\n');
        // Insert vision tokens before the last user message's content
        if i == len - 1 && role == "user" {
            for _ in 0..num_images {
                out.push_str("<|vision_start|><|image_pad|><|vision_end|>");
            }
        }
        out.push_str(content);
        out.push_str("<|im_end|>\n");
    }
    out.push_str("<|im_start|>assistant\n");
    out
}
