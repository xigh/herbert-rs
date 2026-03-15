//! Shared decode utilities used by both CLI and server.

/// Why decode stopped.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StopReason {
    Eos,
    ImEnd,
    MaxTokens,
    RepetitionLoop,
}

/// Check if the last tokens form a repeating loop (same pattern × 3).
/// Returns the pattern length if detected.
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
