//! Shared prompt-to-token conversion used across CLI commands.

use tokenizers::Tokenizer;

/// Tokenize a prompt string, falling back to comma-separated token IDs
/// when no tokenizer is available.
pub fn parse_tokens(tokenizer: Option<&Tokenizer>, prompt: &str) -> anyhow::Result<Vec<u32>> {
    if let Some(tok) = tokenizer {
        let encoding = tok
            .encode(prompt.to_string(), false)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
        Ok(encoding.get_ids().to_vec())
    } else if prompt.contains(',') {
        prompt
            .split(',')
            .map(|s| {
                s.trim()
                    .parse::<u32>()
                    .map_err(|e| anyhow::anyhow!("Invalid token ID '{}': {}", s.trim(), e))
            })
            .collect::<anyhow::Result<Vec<_>>>()
    } else {
        Err(anyhow::anyhow!(
            "Need tokenizer.json or comma-separated token IDs"
        ))
    }
}
