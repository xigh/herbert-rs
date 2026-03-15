//! Shared progress bar helpers for model loaders.

use indicatif::{ProgressBar, ProgressStyle};

/// Create a per-tensor progress bar for model loading with layer indication.
pub fn model_bar(total_tensors: u64, num_layers: usize) -> ProgressBar {
    let pb = ProgressBar::new(total_tensors);
    pb.set_style(
        ProgressStyle::with_template("  {msg} [{bar:30}] {pos}/{len}")
            .expect("valid progress template")
            .progress_chars("█░░"),
    );
    pb.set_message(format!("Loading model (layer 0/{})", num_layers));
    pb
}

/// Create a progress bar for loading from the weight cache.
pub fn cache_bar(num_layers: usize) -> ProgressBar {
    let pb = ProgressBar::new(num_layers as u64);
    pb.set_style(
        ProgressStyle::with_template("  Loading cache [{bar:30}] {pos}/{len} layers")
            .expect("valid progress template")
            .progress_chars("█░░"),
    );
    pb
}
