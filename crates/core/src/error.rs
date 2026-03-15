//! Error types for Herbert.

use thiserror::Error;

pub type Result<T> = std::result::Result<T, HerbertError>;

/// Error type covering all failure modes across backends.
#[derive(Error, Debug)]
pub enum HerbertError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Model loading error: {0}")]
    ModelLoad(String),

    #[error("Tensor operation error: {0}")]
    Tensor(String),

    #[error("Invalid configuration: {0}")]
    Config(String),

    #[error("Backend error: {0}")]
    Backend(String),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("SafeTensors error: {0}")]
    SafeTensors(String),
}

impl From<safetensors::SafeTensorError> for HerbertError {
    fn from(err: safetensors::SafeTensorError) -> Self {
        HerbertError::SafeTensors(err.to_string())
    }
}
