use herbert_core::backend::{Backend, VisionEmbedding};
use herbert_core::config::Config;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::Mutex;
use tokenizers::Tokenizer;

pub struct ModelState {
    pub backend: Box<dyn Backend>,
    pub tokenizer: Tokenizer,
    pub config: Config,
    pub eos_token_id: u32,
    pub im_end_id: u32,
    pub think_id: Option<u32>,
    #[allow(dead_code)] // reserved for tool-calling UI (Phase 4)
    pub end_think_id: Option<u32>,
    pub model_path: PathBuf,
}

pub struct AppState {
    pub model: Mutex<Option<ModelState>>,
    pub generating: AtomicBool,
    pub cancel_flag: AtomicBool,
    pub data_dir: PathBuf,
    pub vision_embeddings: Mutex<HashMap<String, VisionEmbedding>>,
    pub cancel_vision: Mutex<HashSet<String>>,
}

impl AppState {
    pub fn new() -> Self {
        let data_dir = dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".herbert");
        Self {
            model: Mutex::new(None),
            generating: AtomicBool::new(false),
            cancel_flag: AtomicBool::new(false),
            data_dir,
            vision_embeddings: Mutex::new(HashMap::new()),
            cancel_vision: Mutex::new(HashSet::new()),
        }
    }
}
