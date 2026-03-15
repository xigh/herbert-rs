//! Herbert HTTP server for Anthropic Messages API-compatible inference.

mod handlers;
mod inference;
mod metrics;
mod prompt;
mod scheduler;
mod session_log;
mod sse;
mod types;

use axum::routing::{get, post};
use axum::Router;
use clap::Parser;
use herbert_backend_registry::{
    available_backends_help, create_backend as create_registered_backend, BackendConsumer,
};
use herbert_core::backend::{Backend, LoadOpts};
use herbert_core::cpu_detection::get_performance_cpu_count;
use inference::{GpuServerBackend, ServerInference};
use scheduler::InferenceScheduler;
use std::path::PathBuf;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tower_http::cors::CorsLayer;
use tracing::{debug, info};

#[derive(Parser)]
#[command(name = "herbert-server")]
#[command(about = "Herbert – Anthropic Messages API server for local LLM inference")]
struct Cli {
    /// Model directory (required)
    #[arg(long)]
    model: Option<PathBuf>,

    /// Backend name (default: auto; use --backend help to list options)
    #[arg(long, default_value = "auto")]
    backend: String,

    /// Listen address (required, e.g. 0.0.0.0:8080)
    #[arg(long)]
    addr: Option<String>,

    /// API key for authentication
    #[arg(long)]
    api_key: Option<String>,

    /// Thread count override
    #[arg(long)]
    num_threads: Option<usize>,

    /// Sampling temperature (0 = greedy argmax)
    #[arg(long, default_value_t = 0.4)]
    temperature: f32,

    /// Top-k sampling (0 = disabled)
    #[arg(long, default_value_t = 40)]
    top_k: usize,

    /// Top-p / nucleus sampling (1.0 = disabled)
    #[arg(long, default_value_t = 0.9)]
    top_p: f32,

    /// Max tokens in <think> before forcing </think> (0 = unlimited)
    #[arg(long, default_value_t = 0)]
    think_budget: usize,

    /// Disable thinking (prefill empty <think> block per Qwen3 template)
    #[arg(long)]
    nothink: bool,

    /// Disable quantized weight cache (force re-quantization)
    #[arg(long)]
    no_cache: bool,

    /// Prefill chunk size (split long prefills into chunks of N tokens)
    #[arg(long)]
    prefill_chunk_size: Option<usize>,

    /// KV cache quantization: f32, bf16, or int8 (default, halves KV bandwidth)
    #[arg(long, default_value = "int8")]
    kv_quant: String,

    /// Max concurrent inference requests (default: 1 = serial)
    #[arg(long, default_value_t = 1)]
    max_concurrent: usize,

    /// KV cache size in tokens (default: 20480, increase for long prompts)
    #[arg(long)]
    kv_size: Option<usize>,

    /// H2O KV cache budget: max cached positions before eviction (default: disabled)
    #[arg(long)]
    kv_budget: Option<usize>,

    /// Embedding model directory (enables /v1/embeddings endpoint)
    #[arg(long)]
    embed_model: Option<PathBuf>,

    /// Embedding KV cache size in tokens (default: 512)
    #[arg(long, default_value_t = 512)]
    embed_kv_size: usize,
}

pub(crate) struct EmbedState {
    pub(crate) backend: std::sync::Mutex<Box<dyn Backend>>,
    pub(crate) tokenizer: Arc<Tokenizer>,
    pub(crate) model_name: String,
    pub(crate) semaphore: tokio::sync::Semaphore,
}

pub(crate) struct AppState {
    pub(crate) scheduler: Option<InferenceScheduler>,
    pub(crate) api_key: Option<String>,
    pub(crate) embed: Option<EmbedState>,
}

fn create_backend(name: &str) -> anyhow::Result<Box<dyn Backend>> {
    create_registered_backend(BackendConsumer::Server, name)
}

fn load_backend(
    name: &str,
    model_path: &PathBuf,
    opts: LoadOpts,
) -> anyhow::Result<Arc<dyn ServerInference>> {
    let mut backend = create_backend(name)?;
    backend.load(model_path, opts)?;
    let server_backend = GpuServerBackend::new(backend)?;
    Ok(Arc::new(server_backend))
}

fn effective_thread_count(cli_threads: Option<usize>) -> usize {
    cli_threads
        .or_else(|| {
            std::env::var("NUM_THREADS")
                .ok()
                .and_then(|v| v.parse().ok())
        })
        .filter(|v| *v > 0)
        .unwrap_or_else(get_performance_cpu_count)
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .without_time()
        .init();

    let cli = Cli::parse();

    if cli.backend == "help" {
        eprint!("{}", available_backends_help(BackendConsumer::Server));
        return Ok(());
    }

    // Embed-only mode: --embed-model without --model
    let embed_only = cli.model.is_none() && cli.embed_model.is_some();
    if cli.model.is_none() && cli.embed_model.is_none() {
        anyhow::bail!("--model (or --embed-model for embed-only mode) is required");
    }

    let addr = cli
        .addr
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("--addr is required"))?;

    let thread_count = effective_thread_count(cli.num_threads);
    let wall_start = std::time::Instant::now();

    // Load LLM backend (unless embed-only mode)
    let backend_name = &cli.backend;
    let scheduler = if let Some(ref model) = cli.model {
        let kv_quant: herbert_core::config::KvQuantType = cli.kv_quant.parse()
            .map_err(|e: String| anyhow::anyhow!(e))?;

        let prefill_chunk_size = cli
            .prefill_chunk_size
            .or_else(|| {
                std::env::var("HERBERT_PREFILL_CHUNK_SIZE")
                    .ok()
                    .and_then(|v| v.parse().ok())
            })
            .filter(|v| *v > 0);

        debug!("Loading tokenizer");
        let tokenizer_path = model.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        let reserve_hint = cli.kv_size.unwrap_or_else(|| {
            let config_path = model.join("config.json");
            let model_max = std::fs::read_to_string(&config_path)
                .ok()
                .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
                .and_then(|v| {
                    v.get("max_position_embeddings")
                        .or_else(|| v.get("text_config").and_then(|tc| tc.get("max_position_embeddings")))
                        .and_then(|v| v.as_u64())
                        .map(|v| v as usize)
                })
                .unwrap_or(0);
            model_max.max(4096).min(32768)
        });

        debug!(backend = %backend_name, "Loading LLM backend");
        let t0 = std::time::Instant::now();

        let backend = load_backend(
            backend_name,
            model,
            LoadOpts {
                num_threads: Some(thread_count),
                kv_reserve_tokens: Some(reserve_hint),
                show_progress: true,
                no_cache: cli.no_cache,
                prefill_chunk_size,
                kv_quant,
                kv_budget: cli.kv_budget,
                ..Default::default()
            },
        )?;
        let model_load_time = t0.elapsed();

        if backend.config().is_moe() {
            herbert_core::moe_stats::global().enable();
        }

        let eos_token_id = backend.config().eos_token();

        let im_end_id = tokenizer
            .token_to_id("<|im_end|>")
            .or_else(|| tokenizer.token_to_id("<|end|>"))
            .unwrap_or(151645);

        let (tool_call_open_id, tool_call_close_id) = {
            let qwen_open = tokenizer.token_to_id("<tool_call>");
            let qwen_close = tokenizer.token_to_id("</tool_call>");
            if qwen_open.is_some() && qwen_close.is_some() {
                (qwen_open, qwen_close)
            } else {
                let lfm2_open = tokenizer.token_to_id("<|tool_call_start|>");
                let lfm2_close = tokenizer.token_to_id("<|tool_call_end|>");
                if lfm2_open.is_some() && lfm2_close.is_some() {
                    (lfm2_open, lfm2_close)
                } else {
                    let phi4_open = tokenizer.token_to_id("<|tool_call|>");
                    let phi4_close = tokenizer.token_to_id("<|/tool_call|>");
                    if phi4_open.is_some() && phi4_close.is_some() {
                        (phi4_open, phi4_close)
                    } else {
                        (qwen_open, qwen_close)
                    }
                }
            }
        };

        let think_open_id = tokenizer.token_to_id("<think>").unwrap_or(151667);
        let think_close_id = tokenizer.token_to_id("</think>").unwrap_or(151668);
        let newline_id = tokenizer
            .encode("\n", false)
            .ok()
            .and_then(|enc| {
                let ids = enc.get_ids();
                if ids.len() == 1 { Some(ids[0]) } else { None }
            })
            .unwrap_or(198);

        let model_name = model.file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();

        eprintln!(
            "Model loaded in {:.3}s ({} threads, backend={}, max_concurrent={})",
            model_load_time.as_secs_f64(),
            thread_count,
            backend_name,
            cli.max_concurrent,
        );
        eprintln!(
            "Tool tokens: open={:?}, close={:?}, think_open={}, think_close={}, im_end={}",
            tool_call_open_id, tool_call_close_id, think_open_id, think_close_id, im_end_id,
        );

        Some(InferenceScheduler {
            backend,
            tokenizer: Arc::new(tokenizer),
            semaphore: Arc::new(tokio::sync::Semaphore::new(cli.max_concurrent)),
            metrics: Arc::new(metrics::InferenceMetrics::new()),
            eos_token_id,
            im_end_id,
            think_open_id,
            think_close_id,
            newline_id,
            tool_call_open_id,
            tool_call_close_id,
            model_name,
            default_temperature: cli.temperature,
            default_top_k: cli.top_k,
            default_top_p: cli.top_p,
            think_budget: cli.think_budget,
            nothink: cli.nothink,
            max_concurrent: cli.max_concurrent,
        })
    } else {
        None
    };

    // Load embedding model (if --embed-model is specified)
    let embed_backend_name = backend_name.as_str();
    let embed_state = if let Some(ref embed_path) = cli.embed_model {
        eprintln!("Loading embedding model...");
        let t0_embed = std::time::Instant::now();
        let mut embed_backend = create_backend(embed_backend_name)?;
        embed_backend.load(
            embed_path,
            LoadOpts {
                num_threads: Some(thread_count),
                kv_reserve_tokens: Some(cli.embed_kv_size),
                show_progress: false,
                no_cache: cli.no_cache,
                ..Default::default()
            },
        )?;
        let embed_tokenizer_path = embed_path.join("tokenizer.json");
        let embed_tokenizer = Tokenizer::from_file(&embed_tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load embed tokenizer: {}", e))?;
        let embed_model_name = embed_path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        let embed_load_time = t0_embed.elapsed();
        eprintln!(
            "Embedding model loaded in {:.3}s ({})",
            embed_load_time.as_secs_f64(),
            embed_model_name,
        );
        Some(EmbedState {
            backend: std::sync::Mutex::new(embed_backend),
            tokenizer: Arc::new(embed_tokenizer),
            model_name: embed_model_name,
            semaphore: tokio::sync::Semaphore::new(1),
        })
    } else {
        None
    };

    info!(
        wall_secs = wall_start.elapsed().as_secs_f64(),
        embed_only = embed_only,
        "Server ready"
    );

    let state = Arc::new(AppState {
        scheduler,
        api_key: cli.api_key.clone(),
        embed: embed_state,
    });

    let mut app = Router::new()
        .route("/", get(handlers::health_check))
        .route("/v1/embeddings", post(handlers::embeddings_handler))
        .route("/v1/tokenize", post(handlers::tokenize_handler));

    if !embed_only {
        app = app
            .route("/v1/messages", post(handlers::messages_handler))
            .route("/v1/messages/count_tokens", post(handlers::count_tokens_handler))
            .route("/v1/metrics", get(handlers::metrics_handler));
    }

    let app = app
        .layer(CorsLayer::permissive())
        .with_state(state);

    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        let listener = tokio::net::TcpListener::bind(addr).await?;
        if embed_only {
            eprintln!("Server listening on {} (embed-only mode: /v1/embeddings + /v1/tokenize)", addr);
        } else {
            eprintln!("Server listening on {}", addr);
        }
        if cli.api_key.is_some() {
            eprintln!("API key authentication enabled");
        } else {
            eprintln!("WARNING: No API key configured, server is open to all requests");
        }
        if !embed_only {
            eprintln!();
            eprintln!("Usage with Claude Code:");
            eprintln!("  export ANTHROPIC_BASE_URL=http://{}", addr);
            if cli.api_key.is_some() {
                eprintln!("  export ANTHROPIC_API_KEY=<your-key>");
            }
        }

        axum::serve(listener, app).await?;
        Ok::<(), anyhow::Error>(())
    })?;

    Ok(())
}
