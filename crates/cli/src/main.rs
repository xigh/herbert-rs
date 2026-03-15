//! Herbert CLI for local LLM inference.

mod arch;
mod chat;
mod sampler;
mod token_parser;
mod tools;

use clap::Parser;
use herbert_backend_registry::{
    available_backends_help, create_backend, BackendConsumer,
};
use herbert_core::backend::{LoadOpts, RunOpts, VisionEmbedding};
use herbert_core::kv_cache::KvHandle;
use herbert_core::cpu_detection::get_performance_cpu_count;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use token_parser::parse_tokens;
use tokenizers::Tokenizer;
use tracing::{debug, info};

/// Global shutdown flag — set by SIGINT/SIGTERM/SIGHUP handler.
/// Checked during decode loop to allow graceful Metal cleanup.
static SHUTDOWN: AtomicBool = AtomicBool::new(false);

/// Returns true if a graceful shutdown has been requested.
fn shutdown_requested() -> bool {
    SHUTDOWN.load(Ordering::Relaxed)
}

/// Install signal handlers for graceful shutdown.
/// First signal: set SHUTDOWN flag (decode loop breaks, Metal cleans up via Drop).
/// Second signal: force exit (in case cleanup hangs).
fn install_signal_handlers() {
    unsafe {
        for sig in [libc::SIGINT, libc::SIGTERM, libc::SIGHUP] {
            libc::signal(sig, graceful_signal_handler as libc::sighandler_t);
        }
    }
}

extern "C" fn graceful_signal_handler(_sig: libc::c_int) {
    static SIGNAL_COUNT: AtomicU32 = AtomicU32::new(0);
    let n = SIGNAL_COUNT.fetch_add(1, Ordering::Relaxed);
    SHUTDOWN.store(true, Ordering::Relaxed);
    if n >= 1 {
        // Second signal: force exit immediately
        unsafe { libc::_exit(130) };
    }
}

use std::sync::atomic::AtomicU32;

#[derive(Parser)]
#[command(name = "herbert")]
#[command(about = "Herbert – local LLM inference")]
struct Cli {
    /// Show hardware info (CPU, SIMD, cache, RAM, bandwidth) and exit
    #[arg(long)]
    arch: bool,

    /// Model directory
    #[arg(long)]
    model: Option<PathBuf>,

    /// Backend name (default: auto; use --backend help to list options)
    #[arg(long, default_value = "auto")]
    backend: String,

    /// Input prompt (if omitted and no --prompt-file, starts interactive chat mode)
    #[arg(long)]
    prompt: Option<String>,

    /// Read prompt from a file
    #[arg(long)]
    prompt_file: Option<PathBuf>,

    /// System prompt (prepended with chat template)
    #[arg(long)]
    system: Option<String>,

    /// Read system prompt from a file
    #[arg(long)]
    system_file: Option<PathBuf>,

    /// Image file(s) for VL models (repeatable)
    #[arg(long)]
    image: Vec<PathBuf>,

    /// Max tokens to generate (default: 2048)
    #[arg(long, default_value_t = 2048)]
    max_tokens: usize,

    /// Ignore EOS token (force generation up to max-tokens)
    #[arg(long)]
    ignore_eos: bool,

    /// Disable repetition loop detection (allow repeated output)
    #[arg(long)]
    no_repeat_check: bool,

    /// Stop sequence (repeatable, supports \n etc.)
    #[arg(long = "stop")]
    stop: Vec<String>,

    /// Thread count override
    #[arg(long)]
    num_threads: Option<usize>,

    /// Prefill chunk size (split long prefills into chunks of N tokens)
    #[arg(long)]
    prefill_chunk_size: Option<usize>,

    /// Max threads during decode (0=no cap, default=auto: 2/3 of thread count)
    #[arg(long)]
    decode_threads: Option<usize>,

    /// Directory for prefix KV cache (enables caching of repeated prefills)
    #[arg(long)]
    prefix_cache_dir: Option<PathBuf>,

    /// Disable quantized weight cache (force re-quantization)
    #[arg(long)]
    no_cache: bool,

    /// Show stats after generation
    #[arg(long)]
    verbose: bool,

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

    /// Show special tokens in output (e.g. <think>, </think>)
    #[arg(long)]
    show_specials: bool,

    /// Enable tool calling (get_datetime, calculate, list_directory, read_file)
    #[arg(long)]
    tools: bool,

    /// Embedding model directory (for vision-language RAG, e.g. Qwen3-Embedding-0.6B)
    #[arg(long)]
    embed_model: Option<PathBuf>,

    /// Split model across two GPUs at this layer index (dual-GPU mode).
    /// Use a number (e.g. 24) or "auto" for num_layers/2.
    #[arg(long)]
    gpu_split: Option<String>,

    /// KV cache quantization: f32, bf16, or int8 (default, halves KV bandwidth)
    #[arg(long, default_value = "int8")]
    kv_quant: String,

    /// Q format for attention dot products: f32 (default) or bf16 (VDPBF16PS optimization)
    #[arg(long, default_value = "f32")]
    q_format: String,

    /// KV cache budget for H2O eviction (e.g. 4096). When the cache exceeds this
    /// size, low-importance positions are evicted. Disabled by default.
    #[arg(long)]
    kv_budget: Option<usize>,

    /// GPU device index (0=first discrete, 1000+=global for iGPU). Use "list" to show available GPUs.
    #[arg(long, value_name = "INDEX")]
    gpu: Option<String>,

    /// Draft model directory for speculative decoding
    #[arg(long)]
    draft_model: Option<PathBuf>,

    /// EAGLE-3 head directory for speculative decoding
    #[arg(long)]
    eagle3_head: Option<PathBuf>,

    /// Number of draft tokens per speculative step (default: 5)
    #[arg(long, default_value_t = 5)]
    draft_k: usize,
}

fn make_backend(name: &str, _gpu_split: Option<usize>) -> anyhow::Result<Box<dyn herbert_core::backend::Backend>> {
    create_backend(BackendConsumer::Cli, name)
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

fn decode_escaped_text(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut chars = raw.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('n') => out.push('\n'),
            Some('r') => out.push('\r'),
            Some('t') => out.push('\t'),
            Some('0') => out.push('\0'),
            Some('\\') => out.push('\\'),
            Some(other) => {
                out.push('\\');
                out.push(other);
            }
            None => out.push('\\'),
        }
    }
    out
}

fn find_stop_sequence<'a>(text: &str, stop_sequences: &'a [String]) -> Option<&'a str> {
    let max_stop_len = stop_sequences.iter().map(|s| s.len()).max().unwrap_or(0);
    // Only search the tail of the text — stop sequences can't start before (len - max_stop_len)
    let search_start = text.len().saturating_sub(max_stop_len + 64);
    let search_text = &text[search_start..];
    for seq in stop_sequences {
        if search_text.contains(seq.as_str()) {
            return Some(seq.as_str());
        }
    }
    None
}

fn normalize_stop_sequences(raw: Vec<String>) -> Vec<String> {
    let mut out = Vec::new();
    for seq in raw {
        let decoded = decode_escaped_text(seq.trim());
        if !decoded.is_empty() && !out.contains(&decoded) {
            out.push(decoded);
        }
    }
    out
}

pub(crate) fn build_prompt(user_prompt: &str, system_prompt: Option<&str>, num_images: usize, model_family: herbert_core::config::ModelFamily) -> String {
    let mut out = String::new();

    match model_family {
        herbert_core::config::ModelFamily::Mistral3 => {
            // Mistral instruct template: <s>[SYSTEM_PROMPT]sys[/SYSTEM_PROMPT][INST]...[/INST]
            out.push_str("<s>");
            if let Some(sys) = system_prompt {
                out.push_str("[SYSTEM_PROMPT]");
                out.push_str(sys);
                out.push_str("[/SYSTEM_PROMPT]");
            }
            out.push_str("[INST]");
            // Pixtral uses [IMG] token before user text
            for _ in 0..num_images {
                out.push_str("[IMG]");
            }
            out.push_str(user_prompt);
            out.push_str("[/INST]");
        }
        _ => {
            // ChatML template for Qwen3
            if let Some(sys) = system_prompt {
                out.push_str("<|im_start|>system\n");
                out.push_str(sys);
                out.push_str("<|im_end|>\n");
            }
            out.push_str("<|im_start|>user\n");
            // Qwen3-VL uses <|vision_start|><|image_pad|><|vision_end|>
            for _ in 0..num_images {
                out.push_str("<|vision_start|><|image_pad|><|vision_end|>");
            }
            out.push_str(user_prompt);
            out.push_str("<|im_end|>\n<|im_start|>assistant\n");
        }
    }
    out
}

/// Load an image file and return raw RGB bytes (H*W*3).
pub(crate) fn load_image_rgb(path: &std::path::Path) -> anyhow::Result<Vec<u8>> {
    let img = image::open(path)
        .map_err(|e| anyhow::anyhow!("Failed to load image {:?}: {}", path, e))?;
    let rgb = img.to_rgb8();
    Ok(rgb.into_raw())
}

/// Get image dimensions (height, width) from an image file.
pub(crate) fn image_dimensions(path: &std::path::Path) -> anyhow::Result<(usize, usize)> {
    let (w, h) = image::image_dimensions(path)
        .map_err(|e| anyhow::anyhow!("Failed to read image dimensions {:?}: {}", path, e))?;
    Ok((h as usize, w as usize))
}

/// Build the system prefix text (with chat template markers, without the rest of the prompt).
fn system_prefix_text(system: &str, model_family: herbert_core::config::ModelFamily) -> String {
    match model_family {
        herbert_core::config::ModelFamily::Mistral3 => {
            format!("<s>[SYSTEM_PROMPT]{}[/SYSTEM_PROMPT]", system)
        }
        _ => {
            // ChatML template for Qwen3 and Qwen3.5
            format!("<|im_start|>system\n{}<|im_end|>\n", system)
        }
    }
}

/// Get KV cache seq_len from a KvHandle (for /stats).
fn kv_seq_len(kv: &herbert_core::kv_cache::KvHandle) -> usize {
    kv.get_ref::<herbert_backend_common::kv_cache::CpuKvCache>()
        .map(|c| c.kv.seq_len)
        .unwrap_or(0)
}

fn main() -> anyhow::Result<()> {
    install_signal_handlers();

    // Init tracing: RUST_LOG=debug or RUST_LOG=trace to see details
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .without_time()
        .init();


    // Check THP status when hugepages feature is enabled
    #[cfg(all(target_os = "linux", feature = "hugepages"))]
    {
        if let Ok(content) = std::fs::read_to_string("/sys/kernel/mm/transparent_hugepage/enabled") {
            let trimmed = content.trim();
            if !trimmed.contains("[madvise]") && !trimmed.contains("[always]") {
                eprintln!(
                    "[hugepages] WARNING: THP is '{}'. For best performance, run:",
                    trimmed
                );
                eprintln!("  echo madvise | sudo tee /sys/kernel/mm/transparent_hugepage/enabled");
            }
        }
    }

    let cli = Cli::parse();

    if cli.arch {
        arch::print_arch();
        return Ok(());
    }

    // --backend help: list available backends and exit
    if cli.backend == "help" {
        eprint!("{}", available_backends_help(BackendConsumer::Cli));
        return Ok(());
    }

    // --gpu list: enumerate Vulkan devices and exit
    if cli.gpu.as_deref() == Some("list") {
        herbert_backend_vulkan::VulkanBackend::list_devices()?;
        return Ok(());
    }

    // Parse --gpu as usize if provided
    let gpu_index: Option<usize> = match &cli.gpu {
        Some(val) => Some(val.parse::<usize>().map_err(|_| {
            anyhow::anyhow!("--gpu must be a number or \"list\", got: {}", val)
        })?),
        None => None,
    };

    let model = cli.model.as_ref().ok_or_else(|| anyhow::anyhow!("--model is required"))?;
    let backend_name = &cli.backend;

    // Resolve system prompt from --system or --system-file
    let system_prompt = match (&cli.system, &cli.system_file) {
        (Some(s), _) => Some(s.clone()),
        (None, Some(path)) => Some(std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("Failed to read system file {:?}: {}", path, e))?),
        (None, None) => None,
    };

    let thread_count = effective_thread_count(cli.num_threads);

    // Resolve prefill-chunk-size: CLI flag > env var > None (no chunking)
    let prefill_chunk_size = cli
        .prefill_chunk_size
        .or_else(|| {
            std::env::var("HERBERT_PREFILL_CHUNK_SIZE")
                .ok()
                .and_then(|v| v.parse().ok())
        })
        .filter(|v| *v > 0);

    // Resolve prefix-cache-dir: CLI flag > env var > auto (when --system is used) > None
    let prefix_cache_dir = cli
        .prefix_cache_dir
        .or_else(|| {
            std::env::var("HERBERT_PREFIX_CACHE_DIR")
                .ok()
                .map(PathBuf::from)
        })
        .or_else(|| {
            if system_prompt.is_some() {
                std::env::var("HOME").ok().map(|h| {
                    PathBuf::from(h).join(".cache").join("qwen3").join("prefix_kv")
                })
            } else {
                None
            }
        });
    let prefix_cache_dir_display = prefix_cache_dir.as_ref().map(|d| d.display().to_string());

    let stop_sequences = normalize_stop_sequences(cli.stop);
    let use_stop_sequences = !stop_sequences.is_empty();
    let max_tokens = cli.max_tokens;

    // Detect chat mode: no --prompt and no --prompt-file
    let chat_mode = cli.prompt.is_none() && cli.prompt_file.is_none();

    // Resolve prompt (only needed for single-shot mode)
    let prompt = match (&cli.prompt, &cli.prompt_file) {
        (Some(p), _) => Some(p.clone()),
        (None, Some(path)) => Some(std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("Failed to read prompt file {:?}: {}", path, e))?),
        (None, None) => None,
    };

    // Load tokenizer
    debug!("Loading tokenizer");
    let tokenizer_path = model.join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    // Parse model config early (needed for model-family-specific prompt formatting and gpu-split auto)
    let model_config = herbert_core::config::Config::from_file(&model.join("config.json")).ok();
    let model_family = model_config.as_ref()
        .map(|c| c.model_family)
        .unwrap_or(herbert_core::config::ModelFamily::Qwen3);

    // Parse --gpu-split: "auto" → num_layers/2, or a specific layer number
    let gpu_split: Option<usize> = if let Some(ref split_str) = cli.gpu_split {
        if split_str == "auto" {
            let num_layers = model_config.as_ref()
                .map(|c| c.num_layers)
                .ok_or_else(|| anyhow::anyhow!("--gpu-split auto requires a valid config.json"))?;
            let split = num_layers / 2;
            eprintln!("[gpu-split] auto → split at layer {} (of {})", split, num_layers);
            Some(split)
        } else {
            let layer: usize = split_str.parse()
                .map_err(|_| anyhow::anyhow!("--gpu-split must be a number or 'auto', got: {}", split_str))?;
            Some(layer)
        }
    } else {
        None
    };

    // Parse KV cache quantization type
    let kv_quant: herbert_core::config::KvQuantType = cli.kv_quant.parse()
        .map_err(|e: String| anyhow::anyhow!(e))?;

    // Parse Q format for attention dot products
    let use_q_bf16 = match cli.q_format.as_str() {
        "bf16" => true,
        "f32" => false,
        _ => anyhow::bail!("Invalid --q-format: '{}'. Expected 'f32' or 'bf16'.", cli.q_format),
    };

    // Create and load backend
    let wall_start = std::time::Instant::now();

    debug!(backend = %backend_name, %kv_quant, "Loading LLM backend");
    let t0 = std::time::Instant::now();
    let mut backend: Box<dyn herbert_core::backend::Backend> = make_backend(backend_name, gpu_split)?;

    // For KV reserve hint: prompt tokens + max_tokens + headroom.
    // No artificial 64K floor — let the user control allocation via --kv-budget.
    let reserve_hint = if let Some(ref p) = prompt {
        let full = build_prompt(p, system_prompt.as_deref(), cli.image.len(), model_family);
        let toks = parse_tokens(Some(&tokenizer), &full)?;
        // 2× headroom for thinking tokens, clamped to reasonable bounds
        let needed = toks.len().saturating_add(max_tokens * 2);
        needed.max(4096).min(65536)
    } else {
        // Chat mode: reserve for multi-turn conversation
        (8192 + max_tokens * 2).min(65536)
    };

    backend.load(
        model,
        LoadOpts {
            num_threads: Some(thread_count),
            kv_reserve_tokens: Some(reserve_hint),
            show_progress: true,
            no_cache: cli.no_cache,
            prefill_chunk_size,
            decode_threads: cli.decode_threads,
            prefix_cache_dir,
            kv_quant,
            use_q_bf16,
            kv_budget: cli.kv_budget,
            gpu_index,
            ..Default::default()
        },
    )?;
    let model_load_time = t0.elapsed();

    // Load embedding model (if --embed-model is specified)
    let mut embed_backend: Option<Box<dyn herbert_core::backend::Backend>> = if let Some(ref embed_path) = cli.embed_model {
        eprintln!("Loading embedding model...");
        let t0_embed = std::time::Instant::now();
        let mut embed = make_backend(backend_name, None)?;
        embed.load(
            embed_path,
            LoadOpts {
                num_threads: Some(thread_count),
                kv_reserve_tokens: Some(512),
                show_progress: false,
                no_cache: cli.no_cache,
                prefill_chunk_size: None,
                prefix_cache_dir: None,
                ..Default::default()
            },
        )?;
        let embed_load_time = t0_embed.elapsed();
        eprintln!("Embedding model loaded in {:.3}s", embed_load_time.as_secs_f64());
        info!(elapsed_ms = embed_load_time.as_millis() as u64, "Embedding model loaded");
        Some(embed)
    } else {
        None
    };

    // Load draft model for speculative decoding (if --draft-model is specified)
    let mut draft_backend: Option<Box<dyn herbert_core::backend::Backend>> = if let Some(ref draft_path) = cli.draft_model {
        eprintln!("Loading draft model for speculative decoding...");
        let t0_draft = std::time::Instant::now();
        let mut draft = make_backend(backend_name, None)?;
        draft.load(
            draft_path,
            LoadOpts {
                num_threads: Some(thread_count),
                kv_reserve_tokens: Some(reserve_hint),
                show_progress: false,
                no_cache: cli.no_cache,
                prefill_chunk_size,
                decode_threads: cli.decode_threads,
                prefix_cache_dir: None,
                kv_quant,
                use_q_bf16,
                kv_budget: None,
                ..Default::default()
            },
        )?;
        let draft_load_time = t0_draft.elapsed();
        eprintln!("Draft model loaded in {:.3}s (K={})", draft_load_time.as_secs_f64(), cli.draft_k);
        info!(elapsed_ms = draft_load_time.as_millis() as u64, k = cli.draft_k, "Draft model loaded");
        Some(draft)
    } else {
        None
    };

    // Load EAGLE-3 head for speculative decoding (if --eagle3-head is specified)
    let mut eagle3_head: Option<herbert_backend_common::eagle3::Eagle3Head> = if let Some(ref eagle3_path) = cli.eagle3_head {
        eprintln!("Loading EAGLE-3 head for speculative decoding...");
        let t0_eagle3 = std::time::Instant::now();
        let target_config = backend.config()
            .ok_or_else(|| anyhow::anyhow!("Target model config not available for EAGLE-3"))?
            .clone();
        let head = herbert_backend_common::eagle3::Eagle3Head::load(eagle3_path, &target_config)?;
        let eagle3_load_time = t0_eagle3.elapsed();
        eprintln!("EAGLE-3 head loaded in {:.3}s (K={})", eagle3_load_time.as_secs_f64(), cli.draft_k);
        Some(head)
    } else {
        None
    };

    // Load embedding tokenizer (separate from main tokenizer)
    let embed_tokenizer = if let Some(ref embed_path) = cli.embed_model {
        let embed_tokenizer_path = embed_path.join("tokenizer.json");
        let tok = Tokenizer::from_file(&embed_tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load embed tokenizer: {}", e))?;
        Some(tok)
    } else {
        None
    };

    let eos_token_id = backend
        .config()
        .map(|c| c.eos_token())
        .unwrap_or(151643);

    // Enable MoE stats for MoE models
    if let Some(cfg) = backend.config() {
        if cfg.is_moe() {
            herbert_core::moe_stats::global().enable();
        }
    }

    // Lookup turn-end token ID: <|im_end|> (ChatML/Qwen) or <|end|> (Phi)
    let im_end_id = tokenizer
        .token_to_id("<|im_end|>")
        .or_else(|| tokenizer.token_to_id("<|end|>"))
        .unwrap_or(151645);

    // Lookup tool call token IDs: try Qwen3 first, fall back to LFM2, then Phi4
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
    // Lookup Mistral3 [TOOL_CALLS] token ID
    let mistral_tool_calls_id = tokenizer.token_to_id("[TOOL_CALLS]");

    if cli.tools {
        match (tool_call_open_id, tool_call_close_id) {
            (Some(open), Some(close)) => {
                info!(open_id = open, close_id = close, "Tool call tokens found");
            }
            _ => {
                if let Some(tc_id) = mistral_tool_calls_id {
                    info!(tool_calls_id = tc_id, "Mistral [TOOL_CALLS] token found");
                } else {
                    eprintln!(
                        "[tools] WARNING: tool call tokens not found in tokenizer, using text-based fallback",
                    );
                }
            }
        }
    }

    // Create sampler from CLI flags
    let mut sampler = sampler::Sampler::new(sampler::SamplerConfig {
        temperature: cli.temperature,
        top_k: cli.top_k,
        top_p: cli.top_p,
    });
    let use_sampling = !sampler.is_greedy();

    // Lookup think token IDs for think budget
    let think_open_id = tokenizer.token_to_id("<think>").unwrap_or(151667);
    let think_close_id = tokenizer.token_to_id("</think>").unwrap_or(151668);
    eprintln!("DEBUG think tokens: <think>={} (lookup={:?}), </think>={} (lookup={:?})",
        think_open_id, tokenizer.token_to_id("<think>"),
        think_close_id, tokenizer.token_to_id("</think>"));
    let newline_id = tokenizer
        .encode("\n", false)
        .ok()
        .and_then(|enc| {
            let ids = enc.get_ids();
            if ids.len() == 1 { Some(ids[0]) } else { None }
        })
        .unwrap_or(198);
    // --nothink: template prefix + logits backup (budget=1 forces \n then </think>
    // if the model ignores the empty <think></think> prefix — needed for Thinking variants)
    let effective_think_budget = if cli.nothink { 1 } else { cli.think_budget };
    let mut think_budget = sampler::ThinkBudget::new(
        effective_think_budget,
        think_open_id,
        think_close_id,
        newline_id,
    );
    let need_logits = use_sampling || think_budget.is_active();

    // When tools are enabled, inject tool definitions into the system prompt
    let tools_system_prompt = if cli.tools {
        Some(tools::build_tools_system_prompt(system_prompt.as_deref()))
    } else {
        None
    };
    let effective_system = if cli.tools {
        tools_system_prompt.as_deref()
    } else {
        system_prompt.as_deref()
    };

    // Compute system prefix length for prefix cache.
    let system_prefix_len = if let Some(sys) = effective_system {
        let sys_text = system_prefix_text(sys, model_family);
        let sys_tokens = parse_tokens(Some(&tokenizer), &sys_text)?;
        let len = sys_tokens.len();
        info!(system_prefix_len = len, "System prefix tokenized");
        Some(len)
    } else {
        None
    };

    if chat_mode {
        // ── Interactive chat mode ──
        eprintln!();
        if cli.tools {
            eprintln!("Tools enabled. Type /tools to see available tools.");
        }
        eprintln!("Type /help for commands, /quit to exit.");

        let chat_cfg = chat::ChatConfig {
            system_prompt: effective_system,
            max_tokens,
            eos_token_id,
            im_end_id,
            skip_special_tokens: !cli.show_specials,
            tools_enabled: cli.tools,
            tool_call_open_id,
            tool_call_close_id,
            mistral_tool_calls_id,
            system_prefix_len,
            model_path: Some(model.as_path()),
            no_repeat_check: cli.no_repeat_check,
            model_family,
            embed_model_path: cli.embed_model.as_deref(),
        };

        // Enter chat loop directly — it handles the first message, slash commands, everything.
        chat::run_chat(
            &mut backend,
            None,
            &tokenizer,
            chat::ChatSession::new(),
            chat::StopReason::ImEnd,
            &chat_cfg,
            &mut sampler,
            &mut think_budget,
            cli.nothink,
            &mut embed_backend,
            embed_tokenizer.as_ref(),
        )?;

        return Ok(());
    }

    // ── Single-shot mode (existing behavior) ──
    let prompt = prompt.unwrap(); // safe: chat_mode is false

    // Build prompt with chat template
    let full_prompt = build_prompt(&prompt, effective_system, cli.image.len(), model_family);
    debug!(prompt_len = full_prompt.len(), "Built chat prompt");

    // Tokenize
    let tokens = parse_tokens(Some(&tokenizer), &full_prompt)?;
    if tokens.is_empty() {
        return Err(anyhow::anyhow!("Prompt produced zero tokens"));
    }
    info!(num_tokens = tokens.len(), "Tokenized prompt");

    // Debug: dump token IDs for comparison with llama.cpp (set HERBERT_DUMP_TOKENS=1)
    if std::env::var("HERBERT_DUMP_TOKENS").is_ok() {
        eprintln!("[TOKEN_DUMP] prompt_text: {:?}", full_prompt);
        eprintln!("[TOKEN_DUMP] num_tokens: {}", tokens.len());
        eprintln!("[TOKEN_DUMP] token_ids: {:?}", tokens);
        // Decode each token individually to show the text representation
        for (i, &tid) in tokens.iter().enumerate() {
            let text = tokenizer.decode(&[tid], false).unwrap_or_else(|_| "<err>".to_string());
            eprintln!("[TOKEN_DUMP] [{:4}] id={:6} text={:?}", i, tid, text);
        }
        eprintln!("[TOKEN_DUMP] end");
    }

    // Show full prompt with special tokens
    if cli.show_specials {
        let prompt_text = tokenizer
            .decode(&tokens, false)
            .map_err(|e| anyhow::anyhow!("Detokenization failed: {}", e))?;
        eprintln!("--- Prompt ({} tokens) ---", tokens.len());
        eprintln!("{}", prompt_text);
        eprintln!("--- End prompt ---");
    }

    let run_opts = RunOpts {
        return_logits: need_logits,
        profile: false,
        decode_tokens: Some(max_tokens),
        ignore_eos: cli.ignore_eos,
        system_prefix_len,
    };

    // Vision stats (populated if --image is used)
    let mut vision_load_time = std::time::Duration::ZERO;
    let mut vision_encode_time = std::time::Duration::ZERO;
    let mut vision_image_info: Option<String> = None;
    let mut vision_grid_info: Option<String> = None;
    let mut prefill_token_count: usize = tokens.len();

    // Prefill (VL or text-only)
    let (mut kv, prefill_output, ttft) = if !cli.image.is_empty() {
        let image_path = &cli.image[0];
        debug!(path = %image_path.display(), "Loading image");
        let rgb_bytes = load_image_rgb(image_path)?;
        let (img_h, img_w) = image_dimensions(image_path)?;
        let img_file_size = std::fs::metadata(image_path)?.len();
        info!(height = img_h, width = img_w, bytes = rgb_bytes.len(), "Image loaded");

        vision_image_info = Some(format!(
            "{} ({}x{}, {:.1} KB)",
            image_path.file_name().unwrap_or_default().to_string_lossy(),
            img_w, img_h,
            img_file_size as f64 / 1024.0,
        ));

        let image_label = format!(
            "{} ({}x{})",
            image_path.file_name().unwrap_or_default().to_string_lossy(),
            img_w, img_h,
        );

        // ── Vision encoder (family-dispatched) ──
        let (vision_embed, image_pad_id) = match model_family {
            herbert_core::config::ModelFamily::Mistral3 => {
                // ── Pixtral vision encoder (Mistral3/Devstral) ──
                debug!("Loading Pixtral vision config");
                let t0 = std::time::Instant::now();
                let pixtral_config =
                    herbert_vision::pixtral_config::PixtralVisionConfig::from_file(&model.join("config.json"))?;
                info!(num_layers = pixtral_config.num_layers, hidden_size = pixtral_config.hidden_size, "Pixtral vision config loaded");

                let (patches, grid_h, grid_w) = herbert_vision::pixtral::pixtral_preprocess(
                    &rgb_bytes, img_h, img_w, &pixtral_config,
                )?;
                let num_patches = grid_h * grid_w;
                let merge = pixtral_config.spatial_merge_size;
                let merged_h = grid_h / merge;
                let merged_w = grid_w / merge;
                let num_vision_tokens = merged_h * merged_w;
                info!(grid_h, grid_w, num_patches, num_vision_tokens, "Image preprocessed (Pixtral)");

                vision_grid_info = Some(format!(
                    "{}x{} grid, {} patches, {} vision tokens",
                    grid_h, grid_w, num_patches, num_vision_tokens,
                ));

                debug!("Loading Pixtral vision encoder weights (CPU)");
                let pixtral_encoder =
                    herbert_vision::pixtral::load_pixtral_encoder_with_progress(model, &pixtral_config)?;
                vision_load_time = t0.elapsed();
                info!(elapsed_ms = vision_load_time.as_millis() as u64, "Pixtral encoder loaded");

                let t0_enc = std::time::Instant::now();
                let pixtral_output = pixtral_encoder.forward_with_label(&patches, grid_h, grid_w, &image_label)?;
                info!(
                    num_tokens = pixtral_output.num_tokens,
                    elapsed_ms = t0_enc.elapsed().as_millis() as u64,
                    "Pixtral vision encoder done (CPU)"
                );
                vision_encode_time = t0_enc.elapsed();

                let embed = VisionEmbedding {
                    hidden_states: pixtral_output.hidden_states,
                    num_tokens: pixtral_output.num_tokens,
                    grid_h: merged_h,
                    grid_w: merged_w,
                    deepstack_features: vec![],
                };

                // Mistral3 uses [IMG] token (id=10)
                let image_pad_id = model_config.as_ref()
                    .and_then(|c| c.image_token_id)
                    .unwrap_or(10);
                (embed, image_pad_id)
            }
            _ => {
                // ── Qwen3-VL vision encoder ──
                debug!("Loading Qwen3-VL vision config");
                let t0 = std::time::Instant::now();
                let vision_config =
                    herbert_vision::config::VisionConfig::from_file(&model.join("config.json"))?;
                info!(num_layers = vision_config.num_layers, hidden_size = vision_config.hidden_size, "Vision config loaded");

                let (patches, grid_t, grid_h, grid_w) = herbert_vision::image_process::preprocess_rgb(
                    &rgb_bytes, img_h, img_w, &vision_config, 65536, 16777216,
                )?;
                let num_patches = grid_t * grid_h * grid_w;
                let merged_h = grid_h / vision_config.spatial_merge_size;
                let merged_w = grid_w / vision_config.spatial_merge_size;
                let num_vision_tokens = grid_t * merged_h * merged_w;
                info!(grid_t, grid_h, grid_w, num_patches, num_vision_tokens, "Image preprocessed");

                vision_grid_info = Some(format!(
                    "{}x{} grid, {} patches, {} vision tokens",
                    grid_h, grid_w, num_patches, num_vision_tokens,
                ));

                // Try GPU vision encoder first, fall back to CPU
                debug!("Running vision encoder forward");
                let t0_enc = std::time::Instant::now();
                let embed = if let Some(gpu_result) = backend.encode_vision(&patches, grid_t, grid_h, grid_w) {
                    vision_load_time = t0.elapsed();
                    let embed = gpu_result?;
                    info!(
                        num_tokens = embed.num_tokens,
                        deepstack_layers = embed.deepstack_features.len(),
                        elapsed_ms = t0_enc.elapsed().as_millis() as u64,
                        "Vision encoder done (GPU)"
                    );
                    embed
                } else {
                    debug!("Loading vision encoder weights (CPU fallback)");
                    let vision_encoder =
                        herbert_vision::loader::load_vision_encoder_with_progress(model, &vision_config)?;
                    vision_load_time = t0.elapsed();
                    info!(elapsed_ms = vision_load_time.as_millis() as u64, "Vision encoder loaded");

                    let vision_output = vision_encoder.forward_with_label(&patches, grid_t, grid_h, grid_w, &image_label)?;
                    info!(
                        num_tokens = vision_output.num_tokens,
                        deepstack_layers = vision_output.deepstack_features.len(),
                        elapsed_ms = t0_enc.elapsed().as_millis() as u64,
                        "Vision encoder done (CPU)"
                    );
                    VisionEmbedding {
                        hidden_states: vision_output.hidden_states,
                        num_tokens: vision_output.num_tokens,
                        grid_h: merged_h,
                        grid_w: merged_w,
                        deepstack_features: vision_output.deepstack_features,
                    }
                };
                vision_encode_time = t0_enc.elapsed();

                // <|image_pad|> token ID = 151655 for Qwen3 tokenizer
                (embed, 151655u32)
            }
        };

        let image_pos = tokens
            .iter()
            .position(|&t| t == image_pad_id)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "No image placeholder token (id={}) found in prompt. Tokens: {:?}",
                    image_pad_id,
                    &tokens[..tokens.len().min(20)],
                )
            })?;
        let num_vis = vision_embed.num_tokens;
        let mut expanded_tokens = Vec::with_capacity(tokens.len() - 1 + num_vis);
        expanded_tokens.extend_from_slice(&tokens[..image_pos]);
        expanded_tokens.resize(expanded_tokens.len() + num_vis, image_pad_id);
        expanded_tokens.extend_from_slice(&tokens[image_pos + 1..]);
        prefill_token_count = expanded_tokens.len();
        info!(
            image_pos,
            num_vis,
            expanded_len = expanded_tokens.len(),
            "Token sequence expanded with vision tokens"
        );

        debug!("Starting VL prefill");
        let prefill_start = std::time::Instant::now();
        let result = backend.prefill_vl(
            &expanded_tokens,
            &[vision_embed],
            &[image_pos],
            run_opts.clone(),
        )?;
        let elapsed = prefill_start.elapsed();
        info!("VL prefill done");
        (result.0, result.1, elapsed)
    } else {
        debug!(num_tokens = tokens.len(), "Starting text prefill");
        let prefill_start = std::time::Instant::now();
        let result = backend.prefill(&tokens, run_opts.clone())?;
        let elapsed = prefill_start.elapsed();
        info!("Text prefill done");
        (result.0, result.1, elapsed)
    };

    // Extract prefix cache hit info for stats
    let cached_prefix_len = prefill_output.cached_prefix_len;

    // Decode loop — apply think budget to logits, then sample
    let target_logits = prefill_output.logits;
    let first_token = if let Some(ref logits) = target_logits {
        let mut logits_buf = logits.clone();
        think_budget.apply_to_logits(&mut logits_buf);
        if use_sampling { sampler.sample(&logits_buf) } else { sampler::argmax(&logits_buf) }
    } else {
        prefill_output.first_token
    };
    think_budget.track(first_token);

    let decode_start = std::time::Instant::now();

    // Speculative decoding: prefill draft model with same tokens
    let mut draft_kv: Option<KvHandle> = if draft_backend.is_some() {
        let draft = draft_backend.as_mut().unwrap();
        let draft_run_opts = RunOpts {
            return_logits: false,
            profile: false,
            decode_tokens: Some(max_tokens),
            ignore_eos: cli.ignore_eos,
            system_prefix_len: None,
        };
        eprintln!("[spec-dec] Prefilling draft model...");
        let t0_draft_prefill = std::time::Instant::now();
        let (draft_kv_handle, _draft_prefill_output) = draft.prefill(&tokens, draft_run_opts)?;
        eprintln!("[spec-dec] Draft prefill done in {:.3}s", t0_draft_prefill.elapsed().as_secs_f64());
        Some(draft_kv_handle)
    } else {
        None
    };

    // Enable EAGLE-3 hidden state extraction on KV cache
    if let Some(ref eagle3) = eagle3_head {
        backend.enable_eagle3_extraction(&mut kv, eagle3.extract_layers)?;
    }

    let decode_result = {
        // Use full-sequence decode to handle tokenizers with Strip decoders
        // Track printed_len (byte offset) and never print trailing U+FFFD (may resolve later).
        let mut generated_tokens = vec![first_token];
        let mut generated_text;
        let mut stop_sequence_hit: Option<String> = None;
        let mut printed_len: usize;

        // Decode and print first token
        {
            let full_text = tokenizer
                .decode(&generated_tokens, !cli.show_specials)
                .map_err(|e| anyhow::anyhow!("Detokenization failed: {}", e))?;
            let safe = chat::safe_print_len(&full_text);
            if safe > 0 {
                print!("{}", &full_text[..safe]);
            }
            printed_len = safe;
            generated_text = full_text;
            if use_stop_sequences {
                if let Some(seq) = find_stop_sequence(&generated_text, &stop_sequences) {
                    stop_sequence_hit = Some(seq.to_string());
                }
            }
            std::io::Write::flush(&mut std::io::stdout())?;
        }

        if stop_sequence_hit.is_none() && (first_token != eos_token_id) {
            let mut current_token = first_token;

            // Decode speed tracking (--verbose): report tok/s ~16 times during decode
            let chunk_size = (max_tokens / 16).max(32);
            let mut chunk_start = std::time::Instant::now();
            let mut chunk_count: usize = 0;
            let mut decode_speed_log: Vec<(usize, f64)> = Vec::new();

            let profile_overhead = std::env::var("PROFILE_OVERHEAD").is_ok();
            let mut overhead_decode_us: u64 = 0;
            let mut overhead_sampling_us: u64 = 0;
            let mut overhead_tokenizer_us: u64 = 0;
            let mut overhead_print_us: u64 = 0;
            let mut overhead_total_us: u64 = 0;
            let mut overhead_count: u64 = 0;

            // ── Speculative decoding path ──
            if let (Some(ref mut draft), Some(ref mut draft_kv_handle)) = (&mut draft_backend, &mut draft_kv) {
                let mut spec_decoder = herbert_backend_common::speculative::SpeculativeDecoder::new(cli.draft_k);
                let mut total_spec_tokens = 0usize;

                while total_spec_tokens < max_tokens {
                    if shutdown_requested() {
                        eprintln!("\n[herbert] Interrupted — cleaning up...");
                        break;
                    }

                    // Feed current_token to draft model to align KV caches
                    // (draft needs to process the same last token the target did)
                    let step_result = spec_decoder.step(
                        draft,
                        &mut backend,
                        draft_kv_handle,
                        &mut kv,
                        current_token,
                    )?;

                    // Process all produced tokens
                    let mut should_stop = false;
                    for &tok in &step_result.final_tokens {
                        generated_tokens.push(tok);
                        total_spec_tokens += 1;

                        // Detokenize and print
                        let full_text = tokenizer
                            .decode(&generated_tokens, !cli.show_specials)
                            .map_err(|e| anyhow::anyhow!("Detokenization failed: {}", e))?;
                        let safe = chat::safe_print_len(&full_text);
                        if safe > printed_len {
                            print!("{}", &full_text[printed_len..safe]);
                        }
                        printed_len = safe;
                        generated_text = full_text;
                        std::io::Write::flush(&mut std::io::stdout())?;

                        if tok == eos_token_id && !cli.ignore_eos {
                            should_stop = true;
                            break;
                        }
                        if use_stop_sequences {
                            if let Some(seq) = find_stop_sequence(&generated_text, &stop_sequences) {
                                stop_sequence_hit = Some(seq.to_string());
                                should_stop = true;
                                break;
                            }
                        }
                    }

                    if should_stop {
                        break;
                    }

                    // Update current_token to the last produced token
                    if let Some(&last) = step_result.final_tokens.last() {
                        current_token = last;
                    }

                    chunk_count += step_result.produced;
                    if cli.verbose && chunk_count >= chunk_size {
                        let chunk_elapsed = chunk_start.elapsed().as_secs_f64();
                        let tps = chunk_count as f64 / chunk_elapsed;
                        let total_decoded = generated_tokens.len() - 1;
                        decode_speed_log.push((total_decoded, tps));
                        chunk_count = 0;
                        chunk_start = std::time::Instant::now();
                    }
                }

                // Print speculative decoding stats
                eprintln!("\n[spec-dec] Steps: {}, Acceptance rate: {:.1}%, Tokens/step: {:.2}",
                    spec_decoder.total_steps,
                    spec_decoder.acceptance_rate() * 100.0,
                    spec_decoder.tokens_per_step(),
                );
            } else if let Some(ref mut eagle3) = eagle3_head {
            // ── EAGLE-3 speculative decoding path ──

            let embed_tokens = backend.embed_tokens_bf16()?.to_vec();
            let mut eagle3_spec = herbert_backend_common::eagle3_speculative::Eagle3SpecDecoder::new(cli.draft_k);
            let mut total_eagle3_tokens = 0usize;

            while total_eagle3_tokens < max_tokens {
                if shutdown_requested() {
                    eprintln!("\n[herbert] Interrupted — cleaning up...");
                    break;
                }

                // Reset EAGLE KV cache each round
                eagle3.reset_kv();

                let step_result = eagle3_spec.step(
                    eagle3,
                    &mut backend,
                    &mut kv,
                    current_token,
                    &embed_tokens,
                )?;

                // Process all produced tokens
                let mut should_stop = false;
                for &tok in &step_result.final_tokens {
                    generated_tokens.push(tok);
                    total_eagle3_tokens += 1;

                    let full_text = tokenizer
                        .decode(&generated_tokens, !cli.show_specials)
                        .map_err(|e| anyhow::anyhow!("Detokenization failed: {}", e))?;
                    let safe = chat::safe_print_len(&full_text);
                    if safe > printed_len {
                        print!("{}", &full_text[printed_len..safe]);
                    }
                    printed_len = safe;
                    generated_text = full_text;
                    std::io::Write::flush(&mut std::io::stdout())?;

                    if tok == eos_token_id && !cli.ignore_eos {
                        should_stop = true;
                        break;
                    }
                    if use_stop_sequences {
                        if let Some(seq) = find_stop_sequence(&generated_text, &stop_sequences) {
                            stop_sequence_hit = Some(seq.to_string());
                            should_stop = true;
                            break;
                        }
                    }
                }

                if should_stop {
                    break;
                }

                if let Some(&last) = step_result.final_tokens.last() {
                    current_token = last;
                }

                chunk_count += step_result.produced;
                if cli.verbose && chunk_count >= chunk_size {
                    let chunk_elapsed = chunk_start.elapsed().as_secs_f64();
                    let tps = chunk_count as f64 / chunk_elapsed;
                    let total_decoded = generated_tokens.len() - 1;
                    decode_speed_log.push((total_decoded, tps));
                    chunk_count = 0;
                    chunk_start = std::time::Instant::now();
                }
            }

            // Print EAGLE-3 speculative decoding stats
            eprintln!("\n[eagle3] Steps: {}, Acceptance rate: {:.1}%, Tokens/step: {:.2}",
                eagle3_spec.total_steps,
                eagle3_spec.acceptance_rate() * 100.0,
                eagle3_spec.tokens_per_step(),
            );
            } else {
            // ── Normal (non-speculative) decode path ──

            for _step in 0..max_tokens {
                if shutdown_requested() {
                    eprintln!("\n[herbert] Interrupted — cleaning up Metal resources...");
                    break;
                }
                let t_iter = std::time::Instant::now();

                let t_decode = std::time::Instant::now();
                let decode_output = backend.decode_next(
                    &mut kv,
                    current_token,
                    RunOpts {
                        return_logits: need_logits,
                        profile: false,
                        decode_tokens: Some(max_tokens),
                        ignore_eos: cli.ignore_eos,
                        system_prefix_len: None,
                    },
                )?;
                let decode_us = t_decode.elapsed().as_micros() as u64;

                let t_sampling = std::time::Instant::now();
                let next_token = if let Some(mut logits) = decode_output.logits {
                    think_budget.apply_to_logits(&mut logits);
                    if use_sampling { sampler.sample(&logits) } else { sampler::argmax(&logits) }
                } else {
                    decode_output.token
                };
                think_budget.track(next_token);
                let sampling_us = t_sampling.elapsed().as_micros() as u64;

                generated_tokens.push(next_token);
                chunk_count += 1;

                // Incremental decode: decode last 2 tokens to capture cross-token
                // UTF-8 byte merging, then extract the new suffix.
                // Falls back to full-sequence decode only when the join changes
                // earlier bytes (extremely rare for Qwen3's tiktoken tokenizer).
                let t_tok = std::time::Instant::now();
                let full_text = {
                    let skip = !cli.show_specials;
                    let n = generated_tokens.len();
                    if n <= 2 {
                        // Short sequence: full decode is cheap
                        tokenizer.decode(&generated_tokens, skip)
                            .map_err(|e| anyhow::anyhow!("Detokenization failed: {}", e))?
                    } else {
                        // Decode last 2 tokens to get correct join behavior
                        let pair = tokenizer.decode(&generated_tokens[n-2..], skip)
                            .map_err(|e| anyhow::anyhow!("Detokenization failed: {}", e))?;
                        let prev_single = tokenizer.decode(&generated_tokens[n-2..n-1], skip)
                            .map_err(|e| anyhow::anyhow!("Detokenization failed: {}", e))?;
                        if pair.starts_with(&prev_single) {
                            // Normal case: just append the new suffix
                            let new_chars = &pair[prev_single.len()..];
                            format!("{}{}", &generated_text, new_chars)
                        } else {
                            // Rare: cross-token UTF-8 or Strip tokenizer revised bytes.
                            // Fall back to full decode.
                            tokenizer.decode(&generated_tokens, skip)
                                .map_err(|e| anyhow::anyhow!("Detokenization failed: {}", e))?
                        }
                    }
                };
                let tokenizer_us = t_tok.elapsed().as_micros() as u64;

                // Find safe print boundary (skip trailing FFFD)
                let t_print = std::time::Instant::now();
                let print_from = if full_text.is_char_boundary(printed_len)
                    && full_text.as_bytes().get(..printed_len) == generated_text.as_bytes().get(..printed_len)
                {
                    printed_len
                } else {
                    // Tokenizer revised earlier bytes; walk back to safe boundary
                    let common = generated_text.bytes().zip(full_text.bytes())
                        .take_while(|(a, b)| a == b).count();
                    let mut start = common.min(printed_len);
                    while start > 0 && !full_text.is_char_boundary(start) { start -= 1; }
                    start
                };
                let safe = chat::safe_print_len(&full_text);
                if safe > print_from {
                    print!("{}", &full_text[print_from..safe]);
                }
                printed_len = safe;
                generated_text = full_text;

                if use_stop_sequences && stop_sequence_hit.is_none() {
                    if let Some(seq) = find_stop_sequence(&generated_text, &stop_sequences) {
                        stop_sequence_hit = Some(seq.to_string());
                    }
                }

                std::io::Write::flush(&mut std::io::stdout())?;
                let print_us = t_print.elapsed().as_micros() as u64;

                if profile_overhead {
                    let total_us = t_iter.elapsed().as_micros() as u64;
                    overhead_decode_us += decode_us;
                    overhead_sampling_us += sampling_us;
                    overhead_tokenizer_us += tokenizer_us;
                    overhead_print_us += print_us;
                    overhead_total_us += total_us;
                    overhead_count += 1;
                }

                // Log chunk speed
                if cli.verbose && chunk_count >= chunk_size {
                    let chunk_elapsed = chunk_start.elapsed().as_secs_f64();
                    let tps = chunk_count as f64 / chunk_elapsed;
                    let total_decoded = generated_tokens.len() - 1; // exclude first token from prefill
                    decode_speed_log.push((total_decoded, tps));
                    chunk_count = 0;
                    chunk_start = std::time::Instant::now();
                }

                if next_token == eos_token_id && !cli.ignore_eos {
                    break;
                }
                if stop_sequence_hit.is_some() {
                    break;
                }

                // Repetition loop detection
                if !cli.no_repeat_check {
                    if let Some(pat_len) = chat::detect_repetition_loop(&generated_tokens) {
                        eprintln!(
                            "\n[Repetition loop detected ({} tokens × 3), stopping]",
                            pat_len
                        );
                        break;
                    }
                }

                current_token = next_token;
            }

            if profile_overhead && overhead_count > 0 {
                let n = overhead_count as f64;
                let other_us = overhead_total_us.saturating_sub(overhead_decode_us + overhead_sampling_us + overhead_tokenizer_us + overhead_print_us);
                eprintln!("\n[PROFILE-OVERHEAD] Per-token breakdown ({} tokens):", overhead_count);
                eprintln!("  {:<20} {:>8.1} µs  ({:>5.1}%)", "decode_next", overhead_decode_us as f64 / n, overhead_decode_us as f64 / overhead_total_us as f64 * 100.0);
                eprintln!("  {:<20} {:>8.1} µs  ({:>5.1}%)", "sampling+think", overhead_sampling_us as f64 / n, overhead_sampling_us as f64 / overhead_total_us as f64 * 100.0);
                eprintln!("  {:<20} {:>8.1} µs  ({:>5.1}%)", "tokenizer", overhead_tokenizer_us as f64 / n, overhead_tokenizer_us as f64 / overhead_total_us as f64 * 100.0);
                eprintln!("  {:<20} {:>8.1} µs  ({:>5.1}%)", "print+flush+stop", overhead_print_us as f64 / n, overhead_print_us as f64 / overhead_total_us as f64 * 100.0);
                eprintln!("  {:<20} {:>8.1} µs  ({:>5.1}%)", "other (loop)", other_us as f64 / n, other_us as f64 / overhead_total_us as f64 * 100.0);
                eprintln!("  {:<20} {:>8.1} µs  = {:.1} ms/tok", "TOTAL", overhead_total_us as f64 / n, overhead_total_us as f64 / n / 1000.0);
                eprintln!("  {:<20} {:>8.1} tok/s", "effective", n / (overhead_total_us as f64 / 1_000_000.0));
            }

            // Log final partial chunk
            if cli.verbose && chunk_count > 0 {
                let chunk_elapsed = chunk_start.elapsed().as_secs_f64();
                if chunk_elapsed > 0.0 {
                    let tps = chunk_count as f64 / chunk_elapsed;
                    let total_decoded = generated_tokens.len() - 1;
                    decode_speed_log.push((total_decoded, tps));
                }
            }

            // Print decode speed progression
            if cli.verbose && !decode_speed_log.is_empty() {
                eprintln!();
                eprintln!("--- Decode speed ---");
                for (tok, tps) in &decode_speed_log {
                    let bar_len = (*tps as usize).min(60);
                    let bar: String = "█".repeat(bar_len);
                    eprintln!("  {:>4} tok  {:>5.1} tok/s  {}", tok, tps, bar);
                }
            }
            } // close else (non-speculative path)
        }

        let detected_loop = !cli.no_repeat_check && chat::detect_repetition_loop(&generated_tokens).is_some();
        chat::DecodeResult {
            generated_count: generated_tokens.len(),
            stop_reason: if generated_tokens.last() == Some(&eos_token_id)
                || stop_sequence_hit.is_some()
            {
                chat::StopReason::Eos
            } else if detected_loop {
                chat::StopReason::RepetitionLoop
            } else {
                chat::StopReason::MaxTokens
            },
            decode_secs: decode_start.elapsed().as_secs_f64(),
            generated_text,
            all_tokens: generated_tokens,
        }
    };

    let generated_tokens = &decode_result.all_tokens;
    let decode_elapsed = decode_start.elapsed();
    let wall_elapsed = wall_start.elapsed();

    // Ensure output ends with a newline
    println!();

    // MoE stats (always print for MoE models)
    herbert_core::moe_stats::global().print_summary();

    // Verbose stats
    if cli.verbose {
        let output_tokens = generated_tokens.len();
        let prefill_s = ttft.as_secs_f64();
        let decode_s = decode_elapsed.as_secs_f64();
        let total_s = wall_elapsed.as_secs_f64();
        let decode_token_count = output_tokens.saturating_sub(1);

        let prefill_tps = if prefill_s > 0.0 {
            prefill_token_count as f64 / prefill_s
        } else {
            0.0
        };
        let decode_tps = if decode_s > 0.0 {
            decode_token_count as f64 / decode_s
        } else {
            0.0
        };

        let total_cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        eprintln!("--- Stats ---");
        eprintln!("Threads:         {} (of {} available)", thread_count, total_cpus);
        if let Some(cs) = prefill_chunk_size {
            eprintln!("Prefill chunks:  {} tokens", cs);
        }
        if let Some(ref dir) = prefix_cache_dir_display {
            eprintln!("Prefix cache:    {}", dir);
        }
        eprintln!("Backend:         {}", backend_name);
        eprintln!("Model load:      {:.3}s", model_load_time.as_secs_f64());
        if let Some(ref img_info) = vision_image_info {
            eprintln!("Vision load:     {:.3}s", vision_load_time.as_secs_f64());
            eprintln!("Image:           {}", img_info);
            if let Some(ref grid_info) = vision_grid_info {
                eprintln!("                 {}", grid_info);
            }
            eprintln!("Vision encode:   {:.3}s", vision_encode_time.as_secs_f64());
        }
        if let Some(sys_len) = system_prefix_len {
            let question_len = prefill_token_count.saturating_sub(sys_len);
            let cache_status = if cached_prefix_len >= sys_len { "cached" } else { "computed" };
            eprintln!("System tokens:   {} ({})", sys_len, cache_status);
            eprintln!("Question tokens: {}", question_len);
        } else {
            eprintln!("Input tokens:    {}", prefill_token_count);
        }
        eprintln!("Output tokens:   {}", output_tokens);
        if let Some(sys_len) = system_prefix_len {
            if cached_prefix_len >= sys_len {
                let question_len = prefill_token_count.saturating_sub(sys_len);
                let question_tps = if prefill_s > 0.0 {
                    question_len as f64 / prefill_s
                } else {
                    0.0
                };
                eprintln!(
                    "Prefill:         {:.3}s ({:.1} tok/s, question only)",
                    prefill_s, question_tps
                );
            } else {
                eprintln!(
                    "Prefill:         {:.3}s ({:.1} tok/s)",
                    prefill_s, prefill_tps
                );
            }
        } else {
            eprintln!(
                "Prefill:         {:.3}s ({:.1} tok/s)",
                prefill_s, prefill_tps
            );
        }
        eprintln!(
            "Decode:          {:.3}s ({:.1} tok/s)",
            decode_s, decode_tps
        );
        eprintln!("Total:           {:.3}s", total_s);
    }

    // Print MoE decode profiling stats before exit
    #[cfg(feature = "profile-moe-decode")]
    herbert_backend_common::profiler::report_moe_decode();

    // Print int8 KV cache profiling stats before exit
    #[cfg(feature = "profile-kvcache-int8")]
    herbert_backend_common::profiler::flush_kvcache_int8_stats();

    Ok(())
}
