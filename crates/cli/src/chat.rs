//! Interactive multi-turn chat mode.

use crate::sampler::{Sampler, ThinkBudget};
use crate::tools;
use herbert_core::backend::{Backend, RunOpts, VisionEmbedding};
use herbert_core::kv_cache::KvHandle;
use rustyline::completion::{Completer, FilenameCompleter, Pair};
use rustyline::error::ReadlineError;
use rustyline::highlight::Highlighter;
use rustyline::hint::Hinter;
use rustyline::validate::Validator;
use rustyline::{Context, Editor, Helper};
use std::path::{Path, PathBuf};
use std::time::Instant;
use tokenizers::Tokenizer;
use tracing::{debug, info};

/// Tracks chat session state across turns.
pub struct ChatSession {
    pub turn_count: usize,
    pub total_generated: usize,
    pub total_prefill_tokens: usize,
    pub total_prefill_secs: f64,
    pub total_decode_tokens: usize,
    pub total_decode_secs: f64,
}

impl ChatSession {
    pub fn new() -> Self {
        Self {
            turn_count: 0,
            total_generated: 0,
            total_prefill_tokens: 0,
            total_prefill_secs: 0.0,
            total_decode_tokens: 0,
            total_decode_secs: 0.0,
        }
    }
}

pub use herbert_core::decode_utils::{StopReason, detect_repetition_loop};

/// Result of one decode loop.
pub struct DecodeResult {
    pub generated_count: usize,
    pub stop_reason: StopReason,
    pub decode_secs: f64,
    /// Full generated text (accumulated from all tokens, with special tokens).
    pub generated_text: String,
    /// Raw token IDs generated (for tool call detection by token ID).
    pub all_tokens: Vec<u32>,
}

/// Configuration for the chat loop (bundles token IDs and limits).
pub struct ChatConfig<'a> {
    pub system_prompt: Option<&'a str>,
    pub max_tokens: usize,
    pub eos_token_id: u32,
    pub im_end_id: u32,
    pub skip_special_tokens: bool,
    pub tools_enabled: bool,
    /// Token ID for `<tool_call>` (None if not found in tokenizer).
    pub tool_call_open_id: Option<u32>,
    /// Token ID for `</tool_call>` (None if not found in tokenizer).
    pub tool_call_close_id: Option<u32>,
    /// Token ID for `[TOOL_CALLS]` (Mistral3, None if not found in tokenizer).
    pub mistral_tool_calls_id: Option<u32>,
    /// System prefix token length (for prefix cache save on fresh prefill).
    pub system_prefix_len: Option<usize>,
    /// Model directory path (needed for loading vision encoder on /image).
    pub model_path: Option<&'a Path>,
    /// Disable repetition loop detection.
    pub no_repeat_check: bool,
    /// Model family (needed for model-specific prompt formatting).
    pub model_family: herbert_core::config::ModelFamily,
    /// Embedding model directory path (needed for vision embedding with /embed and /dist).
    pub embed_model_path: Option<&'a Path>,
}

/// Lazily-loaded vision encoder state (loaded on first `/image` command).
struct VisionState {
    config: herbert_vision::config::VisionConfig,
    encoder: herbert_vision::encoder::VisionEncoder,
}

// ── Rustyline helper: slash-command completion + filename completion for /image ──

const SLASH_COMMANDS: &[&str] = &[
    "/arch", "/clear", "/config", "/dist", "/embed", "/exit", "/greedy", "/h", "/help",
    "/image", "/nothink", "/q", "/quit", "/stats", "/temp", "/think", "/tools",
    "/topk", "/topp",
];

struct ChatHelper {
    file_completer: FilenameCompleter,
}

impl ChatHelper {
    fn new() -> Self {
        Self {
            file_completer: FilenameCompleter::new(),
        }
    }
}

impl Completer for ChatHelper {
    type Candidate = Pair;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        ctx: &Context<'_>,
    ) -> rustyline::Result<(usize, Vec<Pair>)> {
        // After "/image ", delegate to file completer
        if line.starts_with("/image ") && pos >= 7 {
            return self.file_completer.complete(line, pos, ctx);
        }
        // Complete slash commands at beginning of line
        if line.starts_with('/') {
            let prefix = &line[..pos];
            let matches: Vec<Pair> = SLASH_COMMANDS
                .iter()
                .filter(|cmd| cmd.starts_with(prefix) && **cmd != prefix)
                .map(|cmd| Pair {
                    display: cmd.to_string(),
                    replacement: cmd.to_string(),
                })
                .collect();
            return Ok((0, matches));
        }
        Ok((0, Vec::new()))
    }
}

impl Hinter for ChatHelper {
    type Hint = String;
}

impl Highlighter for ChatHelper {}

impl Validator for ChatHelper {}

impl Helper for ChatHelper {}

/// Byte length of `s` excluding any trailing U+FFFD replacement characters.
/// The tokenizer emits U+FFFD for incomplete multi-byte sequences (e.g. half
/// an emoji). We hold off printing those — the next token will either complete
/// the character (replacing the FFFD) or confirm it's genuinely a FFFD.
fn safe_len(s: &str) -> usize {
    s.trim_end_matches('\u{FFFD}').len()
}

/// Public alias for safe_len (used by main.rs for one-shot decode).
pub fn safe_print_len(s: &str) -> usize {
    safe_len(s)
}

/// Run the streaming decode loop. Prints tokens as they are generated.
/// Accumulates all token IDs and decodes the full sequence each step,
/// printing only the new text delta — this ensures multi-token UTF-8
/// characters (e.g. emojis) are printed correctly without residual
/// replacement characters.
pub fn run_decode_loop(
    backend: &mut Box<dyn Backend>,
    kv: &mut KvHandle,
    first_token: u32,
    tokenizer: &Tokenizer,
    cfg: &ChatConfig,
    sampler: &mut Sampler,
    think_budget: &mut ThinkBudget,
) -> anyhow::Result<DecodeResult> {
    let decode_start = Instant::now();
    let mut all_tokens = vec![first_token];

    // Mute output when inside a <tool_call> block: the open/close tokens are
    // special (skipped by tokenizer), but the JSON body between them is normal
    // text that would otherwise stream to stdout.
    // Also mute for Mistral3 [TOOL_CALLS] token.
    let mut muted = cfg.tools_enabled
        && (cfg.tool_call_open_id == Some(first_token)
            || cfg.mistral_tool_calls_id == Some(first_token));

    // How many bytes of the decoded text we have actually printed.
    // We never print trailing U+FFFD — they may resolve to real chars.
    let full_text = tokenizer
        .decode(&all_tokens, cfg.skip_special_tokens)
        .map_err(|e| anyhow::anyhow!("Detokenization failed: {}", e))?;
    let mut printed_len = safe_len(&full_text);
    if printed_len > 0 && !muted {
        print!("{}", &full_text[..printed_len]);
        std::io::Write::flush(&mut std::io::stdout())?;
    }
    if muted {
        printed_len = 0;
    }
    let mut prev_full = full_text;

    if first_token == cfg.eos_token_id {
        let raw_text = tokenizer.decode(&all_tokens, false).unwrap_or_default();
        return Ok(DecodeResult {
            generated_count: 1,
            stop_reason: StopReason::Eos,
            decode_secs: decode_start.elapsed().as_secs_f64(),
            generated_text: raw_text,
            all_tokens,
        });
    }
    if first_token == cfg.im_end_id {
        let raw_text = tokenizer.decode(&all_tokens, false).unwrap_or_default();
        return Ok(DecodeResult {
            generated_count: 1,
            stop_reason: StopReason::ImEnd,
            decode_secs: decode_start.elapsed().as_secs_f64(),
            generated_text: raw_text,
            all_tokens,
        });
    }

    let mut current_token = first_token;
    let use_sampling = !sampler.is_greedy();
    let need_logits = use_sampling || think_budget.is_active();
    let decode_opts = RunOpts {
        return_logits: need_logits,
        profile: false,
        decode_tokens: Some(cfg.max_tokens),
        ignore_eos: false,
        system_prefix_len: None,
    };

    for _ in 1..cfg.max_tokens {
        let output = backend.decode_next(kv, current_token, decode_opts.clone())?;
        let next_token = if let Some(mut logits) = output.logits {
            think_budget.apply_to_logits(&mut logits);
            if use_sampling {
                sampler.sample(&logits)
            } else {
                crate::sampler::argmax(&logits)
            }
        } else {
            output.token
        };
        think_budget.track(next_token);
        all_tokens.push(next_token);

        // Mute when entering a tool call block
        if !muted && cfg.tools_enabled
            && (cfg.tool_call_open_id == Some(next_token)
                || cfg.mistral_tool_calls_id == Some(next_token))
        {
            muted = true;
        }

        let full_text = tokenizer
            .decode(&all_tokens, cfg.skip_special_tokens)
            .map_err(|e| anyhow::anyhow!("Detokenization failed: {}", e))?;

        if !muted {
            // Find how much new text to print.
            // Common case: prev_full[..printed_len] is still a prefix of full_text.
            // Rare case: tokenizer changed bytes before printed_len (shouldn't happen
            // since we never print FFFD, but handle it gracefully).
            let print_from = if full_text.is_char_boundary(printed_len)
                && full_text.as_bytes().get(..printed_len) == prev_full.as_bytes().get(..printed_len)
            {
                printed_len
            } else {
                // Walk back to a safe char boundary that still matches
                let common = prev_full
                    .bytes()
                    .zip(full_text.bytes())
                    .take_while(|(a, b)| a == b)
                    .count();
                let mut start = common.min(printed_len);
                while start > 0 && !full_text.is_char_boundary(start) {
                    start -= 1;
                }
                start
            };

            let safe = safe_len(&full_text);
            if safe > print_from {
                print!("{}", &full_text[print_from..safe]);
                std::io::Write::flush(&mut std::io::stdout())?;
            }
            printed_len = safe;
        }
        prev_full = full_text;

        if next_token == cfg.eos_token_id {
            if !muted && printed_len < prev_full.len() {
                print!("{}", &prev_full[printed_len..]);
                std::io::Write::flush(&mut std::io::stdout())?;
            }
            // Always decode without skipping special tokens so tool tags are preserved
            let raw_text = tokenizer.decode(&all_tokens, false).unwrap_or_default();
            return Ok(DecodeResult {
                generated_count: all_tokens.len(),
                stop_reason: StopReason::Eos,
                decode_secs: decode_start.elapsed().as_secs_f64(),
                generated_text: raw_text,
                all_tokens,
            });
        }
        if next_token == cfg.im_end_id {
            if !muted && printed_len < prev_full.len() {
                print!("{}", &prev_full[printed_len..]);
                std::io::Write::flush(&mut std::io::stdout())?;
            }
            // Always decode without skipping special tokens so tool tags are preserved
            let raw_text = tokenizer.decode(&all_tokens, false).unwrap_or_default();
            return Ok(DecodeResult {
                generated_count: all_tokens.len(),
                stop_reason: StopReason::ImEnd,
                decode_secs: decode_start.elapsed().as_secs_f64(),
                generated_text: raw_text,
                all_tokens,
            });
        }

        // Repetition loop detection
        if !cfg.no_repeat_check {
            if let Some(pat_len) = detect_repetition_loop(&all_tokens) {
                if !muted && printed_len < prev_full.len() {
                    print!("{}", &prev_full[printed_len..]);
                    std::io::Write::flush(&mut std::io::stdout())?;
                }
                eprintln!(
                    "\n[Repetition loop detected ({} tokens × 3), stopping]",
                    pat_len
                );
                let raw_text = tokenizer.decode(&all_tokens, false).unwrap_or_default();
                return Ok(DecodeResult {
                    generated_count: all_tokens.len(),
                    stop_reason: StopReason::RepetitionLoop,
                    decode_secs: decode_start.elapsed().as_secs_f64(),
                    generated_text: raw_text,
                    all_tokens,
                });
            }
        }

        current_token = next_token;
    }

    // Flush any buffered text at max_tokens
    if !muted && printed_len < prev_full.len() {
        print!("{}", &prev_full[printed_len..]);
        std::io::Write::flush(&mut std::io::stdout())?;
    }

    // Always decode without skipping special tokens so tool tags are preserved
    let raw_text = tokenizer.decode(&all_tokens, false).unwrap_or_default();
    Ok(DecodeResult {
        generated_count: all_tokens.len(),
        stop_reason: StopReason::MaxTokens,
        decode_secs: decode_start.elapsed().as_secs_f64(),
        generated_text: raw_text,
        all_tokens,
    })
}

/// Format the turn tokens for a new user message in an ongoing conversation.
fn format_turn_tokens(
    tokenizer: &Tokenizer,
    user_msg: &str,
    needs_im_end_prefix: bool,
    _model_family: herbert_core::config::ModelFamily,
) -> anyhow::Result<Vec<u32>> {
    // ChatML template for Qwen3 and Qwen3.5
    let mut turn_text = String::new();
    if needs_im_end_prefix {
        turn_text.push_str("<|im_end|>\n");
    }
    turn_text.push_str("<|im_start|>user\n");
    turn_text.push_str(user_msg);
    turn_text.push_str("<|im_end|>\n<|im_start|>assistant\n");

    let encoding = tokenizer
        .encode(turn_text.as_str(), false)
        .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
    Ok(encoding.get_ids().to_vec())
}

/// Format turn tokens with a Tool RAG system block injected before the user message.
///
/// For ChatML: `<|im_end|>\n<|im_start|>system\n{tools_prompt}<|im_end|>\n<|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n`
/// Compute a text embedding using the dedicated embedding backend.
fn compute_embedding(
    embed_backend: &mut Box<dyn Backend>,
    tokenizer: &Tokenizer,
    text: &str,
) -> anyhow::Result<Vec<f32>> {
    let prompt = format!(
        "<|im_start|>system\nRepresent the user's input.<|im_end|>\n\
         <|im_start|>user\n{}<|im_end|>\n\
         <|im_start|>assistant\n",
        text
    );
    let encoding = tokenizer
        .encode(prompt.as_str(), false)
        .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
    let tokens = encoding.get_ids().to_vec();
    let num_tokens = tokens.len();
    let t0 = Instant::now();
    let embedding = embed_backend.embed(&tokens)
        .map_err(|e| anyhow::anyhow!("Embedding failed: {}", e))?;
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "[Embedding: {} tokens, {} dims, {:.1}ms]",
        num_tokens, embedding.len(), elapsed_ms,
    );
    Ok(embedding)
}

/// Check if a string looks like an image file path (by extension AND existence).
fn is_image_path(s: &str) -> bool {
    let path = Path::new(s);
    if !path.exists() {
        return false;
    }
    match path.extension().and_then(|e| e.to_str()) {
        Some(ext) => matches!(ext.to_ascii_lowercase().as_str(), "jpg" | "jpeg" | "png" | "webp" | "bmp"),
        None => false,
    }
}

/// Compute an embedding for an image using the vision-language embedding model.
fn compute_image_embedding(
    embed_backend: &mut Box<dyn Backend>,
    tokenizer: &Tokenizer,
    embed_model_path: &Path,
    image_path: &Path,
) -> anyhow::Result<Vec<f32>> {
    // 1. Load image
    let rgb_bytes = crate::load_image_rgb(image_path)?;
    let (img_h, img_w) = crate::image_dimensions(image_path)?;

    // 2. Vision config
    let vision_config = herbert_vision::config::VisionConfig::from_file(
        &embed_model_path.join("config.json"),
    )?;

    // 3. Preprocess
    let (patches, grid_t, grid_h, grid_w) = herbert_vision::image_process::preprocess_rgb(
        &rgb_bytes, img_h, img_w, &vision_config, 65536, 16777216,
    )?;

    // 4. Vision encode (GPU if available, else CPU fallback)
    let t0_enc = Instant::now();
    let vision_embed = if let Some(gpu_result) = embed_backend.encode_vision(&patches, grid_t, grid_h, grid_w) {
        let embed = gpu_result.map_err(|e| anyhow::anyhow!("GPU vision encode failed: {}", e))?;
        eprintln!(
            "[Vision encode (GPU): {} tokens, {:.1}ms]",
            embed.num_tokens, t0_enc.elapsed().as_secs_f64() * 1000.0,
        );
        embed
    } else {
        let vision_encoder = herbert_vision::loader::load_vision_encoder_with_progress(
            embed_model_path, &vision_config,
        )?;
        let merged_h = grid_h / vision_config.spatial_merge_size;
        let merged_w = grid_w / vision_config.spatial_merge_size;
        let vision_output = vision_encoder.forward_with_label(
            &patches, grid_t, grid_h, grid_w,
            &image_path.display().to_string(),
        )?;
        eprintln!(
            "[Vision encode (CPU): {} tokens, {:.1}ms]",
            vision_output.num_tokens, t0_enc.elapsed().as_secs_f64() * 1000.0,
        );
        VisionEmbedding {
            hidden_states: vision_output.hidden_states,
            num_tokens: vision_output.num_tokens,
            grid_h: merged_h,
            grid_w: merged_w,
            deepstack_features: vision_output.deepstack_features,
        }
    };

    // 5. Build prompt with image placeholder
    let prompt = "<|im_start|>system\nRepresent the user's input.<|im_end|>\n\
                  <|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|><|im_end|>\n\
                  <|im_start|>assistant\n";

    // 6. Tokenize and expand image_pad tokens
    let encoding = tokenizer
        .encode(prompt, false)
        .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
    let base_tokens = encoding.get_ids().to_vec();
    let image_pad_id = 151655u32; // <|image_pad|>

    let image_pos = base_tokens
        .iter()
        .position(|&t| t == image_pad_id)
        .ok_or_else(|| anyhow::anyhow!("No image_pad token found in prompt"))?;

    let num_vis = vision_embed.num_tokens;
    let mut expanded_tokens = Vec::with_capacity(base_tokens.len() - 1 + num_vis);
    expanded_tokens.extend_from_slice(&base_tokens[..image_pos]);
    expanded_tokens.resize(expanded_tokens.len() + num_vis, image_pad_id);
    expanded_tokens.extend_from_slice(&base_tokens[image_pos + 1..]);

    // 7. Call embed_vl
    let t0 = Instant::now();
    let embedding = embed_backend
        .embed_vl(&expanded_tokens, &[vision_embed], &[image_pos])
        .map_err(|e| anyhow::anyhow!("VL embedding failed: {}", e))?;
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "[VL Embedding: {} tokens ({} vision), {} dims, {:.1}ms]",
        expanded_tokens.len(), num_vis, embedding.len(), elapsed_ms,
    );
    Ok(embedding)
}

/// Parse `"phrase A" "phrase B" [method]` from an input string.
/// Returns `(phrase_a, phrase_b, optional_method)` or `None` on failure.
fn parse_two_quoted(input: &str) -> Option<(String, String, Option<String>)> {
    let rest = input.strip_prefix('"')?;
    let end = rest.find('"')?;
    let a = rest[..end].to_string();
    let rest = &rest[end + 1..];
    let rest = rest.trim_start();
    let rest = rest.strip_prefix('"')?;
    let end = rest.find('"')?;
    let b = rest[..end].to_string();
    let rest = rest[end + 1..].trim();
    let method = if rest.is_empty() { None } else { Some(rest.to_string()) };
    Some((a, b, method))
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na * nb) }
}

fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>().sqrt()
}

fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn print_help() {
    eprintln!("--- Commands ---");
    eprintln!("  /help, /h       Show this help");
    eprintln!("  /quit, /q       Quit");
    eprintln!("  /exit           Quit");
    eprintln!("  /clear          Clear conversation (re-prefill system prompt)");
    eprintln!("  /stats          Show session statistics");
    eprintln!("  /config         Show sampling config");
    eprintln!("  /greedy         Set greedy decoding (temp=0)");
    eprintln!("  /temp <val>     Set temperature (0 = greedy)");
    eprintln!("  /topk <val>     Set top-k (0 = disabled)");
    eprintln!("  /topp <val>     Set top-p (1 = disabled)");
    eprintln!("  /think <N>      Set think budget (max tokens in <think>, 0 = unlimited)");
    eprintln!("  /nothink        Disable thinking (prefill empty <think> block)");
    eprintln!("  /tools          Show available tools and status");
    eprintln!("  /image <path>   Load image for next message (VL models)");
    eprintln!("  /image          Show pending image");
    eprintln!("  /image clear    Clear pending image");
    eprintln!("  /embed \"text\" or path.jpg  Generate embedding (requires --embed-model)");
    eprintln!("  /dist \"A\" \"B\" [method]  Compare embeddings, text or images (requires --embed-model)");
    eprintln!("  /arch           Show hardware info (CPU, SIMD, cache, RAM, bandwidth)");
    eprintln!("  !<command>        Run shell command (e.g. !ls, !pwd)");
    eprintln!();
    eprintln!("  Use \\ at end of line to continue input on next line.");
}

fn print_config(sampler: &Sampler, think_budget: &ThinkBudget, nothink: bool) {
    let mode = if sampler.is_greedy() { "greedy" } else { "sampling" };
    eprintln!("--- Config ({}) ---", mode);
    eprintln!("  temperature:  {}", sampler.temperature());
    eprintln!("  top_k:        {}", sampler.top_k());
    eprintln!("  top_p:        {}", sampler.top_p());
    if nothink {
        eprintln!("  thinking:     disabled (template prefix + logits backup)");
    } else {
        let tb = think_budget.max_tokens();
        if tb == 0 {
            eprintln!("  think_budget: unlimited");
        } else {
            eprintln!("  think_budget: {} tokens", tb);
        }
    }
}

fn tps(tokens: usize, secs: f64) -> f64 {
    if secs > 0.0 { tokens as f64 / secs } else { 0.0 }
}

/// Run the interactive chat loop.
///
/// `kv` and `session` represent the state after the first turn has already been decoded.
/// `last_stop` is the stop reason from the first turn's decode.
#[allow(clippy::too_many_arguments)]
pub fn run_chat(
    backend: &mut Box<dyn Backend>,
    mut kv: Option<KvHandle>,
    tokenizer: &Tokenizer,
    mut session: ChatSession,
    mut last_stop: StopReason,
    cfg: &ChatConfig,
    sampler: &mut Sampler,
    think_budget: &mut ThinkBudget,
    initial_nothink: bool,
    embed_backend: &mut Option<Box<dyn Backend>>,
    embed_tokenizer: Option<&Tokenizer>,
) -> anyhow::Result<()> {
    let mut nothink = initial_nothink;
    let mut vision_state: Option<VisionState> = None;
    let mut pending_image: Option<PathBuf> = None;

    // Rustyline editor with history + slash-command completion
    let history_path = std::env::var("HOME")
        .ok()
        .map(|h| {
            let dir = format!("{}/.herbert", h);
            let _ = std::fs::create_dir_all(&dir);
            format!("{}/history.txt", dir)
        });
    let mut rl = Editor::<ChatHelper, rustyline::history::DefaultHistory>::new()?;
    rl.set_helper(Some(ChatHelper::new()));
    if let Some(ref p) = history_path {
        let _ = rl.load_history(p);
    }

    loop {
        let input = match rl.readline("\n>>> ") {
            Ok(line) => {
                let _ = rl.add_history_entry(&line);
                // Handle backslash continuation
                let trimmed_end = line.trim_end_matches('\n').trim_end_matches('\r');
                if let Some(stripped) = trimmed_end.strip_suffix('\\') {
                    let mut full = stripped.to_string();
                    full.push('\n');
                    loop {
                        match rl.readline("... ") {
                            Ok(cont) => {
                                let t = cont.trim_end_matches('\n').trim_end_matches('\r');
                                if let Some(s) = t.strip_suffix('\\') {
                                    full.push_str(s);
                                    full.push('\n');
                                } else {
                                    full.push_str(t);
                                    break;
                                }
                            }
                            Err(ReadlineError::Eof) => break,
                            Err(ReadlineError::Interrupted) => break,
                            Err(e) => return Err(e.into()),
                        }
                    }
                    full
                } else {
                    line
                }
            }
            Err(ReadlineError::Eof) => break,
            Err(ReadlineError::Interrupted) => continue,
            Err(e) => return Err(e.into()),
        };

        let trimmed = input.trim();
        if trimmed.is_empty() {
            continue;
        }

        // Shell escape: !command
        if let Some(shell_cmd) = trimmed.strip_prefix('!') {
            let shell_cmd = shell_cmd.trim();
            if shell_cmd.is_empty() {
                eprintln!("Usage: !<command>  (e.g. !ls, !pwd)");
            } else {
                let _ = std::process::Command::new("sh")
                    .arg("-c")
                    .arg(shell_cmd)
                    .status();
            }
            continue;
        }

        // Slash commands
        if trimmed.starts_with('/') {
            let (cmd, arg) = match trimmed.find(' ') {
                Some(pos) => (&trimmed[..pos], Some(trimmed[pos+1..].trim())),
                None => (trimmed, None),
            };
            match cmd {
                "/arch" => {
                    crate::arch::print_arch();
                    continue;
                }
                "/help" | "/h" => {
                    print_help();
                    continue;
                }
                "/quit" | "/q" | "/exit" => {
                    break;
                }
                "/clear" => {
                    kv = None;
                    session = ChatSession::new();
                    last_stop = StopReason::ImEnd;
                    pending_image = None;
                    eprintln!("[Conversation cleared]");
                    continue;
                }
                "/stats" => {
                    eprintln!("--- Chat Stats ---");
                    eprintln!("Turns:           {}", session.turn_count);
                    if let Some(ref kv_handle) = kv {
                        let seq_len = crate::kv_seq_len(kv_handle);
                        eprintln!("KV cache tokens: {}", seq_len);
                    } else {
                        eprintln!("KV cache tokens: 0 (cleared)");
                    }
                    eprintln!(
                        "Prefill:         {} tokens in {:.3}s ({:.1} tok/s)",
                        session.total_prefill_tokens,
                        session.total_prefill_secs,
                        tps(session.total_prefill_tokens, session.total_prefill_secs),
                    );
                    eprintln!(
                        "Decode:          {} tokens in {:.3}s ({:.1} tok/s)",
                        session.total_decode_tokens,
                        session.total_decode_secs,
                        tps(session.total_decode_tokens, session.total_decode_secs),
                    );
                    eprintln!(
                        "Total generated: {}",
                        session.total_generated,
                    );
                    continue;
                }
                "/config" => {
                    print_config(sampler, think_budget, nothink);
                    continue;
                }
                "/greedy" => {
                    sampler.set_temperature(0.0);
                    sampler.set_top_k(0);
                    sampler.set_top_p(1.0);
                    print_config(sampler, think_budget, nothink);
                    continue;
                }
                "/temp" => {
                    if let Some(val) = arg.and_then(|s| s.parse::<f32>().ok()) {
                        if val < 0.0 {
                            eprintln!("Temperature must be >= 0");
                        } else {
                            sampler.set_temperature(val);
                            print_config(sampler, think_budget, nothink);
                        }
                    } else {
                        eprintln!("Usage: /temp <value>  (e.g. /temp 0.6)");
                    }
                    continue;
                }
                "/topk" => {
                    if let Some(val) = arg.and_then(|s| s.parse::<usize>().ok()) {
                        sampler.set_top_k(val);
                        print_config(sampler, think_budget, nothink);
                    } else {
                        eprintln!("Usage: /topk <value>  (e.g. /topk 50)");
                    }
                    continue;
                }
                "/topp" => {
                    if let Some(val) = arg.and_then(|s| s.parse::<f32>().ok()) {
                        if !(0.0..=1.0).contains(&val) {
                            eprintln!("Top-p must be between 0 and 1");
                        } else {
                            sampler.set_top_p(val);
                            print_config(sampler, think_budget, nothink);
                        }
                    } else {
                        eprintln!("Usage: /topp <value>  (e.g. /topp 0.9)");
                    }
                    continue;
                }
                "/think" => {
                    if let Some(val) = arg.and_then(|s| s.parse::<usize>().ok()) {
                        nothink = false;
                        think_budget.set_max_tokens(val);
                        print_config(sampler, think_budget, nothink);
                    } else {
                        eprintln!("Usage: /think <N>  (e.g. /think 200, /think 0 for unlimited)");
                    }
                    continue;
                }
                "/nothink" => {
                    nothink = true;
                    // Template approach: prefill empty <think></think> block
                    // Logits backup: budget=1 forces \n then </think> if model ignores the prefix
                    think_budget.set_max_tokens(1);
                    print_config(sampler, think_budget, nothink);
                    continue;
                }
                "/tools" => {
                    tools::print_tools_status(cfg.tools_enabled);
                    continue;
                }
                "/embed" => {
                    let text = arg.unwrap_or("").trim();
                    // Strip surrounding quotes if present
                    let text = text.strip_prefix('"')
                        .and_then(|s| s.strip_suffix('"'))
                        .unwrap_or(text);
                    if text.is_empty() {
                        eprintln!("Usage: /embed \"text\" or /embed path.jpg");
                        continue;
                    }
                    match embed_backend {
                        Some(ref mut eb) => {
                            let et = embed_tokenizer.unwrap();
                            let result = if is_image_path(text) {
                                match cfg.embed_model_path {
                                    Some(emp) => compute_image_embedding(eb, et, emp, Path::new(text)),
                                    None => {
                                        eprintln!("[Error: embed_model_path not available for image embedding]");
                                        continue;
                                    }
                                }
                            } else {
                                compute_embedding(eb, et, text)
                            };
                            match result {
                                Ok(embedding) => {
                                    let parts: Vec<String> = embedding.iter()
                                        .map(|v| format!("{:.6}", v))
                                        .collect();
                                    println!("[{}]", parts.join(", "));
                                }
                                Err(e) => {
                                    eprintln!("[Embedding error: {}]", e);
                                }
                            }
                        }
                        None => {
                            eprintln!("[No embedding model loaded. Use --embed-model to specify one.]");
                        }
                    }
                    continue;
                }
                "/dist" => {
                    let arg = arg.unwrap_or("").trim();
                    match parse_two_quoted(arg) {
                        Some((phrase_a, phrase_b, method_opt)) => {
                            let method = method_opt.as_deref().unwrap_or("cosine");
                            if !matches!(method, "cosine" | "euclidean" | "dot") {
                                eprintln!("Unknown method '{}'. Use: cosine, euclidean, or dot.", method);
                                continue;
                            }
                            match embed_backend {
                                Some(ref mut eb) => {
                                    let et = embed_tokenizer.unwrap();
                                    let embed_phrase = |backend: &mut Box<dyn Backend>, phrase: &str| -> anyhow::Result<Vec<f32>> {
                                        if is_image_path(phrase) {
                                            match cfg.embed_model_path {
                                                Some(emp) => compute_image_embedding(backend, et, emp, Path::new(phrase)),
                                                None => Err(anyhow::anyhow!("embed_model_path not available for image embedding")),
                                            }
                                        } else {
                                            compute_embedding(backend, et, phrase)
                                        }
                                    };
                                    let emb_a = embed_phrase(eb, &phrase_a);
                                    let emb_b = embed_phrase(eb, &phrase_b);
                                    match (emb_a, emb_b) {
                                        (Ok(a), Ok(b)) => {
                                            let val = match method {
                                                "euclidean" => euclidean_distance(&a, &b),
                                                "dot" => dot_product(&a, &b),
                                                _ => cosine_similarity(&a, &b),
                                            };
                                            println!("[Distance ({}): {:.4}]", method, val);
                                        }
                                        (Err(e), _) | (_, Err(e)) => {
                                            eprintln!("[Embedding error: {}]", e);
                                        }
                                    }
                                }
                                None => {
                                    eprintln!("[No embedding model loaded. Use --embed-model to specify one.]");
                                }
                            }
                        }
                        None => {
                            eprintln!("Usage: /dist \"phrase A\" \"phrase B\" [cosine|euclidean|dot]");
                        }
                    }
                    continue;
                }
                "/image" => {
                    match arg {
                        None => {
                            // Show pending image status
                            if let Some(ref path) = pending_image {
                                eprintln!("[Pending image: {}]", path.display());
                            } else {
                                eprintln!("[No pending image]");
                            }
                        }
                        Some("clear") => {
                            pending_image = None;
                            eprintln!("[Image cleared]");
                        }
                        Some(path_str) => {
                            if cfg.model_path.is_none() {
                                eprintln!("[Error: model path not available for vision]");
                                continue;
                            }
                            let path = PathBuf::from(path_str);
                            if !path.exists() {
                                eprintln!("[Error: file not found: {}]", path.display());
                                continue;
                            }
                            match crate::image_dimensions(&path) {
                                Ok((h, w)) => {
                                    let file_size = std::fs::metadata(&path)
                                        .map(|m| m.len())
                                        .unwrap_or(0);
                                    let name = path.file_name()
                                        .unwrap_or_default()
                                        .to_string_lossy();
                                    eprintln!(
                                        "[Image loaded: {} ({}x{}, {:.1} KB)]",
                                        name, w, h, file_size as f64 / 1024.0,
                                    );
                                    pending_image = Some(path);
                                }
                                Err(e) => {
                                    eprintln!("[Error reading image: {}]", e);
                                }
                            }
                        }
                    }
                    continue;
                }
                _ => {
                    eprintln!("Unknown command: {}. Type /help for available commands.", cmd);
                    continue;
                }
            }
        }

        // Prefill: vision (pending image) or text-only
        let use_sampling = !sampler.is_greedy();
        let need_logits = use_sampling || think_budget.is_active();
        let prefill_start = Instant::now();
        let (prefill_output, prefill_token_count) = if let Some(image_path) = pending_image.take() {
            // ── Vision prefill (fresh, resets KV) ──
            let model_path = cfg.model_path.unwrap(); // checked when /image was set

            // Lazy-load Qwen3-VL vision encoder
            if vision_state.is_none() {
                let t0 = Instant::now();
                eprintln!("[Loading vision encoder...]");
                let vision_config = herbert_vision::config::VisionConfig::from_file(
                    &model_path.join("config.json"),
                )?;
                let vision_encoder =
                    herbert_vision::loader::load_vision_encoder_with_progress(model_path, &vision_config)?;
                let elapsed = t0.elapsed();
                info!(elapsed_ms = elapsed.as_millis() as u64, "Vision encoder loaded");
                eprintln!("[Vision encoder loaded in {:.1}s]", elapsed.as_secs_f64());
                vision_state = Some(VisionState {
                    config: vision_config,
                    encoder: vision_encoder,
                });
            }
            let vs = vision_state.as_ref().unwrap();

            // Load and preprocess image
            let rgb_bytes = crate::load_image_rgb(&image_path)?;
            let (img_h, img_w) = crate::image_dimensions(&image_path)?;

            let image_label = format!(
                "{} ({}x{})",
                image_path.file_name().unwrap_or_default().to_string_lossy(),
                img_w, img_h,
            );

            // Encode image with Qwen3-VL encoder
            let (vision_embed, image_pad_id) = {
                let vision_config = &vs.config;
                let vision_encoder = &vs.encoder;

                let (patches, grid_t, grid_h, grid_w) = herbert_vision::image_process::preprocess_rgb(
                    &rgb_bytes, img_h, img_w, vision_config, 65536, 16777216,
                )?;
                let merged_h = grid_h / vision_config.spatial_merge_size;
                let merged_w = grid_w / vision_config.spatial_merge_size;

                let t0 = Instant::now();
                let vision_output = vision_encoder.forward_with_label(
                    &patches, grid_t, grid_h, grid_w, &image_label,
                )?;
                let encode_elapsed = t0.elapsed();
                info!(
                    num_tokens = vision_output.num_tokens,
                    elapsed_ms = encode_elapsed.as_millis() as u64,
                    "Vision encoder done"
                );

                let embed = VisionEmbedding {
                    hidden_states: vision_output.hidden_states,
                    num_tokens: vision_output.num_tokens,
                    grid_h: merged_h,
                    grid_w: merged_w,
                    deepstack_features: vision_output.deepstack_features,
                };
                (embed, 151655u32) // <|image_pad|>
            };

            // Build prompt with image placeholder, tokenize, expand
            let full_prompt = crate::build_prompt(trimmed, cfg.system_prompt, 1, cfg.model_family);
            let full_tokens = tokenizer
                .encode(full_prompt.as_str(), false)
                .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?
                .get_ids()
                .to_vec();

            let image_pos = full_tokens
                .iter()
                .position(|&t| t == image_pad_id)
                .ok_or_else(|| anyhow::anyhow!("No image placeholder token (id={}) found in prompt", image_pad_id))?;

            let num_vis = vision_embed.num_tokens;
            let mut expanded_tokens = Vec::with_capacity(full_tokens.len() - 1 + num_vis);
            expanded_tokens.extend_from_slice(&full_tokens[..image_pos]);
            expanded_tokens.resize(expanded_tokens.len() + num_vis, image_pad_id);
            expanded_tokens.extend_from_slice(&full_tokens[image_pos + 1..]);
            let token_count = expanded_tokens.len();

            if !cfg.skip_special_tokens {
                eprintln!("--- VL Prompt ({} tokens, {} vision) ---", token_count, num_vis);
            }

            let opts = RunOpts {
                return_logits: need_logits,
                profile: false,
                decode_tokens: Some(cfg.max_tokens),
                ignore_eos: false,
                system_prefix_len: None,
            };
            let (new_kv, output) = backend.prefill_vl(
                &expanded_tokens,
                &[vision_embed],
                &[image_pos],
                opts,
            )?;
            kv = Some(new_kv);
            (output, token_count)
        } else {
            // ── Text-only prefill ──
            let needs_im_end_prefix = last_stop == StopReason::MaxTokens
                || last_stop == StopReason::RepetitionLoop;
            let turn_tokens = format_turn_tokens(tokenizer, trimmed, needs_im_end_prefix, cfg.model_family)?;
            let turn_token_count = turn_tokens.len();

            if let Some(ref mut kv_handle) = kv {
                // ── Continue prefill (subsequent turn) ──
                if !cfg.skip_special_tokens {
                    let turn_text = tokenizer
                        .decode(&turn_tokens, false)
                        .unwrap_or_default();
                    eprintln!("--- Turn ({} tokens) ---", turn_tokens.len());
                    eprintln!("{}", turn_text);
                    eprintln!("--- End turn ---");
                }
                let opts = RunOpts {
                    return_logits: need_logits,
                    profile: false,
                    decode_tokens: Some(cfg.max_tokens),
                    ignore_eos: false,
                    system_prefix_len: None,
                };
                let output = backend.continue_prefill(kv_handle, &turn_tokens, opts)?;
                (output, turn_token_count)
            } else {
                // ── First turn ──
                let full_prompt = crate::build_prompt(trimmed, cfg.system_prompt, 0, cfg.model_family);
                let full_tokens = tokenizer
                    .encode(full_prompt.as_str(), false)
                    .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?
                    .get_ids()
                    .to_vec();
                if !cfg.skip_special_tokens {
                    let prompt_text = tokenizer
                        .decode(&full_tokens, false)
                        .unwrap_or_default();
                    eprintln!("--- Prompt ({} tokens) ---", full_tokens.len());
                    eprintln!("{}", prompt_text);
                    eprintln!("--- End prompt ---");
                }
                let actual_count = full_tokens.len();
                let opts = RunOpts {
                    return_logits: need_logits,
                    profile: false,
                    decode_tokens: Some(cfg.max_tokens),
                    ignore_eos: false,
                    system_prefix_len: cfg.system_prefix_len,
                };
                let (new_kv, output) = backend.prefill(&full_tokens, opts)?;
                kv = Some(new_kv);
                session.total_prefill_tokens += actual_count.saturating_sub(turn_token_count);
                (output, turn_token_count)
            }
        };
        let prefill_secs = prefill_start.elapsed().as_secs_f64();
        session.total_prefill_tokens += prefill_token_count;
        session.total_prefill_secs += prefill_secs;

        // Apply think budget to logits, then sample
        think_budget.reset();
        let target_logits = prefill_output.logits;
        let first_token = if let Some(ref logits) = target_logits {
            let mut logits_buf = logits.clone();
            think_budget.apply_to_logits(&mut logits_buf);
            if use_sampling {
                sampler.sample(&logits_buf)
            } else {
                crate::sampler::argmax(&logits_buf)
            }
        } else {
            prefill_output.first_token
        };
        think_budget.track(first_token);

        // Decode loop
        let kv_handle = kv.as_mut().unwrap();
        let result = run_decode_loop(
            backend,
            kv_handle,
            first_token,
            tokenizer,
            cfg,
            sampler,
            think_budget,
        )?;

        // Ensure a newline after streamed decode output so rustyline's
        // next prompt doesn't overwrite the last line of model output.
        println!();

        last_stop = result.stop_reason;
        session.turn_count += 1;
        session.total_generated += result.generated_count;
        session.total_decode_tokens += result.generated_count;
        session.total_decode_secs += result.decode_secs;

        debug!(
            stop = ?last_stop,
            tools_enabled = cfg.tools_enabled,
            generated_text_len = result.generated_text.len(),
            "decode done"
        );

        // Tool call loop: if tools are enabled and model stopped naturally, check for tool calls.
        // Check both Eos and ImEnd — some models use eos_token_id == im_end_id,
        // so the Eos check fires first in run_decode_loop.
        if cfg.tools_enabled && (last_stop == StopReason::ImEnd || last_stop == StopReason::Eos) {
            let mut generated_text = result.generated_text;
            let mut current_tokens = result.all_tokens;
            let mut tool_iterations = 0;
            while tool_iterations < tools::MAX_TOOL_ITERATIONS {
                // Primary: detect by special token IDs (robust against tokenizer quirks)
                // Detection chain: Mistral [TOOL_CALLS] → token-based → text-based → raw JSON
                let calls = {
                    let mut found = Vec::new();

                    // 1. Token-based detection (most robust for Qwen3)
                    if found.is_empty() {
                        if let (Some(open_id), Some(close_id)) =
                            (cfg.tool_call_open_id, cfg.tool_call_close_id)
                        {
                            let has_open = current_tokens.contains(&open_id);
                            let has_close = current_tokens.contains(&close_id);
                            debug!(
                                num_tokens = current_tokens.len(),
                                has_open_token = has_open,
                                has_close_token = has_close,
                                open_id,
                                close_id,
                                "checking for tool calls (token-based)"
                            );
                            if has_open && has_close {
                                found = tools::extract_tool_calls_from_tokens(
                                    &current_tokens,
                                    open_id,
                                    close_id,
                                    tokenizer,
                                );
                            }
                        }
                    }

                    // 2. Text-based fallback (<tool_call>...</tool_call> in text)
                    if found.is_empty() {
                        debug!(
                            text_len = generated_text.len(),
                            "checking for tool calls (text-based fallback)"
                        );
                        found = tools::parse_tool_calls(&generated_text);
                    }

                    // 3. Raw JSON fallback (Qwen2.5 models that omit the wrapper)
                    if found.is_empty() {
                        debug!(
                            text_len = generated_text.len(),
                            "checking for tool calls (raw JSON fallback)"
                        );
                        found = tools::parse_raw_tool_call(&generated_text);
                    }

                    found
                };

                if calls.is_empty() {
                    debug!("no tool calls found");
                    break;
                }
                debug!(count = calls.len(), "found tool calls");

                // Execute each tool call and collect results
                let mut results: Vec<(String, String)> = Vec::new();
                for call in &calls {
                    let args_str = call.args_display();
                    if args_str == "{}" {
                        eprintln!("[calling {}]", call.name);
                    } else {
                        eprintln!("[calling {}({})]", call.name, args_str);
                    }
                    debug!(tool_name = &*call.name, tool_args = &*args_str, "tool call");
                    let result = tools::execute_tool(call);
                    let display_result = if result.len() > 200 {
                        format!("{}...", &result[..200])
                    } else {
                        result.clone()
                    };
                    debug!(tool_name = &*call.name, tool_result = &*display_result, "tool result");
                    results.push((call.name.clone(), result));
                }

                // Format tool response and tokenize
                let response_text = tools::format_tool_response(&results);
                let response_encoding = tokenizer
                    .encode(response_text.as_str(), false)
                    .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
                let response_tokens = response_encoding.get_ids().to_vec();
                let response_token_count = response_tokens.len();

                if !cfg.skip_special_tokens {
                    let decoded = tokenizer
                        .decode(&response_tokens, false)
                        .unwrap_or_default();
                    eprintln!("--- Tool response ({} tokens) ---", response_tokens.len());
                    eprintln!("{}", decoded);
                    eprintln!("--- End tool response ---");
                }

                // Continue prefill with tool response
                let kv_handle = kv.as_mut().unwrap();
                let use_sampling = !sampler.is_greedy();
                let need_logits = use_sampling || think_budget.is_active();
                let prefill_start = Instant::now();
                let opts = RunOpts {
                    return_logits: need_logits,
                    profile: false,
                    decode_tokens: Some(cfg.max_tokens),
                    ignore_eos: false,
                    system_prefix_len: None,
                };
                let prefill_output = backend.continue_prefill(kv_handle, &response_tokens, opts)?;
                let prefill_secs = prefill_start.elapsed().as_secs_f64();
                session.total_prefill_tokens += response_token_count;
                session.total_prefill_secs += prefill_secs;

                // Sample first token of model's response
                think_budget.reset();
                let first_token = if let Some(mut logits) = prefill_output.logits {
                    think_budget.apply_to_logits(&mut logits);
                    if use_sampling {
                        sampler.sample(&logits)
                    } else {
                        crate::sampler::argmax(&logits)
                    }
                } else {
                    prefill_output.first_token
                };
                think_budget.track(first_token);

                // Decode again
                let kv_handle = kv.as_mut().unwrap();
                let decode_result = run_decode_loop(
                    backend,
                    kv_handle,
                    first_token,
                    tokenizer,
                    cfg,
                    sampler,
                    think_budget,
                )?;

                last_stop = decode_result.stop_reason;
                session.total_generated += decode_result.generated_count;
                session.total_decode_tokens += decode_result.generated_count;
                session.total_decode_secs += decode_result.decode_secs;
                generated_text = decode_result.generated_text;
                current_tokens = decode_result.all_tokens;
                tool_iterations += 1;

                // Only continue the loop if we stopped naturally (might be another tool call)
                if last_stop != StopReason::ImEnd && last_stop != StopReason::Eos {
                    break;
                }
            }
        }
    }

    if let Some(ref p) = history_path {
        let _ = rl.save_history(p);
    }
    Ok(())
}
