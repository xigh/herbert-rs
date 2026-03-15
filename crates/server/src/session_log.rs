//! Incremental YAML session logging for API requests.
//!
//! Writes the session file at each key moment:
//! - Request received (input prompt, status=prefilling)
//! - After prefill (status=decoding)
//! - Every N tokens during decode (partial output)
//! - Final completion (stop_reason, total timing)

use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tracing::error;

/// How often to flush partial output during decode.
pub const DECODE_FLUSH_INTERVAL: usize = 128;

pub struct SessionLog {
    path: String,
    iso_ts: String,
    start_time: Instant,
    // Request
    pub msg_id: String,
    pub model: String,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub system_text: Option<String>,
    pub messages: Vec<(String, String)>,
    pub input_tokens: usize,
    // Response (updated incrementally)
    pub status: String,
    pub output_tokens: usize,
    pub stop_reason: String,
    pub thinking_text: String,
    pub generated_text: String,
    pub prefill_us: u64,
    pub decode_us: u64,
}

impl SessionLog {
    /// Create a new session log and write the initial file (status=prefilling).
    pub fn new(
        msg_id: String,
        model: String,
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        system_text: Option<String>,
        messages: Vec<(String, String)>,
        input_tokens: usize,
    ) -> Option<Self> {
        let (iso_ts, date, time) = utc_now();
        let home = std::env::var("HOME").ok()?;
        let dir = format!("{}/.herbert/sessions", home);
        if let Err(e) = std::fs::create_dir_all(&dir) {
            error!("Failed to create session log dir: {}", e);
            return None;
        }
        let path = format!("{}/{}_{}-{}.yaml", dir, date, time, msg_id);

        let log = Self {
            path,
            iso_ts,
            start_time: Instant::now(),
            msg_id,
            model,
            max_tokens,
            temperature,
            top_k,
            top_p,
            system_text,
            messages,
            input_tokens,
            status: "prefilling".to_string(),
            output_tokens: 0,
            stop_reason: String::new(),
            thinking_text: String::new(),
            generated_text: String::new(),
            prefill_us: 0,
            decode_us: 0,
        };
        log.flush();
        Some(log)
    }

    /// Rewrite the session file with current state.
    pub fn flush(&self) {
        let duration_ms = self.start_time.elapsed().as_millis();

        let mut doc = String::with_capacity(4096);
        doc.push_str("---\n");
        doc.push_str(&format!("id: {}\n", self.msg_id));
        doc.push_str(&format!("timestamp: {}\n", yaml_escape_inline(&self.iso_ts)));
        doc.push_str(&format!("model: {}\n", self.model));
        doc.push_str(&format!("status: {}\n", self.status));
        doc.push_str(&format!("duration_ms: {}\n", duration_ms));
        if self.prefill_us > 0 {
            doc.push_str(&format!("prefill_ms: {}\n", self.prefill_us / 1000));
        }
        if self.decode_us > 0 {
            doc.push_str(&format!("decode_ms: {}\n", self.decode_us / 1000));
        }

        doc.push_str("request:\n");
        doc.push_str(&format!("  max_tokens: {}\n", self.max_tokens));
        doc.push_str(&format!("  temperature: {}\n", self.temperature));
        doc.push_str(&format!("  top_k: {}\n", self.top_k));
        doc.push_str(&format!("  top_p: {}\n", self.top_p));
        doc.push_str(&format!("  input_tokens: {}\n", self.input_tokens));

        match &self.system_text {
            Some(s) => {
                if s.contains('\n') {
                    doc.push_str(&format!("  system: {}", yaml_block_scalar(s, 4)));
                } else {
                    doc.push_str(&format!("  system: {}\n", yaml_escape_inline(s)));
                }
            }
            None => doc.push_str("  system: null\n"),
        }

        doc.push_str("  messages:\n");
        for (role, content) in &self.messages {
            doc.push_str(&format!("    - role: {}\n", role));
            if content.contains('\n') {
                doc.push_str(&format!("      content: {}", yaml_block_scalar(content, 8)));
            } else {
                doc.push_str(&format!("      content: {}\n", yaml_escape_inline(content)));
            }
        }

        doc.push_str("response:\n");
        doc.push_str(&format!("  output_tokens: {}\n", self.output_tokens));
        if !self.stop_reason.is_empty() {
            doc.push_str(&format!("  stop_reason: {}\n", self.stop_reason));
        }

        if !self.thinking_text.is_empty() {
            doc.push_str(&format!("  thinking: {}", yaml_block_scalar(&self.thinking_text, 4)));
        }

        if !self.generated_text.is_empty() {
            if self.generated_text.contains('\n') {
                doc.push_str(&format!("  text: {}", yaml_block_scalar(&self.generated_text, 4)));
            } else {
                doc.push_str(&format!("  text: {}\n", yaml_escape_inline(&self.generated_text)));
            }
        }

        if let Err(e) = std::fs::write(&self.path, doc.as_bytes()) {
            error!("Failed to write session log: {}", e);
        }
    }
}

fn utc_now() -> (String, String, String) {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let time_of_day = secs % 86400;
    let hours = time_of_day / 3600;
    let minutes = (time_of_day % 3600) / 60;
    let seconds = time_of_day % 60;

    let z = (secs / 86400) as i64 + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };

    let iso = format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        y, m, d, hours, minutes, seconds
    );
    let date = format!("{:04}-{:02}-{:02}", y, m, d);
    let time = format!("{:02}-{:02}-{:02}", hours, minutes, seconds);
    (iso, date, time)
}

fn yaml_escape_inline(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            _ => out.push(c),
        }
    }
    out.push('"');
    out
}

fn yaml_block_scalar(s: &str, indent: usize) -> String {
    if s.is_empty() {
        return "\"\"".to_string();
    }
    let prefix = " ".repeat(indent);
    let mut out = String::from("|\n");
    for line in s.lines() {
        out.push_str(&prefix);
        out.push_str(line);
        out.push('\n');
    }
    out
}
