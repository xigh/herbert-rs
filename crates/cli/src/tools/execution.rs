use std::process::Command;

use super::types::ToolCall;

/// Maximum file size (in bytes) we'll read with read_file.
const MAX_FILE_SIZE: usize = 64 * 1024; // 64 KB

/// Maximum lines returned by read_file.
const DEFAULT_MAX_LINES: usize = 100;

/// Execute a tool call and return the result string.
pub fn execute_tool(call: &ToolCall) -> String {
    match call.name.as_str() {
        "get_datetime" => tool_datetime(),
        "calculate" => {
            let expr = call
                .arguments
                .get("expression")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            tool_calculate(expr)
        }
        "list_directory" => {
            let path = call
                .arguments
                .get("path")
                .and_then(|v| v.as_str())
                .unwrap_or(".");
            tool_list_dir(path)
        }
        "read_file" => {
            let path = call
                .arguments
                .get("path")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let max_lines = call
                .arguments
                .get("max_lines")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(DEFAULT_MAX_LINES);
            tool_read_file(path, max_lines)
        }
        other => format!("Error: unknown tool '{}'", other),
    }
}

/// Format tool results as a tool_response turn for continue_prefill.
///
/// Format tool response in Qwen3 format.
pub fn format_tool_response(results: &[(String, String)]) -> String {
    let mut out = String::new();
    out.push_str("<|im_end|>\n<|im_start|>user\n");
    for (_name, result) in results {
        out.push_str("<tool_response>\n");
        out.push_str(result);
        if !result.ends_with('\n') {
            out.push('\n');
        }
        out.push_str("</tool_response>\n");
    }
    out.push_str("<|im_end|>\n<|im_start|>assistant\n");
    out
}

/// Print the list of available tools to stderr.
pub fn print_tools_status(enabled: bool) {
    if enabled {
        eprintln!("--- Tools (enabled) ---");
    } else {
        eprintln!("--- Tools (disabled) ---");
    }
    eprintln!("  get_datetime    - Get the current date and time");
    eprintln!("  calculate       - Evaluate a math expression (via bc)");
    eprintln!("  list_directory  - List files in a directory");
    eprintln!("  read_file       - Read a text file (up to 100 lines)");
}

// ── Tool implementations ──

fn tool_datetime() -> String {
    match Command::new("date").arg("+%Y-%m-%d %H:%M:%S %Z").output() {
        Ok(output) => {
            let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if output.status.success() && !stdout.is_empty() {
                stdout
            } else {
                "Error: date command failed".to_string()
            }
        }
        Err(e) => format!("Error: {}", e),
    }
}

fn tool_calculate(expression: &str) -> String {
    if expression.is_empty() {
        return "Error: empty expression".to_string();
    }
    // Sanitize: only allow safe characters for bc
    let safe = expression
        .chars()
        .all(|c| c.is_ascii_digit() || "+-*/%^().= \t\nsqrtleioa".contains(c));
    if !safe {
        return "Error: expression contains disallowed characters".to_string();
    }
    match Command::new("bc")
        .arg("-l")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
    {
        Ok(mut child) => {
            use std::io::Write;
            if let Some(ref mut stdin) = child.stdin {
                let _ = stdin.write_all(expression.as_bytes());
                let _ = stdin.write_all(b"\n");
            }
            match child.wait_with_output() {
                Ok(output) => {
                    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
                    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
                    if !stdout.is_empty() {
                        stdout
                    } else if !stderr.is_empty() {
                        format!("Error: {}", stderr)
                    } else {
                        "Error: no output from bc".to_string()
                    }
                }
                Err(e) => format!("Error: {}", e),
            }
        }
        Err(e) => format!("Error: bc not available: {}", e),
    }
}

fn tool_list_dir(path: &str) -> String {
    if path.is_empty() {
        return "Error: empty path".to_string();
    }
    match std::fs::read_dir(path) {
        Ok(entries) => {
            let mut items: Vec<String> = Vec::new();
            for entry in entries {
                match entry {
                    Ok(e) => {
                        let name = e.file_name().to_string_lossy().to_string();
                        let is_dir = e.file_type().map(|ft| ft.is_dir()).unwrap_or(false);
                        if is_dir {
                            items.push(format!("{}/", name));
                        } else {
                            items.push(name);
                        }
                    }
                    Err(e) => items.push(format!("(error: {})", e)),
                }
            }
            items.sort();
            if items.is_empty() {
                "(empty directory)".to_string()
            } else {
                items.join("\n")
            }
        }
        Err(e) => format!("Error: {}", e),
    }
}

fn tool_read_file(path: &str, max_lines: usize) -> String {
    if path.is_empty() {
        return "Error: empty path".to_string();
    }
    // Check file size first
    match std::fs::metadata(path) {
        Ok(meta) => {
            if meta.len() > MAX_FILE_SIZE as u64 {
                return format!(
                    "Error: file too large ({} bytes, max {} bytes)",
                    meta.len(),
                    MAX_FILE_SIZE
                );
            }
        }
        Err(e) => return format!("Error: {}", e),
    }
    match std::fs::read_to_string(path) {
        Ok(content) => {
            let lines: Vec<&str> = content.lines().collect();
            let total = lines.len();
            let limit = max_lines.min(total);
            let mut out = lines[..limit].join("\n");
            if limit < total {
                out.push_str(&format!("\n... ({} more lines)", total - limit));
            }
            out
        }
        Err(e) => format!("Error: {}", e),
    }
}
