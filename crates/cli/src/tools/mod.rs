//! Tool calling support for Qwen3 chat mode.
//!
//! Implements 4 local tools: get_datetime, calculate, list_directory, read_file.
//! Follows the Qwen3 native tool call format (`<tool_call>...</tool_call>` XML tags).

mod types;
mod parsing;
mod execution;
mod prompt;

pub use types::MAX_TOOL_ITERATIONS;
pub use parsing::{
    extract_tool_calls_from_tokens,
    parse_tool_calls,
    parse_raw_tool_call,
};
pub use execution::{execute_tool, format_tool_response, print_tools_status};
pub use prompt::build_tools_system_prompt;
