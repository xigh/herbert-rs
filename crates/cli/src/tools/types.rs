/// Maximum number of tool call round-trips before forcing a stop.
pub const MAX_TOOL_ITERATIONS: usize = 10;

/// Tool definitions in Qwen3 JSON format.
pub(super) const TOOL_DEFINITIONS: &[&str] = &[
    r#"{"type": "function", "function": {"name": "get_datetime", "description": "Get the current date and time. Only call this when the user explicitly asks for the current date or time.", "parameters": {"type": "object", "properties": {}, "required": []}}}"#,
    r#"{"type": "function", "function": {"name": "calculate", "description": "Evaluate a mathematical expression using bc calculator. Only call this when the user asks to compute a specific calculation.", "parameters": {"type": "object", "properties": {"expression": {"type": "string", "description": "Mathematical expression to evaluate (e.g. '2+3', 'sqrt(144)', '3.14*2^10')"}}, "required": ["expression"]}}}"#,
    r#"{"type": "function", "function": {"name": "list_directory", "description": "List files and directories at a given path. Only call this when the user asks to see the contents of a directory.", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "Directory path to list (e.g. '/home/user', '.')"}}, "required": ["path"]}}}"#,
    r#"{"type": "function", "function": {"name": "read_file", "description": "Read the contents of a text file. Only call this when the user asks to read or view a specific file.", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "File path to read"}, "max_lines": {"type": "integer", "description": "Maximum number of lines to return (default: 100)"}}, "required": ["path"]}}}"#,
];

/// Known tool names for raw JSON detection.
pub(super) const KNOWN_TOOLS: &[&str] = &["get_datetime", "calculate", "list_directory", "read_file"];

/// A parsed tool call.
#[derive(Debug)]
pub struct ToolCall {
    pub name: String,
    pub arguments: serde_json::Value,
}

impl ToolCall {
    /// Short display string for the arguments (for logging).
    pub fn args_display(&self) -> String {
        let s = self.arguments.to_string();
        if s.len() > 120 {
            format!("{}...", &s[..120])
        } else {
            s
        }
    }
}
