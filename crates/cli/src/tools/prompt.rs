use super::types::TOOL_DEFINITIONS;

/// Build the system prompt with tool definitions injected (Qwen3 format).
pub fn build_tools_system_prompt(user_system: Option<&str>) -> String {
    let mut out = String::new();
    if let Some(sys) = user_system {
        out.push_str(sys);
        out.push_str("\n\n");
    }

    // Qwen3 format: JSON inside <tool_call> XML tags
    out.push_str("# Tools\n\n");
    out.push_str("You may call one or more functions to assist with the user query.\n\n");
    out.push_str("You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n");
    for def in TOOL_DEFINITIONS {
        out.push_str(def);
        out.push('\n');
    }
    out.push_str("</tools>\n\n");
    out.push_str("IMPORTANT: Do NOT call any tool unless the user's message explicitly requires it. For greetings, chitchat, opinions, or general knowledge questions, respond directly without calling any tool.\n\n");
    out.push_str("When a tool is needed, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n");
    out.push_str("<tool_call>\n");
    out.push_str(r#"{"name": <function-name>, "arguments": <args-json-object>}"#);
    out.push_str("\n</tool_call>");
    out
}
