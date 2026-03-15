fn env_truthy(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .map(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
        .unwrap_or(false)
}

pub fn cpu_experimental_enabled() -> bool {
    env_truthy("HERBERT_CPU_EXPERIMENTAL")
}

pub fn deferred_accum_enabled() -> bool {
    env_truthy("HERBERT_DEFERRED_ACCUM")
}
