use anyhow::bail;
use herbert_core::backend::Backend;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BackendConsumer {
    Cli,
    Server,
}

impl BackendConsumer {
    fn binary_name(self) -> &'static str {
        match self {
            Self::Cli => "herbert",
            Self::Server => "herbert-server",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BackendAvailability {
    Available,
    Unavailable(&'static str),
}

impl BackendAvailability {
    fn is_available(self) -> bool {
        matches!(self, Self::Available)
    }

    fn reason(self) -> Option<&'static str> {
        match self {
            Self::Available => None,
            Self::Unavailable(reason) => Some(reason),
        }
    }
}

#[derive(Clone, Copy)]
struct BackendSpec {
    name: &'static str,
    supports_cli: bool,
    supports_server: bool,
    availability: BackendAvailability,
    factory: fn() -> Box<dyn Backend>,
}

impl BackendSpec {
    fn supports(self, consumer: BackendConsumer) -> bool {
        match consumer {
            BackendConsumer::Cli => self.supports_cli,
            BackendConsumer::Server => self.supports_server,
        }
    }
}

#[cfg(target_os = "macos")]
const METAL_AVAILABILITY: BackendAvailability = BackendAvailability::Available;
#[cfg(not(target_os = "macos"))]
const METAL_AVAILABILITY: BackendAvailability = BackendAvailability::Unavailable("requires macOS");

#[cfg(target_os = "linux")]
const VULKAN_AVAILABILITY: BackendAvailability = BackendAvailability::Available;
#[cfg(not(target_os = "linux"))]
const VULKAN_AVAILABILITY: BackendAvailability = BackendAvailability::Unavailable("requires Linux");

fn make_q4() -> Box<dyn Backend> {
    Box::new(herbert_backend_q4::Q4Backend::new())
}

fn make_bf16() -> Box<dyn Backend> {
    Box::new(herbert_backend_bf16::Bf16Backend::new())
}

fn make_bf16_avx512() -> Box<dyn Backend> {
    Box::new(herbert_backend_bf16_avx512::Bf16Avx512Backend::new())
}

fn make_int8_avx512() -> Box<dyn Backend> {
    Box::new(herbert_backend_int8_avx512::Int8Avx512Backend::new())
}

fn make_metal_q4() -> Box<dyn Backend> {
    Box::new(herbert_backend_metal::MetalBackend::with_quant_mode(
        herbert_backend_metal::QuantMode::Q4,
    ))
}

fn make_metal_int8() -> Box<dyn Backend> {
    Box::new(herbert_backend_metal::MetalBackend::with_quant_mode(
        herbert_backend_metal::QuantMode::Int8,
    ))
}

fn make_metal_bf16() -> Box<dyn Backend> {
    Box::new(herbert_backend_metal::MetalBackend::with_quant_mode(
        herbert_backend_metal::QuantMode::BF16,
    ))
}

fn make_vulkan_q4() -> Box<dyn Backend> {
    Box::new(herbert_backend_vulkan::VulkanBackend::with_quant_mode(
        herbert_backend_vulkan::QuantMode::Q4,
    ))
}

fn make_vulkan_int8() -> Box<dyn Backend> {
    Box::new(herbert_backend_vulkan::VulkanBackend::with_quant_mode(
        herbert_backend_vulkan::QuantMode::Int8,
    ))
}

fn make_vulkan_bf16() -> Box<dyn Backend> {
    Box::new(herbert_backend_vulkan::VulkanBackend::with_quant_mode(
        herbert_backend_vulkan::QuantMode::BF16,
    ))
}

const ALL_BACKENDS: &[BackendSpec] = &[
    BackendSpec {
        name: "q4",
        supports_cli: true,
        supports_server: true,
        availability: BackendAvailability::Available,
        factory: make_q4,
    },
    BackendSpec {
        name: "bf16",
        supports_cli: true,
        supports_server: true,
        availability: BackendAvailability::Available,
        factory: make_bf16,
    },
    BackendSpec {
        name: "bf16-avx512",
        supports_cli: true,
        supports_server: true,
        availability: BackendAvailability::Available,
        factory: make_bf16_avx512,
    },
    BackendSpec {
        name: "int8-avx512",
        supports_cli: true,
        supports_server: true,
        availability: BackendAvailability::Available,
        factory: make_int8_avx512,
    },
    BackendSpec {
        name: "metal-q4",
        supports_cli: true,
        supports_server: true,
        availability: METAL_AVAILABILITY,
        factory: make_metal_q4,
    },
    BackendSpec {
        name: "metal-int8",
        supports_cli: true,
        supports_server: true,
        availability: METAL_AVAILABILITY,
        factory: make_metal_int8,
    },
    BackendSpec {
        name: "metal-bf16",
        supports_cli: true,
        supports_server: true,
        availability: METAL_AVAILABILITY,
        factory: make_metal_bf16,
    },
    BackendSpec {
        name: "vulkan-q4",
        supports_cli: true,
        supports_server: false,
        availability: VULKAN_AVAILABILITY,
        factory: make_vulkan_q4,
    },
    BackendSpec {
        name: "vulkan-int8",
        supports_cli: true,
        supports_server: false,
        availability: VULKAN_AVAILABILITY,
        factory: make_vulkan_int8,
    },
    BackendSpec {
        name: "vulkan-bf16",
        supports_cli: true,
        supports_server: false,
        availability: VULKAN_AVAILABILITY,
        factory: make_vulkan_bf16,
    },
];

pub fn available_backend_names(consumer: BackendConsumer) -> Vec<&'static str> {
    ALL_BACKENDS
        .iter()
        .copied()
        .filter(|spec| spec.supports(consumer) && spec.availability.is_available())
        .map(|spec| spec.name)
        .collect()
}

pub fn available_backends_help(consumer: BackendConsumer) -> String {
    let mut out = String::from("Available backends:\n");
    let auto = resolve_auto(consumer);
    out.push_str(&format!("  auto (=> {})\n", auto));
    for name in available_backend_names(consumer) {
        out.push_str("  ");
        out.push_str(name);
        out.push('\n');
    }
    out
}

/// Preferred backends for auto-selection, in priority order.
/// First available + supported backend wins.
const AUTO_PRIORITY: &[&str] = &[
    "metal-q4",   // macOS Apple Silicon
    "vulkan-q4",  // Linux/Windows with GPU
    "q4",         // CPU fallback (always available)
];

/// Resolve "auto" to the best available backend name for this platform.
pub fn resolve_auto(consumer: BackendConsumer) -> &'static str {
    for &name in AUTO_PRIORITY {
        if let Some(spec) = ALL_BACKENDS.iter().copied().find(|s| s.name == name) {
            if spec.supports(consumer) && spec.availability.is_available() {
                return spec.name;
            }
        }
    }
    "q4" // ultimate fallback
}

pub fn create_backend(consumer: BackendConsumer, name: &str) -> anyhow::Result<Box<dyn Backend>> {
    let resolved = if name == "auto" { resolve_auto(consumer) } else { name };

    let Some(spec) = ALL_BACKENDS.iter().copied().find(|spec| spec.name == resolved) else {
        bail!(
            "Unknown backend: {}. Available options on this build: {}",
            resolved,
            join_backend_names(consumer)
        );
    };

    if !spec.supports(consumer) {
        bail!(
            "Backend '{}' is not supported by {}. Available options on this build: {}",
            resolved,
            consumer.binary_name(),
            join_backend_names(consumer)
        );
    }

    if let Some(reason) = spec.availability.reason() {
        bail!(
            "Backend '{}' is not available on this build: {}. Available options on this build: {}",
            resolved,
            reason,
            join_backend_names(consumer)
        );
    }

    if name == "auto" {
        eprintln!("[auto] Selected backend: {}", resolved);
    }

    Ok((spec.factory)())
}

fn join_backend_names(consumer: BackendConsumer) -> String {
    let names = available_backend_names(consumer);
    if names.is_empty() {
        "none".to_string()
    } else {
        names.join(", ")
    }
}

#[cfg(test)]
mod tests {
    use super::{
        available_backend_names, available_backends_help, create_backend, resolve_auto,
        BackendConsumer,
    };
    use std::collections::HashSet;

    #[test]
    fn cli_backend_names_are_unique() {
        let names = available_backend_names(BackendConsumer::Cli);
        let unique: HashSet<_> = names.iter().copied().collect();
        assert_eq!(names.len(), unique.len());
        assert!(!names.contains(&"cpu-experimental"));
    }

    #[test]
    fn server_excludes_non_server_backends() {
        let names = available_backend_names(BackendConsumer::Server);
        assert!(!names.iter().any(|name| name.starts_with("vulkan-")));
    }

    #[test]
    fn help_text_uses_shared_format() {
        let help = available_backends_help(BackendConsumer::Cli);
        assert!(help.starts_with("Available backends:\n"));
        assert!(help.lines().skip(1).all(|line: &str| line.starts_with("  ")));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn cli_lists_metal_backends_on_macos() {
        let names = available_backend_names(BackendConsumer::Cli);
        assert!(names.contains(&"metal-q4"));
        assert!(names.contains(&"metal-int8"));
        assert!(names.contains(&"metal-bf16"));
    }

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn metal_backends_report_platform_specific_error() {
        let err = match create_backend(BackendConsumer::Cli, "metal-q4") {
            Ok(_) => panic!("metal-q4 should be unavailable on this target"),
            Err(err) => err.to_string(),
        };
        assert!(err.contains("requires macOS"));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn cli_lists_vulkan_backends_on_linux() {
        let names = available_backend_names(BackendConsumer::Cli);
        assert!(names.contains(&"vulkan-q4"));
        assert!(names.contains(&"vulkan-int8"));
        assert!(names.contains(&"vulkan-bf16"));
    }

    #[cfg(not(target_os = "linux"))]
    #[test]
    fn vulkan_backends_report_platform_specific_error() {
        let err = match create_backend(BackendConsumer::Cli, "vulkan-q4") {
            Ok(_) => panic!("vulkan-q4 should be unavailable on this target"),
            Err(err) => err.to_string(),
        };
        assert!(err.contains("requires Linux"));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn auto_resolves_to_metal_q4_on_macos() {
        assert_eq!(resolve_auto(BackendConsumer::Cli), "metal-q4");
        assert_eq!(resolve_auto(BackendConsumer::Server), "metal-q4");
    }

    #[test]
    fn auto_backend_creates_successfully() {
        let backend = create_backend(BackendConsumer::Cli, "auto");
        assert!(backend.is_ok());
    }

    #[test]
    fn help_text_includes_auto() {
        let help = available_backends_help(BackendConsumer::Cli);
        assert!(help.contains("auto"));
    }
}
