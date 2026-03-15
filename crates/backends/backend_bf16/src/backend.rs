//! Backend implementation for BF16 (no quantization).

use herbert_backend_common::cpu_backend::{CpuBackendConfig, GenericCpuBackend};
use herbert_backend_common::generic_model::GenericModel;
use herbert_core::backend::LoadOpts;
use herbert_core::config::Config;
use herbert_core::error::{HerbertError, Result};
use std::path::Path;

use crate::ops::Bf16Ops;

pub struct Bf16Config;

impl CpuBackendConfig for Bf16Config {
    type Ops = Bf16Ops;
    const NAME: &'static str = "bf16";

    fn configure_thread_pool(num_threads: usize) -> Result<()> {
        crate::thread_pool::configure_global_pool(num_threads)
            .map_err(|e| HerbertError::Backend(format!("failed to configure thread pool: {}", e)))
    }

    fn load_model(dir: &Path, opts: LoadOpts) -> Result<(Config, GenericModel<Bf16Ops>)> {
        crate::loader::load_model(dir, opts)
    }
}

pub type Bf16Backend = GenericCpuBackend<Bf16Config>;
