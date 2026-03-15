//! Backend implementation for INT8 AVX-512.

use herbert_backend_common::cpu_backend::{CpuBackendConfig, GenericCpuBackend};
use herbert_backend_common::generic_model::GenericModel;
use herbert_core::backend::LoadOpts;
use herbert_core::config::Config;
use herbert_core::error::{HerbertError, Result};
use std::path::Path;

use crate::ops::Int8Avx512Ops;

pub struct Int8Avx512Config;

impl CpuBackendConfig for Int8Avx512Config {
    type Ops = Int8Avx512Ops;
    const NAME: &'static str = "int8-avx512";

    fn configure_thread_pool(num_threads: usize) -> Result<()> {
        crate::thread_pool::configure_global_pool(num_threads)
            .map_err(|e| HerbertError::Backend(format!("failed to configure thread pool: {}", e)))
    }

    fn load_model(dir: &Path, opts: LoadOpts) -> Result<(Config, GenericModel<Int8Avx512Ops>)> {
        crate::loader::load_model(dir, opts)
    }
}

pub type Int8Avx512Backend = GenericCpuBackend<Int8Avx512Config>;
