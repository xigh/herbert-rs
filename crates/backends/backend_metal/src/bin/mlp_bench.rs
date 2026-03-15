#[cfg(target_os = "macos")]
mod macos {
    use std::path::{Path, PathBuf};
    use std::time::Instant;

    use herbert_backend_metal::cache::{self, CacheMmap};
    use herbert_backend_metal::context::MetalContext;
    use herbert_backend_metal::kernels;
    use herbert_backend_metal::loader::{self, QuantMode};
    use herbert_backend_metal::memory::MetalBuffer;
    use herbert_backend_metal::model::{MetalLayerMLP, MetalModel, MetalWeight};
    use herbert_core::config::Config;
    use herbert_core::error::{HerbertError, Result};
    use objc2::runtime::ProtocolObject;
    use objc2_metal::{MTLCommandEncoder, MTLComputeCommandEncoder};

    #[derive(Clone, Copy)]
    enum BenchMode {
        DownSeparate,
        DownFused,
        DownFused8Row,
        GateUpSeparate,
        GateUpFusedQ4,
    }

    impl BenchMode {
        fn label(self) -> &'static str {
            match self {
                BenchMode::DownSeparate => "down_separate",
                BenchMode::DownFused => "down_fused",
                BenchMode::DownFused8Row => "down_fused_8row",
                BenchMode::GateUpSeparate => "gateup_separate",
                BenchMode::GateUpFusedQ4 => "gateup_fused_q4",
            }
        }
    }

    struct Args {
        model: PathBuf,
        layer: Option<usize>,
        iters: usize,
        warmup: usize,
        max_tokens: usize,
    }

    struct LayerBuffers {
        gate_out: MetalBuffer,
        up_out: MetalBuffer,
        fused_gate_out: MetalBuffer,
        down_tmp: MetalBuffer,
        down_out: MetalBuffer,
    }

    fn parse_args() -> Result<Args> {
        let mut model = None;
        let mut layer = None;
        let mut iters = 200usize;
        let mut warmup = 20usize;
        let mut max_tokens = 256usize;

        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--model" => model = args.next().map(PathBuf::from),
                "--layer" => {
                    let v = args.next().ok_or_else(|| {
                        HerbertError::Backend("--layer requires a value".into())
                    })?;
                    layer = Some(v.parse::<usize>().map_err(|_| {
                        HerbertError::Backend(format!("invalid --layer value: {v}"))
                    })?);
                }
                "--iters" => {
                    let v = args.next().ok_or_else(|| {
                        HerbertError::Backend("--iters requires a value".into())
                    })?;
                    iters = v.parse::<usize>().map_err(|_| {
                        HerbertError::Backend(format!("invalid --iters value: {v}"))
                    })?;
                }
                "--warmup" => {
                    let v = args.next().ok_or_else(|| {
                        HerbertError::Backend("--warmup requires a value".into())
                    })?;
                    warmup = v.parse::<usize>().map_err(|_| {
                        HerbertError::Backend(format!("invalid --warmup value: {v}"))
                    })?;
                }
                "--max-tokens" => {
                    let v = args.next().ok_or_else(|| {
                        HerbertError::Backend("--max-tokens requires a value".into())
                    })?;
                    max_tokens = v.parse::<usize>().map_err(|_| {
                        HerbertError::Backend(format!("invalid --max-tokens value: {v}"))
                    })?;
                }
                "--help" | "-h" => {
                    print_help();
                    std::process::exit(0);
                }
                other => {
                    return Err(HerbertError::Backend(format!("unknown arg: {other}")));
                }
            }
        }

        let model = model.ok_or_else(|| {
            HerbertError::Backend("missing required --model <dir>".into())
        })?;

        Ok(Args { model, layer, iters, warmup, max_tokens })
    }

    fn print_help() {
        eprintln!(
            "Usage: cargo run -p herbert-backend-metal --bin mlp_bench -- --model <dir> [--layer N] [--iters N] [--warmup N] [--max-tokens N]"
        );
    }

    fn load_q4_model(
        model_path: &Path,
        ctx: &MetalContext,
        max_tokens: usize,
    ) -> Result<(Config, MetalModel, Option<CacheMmap>)> {
        if let Some((config, model, mmap)) = cache::try_load_cache(
            &ctx.device, model_path, max_tokens, QuantMode::Q4,
        ) {
            Ok((config, model, Some(mmap)))
        } else {
            let (config, model) = loader::load_model(
                model_path, &ctx.device, max_tokens, QuantMode::Q4,
            )?;
            if let Err(e) = cache::save_cache(&model, &config, model_path, QuantMode::Q4) {
                eprintln!("[mlp-bench] Warning: failed to save cache: {e}");
            }
            Ok((config, model, None))
        }
    }

    fn dense_layer_indices(config: &Config) -> Vec<usize> {
        (0..config.num_layers)
            .filter(|&l| !config.is_moe_layer(l))
            .collect()
    }

    fn make_input(len: usize, base: f32) -> Vec<f32> {
        (0..len)
            .map(|i| {
                let x = i as f32;
                (x * 0.013 + base).sin() * 0.5 + (x * 0.007 + base * 0.3).cos() * 0.25
            })
            .collect()
    }

    fn bytes_of_f32(data: &[f32]) -> &[u8] {
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) }
    }

    fn matvec_weight(
        ctx: &MetalContext,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        input: &MetalBuffer,
        weight: &MetalWeight,
        output: &MetalBuffer,
    ) {
        match weight {
            MetalWeight::BF16(w) => {
                kernels::matvec::bf16_matvec(
                    ctx, encoder, input, &w.packed, output, w.n as u32, w.k as u32,
                );
            }
            MetalWeight::Int8(w) => {
                kernels::matvec::int8_matvec(
                    ctx, encoder, input, &w.packed, &w.scales, output,
                    w.n as u32, w.k as u32,
                );
            }
            MetalWeight::Q4(w) => {
                kernels::matvec::q4_matvec(
                    ctx, encoder, input, &w.packed, &w.scales, output,
                    w.n as u32, w.k as u32,
                );
            }
        }
    }

    fn matvec_weight_residual_add(
        ctx: &MetalContext,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        input: &MetalBuffer,
        weight: &MetalWeight,
        residual: &MetalBuffer,
        output: &MetalBuffer,
    ) -> Result<()> {
        match weight {
            MetalWeight::Q4(w) => {
                kernels::matvec::q4_matvec_residual_add(
                    ctx, encoder, input, &w.packed, &w.scales, residual, output,
                    w.n as u32, w.k as u32,
                );
                Ok(())
            }
            _ => Err(HerbertError::Backend(
                "q4_matvec_residual_add benchmark expects Q4 weights".into(),
            )),
        }
    }

    fn matvec_weight_residual_add_8row(
        ctx: &MetalContext,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        input: &MetalBuffer,
        weight: &MetalWeight,
        residual: &MetalBuffer,
        output: &MetalBuffer,
    ) -> Result<()> {
        match weight {
            MetalWeight::Q4(w) => {
                if !(w.n as u32).is_multiple_of(8) {
                    return Err(HerbertError::Backend(
                        "q4_matvec_residual_add_8row requires output rows divisible by 8".into(),
                    ));
                }
                kernels::matvec::q4_matvec_residual_add_8row(
                    ctx, encoder, input, &w.packed, &w.scales, residual, output,
                    w.n as u32, w.k as u32,
                );
                Ok(())
            }
            _ => Err(HerbertError::Backend(
                "q4_matvec_residual_add_8row benchmark expects Q4 weights".into(),
            )),
        }
    }

    fn encode_mode(
        ctx: &MetalContext,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        model: &MetalModel,
        config: &Config,
        layer_idx: usize,
        bufs: &LayerBuffers,
        hidden_input: &MetalBuffer,
        inter_input: &MetalBuffer,
        mode: BenchMode,
    ) -> Result<()> {
        let layer = &model.layers[layer_idx];
        let MetalLayerMLP::Dense { gate_proj, up_proj, down_proj } = &layer.mlp else {
            return Err(HerbertError::Backend(format!(
                "layer {layer_idx} is not dense"
            )));
        };

        match mode {
            BenchMode::DownSeparate => {
                matvec_weight(ctx, encoder, inter_input, down_proj, &bufs.down_tmp);
                kernels::activation::residual_add(
                    ctx, encoder, &bufs.down_out, &bufs.down_tmp, config.hidden_size as u32,
                );
            }
            BenchMode::DownFused => {
                matvec_weight_residual_add(
                    ctx, encoder, inter_input, down_proj, &bufs.down_out, &bufs.down_out,
                )?;
            }
            BenchMode::DownFused8Row => {
                matvec_weight_residual_add_8row(
                    ctx, encoder, inter_input, down_proj, &bufs.down_out, &bufs.down_out,
                )?;
            }
            BenchMode::GateUpSeparate => {
                matvec_weight(ctx, encoder, hidden_input, gate_proj, &bufs.gate_out);
                matvec_weight(ctx, encoder, hidden_input, up_proj, &bufs.up_out);
                kernels::activation::swiglu(
                    ctx, encoder, &bufs.gate_out, &bufs.up_out,
                    config.intermediate_size as u32,
                );
            }
            BenchMode::GateUpFusedQ4 => {
                let used = kernels::moe::fused_gate_up_swiglu_weight(
                    ctx, encoder, hidden_input, gate_proj, up_proj, &bufs.fused_gate_out,
                );
                if !used {
                    return Err(HerbertError::Backend(
                        "fused_gate_up_swiglu_weight not available for these weights".into(),
                    ));
                }
            }
        }

        Ok(())
    }

    fn run_once(
        ctx: &MetalContext,
        model: &MetalModel,
        config: &Config,
        layers: &[usize],
        bufs: &[LayerBuffers],
        hidden_input: &MetalBuffer,
        inter_input: &MetalBuffer,
        residual_template: &[u8],
        mode: BenchMode,
    ) -> Result<()> {
        if matches!(mode, BenchMode::DownSeparate | BenchMode::DownFused | BenchMode::DownFused8Row) {
            for buf in bufs {
                buf.down_out.write_bytes(residual_template);
            }
        }

        let cb = ctx.begin_command_buffer()?;
        let encoder = MetalContext::new_compute_encoder(&cb)?;
        for (slot, &layer_idx) in layers.iter().enumerate() {
            encode_mode(
                ctx, &encoder, model, config, layer_idx, &bufs[slot],
                hidden_input, inter_input, mode,
            )?;
        }
        encoder.endEncoding();
        MetalContext::submit_and_wait(&cb)
    }

    fn bench_mode(
        ctx: &MetalContext,
        model: &MetalModel,
        config: &Config,
        layers: &[usize],
        bufs: &[LayerBuffers],
        hidden_input: &MetalBuffer,
        inter_input: &MetalBuffer,
        residual_template: &[u8],
        mode: BenchMode,
        warmup: usize,
        iters: usize,
    ) -> Result<f64> {
        for _ in 0..warmup {
            run_once(
                ctx, model, config, layers, bufs, hidden_input, inter_input,
                residual_template, mode,
            )?;
        }

        let t0 = Instant::now();
        for _ in 0..iters {
            run_once(
                ctx, model, config, layers, bufs, hidden_input, inter_input,
                residual_template, mode,
            )?;
        }
        Ok(t0.elapsed().as_secs_f64() * 1000.0 / iters as f64)
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    fn check_gateup_diff(
        ctx: &MetalContext,
        model: &MetalModel,
        config: &Config,
        layer_idx: usize,
        bufs: &LayerBuffers,
        hidden_input: &MetalBuffer,
        inter_input: &MetalBuffer,
        residual_template: &[u8],
    ) -> Result<f32> {
        run_once(
            ctx, model, config, &[layer_idx], std::slice::from_ref(bufs),
            hidden_input, inter_input, residual_template, BenchMode::GateUpSeparate,
        )?;
        let separate = bufs.gate_out.read_f32(config.intermediate_size);

        run_once(
            ctx, model, config, &[layer_idx], std::slice::from_ref(bufs),
            hidden_input, inter_input, residual_template, BenchMode::GateUpFusedQ4,
        )?;
        let fused = bufs.fused_gate_out.read_f32(config.intermediate_size);

        Ok(max_abs_diff(&separate, &fused))
    }

    fn check_down_diff(
        ctx: &MetalContext,
        model: &MetalModel,
        config: &Config,
        layer_idx: usize,
        bufs: &LayerBuffers,
        hidden_input: &MetalBuffer,
        inter_input: &MetalBuffer,
        residual_template: &[u8],
    ) -> Result<f32> {
        run_once(
            ctx, model, config, &[layer_idx], std::slice::from_ref(bufs),
            hidden_input, inter_input, residual_template, BenchMode::DownSeparate,
        )?;
        let separate = bufs.down_out.read_f32(config.hidden_size);

        run_once(
            ctx, model, config, &[layer_idx], std::slice::from_ref(bufs),
            hidden_input, inter_input, residual_template, BenchMode::DownFused,
        )?;
        let fused = bufs.down_out.read_f32(config.hidden_size);

        Ok(max_abs_diff(&separate, &fused))
    }

    pub fn run() -> Result<()> {
        let args = parse_args()?;

        let ctx = MetalContext::new()?;
        eprintln!("[mlp-bench] Using device: {}", ctx.device_name());

        let (config, model, _cache_mmap) = load_q4_model(&args.model, &ctx, args.max_tokens)?;
        let dense_layers = dense_layer_indices(&config);
        if dense_layers.is_empty() {
            return Err(HerbertError::Backend("model has no dense layers".into()));
        }

        let layers: Vec<usize> = if let Some(layer) = args.layer {
            if !dense_layers.contains(&layer) {
                return Err(HerbertError::Backend(format!(
                    "layer {layer} is not a dense layer"
                )));
            }
            vec![layer]
        } else {
            dense_layers
        };

        let hidden = config.hidden_size;
        let inter = config.intermediate_size;
        let hidden_input_data = make_input(hidden, 0.3);
        let inter_input_data = make_input(inter, 0.7);
        let residual_data = make_input(hidden, 1.1);
        let residual_bytes = bytes_of_f32(&residual_data);

        let hidden_input = MetalBuffer::from_f32(&ctx.device, &hidden_input_data)?;
        let inter_input = MetalBuffer::from_f32(&ctx.device, &inter_input_data)?;

        let mut bufs = Vec::with_capacity(layers.len());
        for _ in &layers {
            bufs.push(LayerBuffers {
                gate_out: MetalBuffer::new_uninit(&ctx.device, (inter * 4) as u64)?,
                up_out: MetalBuffer::new_uninit(&ctx.device, (inter * 4) as u64)?,
                fused_gate_out: MetalBuffer::new_uninit(&ctx.device, (inter * 4) as u64)?,
                down_tmp: MetalBuffer::new_uninit(&ctx.device, (hidden * 4) as u64)?,
                down_out: MetalBuffer::from_f32(&ctx.device, &residual_data)?,
            });
        }

        eprintln!(
            "[mlp-bench] dense_layers={} selected_layers={} hidden={} inter={} iters={} warmup={}",
            dense_layer_indices(&config).len(),
            layers.len(),
            hidden,
            inter,
            args.iters,
            args.warmup,
        );
        if let Some(layer) = args.layer {
            eprintln!("[mlp-bench] layer={layer}");
        } else {
            eprintln!("[mlp-bench] layer=all-dense");
        }

        let gateup_diff = check_gateup_diff(
            &ctx, &model, &config, layers[0], &bufs[0], &hidden_input, &inter_input, residual_bytes,
        )?;
        let down_diff = check_down_diff(
            &ctx, &model, &config, layers[0], &bufs[0], &hidden_input, &inter_input, residual_bytes,
        )?;

        let down_sep_ms = bench_mode(
            &ctx, &model, &config, &layers, &bufs, &hidden_input, &inter_input,
            residual_bytes, BenchMode::DownSeparate, args.warmup, args.iters,
        )?;
        let down_fused_ms = bench_mode(
            &ctx, &model, &config, &layers, &bufs, &hidden_input, &inter_input,
            residual_bytes, BenchMode::DownFused, args.warmup, args.iters,
        )?;
        let down_fused_8row_ms = bench_mode(
            &ctx, &model, &config, &layers, &bufs, &hidden_input, &inter_input,
            residual_bytes, BenchMode::DownFused8Row, args.warmup, args.iters,
        )?;
        let gateup_sep_ms = bench_mode(
            &ctx, &model, &config, &layers, &bufs, &hidden_input, &inter_input,
            residual_bytes, BenchMode::GateUpSeparate, args.warmup, args.iters,
        )?;
        let gateup_fused_ms = bench_mode(
            &ctx, &model, &config, &layers, &bufs, &hidden_input, &inter_input,
            residual_bytes, BenchMode::GateUpFusedQ4, args.warmup, args.iters,
        )?;

        let layer_count = layers.len() as f64;
        println!("MLP Bench");
        println!("device: {}", ctx.device_name());
        println!("model: {}", args.model.display());
        println!("selected_layers: {}", layers.len());
        println!("hidden_size: {}", hidden);
        println!("intermediate_size: {}", inter);
        println!();
        println!(
            "{:<22} {:>10} {:>12}",
            "mode", "ms/iter", "ms/layer",
        );
        println!(
            "{:<22} {:>10.3} {:>12.3}",
            BenchMode::DownSeparate.label(), down_sep_ms, down_sep_ms / layer_count,
        );
        println!(
            "{:<22} {:>10.3} {:>12.3}",
            BenchMode::DownFused.label(), down_fused_ms, down_fused_ms / layer_count,
        );
        println!(
            "{:<22} {:>10.3} {:>12.3}",
            BenchMode::DownFused8Row.label(), down_fused_8row_ms, down_fused_8row_ms / layer_count,
        );
        println!(
            "{:<22} {:>10.3} {:>12.3}",
            BenchMode::GateUpSeparate.label(), gateup_sep_ms, gateup_sep_ms / layer_count,
        );
        println!(
            "{:<22} {:>10.3} {:>12.3}",
            BenchMode::GateUpFusedQ4.label(), gateup_fused_ms, gateup_fused_ms / layer_count,
        );
        println!();
        println!(
            "down_fused_speedup: {:.3}x",
            down_sep_ms / down_fused_ms,
        );
        println!(
            "down_fused_8row_speedup: {:.3}x",
            down_sep_ms / down_fused_8row_ms,
        );
        println!(
            "gateup_fused_speedup: {:.3}x",
            gateup_sep_ms / gateup_fused_ms,
        );
        println!("down_max_abs_diff: {:.6}", down_diff);
        println!("gateup_max_abs_diff: {:.6}", gateup_diff);

        Ok(())
    }
}

#[cfg(target_os = "macos")]
fn main() -> herbert_core::error::Result<()> {
    macos::run()
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("mlp_bench is only available on macOS");
    std::process::exit(1);
}
