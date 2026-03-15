//! SPIR-V shader loading and compute pipeline creation.
//!
//! Supports runtime selection of subgroup-size-specific shader variants (w32/w64).

use ash::vk;
use herbert_core::error::{HerbertError, Result};

/// Embed a subgroup-dependent shader (select w32 or w64 at runtime).
macro_rules! include_spv_sg {
    ($name:literal, $sg:expr) => {
        match $sg {
            64 => include_bytes!(concat!(env!("OUT_DIR"), "/", $name, "_w64.spv")) as &[u8],
            _ => include_bytes!(concat!(env!("OUT_DIR"), "/", $name, "_w32.spv")) as &[u8],
        }
    };
}

/// Embed a fixed (subgroup-independent) shader.
macro_rules! include_spv {
    ($name:literal) => {
        include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv"))
    };
}

#[allow(unused_imports)]
pub(crate) use include_spv;
#[allow(unused_imports)]
pub(crate) use include_spv_sg;

/// Create a Vulkan shader module from raw SPIR-V bytes.
fn create_shader_module(device: &ash::Device, spirv_bytes: &[u8]) -> Result<vk::ShaderModule> {
    assert!(
        spirv_bytes.len() % 4 == 0,
        "SPIR-V bytes must be u32-aligned"
    );
    let code: Vec<u32> = spirv_bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let create_info = vk::ShaderModuleCreateInfo::default().code(&code);
    unsafe { device.create_shader_module(&create_info, None) }
        .map_err(|e| HerbertError::Backend(format!("Failed to create shader module: {:?}", e)))
}

/// A single Vulkan compute pipeline with its layout and descriptor set layout.
pub struct ComputePipeline {
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
}

/// Create a compute pipeline from SPIR-V bytes.
pub fn create_compute_pipeline(
    device: &ash::Device,
    spirv_bytes: &[u8],
    num_bindings: u32,
    push_constant_size: u32,
    required_subgroup_size: u32,
) -> Result<ComputePipeline> {
    let module = create_shader_module(device, spirv_bytes)?;

    let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..num_bindings)
        .map(|i| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(i)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        })
        .collect();

    let ds_layout_info = vk::DescriptorSetLayoutCreateInfo::default()
        .flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR)
        .bindings(&bindings);

    let descriptor_set_layout =
        unsafe { device.create_descriptor_set_layout(&ds_layout_info, None) }.map_err(|e| {
            HerbertError::Backend(format!(
                "Failed to create descriptor set layout: {:?}",
                e
            ))
        })?;

    let push_ranges = if push_constant_size > 0 {
        vec![vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(push_constant_size)]
    } else {
        vec![]
    };

    let layouts = [descriptor_set_layout];
    let layout_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&layouts)
        .push_constant_ranges(&push_ranges);

    let layout = unsafe { device.create_pipeline_layout(&layout_info, None) }.map_err(|e| {
        HerbertError::Backend(format!("Failed to create pipeline layout: {:?}", e))
    })?;

    let entry_name = std::ffi::CString::new("main").unwrap();

    let mut subgroup_size_info =
        vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default()
            .required_subgroup_size(required_subgroup_size);

    let mut stage_info = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(module)
        .name(&entry_name);

    if required_subgroup_size > 0 {
        stage_info = stage_info
            .flags(vk::PipelineShaderStageCreateFlags::REQUIRE_FULL_SUBGROUPS)
            .push_next(&mut subgroup_size_info);
    }

    let pipeline_info = vk::ComputePipelineCreateInfo::default()
        .stage(stage_info)
        .layout(layout);

    let pipeline = unsafe {
        device.create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
    }
    .map_err(|e| HerbertError::Backend(format!("Failed to create compute pipeline: {:?}", e)))?
        [0];

    unsafe {
        device.destroy_shader_module(module, None);
    }

    Ok(ComputePipeline {
        pipeline,
        layout,
        descriptor_set_layout,
    })
}

/// Helper to destroy a ComputePipeline's Vulkan objects.
unsafe fn destroy_compute_pipeline(device: &ash::Device, cp: &ComputePipeline) {
    device.destroy_pipeline(cp.pipeline, None);
    device.destroy_pipeline_layout(cp.layout, None);
    device.destroy_descriptor_set_layout(cp.descriptor_set_layout, None);
}

/// All compute pipelines used by the Vulkan backend.
pub struct Pipelines {
    // Subgroup-dependent pipelines
    pub rms_norm: ComputePipeline,
    pub rms_norm_batch: ComputePipeline,
    pub head_rms_norm: ComputePipeline,
    pub softmax: ComputePipeline,
    pub argmax: ComputePipeline,
    pub f32_matvec: ComputePipeline,
    pub f32_matmul: ComputePipeline,
    pub attention_decode: ComputePipeline,
    pub attention_prefill: ComputePipeline,
    // Subgroup-independent pipelines
    pub residual_add: ComputePipeline,
    pub embedding: ComputePipeline,
    pub swiglu: ComputePipeline,
    pub rope_single: ComputePipeline,
    pub rope_batch: ComputePipeline,
    pub copy_buffer: ComputePipeline,
    pub kv_cache_append: ComputePipeline,
    pub kv_cache_append_batch: ComputePipeline,
    pub bias_add_batch: ComputePipeline,
    // MoE pipelines (subgroup-independent)
    pub scaled_add: ComputePipeline,
    pub moe_gather: ComputePipeline,
    pub moe_scatter_add: ComputePipeline,
    // Int8 pipelines (subgroup-dependent)
    pub int8_matvec: ComputePipeline,
    pub int8_matmul: ComputePipeline,
    // BF16 pipelines (subgroup-dependent)
    pub bf16_matvec: ComputePipeline,
    pub bf16_matmul: ComputePipeline,
    // Q4 pipelines (subgroup-dependent)
    pub q4_matvec: ComputePipeline,
    pub q4_matmul: ComputePipeline,
    // Q4 v5 optimized pipelines (vectorized K-group loads)
    pub q4_matvec_v5: ComputePipeline,
    pub q4_matmul_v5: ComputePipeline,
    pub use_q4_v5: bool,
    // Q4 cooperative matrix matmul (WMMA 16×16×16, prefill only)
    pub q4_matmul_coopmat: Option<ComputePipeline>,
    // Q4 cooperative matrix matmul — INT8 WMMA (NVIDIA-only, 16×16×32)
    pub q4_matmul_coopmat_i8: Option<ComputePipeline>,
}

impl Pipelines {
    /// Create all compute pipelines.
    pub fn new(device: &ash::Device, subgroup_size: u32, has_coopmat: bool, is_nvidia: bool) -> Result<Self> {
        let sg = subgroup_size;

        // ----- Subgroup-dependent pipelines -----
        let rms_norm = create_compute_pipeline(
            device, include_spv_sg!("rms_norm", sg), 3, 8, sg,
        )?;
        let rms_norm_batch = create_compute_pipeline(
            device, include_spv_sg!("rms_norm_batch", sg), 3, 12, sg,
        )?;
        let head_rms_norm = create_compute_pipeline(
            device, include_spv_sg!("head_rms_norm", sg), 2, 12, sg,
        )?;
        let softmax = create_compute_pipeline(
            device, include_spv_sg!("softmax", sg), 1, 4, sg,
        )?;
        let argmax = create_compute_pipeline(
            device, include_spv_sg!("argmax", sg), 2, 4, sg,
        )?;
        let f32_matvec = create_compute_pipeline(
            device, include_spv_sg!("f32_matvec", sg), 3, 4, sg,
        )?;
        let f32_matmul = create_compute_pipeline(
            device, include_spv_sg!("f32_matmul", sg), 3, 8, sg,
        )?;
        let attention_decode = create_compute_pipeline(
            device, include_spv_sg!("attention_decode", sg), 5, 24, sg,
        )?;
        let attention_prefill = create_compute_pipeline(
            device, include_spv_sg!("attention_prefill", sg), 5, 36, sg,
        )?;

        // ----- Subgroup-independent pipelines -----
        let residual_add = create_compute_pipeline(
            device, include_spv!("residual_add"), 2, 4, 0,
        )?;
        let embedding = create_compute_pipeline(
            device, include_spv!("embedding"), 3, 8, 0,
        )?;
        let swiglu = create_compute_pipeline(
            device, include_spv!("swiglu"), 2, 4, 0,
        )?;
        let rope_single = create_compute_pipeline(
            device, include_spv!("rope_single"), 3, 8, 0,
        )?;
        let rope_batch = create_compute_pipeline(
            device, include_spv!("rope_batch"), 3, 16, 0,
        )?;
        let copy_buffer = create_compute_pipeline(
            device, include_spv!("copy_buffer"), 2, 4, 0,
        )?;
        let kv_cache_append = create_compute_pipeline(
            device, include_spv!("kv_cache_append"), 2, 8, 0,
        )?;
        let kv_cache_append_batch = create_compute_pipeline(
            device, include_spv!("kv_cache_append_batch"), 2, 12, 0,
        )?;
        let bias_add_batch = create_compute_pipeline(
            device, include_spv!("bias_add_batch"), 2, 8, 0,
        )?;

        // ----- MoE pipelines (subgroup-independent) -----
        let scaled_add = create_compute_pipeline(
            device, include_spv!("scaled_add"), 2, 8, 0,
        )?;
        let moe_gather = create_compute_pipeline(
            device, include_spv!("moe_gather"), 3, 8, 0,
        )?;
        let moe_scatter_add = create_compute_pipeline(
            device, include_spv!("moe_scatter_add"), 4, 8, 0,
        )?;

        // ----- Int8 pipelines (subgroup-dependent) -----
        let int8_matvec = create_compute_pipeline(
            device, include_spv_sg!("int8_matvec", sg), 4, 4, sg,
        )?;
        let int8_matmul = create_compute_pipeline(
            device, include_spv_sg!("int8_matmul", sg), 4, 8, sg,
        )?;

        // ----- BF16 pipelines (subgroup-dependent) -----
        let bf16_matvec = create_compute_pipeline(
            device, include_spv_sg!("bf16_matvec", sg), 3, 4, sg,
        )?;
        let bf16_matmul = create_compute_pipeline(
            device, include_spv_sg!("bf16_matmul", sg), 3, 8, sg,
        )?;

        // ----- Q4 pipelines (subgroup-dependent) -----
        let q4_matvec = create_compute_pipeline(
            device, include_spv_sg!("q4_matvec", sg),
            4, // x, w_packed, scales, y
            8, // N (u32) + K (u32)
            sg,
        )?;
        let q4_matmul = create_compute_pipeline(
            device, include_spv_sg!("q4_matmul", sg),
            4, // A, W_packed, scales, C
            12, // M (u32) + N (u32) + K (u32)
            sg,
        )?;

        // ----- Q4 v5 optimized pipelines (subgroup-dependent) -----
        let q4_matvec_v5 = create_compute_pipeline(
            device, include_spv_sg!("q4_matvec_v5", sg), 4, 8, sg,
        )?;
        let q4_matmul_v5 = create_compute_pipeline(
            device, include_spv_sg!("q4_matmul_v5", sg), 4, 12, sg,
        )?;
        let use_q4_v5 = std::env::var("VULKAN_Q4_SHADER").as_deref() != Ok("v0");
        if use_q4_v5 {
            eprintln!("[vulkan] Q4 shader: v5 (vectorized K-group)");
        } else {
            eprintln!("[vulkan] Q4 shader: v0 (baseline)");
        }

        // Q4 cooperative matrix matmul: only if device supports it and SPIR-V compiled.
        // Set VULKAN_NO_COOPMAT=1 to force v5 for A/B comparison benchmarks.
        let no_coopmat = std::env::var("VULKAN_NO_COOPMAT").is_ok();
        let q4_matmul_coopmat = if has_coopmat && !no_coopmat {
            let spirv = include_spv_sg!("q4_matmul_coopmat", sg);
            if spirv.is_empty() {
                eprintln!("[vulkan] Q4 coopmat: SPIR-V not available (glslc too old?)");
                None
            } else {
                match create_compute_pipeline(
                    device, spirv,
                    4,  // A, W_packed, scales, C
                    12, // M + N + K
                    sg,
                ) {
                    Ok(p) => {
                        eprintln!("[vulkan] Q4 coopmat: pipeline created (prefill M>=16)");
                        Some(p)
                    }
                    Err(e) => {
                        eprintln!("[vulkan] Q4 coopmat: pipeline creation failed: {}", e);
                        None
                    }
                }
            }
        } else {
            if no_coopmat {
                eprintln!("[vulkan] Q4 coopmat: disabled by VULKAN_NO_COOPMAT");
            }
            None
        };

        // Q4 cooperative matrix INT8 WMMA: only on NVIDIA with coopmat support.
        let q4_matmul_coopmat_i8 = if has_coopmat && !no_coopmat && is_nvidia {
            let spirv = include_spv_sg!("q4_matmul_coopmat_i8", sg);
            if spirv.is_empty() {
                eprintln!("[vulkan] Q4 coopmat i8: SPIR-V not available");
                None
            } else {
                match create_compute_pipeline(
                    device, spirv,
                    4,  // A, W_packed, scales, C
                    12, // M + N + K
                    sg,
                ) {
                    Ok(p) => {
                        eprintln!("[vulkan] Q4 coopmat i8: pipeline created (NVIDIA INT8 WMMA)");
                        Some(p)
                    }
                    Err(e) => {
                        eprintln!("[vulkan] Q4 coopmat i8: pipeline creation failed: {}", e);
                        None
                    }
                }
            }
        } else {
            None
        };

        let pipeline_count = 31
            + if q4_matmul_coopmat.is_some() { 1 } else { 0 }
            + if q4_matmul_coopmat_i8.is_some() { 1 } else { 0 };
        eprintln!(
            "[vulkan] All {} pipelines created (subgroup={}, coopmat={})",
            pipeline_count, subgroup_size, has_coopmat
        );

        Ok(Self {
            rms_norm,
            rms_norm_batch,
            head_rms_norm,
            softmax,
            argmax,
            f32_matvec,
            f32_matmul,
            attention_decode,
            attention_prefill,
            residual_add,
            embedding,
            swiglu,
            rope_single,
            rope_batch,
            copy_buffer,
            kv_cache_append,
            kv_cache_append_batch,
            bias_add_batch,
            scaled_add,
            moe_gather,
            moe_scatter_add,
            int8_matvec,
            int8_matmul,
            bf16_matvec,
            bf16_matmul,
            q4_matvec,
            q4_matmul,
            q4_matvec_v5,
            q4_matmul_v5,
            use_q4_v5,
            q4_matmul_coopmat,
            q4_matmul_coopmat_i8,
        })
    }

    /// Destroy all pipeline Vulkan objects. Must be called before destroying the device.
    pub unsafe fn destroy(&self, device: &ash::Device) {
        destroy_compute_pipeline(device, &self.rms_norm);
        destroy_compute_pipeline(device, &self.rms_norm_batch);
        destroy_compute_pipeline(device, &self.head_rms_norm);
        destroy_compute_pipeline(device, &self.softmax);
        destroy_compute_pipeline(device, &self.argmax);
        destroy_compute_pipeline(device, &self.f32_matvec);
        destroy_compute_pipeline(device, &self.f32_matmul);
        destroy_compute_pipeline(device, &self.attention_decode);
        destroy_compute_pipeline(device, &self.attention_prefill);
        destroy_compute_pipeline(device, &self.residual_add);
        destroy_compute_pipeline(device, &self.embedding);
        destroy_compute_pipeline(device, &self.swiglu);
        destroy_compute_pipeline(device, &self.rope_single);
        destroy_compute_pipeline(device, &self.rope_batch);
        destroy_compute_pipeline(device, &self.copy_buffer);
        destroy_compute_pipeline(device, &self.kv_cache_append);
        destroy_compute_pipeline(device, &self.kv_cache_append_batch);
        destroy_compute_pipeline(device, &self.bias_add_batch);
        destroy_compute_pipeline(device, &self.scaled_add);
        destroy_compute_pipeline(device, &self.moe_gather);
        destroy_compute_pipeline(device, &self.moe_scatter_add);
        destroy_compute_pipeline(device, &self.int8_matvec);
        destroy_compute_pipeline(device, &self.int8_matmul);
        destroy_compute_pipeline(device, &self.bf16_matvec);
        destroy_compute_pipeline(device, &self.bf16_matmul);
        destroy_compute_pipeline(device, &self.q4_matvec);
        destroy_compute_pipeline(device, &self.q4_matmul);
        destroy_compute_pipeline(device, &self.q4_matvec_v5);
        destroy_compute_pipeline(device, &self.q4_matmul_v5);
        if let Some(ref p) = self.q4_matmul_coopmat {
            destroy_compute_pipeline(device, p);
        }
        if let Some(ref p) = self.q4_matmul_coopmat_i8 {
            destroy_compute_pipeline(device, p);
        }
    }
}
