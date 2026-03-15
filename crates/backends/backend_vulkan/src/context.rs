//! Vulkan device context: instance, physical device, logical device, queues.
//!
//! Vulkan 1.3, runtime subgroup size selection (32 or 64) + cooperative matrix detection.

use ash::khr;
use ash::vk;
use herbert_core::error::{HerbertError, Result};

use crate::shaders::Pipelines;

const VENDOR_NVIDIA: u32 = 0x10DE;
const VENDOR_AMD: u32 = 0x1002;

/// Holds the Vulkan instance, device, queue, and capability flags.
///
/// All other GPU modules (memory, shaders, compute) depend on this context.
pub struct VulkanContext {
    // Pipelines are cleaned up explicitly in our Drop impl (via `pipelines.destroy()`)
    // before the device is destroyed -- no implicit Drop ordering concerns here.
    pub pipelines: Pipelines,
    pub instance: ash::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device: ash::Device,
    pub queue: vk::Queue,
    pub command_pool: vk::CommandPool,
    pub queue_family_index: u32,
    pub push_descriptor: khr::push_descriptor::Device,
    pub subgroup_size: u32,  // 32 (NVIDIA) or 64 (AMD RDNA3)
    pub has_coopmat: bool,   // VK_KHR_cooperative_matrix available
    pub is_nvidia: bool,     // vendor 0x10DE
    pub gpu_index: usize,    // physical device index passed to new()
    _entry: ash::Entry,
}

impl VulkanContext {
    /// Create a new Vulkan context with compute queue and runtime capability detection.
    ///
    /// Detects GPU vendor and capabilities at runtime:
    /// - Subgroup size: 32 (NVIDIA) or 64 (AMD RDNA3 with wave64 support)
    /// - Cooperative matrix support (VK_KHR_cooperative_matrix)
    ///
    /// `device_index` selects GPU:
    /// - 0..N: selects the Nth discrete GPU (0-based)
    /// - 1000+: selects from all physical devices (global index = device_index - 1000)
    pub fn new(device_index: usize) -> Result<Self> {
        // Load the Vulkan library
        let entry = unsafe { ash::Entry::load() }
            .map_err(|e| HerbertError::Backend(format!("Failed to load Vulkan: {:?}", e)))?;

        // Create Vulkan 1.3 instance
        let app_info =
            vk::ApplicationInfo::default().api_version(vk::make_api_version(0, 1, 3, 0));
        let instance_info = vk::InstanceCreateInfo::default().application_info(&app_info);
        let instance = unsafe { entry.create_instance(&instance_info, None) }
            .map_err(|e| HerbertError::Backend(format!("Failed to create Vulkan instance: {:?}", e)))?;

        // Enumerate physical devices and filter to discrete GPUs
        let physical_devices = unsafe { instance.enumerate_physical_devices() }
            .map_err(|e| HerbertError::Backend(format!("Failed to enumerate devices: {:?}", e)))?;
        if physical_devices.is_empty() {
            return Err(HerbertError::Backend("No Vulkan GPU found".into()));
        }

        let discrete_gpus: Vec<vk::PhysicalDevice> = physical_devices
            .iter()
            .filter(|&&pd| {
                let props = unsafe { instance.get_physical_device_properties(pd) };
                props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU
            })
            .copied()
            .collect();

        // Select physical device based on device_index convention:
        // - 0..N: index into discrete GPUs
        // - 1000+: index into all physical devices (global index = device_index - 1000)
        let (physical_device, gpu_index_display) = if device_index >= 1000 {
            let global_idx = device_index - 1000;
            if global_idx < physical_devices.len() {
                (physical_devices[global_idx], global_idx)
            } else {
                eprintln!(
                    "[vulkan] device_index {} maps to global index {} >= {} physical devices, falling back to GPU 0",
                    device_index, global_idx, physical_devices.len()
                );
                (physical_devices[0], 0)
            }
        } else if discrete_gpus.is_empty() {
            (physical_devices[0], 0)
        } else if device_index < discrete_gpus.len() {
            (discrete_gpus[device_index], device_index)
        } else {
            eprintln!(
                "[vulkan] device_index {} >= {} discrete GPUs, falling back to discrete GPU 0",
                device_index,
                discrete_gpus.len()
            );
            (discrete_gpus[0], 0)
        };

        // Query device name and vendor
        let dev_props = unsafe { instance.get_physical_device_properties(physical_device) };
        let dev_name = unsafe { std::ffi::CStr::from_ptr(dev_props.device_name.as_ptr()) }
            .to_string_lossy();
        let vendor_id = dev_props.vendor_id;
        let is_nvidia = vendor_id == VENDOR_NVIDIA;
        let is_amd = vendor_id == VENDOR_AMD;
        let vendor_str = if is_nvidia {
            "NVIDIA"
        } else if is_amd {
            "AMD"
        } else {
            "other"
        };
        eprintln!(
            "[vulkan] GPU[{}]: {} (vendor: {})",
            gpu_index_display, dev_name, vendor_str
        );

        // Query subgroup size control properties
        let mut subgroup_props = vk::PhysicalDeviceSubgroupSizeControlProperties::default();
        let mut props2 =
            vk::PhysicalDeviceProperties2::default().push_next(&mut subgroup_props);
        unsafe {
            instance.get_physical_device_properties2(physical_device, &mut props2);
        }

        let max_subgroup_size = subgroup_props.max_subgroup_size;
        eprintln!(
            "[vulkan] Subgroup size: min={}, max={}, required stages={:?}",
            subgroup_props.min_subgroup_size,
            max_subgroup_size,
            subgroup_props.required_subgroup_size_stages
        );

        if max_subgroup_size < 32 {
            return Err(HerbertError::Backend(format!(
                "GPU does not support subgroup size 32 (maxSubgroupSize={}).",
                max_subgroup_size
            )));
        }

        // Choose subgroup size: AMD with wave64 support -> 64, else -> 32
        let subgroup_size = if is_amd && max_subgroup_size >= 64 {
            64
        } else {
            32
        };
        eprintln!("[vulkan] Selected subgroup size: {}", subgroup_size);

        // Check for VK_KHR_cooperative_matrix extension
        let device_extensions = unsafe {
            instance.enumerate_device_extension_properties(physical_device)
        }
        .unwrap_or_default();

        let coopmat_ext_name =
            std::ffi::CStr::from_bytes_with_nul(b"VK_KHR_cooperative_matrix\0").unwrap();
        let has_coopmat = device_extensions.iter().any(|ext| {
            let name = unsafe { std::ffi::CStr::from_ptr(ext.extension_name.as_ptr()) };
            name == coopmat_ext_name
        });
        eprintln!(
            "[vulkan] VK_KHR_cooperative_matrix: {}",
            if has_coopmat { "yes" } else { "no" }
        );

        // Find a compute queue family
        let queue_families = unsafe {
            instance.get_physical_device_queue_family_properties(physical_device)
        };
        let queue_family_index = queue_families
            .iter()
            .position(|qf| qf.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .ok_or_else(|| HerbertError::Backend("No compute queue family found".into()))?
            as u32;

        // Prepare device features
        let mut storage_8bit = vk::PhysicalDevice8BitStorageFeatures::default()
            .storage_buffer8_bit_access(true);
        let mut subgroup_size_ctrl = vk::PhysicalDeviceSubgroupSizeControlFeatures::default()
            .subgroup_size_control(true)
            .compute_full_subgroups(true);

        let queue_priorities = [1.0f32];
        let queue_create_infos = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&queue_priorities)];

        // Build extension list: always push_descriptor, optionally cooperative_matrix
        let mut extension_names_vec: Vec<*const std::ffi::c_char> =
            vec![khr::push_descriptor::NAME.as_ptr()];
        if has_coopmat {
            extension_names_vec.push(coopmat_ext_name.as_ptr());
        }

        let mut coopmat_features = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default()
            .cooperative_matrix(true);

        let mut device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&extension_names_vec)
            .push_next(&mut storage_8bit)
            .push_next(&mut subgroup_size_ctrl);

        if has_coopmat {
            device_create_info = device_create_info.push_next(&mut coopmat_features);
        }

        // Create logical device
        let device = unsafe {
            instance.create_device(physical_device, &device_create_info, None)
        }
        .map_err(|e| HerbertError::Backend(format!("Failed to create Vulkan device: {:?}", e)))?;

        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        // Create command pool with reset-command-buffer flag
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool = unsafe { device.create_command_pool(&pool_info, None) }
            .map_err(|e| HerbertError::Backend(format!("Failed to create command pool: {:?}", e)))?;

        // Create push descriptor extension device
        let push_descriptor = khr::push_descriptor::Device::new(&instance, &device);

        // Create all compute pipelines
        let pipelines = Pipelines::new(&device, subgroup_size, has_coopmat, is_nvidia)?;

        eprintln!(
            "[vulkan] Vulkan 1.3 initialized, subgroup={}, coopmat={}",
            subgroup_size, has_coopmat
        );

        Ok(Self {
            pipelines,
            instance,
            physical_device,
            device,
            queue,
            command_pool,
            queue_family_index,
            push_descriptor,
            subgroup_size,
            has_coopmat,
            is_nvidia,
            gpu_index: gpu_index_display,
            _entry: entry,
        })
    }

    // ========================================================================
    // Command buffer helpers
    // ========================================================================

    /// Allocate a primary command buffer and begin recording (ONE_TIME_SUBMIT).
    pub fn cmd_begin(&self) -> Result<vk::CommandBuffer> {
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let cb = unsafe { self.device.allocate_command_buffers(&alloc_info) }
            .map_err(|e| {
                HerbertError::Backend(format!("Failed to allocate command buffer: {:?}", e))
            })?[0];

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { self.device.begin_command_buffer(cb, &begin_info) }
            .map_err(|e| {
                HerbertError::Backend(format!("Failed to begin command buffer: {:?}", e))
            })?;

        Ok(cb)
    }

    /// End recording, submit to the compute queue, wait for completion, and free
    /// the command buffer.
    pub fn cmd_end_submit_wait(&self, cb: vk::CommandBuffer) -> Result<()> {
        unsafe {
            self.device
                .end_command_buffer(cb)
                .map_err(|e| {
                    HerbertError::Backend(format!("Failed to end command buffer: {:?}", e))
                })?;

            let cbs = [cb];
            let submit_info = vk::SubmitInfo::default().command_buffers(&cbs);
            self.device
                .queue_submit(self.queue, &[submit_info], vk::Fence::null())
                .map_err(|e| {
                    HerbertError::Backend(format!("Failed to submit command buffer: {:?}", e))
                })?;

            self.device
                .queue_wait_idle(self.queue)
                .map_err(|e| {
                    HerbertError::Backend(format!("Failed to wait for queue idle: {:?}", e))
                })?;

            self.device
                .free_command_buffers(self.command_pool, &[cb]);
        }
        Ok(())
    }

    // ========================================================================
    // Pipeline barrier helpers
    // ========================================================================

    /// Insert a compute-to-compute memory barrier (shader write -> shader read).
    pub fn cmd_pipeline_barrier(&self, cb: vk::CommandBuffer) {
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ);
        unsafe {
            self.device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
        }
    }

    /// Insert a compute-to-transfer memory barrier (shader write -> transfer read).
    pub fn cmd_compute_to_transfer_barrier(&self, cb: vk::CommandBuffer) {
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
        unsafe {
            self.device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
        }
    }

    // ========================================================================
    // Dispatch helpers
    // ========================================================================

    /// Push storage buffer descriptors for a compute pipeline.
    ///
    /// Each entry in `buffers` is `(vk::Buffer, size_in_bytes)`.
    /// Bindings are assigned sequentially starting at binding 0.
    /// A size of 0 means `VK_WHOLE_SIZE`.
    pub fn cmd_push_descriptors(
        &self,
        cb: vk::CommandBuffer,
        layout: vk::PipelineLayout,
        buffers: &[(vk::Buffer, u64)],
    ) {
        let buffer_infos: Vec<vk::DescriptorBufferInfo> = buffers
            .iter()
            .map(|(buf, size)| {
                vk::DescriptorBufferInfo::default()
                    .buffer(*buf)
                    .offset(0)
                    .range(if *size == 0 { vk::WHOLE_SIZE } else { *size })
            })
            .collect();

        let writes: Vec<vk::WriteDescriptorSet> = buffer_infos
            .iter()
            .enumerate()
            .map(|(i, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_binding(i as u32)
                    .descriptor_count(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(info))
            })
            .collect();

        unsafe {
            self.push_descriptor.cmd_push_descriptor_set(
                cb,
                vk::PipelineBindPoint::COMPUTE,
                layout,
                0,
                &writes,
            );
        }
    }

    /// Push constants to a compute pipeline.
    pub fn cmd_push_constants(
        &self,
        cb: vk::CommandBuffer,
        layout: vk::PipelineLayout,
        data: &[u8],
    ) {
        unsafe {
            self.device.cmd_push_constants(
                cb,
                layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                data,
            );
        }
    }

    /// Bind pipeline, push descriptors, push constants, and dispatch compute work groups.
    pub fn cmd_dispatch(
        &self,
        cb: vk::CommandBuffer,
        pipeline: vk::Pipeline,
        layout: vk::PipelineLayout,
        buffers: &[(vk::Buffer, u64)],
        push_constants: &[u8],
        group_count_x: u32,
        group_count_y: u32,
        group_count_z: u32,
    ) {
        unsafe {
            self.device
                .cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pipeline);
        }
        self.cmd_push_descriptors(cb, layout, buffers);
        if !push_constants.is_empty() {
            self.cmd_push_constants(cb, layout, push_constants);
        }
        unsafe {
            self.device
                .cmd_dispatch(cb, group_count_x, group_count_y, group_count_z);
        }
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().ok();
            // Destroy pipelines before the device (they hold Vulkan pipeline handles)
            self.pipelines.destroy(&self.device);
            self.device
                .destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}
