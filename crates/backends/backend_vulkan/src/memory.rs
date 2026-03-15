//! GPU memory management: device-local buffers, staging, transfers.

use ash::vk;

use crate::context::VulkanContext;
use herbert_core::error::{HerbertError, Result};

/// A Vulkan buffer with its associated device memory.
///
/// Owns both the `vk::Buffer` and `vk::DeviceMemory` handles.
/// Automatically destroys them on drop via the cloned device handle.
pub struct VulkanBuffer {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size: u64,
    device: ash::Device, // clone of device handle for Drop
}

impl VulkanBuffer {
    /// Allocate a device-local buffer (GPU-only, not CPU accessible).
    pub fn device_local(
        ctx: &VulkanContext,
        size: u64,
        usage: vk::BufferUsageFlags,
    ) -> Result<Self> {
        let usage = usage
            | vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::TRANSFER_SRC;

        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe { ctx.device.create_buffer(&buffer_info, None) }
            .map_err(|e| HerbertError::Backend(format!("Failed to create device buffer: {:?}", e)))?;

        let mem_reqs = unsafe { ctx.device.get_buffer_memory_requirements(buffer) };
        let mem_type_index = find_memory_type(
            ctx,
            mem_reqs.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_reqs.size)
            .memory_type_index(mem_type_index);

        let memory = unsafe { ctx.device.allocate_memory(&alloc_info, None) }
            .map_err(|e| HerbertError::Backend(format!("Failed to allocate device memory: {:?}", e)))?;

        unsafe { ctx.device.bind_buffer_memory(buffer, memory, 0) }
            .map_err(|e| HerbertError::Backend(format!("Failed to bind buffer memory: {:?}", e)))?;

        Ok(Self {
            buffer,
            memory,
            size,
            device: ctx.device.clone(),
        })
    }

    /// Allocate a host-visible, host-coherent buffer (CPU accessible for staging).
    pub fn host_visible(
        ctx: &VulkanContext,
        size: u64,
        usage: vk::BufferUsageFlags,
    ) -> Result<Self> {
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe { ctx.device.create_buffer(&buffer_info, None) }
            .map_err(|e| HerbertError::Backend(format!("Failed to create host buffer: {:?}", e)))?;

        let mem_reqs = unsafe { ctx.device.get_buffer_memory_requirements(buffer) };
        let mem_type_index = find_memory_type(
            ctx,
            mem_reqs.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_reqs.size)
            .memory_type_index(mem_type_index);

        let memory = unsafe { ctx.device.allocate_memory(&alloc_info, None) }
            .map_err(|e| HerbertError::Backend(format!("Failed to allocate host memory: {:?}", e)))?;

        unsafe { ctx.device.bind_buffer_memory(buffer, memory, 0) }
            .map_err(|e| HerbertError::Backend(format!("Failed to bind buffer memory: {:?}", e)))?;

        Ok(Self {
            buffer,
            memory,
            size,
            device: ctx.device.clone(),
        })
    }

    /// Upload `data` to a new device-local buffer via a staging buffer.
    pub fn upload_to_device_local(
        ctx: &VulkanContext,
        data: &[u8],
        usage: vk::BufferUsageFlags,
    ) -> Result<Self> {
        let size = data.len() as u64;
        if size == 0 {
            return Self::device_local(ctx, 4, usage);
        }

        let staging = Self::host_visible(ctx, size, vk::BufferUsageFlags::TRANSFER_SRC)?;
        unsafe {
            let ptr = ctx
                .device
                .map_memory(staging.memory, 0, size, vk::MemoryMapFlags::empty())
                .map_err(|e| {
                    HerbertError::Backend(format!("Failed to map staging memory: {:?}", e))
                })?;
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut u8, data.len());
            ctx.device.unmap_memory(staging.memory);
        }

        let dst = Self::device_local(ctx, size, usage)?;

        let cb = ctx.cmd_begin()?;
        let region = vk::BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size,
        };
        unsafe {
            ctx.device
                .cmd_copy_buffer(cb, staging.buffer, dst.buffer, &[region]);
        }
        ctx.cmd_end_submit_wait(cb)?;

        drop(staging);

        Ok(dst)
    }

    /// Create a device-local buffer filled with zeros.
    pub fn device_local_zeroed(
        ctx: &VulkanContext,
        size: u64,
        usage: vk::BufferUsageFlags,
    ) -> Result<Self> {
        let actual_size = if size == 0 { 4 } else { size };
        let dst = Self::device_local(ctx, actual_size, usage)?;

        let cb = ctx.cmd_begin()?;
        unsafe {
            ctx.device
                .cmd_fill_buffer(cb, dst.buffer, 0, vk::WHOLE_SIZE, 0);
        }
        ctx.cmd_end_submit_wait(cb)?;

        Ok(dst)
    }

    /// Create a host-visible buffer and directly write `data` into it via map_memory.
    pub fn host_visible_with_data(
        ctx: &VulkanContext,
        data: &[u8],
        usage: vk::BufferUsageFlags,
    ) -> Result<Self> {
        let size = if data.is_empty() {
            4u64
        } else {
            data.len() as u64
        };

        let buf = Self::host_visible(ctx, size, usage)?;

        if !data.is_empty() {
            unsafe {
                let ptr = ctx
                    .device
                    .map_memory(buf.memory, 0, size, vk::MemoryMapFlags::empty())
                    .map_err(|e| {
                        HerbertError::Backend(format!("Failed to map memory: {:?}", e))
                    })?;
                std::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut u8, data.len());
                ctx.device.unmap_memory(buf.memory);
            }
        }

        Ok(buf)
    }

    /// Read back `byte_count` raw bytes from a device-local buffer.
    pub fn read_bytes(&self, ctx: &VulkanContext, byte_count: usize) -> Result<Vec<u8>> {
        let size = byte_count as u64;

        let staging =
            Self::host_visible(ctx, size, vk::BufferUsageFlags::TRANSFER_DST)?;

        let cb = ctx.cmd_begin()?;
        let region = vk::BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size,
        };
        unsafe {
            ctx.device
                .cmd_copy_buffer(cb, self.buffer, staging.buffer, &[region]);
        }
        ctx.cmd_end_submit_wait(cb)?;

        let ptr = unsafe {
            ctx.device
                .map_memory(staging.memory, 0, size, vk::MemoryMapFlags::empty())
        }
        .map_err(|e| {
            HerbertError::Backend(format!("Failed to map staging for read: {:?}", e))
        })?;

        let slice = unsafe { std::slice::from_raw_parts(ptr as *const u8, byte_count) };
        let result = slice.to_vec();

        unsafe {
            ctx.device.unmap_memory(staging.memory);
        }

        drop(staging);

        Ok(result)
    }

    /// Read back `count` f32 values from a device-local buffer.
    pub fn read_f32(&self, ctx: &VulkanContext, count: usize) -> Result<Vec<f32>> {
        let size = (count * std::mem::size_of::<f32>()) as u64;

        let staging =
            Self::host_visible(ctx, size, vk::BufferUsageFlags::TRANSFER_DST)?;

        let cb = ctx.cmd_begin()?;
        let region = vk::BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size,
        };
        unsafe {
            ctx.device
                .cmd_copy_buffer(cb, self.buffer, staging.buffer, &[region]);
        }
        ctx.cmd_end_submit_wait(cb)?;

        let ptr = unsafe {
            ctx.device
                .map_memory(staging.memory, 0, size, vk::MemoryMapFlags::empty())
        }
        .map_err(|e| {
            HerbertError::Backend(format!("Failed to map staging for read: {:?}", e))
        })?;

        let slice = unsafe { std::slice::from_raw_parts(ptr as *const f32, count) };
        let result = slice.to_vec();

        unsafe {
            ctx.device.unmap_memory(staging.memory);
        }

        drop(staging);

        Ok(result)
    }

    /// Read back `count` u32 values from a device-local buffer.
    pub fn read_u32(&self, ctx: &VulkanContext, count: usize) -> Result<Vec<u32>> {
        let size = (count * std::mem::size_of::<u32>()) as u64;

        let staging =
            Self::host_visible(ctx, size, vk::BufferUsageFlags::TRANSFER_DST)?;

        let cb = ctx.cmd_begin()?;
        let region = vk::BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size,
        };
        unsafe {
            ctx.device
                .cmd_copy_buffer(cb, self.buffer, staging.buffer, &[region]);
        }
        ctx.cmd_end_submit_wait(cb)?;

        let ptr = unsafe {
            ctx.device
                .map_memory(staging.memory, 0, size, vk::MemoryMapFlags::empty())
        }
        .map_err(|e| {
            HerbertError::Backend(format!("Failed to map staging for read: {:?}", e))
        })?;

        let slice = unsafe { std::slice::from_raw_parts(ptr as *const u32, count) };
        let result = slice.to_vec();

        unsafe {
            ctx.device.unmap_memory(staging.memory);
        }

        drop(staging);

        Ok(result)
    }
}

impl Drop for VulkanBuffer {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_buffer(self.buffer, None);
            self.device.free_memory(self.memory, None);
        }
    }
}

/// Find a suitable memory type index.
fn find_memory_type(
    ctx: &VulkanContext,
    type_bits: u32,
    properties: vk::MemoryPropertyFlags,
) -> Result<u32> {
    let mem_props = unsafe {
        ctx.instance
            .get_physical_device_memory_properties(ctx.physical_device)
    };
    for i in 0..mem_props.memory_type_count {
        if (type_bits & (1 << i)) != 0
            && (mem_props.memory_types[i as usize].property_flags & properties) == properties
        {
            return Ok(i);
        }
    }
    Err(HerbertError::Backend(format!(
        "Failed to find memory type with properties {:?}",
        properties
    )))
}
