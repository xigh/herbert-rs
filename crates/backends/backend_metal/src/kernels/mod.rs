//! Metal kernel dispatch wrappers.
//!
//! Each function sets the compute pipeline state, binds buffers, sets push
//! constants as raw bytes, and dispatches the threadgroups.

pub mod norm;
pub mod activation;
pub mod rope;
pub mod matvec;
pub mod matmul;
pub mod attention;
pub mod moe;
pub mod vision;

use std::ptr::NonNull;
use objc2::runtime::ProtocolObject;
use objc2_metal::*;

use crate::memory::MetalBuffer;
use crate::shaders::ComputePipeline;

/// Helper: round up `n / d` (ceiling division).
#[inline]
pub fn div_ceil(n: u32, d: u32) -> u32 {
    (n + d - 1) / d
}

/// Bind pipeline, buffers, and optional push-constant bytes, then dispatch.
///
/// Buffers are bound at indices 0..buffers.len().
/// If `params` is non-empty it is bound as raw bytes at index `buffers.len()`.
pub fn dispatch_kernel(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &ComputePipeline,
    buffers: &[&MetalBuffer],
    params: &[u8],
    grid: MTLSize,
    threads: MTLSize,
) {
    encoder.setComputePipelineState(&pipeline.state);

    for (i, buf) in buffers.iter().enumerate() {
        unsafe { encoder.setBuffer_offset_atIndex(Some(&buf.buffer), buf.offset, i) };
    }

    if !params.is_empty() {
        // SAFETY: params is a stack-allocated byte slice; the pointer is valid
        // for the duration of this call (Metal copies the data immediately).
        let ptr = NonNull::new(params.as_ptr() as *mut std::ffi::c_void).unwrap();
        unsafe {
            encoder.setBytes_length_atIndex(ptr, params.len(), buffers.len());
        }
    }

    encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, threads);
}

/// Like `dispatch_kernel`, but also sets dynamic threadgroup memory at index 0.
/// This allows shaders to use `threadgroup float* shared_x [[threadgroup(0)]]`
/// instead of static `threadgroup float shared_x[8192]`, enabling the GPU to
/// run more threadgroups per core when the actual K is smaller than 8192.
pub fn dispatch_kernel_with_tgmem(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &ComputePipeline,
    buffers: &[&MetalBuffer],
    params: &[u8],
    tgmem_bytes: usize,
    grid: MTLSize,
    threads: MTLSize,
) {
    encoder.setComputePipelineState(&pipeline.state);

    for (i, buf) in buffers.iter().enumerate() {
        unsafe { encoder.setBuffer_offset_atIndex(Some(&buf.buffer), buf.offset, i) };
    }

    if !params.is_empty() {
        let ptr = NonNull::new(params.as_ptr() as *mut std::ffi::c_void).unwrap();
        unsafe {
            encoder.setBytes_length_atIndex(ptr, params.len(), buffers.len());
        }
    }

    unsafe {
        encoder.setThreadgroupMemoryLength_atIndex(tgmem_bytes, 0);
    }

    encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, threads);
}

/// Like `dispatch_kernel`, but reads the threadgroup grid from a GPU buffer
/// (Metal indirect dispatch). The grid dimensions are 3 × u32 at `indirect_offset`
/// bytes into `indirect_buffer`.
pub fn dispatch_kernel_indirect(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &ComputePipeline,
    buffers: &[&MetalBuffer],
    params: &[u8],
    indirect_buffer: &MetalBuffer,
    indirect_offset: usize,
    threads: MTLSize,
) {
    encoder.setComputePipelineState(&pipeline.state);

    for (i, buf) in buffers.iter().enumerate() {
        unsafe { encoder.setBuffer_offset_atIndex(Some(&buf.buffer), buf.offset, i) };
    }

    if !params.is_empty() {
        let ptr = NonNull::new(params.as_ptr() as *mut std::ffi::c_void).unwrap();
        unsafe {
            encoder.setBytes_length_atIndex(ptr, params.len(), buffers.len());
        }
    }

    unsafe {
        encoder.dispatchThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerThreadgroup(
            &indirect_buffer.buffer,
            indirect_offset,
            threads,
        );
    }
}
