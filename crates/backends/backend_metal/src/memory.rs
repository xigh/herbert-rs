//! Metal buffer: unified memory model using StorageModeShared.
//!
//! Apple Silicon has unified memory — CPU and GPU share the same physical
//! memory, so there's no need for staging buffers or device-local transfers.

use herbert_core::error::{HerbertError, Result};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::*;

/// A Metal buffer with shared CPU/GPU access.
///
/// Supports sub-buffer views via `slice()`: the view shares the same underlying
/// Metal buffer (refcounted) but binds at a byte offset for GPU access.
pub struct MetalBuffer {
    pub buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub size: u64,
    /// Byte offset for sub-buffer views (0 for root buffers).
    pub offset: usize,
}

// SAFETY: MetalBuffer holds a reference-counted Metal buffer handle.
// Metal buffers are thread-safe when not written to concurrently.
// Our backend ensures sequential command buffer submission.
unsafe impl Send for MetalBuffer {}
unsafe impl Sync for MetalBuffer {}

impl MetalBuffer {
    fn allocate(
        device: &ProtocolObject<dyn MTLDevice>,
        size: u64,
        zero_init: bool,
    ) -> Result<Self> {
        let buffer = device.newBufferWithLength_options(
            size as usize,
            MTLResourceOptions::StorageModeShared,
        ).ok_or_else(|| HerbertError::Backend(
            format!("Failed to allocate Metal buffer of {} bytes", size),
        ))?;

        if zero_init {
            unsafe {
                let ptr = buffer.contents().as_ptr() as *mut u8;
                std::ptr::write_bytes(ptr, 0, size as usize);
            }
        }

        Ok(Self { buffer, size, offset: 0 })
    }

    /// Create a new zero-initialized shared buffer.
    pub fn new(device: &ProtocolObject<dyn MTLDevice>, size: u64) -> Result<Self> {
        Self::allocate(device, size, true)
    }

    /// Create a new shared buffer without clearing it on the CPU.
    ///
    /// Use only for scratch/output buffers that are fully overwritten by the
    /// caller before any read.
    pub fn new_uninit(device: &ProtocolObject<dyn MTLDevice>, size: u64) -> Result<Self> {
        Self::allocate(device, size, false)
    }

    /// Create a buffer from raw bytes (memcpy into shared memory).
    pub fn from_data(device: &ProtocolObject<dyn MTLDevice>, data: &[u8]) -> Result<Self> {
        let size = data.len() as u64;
        let buffer = Self::new_uninit(device, size)?;
        unsafe {
            let ptr = buffer.buffer.contents().as_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        }
        Ok(buffer)
    }

    /// Create a zero-copy buffer wrapping page-aligned mmap'd memory.
    ///
    /// The memory is NOT copied — the Metal buffer points directly into the mmap.
    /// The caller must ensure the mmap outlives all Metal buffers created from it.
    ///
    /// # Safety
    /// - `ptr` must be page-aligned (4096 bytes)
    /// - The memory at `ptr..ptr+size` must remain valid for the lifetime of the buffer
    /// - `deallocator` must be None (we manage the mmap lifetime externally)
    pub unsafe fn from_ptr_nocopy(
        device: &ProtocolObject<dyn MTLDevice>,
        ptr: *mut u8,
        size: usize,
    ) -> Result<Self> {
        use std::ptr::NonNull;
        use std::ffi::c_void;

        if size == 0 {
            return Self::new(device, 4); // Metal doesn't allow zero-size NoCopy
        }

        let nn = NonNull::new(ptr as *mut c_void).ok_or_else(|| {
            HerbertError::Backend("Null pointer for NoCopy buffer".into())
        })?;

        let buffer = device.newBufferWithBytesNoCopy_length_options_deallocator(
            nn,
            size,
            MTLResourceOptions::StorageModeShared,
            None, // We manage the mmap lifetime externally
        ).ok_or_else(|| {
            HerbertError::Backend(format!(
                "Failed to create NoCopy Metal buffer (size={}, ptr={:?}, page_aligned={})",
                size, ptr, (ptr as usize) % 4096 == 0,
            ))
        })?;

        Ok(Self { buffer, size: size as u64, offset: 0 })
    }

    /// Create a sub-buffer view sharing the same underlying Metal buffer.
    ///
    /// The view has its own offset and logical size but references the same
    /// GPU memory. Uses reference counting so the Metal buffer stays alive
    /// as long as any view exists.
    pub fn slice(&self, byte_offset: usize, byte_size: usize) -> Self {
        Self {
            buffer: self.buffer.clone(),
            size: byte_size as u64,
            offset: self.offset + byte_offset,
        }
    }

    /// Create a buffer from f32 slice.
    pub fn from_f32(device: &ProtocolObject<dyn MTLDevice>, data: &[f32]) -> Result<Self> {
        let bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
        };
        Self::from_data(device, bytes)
    }

    /// Get raw pointer to buffer contents (CPU-accessible).
    ///
    /// Only valid for root buffers (offset == 0). Sub-buffer views should
    /// only be used for GPU-side binding via `dispatch_kernel`.
    pub fn contents_ptr(&self) -> *mut u8 {
        debug_assert!(self.offset == 0, "contents_ptr called on sub-buffer view (offset={})", self.offset);
        self.buffer.contents().as_ptr() as *mut u8
    }

    /// Read f32 values from the buffer (zero-copy on unified memory).
    pub fn read_f32(&self, count: usize) -> Vec<f32> {
        debug_assert!(self.offset == 0, "read_f32 called on sub-buffer view (offset={})", self.offset);
        let mut result = vec![0.0f32; count];
        unsafe {
            let src = self.buffer.contents().as_ptr() as *const f32;
            std::ptr::copy_nonoverlapping(src, result.as_mut_ptr(), count);
        }
        result
    }

    /// Read a single u32 value at the start of the buffer.
    pub fn read_u32(&self) -> u32 {
        debug_assert!(self.offset == 0, "read_u32 called on sub-buffer view (offset={})", self.offset);
        unsafe {
            let ptr = self.buffer.contents().as_ptr() as *const u32;
            std::ptr::read(ptr)
        }
    }

    /// Read multiple u32 values from the buffer.
    pub fn read_u32_vec(&self, count: usize) -> Vec<u32> {
        debug_assert!(self.offset == 0, "read_u32_vec called on sub-buffer view (offset={})", self.offset);
        let mut result = vec![0u32; count];
        unsafe {
            let src = self.buffer.contents().as_ptr() as *const u32;
            std::ptr::copy_nonoverlapping(src, result.as_mut_ptr(), count);
        }
        result
    }

    /// Write a single u32 value at the start of the buffer.
    pub fn write_u32(&self, val: u32) {
        debug_assert!(self.offset == 0, "write_u32 called on sub-buffer view (offset={})", self.offset);
        unsafe {
            let ptr = self.buffer.contents().as_ptr() as *mut u32;
            std::ptr::write(ptr, val);
        }
    }

    /// Write raw bytes into the buffer.
    pub fn write_bytes(&self, data: &[u8]) {
        debug_assert!(self.offset == 0, "write_bytes called on sub-buffer view (offset={})", self.offset);
        assert!(data.len() as u64 <= self.size, "write_bytes: data exceeds buffer size");
        unsafe {
            let ptr = self.buffer.contents().as_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        }
    }

    /// Zero-fill the buffer.
    pub fn zero_fill(&self) {
        debug_assert!(self.offset == 0, "zero_fill called on sub-buffer view (offset={})", self.offset);
        unsafe {
            let ptr = self.buffer.contents().as_ptr() as *mut u8;
            std::ptr::write_bytes(ptr, 0, self.size as usize);
        }
    }
}
