//! Metal GPU context: device, command queue, and compute pipelines.

use herbert_core::error::{HerbertError, Result};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::*;

use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use crate::shaders::Pipelines;

/// Maximum time to wait for a single command buffer to complete.
/// If exceeded, we return an error instead of blocking forever.
/// 30s is generous — individual CBs should complete in <15s with LAYERS_PER_CB=4.
const CB_TIMEOUT: Duration = Duration::from_secs(30);

/// GPU family detected at runtime. Used to gate Metal 4 features.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum GpuFamily {
    /// Pre-Apple7 — simdgroup_matrix NOT available.
    Unknown,
    /// M1/M1 Pro/M1 Max — simdgroup_matrix 8×8 available.
    Apple7,
    /// M2 family.
    Apple8,
    /// M3 family — dynamic caching.
    Apple9,
    /// M4/M5 family.
    Apple10,
}

impl GpuFamily {
    /// True if this GPU supports simdgroup_matrix (Apple7+).
    pub fn supports_simdgroup_matrix(&self) -> bool {
        *self >= GpuFamily::Apple7
    }
}

/// GPU counter sampling state for METAL_PROFILE=3.
pub struct GpuCounters {
    pub sample_buffer: Retained<ProtocolObject<dyn MTLCounterSampleBuffer>>,
    pub max_samples: usize,
}

/// Metal GPU context holding the device, command queue, and all compute pipelines.
pub struct MetalContext {
    pub device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pub pipelines: Pipelines,
    pub gpu_counters: Option<GpuCounters>,
    pub gpu_family: GpuFamily,
    /// True if GPU supports Metal 4 (MTLGPUFamilyMetal4). Orthogonal to Apple GPU family.
    pub metal4: bool,
}

impl MetalContext {
    /// Create a new Metal context with the system default device.
    pub fn new() -> Result<Self> {
        let device = {
            let ptr = MTLCreateSystemDefaultDevice();
            match ptr {
                Some(d) => d,
                None => return Err(HerbertError::Backend("No Metal device found".into())),
            }
        };

        let queue = device.newCommandQueue()
            .ok_or_else(|| HerbertError::Backend("Failed to create Metal command queue".into()))?;

        // Log GPU capabilities for diagnostic / tuning
        let device_name = device.name().to_string();
        let apple7 = device.supportsFamily(MTLGPUFamily::Apple7);
        let apple8 = device.supportsFamily(MTLGPUFamily::Apple8);
        let apple9 = device.supportsFamily(MTLGPUFamily::Apple9);
        let metal3 = device.supportsFamily(MTLGPUFamily::Metal3);
        let tg_mem = device.maxThreadgroupMemoryLength();
        let max_threads = device.maxThreadsPerThreadgroup();
        let max_wset = device.recommendedMaxWorkingSetSize();
        let max_buf = device.maxBufferLength();

        let apple10 = device.supportsFamily(MTLGPUFamily::Apple10);

        // Detect Metal 4 support (MTLGPUFamilyMetal4 = 5002, not yet in objc2-metal bindings).
        // Metal 4 is orthogonal to Apple GPU family — M5 is Apple10 + Metal4.
        let metal4 = {
            let metal4_raw = MTLGPUFamily(5002);
            device.supportsFamily(metal4_raw)
        };

        let (gpu_family, family_str) = if apple10 {
            (GpuFamily::Apple10, "Apple10")
        } else if apple9 {
            (GpuFamily::Apple9, "Apple9")
        } else if apple8 {
            (GpuFamily::Apple8, "Apple8")
        } else if apple7 {
            (GpuFamily::Apple7, "Apple7")
        } else {
            (GpuFamily::Unknown, "Unknown (<Apple7)")
        };

        eprintln!(
            "[metal] {} — family={} metal3={} metal4={} tg_mem={}KB max_threads={}x{}x{} vram={}GB max_buf={}GB",
            device_name, family_str, metal3, metal4,
            tg_mem / 1024,
            max_threads.width, max_threads.height, max_threads.depth,
            max_wset / (1024 * 1024 * 1024),
            max_buf / (1024 * 1024 * 1024),
        );

        if gpu_family == GpuFamily::Unknown {
            eprintln!("[metal] WARNING: GPU family < Apple7 — simdgroup_matrix not available");
        }
        if metal4 {
            eprintln!("[metal] Metal 4 features available: MSL 4.0 shaders, barrier API");
        }

        let pipelines = Pipelines::new(&device, metal4)?;

        // GPU counter profiling setup (METAL_PROFILE=3)
        let gpu_counters = Self::setup_gpu_counters(&device);

        Ok(Self { device, queue, pipelines, gpu_counters, gpu_family, metal4 })
    }

    /// Detect GPU counter support and create a sample buffer for METAL_PROFILE=3.
    fn setup_gpu_counters(device: &ProtocolObject<dyn MTLDevice>) -> Option<GpuCounters> {
        let profile_level = std::env::var("METAL_PROFILE")
            .ok()
            .and_then(|v| v.parse::<u8>().ok())
            .unwrap_or(0);
        if profile_level < 3 {
            return None;
        }

        // Check dispatch boundary support
        let supports_dispatch = device.supportsCounterSampling(
            MTLCounterSamplingPoint::AtDispatchBoundary,
        );
        let supports_stage = device.supportsCounterSampling(
            MTLCounterSamplingPoint::AtStageBoundary,
        );

        eprintln!(
            "[metal-profile] Counter sampling: dispatch_boundary={} stage_boundary={}",
            supports_dispatch, supports_stage,
        );

        if !supports_dispatch {
            eprintln!("[metal-profile] WARNING: GPU does not support AtDispatchBoundary counter sampling");
            eprintln!("[metal-profile]          Falling back to METAL_PROFILE=2 (CB splitting)");
            return None;
        }

        // Find timestamp counter set
        let counter_sets = device.counterSets()?;
        let n_sets = counter_sets.len();

        let mut timestamp_set_idx: Option<usize> = None;
        for i in 0..n_sets {
            let cs = counter_sets.objectAtIndex(i);
            let name_str = cs.name().to_string();
            eprintln!("[metal-profile] Found counter set: \"{}\"", name_str);
            if name_str == "timestamp" {
                timestamp_set_idx = Some(i);
            }
        }

        let timestamp_set_idx = match timestamp_set_idx {
            Some(idx) => idx,
            None => {
                eprintln!("[metal-profile] WARNING: No timestamp counter set found");
                return None;
            }
        };
        let timestamp_set = counter_sets.objectAtIndex(timestamp_set_idx);

        // Log available counters in the timestamp set
        let counters = timestamp_set.counters();
        for i in 0..counters.count() {
            let c = counters.objectAtIndex(i);
            eprintln!("[metal-profile]   counter: \"{}\"", c.name());
        }

        // Create sample buffer (512 slots — enough for ~48 layers × 10 phases + globals)
        let max_samples: usize = 512;
        let desc = MTLCounterSampleBufferDescriptor::new();
        desc.setCounterSet(Some(&timestamp_set));
        desc.setLabel(&objc2_foundation::NSString::from_str("herbert_profile"));
        desc.setStorageMode(MTLStorageMode::Shared);
        unsafe { desc.setSampleCount(max_samples) };

        match device.newCounterSampleBufferWithDescriptor_error(&desc) {
            Ok(buf) => {
                eprintln!(
                    "[metal-profile] GPU counter sampling enabled ({} slots, timestamp)",
                    max_samples,
                );
                Some(GpuCounters {
                    sample_buffer: buf,
                    max_samples,
                })
            }
            Err(e) => {
                eprintln!("[metal-profile] Failed to create counter sample buffer: {:?}", e);
                None
            }
        }
    }

    /// Create a new command buffer from the queue.
    pub fn begin_command_buffer(&self) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>> {
        self.queue.commandBuffer()
            .ok_or_else(|| HerbertError::Backend("Failed to create Metal command buffer".into()))
    }

    /// Create a new compute command encoder from a command buffer.
    pub fn new_compute_encoder(
        cb: &ProtocolObject<dyn MTLCommandBuffer>,
    ) -> Result<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>> {
        cb.computeCommandEncoder()
            .ok_or_else(|| HerbertError::Backend("Failed to create compute encoder".into()))
    }

    /// Submit a command buffer and wait for completion with a timeout.
    ///
    /// Uses `addCompletedHandler` + `Condvar` instead of `waitUntilCompleted()`.
    /// This makes the wait interruptible — if the GPU takes longer than
    /// `CB_TIMEOUT` (30s), we return an error instead of blocking forever
    /// in an uninterruptible kernel sleep.
    ///
    /// This prevents the scenario where `kill -9` during a long
    /// `waitUntilCompleted()` leaves GPU wired memory permanently stuck.
    pub fn submit_and_wait(
        cb: &ProtocolObject<dyn MTLCommandBuffer>,
    ) -> Result<()> {
        // METAL_UNSAFE_BLOCKING=1 uses the old waitUntilCompleted() for benchmarking.
        // DO NOT use in production — causes unrecoverable wired memory leak on kill.
        if std::env::var("METAL_UNSAFE_BLOCKING").ok().as_deref() == Some("1") {
            cb.commit();
            cb.waitUntilCompleted();
            if let Some(error) = cb.error() {
                return Err(HerbertError::Backend(
                    format!("Metal command buffer error: {:?}", error),
                ));
            }
            return Ok(());
        }

        let pair = Arc::new((Mutex::new(false), Condvar::new()));
        let pair_clone = pair.clone();

        let block = block2::RcBlock::new(
            move |_cb: core::ptr::NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
                let (lock, cvar) = &*pair_clone;
                if let Ok(mut done) = lock.lock() {
                    *done = true;
                    cvar.notify_one();
                }
            },
        );

        unsafe {
            cb.addCompletedHandler(&*block as *const _ as *mut _);
        }
        cb.commit();

        // Wait with timeout — thread is NOT in uninterruptible kernel sleep
        let (lock, cvar) = &*pair;
        let mut done = lock.lock().map_err(|_| {
            HerbertError::Backend("Metal CB wait: mutex poisoned".into())
        })?;
        while !*done {
            let (guard, timeout_result) = cvar.wait_timeout(done, CB_TIMEOUT).map_err(|_| {
                HerbertError::Backend("Metal CB wait: condvar error".into())
            })?;
            done = guard;
            if timeout_result.timed_out() && !*done {
                return Err(HerbertError::Backend(format!(
                    "Metal command buffer timed out after {}s — GPU may be hung. \
                     Use METAL_LAYERS_PER_CB=2 to reduce CB size, or check GPU load.",
                    CB_TIMEOUT.as_secs(),
                )));
            }
        }

        // Check for GPU errors
        if let Some(error) = cb.error() {
            return Err(HerbertError::Backend(
                format!("Metal command buffer error: {:?}", error),
            ));
        }

        Ok(())
    }

    /// Commit a command buffer without waiting for completion.
    /// Used for async pipelines where only the final CB needs a blocking wait.
    pub fn commit_no_wait(cb: &ProtocolObject<dyn MTLCommandBuffer>) {
        cb.commit();
    }

    /// Get the device name (for logging).
    pub fn device_name(&self) -> String {
        self.device.name().to_string()
    }
}
