//! Reusable thread pool with persistent workers.
//!
//! Exposes `parallel_for` that reuses worker threads and a shared job descriptor.
//! This avoids per-call channel allocation and boxed task allocation on hot kernel paths.

use herbert_core::cpu_detection::get_performance_cpu_count;
use std::any::Any;
use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Instant;

// ============================================================================
// Dispatch priority (QoS for concurrent inference requests)
// ============================================================================

thread_local! {
    /// Priority for the current thread's `parallel_for` dispatch.
    /// Lower = higher priority: 0 = highest, 9 = lowest, 5 = default.
    static DISPATCH_PRIORITY: Cell<u8> = const { Cell::new(5) };
}

/// Set the dispatch priority for the current thread's `parallel_for` calls.
///
/// When multiple concurrent requests are waiting to dispatch work on the
/// thread pool, the request with the lowest priority value goes first.
/// At equal priority, FIFO ordering is preserved.
///
/// - `0` = highest priority (interactive, real-time streaming)
/// - `5` = default (normal request)
/// - `9` = lowest priority (background/batch processing)
pub fn set_dispatch_priority(priority: u8) {
    DISPATCH_PRIORITY.with(|p| p.set(priority.min(9)));
}

/// Get the current thread's dispatch priority.
pub fn get_dispatch_priority() -> u8 {
    DISPATCH_PRIORITY.with(|p| p.get())
}

/// Wrapper around `*const T` that implements `Send + Sync`.
///
/// # Safety
///
/// The caller must guarantee that:
/// - The pointed-to memory remains valid and immutable for the lifetime of this handle.
/// - This is typically ensured by `parallel_for` blocking until all workers complete,
///   so the source slice on the caller's stack outlives all worker accesses.
pub struct SendPtr<T>(*const T);

// Manual Copy/Clone impls to avoid derive's incorrect `T: Copy` bound.
// Raw pointers are always Copy regardless of T.
impl<T> Copy for SendPtr<T> {}
impl<T> Clone for SendPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}

unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}

impl<T> SendPtr<T> {
    pub fn new(ptr: *const T) -> Self {
        Self(ptr)
    }
    pub fn ptr(self) -> *const T {
        self.0
    }
}

/// Wrapper around `*mut T` that implements `Send + Sync`.
///
/// # Safety
///
/// The caller must guarantee that:
/// - The pointed-to memory remains valid for the lifetime of this handle.
/// - Concurrent writes from different workers target **disjoint** index ranges.
/// - This is typically ensured by `parallel_for` assigning non-overlapping
///   `[start, end)` ranges to each worker.
pub struct SendMutPtr<T>(*mut T);

impl<T> Copy for SendMutPtr<T> {}
impl<T> Clone for SendMutPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}

unsafe impl<T> Send for SendMutPtr<T> {}
unsafe impl<T> Sync for SendMutPtr<T> {}

impl<T> SendMutPtr<T> {
    pub fn new(ptr: *mut T) -> Self {
        Self(ptr)
    }
    pub fn ptr(self) -> *mut T {
        self.0
    }
}

type ParallelFn = unsafe fn(*const (), usize, usize, usize);

#[derive(Clone, Copy)]
struct ParallelJob {
    total_items: usize,
    chunk_size: usize,
    active_workers: usize,
    ctx: *const (),
    func: ParallelFn,
}

// SAFETY: The function pointer and context pointer are only created by `parallel_for` and are
// consumed while `parallel_for` is blocked waiting for completion. They are never retained.
unsafe impl Send for ParallelJob {}
unsafe impl Sync for ParallelJob {}

#[derive(Default)]
struct SharedState {
    shutdown: bool,
    generation: u64,
    current_job: Option<ParallelJob>,
    completed_workers: usize,
    panic: Option<(usize, String)>,
}

struct Shared {
    state: Mutex<SharedState>,
    // Dedicated CV for worker start/wakeup on new job generation.
    worker_cv: Condvar,
    // Dedicated completion CV for the coordinator (parallel_for caller).
    done_cv: Condvar,
}

#[derive(Default)]
struct SyncStats {
    jobs_submitted: AtomicU64,
    active_workers_total: AtomicU64,
    workers_woken: AtomicU64,
    worker_wait_loops: AtomicU64,
    coordinator_wait_loops: AtomicU64,
    notify_worker_calls: AtomicU64,
    notify_done_calls: AtomicU64,
    worker_wait_ns: AtomicU64,
    coordinator_wait_ns: AtomicU64,
    panic_count: AtomicU64,
}

impl SyncStats {
    const fn new() -> Self {
        Self {
            jobs_submitted: AtomicU64::new(0),
            active_workers_total: AtomicU64::new(0),
            workers_woken: AtomicU64::new(0),
            worker_wait_loops: AtomicU64::new(0),
            coordinator_wait_loops: AtomicU64::new(0),
            notify_worker_calls: AtomicU64::new(0),
            notify_done_calls: AtomicU64::new(0),
            worker_wait_ns: AtomicU64::new(0),
            coordinator_wait_ns: AtomicU64::new(0),
            panic_count: AtomicU64::new(0),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct ThreadPoolSyncStats {
    pub jobs_submitted: u64,
    pub active_workers_total: u64,
    pub workers_woken: u64,
    pub worker_wait_loops: u64,
    pub coordinator_wait_loops: u64,
    pub worker_wait_ns: u64,
    pub coordinator_wait_ns: u64,
    pub panic_count: u64,
}

fn sync_debug_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("HERBERT_SYNC_DEBUG")
            .map(|v| v != "0" && v.to_lowercase() != "false")
            .unwrap_or(false)
    })
}

static SYNC_STATS: SyncStats = SyncStats::new();

fn sync_stat_inc(counter: &AtomicU64, value: u64) {
    if sync_debug_enabled() {
        counter.fetch_add(value, Ordering::Relaxed);
    }
}

pub(crate) fn sync_stats_snapshot() -> Option<ThreadPoolSyncStats> {
    if !sync_debug_enabled() {
        return None;
    }
    Some(ThreadPoolSyncStats {
        jobs_submitted: SYNC_STATS.jobs_submitted.load(Ordering::Relaxed),
        active_workers_total: SYNC_STATS.active_workers_total.load(Ordering::Relaxed),
        workers_woken: SYNC_STATS.workers_woken.load(Ordering::Relaxed),
        worker_wait_loops: SYNC_STATS.worker_wait_loops.load(Ordering::Relaxed),
        coordinator_wait_loops: SYNC_STATS.coordinator_wait_loops.load(Ordering::Relaxed),
        worker_wait_ns: SYNC_STATS.worker_wait_ns.load(Ordering::Relaxed),
        coordinator_wait_ns: SYNC_STATS.coordinator_wait_ns.load(Ordering::Relaxed),
        panic_count: SYNC_STATS.panic_count.load(Ordering::Relaxed),
    })
}

fn log_sync_stats(prefix: &str) {
    if let Some(stats) = sync_stats_snapshot() {
        tracing::debug!(
            prefix,
            jobs = stats.jobs_submitted,
            active_total = stats.active_workers_total,
            workers_woken = stats.workers_woken,
            worker_wait_loops = stats.worker_wait_loops,
            coordinator_wait_loops = stats.coordinator_wait_loops,
            worker_wait_ms = stats.worker_wait_ns as f64 / 1_000_000.0,
            coordinator_wait_ms = stats.coordinator_wait_ns as f64 / 1_000_000.0,
            panic_count = stats.panic_count,
            "sync stats"
        );
    }
}

/// Error returned by thread-pool scoped execution
#[derive(Debug)]
pub enum ThreadPoolError {
    WorkerPanic { worker_id: usize, message: String },
    PoolAlreadyInitialized { existing: usize, requested: usize },
}

impl std::fmt::Display for ThreadPoolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ThreadPoolError::WorkerPanic { worker_id, message } => {
                write!(f, "worker {} panicked: {}", worker_id, message)
            }
            ThreadPoolError::PoolAlreadyInitialized {
                existing,
                requested,
            } => {
                write!(
                    f,
                    "thread pool already initialized with {} workers (requested {})",
                    existing, requested
                )
            }
        }
    }
}

impl std::error::Error for ThreadPoolError {}

// ============================================================================
// Priority dispatcher: serializes parallel_for with priority ordering
// ============================================================================

struct DispatchState {
    /// Whether a parallel_for is currently executing.
    active: bool,
    /// Monotonic ticket counter for FIFO ordering within same priority.
    next_ticket: u64,
    /// Ticket of the waiter that should proceed next (set by release).
    serving: Option<u64>,
    /// Waiting callers: (priority, ticket). Lower priority = higher precedence.
    waiters: Vec<(u8, u64)>,
}

/// Priority-aware dispatcher that replaces a simple `Mutex<()>`.
///
/// When multiple concurrent requests want to dispatch `parallel_for` work,
/// the request with the lowest priority value (highest precedence) goes first.
/// At equal priority, FIFO order is preserved via monotonic tickets.
struct PriorityDispatcher {
    state: Mutex<DispatchState>,
    cv: Condvar,
}

impl PriorityDispatcher {
    fn new() -> Self {
        Self {
            state: Mutex::new(DispatchState {
                active: false,
                next_ticket: 0,
                serving: None,
                waiters: Vec::new(),
            }),
            cv: Condvar::new(),
        }
    }

    /// Acquire the dispatch slot. Blocks until this caller has the highest
    /// priority among all waiters. Returns a guard that releases on drop.
    fn acquire(&self) -> DispatchGuard<'_> {
        let priority = DISPATCH_PRIORITY.with(|p| p.get());
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());

        if !state.active {
            // No one active, proceed immediately
            state.active = true;
            return DispatchGuard { dispatcher: self };
        }

        // Someone is active — queue ourselves and wait
        let ticket = state.next_ticket;
        state.next_ticket += 1;
        state.waiters.push((priority, ticket));

        loop {
            state = self.cv.wait(state).unwrap_or_else(|e| e.into_inner());
            if state.serving == Some(ticket) {
                state.serving = None; // consumed
                return DispatchGuard { dispatcher: self };
            }
        }
    }

    /// Release the dispatch slot and wake the highest-priority waiter.
    fn release(&self) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());

        if state.waiters.is_empty() {
            state.active = false;
        } else {
            // Find highest-priority waiter (lowest priority number, FIFO for ties)
            let best_idx = state
                .waiters
                .iter()
                .enumerate()
                .min_by_key(|(_, (prio, ticket))| (*prio, *ticket))
                .map(|(idx, _)| idx)
                .unwrap();
            let (_prio, ticket) = state.waiters.swap_remove(best_idx);
            state.serving = Some(ticket);
            self.cv.notify_all();
        }
    }
}

/// RAII guard that releases the dispatch slot when dropped.
struct DispatchGuard<'a> {
    dispatcher: &'a PriorityDispatcher,
}

impl Drop for DispatchGuard<'_> {
    fn drop(&mut self) {
        self.dispatcher.release();
    }
}

/// Thread pool with persistent workers and shared job descriptor.
///
/// The pool supports a single `parallel_for` at a time. When multiple threads
/// call `parallel_for` concurrently (e.g., concurrent inference requests),
/// the priority dispatcher serializes them — higher-priority requests get
/// the thread pool first, with FIFO ordering at equal priority.
/// Requests interleave at the granularity of individual `parallel_for` calls
/// (~1-3ms per layer).
pub struct ThreadPool {
    /// Number of workers
    num_workers: usize,
    /// Shared scheduler state for all workers.
    shared: Arc<Shared>,
    /// Worker handles (for cleanup)
    workers: Vec<JoinHandle<()>>,
    /// Priority-aware dispatcher for concurrent callers.
    dispatcher: PriorityDispatcher,
}

impl ThreadPool {
    /// Create a new thread pool with the given number of workers.
    ///
    /// On Linux, workers are pinned to physical cores (one per core, skipping SMT siblings)
    /// to give each thread a full L2 cache instead of sharing it with a sibling.
    pub fn new(num_workers: usize) -> Self {
        let num_workers = num_workers.max(1);
        let shared = Arc::new(Shared {
            state: Mutex::new(SharedState::default()),
            worker_cv: Condvar::new(),
            done_cv: Condvar::new(),
        });
        let mut workers = Vec::with_capacity(num_workers);

        let core_ids = crate::topology::physical_core_ids();

        for worker_id in 0..num_workers {
            let shared_clone = Arc::clone(&shared);
            let pin_to = if worker_id < core_ids.len() {
                Some(core_ids[worker_id])
            } else {
                None
            };
            let handle = thread::spawn(move || {
                if let Some(cpu_id) = pin_to {
                    set_thread_affinity(cpu_id);
                }
                worker_loop(worker_id, shared_clone);
            });
            workers.push(handle);
        }

        ThreadPool {
            num_workers,
            shared,
            workers,
            dispatcher: PriorityDispatcher::new(),
        }
    }

    /// Get the number of workers in this pool
    pub fn num_workers(&self) -> usize {
        self.num_workers
    }

    /// Get the effective number of workers for the current execution phase.
    ///
    /// During decode, this may return fewer workers than `num_workers()` if a
    /// decode thread cap is configured (via `--decode-threads`).
    pub fn effective_workers(&self) -> usize {
        crate::execution_phase::effective_max_workers(self.num_workers)
    }

    /// Run a parallel-for job using the phase-aware worker count.
    ///
    /// Equivalent to `parallel_for_with_max_workers(total, self.effective_workers(), f)`.
    pub fn parallel_for_phase_aware<F>(
        &self,
        total_items: usize,
        f: F,
    ) -> Result<(), ThreadPoolError>
    where
        F: Fn(usize, usize, usize) + Sync,
    {
        let max = self.effective_workers();
        self.parallel_for_with_max_workers(total_items, max, f)
    }

    /// Run a parallel-for job over `[0, total_items)` with a contiguous range per worker.
    ///
    /// The callback receives `(worker_id, start, end)`.
    pub fn parallel_for<F>(&self, total_items: usize, f: F) -> Result<(), ThreadPoolError>
    where
        F: Fn(usize, usize, usize) + Sync,
    {
        self.parallel_for_with_max_workers(total_items, self.num_workers, f)
    }

    /// Run a parallel-for job over `[0, total_items)` while capping active workers.
    ///
    /// The callback receives `(worker_id, start, end)`.
    ///
    /// # Safety contract for closures using raw pointers
    ///
    /// `parallel_for` **blocks** until all workers have completed. This guarantees:
    ///
    /// 1. **Lifetime**: Any data referenced via `SendPtr`/`SendMutPtr` captured in the
    ///    closure remains valid for the entire duration of the parallel region, because
    ///    the caller's stack frame (which owns the data) cannot be exited until
    ///    `parallel_for` returns.
    ///
    /// 2. **Disjoint writes**: Each worker receives a non-overlapping `[start, end)` range.
    ///    Workers writing to output buffers via `SendMutPtr` must only access indices within
    ///    their assigned range (or a deterministic function of it, e.g. tile-based).
    ///
    /// 3. **No aliasing**: Shared inputs (`SendPtr`) are read-only during the parallel
    ///    region. Mutable outputs (`SendMutPtr`) are written only by their assigned worker.
    pub fn parallel_for_with_max_workers<F>(
        &self,
        total_items: usize,
        max_workers: usize,
        f: F,
    ) -> Result<(), ThreadPoolError>
    where
        F: Fn(usize, usize, usize) + Sync,
    {
        if total_items == 0 {
            return Ok(());
        }

        let worker_cap = max_workers.max(1);
        let active_workers = self.num_workers.min(worker_cap).min(total_items).max(1);
        let chunk_size = total_items.div_ceil(active_workers);

        if active_workers == 1 {
            f(0, 0, total_items);
            return Ok(());
        }

        // Serialize concurrent callers with priority ordering. Higher-priority
        // requests (lower priority number) get the thread pool first.
        // Requests interleave at the granularity of individual parallel_for
        // calls (~1-3ms per layer).
        let _dispatch_guard = self.dispatcher.acquire();

        unsafe fn call_f<F>(ctx: *const (), worker_id: usize, start: usize, end: usize)
        where
            F: Fn(usize, usize, usize) + Sync,
        {
            // SAFETY: `ctx` points to `f` on the caller stack and remains valid until all workers
            // completed. `parallel_for` blocks until completion before returning.
            let f = unsafe { &*(ctx as *const F) };
            (f)(worker_id, start, end);
        }

        let job = ParallelJob {
            total_items,
            chunk_size,
            active_workers,
            ctx: &f as *const F as *const (),
            func: call_f::<F>,
        };

        sync_stat_inc(&SYNC_STATS.jobs_submitted, 1);
        sync_stat_inc(&SYNC_STATS.active_workers_total, active_workers as u64);

        let mut state = self
            .shared
            .state
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        state.completed_workers = 0;
        state.panic = None;
        state.current_job = Some(job);
        state.generation = state.generation.wrapping_add(1);
        self.shared.worker_cv.notify_all();
        sync_stat_inc(&SYNC_STATS.notify_worker_calls, 1);
        sync_stat_inc(&SYNC_STATS.workers_woken, self.num_workers as u64);

        while state.completed_workers < active_workers {
            sync_stat_inc(&SYNC_STATS.coordinator_wait_loops, 1);
            let wait_start = if sync_debug_enabled() {
                Some(Instant::now())
            } else {
                None
            };
            state = self
                .shared
                .done_cv
                .wait(state)
                .unwrap_or_else(|e| e.into_inner());
            if let Some(start) = wait_start {
                let wait_ns = start.elapsed().as_nanos() as u64;
                sync_stat_inc(&SYNC_STATS.coordinator_wait_ns, wait_ns);
            }
        }

        if let Some((worker_id, message)) = state.panic.take() {
            return Err(ThreadPoolError::WorkerPanic { worker_id, message });
        }
        if sync_debug_enabled() {
            let submitted = SYNC_STATS.jobs_submitted.load(Ordering::Relaxed);
            if submitted.is_multiple_of(5_000) {
                log_sync_stats("periodic");
            }
        }
        Ok(())
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        {
            let mut state = self
                .shared
                .state
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            state.shutdown = true;
            self.shared.worker_cv.notify_all();
            sync_stat_inc(&SYNC_STATS.notify_worker_calls, 1);
            self.shared.done_cv.notify_all();
            sync_stat_inc(&SYNC_STATS.notify_done_calls, 1);
        }
        if sync_debug_enabled() {
            log_sync_stats("drop");
        }

        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

/// Worker loop: wait for generation updates and execute assigned range.
fn worker_loop(worker_id: usize, shared: Arc<Shared>) {
    let mut seen_generation = 0u64;

    loop {
        let job = {
            let mut state = shared.state.lock().unwrap_or_else(|e| e.into_inner());
            while !state.shutdown && state.generation == seen_generation {
                sync_stat_inc(&SYNC_STATS.worker_wait_loops, 1);
                let wait_start = if sync_debug_enabled() {
                    Some(Instant::now())
                } else {
                    None
                };
                state = shared
                    .worker_cv
                    .wait(state)
                    .unwrap_or_else(|e| e.into_inner());
                if let Some(start) = wait_start {
                    let wait_ns = start.elapsed().as_nanos() as u64;
                    sync_stat_inc(&SYNC_STATS.worker_wait_ns, wait_ns);
                }
            }
            if state.shutdown {
                return;
            }
            seen_generation = state.generation;
            state.current_job
        };

        let Some(job) = job else {
            continue;
        };

        if worker_id >= job.active_workers {
            continue;
        }

        let start = worker_id * job.chunk_size;
        let end = (start + job.chunk_size).min(job.total_items);
        if start < end {
            let result = std::panic::catch_unwind(|| {
                // SAFETY: The job descriptor is published under mutex and remains valid until
                // the coordinator observes all worker completions for this generation.
                unsafe { (job.func)(job.ctx, worker_id, start, end) };
            });
            if let Err(payload) = result {
                let message = panic_payload_to_string(payload);
                let mut state = shared.state.lock().unwrap_or_else(|e| e.into_inner());
                if state.panic.is_none() {
                    state.panic = Some((worker_id, message));
                }
                sync_stat_inc(&SYNC_STATS.panic_count, 1);
                state.completed_workers += 1;
                if state.completed_workers >= job.active_workers {
                    shared.done_cv.notify_one();
                    sync_stat_inc(&SYNC_STATS.notify_done_calls, 1);
                }
                continue;
            }
        }

        let mut state = shared.state.lock().unwrap_or_else(|e| e.into_inner());
        state.completed_workers += 1;
        if state.completed_workers >= job.active_workers {
            shared.done_cv.notify_one();
            sync_stat_inc(&SYNC_STATS.notify_done_calls, 1);
        }
    }
}

/// Pin the current thread to the given logical CPU ID.
/// No-op on non-Linux platforms.
#[cfg(target_os = "linux")]
fn set_thread_affinity(cpu_id: usize) {
    unsafe {
        let mut cpuset: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut cpuset);
        libc::CPU_SET(cpu_id, &mut cpuset);
        let ret = libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &cpuset);
        if ret != 0 {
            // Best-effort: log failure but don't abort
            tracing::warn!(cpu_id, error = %std::io::Error::last_os_error(), "sched_setaffinity failed");
        }
    }
}

#[cfg(not(target_os = "linux"))]
fn set_thread_affinity(_cpu_id: usize) {
    // No-op: thread affinity not supported on this platform
}

fn panic_payload_to_string(payload: Box<dyn Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "unknown panic payload".to_string()
    }
}

// Global thread pool instance
use std::sync::OnceLock;

static GLOBAL_POOL: OnceLock<ThreadPool> = OnceLock::new();
static GLOBAL_POOL_THREAD_OVERRIDE: OnceLock<usize> = OnceLock::new();

/// Configure the global thread pool worker count before first use.
pub fn configure_global_pool(num_threads: usize) -> Result<(), ThreadPoolError> {
    let requested = num_threads.max(1);
    if let Some(pool) = GLOBAL_POOL.get() {
        let existing = pool.num_workers();
        if existing != requested {
            return Err(ThreadPoolError::PoolAlreadyInitialized {
                existing,
                requested,
            });
        }
        return Ok(());
    }

    if let Some(existing) = GLOBAL_POOL_THREAD_OVERRIDE.get() {
        if *existing != requested {
            return Err(ThreadPoolError::PoolAlreadyInitialized {
                existing: *existing,
                requested,
            });
        }
        return Ok(());
    }

    let _ = GLOBAL_POOL_THREAD_OVERRIDE.set(requested);
    Ok(())
}

/// Get the global thread pool
///
/// The pool is lazily initialized with the number of performance cores.
pub fn global_pool() -> &'static ThreadPool {
    GLOBAL_POOL.get_or_init(|| {
        let num_threads = GLOBAL_POOL_THREAD_OVERRIDE
            .get()
            .copied()
            .unwrap_or_else(get_performance_cpu_count)
            .max(1);
        ThreadPool::new(num_threads)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn pool_basic_parallel_for() {
        let pool = ThreadPool::new(4);
        assert_eq!(pool.num_workers(), 4);

        let counter = AtomicUsize::new(0);
        pool.parallel_for(100, |_worker, start, end| {
            counter.fetch_add(end - start, Ordering::Relaxed);
        })
        .unwrap();
        assert_eq!(counter.load(Ordering::Relaxed), 100);
    }

    #[test]
    fn pool_parallel_for_phase_aware() {
        let pool = ThreadPool::new(4);

        // parallel_for_phase_aware should work without panicking
        let counter = AtomicUsize::new(0);
        pool.parallel_for_phase_aware(50, |_worker, start, end| {
            counter.fetch_add(end - start, Ordering::Relaxed);
        })
        .unwrap();
        assert_eq!(counter.load(Ordering::Relaxed), 50);
    }

    #[test]
    fn effective_workers_respects_phase() {
        let pool = ThreadPool::new(8);

        // In prefill phase, should return full pool size
        crate::execution_phase::set_phase(crate::execution_phase::ExecutionPhase::Prefill);
        assert_eq!(pool.effective_workers(), 8);

        // Restore
        crate::execution_phase::set_phase(crate::execution_phase::ExecutionPhase::Prefill);
    }

    #[test]
    fn pool_zero_items() {
        let pool = ThreadPool::new(4);
        // Should be a no-op
        pool.parallel_for(0, |_, _, _| {
            panic!("should not be called");
        })
        .unwrap();
    }

    #[test]
    fn pool_single_item() {
        let pool = ThreadPool::new(4);
        let counter = AtomicUsize::new(0);
        pool.parallel_for(1, |_worker, start, end| {
            assert_eq!(start, 0);
            assert_eq!(end, 1);
            counter.fetch_add(1, Ordering::Relaxed);
        })
        .unwrap();
        assert_eq!(counter.load(Ordering::Relaxed), 1);
    }
}
