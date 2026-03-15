//! Thread-local prefill progress callback.
//!
//! Same pattern as `DISPATCH_PRIORITY` in `thread_pool.rs`: the server handler
//! sets a callback before entering the blocking prefill, the model loop calls
//! `report()` after each chunk, and the handler clears it when done.

use std::cell::RefCell;

thread_local! {
    static CALLBACK: RefCell<Option<Box<dyn Fn(usize, usize) + Send + Sync>>> = const { RefCell::new(None) };
}

/// Install a progress callback for the current thread.
///
/// The callback receives `(tokens_processed, tokens_total)`.
pub fn set_callback(cb: Option<Box<dyn Fn(usize, usize) + Send + Sync>>) {
    CALLBACK.with(|c| *c.borrow_mut() = cb);
}

/// Take the progress callback out of thread-local storage (returns it, leaves None).
///
/// Used by the Metal backend to extract the callback and wrap it in an Arc
/// for use in MTLSharedEvent RcBlock callbacks on a dispatch queue thread.
pub fn take_callback() -> Option<Box<dyn Fn(usize, usize) + Send + Sync>> {
    CALLBACK.with(|c| c.borrow_mut().take())
}

/// Report prefill progress. No-op if no callback is installed.
pub fn report(done: usize, total: usize) {
    CALLBACK.with(|c| {
        if let Some(ref cb) = *c.borrow() {
            cb(done, total);
        }
    });
}
