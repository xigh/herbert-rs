//! Type-erased KV cache handle, passed between `Backend::prefill` and `Backend::decode_next`.

use std::any::Any;

/// Wraps a backend-specific KV cache behind `dyn Any`.
///
/// Each backend stores its own concrete type inside; the caller never
/// inspects the contents directly.
pub struct KvHandle {
    inner: Box<dyn Any + Send + Sync>,
}

impl KvHandle {
    pub fn new<T: Any + Send + Sync>(cache: T) -> Self {
        Self {
            inner: Box::new(cache),
        }
    }

    pub fn get_mut<T: Any>(&mut self) -> crate::error::Result<&mut T> {
        self.inner
            .downcast_mut::<T>()
            .ok_or_else(|| crate::error::HerbertError::Backend("KvHandle type mismatch".into()))
    }

    pub fn get_ref<T: Any>(&self) -> crate::error::Result<&T> {
        self.inner
            .downcast_ref::<T>()
            .ok_or_else(|| crate::error::HerbertError::Backend("KvHandle type mismatch".into()))
    }
}
