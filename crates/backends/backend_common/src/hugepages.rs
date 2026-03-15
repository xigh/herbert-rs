//! Hugepage allocation for large buffers (weights, KV cache).
//!
//! Two modes:
//! - `hint_hugepages`: MADV_HUGEPAGE hint for Vec-backed memory (best-effort THP).
//! - `HugeVec<T>`: mmap with MAP_HUGETLB for guaranteed 2MB hugepage backing.
//!   Falls back to regular Vec + MADV_HUGEPAGE hint if hugepage mmap fails.
//!
//! No-op on non-Linux or when the `hugepages` feature is disabled.

use std::fmt;
use std::ops::{Deref, DerefMut};

/// Minimum allocation size to attempt hugepage mmap (2 MB).
#[cfg(all(target_os = "linux", feature = "hugepages"))]
const HUGEPAGE_MIN_BYTES: usize = 2 * 1024 * 1024;

/// 2 MB page size for MAP_HUGETLB.
#[cfg(all(target_os = "linux", feature = "hugepages"))]
const HUGE_PAGE_SIZE: usize = 2 * 1024 * 1024;

// ── HugeVec<T> ─────────────────────────────────────────────────────────

/// A growable buffer backed by either `MAP_HUGETLB` mmap (guaranteed 2MB pages)
/// or a regular `Vec<T>` (fallback).
///
/// Supports both fixed-size use (weights via `from_vec`) and growable use
/// (KV cache via `with_capacity` + `push`/`extend_from_slice`).
///
/// When the `hugepages` feature is enabled on Linux, allocation attempts
/// `mmap(MAP_HUGETLB)`. If that fails (insufficient pre-allocated hugepages),
/// falls back to `Vec<T>` + `MADV_HUGEPAGE` hint.
pub struct HugeVec<T> {
    inner: HugeVecInner<T>,
}

enum HugeVecInner<T> {
    Vec(Vec<T>),
    #[cfg(all(target_os = "linux", feature = "hugepages"))]
    Mmap {
        ptr: *mut T,
        len: usize,       // current number of T elements used
        cap: usize,       // total capacity in T elements
        mmap_len: usize,  // actual mmap size in bytes (rounded up to 2MB)
    },
}

// SAFETY: The mmap'd region is exclusively owned by HugeVec, no aliasing.
unsafe impl<T: Send> Send for HugeVec<T> {}
unsafe impl<T: Sync> Sync for HugeVec<T> {}

impl<T: Copy> HugeVec<T> {
    /// Create a HugeVec from a Vec, attempting to move data to hugepage mmap.
    /// The capacity equals the length (no spare room for growth).
    pub fn from_vec(v: Vec<T>) -> Self {
        #[cfg(all(target_os = "linux", feature = "hugepages"))]
        {
            let byte_len = v.len() * std::mem::size_of::<T>();
            if byte_len >= HUGEPAGE_MIN_BYTES {
                if let Some(inner) = Self::try_mmap_copy(&v, v.len()) {
                    return HugeVec { inner };
                }
                hint_hugepages(&v);
            }
        }
        HugeVec { inner: HugeVecInner::Vec(v) }
    }

    /// Wrap a Vec without attempting hugepage allocation.
    pub fn from_vec_no_huge(v: Vec<T>) -> Self {
        HugeVec { inner: HugeVecInner::Vec(v) }
    }

    /// Allocate with capacity, len=0. Attempts MAP_HUGETLB for the full capacity.
    /// Use `push` / `extend_from_slice` to fill.
    pub fn with_capacity(cap: usize) -> Self {
        #[cfg(all(target_os = "linux", feature = "hugepages"))]
        {
            let byte_cap = cap * std::mem::size_of::<T>();
            if byte_cap >= HUGEPAGE_MIN_BYTES {
                if let Some(inner) = Self::try_mmap_empty(cap) {
                    return HugeVec { inner };
                }
            }
        }
        let v = Vec::with_capacity(cap);
        hint_hugepages(&v);
        HugeVec { inner: HugeVecInner::Vec(v) }
    }

    // ── Mmap helpers ────────────────────────────────────────────────

    /// Mmap a hugepage region, copy src into it. cap = src.len().
    #[cfg(all(target_os = "linux", feature = "hugepages"))]
    fn try_mmap_copy(src: &[T], cap: usize) -> Option<HugeVecInner<T>> {
        let byte_cap = cap.max(src.len()) * std::mem::size_of::<T>();
        let mmap_len = (byte_cap + HUGE_PAGE_SIZE - 1) & !(HUGE_PAGE_SIZE - 1);

        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                mmap_len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_ANONYMOUS | libc::MAP_PRIVATE | libc::MAP_HUGETLB,
                -1,
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            tracing::debug!(
                "MAP_HUGETLB mmap failed for {} bytes: {}",
                byte_cap, std::io::Error::last_os_error()
            );
            return None;
        }
        tracing::debug!(
            "MAP_HUGETLB mmap OK: {} bytes in {} MB hugepage region",
            byte_cap, mmap_len / (1024 * 1024)
        );
        let byte_len = src.len() * std::mem::size_of::<T>();
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr() as *const u8, ptr as *mut u8, byte_len);
        }
        Some(HugeVecInner::Mmap {
            ptr: ptr as *mut T,
            len: src.len(),
            cap,
            mmap_len,
        })
    }

    /// Mmap an empty hugepage region with given capacity.
    #[cfg(all(target_os = "linux", feature = "hugepages"))]
    fn try_mmap_empty(cap: usize) -> Option<HugeVecInner<T>> {
        let byte_cap = cap * std::mem::size_of::<T>();
        let mmap_len = (byte_cap + HUGE_PAGE_SIZE - 1) & !(HUGE_PAGE_SIZE - 1);

        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                mmap_len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_ANONYMOUS | libc::MAP_PRIVATE | libc::MAP_HUGETLB,
                -1,
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return None; // silent fallback for capacity allocs (many small ones)
        }
        Some(HugeVecInner::Mmap {
            ptr: ptr as *mut T,
            len: 0,
            cap,
            mmap_len,
        })
    }

    // ── Public API ──────────────────────────────────────────────────

    pub fn len(&self) -> usize {
        match &self.inner {
            HugeVecInner::Vec(v) => v.len(),
            #[cfg(all(target_os = "linux", feature = "hugepages"))]
            HugeVecInner::Mmap { len, .. } => *len,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn capacity(&self) -> usize {
        match &self.inner {
            HugeVecInner::Vec(v) => v.capacity(),
            #[cfg(all(target_os = "linux", feature = "hugepages"))]
            HugeVecInner::Mmap { cap, .. } => *cap,
        }
    }

    /// Whether this buffer is backed by hugepages (MAP_HUGETLB).
    pub fn is_hugepage_backed(&self) -> bool {
        #[cfg(all(target_os = "linux", feature = "hugepages"))]
        if matches!(self.inner, HugeVecInner::Mmap { .. }) {
            return true;
        }
        false
    }

    pub fn push(&mut self, val: T) {
        match &mut self.inner {
            HugeVecInner::Vec(v) => v.push(val),
            #[cfg(all(target_os = "linux", feature = "hugepages"))]
            HugeVecInner::Mmap { ptr, len, cap, .. } => {
                assert!(*len < *cap, "HugeVec: push beyond capacity ({}/{})", *len, *cap);
                unsafe { ptr.add(*len).write(val); }
                *len += 1;
            }
        }
    }

    pub fn extend_from_slice(&mut self, src: &[T]) {
        match &mut self.inner {
            HugeVecInner::Vec(v) => v.extend_from_slice(src),
            #[cfg(all(target_os = "linux", feature = "hugepages"))]
            HugeVecInner::Mmap { ptr, len, cap, .. } => {
                let new_len = *len + src.len();
                assert!(new_len <= *cap, "HugeVec: extend beyond capacity ({}/{})", new_len, *cap);
                unsafe {
                    std::ptr::copy_nonoverlapping(src.as_ptr(), ptr.add(*len), src.len());
                }
                *len = new_len;
            }
        }
    }

    pub fn truncate(&mut self, new_len: usize) {
        match &mut self.inner {
            HugeVecInner::Vec(v) => v.truncate(new_len),
            #[cfg(all(target_os = "linux", feature = "hugepages"))]
            HugeVecInner::Mmap { len, .. } => {
                if new_len < *len {
                    *len = new_len;
                }
            }
        }
    }

    pub fn clear(&mut self) {
        self.truncate(0);
    }

    pub fn resize(&mut self, new_len: usize, val: T) {
        match &mut self.inner {
            HugeVecInner::Vec(v) => v.resize(new_len, val),
            #[cfg(all(target_os = "linux", feature = "hugepages"))]
            HugeVecInner::Mmap { ptr, len, cap, .. } => {
                if new_len <= *len {
                    *len = new_len;
                } else {
                    assert!(new_len <= *cap, "HugeVec: resize beyond capacity ({}/{})", new_len, *cap);
                    for i in *len..new_len {
                        unsafe { ptr.add(i).write(val); }
                    }
                    *len = new_len;
                }
            }
        }
    }

    /// Iterate over elements.
    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.deref().iter()
    }

    /// Mutable iteration.
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, T> {
        self.deref_mut().iter_mut()
    }
}

// ── Trait implementations ──────────────────────────────────────────────

impl<T: Copy> Deref for HugeVec<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        match &self.inner {
            HugeVecInner::Vec(v) => v,
            #[cfg(all(target_os = "linux", feature = "hugepages"))]
            HugeVecInner::Mmap { ptr, len, .. } => unsafe {
                std::slice::from_raw_parts(*ptr, *len)
            },
        }
    }
}

impl<T: Copy> DerefMut for HugeVec<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        match &mut self.inner {
            HugeVecInner::Vec(v) => v,
            #[cfg(all(target_os = "linux", feature = "hugepages"))]
            HugeVecInner::Mmap { ptr, len, .. } => unsafe {
                std::slice::from_raw_parts_mut(*ptr, *len)
            },
        }
    }
}

impl<T: Copy> Clone for HugeVec<T> {
    fn clone(&self) -> Self {
        match &self.inner {
            HugeVecInner::Vec(v) => HugeVec::from_vec(v.clone()),
            #[cfg(all(target_os = "linux", feature = "hugepages"))]
            HugeVecInner::Mmap { ptr, len, cap, .. } => {
                let slice = unsafe { std::slice::from_raw_parts(*ptr, *len) };
                // Clone preserves capacity for growable buffers
                #[cfg(all(target_os = "linux", feature = "hugepages"))]
                if let Some(inner) = Self::try_mmap_copy(slice, *cap) {
                    return HugeVec { inner };
                }
                let v = slice.to_vec();
                HugeVec { inner: HugeVecInner::Vec(v) }
            }
        }
    }
}

impl<T> Drop for HugeVec<T> {
    fn drop(&mut self) {
        #[cfg(all(target_os = "linux", feature = "hugepages"))]
        if let HugeVecInner::Mmap { ptr, mmap_len, .. } = &self.inner {
            unsafe {
                libc::munmap(*ptr as *mut libc::c_void, *mmap_len);
            }
            return;
        }
        // Vec variant drops automatically
    }
}

impl<T: Copy + fmt::Debug> fmt::Debug for HugeVec<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let slice: &[T] = self;
        write!(f, "HugeVec(len={}, huge={})", slice.len(), self.is_hugepage_backed())
    }
}

// ── Existing hint API (for KV cache etc.) ──────────────────────────────

/// Apply `MADV_HUGEPAGE` hint to a Vec's backing memory.
#[cfg(all(target_os = "linux", feature = "hugepages"))]
pub fn hint_hugepages<T>(v: &Vec<T>) {
    let byte_len = v.capacity() * std::mem::size_of::<T>();
    if byte_len < HUGEPAGE_MIN_BYTES {
        return;
    }
    let ptr = v.as_ptr() as *mut libc::c_void;
    let (aligned_ptr, aligned_len) = align_to_page(ptr, byte_len);
    unsafe {
        libc::madvise(aligned_ptr, aligned_len, libc::MADV_HUGEPAGE);
    }
}

#[cfg(not(all(target_os = "linux", feature = "hugepages")))]
pub fn hint_hugepages<T>(_v: &Vec<T>) {}

#[cfg(all(target_os = "linux", feature = "hugepages"))]
fn align_to_page(ptr: *mut libc::c_void, len: usize) -> (*mut libc::c_void, usize) {
    let page_size = 4096usize;
    let addr = ptr as usize;
    let aligned_addr = addr & !(page_size - 1);
    let aligned_ptr = aligned_addr as *mut libc::c_void;
    let aligned_len = len + (addr - aligned_addr);
    (aligned_ptr, aligned_len)
}
