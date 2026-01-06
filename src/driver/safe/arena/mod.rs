use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use crate::driver::core::{ArenaAllocType, CudaAllocType};
use crate::driver::result::{self, DriverError};
use crate::driver::{CudaSlice, CudaStream, DeviceRepr};

mod const_pool;

#[derive(Debug)]
struct Inner {
    slab_ptr: u64,
    ofs: AtomicUsize,
    len: usize,
    epoch: AtomicUsize,
}

impl Eq for Inner {}
impl PartialEq for Inner {
    fn eq(&self, other: &Self) -> bool {
        self.slab_ptr == other.slab_ptr && self.len == other.len
    }
}

#[derive(Debug, Eq)]
pub struct CudaArena {
    inner: Arc<Inner>,
    stream: Arc<CudaStream>,
}

impl PartialEq for CudaArena {
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner && self.stream == other.stream
    }
}

impl CudaArena {
    pub fn new(stream: Arc<CudaStream>, capacity: usize) -> Result<Self, DriverError> {
        let cu_stream: *mut crate::driver::sys::CUstream_st = stream.cu_stream();

        stream.ctx.bind_to_thread()?;

        let slab_ptr = unsafe {
            if stream.ctx.has_async_alloc {
                result::malloc_async(cu_stream, capacity)?
            } else {
                result::malloc_sync(capacity)?
            }
        };

        let inner = Arc::new(Inner {
            slab_ptr,
            ofs: AtomicUsize::new(0),
            len: capacity,
            epoch: AtomicUsize::new(0),
        });

        Ok(Self { inner, stream })
    }

    pub unsafe fn alloc<T: DeviceRepr>(&self, len: usize) -> Result<CudaSlice<T>, DriverError> {
        let ptr_ofs = self
            .inner
            .ofs
            .fetch_add(len * std::mem::size_of::<T>(), Ordering::Relaxed);

        let ptr = self.inner.slab_ptr + ptr_ofs as u64;
        let mut slice = self.stream.upgrade_device_ptr(ptr as _, len);
        slice.allocation = CudaAllocType::Arena(ArenaAllocType::Pooled);

        Ok(slice)
    }

    pub fn reset(&self) -> Result<(), DriverError> {
        self.inner.ofs.store(0, Ordering::Relaxed);
        self.inner.epoch.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}
