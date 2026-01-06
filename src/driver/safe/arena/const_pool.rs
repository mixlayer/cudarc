use std::{
    any::TypeId,
    collections::HashMap,
    hash::{Hash, Hasher},
    sync::{Arc, RwLock},
};

use crate::driver::{
    core::{ArenaAllocType, CudaAllocType},
    safe::{DeviceRepr, HostSlice},
    sys::CUdeviceptr,
    CudaSlice, CudaStream, DriverError,
};

fn blake3_32(bytes: &[u8]) -> [u8; 32] {
    blake3::hash(bytes).into()
}

#[derive(Clone, Copy, Eq)]
struct InternKey {
    type_id: TypeId,
    len_elems: usize,
    hash: [u8; 32],
}

impl PartialEq for InternKey {
    fn eq(&self, other: &Self) -> bool {
        self.type_id == other.type_id
            && self.len_elems == other.len_elems
            && self.hash == other.hash
    }
}
impl Hash for InternKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.type_id.hash(state);
        self.len_elems.hash(state);
        self.hash.hash(state);
    }
}

/// One interned allocation.
///
/// We store:
/// - canonical host bytes (for collision disambiguation)
/// - the raw device pointer (stable address for graphs)
/// - element count (for upgrade_device_ptr)
/// - an Arc<CudaStream> so we can free later if you ever add eviction/Drop logic.
///
/// NOTE: This design assumes your "interned" allocations are *not freed* when
/// the returned CudaSlice<T> is dropped (e.g., arena-managed drop path).
struct InternEntry {
    bytes: Arc<[u8]>,
    ptr: CUdeviceptr,
    len_elems: usize,
    stream: Arc<CudaStream>,
}

pub struct ConstInternCache {
    // (type, len, hash) -> potentially multiple entries (rare collisions)
    map: RwLock<HashMap<InternKey, Vec<Arc<InternEntry>>>>,
}

impl Default for ConstInternCache {
    fn default() -> Self {
        Self {
            map: RwLock::new(HashMap::new()),
        }
    }
}

impl ConstInternCache {
    /// Intern + upload host data and return a device slice.
    ///
    /// This is shaped to drop-in behind a `clone_htod`-style API.
    ///
    /// ## IMPORTANT
    /// This only works correctly if the returned CudaSlice<T> will *not* free
    /// `ptr` on drop (e.g. arena/const-pool managed). With upstream cudarc,
    /// a plain CudaSlice<T> will free; so you either:
    ///   - integrate this with your arena changes, OR
    ///   - change the API to return Arc<InternEntry> (graph holds the Arc).
    pub fn clone_htod_interned<T, Src>(
        &self,
        stream: &Arc<CudaStream>,
        src: &Src,
    ) -> Result<CudaSlice<T>, DriverError>
    where
        T: DeviceRepr + 'static,
        Src: HostSlice<T> + ?Sized,
    {
        // Respect cudarc host/device sync rules.
        let (host, _guard) = unsafe { src.stream_synced_slice(stream) };

        // Serialize to bytes for hashing + equality checks.
        // SAFETY:
        // - `host` is a real &[T] and thus points to initialized Ts.
        // - For your use-case (dims/strides/scalars), T should be POD-like
        //   (no padding / no uninit bytes). If T can contain padding, hashing
        //   raw bytes can be problematic. In practice, your dtypes (u32, i64,
        //   f16, bf16, f32, etc.) are fine.
        let host_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                host.as_ptr() as *const u8,
                host.len() * std::mem::size_of::<T>(),
            )
        };

        let key = InternKey {
            type_id: TypeId::of::<T>(),
            len_elems: host.len(),
            hash: blake3_32(host_bytes),
        };

        // Fast path: read-lock lookup + byte compare.
        if let Some(existing) = self.lookup_existing(&key, host_bytes) {
            // SAFETY: ptr was originally created from a valid allocation of len_elems T's.
            // This matches cudarc's intended leak/upgrade round-trip usage.
            let slice = unsafe { stream.upgrade_device_ptr::<T>(existing.ptr, existing.len_elems) };
            return Ok(slice);
        }

        // Slow path: allocate/upload, then store.
        let mut map = self.map.write().unwrap();

        // Re-check under write lock to avoid racing inserts.
        if let Some(vec) = map.get(&key) {
            if let Some(existing) = vec.iter().find(|e| e.bytes.as_ref() == host_bytes) {
                let slice =
                    unsafe { stream.upgrade_device_ptr::<T>(existing.ptr, existing.len_elems) };
                return Ok(slice);
            }
        }

        // Allocate + upload normally.
        let dst: CudaSlice<T> = stream.clone_htod(host)?;

        // Take ownership of the raw pointer so we can keep a stable address.
        // `leak()` is designed for exactly this use case. :contentReference[oaicite:2]{index=2}
        let ptr: CUdeviceptr = dst.leak();

        let entry = Arc::new(InternEntry {
            bytes: Arc::<[u8]>::from(host_bytes),
            ptr,
            len_elems: host.len(),
            stream: stream.clone(),
        });

        map.entry(key).or_default().push(entry);

        // Return a "fresh" handle to the interned ptr.
        // Again: this must not free ptr on drop in your final design.
        let mut slice = unsafe { stream.upgrade_device_ptr::<T>(ptr, host.len()) };

        slice.allocation = CudaAllocType::Arena(ArenaAllocType::Interned);

        Ok(slice)
    }

    fn lookup_existing(&self, key: &InternKey, bytes: &[u8]) -> Option<Arc<InternEntry>> {
        let map = self.map.read().ok()?;
        let vec = map.get(key)?;
        vec.iter().find(|e| e.bytes.as_ref() == bytes).cloned()
    }
}
