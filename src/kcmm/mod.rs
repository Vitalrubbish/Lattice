// KCMM (KV Cache Memory Manager) — an OS-style GPU KV Cache memory management service.
//
// Architecture:
//   superblock  — PhysicalBlockAllocator, SuperblockInfo, BlockHandle (extracted from paged_kv)
//   pool        — KcmmPool: lifecycle, block alloc/free, sequence tracking
//   streams     — CudaStream wrappers for evict/restore/prefetch
//   tiering     — TieringEngine: GPU↔CPU↔NVMe migration, EvictionPolicy trait
//   metrics     — UFS-compatible fragmentation metrics (IFR, PME, BU, RFI)
//   sharing     — Prefix sharing manager (step 4, skeleton in step 3)
//   ffi         — C ABI exports for libkcmm.so
//
// `superblock` is always compiled — it contains the physical-block allocator
// types shared with `PagedKvCache`.  The remaining modules are gated behind
// the `kcmm` feature flag so that the existing inference path is unaffected
// when the flag is off.

pub mod superblock;

#[cfg(feature = "kcmm")]
pub mod pool;
#[cfg(feature = "kcmm")]
pub mod streams;
#[cfg(feature = "kcmm")]
pub mod tiering;
#[cfg(feature = "kcmm")]
pub mod metrics;
#[cfg(feature = "kcmm")]
pub mod sharing;
#[cfg(feature = "kcmm")]
pub mod ffi;

pub use superblock::{BlockHandle, PhysicalBlockAllocator, SuperblockInfo, SUPERBLOCK_SIZE};

#[cfg(feature = "kcmm")]
pub use pool::KcmmPool;
#[cfg(feature = "kcmm")]
pub use tiering::EvictionPolicy;
