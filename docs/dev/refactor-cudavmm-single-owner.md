# Refactor: Remove duplicated CudaVmm from PhysicalBlockAllocator

**2026-05-28**

## What changed

Removed `vmm: CudaVmm` field from `PhysicalBlockAllocator`. The `allocate()` method now takes `&CudaVmm` as a parameter, borrowed from the single VMM owner — `PagedKvCache`.

## Why

Both `PagedKvCache` and `PhysicalBlockAllocator` owned separate `CudaVmm` instances, each wrapping the same `device: usize`. Since `CudaVmm` is stateless (all state lives in the CUDA driver), the duplication was purely architectural confusion with no functional benefit.

`PhysicalBlockAllocator` only uses VMM for `create_physical()` when a new superblock is needed. `PagedKvCache` uses VMM for `reserve_address()`, `map()`, and `unmap()`. By making `PagedKvCache` the sole VMM owner and having the allocator borrow it on demand, the ownership boundary is now clear: VMM lifecycle management lives in one place.

## Changes

- `PhysicalBlockAllocator`: removed `vmm` field, removed `device` param from `new()`, `allocate()` now takes `&CudaVmm`
- `PagedKvCache::new()`: updated allocator construction call
- `PagedKvCache::alloc_sequence()`: passes `&self.vmm` to `allocate()`
