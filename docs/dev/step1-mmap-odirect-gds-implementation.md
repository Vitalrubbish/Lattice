# Step 1: mmap, O_DIRECT, and GDS Loader Implementation

## Summary

Implemented three alternative model-weight loading methods alongside the existing `read()` baseline:

| Loader | Mechanism | Kernel path |
|--------|-----------|-------------|
| `read` (existing) | `read(2)` + `cudaMemcpy` H→D | VFS → page cache → block I/O → NVMe |
| `mmap` | `mmap(2)` + page fault + `cudaMemcpy` | filemap_fault → readpage → submit_bio → NVMe |
| `direct` | `O_DIRECT` `pread(2)` + `cudaMemcpy` | generic_file_direct_read → submit_bio → NVMe |
| `gds` | `cuFileRead` (NVMe→GPU DMA) | cuFile driver → NVMe CMB → GPU BAR (no CPU) |

## Files Changed

### Cargo.toml
- Added `memmap2 = "0.9"` for the mmap loader
- Added `[features]` section with `gds = []` feature flag
- Note: aligned buffers use `std::alloc` directly (no `aligned` crate needed)

### src/model/loader.rs
Complete rewrite (~830 lines). Key additions:

1. **`AlignedBuffer` struct** — Safe heap-allocated buffer with configurable alignment, using `std::alloc::Layout` for correct deallocation. Implements `Deref<Target=[u8]>`.

2. **`load_with_mmap()`** — Maps each safetensors shard with `memmap2::Mmap::map()`. The `SafeTensors::deserialize` parses headers from the mmap'd region. `cudaMemcpy` then reads from the mmap'd pages, triggering major page faults on first access that drive the actual NVMe reads via `filemap_fault`.

3. **`load_with_direct()`** — Opens files with `O_DIRECT` flag. Uses `read_whole_file_direct()` which reads in 2 MiB aligned chunks using `libc::pread`. The final partial block (file size not multiple of 4096) is read via a non-O_DIRECT fd. Returns an `AlignedBuffer` whose logical length equals the true file size.

4. **`load_with_gds()`** (feature-gated) — Opens the cuFile driver, registers each file fd, then:
   - cuFileRead the first 256 KB to GPU → D→H copy to CPU → parse JSON to get tensor offsets
   - For each tensor, cuFileRead directly from NVMe to GPU VRAM (no CPU staging)
   - Falls back to a clear error if the `gds` feature is not enabled

5. **`gds_helpers` module** (feature-gated) — `parse_safetensors_header()` parses the safetensors JSON metadata without needing the full file buffer in CPU memory. Returns `GdsTensorMeta` with name, dtype, shape, and byte offsets.

6. **`gds_ffi` module** (feature-gated) — Raw FFI bindings to `libcufile.so`: `cuFileDriverOpen`, `cuFileDriverClose`, `cuFileHandleRegister`, `cuFileHandleDeregister`, `cuFileRead`. Includes `AsDevicePtr` trait to extract `u64` device pointers from `CudaSlice<u8>`.

## Data Path Comparison

| Aspect | `read()` | `mmap` | `O_DIRECT` | `cuFileRead` |
|--------|----------|--------|------------|-------------|
| Kernel VFS entry | `vfs_read` | `filemap_fault` (page fault) | `generic_file_direct_read` | none |
| Page cache | populated | populated via fault | bypassed | bypassed |
| CPU copies | DMA→DRAM + copy_to_user + cudaMemcpy | DMA→DRAM + cudaMemcpy | DMA→user buffer + cudaMemcpy | DMA→GPU (zero CPU copies) |
| Double buffering | yes (page cache + Vec<u8>) | no (page == user page) | no (single user buffer) | no (single GPU buffer) |
| Warm-cache benefit | yes | yes | no | no |

## Compilation

```bash
# Without GDS (default)
cargo build --release

# With GDS support (requires nvidia-fs-dkms, libcufile-dev)
cargo build --release --features gds
```
