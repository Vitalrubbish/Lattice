# KCMM Week 13 Wrap-Up Fix Checklist

> **Source**: Analysis conclusions from `docs/task/kcmm-code-review-fixes.md`
> **Date**: 2026-06-05
> **Branch**: `kcmm`
> **Goal**: Fix blocking issues before Week 14 begins

## Fix Overview

| Priority | Issue | Severity | Affected Files | Estimated Effort |
|----------|-------|----------|---------------|-----------------|
| **P0** | #1 — Deadlock: `install_block` and `free_sequence` have inconsistent lock order | 🔴 Critical | `src/kcmm/pool.rs`, `src/cache/paged_kv.rs` | 30 min |
| **P0** | #3 — `TieringEngine` ignores `cpu_cache_path`, uses anonymous mmap | 🔴 Critical | `src/kcmm/tiering.rs` | 1 hour |
| **P0** | #7 — KCMM module is not feature-gated | 🟠 High | `src/lib.rs`, possibly `src/config.rs` | 30 min |
| **P3** | #15 — `KcmmPool::Drop` does not wait for CUDA Stream completion | 🟢 Low | `src/kcmm/pool.rs` | 10 min |

---

## P0-1: Deadlock — Unify Lock Order in `install_block` and `free_sequence`

### Problem Description

`install_block` and `free_sequence` acquire the `free_block_indices` and `block_info` `parking_lot::Mutex` locks in **opposite order**:

```
install_block:   free_block_indices.lock() → block_info.lock()
free_sequence:   block_info.lock()          → free_block_indices.lock()
```

Two threads calling these functions simultaneously will cause an AB-BA deadlock. This issue exists **in both files**.

### Affected Files

- `src/kcmm/pool.rs:348-420`
- `src/cache/paged_kv.rs:250-325`

### Fix Plan

In `free_sequence`, collect the `block_idx` values to recycle into a temporary `Vec`, release the `block_info` lock, then push them all at once to `free_block_indices`.

**Fix for kcmm/pool.rs:**

```rust
fn free_sequence(&self, block_table: &[u32]) {
    let mut info = self.block_info.lock();
    let num_layers = self.num_layers;
    let mut recycled = Vec::new();

    for &block_idx in block_table {
        let bi = &mut info[block_idx as usize];
        if !bi.in_use {
            continue;
        }
        bi.in_use = false;

        let handle = BlockHandle {
            superblock_idx: bi.superblock_idx,
            block_index: bi.block_index_in_sb,
        };
        for l in 0..num_layers {
            self.k_pools[l].allocator.free(handle);
            self.v_pools[l].allocator.free(handle);
        }
        recycled.push(block_idx);
    }
    drop(info);                                      // Release Lock B
    self.free_block_indices.lock().extend(recycled); // Acquire Lock A
}
```

**Fix for paged_kv.rs:** Same logic, adapted to its field names (`free_block_indices` and `block_info` fields are already consistent).

### Verification

- Compilation passes
- Existing unit tests pass (`cargo test -p baseline-llm-os --lib`)
- Future: add concurrent stress tests (e.g., `loom` or proptest)

---

## P0-3: `TieringEngine` — Replace Anonymous mmap with File-Based mmap

### Problem Description

`TieringEngine::new` uses `MAP_ANONYMOUS`, completely ignoring the `KcmmConfig::cpu_cache_path` configuration:

```rust
// Current code (tiering.rs:109-118)
let ptr = unsafe {
    libc::mmap(
        std::ptr::null_mut(),
        cpu_buffer_size,
        libc::PROT_READ | libc::PROT_WRITE,
        libc::MAP_SHARED | libc::MAP_ANONYMOUS, // ← ignores cpu_cache_path
        -1,                                      // ← fd = -1
        0,
    )
};
```

Impact:
- Cannot share CPU swap buffer across processes (violates KCMM's core multi-engine memory management design)
- Cannot persist swap data on disk
- `config.cpu_buffer_size` does not actually exist (a rough estimate of `max_blocks * block_size * 2` is used instead)

### Affected Files

- `src/kcmm/tiering.rs:107-133`

### Fix Plan

Use `std::fs::OpenOptions` to create/open a file, then `mmap` via the file descriptor:

```rust
use std::fs::OpenOptions;
use std::os::unix::io::AsRawFd;

pub fn new(config: &KcmmConfig) -> Result<Self> {
    // Use buffer size from config, or a reasonable estimate
    let cpu_buffer_size = config.max_blocks * config.block_size * 2; // TODO: add KcmmConfig::cpu_buffer_size

    let cpu_buffer = if cpu_buffer_size > 0 {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&config.cpu_cache_path)?;
        file.set_len(cpu_buffer_size as u64)?;

        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                cpu_buffer_size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                file.as_raw_fd(),
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(anyhow::anyhow!(
                "mmap CPU swap buffer '{}' failed: {}",
                config.cpu_cache_path,
                std::io::Error::last_os_error()
            ));
        }
        ptr as *mut u8
    } else {
        std::ptr::null_mut()
    };

    Ok(Self {
        cpu_buffer,
        cpu_buffer_size,
        nvme_enabled: false,
    })
}
```

**Note**: Currently `KcmmConfig` has no `cpu_buffer_size` field. Options:
1. Add a `cpu_buffer_size` field to `KcmmConfig` (recommended, more flexible)
2. Continue using `max_blocks * block_size * 2` estimate (simpler)

### Verification

- Compilation passes
- Check that `/dev/shm/kcmm_swap` file is correctly created and mapped

---

## P0-7: KCMM Module Feature-Gating

### Problem Description

`Cargo.toml` defines a `kcmm` feature flag, but `lib.rs` unconditionally includes the KCMM module:

```rust
// lib.rs:6 — current
pub mod kcmm;  // ← Always compiled, ignores the feature flag

// config.rs — KcmmConfig also exported unconditionally
pub use config::{KcmmConfig, ModelConfig, ServerConfig};
```

The design document (Backward Compatibility Constraint 0.5) explicitly requires:
> "KCMM enabled as an optional feature flag: `cargo build --features kcmm` compiles the KCMM module and produces `libkcmm.so`"

### Affected Files

- `src/lib.rs:6,10`
- May affect `src/cache/paged_kv.rs:11` (imports types from `kcmm::superblock`)
- May affect `src/cache/cuda_vmm.rs` (if referenced by kcmm)

### Fix Plan

**Step 1:** Add feature gate in `lib.rs`:

```rust
pub mod batch;
pub mod cache;
pub mod config;
pub mod cuda;
pub mod decoder;
#[cfg(feature = "kcmm")]
pub mod kcmm;
pub mod model;
pub mod server;

#[cfg(feature = "kcmm")]
pub use config::KcmmConfig;
pub use config::{ModelConfig, ServerConfig};
```

**Step 2:** Check `src/cache/paged_kv.rs:11`:

```rust
use crate::kcmm::superblock::{
    align_up, BlockHandle, LayerKvPool, SuperblockInfo, SUPERBLOCK_SIZE,
};
```

`paged_kv.rs` must still compile when the `kcmm` feature is not enabled. Options:
- Move `superblock`-related types (`BlockHandle`, `LayerKvPool`, `SuperblockInfo`, `SUPERBLOCK_SIZE`, `align_up`) under `src/cache/`, or keep them in the `kcmm` module but reference them via conditional compilation in `paged_kv.rs`
- Or re-export these types from `src/cache/mod.rs` under `#[cfg(not(feature = "kcmm"))]`

**Recommended approach**: `PhysicalBlockAllocator` and related types (`BlockHandle`, `SuperblockInfo`, etc.) were always designed to be shared base types extracted from `paged_kv.rs`. Keeping them in `kcmm/superblock.rs` (controlled by the kcmm feature gate) while `paged_kv.rs` directly references them means the two are already coupled. Short-term solution:

```rust
// paged_kv.rs
#[cfg(feature = "kcmm")]
use crate::kcmm::superblock::{...};
#[cfg(not(feature = "kcmm"))]
// Fall back to internal definitions in paged_kv (or re-export from a cache submodule)
```

> **More thorough approach** (recommended for Week 14): Move `PhysicalBlockAllocator` and other shared types under `src/cache/` as a neutral module, referenced by both `kcmm` and `paged_kv`.

### Verification

```bash
# Build without kcmm feature (should succeed, no kcmm module)
cargo build

# Build with kcmm feature (should succeed, includes kcmm module + cdylib)
cargo build --features kcmm

# Run tests
cargo test --features kcmm
```

---

## P3-15: Defensive CUDA Stream Synchronization in `KcmmPool::Drop`

### Problem Description

`KcmmPool::Drop` (pool.rs:696-727) directly unmaps physical handles without first synchronizing CUDA streams. If there are outstanding async memcpy operations (added in Week 14), physical memory could be reclaimed during an in-flight transfer, causing use-after-free.

### Affected Files

- `src/kcmm/pool.rs:696-727`

### Fix Plan

Add stream synchronization at the beginning of Drop (defensive programming):

```rust
impl Drop for KcmmPool {
    fn drop(&mut self) {
        // Wait for all outstanding CUDA stream operations to complete
        self.streams.synchronize_all().ok();

        let num_layers = self.num_layers;
        // ... existing unmap/release logic unchanged ...
    }
}
```

### Verification

- Compilation passes
- After async memcpy is added in the future, verify the Drop path via valgrind/cuda-memcheck

---

## Recommended Execution Order

```
1. P0-1 Deadlock fix           (30 min) — concurrency correctness, affects two files
2. P0-7 Feature gating         (30 min) — build system, affects module visibility
3. P0-3 File-based mmap        (1 hr)   — TieringEngine, depends on #7 completion for build config
4. P3-15 Drop defensive sync   (10 min) — can be committed together with any of the above
```
