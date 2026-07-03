# Make freeing CpuResident blocks safe

Status: done
Type: AFK

## What to build

Make sequence cleanup safe when a Block Table contains blocks that have already been evicted from GPU memory and are CpuResident. Freeing a sequence should release logical block indices and CPU swap slots exactly once, without returning the same BlockHandle to the GPU Free List twice.

This must make benchmark cleanup paths safe under memory pressure and protect normal KCMM users from Free List corruption after eviction.

## Acceptance criteria

- [ ] Freeing a sequence with only GpuResident blocks still returns those physical blocks to the GPU Free List.
- [ ] Freeing a sequence with CpuResident blocks does not return already released BlockHandles to the GPU Free List again.
- [ ] Freeing CpuResident blocks releases or accounts for their CPU swap space so long-running workloads do not leak CPU swap.
- [ ] A regression test covers freeing a sequence after at least one block has been evicted.
- [ ] Existing KCMM unit tests and benchmark compile checks pass.

## Blocked by

None - can start immediately.

## Comments

