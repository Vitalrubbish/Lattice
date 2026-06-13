# Make ballast allocation failure visible in integration benchmark

Status: done
Type: AFK

## What to build

`allocate_gpu_ballast` is meant to give the tiering-OFF run the same physical GPU budget as tiering-ON by reserving the staging buffers that the tiering engine would use. If ballast allocation fails, the function returns `None` and the test continues silently, giving OFF more available GPU memory than ON and invalidating the comparison.

Make the failure visible: either log a clear warning that the OFF/ON comparison may be unfair, or fail the test outright. Document the intended ballast size and why it matters in a code comment near the function.

## Acceptance criteria

- [ ] `allocate_gpu_ballast` logs a clear warning or fails the test if it cannot reserve the staging-buffer equivalent memory.
- [ ] The warning/error message explains that tiering-OFF may have more GPU memory available than tiering-ON without the ballast.
- [ ] A code comment documents the ballast size calculation and its fairness purpose.
- [ ] Existing integration benchmark tests still pass when ballast allocation succeeds.
- [ ] `kcmm_engine_integration_single` and `_sweep` are smoke-tested after the change.

## Verification

- `scripts/run_kcmm_integration_bench.sh --single`
- `scripts/run_kcmm_integration_bench.sh --sweep`
- Ballast allocation now fails the test with an explicit fairness error instead of silently returning `None`.

## Blocked by

None - can start immediately.
