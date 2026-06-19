# Add Phase II.A stock vs KCMM A/B gate

Status: done
Type: AFK

## What to build

Add a repeatable Phase II.A A/B gate that compares stock vLLM, KCMM observer,
KCMM shadow allocator, and KCMM-backed allocator modes on the same smoke model
and command shape.

The goal is not a full benchmark suite yet. The gate should make regressions
visible before Phase II.B starts: server startup success, completion success,
request latency, token throughput, GPU memory footprint, and KCMM allocation
stats should be captured in a single report.

## Acceptance criteria

- [x] The gate runs stock vLLM and all enabled KCMM modes with the same prompt and generation parameters.
- [x] The report records success/failure, startup time, request latency, generated token count, GPU memory usage, and KCMM stats where applicable.
- [x] The gate fails if observer or shadow modes cannot produce a completion when stock vLLM can.
- [x] The gate fails if KCMM-backed allocator mode leaks KCMM blocks or leaves the smoke server running.
- [x] The report clearly distinguishes performance warnings from correctness failures.
- [x] The gate can be run locally without downloading a large model.
- [x] The Phase II.A gate result is documented as the prerequisite for starting Phase II.B.

## Implementation

- Added `scripts/kcmm/vllm_ab_gate.py`.
- Extended `scripts/kcmm/vllm_smoke.py` with `nvidia-smi` GPU memory sampling.
- Documented the gate command and Phase II.B prerequisite in
  `docs/dev/kcmm-vllm-cu118-env.md` and
  `docs/adr/0001-vllm-integration-architecture.md`.

## Validation

- `python -m py_compile scripts/kcmm/*.py`
- `python -m scripts.kcmm.vllm_ab_gate`

The local Phase II.A A/B gate passed on 2026-06-19:

- `passed=true`
- `correctness_failures=[]`
- `performance_warnings=[]`
- Completed modes: `stock`, `observer`, `shadow`, `backed`
- Every mode generated 4 completion tokens.
- Shadow and backed reports both recorded `kcmm_allocations=1`,
  `kcmm_frees=1`, `outstanding_mappings=0`, and `error_count=0`.
- Backed mode recorded `blocks_in_use=0` after shutdown.
- Both RTX 3080 GPUs returned to 0 MiB and port `8001` was free after the run.

## Blocked by

- `.scratch/kcmm-vllm-phase-ii-a/issues/01-make-vllm-smoke-runs-self-terminating.md`
- `.scratch/kcmm-vllm-phase-ii-a/issues/06-enable-kcmm-backed-v2-allocator-behind-flag.md`
