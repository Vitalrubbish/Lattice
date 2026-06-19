# Add Phase II.A stock vs KCMM A/B gate

Status: ready-for-agent
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

- [ ] The gate runs stock vLLM and all enabled KCMM modes with the same prompt and generation parameters.
- [ ] The report records success/failure, startup time, request latency, generated token count, GPU memory usage, and KCMM stats where applicable.
- [ ] The gate fails if observer or shadow modes cannot produce a completion when stock vLLM can.
- [ ] The gate fails if KCMM-backed allocator mode leaks KCMM blocks or leaves the smoke server running.
- [ ] The report clearly distinguishes performance warnings from correctness failures.
- [ ] The gate can be run locally without downloading a large model.
- [ ] The Phase II.A gate result is documented as the prerequisite for starting Phase II.B.

## Blocked by

- `.scratch/kcmm-vllm-phase-ii-a/issues/01-make-vllm-smoke-runs-self-terminating.md`
- `.scratch/kcmm-vllm-phase-ii-a/issues/06-enable-kcmm-backed-v2-allocator-behind-flag.md`
