# Clarify memory pressure throughput metric headers

Status: done
Type: AFK

## What to build

Issue #06 already renamed the metric from `throughput_ratio` to `completion_ratio` and added separate `elapsed_throughput` reporting. This issue is a small follow-up cleanup: the sweep table still uses headers `ThrB/s` and `ThrK/s`, which look like "tokens per second". The numbers are actually "completed sequences per second".

Rename the headers to something unambiguous such as `ComplB/s` and `ComplK/s` (or `SeqB/s` / `SeqK/s`) and update the single-config output to explicitly say "completed sequences per second". Update any summary text or script quick-metrics that reference these columns.

## Acceptance criteria

- [ ] Sweep table headers no longer resemble tokens/sec throughput.
- [ ] Single-config log output explicitly labels the elapsed throughput as "completed sequences per second".
- [ ] Any script grep patterns or documentation that parse these columns are updated.
- [ ] `kcmm_bench_memory_pressure_single` and `_sweep` are re-run to confirm output format.

## Verification

- `cargo test --features kcmm --release --test kcmm_bench_memory_pressure kcmm_bench_memory_pressure_single -- --nocapture`
- `cargo test --features kcmm --release --test kcmm_bench_memory_pressure kcmm_bench_memory_pressure_sweep -- --nocapture`
- Sweep output now uses `ComplB/s` and `ComplK/s`; single output says `completed sequences/s`.

## Blocked by

None - can start immediately. This is a cosmetic cleanup on top of completed issue #06.
