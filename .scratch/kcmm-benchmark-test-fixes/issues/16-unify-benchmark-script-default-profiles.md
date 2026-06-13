# Unify default build profile across benchmark runner scripts

Status: done
Type: AFK

## What to build

`scripts/run_kcmm_benches.sh` defaults to `release`, while `scripts/run_kcmm_integration_bench.sh` defaults to `debug`. This is an easy footgun: numbers produced by the two scripts are not comparable unless the user explicitly passes `--release` to the integration script.

Unify the default profile across both scripts. The release profile is preferred for benchmark numbers; if debug is intentionally kept as default for the integration script, document why and add a prominent warning in the help text.

## Acceptance criteria

- [ ] Both `run_kcmm_benches.sh` and `run_kcmm_integration_bench.sh` use the same default profile.
- [ ] Script help text and header comments reflect the chosen default.
- [ ] If `debug` is kept, the help text warns that release is required for performance comparisons.
- [ ] The integration benchmark quick-metrics grep patterns still extract the expected lines.
- [ ] Both scripts are smoke-tested to ensure they still launch the correct tests.

## Verification

- `scripts/run_kcmm_benches.sh --help`
- `scripts/run_kcmm_integration_bench.sh --help`
- `scripts/run_kcmm_integration_bench.sh --single`
- `scripts/run_kcmm_integration_bench.sh --sweep`
- Both scripts now default to release; integration quick metrics still extract the expected ratio/status lines.

## Blocked by

None - can start immediately.
