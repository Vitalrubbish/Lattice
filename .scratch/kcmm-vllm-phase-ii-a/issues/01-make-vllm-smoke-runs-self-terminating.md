# Make vLLM smoke runs self-terminating

Status: done
Type: AFK

## What to build

Create a repeatable smoke-runner for the KCMM vLLM launcher that starts the
known-good tiny local model, waits for the OpenAI-compatible API to become
ready, sends one completion request, and tears down the full vLLM process tree
without leaving a bound port or occupied GPU memory.

This should turn the manually verified Phase I.C server smoke test into an
automation-friendly check that future allocator work can reuse. The important
behavior is end-to-end lifecycle control: build or locate the tiny model, start
the launcher, prove the HTTP path works, then guarantee cleanup even when vLLM's
graceful shutdown hangs.

## Acceptance criteria

- [x] A single command can run the KCMM vLLM smoke test from a clean checkout after the documented conda environment is active.
- [x] The runner starts vLLM through the KCMM launcher with the V2 block manager enabled.
- [x] The runner waits for readiness by querying the API instead of sleeping a fixed amount.
- [x] The runner verifies both model listing and one completion request return HTTP 200.
- [x] The runner terminates the whole subprocess tree and verifies the configured port is no longer listening.
- [x] The runner reports enough stdout/stderr context to diagnose startup failures without opening raw logs manually.
- [x] The runner can run in a mode that skips KCMM observer setup, so stock vLLM and KCMM launcher paths can share the same lifecycle harness.

## Blocked by

None - can start immediately.
