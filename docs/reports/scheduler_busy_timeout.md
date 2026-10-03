# Slot scheduler: overdue rejection while management waits

Historical implementation/evidence report. The
[Parallel Server Behavior Contract](../maintainers/parallel_server_behavior_contract.md)
is authoritative for current expected behavior, tested scopes and known limits;
this report retains its recorded revision's results.


## Problem and fix

An incoming request could wait for its complete `busy_timeout_ms` even when all
active inference slots were already overdue. A pending unload set
`waiting_exclusive_`, which suppressed the scheduler's fail-fast overrun check.
This is HTTP 503 rejection latency, not model generation speed.

Check overdue occupied inference independently of management priority. Ignore
unoccupied slots in that check because waiting management blocks their admission.
Keep exclusive-operation protection, healthy-slot waiting, timeout-zero behavior,
lease ownership, queue counters and wake-up handling. Read the clock once per
overrun scan. Model code, CUDA/Vulkan kernels, slot limits and precision are unchanged.

## Real HTTP validation

Windows/MSVC, RTX 3090, eight CPU threads. CUDA device 0; Vulkan device 1 in this
machine's backend enumeration (device 0 is the AMD integrated GPU). Canary
180M Flash Q8_0, offline ASR, the same checkpoint and model spec before/after.
The baseline is the executable saved at PR #706 revision `2e381d02` before this
scheduler edit, not pristine upstream. Compare within each backend.

The fixture is 23.8 seconds of 16 kHz mono PCM speech (four unchanged copies of
the existing speech fixture). This keeps warm inference active through the
rejection probe. It is longer than the maintainer's 14.2-second fixture. Request
B uses `busy_timeout_ms=15`; normal inference and unload use their existing
long/unbounded waits. Management is `unload_all_models`. Values are median
HTTP round-trip rejection times over five repeats per case, in milliseconds.

| Backend | Slots | Active inference slots | Unload waiting | Before (ms) | After (ms) |
|---|---:|---:|---|---:|---:|
| CUDA | 1 | 1 | no | 1.654 | 1.665 |
| CUDA | 1 | 1 | yes | 28.047 | 1.687 |
| CUDA | 2 | 2 | no | 1.817 | 1.882 |
| CUDA | 2 | 2 | yes | 28.376 | 1.816 |
| CUDA | 2 | 1 | yes | 28.349 | 1.554 |
| CUDA | 4 | 4 | no | 1.936 | 1.997 |
| CUDA | 4 | 4 | yes | 28.621 | 1.857 |
| CUDA | 4 | 1 | yes | 29.872 | 1.627 |
| Vulkan | 1 | 1 | no | 1.811 | 1.690 |
| Vulkan | 1 | 1 | yes | 17.370 | 1.729 |
| Vulkan | 2 | 2 | no | 1.928 | 2.105 |
| Vulkan | 2 | 2 | yes | 17.447 | 1.822 |
| Vulkan | 2 | 1 | yes | 17.413 | 1.762 |
| Vulkan | 4 | 4 | no | 2.083 | 2.181 |
| Vulkan | 4 | 4 | yes | 17.453 | 1.983 |
| Vulkan | 4 | 1 | yes | 17.326 | 1.627 |

Each backend passes 80 rejected-request probes (40 before/40 after). All 160
active inference responses per backend exactly match their corresponding serial
references, excluding only the nondeterministic response `timing` field. All 50
unload/reload sequences per backend succeed and reproduce reference transcripts;
active and queued request counters return to zero. Cold/warm baseline and
candidate transcripts also match. This establishes parity for these fixtures,
not a universal determinism guarantee for other checkpoints or requests.

The harness checks that inference and unload are still pending when B arrives,
and checks the rejection reason, not just a timing threshold. Baseline probes
with waiting unload report a queue timeout; candidate probes report already
overdue active slots. The unit tests synchronize on the management queue state.
HTTP timing includes client/network/worker overhead; timer granularity can make
a full 15 ms wait longer than 15 ms.

## Regression tests and build

The new one-slot regression failed against the original scheduler with
`waiting management suppressed overdue rejection`, then passed after the fix.
Tests also cover full two/four-slot pools, spare slots blocked by management,
two waiting unloads, healthy and overdue slots together, timeout zero,
management priority, queue drainage and lease release. Existing tests retain
coverage for concurrent slot isolation, bounded queue waits, exclusive unload,
exceptions, manager timeouts and one-slot compatibility.

CUDA and Vulkan server builds pass. The five focused CTests pass on both build
configurations: `server_model_slots_test`, `server_single_slot_compat_test`,
`session_pool_test`, `server_busy_guard_test` and `server_config_test`. These
targets already run in the cross-platform server lifecycle CI workflow.
Cross-platform CI and real CPU inference have not been rerun for this local fix.

## Reproduction and evidence

From the repository root, substitute paths to both executables, model and WAV:

```powershell
python tests/server/server_busy_timeout_regression.py `
  --before-server <before.exe> --after-server <after.exe> `
  --model <canary-180m-flash-q8_0.gguf> --spec model_specs/canary_asr.json `
  --audio <speech-16k-mono.wav> --backend cuda --device 0 `
  --output-dir <audit-directory> --repeats 5 --slots 1
```

Repeat for `--backend vulkan` and that backend's device index. The script saves
server configs/logs, fixture PCM, executable SHA-256 provenance, responses and
request timings in `results.json`. Local evidence is retained outside the repo
under `outputs/scheduler-timeout-fix/cuda/` and
`outputs/scheduler-timeout-fix/vulkan-rtx3090/`; build/CTest logs and aggregate
`summary.json` are in the parent directory. An interrupted AMD Vulkan diagnostic
run is excluded from the table and validation counts.

## Framework split scope

The historical multi-slot measurements above were taken with the PR #706 model
adapters/admission rules present. The extracted foundation keeps those rules
empty; reproduce one-slot controls here, and use `--slots 1 2 4` on the model
follow-up. The scheduler implementation is identical. Fresh standalone build
and inference checks are recorded in [the framework split report](common_slot_framework.md).
