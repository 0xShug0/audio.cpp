# Common slot framework extraction

The reusable server/session framework is extracted from PR #706 onto upstream
`77491a33` (2026-09-27). Model implementations, checkpoint-specific guards,
codec fixes and audited admission lists remain in the model follow-up, #706.
The framework alone has empty admission tables and keeps existing models at
one slot. Explicit parallel factories and an audited loaded-model fallback are
available to model adapters; requesting an unsupported count fails before the
pool is published. Continuous token batching is outside this implementation.

## Common implementation

- Per-model `slots` (1-16, default one), slot leases, queues, busy timeouts,
  management priority and exclusive unload/eviction/reconfiguration.
- `IParallelVoiceTaskSessionFactory` and `VoiceTaskSessionPool`, validating clone
  interfaces/capacity and preserving primary/clone lifetime on partial failure.
- Private per-slot prepare/run/stream/native-batch dispatch, serialized first
  publication, failed-load retry, live status and drained management operations.
- Package-owned immutable weight-cache ownership, coordinated first loading,
  failed-load retry and tensor-source mapping protection during concurrent reads,
  copies and storage release. No model-specific arithmetic or GPU kernels.
- The overdue-request regression fix: waiting management cannot suppress the
  overdue check. Spare slots blocked by management are covered as well as full
  pools; healthy blockers and timeout zero retain waiting behavior.

## Fresh standalone validation

Windows/MSVC Release, eight CPU inference threads, RTX 3090 (24 GiB), CUDA
device 0 and Vulkan device 1. CPU, CUDA and Vulkan builds include the full model
set from the current upstream base; CLI and server builds succeed on all three.

| Build | Focused tests | Result |
|---|---:|---|
| CPU | 7 | All pass |
| CUDA | 7 | All pass |
| Vulkan | 8 | All pass |

The seven common tests are `server_single_slot_compat_test`,
`server_model_slots_test`, `session_pool_test`, `shared_weight_cache_test`,
`gguf_tensor_source_test`, `server_config_test` and `server_busy_guard_test`.
Vulkan additionally runs `vulkan_shared_weight_cache_test`. Pool test doubles
exercise parallel offline/native batches, streaming state/reset, independent
sessions, immutable ownership, clone rollback and destruction order. Scheduler
tests cover one/two/four slots, management priority and multiple waiting unloads.
The server lifecycle CI workflow builds/runs the seven common tests across
Linux, macOS and Windows; new remote CI results are pending publication.

Real Canary 180M Flash Q8_0 HTTP smoke checks pass on CPU, CUDA and Vulkan:
default single-slot behavior, repeated exact transcripts, clean two-slot
capacity rejection with no published pool/lease, invalid-input recovery,
unload/reload and drained request counters. Five successful inference responses
and two expected errors per backend are retained in the local smoke results.

Fresh same-backend Canary overdue-request probes compare the saved pre-fix PR
server at `2e381d02` with the standalone framework. The 23.8-second speech
fixture and incoming-request 15 ms timeout are described in the
[scheduler report](scheduler_busy_timeout.md). Medians of three repeats:

| Backend | Unload waiting | Before rejection (ms) | Standalone framework (ms) |
|---|---|---:|---:|
| CUDA | no | 1.562 | 1.676 |
| CUDA | yes | 28.498 | 1.673 |
| Vulkan | no | 1.759 | 1.875 |
| Vulkan | yes | 17.435 | 1.704 |

Each backend passes twelve rejection probes and six unload/reload sequences.
All active inference transcripts match their serial references after excluding
only response timing. These are rejection-latency measurements, not generation
speedups. The older executable is a saved PR build, not pristine upstream.

Real multi-slot model inference belongs to #706; the standalone framework does
not enable arbitrary legacy models. The historical combined-branch multi-slot
measurements in the scheduler report retain that distinction. No real parallel
streaming/native-batch adapter is certified by this extraction. Cross-platform
CI, sanitizers and every possible lifecycle interleaving are not claimed here.

The [parallel model validation procedure](../maintainers/parallel_model_validation.md)
now groups requirements into six shared framework entries (F1-F6) and six
model entries (M1-M6). Shared ownership/transition evidence is collected once per
applicable revision/backend and referenced by models; per-model output failures
remain separate from missing shared coverage. Model runs also sample memory,
avoiding a duplicate identical quality/lifecycle matrix. Performance has its own
verdict, and independent-server comparisons are optional.

Quality requirements are retained: exact outputs/meaningful metadata,
equivalent cache histories where applicable, three fresh starts at every
advertised count, ten mixed warm waves per start at the largest count,
overlap/counter checks, recovery and unload/reload. CPU checkpoint tests depend
on whether CPU behavior is affected; shared compatibility checks still apply.
Framework disconnect/reconfiguration/load-failure/timeout/cross-model/shutdown
and supported sanitizer evidence remain required once, with representative
backend integration. Missing relevant shared evidence still blocks admission.

This documentation update adds no completed test results, runtime changes or
model admission. Historical reports retain their original procedure/build
identity until their evidence is mapped to the M/F records; changing the plan
does not turn pending checks or known AuK/Apollo mismatches into passes.

## Reproduction and evidence

Enable `ENGINE_BUILD_TESTS=ON` and `ENGINE_BUILD_EXTENDED_TESTS=ON`, build the CLI,
server and the focused targets above, then run those named CTests. The common
HTTP harness is `tests/server/server_busy_timeout_regression.py`; use
`--slots 1 --repeats 3` for the standalone framework and the selected backend's
device index. Higher counts require the model-support follow-up.

Local evidence is outside the repository under
`outputs/pr706-framework-split/`: `ctest-{cpu,cuda,vulkan}-framework.log`,
`smoke-results.json`, `canary-{cuda,vulkan}/results.json`, per-server logs/configs,
PCM fixture and executable hashes. Build logs are retained in
`outputs/scheduler-timeout-fix/build-{cpu,cuda,vulkan}-framework.log`.
