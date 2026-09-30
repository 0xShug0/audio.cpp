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

## Legacy default and explicit parallel opt-in (2026-09-30)

Omitting `slots`, or using `slots: 1`, selects the original BusyGuard and direct
session construction. Only an explicit count above one selects the new scheduler
and pool. Legacy bulk unload skips unpublished first loads and stops counting a
session as resident before destruction, using atomic residency instead of racing
on a session pointer. Bulk snapshots use immutable registered IDs without waiting
for metadata under the registry mutex. Queued targeted unload responses retain
upstream behavior.

The guard identity stays fixed at registration. Changing an existing ID between
one and multiple slots returns HTTP 400 before unloading it; use a new ID or
restart. Resizing within the parallel path remains supported. This uses the
existing per-model setting, with no additional CLI flag. The original BusyGuard
is unchanged; no model arithmetic, GPU kernels or admission tables change.

| Validation scope | Result |
|---|---|
| Clean standalone framework CPU build, Windows/MSVC Release | Server built; eight focused CTests pass |
| Clean standalone Canary/Piper CPU checkpoint smoke | Four default/explicit-one cases pass; exact upstream parity, unsupported two-slot rejection, recovery and unload/reload |
| Integrated model-adapter CUDA/Vulkan builds, RTX 3090 | Five focused CTests per backend pass |
| CPU/CUDA/Vulkan Canary ASR and Piper TTS, upstream/default/explicit one | 18 scenarios pass with exact output and meaningful metadata |
| CUDA/Vulkan Canary/Piper concurrent cold/warm, 2 and 4 slots | Eight smoke scenarios pass, including rejection of path changes and resizing within the parallel path |
| Unpublished NeuTTS first load plus idle Piper bulk unload | Three scopes pass: CUDA upstream/candidate and Vulkan candidate |
| Two queued targeted unloads behind Canary inference | Six scopes pass: upstream/candidate on CPU/CUDA/Vulkan |

The standalone CPU tests are `server_config_test`, `server_busy_guard_test`,
`server_single_slot_compat_test`, `model_execution_guard_test`,
`server_model_slots_test`, `session_pool_test`, `shared_weight_cache_test` and
`gguf_tensor_source_test`. The existing cross-platform CI build/filter lists now
include the execution-guard test. The local clean CPU composite includes Canary,
Piper and built-in audio utilities; the remote workflow keeps its existing model
selection.

Checkpoint comparisons use freshly built clean upstream `ff0d9809`; only top-level
timing is excluded from exact response parity. GPU model integration uses the
existing adapter follow-up worktree with this execution-path change applied. Its
other existing model changes are outside this framework commit. Standalone
admission remains empty, so the model smoke tests do not newly admit those models
in this PR. Upstream Vulkan NeuTTS first-load behavior was not compared because
that baseline composite does not link NeuTTS.

Evidence is retained locally in `outputs/slots-legacy-default-20260930/`: the
report, source/executable/checkpoint manifests, raw responses, configs, captured
CMake caches, build/CTest logs, first-load and queued-unload records. The clean
published-source CPU build is recorded in `build-publish-cpu.log` and
`ctest-publish-cpu.xml`; checkpoint smoke is recorded in `publish-cpu-smoke.json`.
Earlier sections describe earlier revisions.

This is focused compatibility evidence, not completion of the full F1-F6/model
validation gate. Metal/macOS, Linux and new sanitizer runs have not been performed
locally. The reported opt-in ordering, fairness, management-priority, first-load
bulk-delay and retirement-accounting issues remain for separate work. No claim
is made that these smoke tests resolve or validate those issues.

## Local follow-up: FIFO admission and retirement, 2026-09-30

The five findings in [the maintainer's report](https://github.com/0xShug0/audio.cpp/pull/715#issuecomment-5914223647)
now have local common-framework fixes: ordered request/management admission and
per-waiter wakeups; binding preparation/deferred callbacks to the same request
lease; ready-model bulk release before blocked drains; logical retirement before
teardown; registry snapshots that do not span metadata waits. Legacy omitted/one
slot execution still uses the unchanged BusyGuard/direct session. No arithmetic,
CUDA/Vulkan kernel or model admission changes were made.

Targeted Windows checks pass: CPU 11/11 CTests, CUDA 5/5, Vulkan 5/5 and MSVC ASan
4/4. Controlled actual-handler tests force each finding at 2/4 slots and cover
stream/batch callback ownership, disconnect and preparation-failure reuse. Real
Piper/Canary probes pass 32 CUDA/Vulkan and 6 CPU case groups with same-backend
pre-fix output parity, queued unload and queued Piper-to-Canary replacement.
Every configured parallel slot is observed in execution logs. The baseline is
the saved pre-fix restored-legacy framework with existing model adapters, not
pristine upstream; backend integration remains separate from this empty-admission
foundation. Evidence is local under `outputs/slot-maintainer-fixes-20260930/` in
the enclosing workspace, including a review patch, manifests, commands, raw logs
and results. Publication excludes the unrelated local HTTP shutdown experiments. The exact
publication checkout builds the server and passes nine focused CPU CTests;
supplemental local ownership/HTTP-worker tests explain the larger local count.
Linux ASan/TSan CI is configured for the published framework tests; results are
pending until those jobs run.

A too-long Piper overlap fixture hit its fixed graph arena in both pre-fix
one-slot and candidate builds. Diagnostic failures are retained; final overlap
tests use a supported shorter prompt without changing model code/settings.
Metal/Linux runtime, TSan and the complete model/framework admission matrix
remain unvalidated here. This targeted fix record does not mark the full gate
complete or replace the maintainer's platform-specific retest.

## CI integration follow-up, 2026-09-30

The first sanitizer run failed while compiling the synthetic merge with newer
upstream, before sanitizer tests executed. Upstream's new `/v1/tasks/batch`
handler still declared `BusyGuard::Lock` for an acquisition that now returns
`ModelExecutionGuard::Lock`. The Linux/macOS build jobs and ASan job reported
the same conversion error; this was not a TSan race report.

Current upstream `9a02e613` is merged here. The generic batch handler now uses
the common request lease, addresses its leased session, and retains ownership
from request preparation through the deferred SSE callback. Legacy one-slot
execution keeps its original acquisition behavior. Added actual-handler tests
cover legacy batches, invalid input, queued replacement at two/four slots, and
two independent deferred batches holding separate slots until completion.

The updated standalone Windows CPU server builds and all nine focused CTests
pass; the three guard/lifecycle MSVC AddressSanitizer tests also pass. Remote
sanitizer and cross-platform results require a fresh CI run; the
earlier GPU/checkpoint results above apply to their recorded revisions, not
automatically to this newer upstream integration. Local logs and JUnit evidence
are retained in `outputs/slot-tsan-ci-20260930/` in the enclosing workspace.

After the compile fix, Linux TSan reached the tests and reported a real race in
cJSON's global error-position writes. The engine JSON wrapper now serializes
only the cJSON parse call; per-tree conversion/deletion remain independent.
An eight-thread valid/invalid/round-trip regression is included in the lifecycle
test. The audit found no other direct cJSON parse/error-hook users in production.

The GCC 13 run also reported timed-mutex unlock warnings in the legacy tests.
This matches the missing `pthread_mutex_clocklock` interceptor documented in
[GCC's r14-905 fix](https://gcc.gnu.org/pipermail/libstdc++-cvs/2023q2/039725.html).
The TSan job selects GCC 14, whose libstdc++ uses an intercepted timed-lock path
under TSan. Production BusyGuard is unchanged; no tests, reports or race checks
are suppressed. Windows CPU 9/9 and MSVC ASan 3/3 pass after the parser fix;
the new remote sanitizer run remains required.
