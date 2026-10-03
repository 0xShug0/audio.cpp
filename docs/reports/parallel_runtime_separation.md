# Startup-selected runtime separation

Implements the [maintainer's Stage 1 proposal](https://github.com/0xShug0/audio.cpp/pull/715#issuecomment-5963682706)
against upstream `2892ed3e94b6ccbdacc111fa2896aaae0a5dc8fc`.
This replaces the previous slot-count-selected guard inside a shared server runtime.

| Startup | Slots field | Result |
|---|---|---|
| Default | Absent | Original `ServerState` |
| Default | Any explicit value, including `1` or `null` | Startup error; dynamic registration returns HTTP 400 |
| `--parallel-jobs` | Absent | `ParallelServerState`, capacity one |
| `--parallel-jobs` | Integer 1-16 | Parallel runtime, subject to advertised model capacity |

The enabled process uses the new pool, leases and FIFO scheduler even with one
slot. Its model capacity can change through one without changing the runtime.
An unsupported count cannot publish a partial pool. Existing model adapters and
admission tables are unchanged: this foundation does not enable catalogue models
for multiple slots. Model support remains in the separate follow-up, PR #706.

## Source boundary

`runtime.h`, `busy_guard.h`, `http.h/.cpp`, `frontend.h/.cpp`, the existing session
interface, CLI files, model implementations and ggml match upstream. The sole
legacy `runtime.cpp` addition rejects explicit slots immediately after parsing
dynamic registration. Execution, ownership, loading, unloading, streaming,
batching, eviction and shutdown bodies retain upstream behavior.

`parallel_runtime.h/.cpp` owns the enabled process's registry, sessions, leases
and locks. Both classes implement the existing interfaces. Startup constructs
one stack-owned concrete class. The flagged default HTTP path now selects
`parallel_http.h/.cpp`, whose owned workers drain before runtime destruction;
default startup still selects the original listener. Custom frontend listeners
retain the existing interface and must drain their workers before returning.
The copied externally linked language-option helper has a distinct name.

The follow-up [self-audit](parallel_runtime_self_audit.md) records scoped fixes
for failure transitions, concurrent registration, transport-policy reads, bulk
barrier ordering and HTTP worker lifetime. The validation below describes the
initial separation revision; the audit has its own evidence/results.

JSON parsing synchronization is a startup-only process opt-in, enabled before
configuration parsing and concurrent users. Default startup and the CLI leave it
off. The only production direct cJSON parse caller found is the framework wrapper.
New direct callers must honor this policy.

Tensor storage and resource cache synchronization require separate explicit
construction options. Both resource-ID and canonical-path caches distinguish
source policies; indexed shards and prefix/folded/composite holders retain the
underlying source and its policy. Existing loaders keep their original calls.

Run `python tests/server/check_parallel_boundaries.py --base origin/main` after
carrying upstream updates into both runtime copies. Source boundary checks do
not certify model output or multi-slot admission.

## Local validation

Windows/MSVC, RTX 3090, CUDA 12.4 and Vulkan SDK 1.4.350.0. GPU builds select
Piper, Canary, NeuTTS and Nemotron diarization, with native model management.

| Check | Result |
|---|---|
| Protected-file and legacy-runtime boundary | Pass |
| CPU server build and focused CTests | 11/11 pass |
| Windows AddressSanitizer focused CTests | 8/8 pass |
| CUDA CLI and server build | Pass |
| Vulkan CLI and server build | Pass |
| Actual process startup and model response schema | 9/9 pass |
| Matched checkpoint integration, both backends | 96/96 requests pass |

Controlled lifecycle tests cover capacities 1, 2 and 4: cold load, queued
unload/reconfiguration, FIFO ordering, management barriers, idle-first bulk
unload, retirement under resident limits 1/2, metadata contention, deferred
stream/native-batch/generic-batch ownership, disconnect, and rejected preparation.
Additional coverage exercises count changes 1/2/1/4/1, invalid counts without
unloading existing state, concurrent valid/invalid JSON, synchronized storage
release/remapping and resource-cache policy separation.

Checkpoint integration compares current upstream, unflagged candidate, flagged
candidate with omitted slots, and flagged candidate with explicit slots=1.
Piper, Canary and Nemotron diarization have exact aligned output/metadata after
removing only top-level execution timing, across cold, two warm requests and
unload/reload on each backend. Model-list schema and drained slot counters are
checked too. This is representative capacity-one integration, not complete
catalogue admission or a throughput benchmark.

Local evidence is retained in `outputs/pr715-stage1-separation-20261003/`: proposal,
source/build/executable manifest, CMake caches and commands, CTest XML, build logs,
startup records, checkpoint harness/results/raw responses/server logs and the
boundary audit. Remote cross-platform and Linux TSan coverage is tracked by CI
on the pushed revision; local results do not substitute for those jobs.
