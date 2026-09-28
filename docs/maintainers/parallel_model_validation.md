# Parallel model validation procedure

Procedure version: **2026-09-28, shared framework + model gate**.
PR #715 owns the common framework; PR #706 owns model adapters and admission.
This procedure defines required evidence, not tests already completed.

Run shared framework checks once for the relevant revision and reference them
from model results. Run model checks separately for CUDA/Vulkan and the exact
checkpoint/task/mode/count being admitted. Performance is a separate report.

| Part | Coverage | Repeat when |
|---|---|---|
| Shared framework: F1-F6 | Scheduling, ownership and transitions, with representative real-server/backend integration | Relevant framework, shared-resource or backend behavior changes |
| Model gate: M1-M6 | Baseline parity, every admitted count, mixed/repeated quality, recovery and memory reuse; applicable execution modes | Adapter, checkpoint, preparation, inference or applicable backend behavior changes |
| Performance | Actual serial versus shared-slot timings; optional independent-server comparison | Reporting performance for a changed workload/build |

## Shared framework gate (once, referenced by models)

Keep one framework record per applicable source/build/platform/backend scope,
instead of copying these requirements into every model checklist. CPU tests
cover common logic; representative real CUDA/Vulkan integration covers backend
ownership/resource behavior. Explain which adapter/construction paths the
representatives cover and expand coverage when another path is affected.

| ID | Required assertions |
|---|---|
| F1 Scheduling and compatibility | Isolated leases, drained counters/queues and management priority without starvation. Test overdue work with waiting management/spare slots, healthy plus overdue slots, multiple managers, manager timeout and timeout zero. Preserve default one-slot and unsupported-count/mode behavior. Test queue overflow only if a capacity exists; otherwise document timeout rejection. |
| F2 Load/configuration transitions | Concurrent first requests publish one complete pool; unload cannot skip a load in progress. Force unload/reconfiguration with queued work and test its documented configuration-selection/rejection policy. No mixed configuration or destroyed-session access. |
| F3 Failure/resource lifetime | Inject primary/clone/weight-load failures, verify rollback/destruction order and retry. Immutable resources outlive users; mutable graphs/caches/RNG/callbacks are private; every error releases ownership. |
| F4 Disconnect/shutdown | Disconnect running/queued clients without abandoning ownership/accounting. Finite active/queued workers and callbacks follow the documented drain/reject policy; shutdown cannot destroy live resources. Forced process termination does not prove graceful drain. |
| F5 Multiple models/eviction | Force two models loading/running/unloading, including residency-limit/idle-eviction contention. No deadlock, contamination, unsafe backend initialization or destruction of another model's live resources. |
| F6 Regressions/sanitizers | Relevant common CTests and reproducible seeded stress after controlled scenarios. Applicable CPU tests under ASan and TSan separately where supported, with explicit platform/unrun-job limitations. Sanitizers do not certify GPU kernels/drivers. |

Use controllable sessions, barriers and injected failures, with representative
real-server integration. Bound completion externally; retain request/session/
configuration/transition traces. Counters and concurrent submission alone do
not prove ownership. These checks retain the
[maintainer's lifecycle concerns](https://github.com/0xShug0/audio.cpp/pull/706#issuecomment-5859339500).

CPU checkpoint smoke tests are required when changes affect CPU behavior.
For unaffected CPU paths, record why and reference common default/unsupported-
mode coverage; every GPU model need not run on CPU. Real streaming/native-batch
admission still requires its own model checks.

## Model gate (six entries per model/backend/mode)

Record applicability and evidence for M1-M6. Collect outputs, counters, timing
and memory in the same runs; do not repeat an identical quality/lifecycle matrix
only to sample memory. Applicable assertions inside each group remain required.

| ID | Required evidence |
|---|---|
| M1 Scope/serial quality | One manifest identifies checkpoint/auxiliaries, quantization, task/mode/settings, fixtures/seeds/references, baseline/candidate source/executable hashes, build flags, OS/device/driver/backend and threads. Compare original and candidate one-slot cold, warm and repeated-warm outputs and meaningful metadata. |
| M2 Every admitted count | At least three fresh server starts at each advertised count that fits. Each start covers true concurrent cold and mixed warm waves, unload/reload, exact comparisons, observed active loaded leases, drained counters and correct serial reuse. |
| M3 Mixed/repeated quality | Effective length, seed, reference/preparation changes where supported, and reversed admission order. At the largest admitted count, at least ten mixed warm waves in each of the three starts. Cache-sensitive models need equivalent ordered per-session histories including resizing/reset; stateless models need no invented history requirement. |
| M4 Model recovery | Reuse M2 unload/reload evidence. Once per model/backend at the largest admitted count: invalid input alongside healthy work, primary-load failure followed by retry, and targeted/all-model unload including during first lazy load. Healthy outputs stay correct; no error strands ownership. Shared queue/timeout/eviction semantics reference F1/F5; test adapter-specific overrides separately. |
| M5 Memory reuse | Sample GPU/host memory during existing cold/preparation/generation/warm/unload/reload runs, including retained idle memory. Report interval, first-request peaks and included weights/context/workspaces. Assess repeated-use stabilization; investigate unexplained continuing growth, OOM/guard failures. Compare the same count/settings with baseline for retention investigations or memory changes. |
| M6 Execution modes | Offline evidence admits offline only. Streaming needs concurrent streams, disconnect/reset and callback ownership; native batching needs concurrent batches with independent results/preparation. Modes not being admitted are not applicable with a reason; test doubles do not validate real model modes. |

### Quality rules that must not be relaxed

- Preserve precision, overlap, generation limits and quality settings across
  comparisons. A VRAM failure is blocked evidence, not permission to lower them.
- Compare all audio/stems, transcripts, artifacts and meaningful metadata.
  Retain raw responses/hashes; exclude only named, justified volatile fields
  such as timing, never an entire metadata object.
- Deterministic output requires byte-identical audio/artifacts and exact text.
  Fixed seeds alone do not establish determinism: measure serial repeat variation.
  If serial parity varies, mark parity unverified unless a model-specific quality
  criterion was defined and justified before the parallel test. Never loosen
  tolerance after a mismatch.
- Cold and warm references are distinct. Compare equivalent single-slot
  histories, not the closest reference. Trace actual session/history selection
  for cache-sensitive models.
- Variations must change effective inputs: ignored ASR text is not mixed-audio
  coverage. Retain audio hashes/lengths and applied options.
- Run the true cold wave before a load-producing probe. Sample loaded leases
  separately from slow GPU-memory queries so short overlap is not hidden.
- A two-slot pass does not admit three/four slots. Three fresh starts per count
  and ten mixed warm waves per start at the largest count are unchanged. Extend
  the same gate to higher advertised counts; never enable an untested count.
- An intermittent mismatch remains failed when later repeats pass. Existing
  AuK/Apollo mismatches are not waived by this procedure change.

M5 needs an evidence-backed assessment of finite repeated reuse, not proof of
leak freedom for every future request. Allocator caching alone is not a leak.
For stable measurements, state the tested history/range. For uncertain growth,
keep the diagnostic and M5 pending; do not invent a passing bound after seeing
it. A baseline issue does not automatically waive an unsafe count.

## Performance report (separate from admission)

Measure the same workload as original one-slot serial, candidate one-slot serial
and candidate N shared slots. Compare actual N-request serial completion time,
not an unlabeled `N * one request` estimate. Run GPU benchmarks sequentially
without unrelated workloads. Keep cold startup/loading separate; use at least
three warm repeats and retain sample count, batch time, request latencies,
failures and memory samples from the existing runs.

N independent one-slot server instances are an **optional** deployment comparison
when memory permits or that comparison is requested. Missing/memory-blocked
independent-server measurements do not fail model quality. Three batches do not
support robust tail-latency claims. Rejection latency is separate: quicker HTTP
503 rejection is not quicker generation.

The comparison table reports backend/model, model checks, shared framework
checks, base one-request time, candidate one-slot time, time saved percentage,
measured base N-request serial time, candidate N-slot completion time and time
saved percentage. `100 * (base - candidate) / base` means elapsed time saved;
positive means less time. Keep quality and memory verdicts visible. Slower
models can pass correctness; faster mismatching models cannot.

## Evidence, reporting and admission

Generate one reusable manifest per unchanged build/package/fixture set, with
model-specific settings attached. Hash unchanged inputs once and reference
them; changed files/settings need new identities. Attach raw outputs, traces,
samples, logs and reproduction commands to the six model entries and the shared
record instead of duplicating every low-level assertion.

| Field | Meaning |
|---|---|
| `model_checks` | M1-M6 for the recorded model/backend/task/mode/count, independent of missing shared tests or optional performance measurements. |
| `framework_checks` | Applicable F1-F6 shared evidence, with record identity/backend scope. Missing shared coverage is pending, not a model output failure. |
| `admission_ready` | True only when model and applicable framework checks pass, with justified not-applicable entries. Performance speed is not part of this boolean. |
| `all_plan_rules` | Legacy alias for `admission_ready` under this recorded procedure version; never imply optional benchmarks passed. |
| `performance` | Measurements/tradeoffs: completed, pending or memory blocked independently. |

Each entry records ID, applicability, PASS/FAIL/PENDING/BLOCKED/UNSUPPORTED/
NOT_APPLICABLE status, tested identity, evidence, reproduction and limitations.
NOT_APPLICABLE needs an implementation/platform reason. Relevant missing shared
evidence still blocks admission; moving it to one record does not waive it.
Disclose unrelated common-suite failures separately.

Execution finished is not validation complete. Report "evidence collected;
validation pending" for unresolved required entries. A group cannot pass by
omitting a failed assertion. Admit only consecutively validated counts; keep
unsupported combinations at one slot. This documentation adds no admission
or runtime behavior.

Label historical evidence with its procedure version and tested hashes. Do not
recalculate old seven-section reports as passes merely by changing the plan.
First map assertions/evidence to M/F entries, retaining failures, missing raw
metadata/histories and changed-binary limitations. New executables do not inherit
quality certification from predecessors. Reviewed evidence reuse must identify
unchanged relevant source/settings and applicability; changed inference,
preparation, ownership or backend paths need new coverage. Changed fixtures or
strengthened observers produce new evidence, not exact historical replay.
Preserve replaced experimental records separately.

For adapter-only changes, rerun the affected model/backend gate and applicable
common regressions; justify reuse of unaffected shared evidence. For framework
changes, rerun F1-F6 and representative checkpoint integration, expanding to
affected paths. Before model-support merge, run the admitted catalogue on
available hardware and disclose blocked/unverified combinations. Local and
remote CI results are distinct. Publishing restrictions neither fail evidence
nor authorize publication.

See the [session contract](parallel_sessions.md),
[framework report](../reports/common_slot_framework.md), and
[scheduler regression](../reports/scheduler_busy_timeout.md) for existing scope
and reproduction. Historical passes do not certify every requirement here.
