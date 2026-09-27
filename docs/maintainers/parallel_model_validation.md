# Parallel model validation procedure

Use this procedure before enabling or increasing parallel slot admission for a
model. The common framework is reviewed in PR #715; model adapters, fixes and
admission are reviewed separately in PR #706. This document defines required
evidence, not a claim that every check below is implemented or already passed.
Historical audit passes must be mapped to this procedure, with gaps marked
pending rather than retroactively declared validated.

The [maintainer's lifecycle review](https://github.com/0xShug0/audio.cpp/pull/706#issuecomment-5859339500)
identified a timeout regression when unload was waiting. That case has a fix
and regression coverage, but output parity and mutual exclusion alone cannot
establish correct request ownership or model state transitions.

## 1. Define the validation case

Record the model family, task, offline/streaming/native-batch mode, checkpoint
and auxiliary-file hashes, quantization, model specification and all runtime
options. Record baseline/candidate commit and executable hashes, build flags,
OS, GPU/device index, driver/backend versions, CPU threads, request fixtures,
seeds and reference inputs. A pass applies to this tested combination; it is
not certification of all checkpoints, options, tasks or backends in a family.

Run CUDA and Vulkan independently, using their actual target device indices.
Verify CPU/default-one-slot and unsupported-mode compatibility when relevant;
GPU results do not enable CPU, streaming or native-batch parallel execution.
Run GPU benchmark cases sequentially so unrelated model workloads do not
distort timing or available memory. Preserve default precision, overlap,
generation limits and quality settings across compared implementations.

## 2. Establish single-slot quality references

Compare the original implementation with the candidate at one slot on the
same backend, checkpoint, request and configuration. Record any baseline
source differences; do not label an older PR executable as pristine upstream.
Generate cold, warm and repeated-warm references. For cache-sensitive models,
record each slot's full ordered history, including prompt/reference changes,
cache resizing and reset/unload. Cold and warm references are separate.

Compare all returned audio/stems, transcripts, artifacts and meaningful
metadata. Exclude only explicitly identified volatile fields such as timing;
retain raw responses and hashes. Prefer byte-identical WAV/artifact output and
exact text for deterministic models. Fixed seeds alone do not prove
determinism: measure serial repeat variation first. If exact serial parity is
unavailable, record parity as unverified unless a model-specific quality
criterion is defined and justified before the parallel test. Do not loosen a
tolerance after observing a mismatch merely to turn the result into a pass.

## 3. Exercise every advertised slot count

Test two, three and four slots individually when advertised and when they fit;
extend the same matrix to any higher advertised count. A two-slot pass does not
justify a larger capacity. Never raise admission for an untested count.

For each count, use at least three fresh server starts. Each start must cover
a concurrent cold wave, warm wave, unload/reload and corresponding output
comparison. Include mixed requests at every count: different text/audio lengths,
seeds, reference voices and preparation settings where supported. Reverse
admission order and compare each result with its equivalent single-slot
history, not whichever reference happens to be closest.

Observe the requested number of active loaded slots; concurrent HTTP submission
alone is not proof of overlapping leases. Record which request occupies which
session/history when needed for the comparison. Check active/queued counters
return to zero, then verify serial reuse after the parallel wave. At the largest
advertised count, additionally run at least ten mixed warm waves to look for
intermittent mismatches, crashes, state contamination or growing memory.
Document repetitions and any reduced stress coverage as an explicit gap.

## 4. Check model integration with the lifecycle

Each model/backend validation includes invalid input alongside valid requests,
queue overflow, busy timeout, targeted/all-model unload during inference,
unload during first lazy load, reload and failure recovery. Check that healthy
requests retain correct output, management waits for ownership to drain, and
no error leaves a stranded slot or queue entry. Exercise idle eviction when
supported and a subsequent cold request. A failure of one request must not
silently poison another session or a later inference.

Streaming additionally needs real concurrent streams, disconnect/reset and
callback ownership checks. Native batching needs concurrent batches with
independent per-request results and preparation state. Offline parity and
test-double interface checks do not validate these real model modes.

## 5. Validate the common ownership and transition contract

These checks belong to the framework suite, with representative real-server
integration. Model admission depends on the relevant framework checks as well
as its own checkpoint tests. Use controllable test sessions, barriers/latches
and injected failures to force interleavings; do not rely only on sleeps or
GPU request duration. Bound test completion externally and preserve a trace
of request, slot/session, model instance/configuration and transition events.

| Forced scenario | Required assertion |
|---|---|
| Client disconnect while queued or running | No abandoned queue accounting. A running lease remains held until its worker finishes or safely stops; disconnect does not imply cancellation of GPU work. |
| Unload/reconfiguration with queued requests | Document and test whether queued work uses the selected configuration, uses a later configuration, or is rejected. No accidental mixing of instance/configuration state or use of a destroyed session. |
| Failure during primary/clone/weight loading | No partially published pool; partial ownership is released in the correct order; later load can retry. |
| Concurrent first requests plus unload | Publish a complete pool once; unload cannot skip a load in progress or destroy sessions still referenced by requests. |
| Overdue inference plus waiting management | Preserve immediate overdue rejection, including partially occupied pools blocked by management. |
| Healthy and overdue slots together | Preserve the documented admission/timeout policy; do not infer that one overdue slot makes every active slot overdue. |
| Multiple managers, manager timeout and timeout zero | Management does not starve, giving up wakes eligible requests, and zero retains its documented waiting behavior. |
| Two different models loading/running/unloading | No deadlocks, cross-model state contamination, unsafe backend initialization or destruction of another model's live resources; include residency-limit/eviction contention. |
| Shutdown with active and queued work | Follow the documented drain/reject policy, with finite test workers; no resource destruction while inference/callbacks still reference it. |

Assert ownership, not just counters: one request owns one leased session;
mutable graphs/caches/RNG/callbacks are private; immutable resources remain
alive while referenced; every completion/error path releases its ownership.
Counter snapshots alone cannot prove these properties. Add seeded stress runs
with reproducible traces after the controlled scenarios. Run applicable CPU
framework tests under AddressSanitizer and ThreadSanitizer separately where
supported; record platform/backend limitations. Sanitizers do not certify GPU
driver/kernel concurrency, so retain real CUDA/Vulkan integration checks.

## 6. Measure memory and performance separately

Measure the actual same workload in three configurations: one slot running
requests sequentially, one loaded model with N slots, and N separate one-slot
server instances where memory permits. Do not substitute N times a single
measurement for a measured sequential run without labeling it an estimate.

Report cold startup/loading separately from warm inference. Use at least three
warm repeats and report sample count, total batch completion time, throughput,
individual latency (median and tail when sample size supports it), failures and
queue/timeout rejection latency. A timeout rejection improvement is not an
inference speedup. Independent slots may improve throughput while increasing
individual latency, and stage serialization can make a correct model slower.

Sample peak GPU memory throughout loading, reference preparation, generation,
unload and reload, including first-request peaks. Record retained idle memory,
host memory and sampling method/interval. State whether values include model
weights, backend/context overhead and workspaces. Compare the same slot count
before/after memory changes. Look for bounded reuse over repeated warm waves;
allocator caching is not by itself proof of a leak. On allocation/VRAM guard
failure, retain diagnostics and mark that count blocked; do not lower quality
settings just to obtain a pass.

## 7. Record the outcome and update admission

Keep correctness, quality, performance, memory and lifecycle results separate.
A correct-but-slower result can support concurrency admission, with the measured
tradeoff disclosed. A quality mismatch, unexplained reset or relevant lifecycle
gap cannot be hidden behind a throughput gain.

| Result | Meaning |
|---|---|
| Validated | Required correctness, quality and lifecycle evidence for the recorded combination/count is complete. |
| Correct but slower | The validation gate passes; measured throughput is worse for this workload. |
| Parity unverified | Execution succeeds, but quality equivalence remains unresolved. |
| Memory blocked | Hardware capacity/guard prevents completion; no correctness pass is claimed at that count. |
| Failed | Reproducible crash, mismatch or ownership/lifecycle failure. |
| Pending / unsupported | A required check has not run, or the backend/mode does not advertise the capability. |

Attach request/config fixtures, raw responses/hashes, timing/memory samples,
logs, repetition counts and reproduction commands. Maintain a per-count
coverage matrix: model reference/parity, mixed/repeat operation, lifecycle,
performance, memory and relevant common-suite evidence. State pending gaps
explicitly. Update the admission header and its capacity assertions only for
consecutively validated counts; keep unsupported combinations at one slot.

For an everyday adapter change, run the affected model/backend gate plus
relevant common regressions. Before merging the model-support PR, rerun the
admitted catalogue on available hardware and report blocked/unverified cases.
After framework changes, rerun common cross-platform tests and representative
checkpoint integration, expanding to affected models when dispatch, ownership
or preparation semantics change. Remote CI and local results are distinct.

## Existing evidence and remaining work

The [framework report](../reports/common_slot_framework.md) lists completed
focused tests and real single-slot server probes. Existing common tests cover
pool contracts/rollback, test-double stream errors, leases/management/timeouts,
shared-cache ownership and concurrent tensor-source access. The HTTP overdue
harness is `tests/server/server_busy_timeout_regression.py`; its standalone
framework configuration uses one slot. Higher counts need model admission.

These checks are partial coverage of section 5, not implementation of its
entire controlled HTTP lifecycle matrix. Real client disconnects, queued
reconfiguration ownership, cross-model contention, shutdown and sanitizer
coverage require explicit evidence before being marked complete. Model
checkpoint validation remains in the model-support follow-up. This procedure
adds no model admission, new runtime behavior or automatically executed tests.
