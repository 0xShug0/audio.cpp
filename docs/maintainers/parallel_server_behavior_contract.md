# Parallel Server Behavior Contract

Version: **2026-10-04**. This is the authoritative behavior contract for the
**experimental** `--parallel-jobs` runtime in PR #715. Other documents describe
implementation, historical measurements or validation procedures; they do not
extend this contract's supported/tested scope. Derived from the
[maintainer's requested state/operation audit](https://github.com/0xShug0/audio.cpp/pull/715#issuecomment-5974471675).

The current focus is startup isolation and the normal request path: independent
leases, cold initialization and serial reuse. Lifecycle/error matrices specify
expected behavior and expose coverage gaps, not a promise that every combination
has been tested. Catalogue multi-slot admission remains a separate model task.
The CLI and unflagged server retain their original implementation.

## Scope and reading the matrices

No flag selects `ServerState`; any explicit `slots`, including 1/null, is rejected.
The flag selects `ParallelServerState` once at startup, including capacity one.
Counts are integers 1..16, further limited by the session's advertised capacity.
The CUDA/Vulkan admission lists in this PR are empty. A model without a parallel
factory therefore supports one slot. Passing a controlled 1/2/4-slot fixture
never admits a real checkpoint at those counts. Startup prints this experimental
scope and the incomplete lifecycle/backend coverage. [T0]

Each cell states **work/wait; response; resulting state**, assuming successful
finite backend operations. Failure/timeout rows below override those outcomes.
A/B are distinct registered models. Responses are HTTP 200 unless the cell says
otherwise. Dynamic management requires `ui_management` (otherwise 403); an
unknown targeted ID returns 404. Bulk unload reports unknown IDs in `not_found`
and does not fail known selections. Registration of an unknown ID is covered
separately from reconfiguration of an existing ID.

**D** means a directly exercised controlled case; **P** links the nearest test
but the entire stated combination has only partial coverage; **U** means no
corresponding behavioral test. These annotations describe coverage, not different
runtime rules. Links resolve to specific test functions. All combinations are
subject to the validation-area scope at the end; D is not a GPU/catalogue claim.

## States and ownership

| State | Definition and allowed next states |
|---|---|
| Unloaded | Registered metadata, no published pool/weights; next inference or explicit load starts loading. Invalid replacement preserves it. |
| Loading | Private weights/primary/clones are being built under initialization protection and an admitted request or exclusive registration lease. Only a complete pool becomes ready; failure returns unloaded. |
| Ready | Complete pool published. May have spare slots, executing requests, active streams/batches and/or waiters. Next management can enter reconfiguring/unloading only after earlier leases drain. |
| Reconfiguring | Exclusive FIFO lease. Validate replacement before teardown; then retire old pool, commit complete metadata and load replacement. Success becomes ready; pre-commit failure preserves old state; post-commit load failure becomes unloaded with new metadata. |
| Unloading | Exclusive ownership, logical residency cleared before sessions/weights are destroyed. Success becomes unloaded. Free-memory queries may still observe retiring allocations. |
| Shutting down | Process/transport phase, not an exposed per-model enum. Stop accepting, cancel socket I/O, join owned workers, then destroy the runtime. Existing requests can still finish admission/backend work. No bounded completion for hung backend calls. |

These are conceptual states, not a new atomic state API. `GET /v1/models` reports
`loaded`, configured `slots`, `active_slots`, `queued_requests` and
`max_parallel_slots` (null until loaded/after retirement). Status is observational;
it neither reserves resources nor gives a globally atomic lifecycle snapshot.

| Request phase | Ownership / transition |
|---|---|
| Queued | In the per-model FIFO; no session/configuration bound yet. Timeout removes the waiter. Disconnect/shutdown does not immediately cancel this scheduler wait. |
| Admitted, waiting for initialization | Owns a slot; waits for the one initializer and possibly guarded cold-load serialization. Counts as active even before inference starts. |
| Preparing / executing | Owns one independent session and its configuration until result serialization completes. Backend work is not preempted by busy timeout. |
| Active stream / deferred batch | The response callback retains the validation lease, including before callback invocation. Completion, exception or callback destruction releases it; stream interface reset is the adapter's responsibility. |

FIFO order is the order callers enter the scheduler, not packet timestamps.
Consecutive requests may fill spare slots. Management blocks later requests,
including when spare slots exist, while earlier admitted/queued requests drain.
Locks/leases release on success and exception. Independent sessions do not imply
simultaneous GPU kernels or a throughput improvement. [T1] [T2] [T3]

## Operations crossed with model states

`M` below is the affected model. Bulk work takes a registry snapshot, publishes
per-model barriers in one short transaction, releases that publication lock,
and drains ready selections before waiting for other selections. It is not an
atomic all-model unload. Eviction/idle retirement try-acquire rather than enqueue.

| M state | Inference | Register / reconfigure M | Unload M | Bulk unload containing M | Eviction / idle retirement | Status | Shutdown |
|---|---|---|---|---|---|---|---|
| Unloaded | Admit; initialize then execute; 200 result, ready. D [T1] | Exclusive; validate/load; 200 loaded, ready. P [T4] | Exclusive; idempotent 200 loaded=false, unloaded. P [T4] | Barrier; include M once in 200 unloaded list; unloaded. P [T5] | No resident victim; skip, unloaded. P [T6] | Read metadata/counters; 200 unloaded/null capacity; no load. P [T7] | No new accepts; drain any admitted work before destruction. P [T8] |
| Loading | Spare slots admit and wait on initialization; full pool queues; 200 results after complete publication, ready. P [T1] | Barrier behind earlier load/requests; 200 new ready state. P [T2] | Barrier waits for first load and earlier work; 200 unloaded. D [T5] | Publish barrier even before pool exists; free ready B first, then M; 200 unloaded. D [T5] | Busy M skipped; use another idle victim or load gets 503; M continues loading. P [T6] | 200 observation, pool not partially exposed; M unchanged. P [T4] | Join finite initialization/request worker; no pool destruction while in use. D controlled default listener [T19] |
| Ready | Admit into spare capacity or queue; 200 result; ready. D [T1] | Earlier work drains, private validation/probe precedes teardown; 200 loaded; ready. D [T2] [T9] | Earlier work drains; 200 loaded=false; unloaded. D [T2] | Per-model barriers; ready-idle M freed first; 200 unloaded. D [T5] | Only idle and queue-free M can retire; no client response; unloaded. P [T6] | 200 counters/metadata, no lease reservation; ready. P [T7] | Cancel I/O, join finite workers/callbacks, then destroy. D [T8] |
| Reconfiguring | Queue behind manager, bind new config after release; 200 result; ready or retry load. D [T2] | Later manager queues; exclusive in scheduler order; 200 final ready config. D [T10] | Later manager queues; after replacement drains, 200 unloaded. P [T10] | Barrier queues behind existing manager; unrelated ready B can retire; 200 unloaded. P [T11] | Busy M skipped; reconfiguration continues. P [T6] | May wait briefly for metadata commit; 200 old/new complete metadata, never partial mix. P [T7] | Existing management finishes before worker join; hung work can block exit. D finite reconfiguration [T19] |
| Unloading | Queue; after teardown reload and run; 200 result, ready. P [T6] | Queue; after teardown validate/load; 200 ready replacement. P [T10] | Queue; idempotent second unload; 200 unloaded. P [T10] | Barrier queues; ready B freed before waiting for M; 200 unloaded. P [T5] | Already retiring M is not a resident victim; other idle victims considered independently. D [T6] | 200 logical unloaded/null capacity while physical teardown can continue. D [T6] | Wait for finite teardown/worker completion before destruction; no forced backend cancellation. D finite teardown [T19] |
| Shutting down | New connections not accepted; preaccepted/queued work may still run; socket response may be lost; resources destroyed after drain. P [T8] | No new accepts; preaccepted manager may finish; no guaranteed delivered response. U [GAP] | No new accepts; preaccepted unload may finish before final destruction. U [GAP] | Existing barriers drain/cancel through ownership; no guaranteed delivered response. U [GAP] | Idle thread stops/joins; resources eventually destroyed after workers. U [GAP] | No guaranteed response; custom/direct callers must stop using the handler before destruction. U [GAP] | Repeated stop signal changes no ownership rule; join remains required. P [T8] |

A **new registration** validates privately, acquires an exclusive lease before
publication and publishes one stable entry before first load. Concurrent same-ID
registration queues on that entry; different IDs use independent entries except
for intentional cold-load guards. First-load failure leaves the registered entry
unloaded for retry; entries are not erased while workers can retain pointers.
A newly registered unsupported count can remain registered/unloaded until corrected;
it never publishes a partial pool. This differs from preserving an existing
working registration on unsupported replacement. [T4] [T12]

## Two-model situations

Here each request/management operation is considered independently against the
starting situation; combined arrival orders follow the next matrix.

| Starting situation | Request to A | Request to B | Reconfigure A | Unload A | Unload A and B |
|---|---|---|---|---|---|
| A unloaded; B ready | Initialize/run; 200; A ready. P [T1] | Run spare slot; 200; B ready. P [T13] | Validate/load exclusively; 200; A ready. P [T4] | Idempotent 200; A unloaded. P [T4] | Ready-idle B freed, A barrier drained; 200; both unloaded. P [T5] |
| A loading; B ready | Admit/wait init or queue; 200; A ready. P [T1] | Warm B bypasses A initialization/load guard; 200; B ready. D [T13] | Wait A's older leases; 200; A ready replacement. P [T2] | Wait first load/older work; 200; A unloaded. D [T5] | Free idle B before A finishes; 200; both unloaded. D [T5] |
| A running with spare slots; B ready | Admit unless older barrier; 200; A ready. P [T3] | Independent run; 200; B ready. P [T13] | Barrier waits all older A work despite spare slots; 200; A ready replacement. D [T2] | Same barrier; 200; A unloaded. D [T2] | B idle retires first; drain A; 200; both unloaded. P [T5] |
| A full with queued requests; B ready | Queue FIFO/timeout 503; after success A ready. D [T3] | Independent run; 200; B ready. P [T13] | Older A queued requests run first; 200; A ready replacement. D [T2] | Older A work runs first; 200; A unloaded. D [T2] | Publish both barriers; free idle B, drain A; 200; both unloaded. P [T5] |
| A reconfiguring; B running | Wait A manager; 200; A ready/new config. D [T2] | Spare B slot runs/full B queues; 200; B ready. P [T13] | Queue behind A manager; 200; final A ready. D [T10] | Queue behind A manager; 200; A unloaded. P [T10] | Barrier A waits manager, B waits older requests; 200; both unloaded. P [T11] |
| A unloading slowly; B ready | Wait, reload/run; 200; A ready. P [T6] | Warm B runs; no registry wait on A destructor; 200; B ready. D [T6] | Wait A teardown then load; 200; A ready. P [T10] | Wait then idempotent 200; A unloaded. P [T10] | Ready-idle B frees before A wait; 200; both unloaded. P [T5] |
| A/B both loading or running | A-only init/queue plus possible global cold guard; 200; A ready. P [T11] | B-only init/queue plus possible global cold guard; 200; B ready. P [T11] | Drain A only; 200; A ready replacement. P [T2] | Drain A only; 200; A unloaded. P [T2] | Publish both barriers before waiting; 200; both unloaded after earlier work. D [T11] |
| Another bulk operation targets A/B | Queue after A barrier; later reload/run; 200; A ready. P [T3] | Queue after B barrier; later reload/run; 200; B ready. P [T3] | Queue on A; 200; A ready replacement. P [T10] | Queue on A; 200; A unloaded. P [T10] | Atomic barrier publication orders overlapping selections; deduplicate; no reverse-order cycle; 200 both unloaded. D [T11] |

## Arrival order and operation interactions

| Interaction | Expected ordering / resulting behavior | Evidence |
|---|---|---|
| Request then request, same model | Spare slots admit both; full capacity queues later callers FIFO. Initialization is shared, session ownership is separate. | D [T3]; P cold overlap [T1] |
| Request then manager, same model | Manager waits all older admitted and queued requests, streams and deferred batches; later requests cannot use spare slots ahead of it. | D [T2] [T14] |
| Manager then request, same model | Request queues behind barrier, then binds the resulting configuration; timeout may return 503 without a lease. | D [T3]; P config change [T2] |
| Manager then manager, same model; reverse operation kinds | Queue in enqueue order, each exclusive. Later unload can retire a replacement; later registration can reload an unloaded entry. | D observed manager order [T10]; P every operation permutation |
| Requests/managers on different models, either order | No registry lock across admission/metadata waits, inference or teardown; unrelated warm work progresses. Intentional guarded cold loads can serialize; UI installer/JSON parser have separate brief/shared scopes. | D [T7] [T13]; P all backend combinations |
| Bulk then overlapping bulk, either/reversed selection order | Serialize only barrier publication, not waits/destruction. Shared models observe the first bulk's barriers before the second's. Non-overlapping ordinary requests bypass this mutex. | D [T11] [T15] |
| Bulk then single manager/request; single operation then bulk | Per-model enqueue order governs each intersection. Bulk snapshot misses registrations published after its snapshot; it is not an all-future-model fence. | P [T10] [T11]; snapshot race U [GAP] |
| Bulk cancellation / partial publication failure | Pending reservations unwind; already granted reservations release. No stranded management ownership. Earlier completed physical unloads are not rolled back. | D reservation RAII [T3]; allocation fault U [GAP] |

## Failure boundaries and recovery

| Boundary / case | Response and required retained/resulting state | Evidence |
|---|---|---|
| Invalid task/preset/spec/schema before live changes | Reject; working configuration/pool unchanged. Slot schema rejected before publication. Generic handler errors become transport 500 unless specifically mapped; do not promise every metadata error is 400. | D [T9]; schema [T0] |
| Unsupported capacity on existing registration, including slots=5 | HTTP 400 invalid_request_error; retain old pool, count, metadata and residency, release manager lease; next inference/valid resize works. | D [T16] |
| Changed multi-slot replacement needs capacity probe | Reuse live advertised capacity for unchanged session inputs. Otherwise create a private primary against effective task/options; reuse immutable loaded assets when possible. Probe fails before commit: old state retained. Changed checkpoint may require temporary duplicate weights; no eviction just to probe. | D controlled option/checkpoint caps [T16]; GPU staging/memory limits U [GAP] |
| After old teardown/new metadata commit, backend or clone failure | No partial pool; new complete config remains registered/unloaded; next request may retry. Prior weights/config are not restored. Multi-slot preflight success does not guarantee the subsequent load succeeds. | D single-slot replacement [T12]; P multi-slot post-probe failure |
| Initial load, null primary, partial clone failure | Private sessions die before weights; no published residency; release admission; retry allowed. Other already admitted cold requests may retry after a failure. | D [T12]; pool [T17] |
| Preparation/execution failure | Release lease, preserve ownership order; HTTP error or stream error if headers sent. Framework does not certify model-specific reset/clean cache after arbitrary backend faults. | D [T12]; real adapter recovery partial |
| Failed publication/allocation before new entry commit | No dangling registry index; staged ownership unwinds. | P [T4]; allocation fault U [GAP] |
| Residency/memory refusal | HTTP 503; no partial new pool. Eviction completed before a failed memory check is not undone. Logical retirement is not physical free memory. | D residency [T6]; device OOM partial |
| Teardown is slow | Keep exclusive ownership; clear logical loaded/capacity first; allow unrelated warm work. No finite completion bound on hung destructor/backend. Throwing destructors are not supported. | D [T6]; hung/throwing teardown U [GAP] |
| UI models-root replacement fails | Stage installer before swapping folder/installer; failure retains consistent old pair. | P GPU-build UI smoke; constructor OOM U [GAP] |

## Timeout, disconnect and shutdown

| Case | Work/wait, response and ownership result | Evidence |
|---|---|---|
| Admission timeout; request or targeted manager | Effective per-model/server ceiling with request override where supported. HTTP 503 server_busy; queued ownership removed. Overdue blockers may cause immediate rejection. Existing work continues. | D [T3] |
| Healthy plus overdue slots, or spare slots behind management | Healthy blocker permits bounded waiting; older manager still prevents barging. If all occupied inference blockers are overdue, fail fast even with spare capacity/queued managers. | D [T3] |
| Timeout zero / bulk drains | Zero means unbounded admission wait; bulk uses zero regardless of normal policy. Not an end-to-end initialization/inference/shutdown deadline. | D [T3] |
| Client disconnect while queued or waiting initialization | No scheduler cancellation token. Wait can continue, then finite work can execute before response I/O notices disconnect. Eventually release on completion/error; prompt cancellation is not promised. | D finite queued/cold/running work [T19] |
| Client disconnect executing offline request | Cannot preempt backend; join/cleanup after finite work and failed write. Response may not be delivered; release lease. | D finite model worker [T19] |
| Disconnect active stream / abandon unstarted response | Write failure unwinds callback; dropping callback releases its bound lease. Model reset/recovery depends on adapter. | D controlled stream [T14] [T18]; real streams partial |
| Shutdown stalled header/live body/non-reading response | Stop accepting; cancel socket I/O (readiness polls at most 250 ms per slice), then join. No registry/model/socket mutex held across worker join. | D loopback [T8] |
| Shutdown active finite offline handler/deferred callback | Listener waits until handler/callback returns; runtime outlives joined workers; response delivery not guaranteed. | D loopback [T8] |
| Shutdown queued request, cold load, manager or hung GPU | Existing scheduler/init/backend waits are not globally canceled. Finite work must drain; timeout-zero/hung work can delay exit indefinitely. Backend termination/cancellation is not implemented. | D finite queued/init/reconfigure/unload [T19]; U hung GPU |
| Custom frontend listener shutdown | Must independently stop callers and drain its workers before return. Default listener tests do not validate HTTPS/WebSocket/external listener ownership. | U [GAP] |

## Validation by area and known limits

Evidence on parent `366e7520`: Windows CPU controlled sessions at 1/2/4,
96 matched capacity-one checkpoint responses across CUDA/Vulkan (Piper TTS,
Canary ASR, Nemotron diarization), and cross-platform CPU/lifecycle/framing plus
Linux ASan/TSan CI. Exact output comparison removes only top-level execution
timing. These results do not certify arbitrary models, modes or counts.
The 2026-10-04 follow-up passes local Windows CPU 14/14 and ASan 11/11,
CUDA/Vulkan CLI/server builds, and 12 real-checkpoint capacity-rejection/reuse
cases (three checkpoints, both backends, legacy/parallel). Those runs also check
four startup-warning cases: present on each flagged build and absent on legacy.
The nine controlled rejection cases preserve pool identity/configuration/count
and allow valid resizing afterward. New follow-up CI results are distinct from
the parent's completed CI and recorded separately in the PR.

| Area | Label | Tested scope / missing evidence |
|---|---|---|
| Concurrent inference | Partially tested | Controlled independent offline/stream/batch sessions and FIFO at 1/2/4; real CUDA/Vulkan only capacity-one serial cold/warm/reload. No catalogue multi-slot admission, real concurrent streams/batches or Metal checkpoint execution. |
| Cold loading | Partially tested | Controlled first load/publication, failure/retry and guarded cold-vs-warm progress; real representative capacity-one loading. Exhaustive concurrent GPU initialization/OOM/platform coverage missing. |
| Reconfiguration | Partially tested | Controlled manager ordering, invalid metadata, count resize, unsupported capability preservation, backend failure/retry. Changed-checkpoint capability probes/temporary staging memory not validated on GPUs. |
| Unloading / eviction | Partially tested | Controlled lazy-load barriers, ready-idle-first bulk, overlapping selections, limits 1/2 and slow logical retirement. Exhaustive device memory guards, backend teardown faults and catalogue recovery missing. |
| Disconnects | Partially tested | Controlled write failure/dropped callbacks; default transport loopback stalled I/O. Default-listener clients disconnected during queued/init/running work drain with real handler ownership [T19]. Prompt queued-request cancellation unimplemented; real model streaming/reset and custom listeners untested. |
| Shutdown | Partially tested | Default listener owns/joins finite offline/deferred workers and cancels stalled socket I/O. Controlled cold-load/management/queued finite worker drain also passes [T19]. No explicit scheduler-wide cancellation; hung GPU and custom listeners untested. |

Known limitations are deliberately explicit: no hard backend cancellation or
bounded shutdown; no immediate disconnected-waiter cancellation; no full atomic
status snapshot; bulk unload is snapshot-based/non-atomic; logical unloaded does
not prove allocations freed; guards can serialize cold loads; warm overlap can
still serialize in the backend; capability preflight can add temporary memory
and a second primary construction; fresh failed registration remains unloaded
until corrected/retried; allocation/thread-construction failure paths are source-
audited where no injection exists. No claim of complete model/platform validation.

The 2026-10-05 admission follow-up [T19] runs the actual default listener and
parallel runtime together. It disconnects admitted and queued clients while
loading, running, reconfiguring or unloading, then stops the listener. Finite
controlled work drains before listener return; queues/leases clear and model
assets outlive sessions. This is controlled CPU evidence, not a real GPU shutdown
or a bounded-cancellation claim.

## Evidence references and maintenance

[GAP]: #validation-by-area-and-known-limits
[T0]: ../../tests/unittests/test_server_runtime_selection.cpp#L10
[T1]: ../../tests/unittests/test_server_parallel_lifecycle.cpp#L438
[T2]: ../../tests/unittests/test_server_parallel_lifecycle.cpp#L282
[T3]: ../../tests/unittests/test_server_model_slots.cpp#L179
[T4]: ../../tests/unittests/test_server_parallel_lifecycle.cpp#L506
[T5]: ../../tests/unittests/test_server_parallel_lifecycle.cpp#L308
[T6]: ../../tests/unittests/test_server_parallel_lifecycle.cpp#L348
[T7]: ../../tests/unittests/test_server_parallel_lifecycle.cpp#L527
[T8]: ../../tests/unittests/test_parallel_http_ownership.cpp#L172
[T9]: ../../tests/unittests/test_server_parallel_lifecycle.cpp#L495
[T10]: ../../tests/unittests/test_server_parallel_lifecycle.cpp#L368
[T11]: ../../tests/unittests/test_server_parallel_lifecycle.cpp#L614
[T12]: ../../tests/unittests/test_server_parallel_lifecycle.cpp#L554
[T13]: ../../tests/unittests/test_server_parallel_lifecycle.cpp#L543
[T14]: ../../tests/unittests/test_server_parallel_lifecycle.cpp#L384
[T15]: ../../tests/unittests/test_server_parallel_lifecycle.cpp#L642
[T16]: ../../tests/unittests/test_server_parallel_lifecycle.cpp#L665
[T17]: ../../tests/unittests/test_session_pool.cpp#L107
[T18]: ../../tests/unittests/test_server_parallel_lifecycle.cpp#L600
[T19]: ../../tests/unittests/test_server_parallel_lifecycle.cpp#L788

[T1] is the nearest initialization/happy-request test where a cold concurrency
combination is marked P, not proof of every such interleaving. Preserve D/P/U
labels when linking a broader fixture. The
[validation procedure](parallel_model_validation.md) defines future model admission
and output quality; the [adapter guide](parallel_sessions.md) describes ownership
interfaces. Historical [self-audit](../reports/parallel_runtime_self_audit.md) and
[separation](../reports/parallel_runtime_separation.md) reports retain their tested
revision/scope and cannot override this contract. For each behavior change,
update the relevant matrix, linked regression and coverage label together.
