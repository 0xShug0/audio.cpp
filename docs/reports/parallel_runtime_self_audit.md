# Parallel runtime self-audit

Audit of PR #715 after startup separation, parent revision
`612c5e9f34713b4e529f944021e7e3eb7db016f5`, against upstream
`2892ed3e94b6ccbdacc111fa2896aaae0a5dc8fc`. Production fixes apply only to
`--parallel-jobs`. The original runtime, HTTP transport, frontend interfaces,
CLI, model implementations and ggml remain unchanged.

## Findings and fixes

| Finding | Failure/interleaving | Scoped fix |
|---|---|---|
| Partial reconfiguration | Old sessions were destroyed before task/preset/spec validation could fail. | Build and validate replacement metadata privately before acquiring the management lease; commit a complete configuration under a short metadata lock. Also include config/weight selectors in reload detection. |
| Invisible first registration | Two same-ID requests could load different objects before registry publication; guarded loads could miss a newly loaded, unpublished entry. | Publish a single stable entry under an already acquired exclusive lease before its first load. Competing registrations use that entry's FIFO. Reserve vector storage before inserting the index, preventing an allocation failure from publishing a dangling index. |
| Unsynchronized live-ingest lookup | The transport read the registry and mutable per-model policy while registration/reconfiguration could write them. | Snapshot the stable entry pointer under the registry mutex, release it, then copy policy under the model metadata lock. |
| Circular bulk waits | Interleaved reverse-order selections could reserve A for one bulk operation and B for another, each waiting for the other. | Serialize only publication of all barriers for one bulk selection. Release this short lock before waits/destruction; deduplicate selected entries. |
| Null primary session | A model returning a null primary was dereferenced before pool validation. | Reject it before accessing session metadata; private model ownership unwinds normally. |
| Partial UI models-root replacement | Installer construction could fail after the visible folder was changed, leaving the old installer paired with a different public root. | Construct the replacement first; swap the root and installer together only after success. |
| Detached HTTP worker lifetime | The original listener can return while detached threads still refer to handlers, model entries or response callbacks. | Add an opt-in listener with owned, reaped/joined workers. On return or an accept/allocation/thread-construction failure, signal cancellation, interrupt socket I/O, then join before runtime destruction. |

The opt-in transport starts from the original framing implementation and has its
own translation unit. The legacy transport is not refactored or modified. It
adds receive readiness/cancellation polling, since Windows `shutdown()` alone
did not wake an already blocked header receive in the regression test. Live body
waits also poll cancellation in short slices while retaining their normal idle
and total deadlines. Accepted sockets are nonblocking; reads/writes poll readiness
and cancellation, and live send deadlines remain enforced in the write loop.
This also permits shutdown when a client stops reading a large response.
A per-worker socket mutex serializes shutdown against close
so a reused handle cannot be shut down accidentally. No such mutex is held while
parsing, invoking a handler/callback, performing model work, or joining.

The shared framing fixture had one stale assertion expecting non-live chunked
bodies to be empty. Current upstream already de-frames those bodies; both
listeners now assert the full four-byte payload. Failed fixture assertions also
drain the listener and print the actual failure rather than terminating on a
joinable thread destructor. These are test changes, not legacy runtime fixes.

## State transitions and ownership

| Operation | Success | Failure / cancellation |
|---|---|---|
| First inference/load | Lease -> private loaded model, primary, clones -> publish complete pool -> prepare/run -> release lease. | Loader/null-primary/clone failure destroys all private sessions before weights, leaves no published pool or residency, and releases lease. Retry is allowed. |
| Existing reconfiguration | Validate private metadata -> exclusive FIFO lease -> unload old pool -> swap complete metadata -> privately construct/publish replacement pool. | Metadata validation failure preserves working state. Backend/pool failure after the swap leaves the new valid entry unloaded for retry; old weights are not retained, avoiding duplicate device residency. |
| New registration | Validate -> pre-acquire management lease -> publish stable entry -> load -> release. | Validation/publication failure leaves no new entry; first-load failure leaves one registered, unloaded entry that can retry. No registry entries are erased while workers may hold pointers. |
| Request preparation/inference | Independent leased session owns mutable request state. | Framework RAII releases admission on exceptions; model-specific reset/recovery remains the model adapter's contract. |
| Deferred stream/native batch | Response callback retains the same shared lease used during preparation/validation. | Dropping an unstarted callback, client-write failure, or callback destruction releases the lease. Runtime lifetime is retained by joining its HTTP workers, not by the lease alone. |
| Unload/eviction | Exclusive lease; clear logical residency/capacity, destroy sessions, then loaded model. | Physical memory can remain allocated during teardown. Memory guards query actual available memory; clearing status is not proof memory is already free. |
| Pending bulk barriers | Reservations grant exclusive management ownership or wait in model FIFO. | Destruction cancels queued reservations or releases already granted ones. Partial barrier-publication failure unwinds all reservations. |

`VoiceTaskSessionPool` destroys clones before the primary, and the loaded-model
holder outlives the privately constructed pool. Registered heap object addresses
remain stable when the registry vector grows. Status reads atomic residency and
capacity, not session pointers that could be destroyed concurrently; status is an
observation, not a globally atomic reservation.

## Where work can wait

| Location | Lock scope / effect on another model |
|---|---|
| Per-model FIFO | Admission or management barrier waits only on that model. Effective busy timeout applies to normal admission; zero permits unbounded waits. Bulk drains deliberately reserve with timeout zero. |
| Per-model initialization mutex | Parallel cold requests for the same model share one pool construction. Already loaded unrelated models do not take this mutex. This wait is not an end-to-end request timeout. |
| Registry mutex | Lookup, snapshot and publication only. Never held across model admission, metadata waits, checkpoint I/O, GPU work or teardown. |
| Metadata shared mutex | Short configuration snapshots/commit. Replacement validation, teardown and loading run outside its exclusive scope. A model's status/policy can wait on its commit; unrelated registry operations remain available. |
| Bulk publication mutex | Only another bulk operation waits here, while barriers are enqueued. Released before waiting on any barrier or destroying any model. |
| Guarded load mutex | Cold load/eviction/memory-check serialization is intentional when `max_loaded_models` or memory headroom is enabled. It can block another cold load; warm inference bypasses it. With both guards disabled, independent cold loads do not take it. Eviction try-acquires victims rather than waiting on them. |
| JSON parser mutex | Enabled only in the flagged process, around cJSON parsing; it briefly serializes parsing across models, not input preparation or inference. |
| Native installer mutex | Shared UI installation/status/root-management operations can wait on one another. Inference never takes this mutex. Installer-internal download/conversion lifetime is outside the slot scheduler and unchanged. |
| Asset/weight caches | Optional locks are per source/bundle/loaded package. Source reads/uploads serialize with storage release; immutable-weight factories may serialize components in that package. They do not form a server-wide inference lock. Registered resources are immutable after publication. |
| Socket I/O / shutdown | Workers own socket I/O and deferred callbacks. Cancellation interrupts reads/writes; joins take no registry/model/socket lock. In-flight backend work is not preempted, so a hung GPU call can still delay shutdown indefinitely. Custom frontend listeners must independently honor the same drain-before-return contract. |

## Validation

Three focused regressions were compiled with the previous parallel runtime and
failed for invalid reconfiguration, concurrent registration and live-policy
metadata access. Their fixed versions pass. Controlled handler tests exercise
capacities 1/2/4, loader/null-primary/partial-clone failures, preparation/run
failure, replacement failure/retry, abandoned deferred responses, reversed and
duplicate bulk selections, unrelated-model progress and guarded cold-load versus
warm inference. Existing FIFO, timeout, stream/disconnect, batch, metadata,
residency and shared-storage regressions remain enabled.

Loopback tests cover the same body framing on both transports, shutdown while
an offline handler or deferred callback is active, a stalled partial header,
stalled live ingest and a client that stops reading a large callback response.
The CI lifecycle and ASan/TSan target lists include the
two new transport tests. Local evidence is retained in
`outputs/pr715-self-audit-20261003/`, including negative-case results, builds,
JUnit records and the final source-boundary check.

This audit does not admit additional model/backend/mode combinations. Local
Windows ASan and representative CUDA/Vulkan checkpoints do not substitute for
Linux/macOS/TSan CI on this changed revision.

| Final local check | Result |
|---|---|
| CPU/framework/framing/ownership CTest | 14/14 pass |
| Windows AddressSanitizer CTest | 11/11 pass |
| Repeated real-handler lifecycle and HTTP ownership tests | 20 consecutive passes each (40 executions) |
| CUDA CLI/server build with native UI manager | Pass |
| Vulkan CLI/server build with native UI manager | Pass |
| Matched upstream/legacy/flagged-omitted/flagged-one checkpoint outputs | 96/96 exact output/metadata matches after removing top-level execution timing |
| Native UI folder change, invalid-folder rejection, state retention and reset | Pass on both builds |
| Source boundary versus current upstream and whitespace check | Pass |

The checkpoint pass uses Piper TTS, Canary ASR and Nemotron diarization across
cold, two warm requests and unload/reload on each backend; it is not a catalogue
multi-slot or performance verdict. The UI smoke verifies normal commit and
early rejection; installer-constructor allocation failure is source-audited,
not fault-injected. These audit changes were validated locally before publication. CI targets
include the new regressions; cross-platform CI results are tracked separately
on the pushed revision.
