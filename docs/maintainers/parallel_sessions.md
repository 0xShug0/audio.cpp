# Parallel execution sessions

The common server framework owns session pools and leases slots to offline,
streaming and native-batch requests. Existing models keep their single-session
behavior by default. Parallel model adapters and audited admission rules are
the separate PR #706 follow-up; no model is newly enabled by this foundation.

The process selects its implementation once at startup:

| Startup | Model configuration | Runtime |
|---|---|---|
| No `--parallel-jobs` | No `slots` field | Original `ServerState` and `BusyGuard` |
| No `--parallel-jobs` | Any explicit `slots`, including `1` or `null` | Configuration/registration error |
| `--parallel-jobs` | No `slots` field | `ParallelServerState`, capacity one |
| `--parallel-jobs` | Integer `slots` from 1 to 16 | `ParallelServerState`, validated per-model capacity |

```sh
audiocpp_server --config server.json
audiocpp_server --config server.json --parallel-jobs
```

Slot count never selects the implementation. Even capacity one in an enabled
process uses a pool, leases, FIFO admission and the parallel lifecycle. Dynamic
registration follows the selected process policy. Reconfiguration can change a
parallel model's count from one to several or back under an exclusive lease;
it cannot switch the process to the legacy implementation. Unsupported capacities
still fail before pool publication. The inference CLI has no new slot option.

The original runtime header, busy guard, transport, frontend interfaces, session
interface, inference CLI, model implementations and ggml files are unchanged
relative to the upstream main used for the change. The original runtime source
adds only explicit-slot rejection in dynamic registration. All execution and
lifecycle changes live in `app/server/parallel_runtime.h/.cpp`. Verify the boundary
with `python tests/server/check_parallel_boundaries.py --base origin/main`. Carry relevant
future upstream changes into both runtime copies without routing legacy calls
through parallel helpers.

The optional loaded-model factory creates extra sessions sequentially before
publishing a pool. Assets can be shared while execution state remains private.
Explicit parallel adapters take precedence, including a rejection of the chosen
backend/task/mode. The admission extension point is initially empty on all
backends. Add a model only after checkpoint-backed cold/warm, mixed-request,
unload/reload, output and memory validation at every advertised slot count.

## Runtime contract

Implement `engine::runtime::IParallelVoiceTaskSessionFactory` alongside the
model's existing session interfaces. Its two methods are:

```cpp
size_t parallel_session_capacity() const noexcept override;
std::unique_ptr<IVoiceTaskSession> create_parallel_session() const override;
```

The capacity includes the primary session. Return `1` for unsupported
combinations and a supported upper bound otherwise. The common limit is
`kMaxParallelSessions` (16). This is an implementation capability, not a promise
that every slot count fits available memory.

Share immutable weights across sessions. Give each session its own mutable
execution state: backend execution context, graph allocator and buffers, KV and
reference caches, streaming state, callbacks and sampler/RNG. Audit any lazy
initialization or global scratch space as well. Do not share a generator merely
because it points to immutable weights. A backend must also permit concurrent
execution with the resulting contexts.

Clones must report the same family, task and mode and implement the same
offline, streaming and native-batch interfaces as the primary. Cloning occurs
before inference begins. The primary outlives every clone, including during
partial construction failure. Propagate allocation failures; the pool owns and
cleans up partially constructed sessions.

`VoiceTaskSessionPool` in `include/engine/framework/runtime/session_pool.h`
validates this contract, owns the primary and clones, and exposes sessions by
slot index. It does not schedule work. It cannot verify that a model adapter
has actually isolated all mutable state; that requires adapter review and tests.

## Server lifecycle

`app/server/main.cpp` constructs exactly one stack-owned concrete runtime and
uses the existing transport/frontend interfaces. The flagged default HTTP path
uses `parallel_http.cpp`, which owns, cancels socket I/O, and joins request workers
before returning. The original listener remains unchanged for default startup.
Custom frontend listeners must likewise drain their workers before returning.
Neither runtime delegates
ownership or lifecycle operations to the other.

`app/server/model_slots.h` is used only on the parallel side and leases one
independent session per request, including when capacity is one. Parallel handlers bind a lease before reading model-dependent
configuration or preparing inputs, and retain it through result serialization,
the entire stream or native batch. Deferred response callbacks own the same
lease used for validation; dropping a callback releases it. RAII
releases it on completion or exception. Native batching within one session and
multiple concurrent sessions are independent capabilities.

Requests and management enter one FIFO admission queue per opt-in model. Order
is defined when a caller joins that queue, not by network arrival timestamps.
Consecutive requests can fill available slots. A management operation is a barrier
for later arrivals: earlier queued requests drain first, then the manager runs
exclusively. Per-waiter notifications assign ownership before waking the caller;
a later caller cannot take an older caller's reservation. The existing busy-timeout
and timeout-zero policy applies; expired/cancelled callers release their queue entry.
Leases must drain before destroying the pool. This does not cancel an in-flight
GPU operation or provide continuous token batching.

Parallel first-load requests serialize pool construction through a per-model
initialization mutex. The complete pool is published only after every clone has
been created successfully. Failed loads can be retried. Eviction checks atomic
logical residency rather than accessing session pointers without a lease.
Residency and public loaded state clear before pool/weight destruction begins,
so a retiring model cannot cause a resident-limit rejection or needless eviction.
This does not mean its physical memory is already free: the optional memory
guard still checks actual available memory, and unload completes after teardown.
Parallel unload leases the model even before its first pool has been published, so an
in-progress lazy load cannot be skipped. Model-list lookup/snapshot locks are
released before waiting for that lease. Bulk unload first reserves barriers for
all selected parallel models, unloads ready idle pools, then waits for blocked
models. An unpublished initial load therefore cannot delay releasing unrelated
idle pools. Full status/voice lookups also release the registry lock before
waiting for model metadata or inspecting filesystem entries.

Bulk barrier publication uses one short mutex to prevent reversed, overlapping
bulk selections from reserving each other's models in a circular wait. It is
released before draining; requests and single-model management do not take it.
Duplicate selected IDs are drained once. Dynamic registration publishes one
stable entry under an exclusive lease before loading, so concurrent registrations
and resident-limit accounting see the same entry. Metadata validation occurs
before replacing live state. If backend loading fails after replacement, the
complete validated new configuration remains registered but unloaded for retry;
this does not retain or recreate the old device weights.

See [the self-audit](../reports/parallel_runtime_self_audit.md) for transition,
ownership, lock-scope and wait-location details, including intentional cold-load
serialization under memory/resident guards and shutdown limits.

The legacy runtime preserves upstream main's execution, locking, loading,
unloading, streaming, batching, eviction and shutdown bodies. Its bulk unload
continues to skip unpublished first loads, exactly as upstream does.

The FIFO, bulk-release and residency behavior has dedicated controlled
real-handler regressions under F1/F2/F5. Passing them alone does not establish
the complete framework gate or admit another model/backend/mode.

Never wait for model metadata or admission while holding the global registry
lock. Do not hold metadata while loading or entering eviction of another model.
The exclusive model lease protects configuration during
reconfiguration after the metadata lock is released.

`GET /v1/models` exposes configured `slots`, current `active_slots`,
`queued_requests`, and `max_parallel_slots`. The latter is `null` until loaded
and is cleared on unload. These are live status observations, not a reservation
of memory or an atomic snapshot of the entire server.

## Explicit shared-framework policies

Parallel startup calls `engine::io::json::enable_serialized_json_parsing()` before
configuration parsing, runtime construction or concurrent parser users. This is
a startup-only process policy with no disable operation. Only the cJSON parsing
call is serialized; conversion, validation, errors and deletion retain upstream
behavior. The current production cJSON parse caller is the framework JSON wrapper;
audit new direct callers before admitting them. Legacy startup and the CLI leave
the policy off.

`TensorSourceOptions{true}` explicitly requests synchronization for a source.
Existing `open_tensor_source` and indexed-shard overloads retain unsynchronized
behavior. New options-taking overloads forward the immutable policy to every
GGUF/Safetensors shard. Prefix, folded and composite views retain the underlying
sources; they do not create independent locks around shared storage.

`ResourceBundle(root, ResourceBundleOptions{true})` separately opts into source
cache synchronization. `bundle.open_tensor_source(id, TensorSourceOptions{true})`
selects synchronized storage. Both the resource-ID and canonical-path caches
separate synchronized and unsynchronized identities. Configure registrations
before publishing a bundle. Model adapters must explicitly request these policies
before sharing assets; adding the overloads alone does not enable any loader.

## Enabling another model

Follow the [parallel model validation procedure](parallel_model_validation.md)
before adding or increasing admission. It uses six model entries (M1-M6) and
references six shared framework entries (F1-F6) once per relevant revision/scope.
Every advertised count still needs three fresh starts, and the largest count
needs ten mixed warm waves per start. Exact output/metadata and applicable cache
history checks remain required. Collect memory in those same runs; independent
server comparisons are optional performance evidence. Model checks, shared
framework checks and performance have separate verdicts; admission needs both
required gates. This defines evidence, not tests that have all already run.

1. Split immutable weights from mutable inference state in the model adapter.
2. Implement the factory and advertise only audited backend/task/mode combinations.
3. Compare isolated single-session and concurrent outputs with fixed seeds and
   equivalent cache state. Include different requests and reference inputs to
   detect contamination between sessions.
4. Exercise lazy load, repeated requests, failures, unload/reload and eviction.
   Streaming adapters also need disconnect/reset tests; native-batch adapters
   need concurrent batches. Measure peak memory and throughput separately.

No family-specific routing change is needed in the server. Existing adapters
without the factory work with one slot and reject larger configurations.

## Framework validation

Unit tests cover clone contracts, ownership/destruction ordering, construction
rollback, independent offline/batch/stream state, slot leases, queue drainage,
exclusive management, timeout-zero behavior and immediate rejection of overdue
work even when unload is waiting. The shared weight cache tests cover concurrent
loading, failed-load retry, immutable ownership and release. Tensor-source tests
cover concurrent copying/scalar reads while mapped storage is released.

The common cross-platform server lifecycle workflow builds and runs these
backend-neutral checks. Real concurrent streaming/native-batch adapters still
need model-specific validation. See [the split validation report](../reports/common_slot_framework.md)
and [the scheduler regression report](../reports/scheduler_busy_timeout.md).
