# Parallel execution sessions

The common server framework owns session pools and leases slots to offline,
streaming and native-batch requests. Existing models keep their single-session
behavior by default. Parallel model adapters and audited admission rules are
the separate PR #706 follow-up; no model is newly enabled by this foundation.

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

`app/server/model_slots.h` leases one session per request. A lease spans
preparation and execution, including the entire stream or native batch. RAII
releases it on completion or exception. Native batching within one session and
multiple concurrent sessions are independent capabilities.

Requests queue when all slots are busy or management blocks admission. The existing busy-timeout policy applies
to the queue. Unload, eviction and reconfiguration require an exclusive lease;
waiting management operations block new requests so management cannot starve.
Leases must drain before destroying the pool. This does not cancel an in-flight
GPU operation or provide continuous token batching.

First-load requests serialize pool construction through a per-model
initialization mutex. The complete pool is published only after every clone has
been created successfully. Failed loads can be retried. Eviction checks atomic
loaded state rather than accessing session pointers without a lease.
Unload leases the model even before its first pool has been published, so an
in-progress lazy load cannot be skipped. Model-list lookup/snapshot locks are
released before waiting for that lease.

Do not hold the model metadata lock while loading: status reads acquire the
global model-list lock before metadata, while loading can acquire the model-list
lock for eviction. The exclusive model lease protects configuration during
reconfiguration after the metadata lock is released.

`GET /v1/models` exposes configured `slots`, current `active_slots`,
`queued_requests`, and `max_parallel_slots`. The latter is `null` until loaded
and is cleared on unload. These are live status observations, not a reservation
of memory or an atomic snapshot of the entire server.

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
