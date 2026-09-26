# Parallel execution sessions

The server uses a common session pool and slot scheduler for all model families.
A model without parallel support keeps its existing single-session behavior.
Multiple slots require explicit support for the selected backend, task, mode and
session options; setting `slots` does not make an arbitrary model thread-safe.

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

Requests queue when all slots are busy. The existing busy-timeout policy applies
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

## Current implementation and validation

Higgs Audio v3 TTS on CUDA in offline mode is the first enabled adapter. Other
models and other Higgs backend/mode combinations retain one slot. The framework
has no family-specific CUDA kernels and does not change shared CUDA operations.

Default unit tests cover the session pool, scheduler and compatibility with the
old single-slot guard. Test doubles cover shared weights, separate request state,
native batches, streaming reset, clone validation, partial failure and lifetime
ordering. Extended tests cover server configuration and the original busy guard.

Local validation built the full CPU model set and passed all 49 default CTests,
plus five focused tests in a CUDA build with Higgs and BS-RoFormer enabled.
RTX 3090 HTTP validation exercised four Higgs slots, a six-request queue,
timeouts, failure recovery, unload/reload and eviction, including unloading
during the first lazy load. All 27 Higgs WAVs matched
their corresponding cold/warm reference byte for byte. A real BS-RoFormer
single-slot request matched both reference stems; requesting two slots was
rejected cleanly. Real concurrent streaming and native-batch model adapters have
not yet been enabled or validated. These checks do not establish multi-slot
support for every model or backend.

The [benchmark report](../reports/common_model_slots.md) compares one through
four shared slots with actual independent single-slot server instances, including
generation time, memory, output parity and reproduction commands.
The [Vulkan report](../reports/common_model_slots_vulkan.md) records single-slot
compatibility, same-backend WAV parity and two reproduced upstream extended-test
failures. Vulkan parallel execution remains disabled.
