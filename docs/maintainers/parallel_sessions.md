# Parallel execution sessions

The common server framework owns session pools and leases slots to offline,
streaming and native-batch requests. Existing models keep their single-session
behavior by default. Parallel model adapters and audited admission rules are
the separate PR #706 follow-up; no model is newly enabled by this foundation.

The [Parallel Server Behavior Contract](parallel_server_behavior_contract.md)
is the single authority for startup selection, state transitions, ordering,
management, errors, status, disconnect and shutdown behavior, and tested scope.
`--parallel-jobs` opts into the experimental path and prints its coverage warning;
the CLI and unflagged server retain their original implementation. This guide
covers adapter interfaces and implementation ownership only.

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

## Adapter ownership interfaces

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

## Server integration

`parallel_runtime.h/.cpp` owns the parallel registry, pool leases and metadata.
`model_slots.h` provides the FIFO admission primitive; `parallel_http.cpp` owns
default-listener workers. Consult the
[behavior matrices](parallel_server_behavior_contract.md#operations-crossed-with-model-states)
for the expected outcomes and linked tests. Custom frontend listeners must
honor that document's drain-before-return contract independently.

Unsupported replacement capacity must be validated before retiring working
sessions. Capability depends on the effective backend/task/mode/session options;
a changed checkpoint may require a private primary probe. See the contract's
failure matrix for pre-commit versus post-commit failure and staging-memory limits.

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
