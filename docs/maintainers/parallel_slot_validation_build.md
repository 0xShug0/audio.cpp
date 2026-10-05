# Experimental model-slot validation builds

Follow-up to PR #715's opt-in `--parallel-jobs` runtime. These build-only hooks
allow collecting model-adapter evidence before adding production admission.
The hooks do not change scheduling, inference, runtime selection, or the default
audited catalogue. No model is admitted by this change.

Backend initialization is separately guarded in `core::init_backend`, across
models and slots, because Vulkan can publish partially initialized device state.
`BestAvailable` follows the same guard. Registry discovery has its own short
guard. These guards protect construction only; they do not serialize uploads,
graph execution, requests, or backend lifetimes. `backend_initialization_test`
and its Vulkan variant cover cold creation, shared device weights and retry.

Configure with `-DENGINE_BUILD_TESTS=ON -DAUDIOCPP_SLOT_VALIDATION=ON`. An optional
`-DAUDIOCPP_SLOT_VALIDATION_CAPACITY_HEADER=/absolute/path/to/fixture.h` selects
temporary admission tables. Both options default off/empty; a capacity header
without the validation switch, missing header, or validation without test builds
is rejected. Keep these binaries separate from deployment builds.

The fixture is included inside `minitts::server`, after `AuditedModelSlots` is
declared. It must define `kAuditedCudaOfflineModels` and
`kAuditedVulkanOfflineModels` as constexpr arrays of that type. Include only the
family/task/backend capacities under test; other families, tasks, CPU and
streaming retain normal admission. The unit-test fixture uses a fictional family.
An override is a test allowance, **not a validation pass**.

Start the server with `--parallel-jobs` and explicit config `slots` as in #715.
For lease-history traces, additionally set
`AUDIOCPP_SLOT_VALIDATION_OBSERVER=1` in the server environment. Offline traces
record model, slot, session address, text, seed, and audio/reference lengths.
Without the environment variable, no trace is emitted. Trace builds may record
request text in logs; collect only the intended validation fixtures.

Use `server_slot_validation_default_test` and
`server_slot_validation_fixture_test` to check default and override isolation.
Run the usual server lifecycle CTests too. These hooks do not certify real
GPU ownership, output quality, unsupported higher counts, or shutdown; use the
[model validation procedure](parallel_model_validation.md) for those verdicts.
