# NEON Library

The native library lives under `framework/neon/`. Its `engine_neon` target
builds independently of model code and GGML. The runtime archive includes these
objects, so consumers keep linking the same `engine_runtime` library.

Build with `-DENGINE_ENABLE_NEON=ON` to opt in. The option defaults to `OFF`.
When disabled, the shared library, Moonshine native decoder and OpenWakeWord
native embedding are excluded from compilation and linking. Their graph paths
remain available; explicit native requests report the required build option.
SanoTTS instead retains main's existing local NEON decoder and selection behavior.
Enabling the option substitutes the shared-library SanoTTS implementation behind
the same execution-plan contract. The two decoder implementations are mutually
exclusive in each build.

## Contracts

- `tensor.h`: F32 feature views, Int32 index views, and explicit channel-first or frame-major layout.
  Sequence dimensions are [channels, frames]; image dimensions are
  [channels, height, width].
  Shapes use inline storage for up to four dimensions.
- `operators.h`: prepared convolution, linear/block-int8 linear, embedding,
  rotary embedding, attention, normalization, activation and tensor composition.
  Operators own immutable weights and can be shared by independent plans.
- `execution.h`: input/value ownership, shape validation, activation lifetimes,
  reusable packing/attention workspace and a persistent worker team.
- `blocks.h`: residual convolution and ConvNeXt compositions built from those
  ordinary operators.

Operator implementations contain the math. The executor runs a generic data-flow
plan; it does not contain decoder/CNN stage enums or model-specific blocks.
Intermediate storage is reused after its last consumer. Eligible elementwise
operators run in place on owned buffers; borrowed inputs stay read-only.
Convolution and attention can write frame-major output directly.

Models declare their architecture and retain weight-name mapping, reference/state
management and model-specific transformations. For example, sanoTTS retains its
acoustic-output/embedding folding; its convolution, normalization, activation and
residual execution use library operators. OpenWakeWord retains its frontend
transforms and streaming history.

Each selected model component has its own native architecture file:
`sanotts/neon_library_decoder.cpp`, `moonshine_asr/neon_decoder.cpp`, and
`open_wake_word/neon_embedding.cpp`. These compile with the repository's
`include/` path and `ENGINE_HAVE_NEON=1`; no GGML include path is needed. Weight names, loading, storage
conversion and backend binding live outside those files.

```cpp
#include "engine/framework/neon/execution.h"

namespace arm = engine::neon;

arm::ExecutionPlan plan;
const auto input = plan.input();
const auto output = plan.add(arm::Conv1d({2, 2, 1}, {1, 0, 0, 1}), {input});
plan.set_output(output, arm::Layout::FrameMajor);

std::vector<float> samples{1, 2, 3, 4};
const auto & result = plan.run({arm::TensorView(samples, {2, 2})}, 4);
// result.values == {1, 3, 2, 4}
```

Input storage must remain valid throughout `run()`. The returned tensor belongs
to the plan and is valid until its next run. A plan is used serially; concurrent
requests use independent plans and may share prepared operators. Changing shapes
or thread counts replans storage or replaces the worker team before dispatch.
Validation failures occur before workers execute, and the plan can be reused.

Use `retain(value)` for intermediate results that must survive the run;
`view(value)` returns a read-only view after successful execution. Moonshine
uses this for KV state, with cache ownership and advancement in the model.
Linear and embedding constructors accept weights by value: lvalues are copied
for intentional reuse, and rvalues transfer ownership into prepared operators.

## Component Selection

Use the existing `<family>.exec_plan` session option and `parse_exec_plan`.
Each model declares its component implementations; `RuntimeSessionBase` remains
unchanged. The framework validates component names, implementation names,
backends, artifacts and parameters. The model checks native availability and
chooses between its existing graph and prepared native component.

- Omission, `{}`, and `impl: "default"` preserve the model's placement defaults.
- `impl: "ggml"` selects the existing graph.
- `impl: "neon"` requires an AArch64 NEON CPU session.
- Component backend overrides must match the resolved session backend.
- These implementations accept no artifacts or parameters.

For example: `--session-option 'moonshine_asr.exec_plan={"decoder":{"impl":"neon"}}'`.
Existing embedded contracts can omit the option; model-local validation admits
only this new key, and `parse_exec_plan` still validates its contents.

Current production selections are `decoder` for sanoTTS nano voices,
`decoder` for Moonshine's complete autoregressive decoder, and `embedding` for OpenWakeWord's CNN.
sanoTTS retains its existing ARM default; Moonshine and OpenWakeWord keep their
graph defaults. There is no additional ARM-specific session selector.

Keep autoregressive caches, speaker state, weight folding tied to a model's
architecture and application policy in the model. Add a framework block when its
math, packing or workspace management can be reused, rather than moving a whole
model implementation into the framework.

## Validation

`neon_library_test` links only the native library. It checks scalar-reference
math, branching, buffer reuse, layouts, shape/thread changes, invalid-input
recovery and concurrent plans sharing weights. It includes ConvNeXt.

`neon_blocks_test` supplements this with kernel geometry, vector tails,
layouts and component selection. The sanoTTS decoder test
checks the full decoder against an independent double-precision implementation.
The independently linked Moonshine decoder test checks its whole decoder
against a double-precision reference, including cache reset and changing memory lengths.

The actual-model tests validate OpenWakeWord frontend/embedding/probabilities
against official intermediates, including
streaming reset and irregular input chunks.
