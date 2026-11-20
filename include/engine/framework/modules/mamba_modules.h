#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/module.h"

#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace engine::modules {

// One Mamba-1 selective state-space layer in the mamba_ssm layout:
// in_proj -> causal depthwise conv + SiLU -> x_proj -> softplus(dt_proj) ->
// selective scan with the D skip and the SiLU(z) gate -> out_proj.
struct Mamba1Config {
    int64_t model_size = 0;
    int64_t inner_size = 0;
    int64_t state_size = 16;
    int64_t conv_kernel = 4;
    int64_t dt_rank = 0;
};

struct Mamba1Weights {
    core::TensorValue in_proj;      // [2 * inner, model]
    core::TensorValue conv_weight;  // [inner, conv_kernel], F32
    core::TensorValue conv_bias;    // [inner], F32
    core::TensorValue x_proj;       // [dt_rank + 2 * state, inner]
    core::TensorValue dt_weight;    // [inner, dt_rank], F32
    core::TensorValue dt_bias;      // [inner], F32
    core::TensorValue a;            // [inner, state], -exp(A_log), F32
    core::TensorValue d;            // [inner], F32
    core::TensorValue out_proj;     // [model, inner]
};

// Recurrent state carried from one call to the next: F32 tensors the caller owns.
struct Mamba1State {
    core::TensorValue conv;  // [batch, inner, conv_kernel - 1]
    core::TensorValue ssm;   // [batch, inner, 1, state]
};

struct Mamba1Outputs {
    core::TensorValue output;  // [batch, length, model]
    // With a carried state: the state after the last token. The caller copies it
    // into Mamba1State after every reader of the old state in the graph.
    std::optional<Mamba1State> next_state;
};

class Mamba1Module {
public:
    explicit Mamba1Module(Mamba1Config config);

    const Mamba1Config & config() const noexcept;

    // input [batch, length, model]; sequence_ids is I32 [batch] holding 0..batch-1.
    // Without a state the scan starts from zeros and the convolution sees zeros
    // before the first token.
    Mamba1Outputs build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const Mamba1Weights & weights,
        const core::TensorValue & sequence_ids,
        const std::optional<Mamba1State> & state = std::nullopt) const;

private:
    Mamba1Config config_;
};

namespace binding {

// Loads mamba_ssm weights under `prefix`. Projections use `storage`; the
// convolution and scan parameters stay F32, as the fused kernels read them.
template <typename Store>
Mamba1Weights mamba1(
    Store & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    const Mamba1Config & config,
    assets::TensorStorageType storage) {
    const int64_t inner = config.inner_size;
    const int64_t state = config.state_size;
    auto a = source.require_f32(prefix + ".A_log", {inner, state});
    for (auto & value : a) {
        value = -std::exp(value);
    }
    using assets::TensorStorageType;
    return {
        store.load_tensor(source, prefix + ".in_proj.weight", storage, {2 * inner, config.model_size}),
        store.load_tensor_as_shape(source, prefix + ".conv1d.weight", TensorStorageType::F32,
            {inner, 1, config.conv_kernel}, core::TensorShape::from_dims({inner, config.conv_kernel})),
        store.load_tensor(source, prefix + ".conv1d.bias", TensorStorageType::F32, {inner}),
        store.load_tensor(source, prefix + ".x_proj.weight", storage, {config.dt_rank + 2 * state, inner}),
        store.load_tensor(source, prefix + ".dt_proj.weight", TensorStorageType::F32, {inner, config.dt_rank}),
        store.load_tensor(source, prefix + ".dt_proj.bias", TensorStorageType::F32, {inner}),
        store.make_f32(core::TensorShape::from_dims({inner, state}), std::move(a)),
        store.load_tensor(source, prefix + ".D", TensorStorageType::F32, {inner}),
        store.load_tensor(source, prefix + ".out_proj.weight", storage, {config.model_size, inner}),
    };
}

}  // namespace binding

}  // namespace engine::modules
