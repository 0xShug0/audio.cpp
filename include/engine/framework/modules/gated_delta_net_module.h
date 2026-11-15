#pragma once

#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"

namespace engine::modules {

// L2 clamps the norm; SumSquares adds epsilon inside the square root.
enum class GatedDeltaNetQkNorm { L2, SumSquares };

struct GatedDeltaNetConfig {
    int64_t hidden_size = 0;
    int64_t key_head_dim = 0;
    int64_t key_heads = 0;
    int64_t value_head_dim = 0;
    int64_t value_heads = 0;
    int64_t conv_kernel = 0;
    float norm_epsilon = 1e-6F;
    bool round_bf16_activations = false;
    RMSNormLowering norm_lowering = RMSNormLowering::Default;
    GatedDeltaNetQkNorm qk_norm = GatedDeltaNetQkNorm::L2;
    float qk_epsilon = 1e-6F;
    ggml_prec projection_precision = GGML_PREC_DEFAULT;
};

struct GatedDeltaNetWeights {
    LinearWeights in_proj_qkv, in_proj_z, in_proj_b, in_proj_a;
    core::TensorValue conv_weight, dt_bias, a_log;
    NormWeights norm;
    LinearWeights out_proj;
    // Optional packed projection with rows [qkv, z, a, b], replacing the four above.
    LinearWeights in_proj{};
};

struct GatedDeltaNetOutput {
    core::TensorValue output;
    core::TensorValue next_conv_tail;
    core::TensorValue next_recurrent_state;
};

// Graph construction only. Input is [batch, time, hidden]. The convolution tail
// is [batch, channels, kernel-1]; omitting it starts with zero history. Recurrent
// state and weights are borrowed. The enclosing runtime commits returned state
// tensors and owns allocation, execution, reset and release.
class GatedDeltaNetModule {
public:
    explicit GatedDeltaNetModule(GatedDeltaNetConfig config);

    const GatedDeltaNetConfig & config() const noexcept;
    const core::ModuleSchema & schema() const noexcept;
    static const core::ModuleSchema & static_schema() noexcept;

    GatedDeltaNetOutput build(core::ModuleBuildContext & ctx,
        const core::TensorValue & input, const core::TensorValue & recurrent_state,
        const GatedDeltaNetWeights & weights,
        const core::TensorValue & conv_tail = {}) const;

private:
    GatedDeltaNetConfig config_;
};

}  // namespace engine::modules
