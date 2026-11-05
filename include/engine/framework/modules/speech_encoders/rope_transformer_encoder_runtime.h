#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"

#include <functional>
#include <memory>
#include <vector>

namespace engine::modules {

enum class EncoderRoPE { Indexed, SplitTables };

struct RopeTransformerEncoderConfig {
    int64_t hidden_size = 0;
    int64_t intermediate_size = 0;
    int64_t heads = 0;
    int64_t rotary_dimensions = 0;
    float norm_epsilon = 1e-5f;
    float rope_theta = 10000.0f;
    EncoderRoPE rotary = EncoderRoPE::Indexed;
    ggml_prec projection_precision = GGML_PREC_DEFAULT;
    bool qk_norm = false;
    bool input_norm = true;
    bool flash_attention = true;
    // Some checkpoints round K/V even when using explicit attention.
    bool f16_kv = false;
};

struct RopeTransformerEncoderLayerWeights {
    NormWeights norm1, norm2, q_norm, k_norm;
    LinearWeights qkv, attention_output, ffn_in, ffn_out;
};

struct RopeTransformerEncoderWeights {
    NormWeights input_norm, final_norm;
    std::vector<RopeTransformerEncoderLayerWeights> layers;
};

struct RopeTransformerEncoderInputs {
    core::TensorValue positions;
    core::TensorValue cosine, sine;
    std::optional<core::TensorValue> attention_mask;
};

// Builds only the encoder: checkpoint names, feature stacking, task heads and
// recurrent speaker policy belong to callers. Activations are [batch, time, hidden].
class RopeTransformerEncoderModule {
public:
    explicit RopeTransformerEncoderModule(RopeTransformerEncoderConfig config);
    core::TensorValue build(core::ModuleBuildContext & ctx, const core::TensorValue & input,
        const RopeTransformerEncoderInputs & inputs, const RopeTransformerEncoderWeights & weights) const;

private:
    RopeTransformerEncoderConfig config_;
};

// Graph composition and allocation are separate from encoder math. The head
// runs only when constructing a graph and may append model-owned inputs/outputs.
struct RopeTransformerEncoderGraphConfig {
    using HeadBuilder = std::function<std::vector<core::TensorValue>(
        core::ModuleBuildContext &, const core::TensorValue &)>;

    std::optional<LinearWeights> input_projection;
    bool masked_attention = false;
    size_t arena_bytes = 0;
    HeadBuilder head;
};

// Weights and the execution context remain caller-owned and must outlive this.
class RopeTransformerEncoderRuntime {
public:
    RopeTransformerEncoderRuntime(const core::ExecutionContext & execution,
        RopeTransformerEncoderConfig config, RopeTransformerEncoderWeights weights,
        RopeTransformerEncoderGraphConfig graph_config = {});
    ~RopeTransformerEncoderRuntime();
    RopeTransformerEncoderRuntime(const RopeTransformerEncoderRuntime &) = delete;
    RopeTransformerEncoderRuntime & operator=(const RopeTransformerEncoderRuntime &) = delete;

    // Bindings are borrowed until a shape change, release, or destruction.
    // Calls and consumers must be sequential, including model-owned extra inputs.
    void prepare(int64_t batch, int64_t frames);
    const core::TensorValue & input() const;
    const core::TensorValue & attention_mask() const;
    const core::TensorValue & output(size_t index = 0) const;
    void run();
    void release_workspace();
    size_t graphs_built() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace engine::modules
