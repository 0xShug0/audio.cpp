#pragma once

#include <array>
#include <optional>

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/attention/scaled_dot_product_attention.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/conditioning_modules.h"
#include "engine/framework/modules/feed_forward_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/positional_modules.h"

#include <string>
#include <vector>

namespace engine::modules {

enum class DitModulation { None, PerLayer, SharedTable };
enum class DitProjectionType { Native, Bf16Autocast };
enum class DitQKNorm { None, Heads, Projection };
enum class DitResidualGate { Linear, SigmoidOneMinus };
enum class DitNormalization { Layer, Rms };
enum class DitFeedForward { Gelu, SwiGlu };
enum class DitEmbeddingActivation { Silu, Gelu };

struct DitAttentionConfig {
    int64_t heads = 0;
    int64_t kv_heads = 0;
    int64_t head_dim = 0;
    DitQKNorm qk_norm = DitQKNorm::None;
    float norm_eps = 1e-6f;
    RoPEConfig rotary;
    ScaledDotProductAttentionLowering lowering = ScaledDotProductAttentionLowering::Flash;
    bool differential = false;
    ggml_prec precision = GGML_PREC_F32;
};

struct DitConfig {
    int64_t input_channels = 0;
    int64_t output_channels = 0;
    int64_t hidden_size = 0;
    int64_t intermediate_size = 0;
    int64_t layers = 0;
    float norm_eps = 1e-6f;
    DitModulation modulation = DitModulation::None;
    DitProjectionType projection_type = DitProjectionType::Native;
    ggml_prec projection_precision = GGML_PREC_DEFAULT;
    DitResidualGate residual_gate = DitResidualGate::Linear;
    DitNormalization normalization = DitNormalization::Layer;
    DitFeedForward feed_forward = DitFeedForward::Gelu;
    GeluApproximation gelu_approximation = GeluApproximation::Tanh;
    DitAttentionConfig self_attention;
    DitAttentionConfig cross_attention;
    int64_t context_dim = 0;
    int64_t global_dim = 0;
    int64_t memory_tokens = 0;
    int64_t local_condition_dim = 0;
    int64_t time_features = 0;
    DitEmbeddingActivation context_activation = DitEmbeddingActivation::Silu;
    // Grouped positional convolutions are causal, stride one, with Mish between stages.
    int64_t position_kernel = 0;
    int64_t position_groups = 1;
};

// Checkpoint naming belongs to binding, not architecture or graph dispatch.
// Empty names denote absent parameters. {layer} is expanded by the loader.
struct DitLinearNames { std::string weight, bias; };
struct DitNormNames { std::string weight, bias; };
struct DitAttentionNames {
    DitLinearNames query, key, value, output;
    DitLinearNames qkv, kv;
    DitNormNames query_norm, key_norm;
    std::vector<DitLinearNames> qkv_parts{};
};
struct DitLayerNames {
    DitNormNames self_norm, cross_norm, feed_forward_norm;
    DitAttentionNames self_attention, cross_attention;
    DitLinearNames feed_forward_in, feed_forward_out;
    DitLinearNames modulation;
    std::vector<DitLinearNames> local_embedding;
    std::string modulation_table;
};
struct DitWeightNames {
    DitLinearNames input, output, preprocess, postprocess;
    std::vector<DitLinearNames> time_embedding, context_embedding, global_embedding, duration_embedding;
    DitLinearNames time_modulation, final_modulation;
    std::string final_modulation_table;
    std::string memory_tokens;
    std::vector<DitLinearNames> global_modulation;
    std::vector<DitLinearNames> position_convolutions;
    DitLayerNames layer;
    // The final affine head's source rows are either [scale,shift] or [shift,scale].
    // This is a checkpoint binding convention; blocks always use named planes.
    std::array<int, 6> modulation_planes{0, 1, 2, 3, 4, 5};
    int final_shift_plane = 0;
    int final_scale_plane = 1;
};

struct DitAttentionWeights {
    LinearWeights query, key, value, output, qkv, kv;
    NormWeights query_norm, key_norm;
};
struct DitLayerWeights {
    NormWeights self_norm, cross_norm, feed_forward_norm;
    DitAttentionWeights self_attention, cross_attention;
    LinearWeights feed_forward_in, feed_forward_out, modulation;
    std::vector<LinearWeights> local_embedding;
    core::TensorValue modulation_table;
};
struct DitWeights {
    LinearWeights input, output, preprocess, postprocess;
    std::vector<LinearWeights> time_embedding, context_embedding, global_embedding, duration_embedding;
    LinearWeights time_modulation, final_modulation;
    core::TensorValue final_modulation_table, memory_tokens;
    std::vector<LinearWeights> global_modulation;
    std::vector<Conv1dWeights> position_convolutions;
    std::vector<DitLayerWeights> layers;
    std::array<int, 6> modulation_planes{0, 1, 2, 3, 4, 5};
    int final_shift_plane = 0;
    int final_scale_plane = 1;
};

DitWeights load_dit_weights(core::BackendWeightStore & store,
    const assets::TensorSource & source, const DitConfig & config,
    const DitWeightNames & names, assets::TensorStorageType storage, core::BackendType backend);

struct DitContextKV { core::TensorValue key, value, differential_key; };
struct DitSelfCacheInputs {
    // Head-major prefix views; destinations are flat B,T,H,D banks.
    std::vector<DitContextKV> prefix, destination;
    core::TensorValue slots;
    int64_t append_tokens = 0;
    std::optional<ggml_type> storage_type;
};
struct DitForwardInputs {
    // Canonical logical layouts are B,T,C and B,F (time/global features).
    core::TensorValue latent, time, context, global, positions, value_mask, local_condition;
    core::TensorValue duration, time_condition, duration_condition, modulations, attention_mask;
    std::optional<DitSelfCacheInputs> self_cache;
    int64_t output_start = 0, output_tokens = 0;
    std::vector<core::TensorValue> frame_conditions, vector_conditions;
    // Optional prepared, head-major cross K/V. The graph must not traverse producers.
    std::vector<DitContextKV> context_kv;
};

// Builds the complete estimator. No graph ownership, execution or checkpoint parsing.
class DitModule {
public:
    explicit DitModule(DitConfig config);
    const DitConfig & config() const noexcept { return config_; }
    core::TensorValue build(core::ModuleBuildContext & ctx,
        const DitForwardInputs & inputs, const DitWeights & weights, ggml_cgraph * graph = nullptr) const;
    core::TensorValue build_modulations(core::ModuleBuildContext & ctx,
        const DitForwardInputs & inputs, const DitWeights & weights) const;
    std::vector<DitContextKV> build_context(core::ModuleBuildContext & ctx,
        const core::TensorValue & context, const DitWeights & weights) const;

private:
    DitConfig config_;
};

} // namespace engine::modules
