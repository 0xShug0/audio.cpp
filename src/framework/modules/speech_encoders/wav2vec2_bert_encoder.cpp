#include "engine/framework/modules/speech_encoders/wav2vec2_bert_encoder.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/weight_binding.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <ggml-alloc.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::modules {
namespace {

namespace binding = engine::modules::binding;
namespace modules = engine::modules;
using Clock = std::chrono::steady_clock;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

void validate_config(const Wav2Vec2BertEncoderConfig & config) {
    if (config.feature_dim <= 0 || config.hidden_size <= 0 || config.intermediate_size <= 0 ||
        config.num_hidden_layers <= 0 || config.output_hidden_layer < 0 || config.num_attention_heads <= 0 ||
        config.relative_positions <= 0 || config.conv_kernel <= 0) {
        throw std::runtime_error("Wav2Vec2-BERT config contains invalid dimensions");
    }
    if (config.output_hidden_layer > config.num_hidden_layers) {
        throw std::runtime_error("Wav2Vec2-BERT output layer cannot exceed hidden layer count");
    }
    if (config.hidden_size % config.num_attention_heads != 0) {
        throw std::runtime_error("Wav2Vec2-BERT hidden size must be divisible by attention heads");
    }
    if (config.relative_left < 0 || config.relative_right < 0 ||
        config.relative_left + config.relative_right + 1 > config.relative_positions) {
        throw std::runtime_error("Wav2Vec2-BERT relative position config is inconsistent");
    }
}

std::string join_name(const std::string & lhs, const std::string & rhs) {
    if (lhs.empty()) {
        return rhs;
    }
    if (rhs.empty()) {
        return lhs;
    }
    return lhs + "." + rhs;
}

std::string indexed_prefix(const std::string & prefix, int64_t index) {
    return join_name(prefix, std::to_string(index));
}

core::TensorValue scale_tensor(core::ModuleBuildContext & ctx, const core::TensorValue & input, float scale) {
    return core::wrap_tensor(ggml_scale(ctx.ggml, input.tensor, scale), input.shape, GGML_TYPE_F32);
}

core::TensorValue add_scaled_residual(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & residual,
    const core::TensorValue & update,
    float update_scale) {
    return modules::AddModule{}.build(ctx, residual, scale_tensor(ctx, update, update_scale));
}

core::TensorValue reshape_heads(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const Wav2Vec2BertEncoderConfig & config) {
    const auto contiguous = core::ensure_backend_addressable_layout(ctx, input);
    return core::reshape_tensor(
        ctx,
        contiguous,
        core::TensorShape::from_dims({1, input.shape.dims[1], config.num_attention_heads, config.hidden_size / config.num_attention_heads}));
}

core::TensorValue repeat_last_dim_mask(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & mask,
    const core::TensorValue & like) {
    auto mask_view = core::reshape_tensor(ctx, mask, core::TensorShape::from_dims({1, mask.shape.dims[0], 1}));
    return modules::RepeatModule({like.shape}).build(ctx, mask_view);
}

core::TensorValue apply_keep_mask(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & keep_mask) {
    return modules::MulModule{}.build(ctx, input, repeat_last_dim_mask(ctx, keep_mask, input));
}

core::TensorValue broadcast_vector(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & vector,
    const core::TensorValue & like) {
    auto view = core::reshape_tensor(ctx, vector, core::TensorShape::from_dims({1, 1, vector.shape.dims[0]}));
    return modules::RepeatModule({like.shape}).build(ctx, view);
}

core::TensorValue build_relative_key_bias(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & q_heads,
    const core::TensorValue & distance_ids,
    const core::TensorValue & distance_embedding,
    const Wav2Vec2BertEncoderConfig & config) {
    const int64_t frames = q_heads.shape.dims[2];
    const int64_t head_dim = config.hidden_size / config.num_attention_heads;
    auto positions = modules::EmbeddingModule({config.relative_positions, head_dim}).build(ctx, distance_ids, distance_embedding);
    auto q_by_row = modules::TransposeModule({{2, 1, 0, 3}, 4}).build(ctx, q_heads);
    q_by_row = core::ensure_backend_addressable_layout(ctx, q_by_row);
    q_by_row = core::reshape_tensor(ctx, q_by_row, core::TensorShape::from_dims({frames, config.num_attention_heads, head_dim}));
    auto positions_by_row = modules::TransposeModule({{0, 2, 1}, 3}).build(ctx, positions);
    positions_by_row = core::ensure_backend_addressable_layout(ctx, positions_by_row);
    auto by_row = modules::MatMulModule{}.build(ctx, q_by_row, positions_by_row);
    by_row = core::ensure_backend_addressable_layout(ctx, by_row);
    by_row = core::reshape_tensor(ctx, by_row, core::TensorShape::from_dims({frames, config.num_attention_heads, 1, frames}));
    return core::ensure_backend_addressable_layout(ctx, modules::TransposeModule({{2, 1, 0, 3}, 4}).build(ctx, by_row));
}

core::TensorValue wav2vec2bert_attention(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & attention_mask,
    const core::TensorValue & distance_ids,
    const Wav2Vec2BertAttentionWeights & weights,
    const Wav2Vec2BertEncoderConfig & config) {
    const int64_t head_dim = config.hidden_size / config.num_attention_heads;
    auto q = modules::LinearModule({config.hidden_size, config.hidden_size, true, GGML_PREC_F32}).build(ctx, input, weights.q);
    auto k = modules::LinearModule({config.hidden_size, config.hidden_size, true, GGML_PREC_F32}).build(ctx, input, weights.k);
    auto v = modules::LinearModule({config.hidden_size, config.hidden_size, true, GGML_PREC_F32}).build(ctx, input, weights.v);
    q = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, reshape_heads(ctx, q, config));
    k = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, reshape_heads(ctx, k, config));
    v = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, reshape_heads(ctx, v, config));

    auto scores = modules::MatMulModule{}.build(
        ctx,
        q,
        modules::TransposeModule({{0, 1, 3, 2}, 4}).build(ctx, k));
    scores = scale_tensor(ctx, scores, 1.0F / std::sqrt(static_cast<float>(head_dim)));
    auto relative = build_relative_key_bias(ctx, q, distance_ids, weights.distance_embedding, config);
    relative = scale_tensor(ctx, relative, 1.0F / std::sqrt(static_cast<float>(head_dim)));
    scores = modules::AddModule{}.build(ctx, scores, relative);
    scores = modules::AddModule{}.build(ctx, scores, attention_mask);
    auto probs = core::wrap_tensor(ggml_soft_max(ctx.ggml, scores.tensor), scores.shape, GGML_TYPE_F32);
    auto out = modules::MatMulModule{}.build(ctx, probs, v);
    out = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, out);
    out = core::ensure_backend_addressable_layout(ctx, out);
    out = core::reshape_tensor(ctx, out, core::TensorShape::from_dims({1, input.shape.dims[1], config.hidden_size}));
    return modules::LinearModule({config.hidden_size, config.hidden_size, true, GGML_PREC_F32}).build(ctx, out, weights.out);
}

core::TensorValue wav2vec2bert_feed_forward(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const modules::LinearWeights & in,
    const modules::LinearWeights & out,
    const Wav2Vec2BertEncoderConfig & config) {
    auto hidden = modules::LinearModule({config.hidden_size, config.intermediate_size, true, GGML_PREC_F32}).build(ctx, input, in);
    hidden = modules::SiluModule{}.build(ctx, hidden);
    return modules::LinearModule({config.intermediate_size, config.hidden_size, true, GGML_PREC_F32}).build(ctx, hidden, out);
}

core::TensorValue wav2vec2bert_conv(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & keep_mask,
    const Wav2Vec2BertConvWeights & weights,
    const Wav2Vec2BertEncoderConfig & config) {
    auto hidden = modules::LayerNormModule({input.shape.last_dim(), config.layer_norm_eps, true, true}).build(ctx, input, weights.layer_norm);
    hidden = apply_keep_mask(ctx, hidden, keep_mask);
    hidden = modules::TransposeModule({{0, 2, 1}, 3}).build(ctx, hidden);
    hidden = modules::Conv1dModule({config.hidden_size, 2 * config.hidden_size, 1, 1, 0, 1, false}).build(ctx, hidden, weights.pointwise_in);
    auto gate = modules::SliceModule({1, 0, config.hidden_size}).build(ctx, hidden);
    auto value = modules::SliceModule({1, config.hidden_size, config.hidden_size}).build(ctx, hidden);
    value = modules::SigmoidModule{}.build(ctx, value);
    hidden = modules::MulModule{}.build(ctx, gate, value);

    auto first = modules::SliceModule({2, 0, 1}).build(ctx, hidden);
    auto left_pad = modules::RepeatModule({core::TensorShape::from_dims({1, config.hidden_size, config.conv_kernel - 1})}).build(ctx, first);
    left_pad = scale_tensor(ctx, left_pad, 0.0F);
    hidden = modules::ConcatModule({2}).build(ctx, left_pad, hidden);
    hidden = modules::DepthwiseConv1dModule({config.hidden_size, config.conv_kernel, 1, 0, 1, false}).build(ctx, hidden, weights.depthwise);
    hidden = modules::TransposeModule({{0, 2, 1}, 3}).build(ctx, hidden);
    hidden = modules::LayerNormModule({hidden.shape.last_dim(), config.layer_norm_eps, true, true}).build(ctx, hidden, weights.depthwise_layer_norm);
    hidden = modules::TransposeModule({{0, 2, 1}, 3}).build(ctx, hidden);
    hidden = modules::SiluModule{}.build(ctx, hidden);
    hidden = modules::Conv1dModule({config.hidden_size, config.hidden_size, 1, 1, 0, 1, false}).build(ctx, hidden, weights.pointwise_out);
    return modules::TransposeModule({{0, 2, 1}, 3}).build(ctx, hidden);
}

core::TensorValue wav2vec2bert_layer(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & keep_mask,
    const core::TensorValue & attention_mask,
    const core::TensorValue & distance_ids,
    const Wav2Vec2BertLayerWeights & weights,
    const Wav2Vec2BertEncoderConfig & config) {
    auto hidden = modules::LayerNormModule({input.shape.last_dim(), config.layer_norm_eps, true, true}).build(ctx, input, weights.ffn1_norm);
    hidden = wav2vec2bert_feed_forward(ctx, hidden, weights.ffn1_in, weights.ffn1_out, config);
    hidden = add_scaled_residual(ctx, input, hidden, 0.5F);

    auto attn = modules::LayerNormModule({hidden.shape.last_dim(), config.layer_norm_eps, true, true}).build(ctx, hidden, weights.self_attn_norm);
    attn = wav2vec2bert_attention(ctx, attn, attention_mask, distance_ids, weights.self_attn, config);
    hidden = modules::AddModule{}.build(ctx, hidden, attn);

    auto conv = wav2vec2bert_conv(ctx, hidden, keep_mask, weights.conv, config);
    hidden = modules::AddModule{}.build(ctx, hidden, conv);

    auto ffn2 = modules::LayerNormModule({hidden.shape.last_dim(), config.layer_norm_eps, true, true}).build(ctx, hidden, weights.ffn2_norm);
    ffn2 = wav2vec2bert_feed_forward(ctx, ffn2, weights.ffn2_in, weights.ffn2_out, config);
    hidden = add_scaled_residual(ctx, hidden, ffn2, 0.5F);
    return modules::LayerNormModule({hidden.shape.last_dim(), config.layer_norm_eps, true, true}).build(ctx, hidden, weights.final_norm);
}

Wav2Vec2BertAttentionWeights load_attention(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    engine::assets::TensorStorageType storage_type,
    const Wav2Vec2BertEncoderConfig & config,
    const Wav2Vec2BertAttentionWeightNames & names) {
    const int64_t head_dim = config.hidden_size / config.num_attention_heads;
    Wav2Vec2BertAttentionWeights weights;
    weights.q = binding::linear_from_source(store, source, join_name(prefix, names.q), storage_type, config.hidden_size, config.hidden_size, true);
    weights.k = binding::linear_from_source(store, source, join_name(prefix, names.k), storage_type, config.hidden_size, config.hidden_size, true);
    weights.v = binding::linear_from_source(store, source, join_name(prefix, names.v), storage_type, config.hidden_size, config.hidden_size, true);
    weights.out = binding::linear_from_source(store, source, join_name(prefix, names.out), storage_type, config.hidden_size, config.hidden_size, true);
    weights.distance_embedding = store.load_tensor(
        source,
        join_name(join_name(prefix, names.distance_embedding), "weight"),
        storage_type,
        {config.relative_positions, head_dim});
    return weights;
}

Wav2Vec2BertConvWeights load_conv_module(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    engine::assets::TensorStorageType storage_type,
    const Wav2Vec2BertEncoderConfig & config,
    const Wav2Vec2BertConvWeightNames & names) {
    Wav2Vec2BertConvWeights weights;
    weights.layer_norm = binding::norm_from_source(store, source, join_name(prefix, names.layer_norm), config.hidden_size);
    weights.pointwise_in = binding::conv1d_from_source(
        store,
        source,
        join_name(prefix, names.pointwise_in),
        storage_type,
        2 * config.hidden_size,
        config.hidden_size,
        1,
        false);
    weights.depthwise = binding::depthwise_conv1d_from_source(
        store,
        source,
        join_name(prefix, names.depthwise),
        storage_type,
        config.hidden_size,
        config.conv_kernel,
        false);
    weights.depthwise_layer_norm = binding::norm_from_source(store, source, join_name(prefix, names.depthwise_layer_norm), config.hidden_size);
    weights.pointwise_out = binding::conv1d_from_source(
        store,
        source,
        join_name(prefix, names.pointwise_out),
        storage_type,
        config.hidden_size,
        config.hidden_size,
        1,
        false);
    return weights;
}

Wav2Vec2BertLayerWeights load_layer(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    int64_t layer_index,
    engine::assets::TensorStorageType matmul_storage_type,
    engine::assets::TensorStorageType conv_storage_type,
    const Wav2Vec2BertEncoderConfig & config,
    const Wav2Vec2BertEncoderWeightBinding & binding_names) {
    const std::string prefix = indexed_prefix(binding_names.encoder_layers, layer_index);
    const auto & names = binding_names.layer;
    Wav2Vec2BertLayerWeights layer;
    layer.ffn1_norm = binding::norm_from_source(store, source, join_name(prefix, names.ffn1_norm), config.hidden_size);
    layer.ffn1_in = binding::linear_from_source(
        store,
        source,
        join_name(prefix, names.ffn1_in),
        matmul_storage_type,
        config.intermediate_size,
        config.hidden_size,
        true);
    layer.ffn1_out = binding::linear_from_source(
        store,
        source,
        join_name(prefix, names.ffn1_out),
        matmul_storage_type,
        config.hidden_size,
        config.intermediate_size,
        true);
    layer.self_attn_norm = binding::norm_from_source(store, source, join_name(prefix, names.self_attn_norm), config.hidden_size);
    layer.self_attn = load_attention(store, source, join_name(prefix, names.self_attn), matmul_storage_type, config, names.attention);
    layer.conv = load_conv_module(store, source, join_name(prefix, names.conv), conv_storage_type, config, names.conv_module);
    layer.ffn2_norm = binding::norm_from_source(store, source, join_name(prefix, names.ffn2_norm), config.hidden_size);
    layer.ffn2_in = binding::linear_from_source(
        store,
        source,
        join_name(prefix, names.ffn2_in),
        matmul_storage_type,
        config.intermediate_size,
        config.hidden_size,
        true);
    layer.ffn2_out = binding::linear_from_source(
        store,
        source,
        join_name(prefix, names.ffn2_out),
        matmul_storage_type,
        config.hidden_size,
        config.intermediate_size,
        true);
    layer.final_norm = binding::norm_from_source(store, source, join_name(prefix, names.final_norm), config.hidden_size);
    return layer;
}

std::vector<float> wav2vec2bert_std(
    const engine::assets::TensorSource & source,
    const Wav2Vec2BertEncoderConfig & config,
    const Wav2Vec2BertEncoderWeightBinding & binding_names) {
    const auto var = source.require_f32(binding_names.stats_variance, {config.hidden_size});
    std::vector<float> stddev(static_cast<size_t>(config.hidden_size), 0.0F);
    for (int64_t i = 0; i < config.hidden_size; ++i) {
        stddev[static_cast<size_t>(i)] = std::sqrt(var[static_cast<size_t>(i)]);
    }
    return stddev;
}

std::vector<int32_t> make_distance_ids(int64_t frames, const Wav2Vec2BertEncoderConfig & config) {
    std::vector<int32_t> ids(static_cast<size_t>(frames * frames), 0);
    for (int64_t row = 0; row < frames; ++row) {
        for (int64_t col = 0; col < frames; ++col) {
            const int64_t distance = std::max<int64_t>(-config.relative_left, std::min<int64_t>(config.relative_right, col - row));
            ids[static_cast<size_t>(row * frames + col)] = static_cast<int32_t>(distance + config.relative_left);
        }
    }
    return ids;
}

std::vector<float> make_attention_mask(
    const std::vector<int32_t> & keep_mask,
    int64_t frames,
    const Wav2Vec2BertEncoderConfig & config) {
    if (static_cast<int64_t>(keep_mask.size()) != frames) {
        throw std::runtime_error("Wav2Vec2-BERT attention mask length mismatch");
    }
    std::vector<float> mask(static_cast<size_t>(config.num_attention_heads * frames * frames), 0.0F);
    const float hidden = std::numeric_limits<float>::lowest();
    for (int64_t head = 0; head < config.num_attention_heads; ++head) {
        for (int64_t row = 0; row < frames; ++row) {
            for (int64_t col = 0; col < frames; ++col) {
                if (keep_mask[static_cast<size_t>(col)] == 0) {
                    mask[static_cast<size_t>((head * frames + row) * frames + col)] = hidden;
                }
            }
        }
    }
    return mask;
}

std::vector<float> make_keep_mask_f32(const std::vector<int32_t> & keep_mask) {
    std::vector<float> out(keep_mask.size(), 0.0F);
    for (size_t i = 0; i < keep_mask.size(); ++i) {
        out[i] = keep_mask[i] == 0 ? 0.0F : 1.0F;
    }
    return out;
}

}  // namespace

class Wav2Vec2BertEncoderComponent::Graph {
public:
    Graph(
        engine::core::ExecutionContext & execution,
        std::shared_ptr<const Wav2Vec2BertEncoderWeights> weights,
        int64_t frames,
        size_t graph_arena_bytes)
        : execution_(execution),
          weights_(std::move(weights)),
          frames_(frames) {
        if (frames_ <= 0) {
            throw std::runtime_error("Wav2Vec2-BERT graph requires positive frame count");
        }
        if (weights_ == nullptr) {
            throw std::runtime_error("Wav2Vec2-BERT graph requires weights");
        }

        const auto build_start = Clock::now();
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize Wav2Vec2-BERT graph context");
        }
        ggml_init_params input_params{64ull * 1024ull * 1024ull, nullptr, true};
        input_ctx_.reset(ggml_init(input_params));
        if (input_ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize Wav2Vec2-BERT input context");
        }
        core::ModuleBuildContext ctx{ctx_.get(), "framework.wav2vec2bert", execution_.backend_type()};
        core::ModuleBuildContext input_ctx{
            input_ctx_.get(),
            "framework.wav2vec2bert.inputs",
            execution_.backend_type()};
        const auto & config = weights_->config;

        features_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, frames_, config.feature_dim})).tensor;
        keep_mask_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({frames_})).tensor;
        attention_mask_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, config.num_attention_heads, frames_, frames_})).tensor;
        distance_ids_ = core::make_tensor(input_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({frames_, frames_})).tensor;
        ggml_set_input(features_);
        ggml_set_input(keep_mask_);
        ggml_set_input(attention_mask_);
        ggml_set_input(distance_ids_);

        auto x = core::wrap_tensor(features_, core::TensorShape::from_dims({1, frames_, config.feature_dim}), GGML_TYPE_F32);
        auto keep = core::wrap_tensor(keep_mask_, core::TensorShape::from_dims({frames_}), GGML_TYPE_F32);
        auto attn_mask = core::wrap_tensor(attention_mask_, core::TensorShape::from_dims({1, config.num_attention_heads, frames_, frames_}), GGML_TYPE_F32);
        auto distances = core::wrap_tensor(distance_ids_, core::TensorShape::from_dims({frames_, frames_}), GGML_TYPE_I32);

        x = modules::LayerNormModule({x.shape.last_dim(), config.layer_norm_eps, true, true}).build(ctx, x, weights_->feature_norm);
        x = modules::LinearModule({config.feature_dim, config.hidden_size, true, GGML_PREC_F32}).build(ctx, x, weights_->feature_projection);
        x = apply_keep_mask(ctx, x, keep);
        for (int64_t layer = 0; layer < config.output_hidden_layer; ++layer) {
            x = wav2vec2bert_layer(ctx, x, keep, attn_mask, distances, weights_->layers[static_cast<size_t>(layer)], config);
        }
        const auto mean = broadcast_vector(ctx, weights_->semantic_mean, x);
        const auto std = broadcast_vector(ctx, weights_->semantic_std, x);
        x = core::wrap_tensor(ggml_sub(ctx.ggml, x.tensor, mean.tensor), x.shape, GGML_TYPE_F32);
        x = core::wrap_tensor(ggml_div(ctx.ggml, x.tensor, std.tensor), x.shape, GGML_TYPE_F32);
        output_ = core::ensure_backend_addressable_layout(ctx, x).tensor;
        ggml_set_output(output_);

        const int64_t graph_nodes = std::max<int64_t>(
            65536,
            config.output_hidden_layer * (frames_ * config.num_attention_heads * 8 + frames_ * 16 + 1024) + 8192);
        graph_ = ggml_new_graph_custom(ctx_.get(), static_cast<size_t>(graph_nodes), false);
        ggml_build_forward_expand(graph_, output_);
        input_buffer_ = ggml_backend_alloc_ctx_tensors(input_ctx_.get(), execution_.backend());
        if (input_buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate Wav2Vec2-BERT input buffer");
        }
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend()));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            clear_graph();
            throw std::runtime_error("failed to allocate Wav2Vec2-BERT graph");
        }

        const auto distances_host = make_distance_ids(frames_, config);
        ggml_backend_tensor_set(distance_ids_, distances_host.data(), 0, distances_host.size() * sizeof(int32_t));
        debug::timing_log_scalar(
            "framework.wav2vec2bert.graph.build_ms",
            engine::debug::elapsed_ms(build_start, Clock::now()));
        debug::trace_log_scalar("framework.wav2vec2bert.frames", frames_);
    }

    ~Graph() {
        clear_graph();
    }

    int64_t frames() const noexcept {
        return frames_;
    }

    Wav2Vec2BertEncoderOutput run(const Wav2Vec2BertEncoderInput & features) {
        const auto & config = weights_->config;
        if (features.frames <= 0 || features.frames > frames_ || features.dims != config.feature_dim) {
            throw std::runtime_error("Wav2Vec2-BERT feature shape does not match prepared graph");
        }
        if (static_cast<int64_t>(features.values.size()) != features.frames * config.feature_dim) {
            throw std::runtime_error("Wav2Vec2-BERT feature value count mismatch");
        }
        if (static_cast<int64_t>(features.attention_mask.size()) != features.frames) {
            throw std::runtime_error("Wav2Vec2-BERT feature attention mask count mismatch");
        }

        auto timing_start = Clock::now();
        std::vector<float> padded_features(static_cast<size_t>(frames_ * config.feature_dim), 0.0F);
        std::copy(features.values.begin(), features.values.end(), padded_features.begin());
        std::vector<int32_t> padded_mask(static_cast<size_t>(frames_), 0);
        std::copy(features.attention_mask.begin(), features.attention_mask.end(), padded_mask.begin());
        const auto keep = make_keep_mask_f32(padded_mask);
        const auto attention = make_attention_mask(padded_mask, frames_, config);
        debug::timing_log_scalar(
            "framework.wav2vec2bert.host_input_prepare_ms",
            engine::debug::elapsed_ms(timing_start, Clock::now()));
        timing_start = Clock::now();
        ggml_backend_tensor_set(features_, padded_features.data(), 0, padded_features.size() * sizeof(float));
        ggml_backend_tensor_set(keep_mask_, keep.data(), 0, keep.size() * sizeof(float));
        ggml_backend_tensor_set(attention_mask_, attention.data(), 0, attention.size() * sizeof(float));
        debug::timing_log_scalar(
            "framework.wav2vec2bert.input_upload_ms",
            engine::debug::elapsed_ms(timing_start, Clock::now()));

        core::set_backend_threads(execution_.backend(), execution_.config().threads);
        timing_start = Clock::now();
        const ggml_status status = engine::core::compute_backend_graph(execution_.backend(), graph_);
        ggml_backend_synchronize(execution_.backend());
        debug::timing_log_scalar(
            "framework.wav2vec2bert.graph.compute_ms",
            engine::debug::elapsed_ms(timing_start, Clock::now()));
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Wav2Vec2-BERT graph compute failed");
        }
        Wav2Vec2BertEncoderOutput out;
        out.frames = features.frames;
        out.dims = config.hidden_size;
        out.values.resize(static_cast<size_t>(features.frames * config.hidden_size));
        timing_start = Clock::now();
        ggml_backend_tensor_get(output_, out.values.data(), 0, out.values.size() * sizeof(float));
        debug::timing_log_scalar(
            "framework.wav2vec2bert.output_read_ms",
            engine::debug::elapsed_ms(timing_start, Clock::now()));
        return out;
    }

private:
    void clear_graph() {
        if (graph_ != nullptr) {
            engine::core::release_backend_graph_resources(execution_.backend(), graph_);
            graph_ = nullptr;
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
            gallocr_ = nullptr;
        }
        if (input_buffer_ != nullptr) {
            ggml_backend_buffer_free(input_buffer_);
            input_buffer_ = nullptr;
        }
    }

    engine::core::ExecutionContext & execution_;
    std::shared_ptr<const Wav2Vec2BertEncoderWeights> weights_;
    int64_t frames_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> input_ctx_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * features_ = nullptr;
    ggml_tensor * keep_mask_ = nullptr;
    ggml_tensor * attention_mask_ = nullptr;
    ggml_tensor * distance_ids_ = nullptr;
    ggml_tensor * output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    ggml_backend_buffer_t input_buffer_ = nullptr;
};

std::shared_ptr<const Wav2Vec2BertEncoderWeights> load_wav2vec2bert_weights(
    std::shared_ptr<const engine::assets::TensorSource> model_source,
    std::shared_ptr<const engine::assets::TensorSource> stats_source,
    ggml_backend_t backend,
    engine::core::BackendType backend_type,
    Wav2Vec2BertEncoderConfig config,
    const Wav2Vec2BertEncoderWeightBinding & weight_binding,
    size_t weight_context_bytes) {
    if (model_source == nullptr || stats_source == nullptr) {
        throw std::runtime_error("Wav2Vec2-BERT requires model and stats tensor sources");
    }
    validate_config(config);
    auto weights = std::make_shared<Wav2Vec2BertEncoderWeights>();
    weights->config = std::move(config);
    weights->store = std::make_shared<engine::core::BackendWeightStore>(
        backend,
        backend_type,
        "framework.wav2vec2bert.weights",
        weight_context_bytes);

    const auto & source = *model_source;
    const auto & config_ref = weights->config;
    weights->feature_norm = binding::norm_from_source(*weights->store, source, weight_binding.feature_norm, config_ref.feature_dim);
    weights->feature_projection = binding::linear_from_source(
        *weights->store,
        source,
        weight_binding.feature_projection,
        weight_binding.matmul_storage_type,
        config_ref.hidden_size,
        config_ref.feature_dim,
        true);
    weights->layers.reserve(static_cast<size_t>(config_ref.num_hidden_layers));
    for (int64_t layer_index = 0; layer_index < config_ref.num_hidden_layers; ++layer_index) {
        weights->layers.push_back(load_layer(
            *weights->store,
            source,
            layer_index,
            weight_binding.matmul_storage_type,
            weight_binding.conv_storage_type,
            config_ref,
            weight_binding));
    }
    weights->semantic_mean = weights->store->make_from_f32(
        engine::core::TensorShape::from_dims({config_ref.hidden_size}),
        weight_binding.stats_storage_type,
        stats_source->require_f32(weight_binding.stats_mean, {config_ref.hidden_size}));
    weights->semantic_std = weights->store->make_from_f32(
        engine::core::TensorShape::from_dims({config_ref.hidden_size}),
        weight_binding.stats_storage_type,
        wav2vec2bert_std(*stats_source, config_ref, weight_binding));

    weights->store->upload();
    model_source->release_storage();
    stats_source->release_storage();
    return weights;
}

Wav2Vec2BertEncoderComponent Wav2Vec2BertEncoderComponent::load_from_tensor_source(
    std::shared_ptr<const engine::assets::TensorSource> model_source,
    std::shared_ptr<const engine::assets::TensorSource> stats_source,
    engine::core::ExecutionContext & execution,
    size_t graph_arena_bytes,
    size_t weight_context_bytes,
    Wav2Vec2BertEncoderConfig config) {
    return load_from_tensor_source(
        std::move(model_source),
        std::move(stats_source),
        execution,
        graph_arena_bytes,
        weight_context_bytes,
        std::move(config),
        Wav2Vec2BertEncoderWeightBinding{});
}

Wav2Vec2BertEncoderComponent Wav2Vec2BertEncoderComponent::load_from_tensor_source(
    std::shared_ptr<const engine::assets::TensorSource> model_source,
    std::shared_ptr<const engine::assets::TensorSource> stats_source,
    engine::core::ExecutionContext & execution,
    size_t graph_arena_bytes,
    size_t weight_context_bytes,
    Wav2Vec2BertEncoderConfig config,
    Wav2Vec2BertEncoderWeightBinding weight_binding) {
    auto weights = load_wav2vec2bert_weights(
        std::move(model_source),
        std::move(stats_source),
        execution.backend(),
        execution.backend_type(),
        std::move(config),
        weight_binding,
        weight_context_bytes);
    return Wav2Vec2BertEncoderComponent(std::move(weights), execution, graph_arena_bytes);
}

Wav2Vec2BertEncoderComponent::Wav2Vec2BertEncoderComponent() = default;

Wav2Vec2BertEncoderComponent::Wav2Vec2BertEncoderComponent(
    std::shared_ptr<const Wav2Vec2BertEncoderWeights> weights,
    engine::core::ExecutionContext & execution,
    size_t graph_arena_bytes)
    : execution_(&execution),
      graph_arena_bytes_(graph_arena_bytes),
      weights_(std::move(weights)) {
    if (weights_ == nullptr) {
        throw std::runtime_error("Wav2Vec2-BERT component requires weights");
    }
    if (graph_arena_bytes_ == 0) {
        throw std::runtime_error("Wav2Vec2-BERT graph arena must be non-zero");
    }
}

Wav2Vec2BertEncoderComponent::~Wav2Vec2BertEncoderComponent() = default;

Wav2Vec2BertEncoderComponent::Wav2Vec2BertEncoderComponent(Wav2Vec2BertEncoderComponent &&) noexcept = default;

Wav2Vec2BertEncoderComponent & Wav2Vec2BertEncoderComponent::operator=(Wav2Vec2BertEncoderComponent &&) noexcept = default;

void Wav2Vec2BertEncoderComponent::prepare(int64_t frames) {
    if (execution_ == nullptr) {
        throw std::runtime_error("Wav2Vec2-BERT runtime execution context is missing");
    }
    if (frames <= 0) {
        throw std::runtime_error("Wav2Vec2-BERT prepare requires positive frames");
    }
    if (graph_ != nullptr && graph_->frames() == frames) {
        return;
    }
    graph_.reset();
    graph_ = std::make_unique<Graph>(*execution_, weights_, frames, graph_arena_bytes_);
}

Wav2Vec2BertEncoderOutput Wav2Vec2BertEncoderComponent::encode(const Wav2Vec2BertEncoderInput & features) {
    if (graph_ == nullptr || graph_->frames() < features.frames) {
        throw std::runtime_error("Wav2Vec2-BERT graph was not prepared for this reference length");
    }
    return graph_->run(features);
}

void Wav2Vec2BertEncoderComponent::release_runtime_graph() {
    graph_.reset();
}

}  // namespace engine::modules
