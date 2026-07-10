#include "engine/models/index_tts2/semantic_encoder.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/weight_binding.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::index_tts2 {
namespace {

namespace binding = engine::modules::binding;
namespace modules = engine::modules;
using Clock = std::chrono::steady_clock;

constexpr int64_t kFeatureDim = 160;
constexpr int64_t kHidden = 1024;
constexpr int64_t kIntermediate = 4096;
constexpr int64_t kLayers = 24;
constexpr int64_t kSemanticOutputHiddenStateIndex = 17;
constexpr int64_t kHeads = 16;
constexpr int64_t kHeadDim = kHidden / kHeads;
constexpr int64_t kRelativePositions = 73;
constexpr int64_t kRelativeLeft = 64;
constexpr int64_t kRelativeRight = 8;
constexpr int64_t kConvKernel = 31;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

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

core::TensorValue reshape_heads(core::ModuleBuildContext & ctx, const core::TensorValue & input) {
    const auto contiguous = core::ensure_backend_addressable_layout(ctx, input);
    return core::reshape_tensor(ctx, contiguous, core::TensorShape::from_dims({1, input.shape.dims[1], kHeads, kHeadDim}));
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
    const core::TensorValue & distance_embedding) {
    const int64_t frames = q_heads.shape.dims[2];
    auto positions = modules::EmbeddingModule({kRelativePositions, kHeadDim}).build(ctx, distance_ids, distance_embedding);
    auto q_by_row = modules::TransposeModule({{2, 1, 0, 3}, 4}).build(ctx, q_heads);
    q_by_row = core::ensure_backend_addressable_layout(ctx, q_by_row);
    q_by_row = core::reshape_tensor(ctx, q_by_row, core::TensorShape::from_dims({frames, kHeads, kHeadDim}));
    auto positions_by_row = modules::TransposeModule({{0, 2, 1}, 3}).build(ctx, positions);
    positions_by_row = core::ensure_backend_addressable_layout(ctx, positions_by_row);
    auto by_row = modules::MatMulModule{}.build(ctx, q_by_row, positions_by_row);
    by_row = core::ensure_backend_addressable_layout(ctx, by_row);
    by_row = core::reshape_tensor(ctx, by_row, core::TensorShape::from_dims({frames, kHeads, 1, frames}));
    return core::ensure_backend_addressable_layout(ctx, modules::TransposeModule({{2, 1, 0, 3}, 4}).build(ctx, by_row));
}

core::TensorValue wav2vec2bert_attention(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & attention_mask,
    const core::TensorValue & distance_ids,
    const IndexTTS2Wav2Vec2BertAttentionWeights & weights) {
    auto q = modules::LinearModule({kHidden, kHidden, true, GGML_PREC_F32}).build(ctx, input, weights.q);
    auto k = modules::LinearModule({kHidden, kHidden, true, GGML_PREC_F32}).build(ctx, input, weights.k);
    auto v = modules::LinearModule({kHidden, kHidden, true, GGML_PREC_F32}).build(ctx, input, weights.v);
    q = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, reshape_heads(ctx, q));
    k = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, reshape_heads(ctx, k));
    v = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, reshape_heads(ctx, v));

    auto scores = modules::MatMulModule{}.build(
        ctx,
        q,
        modules::TransposeModule({{0, 1, 3, 2}, 4}).build(ctx, k));
    scores = scale_tensor(ctx, scores, 1.0F / std::sqrt(static_cast<float>(kHeadDim)));
    auto relative = build_relative_key_bias(ctx, q, distance_ids, weights.distance_embedding);
    relative = scale_tensor(ctx, relative, 1.0F / std::sqrt(static_cast<float>(kHeadDim)));
    scores = modules::AddModule{}.build(ctx, scores, relative);
    scores = modules::AddModule{}.build(ctx, scores, attention_mask);
    auto probs = core::wrap_tensor(ggml_soft_max(ctx.ggml, scores.tensor), scores.shape, GGML_TYPE_F32);
    auto out = modules::MatMulModule{}.build(ctx, probs, v);
    out = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, out);
    out = core::ensure_backend_addressable_layout(ctx, out);
    out = core::reshape_tensor(ctx, out, core::TensorShape::from_dims({1, input.shape.dims[1], kHidden}));
    return modules::LinearModule({kHidden, kHidden, true, GGML_PREC_F32}).build(ctx, out, weights.out);
}

core::TensorValue wav2vec2bert_feed_forward(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const modules::LinearWeights & in,
    const modules::LinearWeights & out) {
    auto hidden = modules::LinearModule({kHidden, kIntermediate, true, GGML_PREC_F32}).build(ctx, input, in);
    hidden = modules::SiluModule{}.build(ctx, hidden);
    return modules::LinearModule({kIntermediate, kHidden, true, GGML_PREC_F32}).build(ctx, hidden, out);
}

core::TensorValue wav2vec2bert_conv(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & keep_mask,
    const IndexTTS2Wav2Vec2BertConvWeights & weights) {
    auto hidden = modules::LayerNormModule({input.shape.last_dim(), 1.0e-5F, true, true}).build(ctx, input, weights.layer_norm);
    hidden = apply_keep_mask(ctx, hidden, keep_mask);
    hidden = modules::TransposeModule({{0, 2, 1}, 3}).build(ctx, hidden);
    hidden = modules::Conv1dModule({kHidden, 2 * kHidden, 1, 1, 0, 1, false}).build(ctx, hidden, weights.pointwise_in);
    auto gate = modules::SliceModule({1, 0, kHidden}).build(ctx, hidden);
    auto value = modules::SliceModule({1, kHidden, kHidden}).build(ctx, hidden);
    value = modules::SigmoidModule{}.build(ctx, value);
    hidden = modules::MulModule{}.build(ctx, gate, value);

    auto first = modules::SliceModule({2, 0, 1}).build(ctx, hidden);
    auto left_pad = modules::RepeatModule({core::TensorShape::from_dims({1, kHidden, kConvKernel - 1})}).build(ctx, first);
    left_pad = scale_tensor(ctx, left_pad, 0.0F);
    hidden = modules::ConcatModule({2}).build(ctx, left_pad, hidden);
    hidden = modules::DepthwiseConv1dModule({kHidden, kConvKernel, 1, 0, 1, false}).build(ctx, hidden, weights.depthwise);
    hidden = modules::TransposeModule({{0, 2, 1}, 3}).build(ctx, hidden);
    hidden = modules::LayerNormModule({hidden.shape.last_dim(), 1.0e-5F, true, true}).build(ctx, hidden, weights.depthwise_layer_norm);
    hidden = modules::TransposeModule({{0, 2, 1}, 3}).build(ctx, hidden);
    hidden = modules::SiluModule{}.build(ctx, hidden);
    hidden = modules::Conv1dModule({kHidden, kHidden, 1, 1, 0, 1, false}).build(ctx, hidden, weights.pointwise_out);
    return modules::TransposeModule({{0, 2, 1}, 3}).build(ctx, hidden);
}

core::TensorValue wav2vec2bert_layer(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & keep_mask,
    const core::TensorValue & attention_mask,
    const core::TensorValue & distance_ids,
    const IndexTTS2Wav2Vec2BertLayerWeights & weights) {
    auto hidden = modules::LayerNormModule({input.shape.last_dim(), 1.0e-5F, true, true}).build(ctx, input, weights.ffn1_norm);
    hidden = wav2vec2bert_feed_forward(ctx, hidden, weights.ffn1_in, weights.ffn1_out);
    hidden = add_scaled_residual(ctx, input, hidden, 0.5F);

    auto attn = modules::LayerNormModule({hidden.shape.last_dim(), 1.0e-5F, true, true}).build(ctx, hidden, weights.self_attn_norm);
    attn = wav2vec2bert_attention(ctx, attn, attention_mask, distance_ids, weights.self_attn);
    hidden = modules::AddModule{}.build(ctx, hidden, attn);

    auto conv = wav2vec2bert_conv(ctx, hidden, keep_mask, weights.conv);
    hidden = modules::AddModule{}.build(ctx, hidden, conv);

    auto ffn2 = modules::LayerNormModule({hidden.shape.last_dim(), 1.0e-5F, true, true}).build(ctx, hidden, weights.ffn2_norm);
    ffn2 = wav2vec2bert_feed_forward(ctx, ffn2, weights.ffn2_in, weights.ffn2_out);
    hidden = add_scaled_residual(ctx, hidden, ffn2, 0.5F);
    return modules::LayerNormModule({hidden.shape.last_dim(), 1.0e-5F, true, true}).build(ctx, hidden, weights.final_norm);
}

IndexTTS2Wav2Vec2BertAttentionWeights load_attention(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    engine::assets::TensorStorageType storage_type) {
    IndexTTS2Wav2Vec2BertAttentionWeights weights;
    weights.q = binding::linear_from_source(store, source, prefix + ".linear_q", storage_type, kHidden, kHidden, true);
    weights.k = binding::linear_from_source(store, source, prefix + ".linear_k", storage_type, kHidden, kHidden, true);
    weights.v = binding::linear_from_source(store, source, prefix + ".linear_v", storage_type, kHidden, kHidden, true);
    weights.out = binding::linear_from_source(store, source, prefix + ".linear_out", storage_type, kHidden, kHidden, true);
    weights.distance_embedding = store.load_tensor(
        source,
        prefix + ".distance_embedding.weight",
        storage_type,
        {kRelativePositions, kHeadDim});
    return weights;
}

IndexTTS2Wav2Vec2BertConvWeights load_conv_module(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    engine::assets::TensorStorageType storage_type) {
    IndexTTS2Wav2Vec2BertConvWeights weights;
    weights.layer_norm = binding::norm_from_source(store, source, prefix + ".layer_norm", kHidden);
    weights.pointwise_in = binding::conv1d_from_source(
        store,
        source,
        prefix + ".pointwise_conv1",
        storage_type,
        2 * kHidden,
        kHidden,
        1,
        false);
    weights.depthwise = binding::depthwise_conv1d_from_source(
        store,
        source,
        prefix + ".depthwise_conv",
        storage_type,
        kHidden,
        kConvKernel,
        false);
    weights.depthwise_layer_norm = binding::norm_from_source(store, source, prefix + ".depthwise_layer_norm", kHidden);
    weights.pointwise_out = binding::conv1d_from_source(
        store,
        source,
        prefix + ".pointwise_conv2",
        storage_type,
        kHidden,
        kHidden,
        1,
        false);
    return weights;
}

IndexTTS2Wav2Vec2BertLayerWeights load_layer(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    int64_t layer_index,
    engine::assets::TensorStorageType matmul_storage_type,
    engine::assets::TensorStorageType conv_storage_type) {
    const std::string prefix = "encoder.layers." + std::to_string(layer_index);
    IndexTTS2Wav2Vec2BertLayerWeights layer;
    layer.ffn1_norm = binding::norm_from_source(store, source, prefix + ".ffn1_layer_norm", kHidden);
    layer.ffn1_in = binding::linear_from_source(
        store,
        source,
        prefix + ".ffn1.intermediate_dense",
        matmul_storage_type,
        kIntermediate,
        kHidden,
        true);
    layer.ffn1_out = binding::linear_from_source(
        store,
        source,
        prefix + ".ffn1.output_dense",
        matmul_storage_type,
        kHidden,
        kIntermediate,
        true);
    layer.self_attn_norm = binding::norm_from_source(store, source, prefix + ".self_attn_layer_norm", kHidden);
    layer.self_attn = load_attention(store, source, prefix + ".self_attn", matmul_storage_type);
    layer.conv = load_conv_module(store, source, prefix + ".conv_module", conv_storage_type);
    layer.ffn2_norm = binding::norm_from_source(store, source, prefix + ".ffn2_layer_norm", kHidden);
    layer.ffn2_in = binding::linear_from_source(
        store,
        source,
        prefix + ".ffn2.intermediate_dense",
        matmul_storage_type,
        kIntermediate,
        kHidden,
        true);
    layer.ffn2_out = binding::linear_from_source(
        store,
        source,
        prefix + ".ffn2.output_dense",
        matmul_storage_type,
        kHidden,
        kIntermediate,
        true);
    layer.final_norm = binding::norm_from_source(store, source, prefix + ".final_layer_norm", kHidden);
    return layer;
}

std::vector<float> wav2vec2bert_std(const engine::assets::TensorSource & source) {
    const auto var = source.require_f32("var", {kHidden});
    std::vector<float> stddev(static_cast<size_t>(kHidden), 0.0F);
    for (int64_t i = 0; i < kHidden; ++i) {
        stddev[static_cast<size_t>(i)] = std::sqrt(var[static_cast<size_t>(i)]);
    }
    return stddev;
}

std::vector<int32_t> make_distance_ids(int64_t frames) {
    std::vector<int32_t> ids(static_cast<size_t>(frames * frames), 0);
    for (int64_t row = 0; row < frames; ++row) {
        for (int64_t col = 0; col < frames; ++col) {
            const int64_t distance = std::max<int64_t>(-kRelativeLeft, std::min<int64_t>(kRelativeRight, col - row));
            ids[static_cast<size_t>(row * frames + col)] = static_cast<int32_t>(distance + kRelativeLeft);
        }
    }
    return ids;
}

std::vector<float> make_attention_mask(const std::vector<int32_t> & keep_mask, int64_t frames) {
    if (static_cast<int64_t>(keep_mask.size()) != frames) {
        throw std::runtime_error("IndexTTS2 Wav2Vec2-BERT attention mask length mismatch");
    }
    std::vector<float> mask(static_cast<size_t>(kHeads * frames * frames), 0.0F);
    const float hidden = std::numeric_limits<float>::lowest();
    for (int64_t head = 0; head < kHeads; ++head) {
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

class IndexTTS2Wav2Vec2BertRuntime::Graph {
public:
    Graph(
        engine::core::ExecutionContext & execution,
        std::shared_ptr<const IndexTTS2Wav2Vec2BertWeights> weights,
        int64_t frames,
        size_t graph_arena_bytes)
        : execution_(execution),
          weights_(std::move(weights)),
          frames_(frames) {
        if (frames_ <= 0) {
            throw std::runtime_error("IndexTTS2 Wav2Vec2-BERT graph requires positive frame count");
        }
        if (weights_ == nullptr) {
            throw std::runtime_error("IndexTTS2 Wav2Vec2-BERT graph requires weights");
        }

        const auto build_start = Clock::now();
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize IndexTTS2 Wav2Vec2-BERT graph context");
        }
        ggml_init_params input_params{64ull * 1024ull * 1024ull, nullptr, true};
        input_ctx_.reset(ggml_init(input_params));
        if (input_ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize IndexTTS2 Wav2Vec2-BERT input context");
        }
        core::ModuleBuildContext ctx{ctx_.get(), "index_tts2.wav2vec2bert", execution_.backend_type()};
        core::ModuleBuildContext input_ctx{
            input_ctx_.get(),
            "index_tts2.wav2vec2bert.inputs",
            execution_.backend_type()};

        features_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, frames_, kFeatureDim})).tensor;
        keep_mask_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({frames_})).tensor;
        attention_mask_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, kHeads, frames_, frames_})).tensor;
        distance_ids_ = core::make_tensor(input_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({frames_, frames_})).tensor;
        ggml_set_input(features_);
        ggml_set_input(keep_mask_);
        ggml_set_input(attention_mask_);
        ggml_set_input(distance_ids_);

        auto x = core::wrap_tensor(features_, core::TensorShape::from_dims({1, frames_, kFeatureDim}), GGML_TYPE_F32);
        auto keep = core::wrap_tensor(keep_mask_, core::TensorShape::from_dims({frames_}), GGML_TYPE_F32);
        auto attn_mask = core::wrap_tensor(attention_mask_, core::TensorShape::from_dims({1, kHeads, frames_, frames_}), GGML_TYPE_F32);
        auto distances = core::wrap_tensor(distance_ids_, core::TensorShape::from_dims({frames_, frames_}), GGML_TYPE_I32);

        x = modules::LayerNormModule({x.shape.last_dim(), 1.0e-5F, true, true}).build(ctx, x, weights_->feature_norm);
        x = modules::LinearModule({kFeatureDim, kHidden, true, GGML_PREC_F32}).build(ctx, x, weights_->feature_projection);
        x = apply_keep_mask(ctx, x, keep);
        for (int64_t layer = 0; layer < kSemanticOutputHiddenStateIndex; ++layer) {
            x = wav2vec2bert_layer(ctx, x, keep, attn_mask, distances, weights_->layers[static_cast<size_t>(layer)]);
        }
        const auto mean = broadcast_vector(ctx, weights_->semantic_mean, x);
        const auto std = broadcast_vector(ctx, weights_->semantic_std, x);
        x = core::wrap_tensor(ggml_sub(ctx.ggml, x.tensor, mean.tensor), x.shape, GGML_TYPE_F32);
        x = core::wrap_tensor(ggml_div(ctx.ggml, x.tensor, std.tensor), x.shape, GGML_TYPE_F32);
        output_ = core::ensure_backend_addressable_layout(ctx, x).tensor;
        ggml_set_output(output_);

        const int64_t graph_nodes = std::max<int64_t>(
            65536,
            kSemanticOutputHiddenStateIndex * (frames_ * kHeads * 8 + frames_ * 16 + 1024) + 8192);
        graph_ = ggml_new_graph_custom(ctx_.get(), static_cast<size_t>(graph_nodes), false);
        ggml_build_forward_expand(graph_, output_);
        input_buffer_ = ggml_backend_alloc_ctx_tensors(input_ctx_.get(), execution_.backend());
        if (input_buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate IndexTTS2 Wav2Vec2-BERT input buffer");
        }
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend()));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            clear_graph();
            throw std::runtime_error("failed to allocate IndexTTS2 Wav2Vec2-BERT graph");
        }

        const auto distances_host = make_distance_ids(frames_);
        ggml_backend_tensor_set(distance_ids_, distances_host.data(), 0, distances_host.size() * sizeof(int32_t));
        debug::timing_log_scalar(
            "index_tts2.wav2vec2bert.graph.build_ms",
            engine::debug::elapsed_ms(build_start, Clock::now()));
        debug::trace_log_scalar("index_tts2.wav2vec2bert.frames", frames_);
    }

    ~Graph() {
        clear_graph();
    }

    int64_t frames() const noexcept {
        return frames_;
    }

    IndexTTS2SemanticEmbedding run(const IndexTTS2SemanticFeatureOutput & features) {
        if (features.frames <= 0 || features.frames > frames_ || features.dims != kFeatureDim) {
            throw std::runtime_error("IndexTTS2 Wav2Vec2-BERT feature shape does not match prepared graph");
        }
        if (static_cast<int64_t>(features.values.size()) != features.frames * kFeatureDim) {
            throw std::runtime_error("IndexTTS2 Wav2Vec2-BERT feature value count mismatch");
        }
        if (static_cast<int64_t>(features.attention_mask.size()) != features.frames) {
            throw std::runtime_error("IndexTTS2 Wav2Vec2-BERT feature attention mask count mismatch");
        }

        auto timing_start = Clock::now();
        std::vector<float> padded_features(static_cast<size_t>(frames_ * kFeatureDim), 0.0F);
        std::copy(features.values.begin(), features.values.end(), padded_features.begin());
        std::vector<int32_t> padded_mask(static_cast<size_t>(frames_), 0);
        std::copy(features.attention_mask.begin(), features.attention_mask.end(), padded_mask.begin());
        const auto keep = make_keep_mask_f32(padded_mask);
        const auto attention = make_attention_mask(padded_mask, frames_);
        ggml_backend_tensor_set(features_, padded_features.data(), 0, padded_features.size() * sizeof(float));
        ggml_backend_tensor_set(keep_mask_, keep.data(), 0, keep.size() * sizeof(float));
        ggml_backend_tensor_set(attention_mask_, attention.data(), 0, attention.size() * sizeof(float));
        debug::timing_log_scalar(
            "index_tts2.wav2vec2bert.input_upload_ms",
            engine::debug::elapsed_ms(timing_start, Clock::now()));

        core::set_backend_threads(execution_.backend(), execution_.config().threads);
        timing_start = Clock::now();
        const ggml_status status = engine::core::compute_backend_graph(execution_.backend(), graph_);
        ggml_backend_synchronize(execution_.backend());
        debug::timing_log_scalar(
            "index_tts2.wav2vec2bert.graph.compute_ms",
            engine::debug::elapsed_ms(timing_start, Clock::now()));
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("IndexTTS2 Wav2Vec2-BERT graph compute failed");
        }
        IndexTTS2SemanticEmbedding out;
        out.frames = features.frames;
        out.dims = kHidden;
        out.values.resize(static_cast<size_t>(features.frames * kHidden));
        timing_start = Clock::now();
        ggml_backend_tensor_get(output_, out.values.data(), 0, out.values.size() * sizeof(float));
        debug::timing_log_scalar(
            "index_tts2.wav2vec2bert.output_read_ms",
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
    std::shared_ptr<const IndexTTS2Wav2Vec2BertWeights> weights_;
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

std::shared_ptr<const IndexTTS2Wav2Vec2BertWeights> load_index_tts2_wav2vec2bert_weights(
    const IndexTTS2Assets & assets,
    ggml_backend_t backend,
    engine::core::BackendType backend_type,
    engine::assets::TensorStorageType matmul_storage_type,
    engine::assets::TensorStorageType conv_storage_type,
    size_t weight_context_bytes) {
    if (assets.wav2vec2bert_weights == nullptr || assets.wav2vec2bert_stats == nullptr) {
        throw std::runtime_error("IndexTTS2 Wav2Vec2-BERT requires model and stats tensor sources");
    }
    auto weights = std::make_shared<IndexTTS2Wav2Vec2BertWeights>();
    weights->store = std::make_shared<engine::core::BackendWeightStore>(
        backend,
        backend_type,
        "index_tts2.wav2vec2bert.weights",
        weight_context_bytes);

    const auto & source = *assets.wav2vec2bert_weights;
    weights->feature_norm = binding::norm_from_source(*weights->store, source, "feature_projection.layer_norm", kFeatureDim);
    weights->feature_projection = binding::linear_from_source(
        *weights->store,
        source,
        "feature_projection.projection",
        matmul_storage_type,
        kHidden,
        kFeatureDim,
        true);
    weights->layers.reserve(static_cast<size_t>(kLayers));
    for (int64_t layer_index = 0; layer_index < kLayers; ++layer_index) {
        weights->layers.push_back(load_layer(
            *weights->store,
            source,
            layer_index,
            matmul_storage_type,
            conv_storage_type));
    }
    weights->semantic_mean = weights->store->make_from_f32(
        engine::core::TensorShape::from_dims({kHidden}),
        matmul_storage_type,
        assets.wav2vec2bert_stats->require_f32("mean", {kHidden}));
    weights->semantic_std = weights->store->make_from_f32(
        engine::core::TensorShape::from_dims({kHidden}),
        matmul_storage_type,
        wav2vec2bert_std(*assets.wav2vec2bert_stats));

    weights->store->upload();
    assets.wav2vec2bert_weights->release_storage();
    assets.wav2vec2bert_stats->release_storage();
    return weights;
}

IndexTTS2Wav2Vec2BertRuntime::IndexTTS2Wav2Vec2BertRuntime(
    std::shared_ptr<const IndexTTS2Assets> assets,
    engine::core::ExecutionContext & execution,
    size_t graph_arena_bytes,
    size_t weight_context_bytes,
    engine::assets::TensorStorageType matmul_storage_type,
    engine::assets::TensorStorageType conv_storage_type)
    : assets_(std::move(assets)),
      execution_(&execution),
      graph_arena_bytes_(graph_arena_bytes) {
    if (assets_ == nullptr) {
        throw std::runtime_error("IndexTTS2 Wav2Vec2-BERT runtime requires assets");
    }
    if (graph_arena_bytes_ == 0) {
        throw std::runtime_error("IndexTTS2 Wav2Vec2-BERT graph arena must be non-zero");
    }
    weights_ = load_index_tts2_wav2vec2bert_weights(
        *assets_,
        execution.backend(),
        execution.backend_type(),
        matmul_storage_type,
        conv_storage_type,
        weight_context_bytes);
}

IndexTTS2Wav2Vec2BertRuntime::~IndexTTS2Wav2Vec2BertRuntime() = default;

void IndexTTS2Wav2Vec2BertRuntime::prepare(int64_t frames) {
    if (execution_ == nullptr) {
        throw std::runtime_error("IndexTTS2 Wav2Vec2-BERT runtime execution context is missing");
    }
    if (frames <= 0) {
        throw std::runtime_error("IndexTTS2 Wav2Vec2-BERT prepare requires positive frames");
    }
    if (graph_ != nullptr && graph_->frames() >= frames) {
        return;
    }
    graph_.reset();
    graph_ = std::make_unique<Graph>(*execution_, weights_, frames, graph_arena_bytes_);
}

IndexTTS2SemanticEmbedding IndexTTS2Wav2Vec2BertRuntime::encode(const IndexTTS2SemanticFeatureOutput & features) {
    if (graph_ == nullptr || graph_->frames() < features.frames) {
        throw std::runtime_error("IndexTTS2 Wav2Vec2-BERT graph was not prepared for this reference length");
    }
    return graph_->run(features);
}

void IndexTTS2Wav2Vec2BertRuntime::release_graph() {
    graph_.reset();
}

}  // namespace engine::models::index_tts2
