#include "engine/models/meanvc2/asr_encoder.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/module.h"
#include "engine/framework/modules/attention/types.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/streaming_conv_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::meanvc2 {
namespace {

constexpr int64_t kFbankFrames = 19;
constexpr int64_t kFbankDims = 80;
constexpr int64_t kConvChannels = 256;
constexpr int64_t kHidden = 256;
constexpr int64_t kIntermediate = 2048;
constexpr int64_t kLayers = 6;
constexpr int64_t kHeads = 4;
constexpr int64_t kHeadDim = 64;
constexpr int64_t kAttentionCacheFrames = 8;
constexpr int64_t kAttentionCacheWidth = 2 * kHeadDim;
constexpr int64_t kConvCacheFrames = 8;
constexpr int64_t kAsrPosMax = 5000;
constexpr int64_t kConvKernel = 3;
constexpr int64_t kConvStride = 2;
constexpr int64_t kEncoderConvKernel = 9;
constexpr int64_t kEncodedFrames = 4;
constexpr int64_t kEncodedFeatures = 19;
constexpr int64_t kEncoderKeyFrames = kAttentionCacheFrames + kEncodedFrames;
constexpr int64_t kSubsamplingFlat = kConvChannels * kEncodedFeatures;
constexpr size_t kGraphNodes = 262144;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

}  // namespace

struct MeanVC2AsrEncoderWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    modules::Conv2dWeights conv0;
    modules::Conv2dWeights conv1;
    modules::LinearWeights out;
    struct Layer {
        modules::NormWeights norm_ff_macaron;
        modules::LinearWeights ff_macaron_w1;
        modules::LinearWeights ff_macaron_w2;
        modules::NormWeights norm_mha;
        modules::RelativeAttentionWeights self_attn;
        modules::NormWeights norm_conv;
        modules::Conv1dWeights conv_pointwise1;
        modules::DepthwiseConv1dWeights conv_depthwise;
        modules::NormWeights conv_norm;
        modules::Conv1dWeights conv_pointwise2;
        modules::NormWeights norm_ff;
        modules::LinearWeights ff_w1;
        modules::LinearWeights ff_w2;
        modules::NormWeights norm_final;
    };
    std::array<Layer, kLayers> layers;
    modules::NormWeights after_norm;
};

namespace {

std::shared_ptr<const MeanVC2AsrEncoderWeights> load_asr_encoder_weights(
    ggml_backend_t backend,
    core::BackendType backend_type,
    const assets::TensorSource & source,
    size_t weight_context_bytes,
    assets::TensorStorageType storage_type) {
    auto weights = std::make_shared<MeanVC2AsrEncoderWeights>();
    weights->store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "meanvc2.asr.weights",
        weight_context_bytes);
    weights->conv0 = modules::binding::conv2d_from_source(
        *weights->store,
        source,
        "encoder.embed.conv.0",
        storage_type,
        kConvChannels,
        1,
        kConvKernel,
        kConvKernel,
        true);
    weights->conv1 = modules::binding::conv2d_from_source(
        *weights->store,
        source,
        "encoder.embed.conv.2",
        storage_type,
        kConvChannels,
        kConvChannels,
        kConvKernel,
        kConvKernel,
        true);
    weights->out = modules::binding::linear_from_source(
        *weights->store,
        source,
        "encoder.embed.out.0",
        storage_type,
        kHidden,
        kSubsamplingFlat,
        true);
    for (int64_t layer_index = 0; layer_index < kLayers; ++layer_index) {
        const std::string prefix = "encoder.encoders." + std::to_string(layer_index);
        auto & layer = weights->layers[static_cast<size_t>(layer_index)];
        layer.norm_ff_macaron = modules::binding::norm_from_source(*weights->store, source, prefix + ".norm_ff_macaron", kHidden);
        layer.ff_macaron_w1 = modules::binding::linear_from_source(
            *weights->store,
            source,
            prefix + ".feed_forward_macaron.w_1",
            storage_type,
            kIntermediate,
            kHidden,
            true);
        layer.ff_macaron_w2 = modules::binding::linear_from_source(
            *weights->store,
            source,
            prefix + ".feed_forward_macaron.w_2",
            storage_type,
            kHidden,
            kIntermediate,
            true);
        layer.norm_mha = modules::binding::norm_from_source(*weights->store, source, prefix + ".norm_mha", kHidden);
        layer.self_attn.attention.q_weight = weights->store->load_tensor(
            source,
            prefix + ".self_attn.linear_q.weight",
            storage_type,
            {kHidden, kHidden});
        layer.self_attn.attention.q_bias = weights->store->load_f32_tensor(source, prefix + ".self_attn.linear_q.bias", {kHidden});
        layer.self_attn.attention.k_weight = weights->store->load_tensor(
            source,
            prefix + ".self_attn.linear_k.weight",
            storage_type,
            {kHidden, kHidden});
        layer.self_attn.attention.k_bias = weights->store->load_f32_tensor(source, prefix + ".self_attn.linear_k.bias", {kHidden});
        layer.self_attn.attention.v_weight = weights->store->load_tensor(
            source,
            prefix + ".self_attn.linear_v.weight",
            storage_type,
            {kHidden, kHidden});
        layer.self_attn.attention.v_bias = weights->store->load_f32_tensor(source, prefix + ".self_attn.linear_v.bias", {kHidden});
        layer.self_attn.attention.out_weight = weights->store->load_tensor(
            source,
            prefix + ".self_attn.linear_out.weight",
            storage_type,
            {kHidden, kHidden});
        layer.self_attn.attention.out_bias = weights->store->load_f32_tensor(source, prefix + ".self_attn.linear_out.bias", {kHidden});
        layer.self_attn.pos_weight = weights->store->load_tensor(
            source,
            prefix + ".self_attn.linear_pos.weight",
            storage_type,
            {kHidden, kHidden});
        layer.self_attn.pos_bias_u = weights->store->load_f32_tensor(source, prefix + ".self_attn.pos_bias_u", {kHeads, kHeadDim});
        layer.self_attn.pos_bias_v = weights->store->load_f32_tensor(source, prefix + ".self_attn.pos_bias_v", {kHeads, kHeadDim});
        layer.norm_conv = modules::binding::norm_from_source(*weights->store, source, prefix + ".norm_conv", kHidden);
        layer.conv_pointwise1 = modules::binding::conv1d_from_source(
            *weights->store,
            source,
            prefix + ".conv_module.streaming_conv.pointwise_conv1",
            storage_type,
            2 * kHidden,
            kHidden,
            1,
            true);
        layer.conv_depthwise = modules::binding::depthwise_conv1d_from_source(
            *weights->store,
            source,
            prefix + ".conv_module.streaming_conv.depthwise_conv",
            storage_type,
            kHidden,
            kEncoderConvKernel,
            true);
        layer.conv_norm = modules::binding::norm_from_source(
            *weights->store,
            source,
            prefix + ".conv_module.streaming_conv.norm",
            kHidden);
        layer.conv_pointwise2 = modules::binding::conv1d_from_source(
            *weights->store,
            source,
            prefix + ".conv_module.streaming_conv.pointwise_conv2",
            storage_type,
            kHidden,
            kHidden,
            1,
            true);
        layer.norm_ff = modules::binding::norm_from_source(*weights->store, source, prefix + ".norm_ff", kHidden);
        layer.ff_w1 = modules::binding::linear_from_source(
            *weights->store,
            source,
            prefix + ".feed_forward.w_1",
            storage_type,
            kIntermediate,
            kHidden,
            true);
        layer.ff_w2 = modules::binding::linear_from_source(
            *weights->store,
            source,
            prefix + ".feed_forward.w_2",
            storage_type,
            kHidden,
            kIntermediate,
            true);
        layer.norm_final = modules::binding::norm_from_source(*weights->store, source, prefix + ".norm_final", kHidden);
    }
    weights->after_norm = modules::binding::norm_from_source(*weights->store, source, "encoder.after_norm", kHidden);
    weights->store->upload();
    return weights;
}

core::TensorValue build_subsampling(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & fbank,
    const MeanVC2AsrEncoderWeights & weights) {
    auto x = core::reshape_tensor(
        ctx,
        fbank,
        core::TensorShape::from_dims({1, 1, kFbankFrames, kFbankDims}));
    x = modules::Conv2dModule({
        1,
        kConvChannels,
        kConvKernel,
        kConvKernel,
        static_cast<int>(kConvStride),
        static_cast<int>(kConvStride),
        0,
        0,
        1,
        1,
        true,
    }).build(ctx, x, weights.conv0);
    x = modules::ReluModule().build(ctx, x);
    x = modules::Conv2dModule({
        kConvChannels,
        kConvChannels,
        kConvKernel,
        kConvKernel,
        static_cast<int>(kConvStride),
        static_cast<int>(kConvStride),
        0,
        0,
        1,
        1,
        true,
    }).build(ctx, x, weights.conv1);
    x = modules::ReluModule().build(ctx, x);
    x = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, x);
    ggml_tensor * contiguous = ggml_cont(ctx.ggml, x.tensor);
    x = core::wrap_tensor(contiguous, x.shape, contiguous->type);
    x = core::reshape_tensor(ctx, x, core::TensorShape::from_dims({1, kEncodedFrames, kSubsamplingFlat}));
    x = modules::LinearModule({kSubsamplingFlat, kHidden, true}).build(ctx, x, weights.out);
    ggml_tensor * scaled = ggml_scale(ctx.ggml, x.tensor, std::sqrt(static_cast<float>(kHidden)));
    return core::wrap_tensor(scaled, x.shape, scaled->type);
}

std::vector<float> make_wenet_positional_encoding(int64_t offset) {
    if (offset < 0 || offset + kEncoderKeyFrames > kAsrPosMax) {
        throw std::runtime_error("MeanVC2 ASR position offset is outside positional table");
    }
    std::vector<float> values(static_cast<size_t>(kEncoderKeyFrames * kHidden), 0.0F);
    constexpr long double kBase = 10000.0L;
    for (int64_t pos = 0; pos < kEncoderKeyFrames; ++pos) {
        const long double position = static_cast<long double>(offset + pos);
        for (int64_t i = 0; i < kHidden / 2; ++i) {
            const long double exponent = static_cast<long double>(2 * i) / static_cast<long double>(kHidden);
            const long double phase = position / std::pow(kBase, exponent);
            const size_t base = static_cast<size_t>(pos * kHidden + 2 * i);
            values[base] = static_cast<float>(std::sin(phase));
            values[base + 1] = static_cast<float>(std::cos(phase));
        }
    }
    return values;
}

core::TensorValue heads_bthd_to_bhtd(core::ModuleBuildContext & ctx, const core::TensorValue & input) {
    auto x = core::reshape_tensor(ctx, input, core::TensorShape::from_dims({1, kEncodedFrames, kHeads, kHeadDim}));
    return modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, x);
}

core::TensorValue bhtd_to_bthd_hidden(core::ModuleBuildContext & ctx, const core::TensorValue & input) {
    auto x = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, input);
    x = core::ensure_backend_addressable_layout(ctx, x);
    return core::reshape_tensor(ctx, x, core::TensorShape::from_dims({1, kEncodedFrames, kHidden}));
}

core::TensorValue add_head_bias(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & bias) {
    auto bias_view = core::reshape_tensor(ctx, bias, core::TensorShape::from_dims({1, kHeads, 1, kHeadDim}));
    auto repeated = modules::RepeatModule({input.shape}).build(ctx, bias_view);
    ggml_tensor * added = ggml_add(ctx.ggml, input.tensor, repeated.tensor);
    return core::wrap_tensor(added, input.shape, added->type);
}

core::TensorValue build_wenet_attention(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & pos_emb,
    const core::TensorValue & att_cache,
    const MeanVC2AsrEncoderWeights::Layer & weights,
    core::TensorValue & next_cache) {
    auto q = modules::LinearModule({kHidden, kHidden, true})
                 .build(ctx, input, {weights.self_attn.attention.q_weight, weights.self_attn.attention.q_bias});
    auto k = modules::LinearModule({kHidden, kHidden, true})
                 .build(ctx, input, {weights.self_attn.attention.k_weight, weights.self_attn.attention.k_bias});
    auto v = modules::LinearModule({kHidden, kHidden, true})
                 .build(ctx, input, {weights.self_attn.attention.v_weight, weights.self_attn.attention.v_bias});
    q = heads_bthd_to_bhtd(ctx, q);
    k = heads_bthd_to_bhtd(ctx, k);
    v = heads_bthd_to_bhtd(ctx, v);

    auto key_cache = modules::SliceModule({3, 0, kHeadDim}).build(ctx, att_cache);
    auto value_cache = modules::SliceModule({3, kHeadDim, kHeadDim}).build(ctx, att_cache);
    auto all_k = modules::ConcatModule({2}).build(ctx, key_cache, k);
    auto all_v = modules::ConcatModule({2}).build(ctx, value_cache, v);

    auto packed = modules::ConcatModule({3}).build(ctx, all_k, all_v);
    next_cache = modules::SliceModule({2, kEncodedFrames, kAttentionCacheFrames}).build(ctx, packed);
    next_cache = core::ensure_backend_addressable_layout(ctx, next_cache);

    auto p = modules::LinearModule({kHidden, kHidden, false})
                 .build(ctx, pos_emb, {weights.self_attn.pos_weight, std::nullopt});
    p = core::reshape_tensor(ctx, p, core::TensorShape::from_dims({1, kEncoderKeyFrames, kHeads, kHeadDim}));
    p = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, p);

    auto q_u = add_head_bias(ctx, q, weights.self_attn.pos_bias_u);
    auto q_v = add_head_bias(ctx, q, weights.self_attn.pos_bias_v);
    auto k_t = modules::TransposeModule({{0, 1, 3, 2}, 4}).build(ctx, all_k);
    auto p_t = modules::TransposeModule({{0, 1, 3, 2}, 4}).build(ctx, p);
    auto scores = modules::MatMulModule().build(ctx, q_u, k_t);
    auto pos_scores = modules::MatMulModule().build(ctx, q_v, p_t);
    ggml_tensor * added_scores = ggml_add(ctx.ggml, scores.tensor, pos_scores.tensor);
    scores = core::wrap_tensor(added_scores, scores.shape, added_scores->type);
    ggml_tensor * scaled_scores = ggml_scale(ctx.ggml, scores.tensor, 1.0F / std::sqrt(static_cast<float>(kHeadDim)));
    scores = core::wrap_tensor(scaled_scores, scores.shape, scaled_scores->type);
    ggml_tensor * softmax = ggml_soft_max(ctx.ggml, core::ensure_backend_addressable_layout(ctx, scores).tensor);
    auto attn = core::wrap_tensor(softmax, scores.shape, softmax->type);
    auto context = modules::MatMulModule().build(ctx, attn, all_v);
    context = bhtd_to_bthd_hidden(ctx, context);
    return modules::LinearModule({kHidden, kHidden, true})
        .build(ctx, context, {weights.self_attn.attention.out_weight, weights.self_attn.attention.out_bias});
}

core::TensorValue build_wenet_streaming_conv(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & conv_cache,
    const MeanVC2AsrEncoderWeights::Layer & weights,
    core::TensorValue & next_cache) {
    auto x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, input);
    auto conv_input = modules::ConcatModule({2}).build(ctx, conv_cache, x);
    next_cache = modules::SliceModule({2, kEncodedFrames, kConvCacheFrames}).build(ctx, conv_input);
    next_cache = core::ensure_backend_addressable_layout(ctx, next_cache);

    x = modules::Conv1dModule({kHidden, 2 * kHidden, 1, 1, 0, 1, true}).build(ctx, conv_input, weights.conv_pointwise1);
    x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
    x = modules::GLUModule().build(ctx, x);
    x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
    x = modules::DepthwiseConv1dModule({kHidden, kEncoderConvKernel, 1, 0, 1, true}).build(ctx, x, weights.conv_depthwise);
    x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
    x = modules::LayerNormModule({kHidden, 1.0e-5F, true, true}).build(ctx, x, weights.conv_norm);
    x = modules::SiluModule().build(ctx, x);
    x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
    x = modules::Conv1dModule({kHidden, kHidden, 1, 1, 0, 1, true}).build(ctx, x, weights.conv_pointwise2);
    return modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
}

core::TensorValue build_feed_forward(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const modules::LinearWeights & w1,
    const modules::LinearWeights & w2) {
    auto x = modules::LinearModule({kHidden, kIntermediate, true}).build(ctx, input, w1);
    x = modules::SiluModule().build(ctx, x);
    return modules::LinearModule({kIntermediate, kHidden, true}).build(ctx, x, w2);
}

core::TensorValue build_encoder_layer(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & pos_emb,
    const core::TensorValue & att_cache,
    const core::TensorValue & conv_cache,
    const MeanVC2AsrEncoderWeights::Layer & weights,
    core::TensorValue & next_att_cache,
    core::TensorValue & next_conv_cache) {
    auto x_norm = modules::LayerNormModule({kHidden, 1.0e-5F, true, true}).build(ctx, input, weights.norm_ff_macaron);
    auto ff = build_feed_forward(ctx, x_norm, weights.ff_macaron_w1, weights.ff_macaron_w2);
    ggml_tensor * macaron = ggml_add(ctx.ggml, input.tensor, ggml_scale(ctx.ggml, ff.tensor, 0.5F));
    auto x = core::wrap_tensor(macaron, input.shape, macaron->type);

    x_norm = modules::LayerNormModule({kHidden, 1.0e-5F, true, true}).build(ctx, x, weights.norm_mha);
    auto attn = build_wenet_attention(ctx, x_norm, pos_emb, att_cache, weights, next_att_cache);
    ggml_tensor * attn_residual = ggml_add(ctx.ggml, x.tensor, attn.tensor);
    x = core::wrap_tensor(attn_residual, x.shape, attn_residual->type);

    x_norm = modules::LayerNormModule({kHidden, 1.0e-5F, true, true}).build(ctx, x, weights.norm_conv);
    auto conv = build_wenet_streaming_conv(ctx, x_norm, conv_cache, weights, next_conv_cache);
    ggml_tensor * conv_residual = ggml_add(ctx.ggml, x.tensor, conv.tensor);
    x = core::wrap_tensor(conv_residual, x.shape, conv_residual->type);

    x_norm = modules::LayerNormModule({kHidden, 1.0e-5F, true, true}).build(ctx, x, weights.norm_ff);
    ff = build_feed_forward(ctx, x_norm, weights.ff_w1, weights.ff_w2);
    ggml_tensor * ff_residual = ggml_add(ctx.ggml, x.tensor, ggml_scale(ctx.ggml, ff.tensor, 0.5F));
    x = core::wrap_tensor(ff_residual, x.shape, ff_residual->type);
    return modules::LayerNormModule({kHidden, 1.0e-5F, true, true}).build(ctx, x, weights.norm_final);
}

}  // namespace

struct MeanVC2AsrEncoderGraph {
    MeanVC2AsrEncoderGraph(
        ggml_backend_t backend,
        core::BackendType backend_type,
        size_t graph_context_bytes,
        std::shared_ptr<const MeanVC2AsrEncoderWeights> weights)
        : backend(backend),
          weights(std::move(weights)) {
        if (backend == nullptr || this->weights == nullptr) {
            throw std::runtime_error("MeanVC2 ASR encoder graph requires backend and weights");
        }

        ggml_init_params params{graph_context_bytes, nullptr, true};
        ctx.reset(ggml_init(params));
        if (ctx == nullptr) {
            throw std::runtime_error("failed to initialize MeanVC2 ASR encoder graph context");
        }

        core::ModuleBuildContext build_ctx{ctx.get(), "meanvc2.asr", backend_type};
        input = core::make_tensor(
            build_ctx,
            GGML_TYPE_F32,
            core::TensorShape::from_dims({1, kFbankFrames, kFbankDims}));
        ggml_set_input(input.tensor);
        pos_emb = core::make_tensor(
            build_ctx,
            GGML_TYPE_F32,
            core::TensorShape::from_dims({1, kEncoderKeyFrames, kHidden}));
        ggml_set_input(pos_emb.tensor);
        for (int64_t layer_index = 0; layer_index < kLayers; ++layer_index) {
            attention_cache[static_cast<size_t>(layer_index)] = core::make_tensor(
                build_ctx,
                GGML_TYPE_F32,
                core::TensorShape::from_dims({1, kHeads, kAttentionCacheFrames, kAttentionCacheWidth}));
            conv_cache[static_cast<size_t>(layer_index)] = core::make_tensor(
                build_ctx,
                GGML_TYPE_F32,
                core::TensorShape::from_dims({1, kHidden, kConvCacheFrames}));
            ggml_set_input(attention_cache[static_cast<size_t>(layer_index)].tensor);
            ggml_set_input(conv_cache[static_cast<size_t>(layer_index)].tensor);
        }

        auto hidden = build_subsampling(build_ctx, input, *this->weights);
        for (int64_t layer_index = 0; layer_index < kLayers; ++layer_index) {
            hidden = build_encoder_layer(
                build_ctx,
                hidden,
                pos_emb,
                attention_cache[static_cast<size_t>(layer_index)],
                conv_cache[static_cast<size_t>(layer_index)],
                this->weights->layers[static_cast<size_t>(layer_index)],
                next_attention_cache[static_cast<size_t>(layer_index)],
                next_conv_cache[static_cast<size_t>(layer_index)]);
            ggml_set_output(next_attention_cache[static_cast<size_t>(layer_index)].tensor);
            ggml_set_output(next_conv_cache[static_cast<size_t>(layer_index)].tensor);
        }
        output = modules::LayerNormModule({kHidden, 1.0e-5F, true, true}).build(build_ctx, hidden, this->weights->after_norm);
        output = core::ensure_backend_addressable_layout(build_ctx, output);
        ggml_set_output(output.tensor);

        graph = ggml_new_graph_custom(ctx.get(), kGraphNodes, false);
        ggml_build_forward_expand(graph, output.tensor);
        for (int64_t layer_index = 0; layer_index < kLayers; ++layer_index) {
            ggml_build_forward_expand(graph, next_attention_cache[static_cast<size_t>(layer_index)].tensor);
            ggml_build_forward_expand(graph, next_conv_cache[static_cast<size_t>(layer_index)].tensor);
        }
        gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (gallocr == nullptr || !ggml_gallocr_reserve(gallocr, graph) || !ggml_gallocr_alloc_graph(gallocr, graph)) {
            throw std::runtime_error("failed to allocate MeanVC2 ASR encoder graph");
        }
    }

    ~MeanVC2AsrEncoderGraph() {
        if (backend != nullptr) {
            core::release_backend_graph_resources(backend, graph);
        }
        if (gallocr != nullptr) {
            ggml_gallocr_free(gallocr);
            gallocr = nullptr;
        }
    }

    std::vector<float> run(
        const MeanVC2FbankWindow & window,
        int64_t position_offset,
        std::vector<float> & attention_cache_values,
        std::vector<float> & conv_cache_values) {
        if (static_cast<int64_t>(window.values.size()) != kFbankFrames * kFbankDims) {
            throw std::runtime_error("MeanVC2 ASR fbank window shape mismatch");
        }
        if (static_cast<int64_t>(attention_cache_values.size()) !=
            kLayers * kHeads * kAttentionCacheFrames * kAttentionCacheWidth) {
            throw std::runtime_error("MeanVC2 ASR attention cache shape mismatch");
        }
        if (static_cast<int64_t>(conv_cache_values.size()) != kLayers * kHidden * kConvCacheFrames) {
            throw std::runtime_error("MeanVC2 ASR convolution cache shape mismatch");
        }
        core::write_tensor_float(input, window.values);
        core::write_tensor_float(pos_emb, make_wenet_positional_encoding(position_offset - kAttentionCacheFrames));
        const size_t attention_layer_values =
            static_cast<size_t>(kHeads * kAttentionCacheFrames * kAttentionCacheWidth);
        const size_t conv_layer_values = static_cast<size_t>(kHidden * kConvCacheFrames);
        for (int64_t layer_index = 0; layer_index < kLayers; ++layer_index) {
            const size_t layer = static_cast<size_t>(layer_index);
            core::write_tensor_float(
                attention_cache[layer],
                std::vector<float>(
                    attention_cache_values.begin() + static_cast<std::ptrdiff_t>(layer * attention_layer_values),
                    attention_cache_values.begin() + static_cast<std::ptrdiff_t>((layer + 1) * attention_layer_values)));
            core::write_tensor_float(
                conv_cache[layer],
                std::vector<float>(
                    conv_cache_values.begin() + static_cast<std::ptrdiff_t>(layer * conv_layer_values),
                    conv_cache_values.begin() + static_cast<std::ptrdiff_t>((layer + 1) * conv_layer_values)));
        }
        const ggml_status status = core::compute_backend_graph(backend, graph, nullptr, "MeanVC2 ASR encoder");
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("MeanVC2 ASR encoder graph compute failed");
        }
        for (int64_t layer_index = 0; layer_index < kLayers; ++layer_index) {
            const size_t layer = static_cast<size_t>(layer_index);
            std::vector<float> next_attention;
            std::vector<float> next_conv;
            core::read_tensor_float_into(next_attention_cache[layer].tensor, next_attention);
            core::read_tensor_float_into(next_conv_cache[layer].tensor, next_conv);
            std::copy(
                next_attention.begin(),
                next_attention.end(),
                attention_cache_values.begin() + static_cast<std::ptrdiff_t>(layer * attention_layer_values));
            std::copy(
                next_conv.begin(),
                next_conv.end(),
                conv_cache_values.begin() + static_cast<std::ptrdiff_t>(layer * conv_layer_values));
        }
        return core::read_tensor_float(output.tensor);
    }

    ggml_backend_t backend = nullptr;
    std::shared_ptr<const MeanVC2AsrEncoderWeights> weights;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx;
    core::TensorValue input;
    core::TensorValue pos_emb;
    std::array<core::TensorValue, kLayers> attention_cache;
    std::array<core::TensorValue, kLayers> conv_cache;
    std::array<core::TensorValue, kLayers> next_attention_cache;
    std::array<core::TensorValue, kLayers> next_conv_cache;
    core::TensorValue output;
    ggml_cgraph * graph = nullptr;
    ggml_gallocr_t gallocr = nullptr;
};

MeanVC2AsrEncoderRuntime::MeanVC2AsrEncoderRuntime(
    std::shared_ptr<const assets::TensorSource> source,
    core::ExecutionContext & execution_context,
    size_t weight_context_bytes,
    size_t graph_context_bytes,
    assets::TensorStorageType weight_storage_type)
    : execution_context_(execution_context),
      graph_context_bytes_(graph_context_bytes),
      source_(std::move(source)) {
    if (source_ == nullptr) {
        throw std::runtime_error("MeanVC2 ASR encoder requires tensor source");
    }
    weights_ = load_asr_encoder_weights(
        execution_context_.backend(),
        execution_context_.backend_type(),
        *source_,
        weight_context_bytes,
        weight_storage_type);
    source_->release_storage();
}

MeanVC2AsrEncoderRuntime::~MeanVC2AsrEncoderRuntime() = default;

void MeanVC2AsrEncoderRuntime::reset() {
    attention_cache_.assign(
        static_cast<size_t>(kLayers * kHeads * kAttentionCacheFrames * kAttentionCacheWidth),
        0.0F);
    conv_cache_.assign(static_cast<size_t>(kLayers * kHidden * kConvCacheFrames), 0.0F);
}

std::vector<float> MeanVC2AsrEncoderRuntime::encode_windows(
    const std::vector<MeanVC2FbankWindow> & windows) {
    if (windows.empty()) {
        return {};
    }
    if (attention_cache_.empty() || conv_cache_.empty()) {
        reset();
    }
    if (graph_ == nullptr) {
        graph_ = std::make_unique<MeanVC2AsrEncoderGraph>(
            execution_context_.backend(),
            execution_context_.backend_type(),
            graph_context_bytes_,
            weights_);
    }
    std::vector<float> out;
    out.reserve(windows.size() * static_cast<size_t>(kEncodedFrames * kHidden));
    for (const auto & window : windows) {
        const auto encoded = graph_->run(window, window.offset, attention_cache_, conv_cache_);
        out.insert(out.end(), encoded.begin(), encoded.end());
    }
    return out;
}

}  // namespace engine::models::meanvc2
