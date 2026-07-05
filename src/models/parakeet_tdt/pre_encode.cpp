#include "engine/models/parakeet_tdt/pre_encode.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/core/module.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/structural_modules.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace engine::models::parakeet_tdt {

namespace {

constexpr size_t kPreEncodeGraphBytes = 64 * 1024 * 1024;
constexpr size_t kPreEncodeGraphNodes = 512;

int64_t conv_output_dim(int64_t input, int64_t kernel, int64_t stride, int64_t padding) {
    return (input + 2 * padding - kernel) / stride + 1;
}

int64_t conv_valid_length(int64_t valid, int64_t kernel, int64_t stride, int64_t padding) {
    if (valid <= 0) {
        return 0;
    }
    const int64_t numerator = valid + 2 * padding - kernel;
    if (numerator < 0) {
        return 0;
    }
    return numerator / stride + 1;
}

engine::core::TensorValue add_channel_bias_4d(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    const engine::core::TensorValue & bias,
    int64_t channels) {
    engine::core::validate_shape(bias, engine::core::TensorShape::from_dims({channels}), "bias");
    const auto bias_view = engine::core::reshape_tensor(
        ctx,
        bias,
        engine::core::TensorShape::from_dims({1, channels, 1, 1}));
    const auto expanded = engine::core::wrap_tensor(
        ggml_repeat(ctx.ggml, bias_view.tensor, input.tensor),
        input.shape,
        GGML_TYPE_F32);
    return engine::core::wrap_tensor(ggml_add(ctx.ggml, input.tensor, expanded.tensor), input.shape, GGML_TYPE_F32);
}

engine::core::TensorValue apply_explicit_time_mask_4d(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    const engine::core::TensorValue & mask) {
    engine::core::validate_shape(mask, engine::core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[2]}), "mask");
    if (mask.type != GGML_TYPE_I32) {
        throw std::runtime_error("pre-encode mask must be GGML_TYPE_I32");
    }
    const auto mask_f32 = engine::core::wrap_tensor(ggml_cast(ctx.ggml, mask.tensor, GGML_TYPE_F32), mask.shape, GGML_TYPE_F32);
    const auto mask_4d = engine::core::reshape_tensor(
        ctx,
        mask_f32,
        engine::core::TensorShape::from_dims({input.shape.dims[0], 1, input.shape.dims[2], 1}));
    const auto mask_broadcast = engine::core::wrap_tensor(ggml_repeat(ctx.ggml, mask_4d.tensor, input.tensor), input.shape, GGML_TYPE_F32);
    return engine::core::wrap_tensor(ggml_mul(ctx.ggml, input.tensor, mask_broadcast.tensor), input.shape, GGML_TYPE_F32);
}

engine::core::TensorValue build_depthwise_conv2d(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    const engine::core::TensorValue & weight,
    const engine::core::TensorValue * bias,
    int64_t channels,
    int kernel,
    int stride,
    int padding) {
    engine::core::validate_shape(
        weight,
        engine::core::TensorShape::from_dims({channels, 1, kernel, kernel}),
        "depthwise_weight");
    const int64_t out_h = conv_output_dim(input.shape.dims[2], kernel, stride, padding);
    const int64_t out_w = conv_output_dim(input.shape.dims[3], kernel, stride, padding);
    auto output = engine::core::wrap_tensor(
        ggml_conv_2d_dw_direct(
            ctx.ggml,
            weight.tensor,
            input.tensor,
            stride,
            stride,
            padding,
            padding,
            1,
            1),
        engine::core::TensorShape::from_dims({input.shape.dims[0], channels, out_h, out_w}),
        GGML_TYPE_F32);
    if (bias != nullptr) {
        output = add_channel_bias_4d(ctx, output, *bias, channels);
    }
    return output;
}

void fill_mask(std::vector<int32_t> & mask, int64_t frames, int64_t valid_frames) {
    mask.assign(static_cast<size_t>(frames), 0);
    const int64_t clamped = std::clamp<int64_t>(valid_frames, 0, frames);
    for (int64_t t = 0; t < clamped; ++t) {
        mask[static_cast<size_t>(t)] = 1;
    }
}

}  // namespace

struct ParakeetPreEncodeLinearGraph {
    int64_t input_frames = -1;
    int64_t input_features = -1;
    int64_t stage1_frames = 0;
    int64_t stage1_features = 0;
    int64_t stage2_frames = 0;
    int64_t stage2_features = 0;
    int64_t stage3_frames = 0;
    int64_t stage3_features = 0;
    ggml_backend_t backend = nullptr;
    ggml_context * ggml = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_cgraph * graph = nullptr;
    engine::core::TensorValue input;
    engine::core::TensorValue mask1;
    engine::core::TensorValue mask2;
    engine::core::TensorValue mask3;
    engine::core::TensorValue output;

    ~ParakeetPreEncodeLinearGraph() {
        if (buffer != nullptr) {
            ggml_backend_buffer_free(buffer);
            buffer = nullptr;
        }
        if (ggml != nullptr) {
            ggml_free(ggml);
            ggml = nullptr;
        }
    }
};

ParakeetPreEncodeScratch::ParakeetPreEncodeScratch() = default;
ParakeetPreEncodeScratch::~ParakeetPreEncodeScratch() = default;
ParakeetPreEncodeScratch::ParakeetPreEncodeScratch(ParakeetPreEncodeScratch && other) noexcept = default;
ParakeetPreEncodeScratch & ParakeetPreEncodeScratch::operator=(ParakeetPreEncodeScratch && other) noexcept = default;

void compute_parakeet_pre_encode(
    const ParakeetFrontendBatch & frontend,
    const ParakeetAssets & assets,
    const ParakeetTDTWeights & weights,
    engine::core::ExecutionContext & execution_context,
    ParakeetPreEncodeBatch & output,
    ParakeetPreEncodeScratch & scratch) {
    if (frontend.batch != 1) {
        throw std::runtime_error("Parakeet pre-encode currently supports batch=1 only");
    }

    const int64_t input_frames = frontend.frames;
    const int64_t input_features = frontend.feature_dim;
    const int64_t channels = assets.model_config.encoder.subsampling_conv_channels;
    const int64_t kernel = assets.model_config.encoder.subsampling_conv_kernel_size;
    const int64_t stride = assets.model_config.encoder.subsampling_conv_stride;
    const int64_t padding = (kernel - 1) / 2;
    const int64_t hidden = assets.model_config.encoder.hidden_size;

    const int64_t stage1_frames = conv_output_dim(input_frames, kernel, stride, padding);
    const int64_t stage1_features = conv_output_dim(input_features, kernel, stride, padding);
    const int64_t stage2_frames = conv_output_dim(stage1_frames, kernel, stride, padding);
    const int64_t stage2_features = conv_output_dim(stage1_features, kernel, stride, padding);
    const int64_t stage3_frames = conv_output_dim(stage2_frames, kernel, stride, padding);
    const int64_t stage3_features = conv_output_dim(stage2_features, kernel, stride, padding);

    const bool graph_mismatch =
        !scratch.graph ||
        scratch.graph->backend != execution_context.backend() ||
        scratch.graph->input_frames != input_frames ||
        scratch.graph->input_features != input_features;

    if (graph_mismatch) {
        auto graph = std::make_unique<ParakeetPreEncodeLinearGraph>();
        graph->input_frames = input_frames;
        graph->input_features = input_features;
        graph->stage1_frames = stage1_frames;
        graph->stage1_features = stage1_features;
        graph->stage2_frames = stage2_frames;
        graph->stage2_features = stage2_features;
        graph->stage3_frames = stage3_frames;
        graph->stage3_features = stage3_features;
        graph->backend = execution_context.backend();

        ggml_init_params params = {};
        params.mem_size = kPreEncodeGraphBytes;
        params.mem_buffer = nullptr;
        params.no_alloc = true;
        graph->ggml = ggml_init(params);
        if (graph->ggml == nullptr) {
            throw std::runtime_error("Failed to initialize Parakeet pre-encode ggml context");
        }

        engine::core::ModuleBuildContext ctx = {};
        ctx.ggml = graph->ggml;
        ctx.module_instance_name = "parakeet_tdt_pre_encode";

        graph->input = engine::core::make_tensor(ctx, GGML_TYPE_F32, engine::core::TensorShape::from_dims({1, 1, input_frames, input_features}));
        graph->mask1 = engine::core::make_tensor(ctx, GGML_TYPE_I32, engine::core::TensorShape::from_dims({1, stage1_frames}));
        graph->mask2 = engine::core::make_tensor(ctx, GGML_TYPE_I32, engine::core::TensorShape::from_dims({1, stage2_frames}));
        graph->mask3 = engine::core::make_tensor(ctx, GGML_TYPE_I32, engine::core::TensorShape::from_dims({1, stage3_frames}));

        const auto & subsampling = weights.encoder.subsampling;

        auto x = engine::modules::Conv2dModule({
            1,
            channels,
            kernel,
            kernel,
            static_cast<int>(stride),
            static_cast<int>(stride),
            static_cast<int>(padding),
            static_cast<int>(padding),
            1,
            1,
            true,
        }).build(ctx, graph->input, subsampling.conv0);
        x = engine::modules::ReluModule().build(ctx, x);
        x = apply_explicit_time_mask_4d(ctx, x, graph->mask1);

        x = build_depthwise_conv2d(
            ctx,
            x,
            subsampling.depthwise1_weight,
            &subsampling.depthwise1_bias,
            channels,
            static_cast<int>(kernel),
            static_cast<int>(stride),
            static_cast<int>(padding));
        x = engine::modules::Conv2dModule({
            channels,
            channels,
            1,
            1,
            1,
            1,
            0,
            0,
            1,
            1,
            true,
        }).build(ctx, x, subsampling.pointwise1);
        x = engine::modules::ReluModule().build(ctx, x);
        x = apply_explicit_time_mask_4d(ctx, x, graph->mask2);

        x = build_depthwise_conv2d(
            ctx,
            x,
            subsampling.depthwise2_weight,
            &subsampling.depthwise2_bias,
            channels,
            static_cast<int>(kernel),
            static_cast<int>(stride),
            static_cast<int>(padding));
        x = engine::modules::Conv2dModule({
            channels,
            channels,
            1,
            1,
            1,
            1,
            0,
            0,
            1,
            1,
            true,
        }).build(ctx, x, subsampling.pointwise2);
        x = engine::modules::ReluModule().build(ctx, x);
        x = apply_explicit_time_mask_4d(ctx, x, graph->mask3);

        x = engine::modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, x);
        x = engine::core::wrap_tensor(ggml_cont(ctx.ggml, x.tensor), x.shape, GGML_TYPE_F32);
        x = engine::modules::ReshapeModule({
            engine::core::TensorShape::from_dims({1, stage3_frames, channels * stage3_features}),
        }).build(ctx, x);
        x = engine::modules::LinearModule({channels * stage3_features, hidden, true}).build(
            ctx,
            x,
            subsampling.linear);
        if (assets.model_config.encoder.scale_input) {
            x = engine::core::wrap_tensor(ggml_scale(ctx.ggml, x.tensor, std::sqrt(static_cast<float>(hidden))), x.shape, GGML_TYPE_F32);
        }

        graph->output = x;
        graph->graph = ggml_new_graph_custom(graph->ggml, kPreEncodeGraphNodes, false);
        ggml_build_forward_expand(graph->graph, graph->output.tensor);
        graph->buffer = ggml_backend_alloc_ctx_tensors(graph->ggml, execution_context.backend());
        if (graph->buffer == nullptr) {
            throw std::runtime_error("Failed to allocate Parakeet pre-encode backend tensors");
        }

        scratch.graph = std::move(graph);
    }

    auto & graph = *scratch.graph;
    engine::core::write_tensor_f32(graph.input, frontend.features);

    int64_t valid1 = frontend.feature_lengths[0];
    valid1 = std::min<int64_t>(stage1_frames, conv_valid_length(valid1, kernel, stride, padding));
    int64_t valid2 = std::min<int64_t>(stage2_frames, conv_valid_length(valid1, kernel, stride, padding));
    int64_t valid3 = std::min<int64_t>(stage3_frames, conv_valid_length(valid2, kernel, stride, padding));

    std::vector<int32_t> mask;
    fill_mask(mask, stage1_frames, valid1);
    engine::core::write_tensor_i32(graph.mask1, mask);
    fill_mask(mask, stage2_frames, valid2);
    engine::core::write_tensor_i32(graph.mask2, mask);
    fill_mask(mask, stage3_frames, valid3);
    engine::core::write_tensor_i32(graph.mask3, mask);

    engine::core::compute_backend_graph(execution_context.backend(), graph.graph);
    engine::core::read_tensor_f32_into(graph.output.tensor, output.hidden);
    output.frames = stage3_frames;
    output.valid_frames = valid3;
}

std::vector<float> compute_parakeet_relative_positional_encoding(
    int64_t batch,
    int64_t hidden_size,
    int64_t frames,
    int64_t max_position_embeddings) {
    if (frames > max_position_embeddings) {
        throw std::runtime_error("Parakeet relative positional encoding exceeds max_position_embeddings");
    }
    if (hidden_size % 2 != 0) {
        throw std::runtime_error("Parakeet relative positional encoding requires even hidden_size");
    }
    const int64_t position_length = 2 * frames - 1;
    std::vector<float> values(static_cast<size_t>(batch * position_length * hidden_size), 0.0f);
    constexpr long double kBase = 10000.0L;
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t pos = 0; pos < position_length; ++pos) {
            const int64_t position_id = frames - 1 - pos;
            for (int64_t i = 0; i < hidden_size / 2; ++i) {
                const long double exponent = static_cast<long double>(2 * i) / static_cast<long double>(hidden_size);
                const long double inv_freq = 1.0L / std::pow(kBase, exponent);
                const long double phase = static_cast<long double>(position_id) * inv_freq;
                const float sinv = static_cast<float>(std::sin(phase));
                const float cosv = static_cast<float>(std::cos(phase));
                const size_t base = static_cast<size_t>((b * position_length + pos) * hidden_size + 2 * i);
                values[base] = sinv;
                values[base + 1] = cosv;
            }
        }
    }
    return values;
}

}  // namespace engine::models::parakeet_tdt
