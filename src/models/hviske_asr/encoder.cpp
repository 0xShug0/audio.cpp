#include "engine/models/hviske_asr/encoder.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/asr_helpers.h"
#include "engine/framework/modules/conformer_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace engine::models::hviske_asr {
namespace {

using Clock = std::chrono::steady_clock;

constexpr size_t kEncoderGraphNodes = 1048576;

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

std::vector<float> make_relative_positional_encoding(int64_t batch, int64_t hidden, int64_t frames, int64_t max_frames) {
    if (frames > max_frames) {
        throw std::runtime_error("Hviske encoder relative position frames exceed configured maximum");
    }
    if (hidden % 2 != 0) {
        throw std::runtime_error("Hviske encoder relative position requires even hidden size");
    }
    const int64_t position_length = 2 * frames - 1;
    std::vector<float> values(static_cast<size_t>(batch * position_length * hidden), 0.0f);
    constexpr long double kBase = 10000.0L;
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t pos = 0; pos < position_length; ++pos) {
            const int64_t position_id = frames - 1 - pos;
            for (int64_t i = 0; i < hidden / 2; ++i) {
                const long double exponent = static_cast<long double>(2 * i) / static_cast<long double>(hidden);
                const long double inv_freq = 1.0L / std::pow(kBase, exponent);
                const long double phase = static_cast<long double>(position_id) * inv_freq;
                const size_t base = static_cast<size_t>((b * position_length + pos) * hidden + 2 * i);
                values[base] = static_cast<float>(std::sin(phase));
                values[base + 1] = static_cast<float>(std::cos(phase));
            }
        }
    }
    return values;
}

}  // namespace

struct HviskeEncoderRuntime::Graph {
    int64_t input_frames = 0;
    int64_t input_features = 0;
    int64_t encoded_frames = 0;
    int64_t encoded_features = 0;
    int64_t hidden = 0;
    int64_t decoder_hidden = 0;
    ggml_backend_t backend = nullptr;
    ggml_context * ggml = nullptr;
    ggml_gallocr_t gallocr = nullptr;
    ggml_gallocr_t pos_gallocr = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_cgraph * pos_projection_graph = nullptr;
    engine::core::TensorValue input;
    engine::core::TensorValue mask1;
    engine::core::TensorValue mask2;
    engine::core::TensorValue mask3;
    engine::core::TensorValue pos_emb;
    engine::core::TensorValue attention_mask;
    engine::core::TensorValue keep_mask;
    std::vector<engine::core::TensorValue> projected_pos_emb;
    std::vector<engine::core::TensorValue> projected_pos_emb_computed;
    engine::core::TensorValue output;

    ~Graph() {
        if (backend != nullptr) {
            engine::core::release_backend_graph_resources(backend, graph);
            engine::core::release_backend_graph_resources(backend, pos_projection_graph);
        }
        if (pos_gallocr != nullptr) {
            ggml_gallocr_free(pos_gallocr);
        }
        if (gallocr != nullptr) {
            ggml_gallocr_free(gallocr);
        }
        if (ggml != nullptr) {
            ggml_free(ggml);
        }
    }
};

HviskeEncoderRuntime::HviskeEncoderRuntime(
    std::shared_ptr<const HviskeAssets> assets,
    std::shared_ptr<const HviskeWeights> weights,
    engine::core::ExecutionContext & execution_context,
    size_t graph_arena_bytes)
    : assets_(std::move(assets)),
      weights_(std::move(weights)),
      execution_context_(&execution_context),
      graph_arena_bytes_(graph_arena_bytes) {
    if (assets_ == nullptr || weights_ == nullptr) {
        throw std::runtime_error("Hviske encoder requires assets and weights");
    }
    if (graph_arena_bytes_ == 0) {
        throw std::runtime_error("Hviske encoder graph arena must be non-zero");
    }
}

HviskeEncoderRuntime::~HviskeEncoderRuntime() = default;
HviskeEncoderRuntime::HviskeEncoderRuntime(HviskeEncoderRuntime &&) noexcept = default;
HviskeEncoderRuntime & HviskeEncoderRuntime::operator=(HviskeEncoderRuntime &&) noexcept = default;

void HviskeEncoderRuntime::ensure_graph(int64_t input_frames, int64_t input_features) {
    if (input_frames <= 0 || input_features <= 0) {
        throw std::runtime_error("Hviske encoder graph requires positive input shape");
    }
    if (graph_ != nullptr &&
        graph_->backend == execution_context_->backend() &&
        graph_->input_frames >= input_frames &&
        graph_->input_features == input_features) {
        debug::timing_log_scalar("hviske_asr.encoder.graph_rebuild_ms", 0.0);
        debug::trace_log_scalar("hviske_asr.encoder.graph_cache_hit", true);
        return;
    }

    const auto build_start = Clock::now();
    const auto & config = assets_->config.encoder;
    const auto & encoder_weights = weights_->encoder;
    const int64_t kernel = 3;
    const int64_t stride = 2;
    const int64_t padding = 1;
    const int64_t channels = config.subsampling_conv_channels;
    const int64_t stage1_frames = conv_output_dim(input_frames, kernel, stride, padding);
    const int64_t stage1_features = conv_output_dim(input_features, kernel, stride, padding);
    const int64_t stage2_frames = conv_output_dim(stage1_frames, kernel, stride, padding);
    const int64_t stage2_features = conv_output_dim(stage1_features, kernel, stride, padding);
    const int64_t stage3_frames = conv_output_dim(stage2_frames, kernel, stride, padding);
    const int64_t stage3_features = conv_output_dim(stage2_features, kernel, stride, padding);
    if (stage3_features * channels != config.subsampling_conv_channels * (config.feat_in / config.subsampling_factor)) {
        throw std::runtime_error("Hviske encoder subsampling linear shape mismatch");
    }

    auto graph = std::make_unique<Graph>();
    graph->input_frames = input_frames;
    graph->input_features = input_features;
    graph->encoded_frames = stage3_frames;
    graph->encoded_features = stage3_features;
    graph->hidden = config.hidden_size;
    graph->decoder_hidden = assets_->config.decoder.hidden_size;
    graph->backend = execution_context_->backend();

    ggml_init_params params = {};
    params.mem_size = graph_arena_bytes_;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    graph->ggml = ggml_init(params);
    if (graph->ggml == nullptr) {
        throw std::runtime_error("Failed to initialize Hviske encoder ggml context");
    }

    engine::core::ModuleBuildContext ctx{graph->ggml, "hviske_asr.encoder", execution_context_->backend_type()};
    graph->input = engine::core::make_tensor(
        ctx,
        GGML_TYPE_F32,
        engine::core::TensorShape::from_dims({1, 1, input_frames, input_features}));
    ggml_set_input(graph->input.tensor);
    ggml_set_output(graph->input.tensor);
    graph->mask1 = engine::core::make_tensor(ctx, GGML_TYPE_I32, engine::core::TensorShape::from_dims({1, stage1_frames}));
    ggml_set_input(graph->mask1.tensor);
    ggml_set_output(graph->mask1.tensor);
    graph->mask2 = engine::core::make_tensor(ctx, GGML_TYPE_I32, engine::core::TensorShape::from_dims({1, stage2_frames}));
    ggml_set_input(graph->mask2.tensor);
    ggml_set_output(graph->mask2.tensor);
    graph->mask3 = engine::core::make_tensor(ctx, GGML_TYPE_I32, engine::core::TensorShape::from_dims({1, stage3_frames}));
    ggml_set_input(graph->mask3.tensor);
    ggml_set_output(graph->mask3.tensor);
    graph->pos_emb = engine::core::make_tensor(
        ctx,
        GGML_TYPE_F32,
        engine::core::TensorShape::from_dims({1, 2 * stage3_frames - 1, config.hidden_size}));
    ggml_set_input(graph->pos_emb.tensor);
    ggml_set_output(graph->pos_emb.tensor);
    graph->attention_mask = engine::core::make_tensor(
        ctx,
        GGML_TYPE_F32,
        engine::core::TensorShape::from_dims({stage3_frames, stage3_frames}));
    ggml_set_input(graph->attention_mask.tensor);
    ggml_set_output(graph->attention_mask.tensor);
    graph->keep_mask = engine::core::make_tensor(
        ctx,
        GGML_TYPE_I32,
        engine::core::TensorShape::from_dims({1, stage3_frames}));
    ggml_set_input(graph->keep_mask.tensor);
    ggml_set_output(graph->keep_mask.tensor);

    auto x = engine::modules::Conv2dModule({
        1,
        channels,
        kernel,
        kernel,
        stride,
        stride,
        padding,
        padding,
        1,
        1,
        true,
    }).build(ctx, graph->input, encoder_weights.subsampling.conv0);
    x = engine::modules::ReluModule().build(ctx, x);
    x = engine::modules::TimeMask4dModule().build(ctx, x, graph->mask1);

    x = engine::modules::DepthwiseConv2dModule({channels, kernel, kernel, stride, stride, padding, padding, 1, 1, true})
            .build(ctx, x, {encoder_weights.subsampling.depthwise1_weight, encoder_weights.subsampling.depthwise1_bias});
    x = engine::modules::TimeMask4dModule().build(ctx, x, graph->mask2);
    x = engine::modules::Conv2dModule({
        channels, channels, 1, 1, 1, 1, 0, 0, 1, 1, true,
    }).build(ctx, x, encoder_weights.subsampling.pointwise1);
    x = engine::modules::ReluModule().build(ctx, x);
    x = engine::modules::TimeMask4dModule().build(ctx, x, graph->mask2);

    x = engine::modules::DepthwiseConv2dModule({channels, kernel, kernel, stride, stride, padding, padding, 1, 1, true})
            .build(ctx, x, {encoder_weights.subsampling.depthwise2_weight, encoder_weights.subsampling.depthwise2_bias});
    x = engine::modules::TimeMask4dModule().build(ctx, x, graph->mask3);
    x = engine::modules::Conv2dModule({
        channels, channels, 1, 1, 1, 1, 0, 0, 1, 1, true,
    }).build(ctx, x, encoder_weights.subsampling.pointwise2);
    x = engine::modules::ReluModule().build(ctx, x);
    x = engine::modules::TimeMask4dModule().build(ctx, x, graph->mask3);

    x = engine::modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, x);
    x = engine::core::wrap_tensor(ggml_cont(ctx.ggml, x.tensor), x.shape, GGML_TYPE_F32);
    x = engine::modules::ReshapeModule({
        engine::core::TensorShape::from_dims({1, stage3_frames, channels * stage3_features}),
    }).build(ctx, x);
    x = engine::modules::LinearModule({channels * stage3_features, config.hidden_size, true}).build(
        ctx,
        x,
        encoder_weights.subsampling.linear);

    graph->projected_pos_emb.reserve(static_cast<size_t>(config.layers));
    graph->projected_pos_emb_computed.reserve(static_cast<size_t>(config.layers));
    for (int64_t layer_idx = 0; layer_idx < config.layers; ++layer_idx) {
        const auto & layer = encoder_weights.layers[static_cast<size_t>(layer_idx)];
        graph->projected_pos_emb.push_back(engine::core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({1, 2 * stage3_frames - 1, config.hidden_size})));
        ggml_set_input(graph->projected_pos_emb.back().tensor);
        ggml_set_output(graph->projected_pos_emb.back().tensor);
        graph->projected_pos_emb_computed.push_back(
            engine::modules::LinearModule({config.hidden_size, config.hidden_size, false}).build(
                ctx,
                graph->pos_emb,
                {layer.self_attn.pos_weight, std::nullopt}));
        ggml_set_output(graph->projected_pos_emb_computed.back().tensor);
    }

    for (int64_t layer_idx = 0; layer_idx < config.layers; ++layer_idx) {
        const auto & layer = encoder_weights.layers[static_cast<size_t>(layer_idx)];
        x = engine::modules::RelativeConformerBlockModule({
            config.hidden_size,
            config.heads,
            config.intermediate_size,
            config.conv_kernel,
            1.0e-5f,
            true,
            -1,
            -1,
            0,
        }).build(
            ctx,
            x,
            std::nullopt,
            {
                layer.norm_feed_forward1,
                layer.ff1_linear1,
                layer.ff1_linear2,
                layer.norm_self_att,
                layer.self_attn,
                {layer.norm_conv, layer.conv_pointwise1, layer.conv_depthwise, {layer.conv_norm.scale, layer.conv_norm.bias}, layer.conv_pointwise2},
                layer.norm_feed_forward2,
                layer.ff2_linear1,
                layer.ff2_linear2,
                layer.norm_out,
            },
            graph->attention_mask,
            graph->keep_mask,
            graph->keep_mask,
            graph->projected_pos_emb[static_cast<size_t>(layer_idx)]);
    }

    graph->output = engine::modules::LinearModule({config.hidden_size, assets_->config.decoder.hidden_size, true}).build(
        ctx,
        x,
        encoder_weights.encoder_projector);
    ggml_set_output(graph->output.tensor);

    graph->pos_projection_graph = ggml_new_graph_custom(graph->ggml, 4096, false);
    for (const auto & projected : graph->projected_pos_emb_computed) {
        ggml_build_forward_expand(graph->pos_projection_graph, projected.tensor);
    }
    graph->graph = ggml_new_graph_custom(graph->ggml, kEncoderGraphNodes, false);
    ggml_build_forward_expand(graph->graph, graph->output.tensor);
    graph->gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(graph->backend));
    if (graph->gallocr == nullptr ||
        !ggml_gallocr_reserve(graph->gallocr, graph->graph) ||
        !ggml_gallocr_alloc_graph(graph->gallocr, graph->graph)) {
        throw std::runtime_error("Failed to allocate Hviske encoder graph tensors");
    }
    graph->pos_gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(graph->backend));
    if (graph->pos_gallocr == nullptr ||
        !ggml_gallocr_reserve(graph->pos_gallocr, graph->pos_projection_graph) ||
        !ggml_gallocr_alloc_graph(graph->pos_gallocr, graph->pos_projection_graph)) {
        throw std::runtime_error("Failed to allocate Hviske encoder position projection graph tensors");
    }
    engine::core::write_tensor_f32(
        graph->pos_emb,
        make_relative_positional_encoding(1, config.hidden_size, graph->encoded_frames, config.pos_emb_max_len));
    const auto pos_status = engine::core::compute_backend_graph(execution_context_->backend(), graph->pos_projection_graph);
    if (pos_status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("Hviske encoder position projection graph compute failed");
    }
    for (size_t i = 0; i < graph->projected_pos_emb.size(); ++i) {
        ggml_backend_tensor_copy(
            graph->projected_pos_emb_computed[i].tensor,
            graph->projected_pos_emb[i].tensor);
    }
    graph_ = std::move(graph);
    const double build_ms = engine::debug::elapsed_ms(build_start, Clock::now());
    debug::timing_log_scalar("hviske_asr.encoder.graph_build_ms", build_ms);
    debug::timing_log_scalar("hviske_asr.encoder.graph_rebuild_ms", build_ms);
    debug::trace_log_scalar("hviske_asr.encoder.graph_cache_hit", false);
    debug::trace_log_scalar("hviske_asr.encoder.graph_frames", input_frames);
    debug::trace_log_scalar("hviske_asr.encoder.encoded_frames", stage3_frames);
}

void HviskeEncoderRuntime::prepare_capacity(int64_t input_frames, int64_t input_features) {
    ensure_graph(input_frames, input_features);
}

HviskeEncodedAudio HviskeEncoderRuntime::encode(const HviskeFrontendFeatures & features) {
    if (execution_context_ == nullptr) {
        throw std::runtime_error("Hviske encoder execution context is null");
    }
    if (features.feature_dim <= 0 || features.frames <= 0) {
        throw std::runtime_error("Hviske encoder requires positive frontend feature shape");
    }
    if (static_cast<int64_t>(features.values.size()) != features.feature_dim * features.frames) {
        throw std::runtime_error("Hviske encoder frontend value count mismatch");
    }

    const auto encode_start = Clock::now();
    ensure_graph(features.frames, features.feature_dim);
    auto & graph = *graph_;

    const auto input_prepare_start = Clock::now();
    time_major_input_.assign(static_cast<size_t>(graph.input_frames * graph.input_features), 0.0f);
    for (int64_t t = 0; t < features.frames; ++t) {
        for (int64_t f = 0; f < features.feature_dim; ++f) {
            time_major_input_[static_cast<size_t>(t * graph.input_features + f)] =
                features.values[static_cast<size_t>(f * features.frames + t)];
        }
    }
    engine::core::write_tensor_f32(graph.input, time_major_input_);
    debug::timing_log_scalar("hviske_asr.encoder.input_prepare_upload_ms", engine::debug::elapsed_ms(input_prepare_start, Clock::now()));

    const auto mask_upload_start = Clock::now();
    int64_t valid1 = std::min<int64_t>(graph.mask1.shape.dims[1], conv_valid_length(features.valid_frames, 3, 2, 1));
    int64_t valid2 = std::min<int64_t>(graph.mask2.shape.dims[1], conv_valid_length(valid1, 3, 2, 1));
    int64_t valid3 = std::min<int64_t>(graph.encoded_frames, conv_valid_length(valid2, 3, 2, 1));

    std::vector<int32_t> mask;
    engine::modules::fill_asr_keep_mask(mask, graph.mask1.shape.dims[1], valid1);
    engine::core::write_tensor_i32(graph.mask1, mask);
    engine::modules::fill_asr_keep_mask(mask, graph.mask2.shape.dims[1], valid2);
    engine::core::write_tensor_i32(graph.mask2, mask);
    engine::modules::fill_asr_keep_mask(mask, graph.mask3.shape.dims[1], valid3);
    engine::core::write_tensor_i32(graph.mask3, mask);
    engine::core::write_tensor_i32(graph.keep_mask, engine::modules::make_asr_keep_mask(graph.encoded_frames, valid3));
    engine::core::write_tensor_f32(graph.attention_mask, engine::modules::make_asr_full_attention_bias(graph.encoded_frames, valid3));
    debug::timing_log_scalar("hviske_asr.encoder.mask_upload_ms", engine::debug::elapsed_ms(mask_upload_start, Clock::now()));

    engine::core::set_backend_threads(execution_context_->backend(), execution_context_->config().threads);
    const auto compute_start = Clock::now();
    const auto status = engine::core::compute_backend_graph(execution_context_->backend(), graph.graph, nullptr, "Hviske encoder");
    ggml_backend_synchronize(execution_context_->backend());
    debug::timing_log_scalar("hviske_asr.encoder.graph.compute_ms", engine::debug::elapsed_ms(compute_start, Clock::now()));
    if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("Hviske encoder graph compute failed");
    }

    const auto output_read_start = Clock::now();
    engine::core::read_tensor_f32_into(graph.output.tensor, output_scratch_);
    if (valid3 < graph.encoded_frames) {
        for (int64_t row = std::max<int64_t>(valid3, 0); row < graph.encoded_frames; ++row) {
            std::fill_n(
                output_scratch_.begin() + static_cast<std::ptrdiff_t>(row * graph.decoder_hidden),
                static_cast<std::ptrdiff_t>(graph.decoder_hidden),
                0.0f);
        }
    }
    debug::timing_log_scalar("hviske_asr.encoder.output_read_ms", engine::debug::elapsed_ms(output_read_start, Clock::now()));

    HviskeEncodedAudio out;
    out.values = output_scratch_;
    out.frames = graph.encoded_frames;
    out.valid_frames = valid3;
    out.hidden_size = graph.decoder_hidden;
    debug::timing_log_scalar("hviske_asr.encoder_ms", engine::debug::elapsed_ms(encode_start, Clock::now()));
    debug::trace_log_scalar("hviske_asr.encoder_valid_frames", out.valid_frames);
    return out;
}

}  // namespace engine::models::hviske_asr
