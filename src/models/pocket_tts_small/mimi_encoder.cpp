#include "engine/models/pocket_tts_small/conditioning.h"
#include "engine/models/pocket_tts_small/assets.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/modules/streaming_conv_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <chrono>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace engine::models::pocket_tts_small {
namespace {

constexpr int64_t kMimiInputChannels = 1;

modules::TransformerEncoderBlockWeights make_transformer_layer_weights(
    core::ModuleBuildContext & ctx,
    const models::pocket_tts_small::BackendWeights & weights,
    int64_t hidden_size,
    int64_t layer) {
    return make_transformer_block_weights(
        ctx,
        weights.mimi_encoder.transformer_layers.at(static_cast<size_t>(layer)),
        hidden_size);
}

core::TensorValue build_residual_block(
    core::ModuleBuildContext & ctx,
    const ResidualBlockWeights & weights,
    const core::TensorValue & input,
    int64_t channels,
    int64_t hidden_channels) {
    auto x = modules::EluModule().build(ctx, input);
    x = modules::StreamingConv1dModule({
        channels,
        hidden_channels,
        3,
        1,
        1,
        true,
        modules::StreamingPadMode::Constant,
    }).build(ctx, x, {
        weights.conv1.weight,
        weights.conv1.bias,
    });
    x = modules::EluModule().build(ctx, x);
    x = modules::StreamingConv1dModule({
        hidden_channels,
        channels,
        1,
        1,
        1,
        true,
        modules::StreamingPadMode::Constant,
    }).build(ctx, x, {
        weights.conv2.weight,
        weights.conv2.bias,
    });
    return modules::ResidualAddModule().build(ctx, input, x);
}

core::TensorValue build_causal_context_mask(
    core::ModuleBuildContext & ctx,
    int64_t query_frames,
    int64_t key_frames) {
    return core::make_tensor(
        ctx,
        GGML_TYPE_F32,
        core::TensorShape::from_dims({1, 1, query_frames, key_frames}));
}

std::vector<float> build_causal_context_mask_values(int64_t query_frames, int64_t key_frames, int64_t context) {
    std::vector<float> values(static_cast<size_t>(query_frames * key_frames), -INFINITY);
    for (int64_t q = 0; q < query_frames; ++q) {
        const int64_t first_k = std::max<int64_t>(0, q - context + 1);
        for (int64_t k = first_k; k <= q && k < key_frames; ++k) {
            values[static_cast<size_t>(q * key_frames + k)] = 0.0F;
        }
    }
    return values;
}

}  // namespace

struct MimiEncoder::Runtime {
    ggml_backend_t backend = nullptr;
    const BackendWeights * weights = nullptr;
    int threads = 0;
    size_t samples = 0;
    size_t context_bytes = 0;
    ggml_context * context = nullptr;
    ggml_gallocr_t allocator = nullptr;
    ggml_backend_buffer_t parameters = nullptr;
    ggml_cgraph * graph = nullptr;
    core::TensorValue input, positions, attention_mask, conditioning;
    std::vector<int32_t> position_values;
    std::vector<float> attention_mask_values;

    ~Runtime() {
        core::release_backend_graph_resources(backend, graph, true);
        if (allocator) ggml_gallocr_free(allocator);
        if (parameters) ggml_backend_buffer_free(parameters);
        if (context) ggml_free(context);
    }
};

MimiEncoder::MimiEncoder() = default;
MimiEncoder::~MimiEncoder() = default;

MimiPromptEmbedding MimiEncoder::encode_prompt_embedding(
    ggml_backend_t backend,
    int threads,
    const models::pocket_tts_small::Assets & manifest,
    const models::pocket_tts_small::BackendWeights & weights,
    const std::vector<float> & waveform,
    size_t graph_context_bytes) const {
    const int64_t flow_hidden = manifest.model_config.flow_dim;
    const int64_t mimi_dimension = manifest.model_config.mimi_seanet_dimension;
    const int64_t mimi_base_channels = manifest.model_config.mimi_base_filters;
    const int64_t mimi_heads = manifest.model_config.mimi_heads;
    const int64_t mimi_intermediate = manifest.model_config.mimi_intermediate_size;
    const int64_t mimi_transformer_layers = manifest.model_config.mimi_layers;
    const int64_t mimi_inner_dimension = manifest.model_config.mimi_inner_dim;
    const int64_t mimi_context = manifest.model_config.mimi_context;
    const auto started = std::chrono::steady_clock::now();
    if (!runtime_ || runtime_->backend != backend || runtime_->weights != &weights ||
        runtime_->threads != threads || runtime_->samples != waveform.size() ||
        runtime_->context_bytes != graph_context_bytes) {
        auto next = std::make_unique<Runtime>();
        next->backend = backend;
        next->weights = &weights;
        next->threads = threads;
        next->samples = waveform.size();
        next->context_bytes = graph_context_bytes;
        next->context = ggml_init({graph_context_bytes, nullptr, true});
        if (!next->context) throw std::runtime_error("Failed to initialize Mimi encoder context");
        auto & capture = *next;
        core::ModuleBuildContext ctx{next->context, "pocket_tts_small.mimi_encoder", weights.backend_type};
            auto input = core::make_tensor(
                ctx,
                GGML_TYPE_F32,
                core::TensorShape::from_dims({1, 1, static_cast<int64_t>(waveform.size())}));
            capture.input = input;

            auto x = modules::StreamingConv1dModule({
                kMimiInputChannels,
                mimi_base_channels,
                7,
                1,
                1,
                true,
                modules::StreamingPadMode::Constant,
            }).build(ctx, input, {
                weights.mimi_encoder.input_conv.weight,
                weights.mimi_encoder.input_conv.bias,
            });
            x = build_residual_block(ctx, weights.mimi_encoder.block0, x, mimi_base_channels, mimi_base_channels / 2);
            x = modules::EluModule().build(ctx, x);
            x = modules::StreamingConv1dModule({
                mimi_base_channels,
                mimi_base_channels * 2,
                8,
                4,
                1,
                true,
                modules::StreamingPadMode::Constant,
            }).build(ctx, x, {
                weights.mimi_encoder.downsample0.weight,
                weights.mimi_encoder.downsample0.bias,
            });
            x = build_residual_block(ctx, weights.mimi_encoder.block1, x, mimi_base_channels * 2, mimi_base_channels);
            x = modules::EluModule().build(ctx, x);
            x = modules::StreamingConv1dModule({
                mimi_base_channels * 2,
                mimi_base_channels * 4,
                10,
                5,
                1,
                true,
                modules::StreamingPadMode::Constant,
            }).build(ctx, x, {
                weights.mimi_encoder.downsample1.weight,
                weights.mimi_encoder.downsample1.bias,
            });
            x = build_residual_block(ctx, weights.mimi_encoder.block2, x, mimi_base_channels * 4, mimi_base_channels * 2);
            x = modules::EluModule().build(ctx, x);
            x = modules::StreamingConv1dModule({
                mimi_base_channels * 4,
                mimi_dimension,
                12,
                6,
                1,
                true,
                modules::StreamingPadMode::Constant,
            }).build(ctx, x, {
                weights.mimi_encoder.downsample2.weight,
                weights.mimi_encoder.downsample2.bias,
            });
            x = modules::EluModule().build(ctx, x);
            x = modules::StreamingConv1dModule({
                mimi_dimension,
                mimi_dimension,
                3,
                1,
                1,
                true,
                modules::StreamingPadMode::Constant,
            }).build(ctx, x, {
                weights.mimi_encoder.output_conv.weight,
                weights.mimi_encoder.output_conv.bias,
            });

            const int64_t encoder_frames = x.shape.dims[2];
            auto positions = core::make_tensor(
                ctx,
                GGML_TYPE_I32,
                core::TensorShape::from_dims({encoder_frames}));
            capture.positions = positions;
            capture.position_values.resize(static_cast<size_t>(encoder_frames));
            for (int64_t i = 0; i < encoder_frames; ++i) {
                capture.position_values[static_cast<size_t>(i)] = static_cast<int32_t>(i);
            }
            std::vector<modules::TransformerEncoderBlockWeights> transformer_layers;
            transformer_layers.reserve(static_cast<size_t>(mimi_transformer_layers));
            for (int64_t layer = 0; layer < mimi_transformer_layers; ++layer) {
                transformer_layers.push_back(make_transformer_layer_weights(ctx, weights, mimi_dimension, layer));
            }
            auto attention_mask = build_causal_context_mask(ctx, encoder_frames, encoder_frames);
            capture.attention_mask = attention_mask;
            capture.attention_mask_values = build_causal_context_mask_values(encoder_frames, encoder_frames, mimi_context);
            x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
            modules::TransformerEncoderBlockConfig transformer_config{
                mimi_dimension, mimi_heads, mimi_intermediate, 1.0e-5F, false};
            transformer_config.gelu_approximation = modules::GeluApproximation::Tanh;
            const modules::StreamingTransformerEncoderBlockModule transformer(transformer_config);
            for (const auto & layer : transformer_layers) {
                x = transformer.build(ctx, x, positions, layer, std::nullopt, std::nullopt, attention_mask).output;
            }
            x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
            x = modules::StreamingConv1dModule({
                mimi_dimension,
                mimi_inner_dimension,
                32,
                16,
                1,
                false,
                modules::StreamingPadMode::Replicate,
            }).build(ctx, x, {
                weights.mimi_encoder.downsample_conv.weight,
                weights.mimi_encoder.downsample_conv.bias,
            });

            auto latents_btf = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
            capture.conditioning = modules::LinearModule({mimi_inner_dimension, flow_hidden, false}).build(
                ctx,
                latents_btf,
                {
                    weights.flow.speaker_proj_weight,
                    std::nullopt,
                });
        ggml_set_input(capture.input.tensor);
        ggml_set_output(capture.positions.tensor);
        ggml_set_output(capture.attention_mask.tensor);
        ggml_set_output(capture.conditioning.tensor);
        capture.graph = ggml_new_graph_custom(capture.context, 32768, false);
        ggml_build_forward_expand(capture.graph, capture.conditioning.tensor);
        if (core::is_host_backend(backend)) {
            capture.parameters = ggml_backend_alloc_ctx_tensors(capture.context, backend);
            if (!capture.parameters) throw std::runtime_error("Mimi encoder parameter allocation failed");
        }
        capture.allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!capture.allocator || !ggml_gallocr_reserve(capture.allocator, capture.graph) ||
            !ggml_gallocr_alloc_graph(capture.allocator, capture.graph))
            throw std::runtime_error("Mimi encoder graph allocation failed");
        core::write_tensor_i32(capture.positions, capture.position_values);
        core::write_tensor_f32(capture.attention_mask, capture.attention_mask_values);
        runtime_ = std::move(next);
    }
    core::write_tensor_f32(runtime_->input, waveform);
    core::set_backend_threads(backend, threads);
    const auto status = core::compute_backend_graph(backend, runtime_->graph);
    if (status != GGML_STATUS_SUCCESS)
        throw std::runtime_error("Mimi encoder graph execution failed: " + std::string(ggml_status_to_string(status)));
    MimiPromptEmbedding result;
    result.values = core::read_tensor_f32(runtime_->conditioning.tensor);
    result.frames = static_cast<int64_t>(result.values.size() / static_cast<size_t>(flow_hidden));
    const double encode_ms = engine::debug::elapsed_ms(started);
    engine::debug::timing_log_scalar("pocket_tts_small.mimi_encoder_ms", encode_ms);
    return result;
}

}  // namespace engine::models::pocket_tts_small
