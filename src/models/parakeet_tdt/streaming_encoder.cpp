#include "engine/models/parakeet_tdt/session.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention_modules.h"
#include "engine/framework/modules/conformer_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"

#include "ggml-backend.h"

#include <memory>
#include <optional>
#include <stdexcept>

namespace engine::models::parakeet_tdt {

namespace {

constexpr size_t kStreamingEncoderGraphBytes = 256 * 1024 * 1024;
constexpr size_t kStreamingEncoderGraphNodes = 1048576;

std::vector<int32_t> make_keep_mask_i32(int64_t frames, int64_t valid_frames) {
    std::vector<int32_t> mask(static_cast<size_t>(frames), 0);
    for (int64_t i = 0; i < frames; ++i) {
        mask[static_cast<size_t>(i)] = i < valid_frames ? 1 : 0;
    }
    return mask;
}

std::vector<float> make_chunked_limited_with_rc_attention_bias(
    int64_t frames,
    int64_t valid_frames,
    int64_t left_context,
    int64_t chunk_frames,
    int64_t right_context) {
    if (chunk_frames <= 0) {
        throw std::runtime_error("Streaming chunked attention requires positive chunk_frames");
    }
    constexpr float kMasked = -10000.0f;
    std::vector<float> mask(static_cast<size_t>(frames * frames), kMasked);
    for (int64_t q = 0; q < frames; ++q) {
        const int64_t chunk_idx = q / chunk_frames;
        const int64_t window_start = std::max<int64_t>(0, chunk_idx * chunk_frames - left_context);
        const int64_t window_end = std::min<int64_t>(frames - 1, chunk_idx * chunk_frames + chunk_frames - 1 + right_context);
        for (int64_t k = window_start; k <= window_end; ++k) {
            mask[static_cast<size_t>(q * frames + k)] = 0.0f;
        }
    }
    for (int64_t q = valid_frames; q < frames; ++q) {
        for (int64_t k = 0; k < frames; ++k) {
            mask[static_cast<size_t>(q * frames + k)] = kMasked;
        }
    }
    for (int64_t k = valid_frames; k < frames; ++k) {
        for (int64_t q = 0; q < frames; ++q) {
            mask[static_cast<size_t>(q * frames + k)] = kMasked;
        }
    }
    return mask;
}

}  // namespace

struct ParakeetTDTSession::StreamingChunkEncoderGraph {
    int64_t frames = 0;

    ggml_backend_t backend = nullptr;
    ggml_context * ggml = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_cgraph * pos_projection_graph = nullptr;
    ggml_backend_buffer_t buffer = nullptr;

    core::TensorValue input_hidden;
    core::TensorValue pos_emb;
    core::TensorValue attention_mask;
    core::TensorValue keep_mask;
    std::vector<core::TensorValue> projected_pos_emb;
    std::vector<core::TensorValue> projected_pos_emb_computed;
    core::TensorValue output_hidden;
    core::TensorValue output_projected;

    ~StreamingChunkEncoderGraph() {
        if (buffer != nullptr) {
            ggml_backend_buffer_free(buffer);
        }
        if (ggml != nullptr) {
            ggml_free(ggml);
        }
    }
};

ParakeetTDTSession::StreamingChunkEncoderState::StreamingChunkEncoderState() = default;
ParakeetTDTSession::StreamingChunkEncoderState::~StreamingChunkEncoderState() = default;
ParakeetTDTSession::StreamingChunkEncoderState::StreamingChunkEncoderState(StreamingChunkEncoderState &&) noexcept = default;
ParakeetTDTSession::StreamingChunkEncoderState &
ParakeetTDTSession::StreamingChunkEncoderState::operator=(StreamingChunkEncoderState &&) noexcept = default;

void ParakeetTDTSession::ensure_streaming_chunk_encoder_graph(int64_t, int64_t, int64_t) {
    const auto ensure_start = std::chrono::steady_clock::now();
    const int64_t fixed_frames = streaming_state_.context.window_frames;
    if (fixed_frames <= 0) {
        throw std::runtime_error("Parakeet streaming encoder graph requires positive frame count");
    }
    if (streaming_chunk_encoder_.graph != nullptr &&
        streaming_chunk_encoder_.graph->backend == execution_context().backend() &&
        streaming_chunk_encoder_.graph->frames == fixed_frames) {
        streaming_state_.timings.encoder_ensure_graph_ms +=
            std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
                std::chrono::steady_clock::now() - ensure_start)
                .count();
        return;
    }

    const auto graph_build_start = std::chrono::steady_clock::now();
    auto graph = std::make_unique<StreamingChunkEncoderGraph>();
    graph->frames = fixed_frames;
    graph->backend = execution_context().backend();

    ggml_init_params params = {};
    params.mem_size = kStreamingEncoderGraphBytes;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    graph->ggml = ggml_init(params);
    if (graph->ggml == nullptr) {
        throw std::runtime_error("Failed to initialize Parakeet streaming encoder ggml context");
    }

    core::ModuleBuildContext ctx = {};
    ctx.ggml = graph->ggml;
    ctx.module_instance_name = "parakeet_tdt_streaming_encoder";

    const int64_t hidden = assets_->model_config.encoder.hidden_size;
    const int64_t heads = assets_->model_config.encoder.num_attention_heads;
    const int64_t intermediate = assets_->model_config.encoder.intermediate_size;
    const int64_t kernel = assets_->model_config.encoder.conv_kernel_size;
    const int64_t num_layers = assets_->model_config.encoder.num_hidden_layers;
    const int64_t decoder_hidden = assets_->model_config.decoder_hidden_size;
    const auto & encoder_weights = weights_->encoder;

    graph->input_hidden = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, fixed_frames, hidden}));
    graph->pos_emb = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 2 * fixed_frames - 1, hidden}));
    graph->attention_mask = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({fixed_frames, fixed_frames}));
    graph->keep_mask = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({1, fixed_frames}));
    graph->projected_pos_emb.reserve(static_cast<size_t>(num_layers));
    graph->projected_pos_emb_computed.reserve(static_cast<size_t>(num_layers));
    for (int64_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
        const auto & layer_weights = encoder_weights.layers[static_cast<size_t>(layer_idx)];
        graph->projected_pos_emb.push_back(
            core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 2 * fixed_frames - 1, hidden})));
        graph->projected_pos_emb_computed.push_back(
            modules::LinearModule({hidden, hidden, false}).build(ctx, graph->pos_emb, {layer_weights.self_attn.pos_weight, std::nullopt}));
    }

    auto x = graph->input_hidden;

    for (int64_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
        const auto & layer_weights = encoder_weights.layers[static_cast<size_t>(layer_idx)];
        x = modules::RelativeConformerBlockModule({
            hidden,
            heads,
            intermediate,
            kernel,
            1.0e-5f,
            false,
            -1,
            -1,
            0,
        }).build(
            ctx,
            x,
            std::nullopt,
            {
                layer_weights.norm_feed_forward1,
                layer_weights.ff1_linear1,
                layer_weights.ff1_linear2,
                layer_weights.norm_self_att,
                layer_weights.self_attn,
                {layer_weights.norm_conv, layer_weights.conv_pointwise_conv1, layer_weights.conv_depthwise_conv, {layer_weights.conv_norm.scale, layer_weights.conv_norm.bias}, layer_weights.conv_pointwise_conv2},
                layer_weights.norm_feed_forward2,
                layer_weights.ff2_linear1,
                layer_weights.ff2_linear2,
                layer_weights.norm_out,
            },
            graph->attention_mask,
            graph->keep_mask,
            graph->keep_mask,
            graph->projected_pos_emb[static_cast<size_t>(layer_idx)]);
    }

    graph->output_hidden = x;
    graph->output_projected = modules::LinearModule({hidden, decoder_hidden, true}).build(
        ctx,
        x,
        encoder_weights.encoder_projector);

    graph->pos_projection_graph = ggml_new_graph_custom(graph->ggml, 4096, false);
    for (const auto & projected : graph->projected_pos_emb_computed) {
        ggml_build_forward_expand(graph->pos_projection_graph, projected.tensor);
    }

    graph->graph = ggml_new_graph_custom(graph->ggml, kStreamingEncoderGraphNodes, false);
    ggml_build_forward_expand(graph->graph, graph->output_projected.tensor);
    graph->buffer = ggml_backend_alloc_ctx_tensors(graph->ggml, graph->backend);
    if (graph->buffer == nullptr) {
        throw std::runtime_error("Failed to allocate Parakeet streaming encoder backend tensors");
    }

    const auto pos_emb = compute_parakeet_relative_positional_encoding(
        1,
        hidden,
        fixed_frames,
        assets_->model_config.encoder.max_position_embeddings);
    core::write_tensor_f32(graph->pos_emb, pos_emb);
    engine::core::compute_backend_graph(graph->backend, graph->pos_projection_graph);
    for (size_t i = 0; i < graph->projected_pos_emb.size(); ++i) {
        ggml_backend_tensor_copy(graph->projected_pos_emb_computed[i].tensor, graph->projected_pos_emb[i].tensor);
    }

    streaming_chunk_encoder_.graph = std::move(graph);
    const auto ensure_end = std::chrono::steady_clock::now();
    streaming_state_.timings.encoder_ensure_graph_ms +=
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(ensure_end - ensure_start).count();
    streaming_state_.timings.encoder_graph_build_ms +=
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(ensure_end - graph_build_start).count();
    streaming_state_.timings.encoder_graph_rebuilds += 1;
}

const std::vector<float> & ParakeetTDTSession::run_streaming_chunk_graph_encoder(
    const ParakeetPreEncodeBatch & pre_encode,
    int64_t left_context,
    int64_t chunk_frames,
    int64_t right_context) {
    const int64_t frames = streaming_state_.context.window_frames;
    const int64_t hidden_out = assets_->model_config.decoder_hidden_size;

    if (frames <= 0) {
        throw std::runtime_error("Parakeet streaming encoder requires positive configured window frames");
    }
    if (pre_encode.frames != frames) {
        throw std::runtime_error("Streaming pre-encode frames do not match fixed streaming window frames");
    }
    ensure_streaming_chunk_encoder_graph(frames, left_context, right_context);
    auto * graph = streaming_chunk_encoder_.graph.get();
    if (graph == nullptr) {
        throw std::runtime_error("Failed to find cached Parakeet streaming encoder graph");
    }

    const auto input_write_start = std::chrono::steady_clock::now();
    core::write_tensor_f32(graph->input_hidden, pre_encode.hidden);
    const auto input_write_end = std::chrono::steady_clock::now();
    streaming_state_.timings.encoder_input_write_ms +=
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(input_write_end - input_write_start).count();

    const auto attention_mask_start = std::chrono::steady_clock::now();
    core::write_tensor_f32(
        graph->attention_mask,
        make_chunked_limited_with_rc_attention_bias(
            frames,
            pre_encode.valid_frames,
            left_context,
            chunk_frames,
            right_context));
    const auto attention_mask_end = std::chrono::steady_clock::now();
    streaming_state_.timings.encoder_attention_mask_ms +=
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(attention_mask_end - attention_mask_start).count();

    const auto keep_mask_start = std::chrono::steady_clock::now();
    core::write_tensor_i32(graph->keep_mask, make_keep_mask_i32(frames, pre_encode.valid_frames));
    const auto keep_mask_end = std::chrono::steady_clock::now();
    streaming_state_.timings.encoder_keep_mask_ms +=
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(keep_mask_end - keep_mask_start).count();

    const auto graph_compute_start = std::chrono::steady_clock::now();
    engine::core::compute_backend_graph(execution_context().backend(), graph->graph);
    const auto graph_compute_end = std::chrono::steady_clock::now();
    streaming_state_.timings.encoder_graph_compute_ms +=
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(graph_compute_end - graph_compute_start).count();

    const auto projected_read_start = std::chrono::steady_clock::now();
    core::read_tensor_f32_into(graph->output_projected.tensor, streaming_chunk_encoder_.projected_output_scratch);
    const auto projected_read_end = std::chrono::steady_clock::now();
    streaming_state_.timings.encoder_projected_read_ms +=
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(projected_read_end - projected_read_start).count();
    if (pre_encode.valid_frames < frames) {
        const auto output_zero_start = std::chrono::steady_clock::now();
        for (int64_t row = std::max<int64_t>(pre_encode.valid_frames, 0); row < frames; ++row) {
            std::fill_n(
                streaming_chunk_encoder_.projected_output_scratch.begin() + static_cast<std::ptrdiff_t>(row * hidden_out),
                static_cast<std::ptrdiff_t>(hidden_out),
                0.0f);
        }
        const auto output_zero_end = std::chrono::steady_clock::now();
        streaming_state_.timings.encoder_output_zero_ms +=
            std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(output_zero_end - output_zero_start).count();
    }
    return streaming_chunk_encoder_.projected_output_scratch;
}

}  // namespace engine::models::parakeet_tdt
