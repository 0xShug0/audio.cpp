#include "engine/models/parakeet_tdt/session.h"

#include "engine/framework/core/backend.h"
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

constexpr size_t kEncoderGraphBytes = 256 * 1024 * 1024;
constexpr size_t kEncoderGraphNodes = 1048576;

}  // namespace

struct ParakeetTDTSession::FullContextEncoderGraph {
    int64_t frames = 0;
    int64_t pos_frames = 0;
    int64_t applied_position_cache_generation = -1;
    int64_t applied_mask_cache_generation = -1;
    int64_t applied_attention_cache_generation = -1;

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

    core::TensorValue layer0_ff1;
    core::TensorValue layer0_attn;
    core::TensorValue layer0_conv;
    core::TensorValue layer0_ff2;
    core::TensorValue layer0_output;

    core::TensorValue layer12_ff1;
    core::TensorValue layer12_attn;
    core::TensorValue layer12_conv;
    core::TensorValue layer12_ff2;
    core::TensorValue layer12_output;

    core::TensorValue layer23_ff1;
    core::TensorValue layer23_attn;
    core::TensorValue layer23_conv;
    core::TensorValue layer23_ff2;
    core::TensorValue layer23_output;

    core::TensorValue output_hidden;
    core::TensorValue output_projected;

    ~FullContextEncoderGraph() {
        if (buffer != nullptr) {
            ggml_backend_buffer_free(buffer);
        }
        if (ggml != nullptr) {
            ggml_free(ggml);
        }
    }
};

ParakeetTDTSession::FullContextEncoderState::FullContextEncoderState() = default;
ParakeetTDTSession::FullContextEncoderState::~FullContextEncoderState() = default;
ParakeetTDTSession::FullContextEncoderState::FullContextEncoderState(FullContextEncoderState &&) noexcept = default;
ParakeetTDTSession::FullContextEncoderState &
ParakeetTDTSession::FullContextEncoderState::operator=(FullContextEncoderState &&) noexcept = default;

ParakeetTDTSession::~ParakeetTDTSession() = default;

void ParakeetTDTSession::ensure_full_context_encoder_graph(FullContextEncoderState & state, int64_t frames) {
    if (frames <= 0) {
        throw std::runtime_error("Parakeet encoder graph requires positive frame count");
    }
    if (state.graph != nullptr &&
        state.graph->backend == execution_context().backend() &&
        state.graph->frames == frames) {
        return;
    }

    auto graph = std::make_unique<FullContextEncoderGraph>();
    graph->frames = frames;
    graph->pos_frames = frames;
    graph->backend = execution_context().backend();

    ggml_init_params params = {};
    params.mem_size = kEncoderGraphBytes;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    graph->ggml = ggml_init(params);
    if (graph->ggml == nullptr) {
        throw std::runtime_error("Failed to initialize Parakeet encoder ggml context");
    }

    core::ModuleBuildContext ctx = {};
    ctx.ggml = graph->ggml;
    ctx.module_instance_name = "parakeet_tdt_encoder";

    const int64_t hidden = assets_->model_config.encoder.hidden_size;
    const int64_t heads = assets_->model_config.encoder.num_attention_heads;
    const int64_t intermediate = assets_->model_config.encoder.intermediate_size;
    const int64_t kernel = assets_->model_config.encoder.conv_kernel_size;
    const int64_t num_layers = assets_->model_config.encoder.num_hidden_layers;
    const int64_t decoder_hidden = assets_->model_config.decoder_hidden_size;
    const auto & encoder_weights = weights_->encoder;

    graph->input_hidden = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, frames, hidden}));
    graph->pos_emb = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 2 * graph->pos_frames - 1, hidden}));
    graph->attention_mask = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({frames, frames}));
    graph->keep_mask = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({1, frames}));
    graph->projected_pos_emb.reserve(static_cast<size_t>(num_layers));
    graph->projected_pos_emb_computed.reserve(static_cast<size_t>(num_layers));
    for (int64_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
        const auto & layer_weights = encoder_weights.layers[static_cast<size_t>(layer_idx)];
        graph->projected_pos_emb.push_back(
            core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 2 * graph->pos_frames - 1, hidden})));
        graph->projected_pos_emb_computed.push_back(
            modules::LinearModule({hidden, hidden, false}).build(
                ctx,
                graph->pos_emb,
                {layer_weights.self_attn.pos_weight, std::nullopt}));
    }

    modules::AddModule add;
    modules::SiluModule silu;
    auto x = graph->input_hidden;
    const int64_t mid_layer = num_layers / 2;
    const int64_t last_layer = num_layers - 1;

    for (int64_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
        const auto & layer_weights = encoder_weights.layers[static_cast<size_t>(layer_idx)];
        const bool tapped = layer_idx == 0 || layer_idx == mid_layer || layer_idx == last_layer;

        if (!tapped) {
            x = modules::RelativeConformerBlockModule({
                hidden,
                heads,
                intermediate,
                kernel,
                1e-5f,
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
            continue;
        }

        auto ff1_in = modules::LayerNormModule({hidden, 1e-5f, true, true}).build(ctx, x, layer_weights.norm_feed_forward1);
        auto ff1 = modules::LinearModule({hidden, intermediate, false}).build(ctx, ff1_in, layer_weights.ff1_linear1);
        ff1 = silu.build(ctx, ff1);
        ff1 = modules::LinearModule({intermediate, hidden, false}).build(ctx, ff1, layer_weights.ff1_linear2);
        auto ff1_half = core::wrap_tensor(ggml_scale(ctx.ggml, ff1.tensor, 0.5f), ff1.shape, GGML_TYPE_F32);
        x = add.build(ctx, x, ff1_half);

        auto attn_in = modules::LayerNormModule({hidden, 1e-5f, true, true}).build(ctx, x, layer_weights.norm_self_att);
        auto attn = modules::RelativeSelfAttentionModule({
            hidden,
            heads,
            false,
            -1,
            -1,
            0,
        }).build(
            ctx,
            attn_in,
            std::nullopt,
            layer_weights.self_attn,
            graph->attention_mask,
            std::nullopt,
            graph->projected_pos_emb[static_cast<size_t>(layer_idx)]);
        x = add.build(ctx, x, attn);

        auto conv = modules::ConformerConvModule({hidden, kernel, false, 1e-5f, 0}).build(
            ctx,
            x,
            {layer_weights.norm_conv, layer_weights.conv_pointwise_conv1, layer_weights.conv_depthwise_conv, {layer_weights.conv_norm.scale, layer_weights.conv_norm.bias}, layer_weights.conv_pointwise_conv2},
            graph->keep_mask);
        x = add.build(ctx, x, conv);

        auto ff2_in = modules::LayerNormModule({hidden, 1e-5f, true, true}).build(ctx, x, layer_weights.norm_feed_forward2);
        auto ff2 = modules::LinearModule({hidden, intermediate, false}).build(ctx, ff2_in, layer_weights.ff2_linear1);
        ff2 = silu.build(ctx, ff2);
        ff2 = modules::LinearModule({intermediate, hidden, false}).build(ctx, ff2, layer_weights.ff2_linear2);
        auto ff2_half = core::wrap_tensor(ggml_scale(ctx.ggml, ff2.tensor, 0.5f), ff2.shape, GGML_TYPE_F32);
        x = add.build(ctx, x, ff2_half);
        x = modules::LayerNormModule({hidden, 1e-5f, true, true}).build(ctx, x, layer_weights.norm_out);

        if (layer_idx == 0) {
            graph->layer0_ff1 = ff1;
            graph->layer0_attn = attn;
            graph->layer0_conv = conv;
            graph->layer0_ff2 = ff2;
            graph->layer0_output = x;
        } else if (layer_idx == mid_layer) {
            graph->layer12_ff1 = ff1;
            graph->layer12_attn = attn;
            graph->layer12_conv = conv;
            graph->layer12_ff2 = ff2;
            graph->layer12_output = x;
        } else {
            graph->layer23_ff1 = ff1;
            graph->layer23_attn = attn;
            graph->layer23_conv = conv;
            graph->layer23_ff2 = ff2;
            graph->layer23_output = x;
        }
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

    graph->graph = ggml_new_graph_custom(graph->ggml, kEncoderGraphNodes, false);
    ggml_build_forward_expand(graph->graph, graph->output_projected.tensor);
    graph->buffer = ggml_backend_alloc_ctx_tensors(graph->ggml, graph->backend);
    if (graph->buffer == nullptr) {
        throw std::runtime_error("Failed to allocate Parakeet encoder backend tensors");
    }

    state.graph = std::move(graph);
}

const std::vector<float> & ParakeetTDTSession::run_offline_graph_encoder(
    const ParakeetPreEncodeBatch & pre_encode,
    FullContextEncoderState & state) {
    const int64_t frames = pre_encode.frames;

    ensure_full_context_encoder_graph(state, frames);
    if (state.graph->applied_position_cache_generation != shared_encoder_.position_cache_generation &&
        shared_encoder_.cached_position_frames == frames &&
        !shared_encoder_.cached_position_long_context &&
        shared_encoder_.cached_pos_frames == frames) {
        core::write_tensor_f32(state.graph->pos_emb, shared_encoder_.pos_emb_cache);
        engine::core::compute_backend_graph(state.graph->backend, state.graph->pos_projection_graph);
        for (size_t i = 0; i < state.graph->projected_pos_emb.size(); ++i) {
            ggml_backend_tensor_copy(
                state.graph->projected_pos_emb_computed[i].tensor,
                state.graph->projected_pos_emb[i].tensor);
        }
        state.graph->applied_position_cache_generation = shared_encoder_.position_cache_generation;
    }
    if (state.graph->applied_attention_cache_generation != state.attention_cache_generation &&
        state.cached_attention_frames == frames &&
        state.cached_attention_valid_frames == pre_encode.valid_frames) {
        core::write_tensor_f32(state.graph->attention_mask, state.attention_bias_cache);
        state.graph->applied_attention_cache_generation = state.attention_cache_generation;
    }
    if (state.graph->applied_mask_cache_generation != shared_encoder_.mask_cache_generation &&
        shared_encoder_.cached_mask_frames == frames &&
        shared_encoder_.cached_mask_valid_frames == pre_encode.valid_frames) {
        core::write_tensor_i32(state.graph->keep_mask, shared_encoder_.keep_mask_cache);
        state.graph->applied_mask_cache_generation = shared_encoder_.mask_cache_generation;
    }

    core::write_tensor_f32(state.graph->input_hidden, pre_encode.hidden);
    engine::core::compute_backend_graph(execution_context().backend(), state.graph->graph);

    core::read_tensor_f32_into(state.graph->output_projected.tensor, shared_encoder_.projected_output_scratch);
    if (pre_encode.valid_frames < frames) {
        const int64_t hidden_out = assets_->model_config.decoder_hidden_size;
        for (int64_t row = std::max<int64_t>(pre_encode.valid_frames, 0); row < frames; ++row) {
            std::fill_n(
                shared_encoder_.projected_output_scratch.begin() + static_cast<std::ptrdiff_t>(row * hidden_out),
                static_cast<std::ptrdiff_t>(hidden_out),
                0.0f);
        }
    }
    return shared_encoder_.projected_output_scratch;
}

}  // namespace engine::models::parakeet_tdt
