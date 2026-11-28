#include "engine/models/nemotron_3_diar/encoder.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace engine::models::nemotron_3_diar {
namespace {

constexpr size_t kGraphNodes = 1048576;

void release_graph(
    ggml_backend_t backend,
    ggml_context * context,
    ggml_cgraph * graph,
    ggml_backend_graph_plan_t plan,
    ggml_gallocr_t allocator) {
    if (plan != nullptr) core::free_backend_graph_plan(backend, plan);
    if (graph != nullptr) core::release_backend_graph_resources(backend, graph, true);
    if (allocator != nullptr) ggml_gallocr_free(allocator);
    if (context != nullptr) ggml_free(context);
}

void allocate_graph(
    const core::ExecutionContext & execution,
    ggml_cgraph * graph,
    ggml_gallocr_t & allocator,
    ggml_backend_graph_plan_t & plan) {
    allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend()));
    if (!ggml_gallocr_alloc_graph(allocator, graph)) {
        throw std::runtime_error("failed to allocate Nemotron diarization graph tensors");
    }
    debug::trace_log_scalar("nemotron_3_diar.graph_buffer_bytes", ggml_gallocr_get_buffer_size(allocator, 0));
    if (execution.uses_host_graph_plan()) {
        plan = core::create_backend_graph_plan_if_host(execution.backend(), graph);
        if (plan == nullptr) throw std::runtime_error("failed to create Nemotron diarization graph plan");
    }
}

}  // namespace

PreEncodeGraph::~PreEncodeGraph() { release_graph(backend, ggml, graph, plan, allocator); }

void ensure_pre_encode_graph(
    std::unique_ptr<PreEncodeGraph> & graph,
    const core::ExecutionContext & execution,
    const Assets & assets,
    const ModelWeights & weights,
    size_t arena_bytes,
    int64_t batch,
    int64_t frames) {
    if (graph != nullptr && graph->batch == batch && graph->frames == frames && graph->backend == execution.backend()) return;
    graph.reset();
    auto next = std::make_unique<PreEncodeGraph>();
    next->batch = batch;
    next->frames = frames;
    next->backend = execution.backend();
    next->threads = std::max(1, execution.config().threads);
    next->ggml = ggml_init({arena_bytes, nullptr, true});
    if (next->ggml == nullptr) throw std::runtime_error("failed to initialize Nemotron pre-encode context");
    core::ModuleBuildContext ctx{
        next->ggml, "nemotron_3_diar.pre_encode", execution.backend_type()};
    const auto & config = assets.model_config.encoder;
    next->input = core::make_tensor(
        ctx, GGML_TYPE_F32,
        core::TensorShape::from_dims({batch, frames, config.feature_size * config.subsampling_factor}));
    next->output = modules::LinearModule({
        config.feature_size * config.subsampling_factor, config.hidden_size, false,
    }).build(ctx, next->input, weights.pre_encode);
    ggml_set_input(next->input.tensor);
    ggml_set_output(next->output.tensor);
    next->graph = ggml_new_graph_custom(next->ggml, kGraphNodes, false);
    ggml_build_forward_expand(next->graph, next->output.tensor);
    allocate_graph(execution, next->graph, next->allocator, next->plan);
    graph = std::move(next);
}

void ensure_encoder_graph(
    std::unique_ptr<Nemotron3DiarTransformerEncoderGraph> & graph,
    const core::ExecutionContext & execution,
    const Assets & assets,
    const ModelWeights & weights,
    size_t arena_bytes,
    int64_t batch,
    int64_t frames,
    bool use_flash_attention) {
    if (!graph) {
        auto next = std::make_unique<Nemotron3DiarTransformerEncoderGraph>();
        const auto config = assets.model_config;
        modules::RopeTransformerEncoderConfig encoder;
        encoder.hidden_size = config.encoder.hidden_size;
        encoder.intermediate_size = config.encoder.intermediate_size;
        encoder.heads = config.encoder.heads;
        encoder.rotary_dimensions = config.encoder.hidden_size / config.encoder.heads;
        encoder.norm_epsilon = config.encoder.layer_norm_eps;
        encoder.rope_theta = config.encoder.rope_theta;
        encoder.rotary = modules::EncoderRoPE::SplitTables;
        encoder.flash_attention = use_flash_attention;
        auto head = [state = next.get(), config, &weights](core::ModuleBuildContext & ctx,
            const core::TensorValue & encoded) {
            const auto batch = encoded.shape.at(0), frames = encoded.shape.at(1);
            state->frame_mask = core::make_tensor(ctx, GGML_TYPE_F32,
                core::TensorShape::from_dims({batch, frames, 1}));
            ggml_set_input(state->frame_mask.tensor);
            auto output = encoded;
            output = modules::LinearModule({config.encoder.hidden_size, config.head.hidden_size, true})
                         .build(ctx, output, weights.encoder_projection);
            // Zero padded frames so the subpixel convolution at the last valid
            // frame sees zero padding, as in an unpadded forward.
            output = core::wrap_tensor(
                ggml_mul(ctx.ggml, output.tensor, state->frame_mask.tensor), output.shape, GGML_TYPE_F32);
            output = modules::TransposeModule({{0, 2, 1}, 3}).build(ctx, output);
            output = modules::Conv1dModule({
                config.head.hidden_size,
                config.head.hidden_size * config.head.upsample_factor,
                config.head.upsample_kernel_size,
                1,
                static_cast<int>(config.head.upsample_kernel_size / 2),
                1,
                true,
            }).build(ctx, output, weights.subpixel_upsample);
            output = modules::TransposeModule({{0, 2, 1}, 3}).build(ctx, output);
            output = core::ensure_backend_addressable_layout(ctx, output);
            output = core::reshape_tensor(
                ctx, output,
                core::TensorShape::from_dims({
                    batch, frames * config.head.upsample_factor, config.head.hidden_size,
                }));
            output = modules::ReluModule().build(ctx, output);
            output = modules::LinearModule({config.head.hidden_size, config.head.hidden_size, true})
                         .build(ctx, output, weights.head_hidden);
            output = modules::ReluModule().build(ctx, output);
            output = modules::LinearModule({config.head.hidden_size, config.num_speakers, true})
                         .build(ctx, output, weights.speaker_head);
            output = modules::SigmoidModule().build(ctx, output);
            return std::vector<core::TensorValue>{output};
        };
        modules::RopeTransformerEncoderGraphConfig graph_config;
        graph_config.head = std::move(head);
        graph_config.masked_attention = true;
        graph_config.arena_bytes = arena_bytes;
        next->runtime = std::make_unique<modules::RopeTransformerEncoderRuntime>(execution,
            encoder, weights.encoder, std::move(graph_config));
        graph = std::move(next);
    }
    graph->runtime->prepare(batch, frames);
}

std::vector<float> attention_mask(const std::vector<int64_t> & lengths, int64_t frames) {
    std::vector<float> values(static_cast<size_t>(lengths.size() * frames * frames), 0.0F);
    for (size_t batch = 0; batch < lengths.size(); ++batch) {
        for (int64_t query = 0; query < frames; ++query) {
            for (int64_t key = lengths[batch]; key < frames; ++key) {
                values[(batch * static_cast<size_t>(frames) + static_cast<size_t>(query)) *
                           static_cast<size_t>(frames) + static_cast<size_t>(key)] = -10000.0F;
            }
        }
    }
    return values;
}

std::vector<float> frame_mask(const std::vector<int64_t> & lengths, int64_t frames) {
    std::vector<float> values(lengths.size() * static_cast<size_t>(frames), 0.0F);
    for (size_t batch = 0; batch < lengths.size(); ++batch) {
        std::fill_n(values.begin() + static_cast<std::ptrdiff_t>(batch * static_cast<size_t>(frames)),
            std::clamp<int64_t>(lengths[batch], 0, frames), 1.0F);
    }
    return values;
}

}  // namespace engine::models::nemotron_3_diar
