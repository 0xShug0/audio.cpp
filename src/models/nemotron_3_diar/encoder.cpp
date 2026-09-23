#include "engine/models/nemotron_3_diar/encoder.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/grouped_query_attention.h"
#include "engine/framework/modules/positional_modules.h"
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

core::TensorValue build_attention(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & mask,
    const core::TensorValue & cos,
    const core::TensorValue & sin,
    const EncoderLayerWeights & weights,
    const EncoderConfig & config) {
    const int64_t head_dim = config.hidden_size / config.heads;
    auto qkv = modules::LinearModule({config.hidden_size, 3 * config.hidden_size, false})
                   .build(ctx, input, {*weights.attention.qkv_weight, std::nullopt});
    qkv = core::ensure_backend_addressable_layout(ctx, qkv);
    auto make_heads = [&](int64_t offset) {
        auto value = modules::SliceModule({2, offset, config.hidden_size}).build(ctx, qkv);
        value = core::ensure_backend_addressable_layout(ctx, value);
        value = core::reshape_tensor(
            ctx, value,
            core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], config.heads, head_dim}));
        return modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, value);
    };
    auto query = modules::SplitRoPEModule({head_dim}).build(ctx, make_heads(0), cos, sin);
    auto key = modules::SplitRoPEModule({head_dim}).build(ctx, make_heads(config.hidden_size), cos, sin);
    auto value = make_heads(2 * config.hidden_size);
    auto context = modules::GroupedQueryAttentionModule({
        head_dim,
        modules::GroupedQueryAttentionLowering::FlashGroupedViewKV,
        GGML_PREC_F32,
        modules::AttentionCausality::NonCausal,
    }).build(ctx, query, key, value, mask);
    context = core::ensure_backend_addressable_layout(ctx, context);
    context = core::reshape_tensor(
        ctx, context,
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], config.hidden_size}));
    return modules::LinearModule({config.hidden_size, config.hidden_size, true})
        .build(ctx, context, {weights.attention.out_weight, weights.attention.out_bias});
}

core::TensorValue build_encoder_layer(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & mask,
    const core::TensorValue & cos,
    const core::TensorValue & sin,
    const EncoderLayerWeights & weights,
    const EncoderConfig & config) {
    auto normalized = modules::LayerNormModule({config.hidden_size, config.layer_norm_eps, true, true})
                          .build(ctx, input, weights.norm1);
    auto output = modules::AddModule().build(
        ctx, input, build_attention(ctx, normalized, mask, cos, sin, weights, config));
    normalized = modules::LayerNormModule({config.hidden_size, config.layer_norm_eps, true, true})
                     .build(ctx, output, weights.norm2);
    auto feed_forward = modules::LinearModule({config.hidden_size, config.intermediate_size, true})
                            .build(ctx, normalized, weights.ffn_in);
    feed_forward = modules::GeluModule({modules::GeluApproximation::ExactErf}).build(ctx, feed_forward);
    feed_forward = modules::LinearModule({config.intermediate_size, config.hidden_size, true})
                       .build(ctx, feed_forward, weights.ffn_out);
    return modules::AddModule().build(ctx, output, feed_forward);
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
EncoderGraph::~EncoderGraph() { release_graph(backend, ggml, graph, plan, allocator); }

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
    std::unique_ptr<EncoderGraph> & graph,
    const core::ExecutionContext & execution,
    const Assets & assets,
    const ModelWeights & weights,
    size_t arena_bytes,
    int64_t batch,
    int64_t frames) {
    if (graph != nullptr && graph->batch == batch && graph->frames == frames && graph->backend == execution.backend()) return;
    graph.reset();
    auto next = std::make_unique<EncoderGraph>();
    next->batch = batch;
    next->frames = frames;
    next->backend = execution.backend();
    next->threads = std::max(1, execution.config().threads);
    next->ggml = ggml_init({arena_bytes, nullptr, true});
    if (next->ggml == nullptr) throw std::runtime_error("failed to initialize Nemotron encoder context");
    core::ModuleBuildContext ctx{
        next->ggml, "nemotron_3_diar.encoder", execution.backend_type()};
    const auto & config = assets.model_config;
    const int64_t head_dim = config.encoder.hidden_size / config.encoder.heads;
    next->input = core::make_tensor(
        ctx, GGML_TYPE_F32,
        core::TensorShape::from_dims({batch, frames, config.encoder.hidden_size}));
    next->attention_mask = core::make_tensor(
        ctx, GGML_TYPE_F16,
        core::TensorShape::from_dims({batch, 1, frames, frames}));
    next->rope_cos = core::make_tensor(
        ctx, GGML_TYPE_F32,
        core::TensorShape::from_dims({1, config.encoder.heads, frames, head_dim / 2}));
    next->rope_sin = core::make_tensor(
        ctx, GGML_TYPE_F32,
        core::TensorShape::from_dims({1, config.encoder.heads, frames, head_dim / 2}));

    auto output = modules::LayerNormModule({
        config.encoder.hidden_size, config.encoder.layer_norm_eps, true, true,
    }).build(ctx, next->input, weights.embed_norm);
    for (size_t index = 0; index < weights.layers.size(); ++index) {
        output = build_encoder_layer(
            ctx, output, next->attention_mask, next->rope_cos, next->rope_sin,
            weights.layers[index], config.encoder);
    }
    output = modules::LayerNormModule({
        config.encoder.hidden_size, config.encoder.layer_norm_eps, true, true,
    }).build(ctx, output, weights.final_norm);
    output = modules::LinearModule({config.encoder.hidden_size, config.head.hidden_size, true})
                 .build(ctx, output, weights.encoder_projection);
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
    next->probabilities = modules::SigmoidModule().build(ctx, output);
    ggml_set_input(next->input.tensor);
    ggml_set_input(next->attention_mask.tensor);
    // These tables are populated once and must survive every graph execution.
    ggml_set_input(next->rope_cos.tensor);
    ggml_set_input(next->rope_sin.tensor);
    ggml_set_output(next->rope_cos.tensor);
    ggml_set_output(next->rope_sin.tensor);
    ggml_set_output(next->probabilities.tensor);
    next->graph = ggml_new_graph_custom(next->ggml, kGraphNodes, false);
    ggml_build_forward_expand(next->graph, next->probabilities.tensor);
    allocate_graph(execution, next->graph, next->allocator, next->plan);
    core::write_tensor_f32(
        next->rope_cos,
        rope_table(1, config.encoder.heads, frames, head_dim, config.encoder.rope_theta, true));
    core::write_tensor_f32(
        next->rope_sin,
        rope_table(1, config.encoder.heads, frames, head_dim, config.encoder.rope_theta, false));
    graph = std::move(next);
}

std::vector<float> rope_table(
    int64_t batch,
    int64_t heads,
    int64_t frames,
    int64_t head_dim,
    float theta,
    bool cosine) {
    const int64_t half = head_dim / 2;
    std::vector<float> values(static_cast<size_t>(batch * heads * frames * half));
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t h = 0; h < heads; ++h) {
            for (int64_t t = 0; t < frames; ++t) {
                for (int64_t i = 0; i < half; ++i) {
                    const double frequency = std::pow(static_cast<double>(theta), -2.0 * i / head_dim);
                    const double phase = static_cast<double>(t) * frequency;
                    const size_t index = static_cast<size_t>(((b * heads + h) * frames + t) * half + i);
                    values[index] = static_cast<float>(cosine ? std::cos(phase) : std::sin(phase));
                }
            }
        }
    }
    return values;
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

}  // namespace engine::models::nemotron_3_diar
