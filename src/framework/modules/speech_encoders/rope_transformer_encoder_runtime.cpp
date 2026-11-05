#include "engine/framework/modules/speech_encoders/rope_transformer_encoder_runtime.h"

#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/grouped_query_attention.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <ggml-alloc.h>

#include <algorithm>
#include <numeric>
#include <cmath>
#include <stdexcept>

namespace engine::modules {

RopeTransformerEncoderModule::RopeTransformerEncoderModule(RopeTransformerEncoderConfig config)
    : config_(config) {
    if (config.hidden_size <= 0 || config.intermediate_size <= 0 || config.heads <= 0 ||
        config.hidden_size % config.heads != 0 || config.rotary_dimensions <= 0 ||
        config.rotary_dimensions % 2 != 0 || config.rotary_dimensions > config.hidden_size / config.heads ||
        !(config.norm_epsilon > 0) || !(config.rope_theta > 0)) {
        throw std::runtime_error("invalid RoPE transformer encoder dimensions or normalization");
    }
}

core::TensorValue RopeTransformerEncoderModule::build(core::ModuleBuildContext & ctx,
    const core::TensorValue & input, const RopeTransformerEncoderInputs & inputs,
    const RopeTransformerEncoderWeights & weights) const {
    core::validate_rank_between(input, 3, 3, "encoder input");
    core::validate_last_dim(input, config_.hidden_size, "encoder input");
    const auto batch = input.shape.at(0), frames = input.shape.at(1);
    const auto d = config_.hidden_size, heads = config_.heads, dk = d / heads;
    auto x = input;
    if (config_.input_norm) {
        x = LayerNormModule({d, config_.norm_epsilon, true, true}).build(ctx, x, weights.input_norm);
    }
    for (const auto & layer : weights.layers) {
        auto normalized = LayerNormModule({d, config_.norm_epsilon, true, true}).build(ctx, x, layer.norm1);
        auto qkv = LinearModule({d, 3 * d, layer.qkv.bias.has_value(), config_.projection_precision})
            .build(ctx, normalized, layer.qkv);
        core::TensorValue attended;
        if (config_.rotary == EncoderRoPE::SplitTables) {
            qkv = core::ensure_backend_addressable_layout(ctx, qkv);
            const auto split = [&](int64_t offset) {
                auto value = SliceModule({2, offset, d}).build(ctx, qkv);
                value = core::ensure_backend_addressable_layout(ctx, value);
                value = core::reshape_tensor(ctx, value, core::TensorShape::from_dims({batch, frames, heads, dk}));
                return TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, value);
            };
            auto q = split(0), k = split(d), v = split(2 * d);
            if (config_.qk_norm) {
                q = LayerNormModule({dk, config_.norm_epsilon, true, true}).build(ctx, q, layer.q_norm);
                k = LayerNormModule({dk, config_.norm_epsilon, true, true}).build(ctx, k, layer.k_norm);
            }
            q = SplitRoPEModule({config_.rotary_dimensions}).build(ctx, q, inputs.cosine, inputs.sine);
            k = SplitRoPEModule({config_.rotary_dimensions}).build(ctx, k, inputs.cosine, inputs.sine);
            if (config_.f16_kv) {
                k = core::wrap_tensor(ggml_cast(ctx.ggml, k.tensor, GGML_TYPE_F16), k.shape, GGML_TYPE_F16);
                v = core::wrap_tensor(ggml_cast(ctx.ggml, v.tensor, GGML_TYPE_F16), v.shape, GGML_TYPE_F16);
            }
            attended = GroupedQueryAttentionModule({dk,
                config_.flash_attention ? GroupedQueryAttentionLowering::FlashGroupedViewKV
                                        : GroupedQueryAttentionLowering::ManualRepeat,
                GGML_PREC_F32, AttentionCausality::NonCausal}).build(ctx, q, k, v, inputs.attention_mask);
            attended = core::ensure_backend_addressable_layout(ctx, attended);
        } else {
            // Indexed RoPE operates on [dimension, heads, time, batch] views.
            // Preserve their frame stride rather than materializing each split.
            const auto split = [&](int offset) {
                auto * value = ggml_view_4d(ctx.ggml, qkv.tensor, dk, heads, frames, batch,
                    dk * sizeof(float), qkv.tensor->nb[1], d * 3 * frames * sizeof(float),
                    offset * d * sizeof(float));
                return core::wrap_tensor(value, core::TensorShape::from_dims({batch, frames, heads, dk}));
            };
            auto q = split(0), k = split(1), v = split(2);
            if (config_.qk_norm) {
                q = LayerNormModule({dk, config_.norm_epsilon, true, true}).build(ctx, q, layer.q_norm);
                k = LayerNormModule({dk, config_.norm_epsilon, true, true}).build(ctx, k, layer.k_norm);
            }
            const RoPEModule rope({config_.rotary_dimensions, GGML_ROPE_TYPE_NEOX, config_.rope_theta});
            q = rope.build(ctx, q, inputs.positions);
            k = rope.build(ctx, k, inputs.positions);
            auto * qp = ggml_permute(ctx.ggml, q.tensor, 0, 2, 1, 3);
            auto * kp = ggml_permute(ctx.ggml, k.tensor, 0, 2, 1, 3);
            auto * vp = ggml_permute(ctx.ggml, v.tensor, 0, 2, 1, 3);
            if (config_.f16_kv) {
                kp = ggml_cast(ctx.ggml, kp, GGML_TYPE_F16);
                vp = ggml_cast(ctx.ggml, vp, GGML_TYPE_F16);
            }
            auto * mask = inputs.attention_mask ? inputs.attention_mask->tensor : nullptr;
            const float scale = 1.0f / std::sqrt(static_cast<float>(dk));
            ggml_tensor * result;
            if (config_.flash_attention) {
                result = ggml_flash_attn_ext(ctx.ggml, qp, kp, vp, mask, scale, 0.0f, 0.0f);
                ggml_flash_attn_ext_set_prec(result, GGML_PREC_F32);
            } else {
                auto * scores = ggml_mul_mat(ctx.ggml, kp, qp);
                ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
                auto * probabilities = ggml_soft_max_ext(ctx.ggml, scores, mask, scale, 0.0f);
                auto * values = ggml_cont(ctx.ggml, ggml_permute(ctx.ggml, vp, 1, 0, 2, 3));
                result = ggml_mul_mat(ctx.ggml, values, probabilities);
                ggml_mul_mat_set_prec(result, GGML_PREC_F32);
                result = ggml_cont(ctx.ggml, ggml_permute(ctx.ggml, result, 0, 2, 1, 3));
            }
            attended = core::wrap_tensor(result, core::TensorShape::from_dims({batch, frames, heads, dk}));
        }
        attended = core::reshape_tensor(ctx, attended, input.shape);
        auto residual = LinearModule({d, d, layer.attention_output.bias.has_value(), config_.projection_precision})
            .build(ctx, attended, layer.attention_output);
        x = AddModule().build(ctx, x, residual);
        normalized = LayerNormModule({d, config_.norm_epsilon, true, true}).build(ctx, x, layer.norm2);
        auto hidden = LinearModule({d, config_.intermediate_size, layer.ffn_in.bias.has_value(), config_.projection_precision})
            .build(ctx, normalized, layer.ffn_in);
        hidden = GeluModule({GeluApproximation::ExactErf}).build(ctx, hidden);
        hidden = LinearModule({config_.intermediate_size, d, layer.ffn_out.bias.has_value(), config_.projection_precision})
            .build(ctx, hidden, layer.ffn_out);
        x = AddModule().build(ctx, x, hidden);
    }
    return LayerNormModule({d, config_.norm_epsilon, true, true}).build(ctx, x, weights.final_norm);
}

struct RopeTransformerEncoderRuntime::Impl {
    struct Graph {
        ggml_backend_t backend;
        ggml_context * context = nullptr;
        ggml_cgraph * graph = nullptr;
        ggml_backend_graph_plan_t plan = nullptr;
        ggml_gallocr_t allocator = nullptr;
        int64_t batch = 0, frames = 0;
        core::TensorValue input;
        RopeTransformerEncoderInputs rotary;
        std::vector<core::TensorValue> outputs;
        explicit Graph(ggml_backend_t backend_in) : backend(backend_in) {}
        ~Graph() {
            if (plan) core::free_backend_graph_plan(backend, plan);
            if (graph) core::release_backend_graph_resources(backend, graph, true);
            if (allocator) ggml_gallocr_free(allocator);
            if (context) ggml_free(context);
        }
    };
    const core::ExecutionContext & execution;
    RopeTransformerEncoderConfig config;
    RopeTransformerEncoderWeights weights;
    RopeTransformerEncoderModule module;
    RopeTransformerEncoderGraphConfig graph_config;
    size_t builds = 0;
    std::unique_ptr<Graph> graph;

    Impl(const core::ExecutionContext & e, RopeTransformerEncoderConfig c,
        RopeTransformerEncoderWeights w, RopeTransformerEncoderGraphConfig g)
        : execution(e), config(c), weights(std::move(w)), module(c), graph_config(std::move(g)) {}

    void prepare(int64_t batch, int64_t frames) {
        if (batch <= 0 || frames <= 0) throw std::invalid_argument("encoder batch and frames must be positive");
        if (graph && graph->batch == batch && graph->frames == frames) return;
        graph.reset();
        auto next = std::make_unique<Graph>(execution.backend());
        next->batch = batch;
        next->frames = frames;
        const size_t nodes = std::max<size_t>(32768, 512 + weights.layers.size() * 512);
        const size_t bytes = graph_config.arena_bytes ? graph_config.arena_bytes :
            ggml_tensor_overhead() * nodes * 2 + ggml_graph_overhead_custom(nodes, false);
        next->context = ggml_init({bytes, nullptr, true});
        if (!next->context) throw std::runtime_error("RoPE encoder context allocation failed");
        core::ModuleBuildContext ctx{next->context, "rope_transformer_encoder", execution.backend_type()};
        const auto & projection = graph_config.input_projection;
        const auto width = projection ? projection->weight.shape.last_dim() : config.hidden_size;
        next->input = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch, frames, width}));
        ggml_set_input(next->input.tensor);
        if (graph_config.masked_attention) {
            next->rotary.attention_mask = core::make_tensor(ctx, GGML_TYPE_F16,
                core::TensorShape::from_dims({batch, 1, frames, frames}));
            ggml_set_input(next->rotary.attention_mask->tensor);
        }
        if (!weights.layers.empty()) {
            if (config.rotary == EncoderRoPE::Indexed) {
                next->rotary.positions = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({frames}));
                ggml_set_input(next->rotary.positions.tensor);
                ggml_set_output(next->rotary.positions.tensor);
            } else {
                const auto shape = core::TensorShape::from_dims({1, config.heads, frames, config.rotary_dimensions / 2});
                next->rotary.cosine = core::make_tensor(ctx, GGML_TYPE_F32, shape);
                next->rotary.sine = core::make_tensor(ctx, GGML_TYPE_F32, shape);
                for (auto value : {next->rotary.cosine, next->rotary.sine}) {
                    ggml_set_input(value.tensor);
                    ggml_set_output(value.tensor);
                }
            }
        }
        auto input = next->input;
        if (projection) input = LinearModule({width, config.hidden_size, projection->bias.has_value(),
            config.projection_precision}).build(ctx, input, *projection);
        auto encoded = module.build(ctx, input, next->rotary, weights);
        next->outputs = graph_config.head ? graph_config.head(ctx, encoded) : std::vector<core::TensorValue>{encoded};
        if (next->outputs.empty()) throw std::invalid_argument("encoder graph requires an output");
        next->graph = ggml_new_graph_custom(next->context, nodes, false);
        for (const auto & output : next->outputs) {
            ggml_set_output(output.tensor);
            ggml_build_forward_expand(next->graph, output.tensor);
        }
        core::validate_backend_graph_supported(execution.backend(), next->graph, "RoPE transformer encoder");
        next->allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend()));
        if (!ggml_gallocr_alloc_graph(next->allocator, next->graph))
            throw std::runtime_error("RoPE encoder graph allocation failed");
        if (execution.uses_host_graph_plan()) {
            next->plan = core::create_backend_graph_plan_if_host(execution.backend(), next->graph);
            if (!next->plan) throw std::runtime_error("RoPE encoder host plan allocation failed");
        }
        if (next->rotary.positions.valid()) {
            std::vector<int32_t> positions(static_cast<size_t>(frames));
            std::iota(positions.begin(), positions.end(), 0);
            core::write_tensor_i32(next->rotary.positions, positions.data(), positions.size());
        }
        if (next->rotary.cosine.valid()) {
            const auto half = config.rotary_dimensions / 2;
            std::vector<float> cosine(static_cast<size_t>(config.heads * frames * half)), sine(cosine.size());
            for (int64_t h = 0; h < config.heads; ++h)
                for (int64_t t = 0; t < frames; ++t)
                    for (int64_t i = 0; i < half; ++i) {
                        const double frequency = std::pow(static_cast<double>(config.rope_theta),
                            -2.0 * i / config.rotary_dimensions);
                        const double phase = static_cast<double>(t) * frequency;
                        const auto index = static_cast<size_t>((h * frames + t) * half + i);
                        cosine[index] = static_cast<float>(std::cos(phase));
                        sine[index] = static_cast<float>(std::sin(phase));
                    }
            core::write_tensor_f32(next->rotary.cosine, cosine);
            core::write_tensor_f32(next->rotary.sine, sine);
        }
        graph = std::move(next);
        ++builds;
    }
};

RopeTransformerEncoderRuntime::RopeTransformerEncoderRuntime(const core::ExecutionContext & execution,
    RopeTransformerEncoderConfig config, RopeTransformerEncoderWeights weights,
    RopeTransformerEncoderGraphConfig graph_config)
    : impl_(std::make_unique<Impl>(execution, config, std::move(weights), std::move(graph_config))) {}
RopeTransformerEncoderRuntime::~RopeTransformerEncoderRuntime() = default;
void RopeTransformerEncoderRuntime::prepare(int64_t batch, int64_t frames) { impl_->prepare(batch, frames); }
const core::TensorValue & RopeTransformerEncoderRuntime::input() const {
    if (!impl_->graph) throw std::logic_error("encoder runtime is not prepared");
    return impl_->graph->input;
}
const core::TensorValue & RopeTransformerEncoderRuntime::attention_mask() const {
    if (!impl_->graph) throw std::logic_error("encoder runtime is not prepared");
    return impl_->graph->rotary.attention_mask.value();
}
const core::TensorValue & RopeTransformerEncoderRuntime::output(size_t index) const {
    if (!impl_->graph) throw std::logic_error("encoder runtime is not prepared");
    return impl_->graph->outputs.at(index);
}
void RopeTransformerEncoderRuntime::run() {
    if (!impl_->graph) throw std::logic_error("encoder runtime is not prepared");
    core::set_backend_threads(impl_->execution.backend(), impl_->execution.config().threads);
    if (core::compute_backend_graph(impl_->execution.backend(), impl_->graph->graph,
            impl_->graph->plan, "RoPE transformer encoder") != GGML_STATUS_SUCCESS)
        throw std::runtime_error("RoPE encoder graph execution failed");
}
void RopeTransformerEncoderRuntime::release_workspace() { impl_->graph.reset(); }
size_t RopeTransformerEncoderRuntime::graphs_built() const noexcept { return impl_->builds; }

} // namespace engine::modules
