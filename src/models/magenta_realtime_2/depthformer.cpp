#include "engine/models/magenta_realtime_2/depthformer.h"

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/modules/feed_forward_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/attention/streaming_frame_attention_cache.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/packed_linear_weights.h"
#include "engine/framework/modules/primitive_modules.h"

#include <ggml-alloc.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <unordered_map>
#include <unordered_set>

namespace engine::models::magenta_realtime_2 {
namespace {

using core::TensorValue;
using core::TensorShape;

struct ContextDeleter {
    void operator()(ggml_context * value) const { if (value) ggml_free(value); }
};

struct DepthformerWeights {
    core::BackendWeightStore store;
    std::unordered_map<std::string, TensorValue> tensors;

    DepthformerWeights(core::ExecutionContext & execution, const MagentaAssets & assets,
                       engine::assets::TensorStorageType storage)
        : store(execution.backend(), execution.backend_type(), "magenta_realtime_2.depthformer.weights", 1024 * 1024) {
        std::unordered_set<std::string> packed_sources;
        for (const auto & stack : {std::make_pair("temporal", assets.config.temporal_layers),
                                   std::make_pair("depth", assets.config.depth_layers)}) {
            const auto dim = std::string(stack.first) == "temporal" ? assets.config.temporal_dim : assets.config.depth_dim;
            for (int64_t i = 0; i < stack.second; ++i) {
                const auto prefix = std::string(stack.first) + "." + std::to_string(i) + ".self";
                std::vector<modules::PackedLinearSource> sources;
                for (const auto * projection : {"q", "k", "v"}) {
                    const auto name = prefix + "." + projection + ".weight";
                    sources.push_back({name, std::nullopt, dim});
                    packed_sources.insert(name);
                }
                const auto packed = modules::PackedLinearWeightsBuilder({dim, std::move(sources), false, true})
                    .build(store, *assets.weights, storage);
                tensors.emplace(prefix + ".qkv.weight", packed.weight);
            }
        }
        for (const auto & metadata : assets.weights->tensors()) {
            if (metadata.name.rfind("codec.", 0) == 0 || packed_sources.count(metadata.name)) continue;
            if (metadata.name.find("per_dim_scale") != std::string::npos) {
                auto values = assets.weights->require_f32(metadata.name);
                for (auto & value : values) value = static_cast<float>(1.442695041 / std::sqrt(128.0) * std::log1p(std::exp(value)));
                tensors.emplace(metadata.name, store.make_f32(TensorShape::from_dims({128}), std::move(values)));
            } else {
                const bool f32 = metadata.shape.size() == 1 || metadata.name.find("sink_") != std::string::npos;
                tensors.emplace(metadata.name, f32 ? store.load_f32_tensor(*assets.weights, metadata.name, metadata.shape) :
                    store.load_tensor(*assets.weights, metadata.name, storage, metadata.shape));
            }
        }
        store.upload();
    }

    const TensorValue & at(const std::string & name) const { return tensors.at(name); }

    TensorValue linear(core::ModuleBuildContext & ctx, const TensorValue & x, const std::string & name) const {
        const auto & weight = at(name + ".weight");
        const auto in = weight.shape.dims[1], out = weight.shape.dims[0];
        const auto bias = tensors.find(name + ".bias");
        const modules::LinearWeights w{weight, bias == tensors.end() ? std::nullopt : std::optional<TensorValue>(bias->second)};
        return modules::LinearModule({in, out, w.bias.has_value(), GGML_PREC_F32}).build(ctx, x, w);
    }

    TensorValue norm(core::ModuleBuildContext & ctx, const TensorValue & x, const std::string & name, bool layer = false) const {
        const auto width = x.shape.last_dim();
        modules::NormWeights w{at(name + ".weight"), std::nullopt};
        if (layer) w.bias = at(name + ".bias");
        const modules::NormConfig config{width, 1e-6F, true, layer};
        return layer ? modules::LayerNormModule(config).build(ctx, x, w) : modules::RMSNormModule(config).build(ctx, x, w);
    }

    TensorValue feed_forward(core::ModuleBuildContext & ctx, const TensorValue & x, const std::string & prefix) const {
        auto y = norm(ctx, x, prefix + ".ffn.pre_norm");
        const auto & up = at(prefix + ".ffn.up.weight");
        const auto & down = at(prefix + ".ffn.down.weight");
        y = modules::FeedForwardModule({up.shape.dims[1], up.shape.dims[0], true,
            modules::GeluApproximation::Tanh, GGML_PREC_F32}).build(ctx, y,
            {up, at(prefix + ".ffn.up.bias"), down, at(prefix + ".ffn.down.bias")});
        y = norm(ctx, y, prefix + ".ffn.post_norm");
        return modules::AddModule().build(ctx, x, y);
    }
};

struct ConditionGraph {
    core::ExecutionContext & execution;
    std::unique_ptr<ggml_context, ContextDeleter> context;
    ggml_cgraph * graph = nullptr;
    ggml_gallocr_t allocator = nullptr;
    ggml_backend_graph_plan_t host_plan = nullptr;
    ggml_tensor * style = nullptr;
    ggml_tensor * controls = nullptr;
    ggml_tensor * output = nullptr;

    ConditionGraph(core::ExecutionContext & execution, const DepthformerWeights & weights, const MagentaConfig & config)
        : execution(execution), context(ggml_init({2 * 1024 * 1024, nullptr, true})) {
        if (!context) throw std::runtime_error("Cannot allocate Magenta conditioning metadata");
        auto * c = context.get();
        core::ModuleBuildContext ctx{c, "magenta_realtime_2.condition", execution.backend_type()};
        const auto style_input = core::make_tensor(ctx, GGML_TYPE_I32, TensorShape::from_dims({12}));
        const auto control_input = core::make_tensor(ctx, GGML_TYPE_I32, TensorShape::from_dims({132}));
        style = style_input.tensor;
        controls = control_input.tensor;
        ggml_set_input(style);
        ggml_set_input(controls);
        const auto & style_weight = weights.at("condition.style.mulan_dequantizer.weight");
        auto style_emb = modules::EmbeddingModule({style_weight.shape.dims[0], 768}).build(ctx, style_input, style_weight);
        style_emb = modules::ReduceSumModule({0}).build(ctx, style_emb);
        style_emb = weights.linear(ctx, style_emb, "condition.style.depth_input_adapter");
        const auto & control_weight = weights.at("condition.controls.weight");
        auto regular = modules::EmbeddingModule({control_weight.shape.dims[0], config.condition_dim}).build(ctx, control_input, control_weight);
        regular = modules::ReduceMeanModule({0}).build(ctx, regular);
        regular = modules::ReshapeModule({TensorShape::from_dims({1, config.condition_dim})}).build(ctx, core::ensure_backend_addressable_layout(ctx, regular));
        auto encoded = modules::ScaleBiasModule({0.5F, 0.0F}).build(ctx, modules::AddModule().build(ctx, style_emb, regular));
        encoded = weights.norm(ctx, encoded, "condition.encoder_ln", true);
        TensorValue combined;
        for (int64_t i = 0; i < config.temporal_layers; ++i) {
            const auto prefix = "temporal." + std::to_string(i) + ".cross";
            for (const auto * type : {"k", "v"}) {
                auto value = weights.linear(ctx, encoded, prefix + "." + type);
                value = modules::ReshapeModule({TensorShape::from_dims({config.temporal_dim})}).build(ctx, value);
                combined = combined.valid() ? modules::ConcatModule({0}).build(ctx, combined, value) : value;
            }
        }
        output = combined.tensor;
        ggml_set_output(output);
        graph = ggml_new_graph_custom(c, 2048, false);
        ggml_build_forward_expand(graph, output);
        allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend()));
        if (!ggml_gallocr_alloc_graph(allocator, graph)) throw std::runtime_error("Cannot allocate Magenta conditioning graph");
        host_plan = core::create_backend_graph_plan_if_host(execution.backend(), graph);
        debug::timing_log_context_reservation("magenta_realtime_2.condition", c);
        debug::timing_log_scalar("magenta_realtime_2.condition.graph.buffer_mb", ggml_gallocr_get_buffer_size(allocator, 0) / (1024.0 * 1024.0));
    }

    ~ConditionGraph() {
        core::free_backend_graph_plan(execution.backend(), host_plan);
        core::release_backend_graph_resources(execution.backend(), graph, true);
        if (allocator) ggml_gallocr_free(allocator);
    }

    std::vector<float> run(const std::array<int32_t, 12> & style_ids, const std::array<int32_t, 132> & control_ids) {
        ggml_backend_tensor_set(style, style_ids.data(), 0, sizeof(style_ids));
        ggml_backend_tensor_set(controls, control_ids.data(), 0, sizeof(control_ids));
        if (core::compute_backend_graph(execution.backend(), graph, host_plan) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("Magenta conditioning graph failed");
        return core::read_tensor_f32(output);
    }
};

class FrameGraph {
public:
    FrameGraph(core::ExecutionContext & execution, const DepthformerWeights & weights, MagentaConfig config, bool greedy)
        : execution_(execution), weights_(weights), config_(config), greedy_(greedy),
          context_(ggml_init({8 * 1024 * 1024, nullptr, true})),
          input_context_(ggml_init({256 * 1024, nullptr, true})) {
        if (!context_ || !input_context_) throw std::runtime_error("Cannot allocate Magenta frame metadata");
        auto * c = context_.get();
        ctx_ = {c, "magenta_realtime_2.frame", execution.backend_type()};
        ctx_.elide_identity_reshapes = true;
        core::ModuleBuildContext inputs{input_context_.get(), "magenta_realtime_2.frame.inputs", execution.backend_type()};
        const auto previous = core::make_tensor(inputs, GGML_TYPE_I32, TensorShape::from_dims({12}));
        const auto mask = core::make_tensor(inputs, GGML_TYPE_F32, TensorShape::from_dims({config_.window + 1}));
        const auto condition = core::make_tensor(inputs, GGML_TYPE_F32,
            TensorShape::from_dims({config_.temporal_layers * 2 * config_.temporal_dim}));
        previous_ = previous.tensor;
        mask_ = mask.tensor;
        condition_ = condition.tensor;
        ggml_set_input(previous_);
        ggml_set_input(mask_);
        ggml_set_input(condition_);
        mask_values_.resize(static_cast<size_t>(config_.window + 1));
        if (!greedy_) {
            uniforms_ = core::make_tensor(inputs, GGML_TYPE_F32, TensorShape::from_dims({12})).tensor;
            temperature_ = core::make_tensor(inputs, GGML_TYPE_F32, TensorShape::from_dims({1})).tensor;
            rank_ = core::make_tensor(inputs, GGML_TYPE_I32, TensorShape::from_dims({1})).tensor;
            ggml_set_input(uniforms_);
            ggml_set_input(temperature_);
            ggml_set_input(rank_);
        }
        const auto & token_weight = weights_.at("token_embedding.weight");
        auto embedded = modules::EmbeddingModule({token_weight.shape.dims[0], config_.temporal_dim}).build(ctx_, previous, token_weight);
        embedded = modules::ScaleBiasModule({std::sqrt(static_cast<float>(config_.temporal_dim)), 0.0F}).build(ctx_, embedded);
        auto x = modules::ReduceMeanModule({0}).build(ctx_, embedded);
        x = modules::ReshapeModule({TensorShape::from_dims({1, config_.temporal_dim})}).build(ctx_, core::ensure_backend_addressable_layout(ctx_, x));
        for (int64_t i = 0; i < config_.temporal_layers; ++i) {
            const auto prefix = "temporal." + std::to_string(i);
            x = attention(x, prefix + ".self", config_.temporal_heads, true, {}, {});
            const int64_t offset = i * 2 * config_.temporal_dim;
            const auto key = modules::SliceModule({0, offset, config_.temporal_dim}).build(ctx_, condition);
            const auto value = modules::SliceModule({0, offset + config_.temporal_dim, config_.temporal_dim}).build(ctx_, condition);
            x = attention(x, prefix + ".cross", config_.temporal_heads, true, key, value);
            x = weights_.feed_forward(ctx_, x, prefix);
        }
        std::vector<TensorValue> depth_keys(static_cast<size_t>(config_.depth_layers));
        std::vector<TensorValue> depth_values(depth_keys.size());
        auto depth_input = x;
        TensorValue codes;
        for (int step = 0; step < 12; ++step) {
            x = weights_.linear(ctx_, depth_input, "depth.depth_input_adapter");
            for (int64_t i = 0; i < config_.depth_layers; ++i) {
                const auto prefix = "depth." + std::to_string(i);
                x = attention(x, prefix + ".self", config_.depth_heads, false,
                              depth_keys[static_cast<size_t>(i)], depth_values[static_cast<size_t>(i)]);
                depth_keys[static_cast<size_t>(i)] = last_key_;
                depth_values[static_cast<size_t>(i)] = last_value_;
                x = weights_.feed_forward(ctx_, x, prefix);
            }
            x = weights_.norm(ctx_, x, "depth.final_ln", true);
            const int64_t offset = 6 + step * 1024;
            const modules::LinearWeights head_weights{
                modules::SliceModule({0, offset, 1024}).build(ctx_, weights_.at("depth.to_logits.weight")),
                modules::SliceModule({0, offset, 1024}).build(ctx_, weights_.at("depth.to_logits.bias"))};
            auto logits = modules::LinearModule({config_.depth_dim, 1024, true, GGML_PREC_F32}).build(ctx_, x, head_weights);
            logits = modules::ScaleBiasModule({1.0F / 30.0F, 0.0F}).build(ctx_, logits);
            logits = modules::TanhModule().build(ctx_, logits);
            logits = modules::ScaleBiasModule({30.0F, 0.0F}).build(ctx_, logits);
            const auto code = core::wrap_tensor(greedy_ ? ggml_argmax(c, logits.tensor) : sample(logits.tensor, step),
                TensorShape::from_dims({1}), GGML_TYPE_I32);
            const auto float_code = core::wrap_tensor(ggml_cast(c, code.tensor, GGML_TYPE_F32), code.shape);
            const auto shifted = modules::ScaleBiasModule({1.0F, static_cast<float>(offset)}).build(ctx_, float_code);
            const auto unique = core::wrap_tensor(ggml_cast(c, shifted.tensor, GGML_TYPE_I32), shifted.shape, GGML_TYPE_I32);
            codes = codes.valid() ? modules::ConcatModule({0}).build(ctx_, codes, float_code) : float_code;
            depth_input = modules::EmbeddingModule({token_weight.shape.dims[0], config_.temporal_dim}).build(ctx_, unique, token_weight);
            depth_input = modules::ScaleBiasModule({std::sqrt(static_cast<float>(config_.temporal_dim)), 0.0F}).build(ctx_, depth_input);
        }
        output_ = ggml_cast(c, codes.tensor, GGML_TYPE_I32);
        ggml_set_output(output_);
        cache_.set_outputs();
        graph_ = ggml_new_graph_custom(c, 32768, false);
        ggml_build_forward_expand(graph_, output_);
        cache_.build_forward_expand(graph_);
        input_buffer_ = ggml_backend_alloc_ctx_tensors(input_context_.get(), execution.backend());
        if (!input_buffer_) throw std::runtime_error("Cannot allocate Magenta persistent frame inputs");
        allocator_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend()));
        if (!ggml_gallocr_alloc_graph(allocator_, graph_)) throw std::runtime_error("Cannot allocate Magenta frame graph");
        host_plan_ = core::create_backend_graph_plan_if_host(execution.backend(), graph_);
        debug::timing_log_context_reservation("magenta_realtime_2.frame", c);
        debug::timing_log_scalar("magenta_realtime_2.frame.graph.buffer_mb", ggml_gallocr_get_buffer_size(allocator_, 0) / (1024.0 * 1024.0));
        debug::timing_log_scalar("magenta_realtime_2.frame.state.buffer_mb", ggml_backend_buffer_get_size(input_buffer_) / (1024.0 * 1024.0));
    }

    ~FrameGraph() {
        core::free_backend_graph_plan(execution_.backend(), host_plan_);
        core::release_backend_graph_resources(execution_.backend(), graph_, true);
        if (allocator_) ggml_gallocr_free(allocator_);
        if (input_buffer_) ggml_backend_buffer_free(input_buffer_);
    }

    void reset(const MagentaSamplingOptions & options) {
        frames_ = 0;
        codes_.fill(0);
        cache_.zero_inputs(execution_.backend());
        if (!greedy_) {
            const float inverse_temperature = 1.0F / options.temperature;
            const int32_t rank = options.top_k == 0 ? 1023 : options.top_k - 1;
            ggml_backend_tensor_set(temperature_, &inverse_temperature, 0, sizeof(float));
            ggml_backend_tensor_set(rank_, &rank, 0, sizeof(rank));
        }
    }

    void condition(const std::vector<float> & values) {
        ggml_backend_tensor_set(condition_, values.data(), 0, values.size() * sizeof(float));
    }

    std::array<int32_t, 12> run(std::mt19937 & rng) {
        ggml_backend_tensor_set(previous_, codes_.data(), 0, sizeof(codes_));
        if (frames_ <= config_.window) {
            std::fill(mask_values_.begin(), mask_values_.end(), 0.0F);
            std::fill_n(mask_values_.begin(), static_cast<size_t>(std::max<int64_t>(0, config_.window - frames_)), -std::numeric_limits<float>::infinity());
            ggml_backend_tensor_set(mask_, mask_values_.data(), 0, mask_values_.size() * sizeof(float));
        }
        if (!greedy_) {
            std::array<float, 12> uniforms;
            for (auto & value : uniforms) value = std::generate_canonical<float, 24>(rng);
            ggml_backend_tensor_set(uniforms_, uniforms.data(), 0, sizeof(uniforms));
        }
        if (core::compute_backend_graph(execution_.backend(), graph_, host_plan_) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("Magenta frame graph failed");
        cache_.commit_outputs(execution_.backend());
        std::array<int32_t, 12> output{};
        ggml_backend_tensor_get(output_, output.data(), 0, sizeof(output));
        for (size_t i = 0; i < codes_.size(); ++i) codes_[i] = 6 + static_cast<int32_t>(i) * 1024 + output[i];
        ++frames_;
        return output;
    }

private:
    TensorValue heads(const TensorValue & x, int64_t count) {
        auto result = x;
        if (!ggml_is_contiguous(result.tensor)) result = core::ensure_backend_addressable_layout(ctx_, result);
        result = modules::ReshapeModule({TensorShape::from_dims({1, count, 128})}).build(ctx_, result);
        result = modules::TransposeModule({{1, 0, 2, 3}, 3}).build(ctx_, result);
        if (!ggml_is_contiguous(result.tensor)) result = core::ensure_backend_addressable_layout(ctx_, result);
        return modules::ReshapeModule({TensorShape::from_dims({1, count, 1, 128})}).build(ctx_, result);
    }

    TensorValue attention(const TensorValue & residual, const std::string & prefix,
                          int64_t count, bool temporal, const TensorValue & cached_key, const TensorValue & cached_value) {
        const auto normalized = weights_.norm(ctx_, residual, prefix + ".pre_norm");
        const bool cross = prefix.find(".cross") != std::string::npos;
        TensorValue q, k, v;
        if (cross) {
            q = heads(weights_.linear(ctx_, normalized, prefix + ".q"), count);
            k = heads(cached_key, count);
            v = heads(cached_value, count);
        } else {
            auto qkv = weights_.linear(ctx_, normalized, prefix + ".qkv");
            const int64_t dim = count * 128;
            qkv = modules::ReshapeModule({TensorShape::from_dims({3 * dim})}).build(ctx_, qkv);
            q = heads(modules::SliceModule({0, 0, dim}).build(ctx_, qkv), count);
            k = heads(modules::SliceModule({0, dim, dim}).build(ctx_, qkv), count);
            v = heads(modules::SliceModule({0, 2 * dim, dim}).build(ctx_, qkv), count);
        }
        if (temporal) {
            core::ModuleBuildContext inputs{input_context_.get(), "magenta_realtime_2.attention.inputs", execution_.backend_type()};
            const auto shape = TensorShape::from_dims({1, count, config_.window, 128});
            const auto key_cache = core::make_tensor(inputs, GGML_TYPE_F32, shape);
            const auto value_cache = core::make_tensor(inputs, GGML_TYPE_F32, shape);
            ggml_set_input(key_cache.tensor);
            ggml_set_input(value_cache.tensor);
            const auto next = modules::StreamingFrameAttentionCacheModule({config_.window}).build(ctx_, key_cache, value_cache, k, v);
            cache_.add_layer(key_cache.tensor, value_cache.tensor, next.next_key_cache.tensor, next.next_value_cache.tensor);
            k = next.key_context;
            v = next.value_context;
        } else if (cached_key.valid()) {
            k = modules::ConcatModule({2}).build(ctx_, cached_key, k);
            v = modules::ConcatModule({2}).build(ctx_, cached_value, v);
        }
        last_key_ = k;
        last_value_ = v;
        const auto scaled_query = core::wrap_tensor(ggml_mul(ctx_.ggml, q.tensor,
            weights_.at(prefix + ".per_dim_scale").tensor), q.shape);
        auto scores = modules::MatMulModule().build(ctx_, scaled_query,
            modules::TransposeModule({{0, 1, 3, 2}, 4}).build(ctx_, k));
        ggml_mul_mat_set_prec(scores.tensor, GGML_PREC_F32);
        if (temporal) {
            const auto sink_key = heads(weights_.at(prefix + ".sink_key_embeddings"), count);
            const auto sink_value = heads(weights_.at(prefix + ".sink_value_embeddings"), count);
            const auto sinks = modules::MatMulModule().build(ctx_, q,
                modules::TransposeModule({{0, 1, 3, 2}, 4}).build(ctx_, sink_key));
            ggml_mul_mat_set_prec(sinks.tensor, GGML_PREC_F32);
            scores = modules::AddModule(false, true).build(ctx_, scores,
                core::wrap_tensor(mask_, TensorShape::from_dims({config_.window + 1})));
            scores = modules::ConcatModule({3}).build(ctx_, sinks, scores);
            v = modules::ConcatModule({2}).build(ctx_, sink_value, v);
        }
        const auto probabilities = modules::SoftmaxModule().build(ctx_, scores);
        auto context = modules::MatMulModule().build(ctx_, probabilities, v);
        ggml_mul_mat_set_prec(context.tensor, GGML_PREC_F32);
        context = modules::ReshapeModule({TensorShape::from_dims({1, count * 128})}).build(ctx_, core::ensure_backend_addressable_layout(ctx_, context));
        context = weights_.linear(ctx_, context, prefix + ".o");
        context = weights_.norm(ctx_, context, prefix + ".post_norm");
        return modules::AddModule().build(ctx_, residual, context);
    }

    ggml_tensor * sample(ggml_tensor * logits, int step) {
        auto * c = context_.get();
        logits = ggml_reshape_1d(c, logits, 1024);
        auto * order = ggml_argsort(c, logits, GGML_SORT_ORDER_DESC);
        auto * sorted = ggml_reshape_1d(c, ggml_get_rows(c, ggml_reshape_2d(c, logits, 1, 1024), order), 1024);
        auto * threshold = ggml_get_rows(c, ggml_reshape_2d(c, sorted, 1, 1024), rank_);
        auto * keep = ggml_scale_bias(c, ggml_step(c, ggml_sub(c, ggml_repeat(c, threshold, sorted), sorted)), -1.0F, 1.0F);
        auto * probability = ggml_mul(c, ggml_soft_max(c, ggml_mul(c, sorted, temperature_)), keep);
        auto * cdf = ggml_cumsum(c, probability);
        auto * total = ggml_view_1d(c, cdf, 1, 1023 * sizeof(float));
        auto * uniform = ggml_view_1d(c, uniforms_, 1, static_cast<size_t>(step) * sizeof(float));
        auto * selected = ggml_cast(c, ggml_sum(c, ggml_step(c, ggml_sub(c, ggml_repeat(c, ggml_mul(c, total, uniform), cdf), cdf))), GGML_TYPE_I32);
        return ggml_cast(c, ggml_get_rows(c, ggml_reshape_2d(c, ggml_cast(c, order, GGML_TYPE_F32), 1, 1024), selected), GGML_TYPE_I32);
    }

    core::ExecutionContext & execution_;
    const DepthformerWeights & weights_;
    MagentaConfig config_;
    bool greedy_;
    std::unique_ptr<ggml_context, ContextDeleter> context_;
    std::unique_ptr<ggml_context, ContextDeleter> input_context_;
    ggml_backend_buffer_t input_buffer_ = nullptr;
    core::ModuleBuildContext ctx_;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t allocator_ = nullptr;
    ggml_backend_graph_plan_t host_plan_ = nullptr;
    modules::StreamingFrameAttentionCache cache_;
    std::vector<float> mask_values_;
    ggml_tensor * previous_ = nullptr;
    ggml_tensor * mask_ = nullptr;
    ggml_tensor * condition_ = nullptr;
    ggml_tensor * uniforms_ = nullptr;
    ggml_tensor * temperature_ = nullptr;
    ggml_tensor * rank_ = nullptr;
    ggml_tensor * output_ = nullptr;
    TensorValue last_key_;
    TensorValue last_value_;
    std::array<int32_t, 12> codes_{};
    int64_t frames_ = 0;
};

}  // namespace

class DepthformerRuntime::Impl {
public:
    Impl(core::ExecutionContext & execution, std::shared_ptr<const MagentaAssets> assets,
         engine::assets::TensorStorageType storage)
        : execution(execution), assets(std::move(assets)), weights(execution, *this->assets, storage),
          conditioning(execution, weights, this->assets->config) {}

    core::ExecutionContext & execution;
    std::shared_ptr<const MagentaAssets> assets;
    DepthformerWeights weights;
    ConditionGraph conditioning;
    std::array<std::unique_ptr<FrameGraph>, 2> graphs;
    FrameGraph * current = nullptr;
    std::mt19937 rng;
    std::vector<float> condition_values;
    std::array<int32_t, 12> last_style{};
    std::array<int32_t, 132> last_controls{};
    bool has_condition = false;
};

DepthformerRuntime::DepthformerRuntime(core::ExecutionContext & execution, std::shared_ptr<const MagentaAssets> assets,
                                     engine::assets::TensorStorageType storage)
    : impl_(std::make_unique<Impl>(execution, std::move(assets), storage)) {}
DepthformerRuntime::~DepthformerRuntime() = default;

void DepthformerRuntime::reset(const MagentaSamplingOptions & options) {
    auto & p = *impl_;
    p.rng.seed(options.seed);
    const size_t mode = options.temperature <= 0 ? 0 : 1;
    if (!p.graphs[mode]) p.graphs[mode] = std::make_unique<FrameGraph>(p.execution, p.weights, p.assets->config, mode == 0);
    p.current = p.graphs[mode].get();
    p.current->reset(options);
    if (p.has_condition) p.current->condition(p.condition_values);
}

void DepthformerRuntime::condition(const std::array<int32_t, 12> & style, const std::array<int32_t, 132> & controls) {
    auto & p = *impl_;
    if (p.has_condition && p.last_style == style && p.last_controls == controls) return;
    p.condition_values = p.conditioning.run(style, controls);
    p.last_style = style;
    p.last_controls = controls;
    p.has_condition = true;
    if (p.current) p.current->condition(p.condition_values);
}

std::array<int32_t, 12> DepthformerRuntime::step() {
    if (!impl_->current || !impl_->has_condition) throw std::runtime_error("Magenta generation requires reset and conditioning");
    return impl_->current->run(impl_->rng);
}

}  // namespace engine::models::magenta_realtime_2
