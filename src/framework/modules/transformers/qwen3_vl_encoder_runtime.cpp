#include "engine/framework/modules/transformers/qwen3_vl_encoder_runtime.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/grouped_query_attention.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/optimizations/fast_projection_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include "ggml-alloc.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace engine::modules {
namespace {

namespace assets = engine::assets;
namespace core = engine::core;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

void validate_config(const Qwen3VlEncoderRuntimeConfig & config) {
    if (config.vocab_size <= 0 ||
        config.stack.hidden_size <= 0 ||
        config.stack.layers <= 0 ||
        config.stack.num_attention_heads <= 0 ||
        config.stack.num_key_value_heads <= 0 ||
        config.stack.head_dim <= 0 ||
        config.stack.intermediate_size <= 0) {
        throw std::runtime_error("Qwen3VlEncoderRuntime requires positive model dimensions");
    }
    if (config.stack.num_attention_heads % config.stack.num_key_value_heads != 0) {
        throw std::runtime_error("Qwen3VlEncoderRuntime attention heads must be divisible by key/value heads");
    }
    if (config.stack.head_dim % 2 != 0) {
        throw std::runtime_error("Qwen3VlEncoderRuntime head_dim must be even");
    }
    if (config.weight_context_bytes == 0 || config.graph_arena_bytes == 0 || config.input_arena_bytes == 0) {
        throw std::runtime_error("Qwen3VlEncoderRuntime requires positive context sizes");
    }
    if (config.model_prefix.empty()) {
        throw std::runtime_error("Qwen3VlEncoderRuntime requires a model weight prefix");
    }
    if (config.output_mode == Qwen3VlEncoderOutputMode::Logits && config.logits_size < 0) {
        throw std::runtime_error("Qwen3VlEncoderRuntime logits size must be non-negative");
    }
    if (config.output_mode == Qwen3VlEncoderOutputMode::Logits && config.lm_head_weight_name.empty()) {
        throw std::runtime_error("Qwen3VlEncoderRuntime logits mode requires an lm_head weight name");
    }
    if (config.readback_round_type.has_value() && *config.readback_round_type != GGML_TYPE_BF16) {
        throw std::runtime_error("Qwen3VlEncoderRuntime readback rounding currently supports only bf16");
    }
}

void validate_options(const Qwen3VlEncoderOptions & options) {
    if (options.layerwise_batch <= 0) {
        throw std::runtime_error("Qwen3VlEncoderRuntime layerwise_batch must be positive");
    }
}

std::string embed_tokens_name(const Qwen3VlEncoderRuntimeConfig & config) {
    return config.model_prefix + ".embed_tokens.weight";
}

std::string final_norm_name(const Qwen3VlEncoderRuntimeConfig & config) {
    return config.final_norm_weight_name.empty()
        ? config.model_prefix + ".norm.weight"
        : config.final_norm_weight_name;
}

std::string layer_prefix(const Qwen3VlEncoderRuntimeConfig & config, int64_t layer) {
    return config.model_prefix + ".layers." + std::to_string(layer) + ".";
}

bool returns_logits(const Qwen3VlEncoderRuntimeConfig & config) {
    return config.output_mode == Qwen3VlEncoderOutputMode::Logits;
}

int64_t resolved_logits_size(const Qwen3VlEncoderRuntimeConfig & config) {
    return config.logits_size > 0 ? config.logits_size : config.vocab_size;
}

bool needs_hidden_readback(const Qwen3VlEncoderRuntimeConfig & config) {
    return config.output_mode == Qwen3VlEncoderOutputMode::Hidden || config.return_hidden;
}

std::vector<std::string> full_required_names(const Qwen3VlEncoderRuntimeConfig & config) {
    std::vector<std::string> names{embed_tokens_name(config)};
    if (returns_logits(config)) {
        names.push_back(final_norm_name(config));
        names.push_back(config.lm_head_weight_name);
        if (config.use_lm_head_bias) {
            names.push_back(config.lm_head_bias_name);
        }
    }
    return names;
}

std::vector<std::string> final_head_required_names(const Qwen3VlEncoderRuntimeConfig & config) {
    std::vector<std::string> names{final_norm_name(config), config.lm_head_weight_name};
    if (config.use_lm_head_bias) {
        names.push_back(config.lm_head_bias_name);
    }
    return names;
}

bool matches_weight_filter(
    std::string_view name,
    const std::vector<std::string> & required_names,
    const std::vector<std::string> & prefix_filters) {
    for (const auto & required : required_names) {
        if (name == required) {
            return true;
        }
    }
    for (const auto & prefix : prefix_filters) {
        if (name.rfind(prefix, 0) == 0) {
            return true;
        }
    }
    return false;
}

std::vector<std::string> layer_prefix_filters(const Qwen3VlEncoderRuntimeConfig & config, int64_t begin, int64_t end) {
    std::vector<std::string> prefixes;
    prefixes.reserve(static_cast<size_t>(end - begin));
    for (int64_t layer = begin; layer < end; ++layer) {
        prefixes.push_back(layer_prefix(config, layer));
    }
    return prefixes;
}

class Qwen3VlEncoderWeightStore {
public:
    Qwen3VlEncoderWeightStore(
        core::ExecutionContext & execution,
        std::shared_ptr<const assets::TensorSource> source,
        const Qwen3VlEncoderRuntimeConfig & config)
        : Qwen3VlEncoderWeightStore(
              execution,
              std::move(source),
              config,
              config.trace_name + ".weights",
              full_required_names(config),
              layer_prefix_filters(config, 0, config.stack.layers)) {}

    Qwen3VlEncoderWeightStore(
        core::ExecutionContext & execution,
        std::shared_ptr<const assets::TensorSource> source,
        const Qwen3VlEncoderRuntimeConfig & config,
        const std::string & label,
        const std::vector<std::string> & required_names,
        const std::vector<std::string> & prefix_filters)
        : source_(std::move(source)),
          store_(
              execution.backend(),
              execution.backend_type(),
              label,
              config.weight_context_bytes) {
        if (source_ == nullptr) {
            throw std::runtime_error("Qwen3VlEncoderRuntime tensor source is missing");
        }
        for (const auto & meta : source_->tensors()) {
            if (!matches_weight_filter(meta.name, required_names, prefix_filters)) {
                continue;
            }
            weights_.emplace(
                meta.name,
                store_.load_tensor(*source_, meta.name, config.weight_storage_type, meta.shape));
        }
        for (const auto & required : required_names) {
            if (weights_.find(required) == weights_.end()) {
                throw std::runtime_error("missing Qwen3-VL encoder tensor: " + required);
            }
        }
        store_.upload();
        source_->release_storage();
    }

    const core::TensorValue & require(std::string_view name) const {
        const auto it = weights_.find(std::string(name));
        if (it == weights_.end()) {
            throw std::runtime_error("missing Qwen3-VL encoder tensor: " + std::string(name));
        }
        return it->second;
    }

    std::optional<core::TensorValue> maybe(std::string_view name) const {
        const auto it = weights_.find(std::string(name));
        if (it == weights_.end()) {
            return std::nullopt;
        }
        return it->second;
    }

private:
    std::shared_ptr<const assets::TensorSource> source_;
    core::BackendWeightStore store_;
    std::unordered_map<std::string, core::TensorValue> weights_;
};

core::TensorValue linear_projection(
    core::ModuleBuildContext & ctx,
    const Qwen3VlEncoderRuntimeConfig & config,
    const core::TensorValue & x,
    const core::TensorValue & weight,
    int64_t in_features,
    int64_t out_features) {
    const ggml_prec precision = ggml_is_quantized(weight.tensor->type) ? GGML_PREC_DEFAULT : config.stack.projection_precision;
    const bool use_fast_projection =
        config.use_cuda_fast_projection &&
        ctx.backend_type == core::BackendType::Cuda &&
        out_features % 4 == 0;
    LinearWeights params{weight, std::nullopt};
    return use_fast_projection
        ? FastPackedProjection4Module({in_features, out_features, precision}).build(ctx, x, params)
        : LinearModule({in_features, out_features, false, precision}).build(ctx, x, params);
}

std::vector<float> split_rope_table_values(
    const Qwen3VlEncoderRuntimeConfig & config,
    int64_t tokens,
    int64_t heads,
    bool cosine) {
    const int64_t half = config.stack.head_dim / 2;
    std::vector<float> out(static_cast<size_t>(heads * tokens * half));
    for (int64_t head = 0; head < heads; ++head) {
        for (int64_t t = 0; t < tokens; ++t) {
            for (int64_t i = 0; i < half; ++i) {
                const float inv = 1.0F / std::pow(
                    config.stack.rope_theta,
                    static_cast<float>(2 * i) / static_cast<float>(config.stack.head_dim));
                out[static_cast<size_t>((head * tokens + t) * half + i)] =
                    cosine ? std::cos(static_cast<float>(t) * inv) : std::sin(static_cast<float>(t) * inv);
            }
        }
    }
    return out;
}

core::TensorValue build_mlp(
    core::ModuleBuildContext & ctx,
    const Qwen3VlEncoderWeightStore & weights,
    const Qwen3VlEncoderRuntimeConfig & config,
    const core::TensorValue & x,
    int64_t layer) {
    const std::string prefix = layer_prefix(config, layer) + "mlp.";
    auto gate = linear_projection(
        ctx,
        config,
        x,
        weights.require(prefix + "gate_proj.weight"),
        config.stack.hidden_size,
        config.stack.intermediate_size);
    auto up = linear_projection(
        ctx,
        config,
        x,
        weights.require(prefix + "up_proj.weight"),
        config.stack.hidden_size,
        config.stack.intermediate_size);
    auto hidden = MulModule{}.build(
        ctx,
        core::ensure_backend_addressable_layout(ctx, SiluModule{}.build(ctx, gate)),
        core::ensure_backend_addressable_layout(ctx, up));
    return linear_projection(
        ctx,
        config,
        hidden,
        weights.require(prefix + "down_proj.weight"),
        config.stack.intermediate_size,
        config.stack.hidden_size);
}

core::TensorValue build_attention(
    core::ModuleBuildContext & ctx,
    const Qwen3VlEncoderWeightStore & weights,
    const Qwen3VlEncoderRuntimeConfig & config,
    const core::TensorValue & x,
    const core::TensorValue & q_cos,
    const core::TensorValue & q_sin,
    const core::TensorValue & k_cos,
    const core::TensorValue & k_sin,
    int64_t layer) {
    const std::string prefix = layer_prefix(config, layer) + "self_attn.";
    const int64_t tokens = x.shape.dims[1];
    const int64_t head_dim = config.stack.head_dim;
    auto q = linear_projection(
        ctx,
        config,
        x,
        weights.require(prefix + "q_proj.weight"),
        config.stack.hidden_size,
        config.stack.num_attention_heads * head_dim);
    auto k = linear_projection(
        ctx,
        config,
        x,
        weights.require(prefix + "k_proj.weight"),
        config.stack.hidden_size,
        config.stack.num_key_value_heads * head_dim);
    auto v = linear_projection(
        ctx,
        config,
        x,
        weights.require(prefix + "v_proj.weight"),
        config.stack.hidden_size,
        config.stack.num_key_value_heads * head_dim);
    q = core::reshape_tensor(ctx, q, core::TensorShape::from_dims({1, tokens, config.stack.num_attention_heads, head_dim}));
    k = core::reshape_tensor(ctx, k, core::TensorShape::from_dims({1, tokens, config.stack.num_key_value_heads, head_dim}));
    v = core::reshape_tensor(ctx, v, core::TensorShape::from_dims({1, tokens, config.stack.num_key_value_heads, head_dim}));
    q = TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, q);
    k = TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, k);
    v = TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, v);
    q = RMSNormModule({head_dim, config.stack.rms_norm_eps, true, false}).build(
        ctx,
        q,
        {weights.require(prefix + "q_norm.weight"), std::nullopt});
    k = RMSNormModule({head_dim, config.stack.rms_norm_eps, true, false}).build(
        ctx,
        k,
        {weights.require(prefix + "k_norm.weight"), std::nullopt});
    q = SplitRoPEModule({head_dim}).build(ctx, q, q_cos, q_sin);
    k = SplitRoPEModule({head_dim}).build(ctx, k, k_cos, k_sin);
    auto h = GroupedQueryAttentionModule({
        head_dim,
        GroupedQueryAttentionLowering::ManualRepeat,
        config.stack.attention_precision,
        AttentionCausality::Causal,
    }).build(ctx, q, k, v);
    h = core::ensure_backend_addressable_layout(ctx, h);
    h = core::reshape_tensor(ctx, h, core::TensorShape::from_dims({1, tokens, config.stack.num_attention_heads * head_dim}));
    return linear_projection(
        ctx,
        config,
        h,
        weights.require(prefix + "o_proj.weight"),
        config.stack.num_attention_heads * head_dim,
        config.stack.hidden_size);
}

core::TensorValue build_layer_group(
    core::ModuleBuildContext & ctx,
    const Qwen3VlEncoderWeightStore & weights,
    const Qwen3VlEncoderRuntimeConfig & config,
    const core::TensorValue & input,
    const core::TensorValue & q_cos,
    const core::TensorValue & q_sin,
    const core::TensorValue & k_cos,
    const core::TensorValue & k_sin,
    int64_t layer_begin,
    int64_t layer_end) {
    auto x = input;
    for (int64_t layer = layer_begin; layer < layer_end; ++layer) {
        const std::string prefix = layer_prefix(config, layer);
        auto h = RMSNormModule({config.stack.hidden_size, config.stack.rms_norm_eps, true, false}).build(
            ctx,
            x,
            {weights.require(prefix + "input_layernorm.weight"), std::nullopt});
        auto attn = build_attention(ctx, weights, config, h, q_cos, q_sin, k_cos, k_sin, layer);
        x = AddModule{}.build(
            ctx,
            core::ensure_backend_addressable_layout(ctx, x),
            core::ensure_backend_addressable_layout(ctx, attn));
        h = RMSNormModule({config.stack.hidden_size, config.stack.rms_norm_eps, true, false}).build(
            ctx,
            x,
            {weights.require(prefix + "post_attention_layernorm.weight"), std::nullopt});
        auto mlp = build_mlp(ctx, weights, config, h, layer);
        x = AddModule{}.build(
            ctx,
            core::ensure_backend_addressable_layout(ctx, x),
            core::ensure_backend_addressable_layout(ctx, mlp));
    }
    return x;
}

core::TensorValue build_encoder(
    core::ModuleBuildContext & ctx,
    const Qwen3VlEncoderWeightStore & weights,
    const Qwen3VlEncoderRuntimeConfig & config,
    const core::TensorValue & ids,
    const core::TensorValue & q_cos,
    const core::TensorValue & q_sin,
    const core::TensorValue & k_cos,
    const core::TensorValue & k_sin) {
    auto x = EmbeddingModule({config.vocab_size, config.stack.hidden_size})
                 .build(ctx, ids, weights.require(embed_tokens_name(config)));
    x = core::reshape_tensor(ctx, x, core::TensorShape::from_dims({1, ids.shape.dims[0], config.stack.hidden_size}));
    x = build_layer_group(ctx, weights, config, x, q_cos, q_sin, k_cos, k_sin, 0, config.stack.layers);
    return core::reshape_tensor(ctx, x, core::TensorShape::from_dims({ids.shape.dims[0], config.stack.hidden_size}));
}

core::TensorValue build_logits(
    core::ModuleBuildContext & ctx,
    const Qwen3VlEncoderWeightStore & weights,
    const Qwen3VlEncoderRuntimeConfig & config,
    const core::TensorValue & hidden) {
    auto logits_input = RMSNormModule({config.stack.hidden_size, config.stack.rms_norm_eps, true, false}).build(
        ctx,
        hidden,
        {weights.require(final_norm_name(config)), std::nullopt});
    if (config.lm_head_input_type.has_value() && logits_input.type != *config.lm_head_input_type) {
        logits_input = core::wrap_tensor(
            ggml_cast(ctx.ggml, logits_input.tensor, *config.lm_head_input_type),
            logits_input.shape,
            *config.lm_head_input_type);
    }
    return LinearModule({
        config.stack.hidden_size,
        resolved_logits_size(config),
        config.use_lm_head_bias,
        config.lm_head_precision,
    }).build(
        ctx,
        logits_input,
        {weights.require(config.lm_head_weight_name), weights.maybe(config.lm_head_bias_name)});
}

struct EncoderGraphReadback {
    core::TensorValue hidden;
    core::TensorValue logits;
};

class EncoderGraph {
public:
    EncoderGraph(
        core::ExecutionContext & execution,
        Qwen3VlEncoderWeightStore & weights,
        const Qwen3VlEncoderRuntimeConfig & config,
        int64_t tokens)
        : execution_(execution),
          config_(config),
          tokens_(tokens) {
        ctx_.reset(ggml_init({config_.graph_arena_bytes, nullptr, true}));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize Qwen3-VL encoder graph context");
        }
        input_ctx_.reset(ggml_init({config_.input_arena_bytes, nullptr, true}));
        if (input_ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize Qwen3-VL encoder input context");
        }
        const std::string input_label = config_.trace_name + ".inputs";
        core::ModuleBuildContext input_ctx{input_ctx_.get(), input_label.c_str(), execution_.backend_type()};
        ids_ = core::make_tensor(input_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({tokens_}));
        q_cos_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, config_.stack.num_attention_heads, tokens_, config_.stack.head_dim / 2}));
        q_sin_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, config_.stack.num_attention_heads, tokens_, config_.stack.head_dim / 2}));
        k_cos_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, config_.stack.num_key_value_heads, tokens_, config_.stack.head_dim / 2}));
        k_sin_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, config_.stack.num_key_value_heads, tokens_, config_.stack.head_dim / 2}));
        ggml_set_input(ids_.tensor);
        ggml_set_input(q_cos_.tensor);
        ggml_set_input(q_sin_.tensor);
        ggml_set_input(k_cos_.tensor);
        ggml_set_input(k_sin_.tensor);

        core::ModuleBuildContext build_ctx{ctx_.get(), config_.trace_name.c_str(), execution_.backend_type()};
        readback_.hidden = build_encoder(build_ctx, weights, config_, ids_, q_cos_, q_sin_, k_cos_, k_sin_);
        if (returns_logits(config_)) {
            readback_.logits = build_logits(build_ctx, weights, config_, readback_.hidden);
        }
        graph_ = ggml_new_graph_custom(ctx_.get(), 262144, false);
        if (needs_hidden_readback(config_)) {
            ggml_set_output(readback_.hidden.tensor);
            ggml_build_forward_expand(graph_, readback_.hidden.tensor);
        }
        if (returns_logits(config_)) {
            ggml_set_output(readback_.logits.tensor);
            ggml_build_forward_expand(graph_, readback_.logits.tensor);
        }
        input_buffer_ = ggml_backend_alloc_ctx_tensors(input_ctx_.get(), execution_.backend());
        if (input_buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate Qwen3-VL encoder inputs");
        }
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend()));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("failed to allocate Qwen3-VL encoder graph");
        }
        core::prepare_host_graph_plan(execution_, graph_, plan_);
    }

    ~EncoderGraph() {
        plan_.reset();
        if (graph_ != nullptr) {
            core::release_backend_graph_resources(execution_.backend(), graph_);
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
        if (input_buffer_ != nullptr) {
            ggml_backend_buffer_free(input_buffer_);
        }
    }

    int64_t tokens() const noexcept {
        return tokens_;
    }

    Qwen3VlEncoderResult run(const std::vector<int32_t> & ids) {
        core::write_tensor_i32(ids_, ids);
        core::write_tensor_f32(q_cos_, split_rope_table_values(config_, tokens_, config_.stack.num_attention_heads, true));
        core::write_tensor_f32(q_sin_, split_rope_table_values(config_, tokens_, config_.stack.num_attention_heads, false));
        core::write_tensor_f32(k_cos_, split_rope_table_values(config_, tokens_, config_.stack.num_key_value_heads, true));
        core::write_tensor_f32(k_sin_, split_rope_table_values(config_, tokens_, config_.stack.num_key_value_heads, false));
        core::set_backend_threads(execution_.backend(), std::max(1, execution_.config().threads));
        const ggml_status status = core::compute_graph(execution_, graph_, plan_, config_.trace_name.c_str());
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Qwen3-VL encoder graph compute failed");
        }
        Qwen3VlEncoderResult result;
        result.steps = tokens_;
        result.hidden_size = config_.stack.hidden_size;
        result.logits_size = returns_logits(config_) ? resolved_logits_size(config_) : 0;
        if (needs_hidden_readback(config_)) {
            result.hidden = core::read_tensor_f32(readback_.hidden.tensor);
        }
        if (returns_logits(config_)) {
            result.logits = core::read_tensor_f32(readback_.logits.tensor);
        }
        return result;
    }

private:
    core::ExecutionContext & execution_;
    Qwen3VlEncoderRuntimeConfig config_;
    int64_t tokens_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> input_ctx_;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    ggml_backend_buffer_t input_buffer_ = nullptr;
    core::HostGraphPlan plan_;
    core::TensorValue ids_;
    core::TensorValue q_cos_;
    core::TensorValue q_sin_;
    core::TensorValue k_cos_;
    core::TensorValue k_sin_;
    EncoderGraphReadback readback_;
};

class EmbeddingGraph {
public:
    EmbeddingGraph(
        core::ExecutionContext & execution,
        Qwen3VlEncoderWeightStore & weights,
        const Qwen3VlEncoderRuntimeConfig & config,
        int64_t tokens)
        : execution_(execution),
          config_(config),
          tokens_(tokens) {
        ctx_.reset(ggml_init({128ull * 1024ull * 1024ull, nullptr, true}));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize Qwen3-VL embedding graph context");
        }
        input_ctx_.reset(ggml_init({config_.input_arena_bytes, nullptr, true}));
        if (input_ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize Qwen3-VL embedding input context");
        }
        const std::string input_label = config_.trace_name + ".embedding.inputs";
        core::ModuleBuildContext input_ctx{input_ctx_.get(), input_label.c_str(), execution_.backend_type()};
        ids_ = core::make_tensor(input_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({tokens_}));
        ggml_set_input(ids_.tensor);
        const std::string build_label = config_.trace_name + ".embedding";
        core::ModuleBuildContext build_ctx{ctx_.get(), build_label.c_str(), execution_.backend_type()};
        auto x = EmbeddingModule({config_.vocab_size, config_.stack.hidden_size})
                     .build(build_ctx, ids_, weights.require(embed_tokens_name(config_)));
        output_ = core::reshape_tensor(build_ctx, x, core::TensorShape::from_dims({1, tokens_, config_.stack.hidden_size}));
        graph_ = ggml_new_graph_custom(ctx_.get(), 65536, false);
        ggml_set_output(output_.tensor);
        ggml_build_forward_expand(graph_, output_.tensor);
        input_buffer_ = ggml_backend_alloc_ctx_tensors(input_ctx_.get(), execution_.backend());
        if (input_buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate Qwen3-VL embedding inputs");
        }
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend()));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("failed to allocate Qwen3-VL embedding graph");
        }
        core::prepare_host_graph_plan(execution_, graph_, plan_);
    }

    ~EmbeddingGraph() {
        plan_.reset();
        if (graph_ != nullptr) {
            core::release_backend_graph_resources(execution_.backend(), graph_);
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
        if (input_buffer_ != nullptr) {
            ggml_backend_buffer_free(input_buffer_);
        }
    }

    std::vector<float> run(const std::vector<int32_t> & ids) {
        core::write_tensor_i32(ids_, ids);
        core::set_backend_threads(execution_.backend(), std::max(1, execution_.config().threads));
        const std::string label = config_.trace_name + ".embedding";
        const ggml_status status = core::compute_graph(execution_, graph_, plan_, label.c_str());
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Qwen3-VL embedding graph compute failed");
        }
        return core::read_tensor_f32(output_.tensor);
    }

private:
    core::ExecutionContext & execution_;
    Qwen3VlEncoderRuntimeConfig config_;
    int64_t tokens_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> input_ctx_;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    ggml_backend_buffer_t input_buffer_ = nullptr;
    core::HostGraphPlan plan_;
    core::TensorValue ids_;
    core::TensorValue output_;
};

class LayerGroupGraph {
public:
    LayerGroupGraph(
        core::ExecutionContext & execution,
        Qwen3VlEncoderWeightStore & weights,
        const Qwen3VlEncoderRuntimeConfig & config,
        int64_t tokens,
        int64_t layer_begin,
        int64_t layer_end)
        : execution_(execution),
          config_(config),
          tokens_(tokens) {
        ctx_.reset(ggml_init({config_.graph_arena_bytes, nullptr, true}));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize Qwen3-VL layerwise graph context");
        }
        input_ctx_.reset(ggml_init({32ull * 1024ull * 1024ull, nullptr, true}));
        if (input_ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize Qwen3-VL layerwise input context");
        }
        const std::string input_label = config_.trace_name + ".layerwise.inputs";
        core::ModuleBuildContext input_ctx{input_ctx_.get(), input_label.c_str(), execution_.backend_type()};
        hidden_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, tokens_, config_.stack.hidden_size}));
        q_cos_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, config_.stack.num_attention_heads, tokens_, config_.stack.head_dim / 2}));
        q_sin_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, config_.stack.num_attention_heads, tokens_, config_.stack.head_dim / 2}));
        k_cos_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, config_.stack.num_key_value_heads, tokens_, config_.stack.head_dim / 2}));
        k_sin_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, config_.stack.num_key_value_heads, tokens_, config_.stack.head_dim / 2}));
        ggml_set_input(hidden_.tensor);
        ggml_set_input(q_cos_.tensor);
        ggml_set_input(q_sin_.tensor);
        ggml_set_input(k_cos_.tensor);
        ggml_set_input(k_sin_.tensor);
        const std::string build_label = config_.trace_name + ".layerwise";
        core::ModuleBuildContext build_ctx{ctx_.get(), build_label.c_str(), execution_.backend_type()};
        output_ = build_layer_group(build_ctx, weights, config_, hidden_, q_cos_, q_sin_, k_cos_, k_sin_, layer_begin, layer_end);
        graph_ = ggml_new_graph_custom(ctx_.get(), 262144, false);
        ggml_set_output(output_.tensor);
        ggml_build_forward_expand(graph_, output_.tensor);
        input_buffer_ = ggml_backend_alloc_ctx_tensors(input_ctx_.get(), execution_.backend());
        if (input_buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate Qwen3-VL layerwise inputs");
        }
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend()));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("failed to allocate Qwen3-VL layerwise graph");
        }
        core::prepare_host_graph_plan(execution_, graph_, plan_);
    }

    ~LayerGroupGraph() {
        plan_.reset();
        if (graph_ != nullptr) {
            core::release_backend_graph_resources(execution_.backend(), graph_);
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
        if (input_buffer_ != nullptr) {
            ggml_backend_buffer_free(input_buffer_);
        }
    }

    std::vector<float> run(
        const std::vector<float> & hidden,
        const std::vector<float> & q_cos,
        const std::vector<float> & q_sin,
        const std::vector<float> & k_cos,
        const std::vector<float> & k_sin) {
        core::write_tensor_f32(hidden_, hidden);
        core::write_tensor_f32(q_cos_, q_cos);
        core::write_tensor_f32(q_sin_, q_sin);
        core::write_tensor_f32(k_cos_, k_cos);
        core::write_tensor_f32(k_sin_, k_sin);
        core::set_backend_threads(execution_.backend(), std::max(1, execution_.config().threads));
        const std::string label = config_.trace_name + ".layerwise";
        const ggml_status status = core::compute_graph(execution_, graph_, plan_, label.c_str());
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Qwen3-VL layerwise graph compute failed");
        }
        return core::read_tensor_f32(output_.tensor);
    }

private:
    core::ExecutionContext & execution_;
    Qwen3VlEncoderRuntimeConfig config_;
    int64_t tokens_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> input_ctx_;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    ggml_backend_buffer_t input_buffer_ = nullptr;
    core::HostGraphPlan plan_;
    core::TensorValue hidden_;
    core::TensorValue q_cos_;
    core::TensorValue q_sin_;
    core::TensorValue k_cos_;
    core::TensorValue k_sin_;
    core::TensorValue output_;
};

class FinalHeadGraph {
public:
    FinalHeadGraph(
        core::ExecutionContext & execution,
        Qwen3VlEncoderWeightStore & weights,
        const Qwen3VlEncoderRuntimeConfig & config,
        int64_t tokens)
        : execution_(execution),
          config_(config),
          tokens_(tokens) {
        ctx_.reset(ggml_init({128ull * 1024ull * 1024ull, nullptr, true}));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize Qwen3-VL final head graph context");
        }
        input_ctx_.reset(ggml_init({config_.input_arena_bytes, nullptr, true}));
        if (input_ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize Qwen3-VL final head input context");
        }
        const std::string input_label = config_.trace_name + ".final_head.inputs";
        core::ModuleBuildContext input_ctx{input_ctx_.get(), input_label.c_str(), execution_.backend_type()};
        hidden_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({tokens_, config_.stack.hidden_size}));
        ggml_set_input(hidden_.tensor);
        const std::string build_label = config_.trace_name + ".final_head";
        core::ModuleBuildContext build_ctx{ctx_.get(), build_label.c_str(), execution_.backend_type()};
        logits_ = build_logits(build_ctx, weights, config_, hidden_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 65536, false);
        ggml_set_output(logits_.tensor);
        ggml_build_forward_expand(graph_, logits_.tensor);
        input_buffer_ = ggml_backend_alloc_ctx_tensors(input_ctx_.get(), execution_.backend());
        if (input_buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate Qwen3-VL final head inputs");
        }
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend()));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("failed to allocate Qwen3-VL final head graph");
        }
        core::prepare_host_graph_plan(execution_, graph_, plan_);
    }

    ~FinalHeadGraph() {
        plan_.reset();
        if (graph_ != nullptr) {
            core::release_backend_graph_resources(execution_.backend(), graph_);
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
        if (input_buffer_ != nullptr) {
            ggml_backend_buffer_free(input_buffer_);
        }
    }

    std::vector<float> run(const std::vector<float> & hidden) {
        core::write_tensor_f32(hidden_, hidden);
        core::set_backend_threads(execution_.backend(), std::max(1, execution_.config().threads));
        const std::string label = config_.trace_name + ".final_head";
        const ggml_status status = core::compute_graph(execution_, graph_, plan_, label.c_str());
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Qwen3-VL final head graph compute failed");
        }
        return core::read_tensor_f32(logits_.tensor);
    }

private:
    core::ExecutionContext & execution_;
    Qwen3VlEncoderRuntimeConfig config_;
    int64_t tokens_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> input_ctx_;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    ggml_backend_buffer_t input_buffer_ = nullptr;
    core::HostGraphPlan plan_;
    core::TensorValue hidden_;
    core::TensorValue logits_;
};

}  // namespace

class Qwen3VlEncoderRuntime::Impl {
public:
    Impl(
        core::ExecutionContext & execution,
        std::shared_ptr<const assets::TensorSource> tensor_source,
        Qwen3VlEncoderRuntimeConfig config)
        : execution_(execution),
          source_(std::move(tensor_source)),
          config_(std::move(config)) {
        validate_config(config_);
        if (source_ == nullptr) {
            throw std::runtime_error("Qwen3VlEncoderRuntime tensor source is missing");
        }
    }

    ~Impl() {
        release_runtime_graphs();
    }

    Qwen3VlEncoderResult encode_text(const std::vector<int32_t> & ids, const Qwen3VlEncoderOptions & options) {
        if (ids.empty()) {
            throw std::runtime_error("Qwen3VlEncoderRuntime encode_text requires tokens");
        }
        validate_options(options);
        const int64_t tokens = static_cast<int64_t>(ids.size());
        auto result = options.layerwise ? encode_text_layerwise(ids, options.layerwise_batch) : encode_text_full(ids);
        round_readback(result.hidden);
        round_readback(result.logits);
        result.steps = tokens;
        result.hidden_size = config_.stack.hidden_size;
        result.logits_size = returns_logits(config_) ? resolved_logits_size(config_) : 0;
        return result;
    }

    void release_runtime_graphs() {
        graph_.reset();
    }

    void release_weights() {
        release_runtime_graphs();
        full_weights_.reset();
        source_->release_storage();
    }

private:
    void round_readback(std::vector<float> & values) const {
        if (!config_.readback_round_type.has_value()) {
            return;
        }
        if (*config_.readback_round_type == GGML_TYPE_BF16) {
            core::round_f32_to_bf16_in_place(values);
        }
    }

    void ensure_full_weights() {
        if (full_weights_ == nullptr) {
            full_weights_ = std::make_unique<Qwen3VlEncoderWeightStore>(execution_, source_, config_);
        }
    }

    Qwen3VlEncoderResult encode_text_full(const std::vector<int32_t> & ids) {
        ensure_full_weights();
        const int64_t tokens = static_cast<int64_t>(ids.size());
        if (graph_ == nullptr || graph_->tokens() != tokens) {
            graph_.reset();
            graph_ = std::make_unique<EncoderGraph>(execution_, *full_weights_, config_, tokens);
        }
        return graph_->run(ids);
    }

    Qwen3VlEncoderResult encode_text_layerwise(const std::vector<int32_t> & ids, int64_t layer_batch) {
        release_weights();
        const int64_t tokens = static_cast<int64_t>(ids.size());
        std::vector<float> hidden;
        {
            Qwen3VlEncoderWeightStore weights(
                execution_,
                source_,
                config_,
                config_.trace_name + ".embedding.weights",
                {embed_tokens_name(config_)},
                {});
            EmbeddingGraph graph(execution_, weights, config_, tokens);
            hidden = graph.run(ids);
        }
        const auto q_cos = split_rope_table_values(config_, tokens, config_.stack.num_attention_heads, true);
        const auto q_sin = split_rope_table_values(config_, tokens, config_.stack.num_attention_heads, false);
        const auto k_cos = split_rope_table_values(config_, tokens, config_.stack.num_key_value_heads, true);
        const auto k_sin = split_rope_table_values(config_, tokens, config_.stack.num_key_value_heads, false);
        for (int64_t layer = 0; layer < config_.stack.layers; layer += layer_batch) {
            const int64_t end = std::min<int64_t>(layer + layer_batch, config_.stack.layers);
            auto prefixes = layer_prefix_filters(config_, layer, end);
            Qwen3VlEncoderWeightStore weights(
                execution_,
                source_,
                config_,
                config_.trace_name + ".layerwise.weights",
                {},
                prefixes);
            LayerGroupGraph graph(execution_, weights, config_, tokens, layer, end);
            hidden = graph.run(hidden, q_cos, q_sin, k_cos, k_sin);
        }
        Qwen3VlEncoderResult result;
        result.steps = tokens;
        result.hidden_size = config_.stack.hidden_size;
        result.logits_size = returns_logits(config_) ? resolved_logits_size(config_) : 0;
        if (returns_logits(config_)) {
            Qwen3VlEncoderWeightStore weights(
                execution_,
                source_,
                config_,
                config_.trace_name + ".final_head.weights",
                final_head_required_names(config_),
                {});
            FinalHeadGraph graph(execution_, weights, config_, tokens);
            result.logits = graph.run(hidden);
        }
        if (needs_hidden_readback(config_)) {
            result.hidden = std::move(hidden);
        }
        return result;
    }

    core::ExecutionContext & execution_;
    std::shared_ptr<const assets::TensorSource> source_;
    Qwen3VlEncoderRuntimeConfig config_;
    std::unique_ptr<Qwen3VlEncoderWeightStore> full_weights_;
    std::unique_ptr<EncoderGraph> graph_;
};

Qwen3VlEncoderRuntime::Qwen3VlEncoderRuntime(
    core::ExecutionContext & execution,
    std::shared_ptr<const assets::TensorSource> tensor_source,
    Qwen3VlEncoderRuntimeConfig config)
    : impl_(std::make_unique<Impl>(execution, std::move(tensor_source), std::move(config))) {}

Qwen3VlEncoderRuntime::~Qwen3VlEncoderRuntime() = default;

Qwen3VlEncoderResult Qwen3VlEncoderRuntime::encode_text(
    const std::vector<int32_t> & ids,
    Qwen3VlEncoderOptions options) {
    return impl_->encode_text(ids, options);
}

void Qwen3VlEncoderRuntime::release_runtime_graphs() {
    impl_->release_runtime_graphs();
}

void Qwen3VlEncoderRuntime::release_weights() {
    impl_->release_weights();
}

}  // namespace engine::modules
