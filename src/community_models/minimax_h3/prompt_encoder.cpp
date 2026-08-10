#include "engine/community_models/minimax_h3/prompt_encoder.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/grouped_query_attention.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/optimizations/fast_projection_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include "ggml-alloc.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>

namespace engine::models::minimax_h3 {

namespace assets = engine::assets;
namespace core = engine::core;
namespace modules = engine::modules;

bool matches_prompt_weight_filter(
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

struct PromptGgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

PromptEncoderWeightStore::PromptEncoderWeightStore(
    core::ExecutionContext & execution_context,
    std::shared_ptr<const assets::TensorSource> tensor_source,
    size_t weight_context_bytes)
    : execution(execution_context),
      source_(std::move(tensor_source)),
      store_(execution.backend(), execution.backend_type(), "minimax_h3.prompt_encoder.weights", weight_context_bytes) {
    if (source_ == nullptr) {
        throw std::runtime_error("MiniMax-H3 prompt encoder tensor source is missing");
    }
    for (const auto & meta : source_->tensors()) {
        weights_.emplace(
            meta.name,
            store_.load_tensor(*source_, meta.name, assets::TensorStorageType::Native, meta.shape));
    }
    store_.upload();
    source_->release_storage();
}

PromptEncoderWeightStore::PromptEncoderWeightStore(
    core::ExecutionContext & execution_context,
    std::shared_ptr<const assets::TensorSource> tensor_source,
    size_t weight_context_bytes,
    const std::vector<std::string> & required_names,
    const std::vector<std::string> & prefix_filters)
    : execution(execution_context),
      source_(std::move(tensor_source)),
      store_(execution.backend(), execution.backend_type(), "minimax_h3.prompt_encoder.layerwise.weights", weight_context_bytes) {
    if (source_ == nullptr) {
        throw std::runtime_error("MiniMax-H3 prompt encoder tensor source is missing");
    }
    for (const auto & meta : source_->tensors()) {
        if (!matches_prompt_weight_filter(meta.name, required_names, prefix_filters)) {
            continue;
        }
        weights_.emplace(
            meta.name,
            store_.load_tensor(*source_, meta.name, assets::TensorStorageType::Native, meta.shape));
    }
    for (const auto & required : required_names) {
        if (weights_.find(required) == weights_.end()) {
            throw std::runtime_error("missing MiniMax-H3 prompt encoder layerwise tensor: " + required);
        }
    }
    store_.upload();
    source_->release_storage();
}

const core::TensorValue & PromptEncoderWeightStore::require(std::string_view name) const {
    const auto it = weights_.find(std::string(name));
    if (it == weights_.end()) {
        throw std::runtime_error("missing MiniMax-H3 prompt encoder tensor: " + std::string(name));
    }
    return it->second;
}

core::TensorValue prompt_repeat_like(core::ModuleBuildContext & ctx, const core::TensorValue & value, const core::TensorValue & like) {
    auto source = value;
    if (source.tensor->type != GGML_TYPE_F32 && source.tensor->type != GGML_TYPE_F16) {
        source = core::wrap_tensor(ggml_cast(ctx.ggml, source.tensor, GGML_TYPE_F16), source.shape, GGML_TYPE_F16);
    }
    auto repeated = modules::RepeatModule({like.shape}).build(ctx, source);
    return core::wrap_tensor(ggml_cont(ctx.ggml, repeated.tensor), repeated.shape, repeated.type);
}

core::TensorValue prompt_linear_projection(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & x,
    const core::TensorValue & weight,
    int64_t in_features,
    int64_t out_features) {
    modules::LinearWeights params{weight, std::nullopt};
    const ggml_prec precision = ggml_is_quantized(weight.tensor->type) ? GGML_PREC_DEFAULT : GGML_PREC_F32;
    const bool use_fast_projection = ctx.backend_type == core::BackendType::Cuda && out_features % 4 == 0;
    return use_fast_projection
        ? modules::FastPackedProjection4Module({in_features, out_features, precision}).build(ctx, x, params)
        : modules::LinearModule({in_features, out_features, false, precision}).build(ctx, x, params);
}

core::TensorValue prompt_apply_rope(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & x,
    const core::TensorValue & cos,
    const core::TensorValue & sin,
    int64_t rot_dim) {
    const int axis = static_cast<int>(x.shape.rank - 1);
    auto x_rot = core::ensure_backend_addressable_layout(ctx, modules::SliceModule({axis, 0, rot_dim}).build(ctx, x));
    auto x_pass = rot_dim == x.shape.last_dim() ? core::TensorValue{} : modules::SliceModule({axis, rot_dim, x.shape.last_dim() - rot_dim}).build(ctx, x);
    auto x1 = core::ensure_backend_addressable_layout(ctx, modules::SliceModule({axis, 0, rot_dim / 2}).build(ctx, x_rot));
    auto x2 = core::ensure_backend_addressable_layout(ctx, modules::SliceModule({axis, rot_dim / 2, rot_dim / 2}).build(ctx, x_rot));
    auto cos_half = modules::SliceModule({static_cast<int>(cos.shape.rank - 1), 0, rot_dim / 2}).build(ctx, cos);
    auto sin_half = modules::SliceModule({static_cast<int>(sin.shape.rank - 1), 0, rot_dim / 2}).build(ctx, sin);
    cos_half = prompt_repeat_like(ctx, cos_half, x1);
    sin_half = prompt_repeat_like(ctx, sin_half, x1);
    auto x1_cos = modules::MulModule{}.build(
        ctx,
        core::ensure_backend_addressable_layout(ctx, x1),
        core::ensure_backend_addressable_layout(ctx, cos_half));
    auto x2_sin = modules::MulModule{}.build(
        ctx,
        core::ensure_backend_addressable_layout(ctx, x2),
        core::ensure_backend_addressable_layout(ctx, sin_half));
    auto first = core::wrap_tensor(ggml_sub(ctx.ggml, x1_cos.tensor, x2_sin.tensor), x1.shape, GGML_TYPE_F32);
    auto x2_cos = modules::MulModule{}.build(
        ctx,
        core::ensure_backend_addressable_layout(ctx, x2),
        core::ensure_backend_addressable_layout(ctx, cos_half));
    auto x1_sin = modules::MulModule{}.build(
        ctx,
        core::ensure_backend_addressable_layout(ctx, x1),
        core::ensure_backend_addressable_layout(ctx, sin_half));
    auto second = modules::AddModule{}.build(
        ctx,
        core::ensure_backend_addressable_layout(ctx, x2_cos),
        core::ensure_backend_addressable_layout(ctx, x1_sin));
    auto out_rot = modules::ConcatModule({axis}).build(ctx, first, second);
    if (!x_pass.valid()) {
        return out_rot;
    }
    return modules::ConcatModule({axis}).build(ctx, out_rot, x_pass);
}

std::vector<float> prompt_rope_values(const MiniMaxH3Config & cfg, int64_t tokens, bool cosine) {
    std::vector<float> out(static_cast<size_t>(tokens * cfg.prompt_head_dim));
    const int64_t half = cfg.prompt_head_dim / 2;
    for (int64_t t = 0; t < tokens; ++t) {
        for (int64_t i = 0; i < half; ++i) {
            const float inv = 1.0F / std::pow(cfg.prompt_rope_theta, static_cast<float>(2 * i) / static_cast<float>(cfg.prompt_head_dim));
            const float v = cosine ? std::cos(static_cast<float>(t) * inv) : std::sin(static_cast<float>(t) * inv);
            out[static_cast<size_t>(t * cfg.prompt_head_dim + i)] = v;
            out[static_cast<size_t>(t * cfg.prompt_head_dim + half + i)] = v;
        }
    }
    return out;
}

core::TensorValue prompt_mlp(
    core::ModuleBuildContext & ctx,
    const PromptEncoderWeightStore & weights,
    const MiniMaxH3Config & cfg,
    const core::TensorValue & x,
    int64_t layer) {
    const std::string p = "model.language_model.layers." + std::to_string(layer) + ".mlp.";
    auto gate = prompt_linear_projection(ctx, x, weights.require(p + "gate_proj.weight"), cfg.prompt_hidden, cfg.prompt_intermediate);
    auto up = prompt_linear_projection(ctx, x, weights.require(p + "up_proj.weight"), cfg.prompt_hidden, cfg.prompt_intermediate);
    auto hidden = modules::MulModule{}.build(
        ctx,
        core::ensure_backend_addressable_layout(ctx, modules::SiluModule{}.build(ctx, gate)),
        core::ensure_backend_addressable_layout(ctx, up));
    return prompt_linear_projection(ctx, hidden, weights.require(p + "down_proj.weight"), cfg.prompt_intermediate, cfg.prompt_hidden);
}

core::TensorValue prompt_attention(
    core::ModuleBuildContext & ctx,
    const PromptEncoderWeightStore & weights,
    const MiniMaxH3Config & cfg,
    const core::TensorValue & x,
    const core::TensorValue & cos,
    const core::TensorValue & sin,
    int64_t layer) {
    const std::string p = "model.language_model.layers." + std::to_string(layer) + ".self_attn.";
    const int64_t tokens = x.shape.dims[1];
    auto q = prompt_linear_projection(ctx, x, weights.require(p + "q_proj.weight"), cfg.prompt_hidden, cfg.prompt_heads * cfg.prompt_head_dim);
    auto k = prompt_linear_projection(ctx, x, weights.require(p + "k_proj.weight"), cfg.prompt_hidden, cfg.prompt_kv_heads * cfg.prompt_head_dim);
    auto v = prompt_linear_projection(ctx, x, weights.require(p + "v_proj.weight"), cfg.prompt_hidden, cfg.prompt_kv_heads * cfg.prompt_head_dim);
    q = core::reshape_tensor(ctx, q, core::TensorShape::from_dims({1, tokens, cfg.prompt_heads, cfg.prompt_head_dim}));
    k = core::reshape_tensor(ctx, k, core::TensorShape::from_dims({1, tokens, cfg.prompt_kv_heads, cfg.prompt_head_dim}));
    v = core::reshape_tensor(ctx, v, core::TensorShape::from_dims({1, tokens, cfg.prompt_kv_heads, cfg.prompt_head_dim}));
    q = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, q);
    k = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, k);
    v = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, v);
    q = modules::RMSNormModule({cfg.prompt_head_dim, cfg.prompt_eps, true, false}).build(
        ctx, q, {weights.require(p + "q_norm.weight"), std::nullopt});
    k = modules::RMSNormModule({cfg.prompt_head_dim, cfg.prompt_eps, true, false}).build(
        ctx, k, {weights.require(p + "k_norm.weight"), std::nullopt});
    q = prompt_apply_rope(ctx, q, cos, sin, cfg.prompt_head_dim);
    k = prompt_apply_rope(ctx, k, cos, sin, cfg.prompt_head_dim);
    auto h = modules::GroupedQueryAttentionModule({
        cfg.prompt_head_dim,
        modules::GroupedQueryAttentionLowering::ManualRepeat,
        GGML_PREC_DEFAULT,
        modules::AttentionCausality::Causal}).build(ctx, q, k, v);
    h = core::ensure_backend_addressable_layout(ctx, h);
    h = core::reshape_tensor(ctx, h, core::TensorShape::from_dims({1, tokens, cfg.prompt_heads * cfg.prompt_head_dim}));
    return prompt_linear_projection(ctx, h, weights.require(p + "o_proj.weight"), cfg.prompt_heads * cfg.prompt_head_dim, cfg.prompt_hidden);
}

core::TensorValue build_prompt_encoder(
    core::ModuleBuildContext & ctx,
    const PromptEncoderWeightStore & weights,
    const MiniMaxH3Config & cfg,
    const core::TensorValue & ids,
    const core::TensorValue & cos,
    const core::TensorValue & sin) {
    auto x = modules::EmbeddingModule({cfg.vocab_size, cfg.prompt_hidden}).build(
        ctx, ids, weights.require("model.language_model.embed_tokens.weight"));
    x = core::reshape_tensor(ctx, x, core::TensorShape::from_dims({1, ids.shape.dims[0], cfg.prompt_hidden}));
    for (int64_t layer = 0; layer < cfg.prompt_layers; ++layer) {
        const std::string p = "model.language_model.layers." + std::to_string(layer) + ".";
        auto h = modules::RMSNormModule({cfg.prompt_hidden, cfg.prompt_eps, true, false}).build(
            ctx, x, {weights.require(p + "input_layernorm.weight"), std::nullopt});
        auto attn = prompt_attention(ctx, weights, cfg, h, cos, sin, layer);
        x = modules::AddModule{}.build(
            ctx,
            core::ensure_backend_addressable_layout(ctx, x),
            core::ensure_backend_addressable_layout(ctx, attn));
        h = modules::RMSNormModule({cfg.prompt_hidden, cfg.prompt_eps, true, false}).build(
            ctx, x, {weights.require(p + "post_attention_layernorm.weight"), std::nullopt});
        auto mlp = prompt_mlp(ctx, weights, cfg, h, layer);
        x = modules::AddModule{}.build(
            ctx,
            core::ensure_backend_addressable_layout(ctx, x),
            core::ensure_backend_addressable_layout(ctx, mlp));
    }
    return core::reshape_tensor(ctx, x, core::TensorShape::from_dims({ids.shape.dims[0], cfg.prompt_hidden}));
}

class PromptEncoderGraph {
public:
    PromptEncoderGraph(PromptEncoderWeightStore & weights, const MiniMaxH3Config & cfg, int64_t tokens)
        : execution_(weights.execution) {
        ctx_.reset(ggml_init({512 * 1024 * 1024, nullptr, true}));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize MiniMax-H3 prompt encoder graph context");
        }
        input_ctx_.reset(ggml_init({16 * 1024 * 1024, nullptr, true}));
        if (input_ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize MiniMax-H3 prompt encoder input context");
        }
        core::ModuleBuildContext input_ctx{input_ctx_.get(), "minimax_h3.prompt_encoder.inputs", execution_.backend_type()};
        ids_ = core::make_tensor(input_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({tokens}));
        cos_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 1, tokens, cfg.prompt_head_dim}));
        sin_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 1, tokens, cfg.prompt_head_dim}));
        ggml_set_input(ids_.tensor);
        ggml_set_input(cos_.tensor);
        ggml_set_input(sin_.tensor);

        core::ModuleBuildContext build_ctx{ctx_.get(), "minimax_h3.prompt_encoder", execution_.backend_type()};
        output_ = build_prompt_encoder(build_ctx, weights, cfg, ids_, cos_, sin_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 262144, false);
        ggml_set_output(output_.tensor);
        ggml_build_forward_expand(graph_, output_.tensor);
        input_buffer_ = ggml_backend_alloc_ctx_tensors(input_ctx_.get(), execution_.backend());
        if (input_buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate MiniMax-H3 prompt encoder inputs");
        }
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend()));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("failed to allocate MiniMax-H3 prompt encoder graph");
        }
        core::prepare_host_graph_plan(execution_, graph_, plan_);
    }

    ~PromptEncoderGraph() {
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

    std::vector<float> run(const MiniMaxH3Config & cfg, const std::vector<int32_t> & ids) {
        core::write_tensor_i32(ids_, ids);
        core::write_tensor_f32(cos_, prompt_rope_values(cfg, ids.size(), true));
        core::write_tensor_f32(sin_, prompt_rope_values(cfg, ids.size(), false));
        core::set_backend_threads(execution_.backend(), 8);
        const ggml_status status = core::compute_graph(execution_, graph_, plan_, "minimax_h3.prompt_encoder");
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("MiniMax-H3 prompt encoder graph compute failed");
        }
        return core::read_tensor_f32(output_.tensor);
    }

private:
    core::ExecutionContext & execution_;
    std::unique_ptr<ggml_context, PromptGgmlContextDeleter> ctx_;
    std::unique_ptr<ggml_context, PromptGgmlContextDeleter> input_ctx_;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    ggml_backend_buffer_t input_buffer_ = nullptr;
    core::HostGraphPlan plan_;
    core::TensorValue ids_;
    core::TensorValue cos_;
    core::TensorValue sin_;
    core::TensorValue output_;
};

class PromptEmbeddingGraph {
public:
    PromptEmbeddingGraph(PromptEncoderWeightStore & weights, const MiniMaxH3Config & cfg, int64_t tokens)
        : execution_(weights.execution),
          tokens_(tokens),
          cfg_(cfg) {
        ctx_.reset(ggml_init({128 * 1024 * 1024, nullptr, true}));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize MiniMax-H3 prompt embedding graph context");
        }
        input_ctx_.reset(ggml_init({16 * 1024 * 1024, nullptr, true}));
        if (input_ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize MiniMax-H3 prompt embedding input context");
        }
        core::ModuleBuildContext input_ctx{input_ctx_.get(), "minimax_h3.prompt_embedding.inputs", execution_.backend_type()};
        ids_ = core::make_tensor(input_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({tokens_}));
        ggml_set_input(ids_.tensor);
        core::ModuleBuildContext build_ctx{ctx_.get(), "minimax_h3.prompt_embedding", execution_.backend_type()};
        auto x = modules::EmbeddingModule({cfg_.vocab_size, cfg_.prompt_hidden}).build(
            build_ctx, ids_, weights.require("model.language_model.embed_tokens.weight"));
        output_ = core::reshape_tensor(build_ctx, x, core::TensorShape::from_dims({1, tokens_, cfg_.prompt_hidden}));
        graph_ = ggml_new_graph_custom(ctx_.get(), 65536, false);
        ggml_set_output(output_.tensor);
        ggml_build_forward_expand(graph_, output_.tensor);
        input_buffer_ = ggml_backend_alloc_ctx_tensors(input_ctx_.get(), execution_.backend());
        if (input_buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate MiniMax-H3 prompt embedding inputs");
        }
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend()));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("failed to allocate MiniMax-H3 prompt embedding graph");
        }
        core::prepare_host_graph_plan(execution_, graph_, plan_);
    }

    ~PromptEmbeddingGraph() {
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
        core::set_backend_threads(execution_.backend(), 8);
        const ggml_status status = core::compute_graph(execution_, graph_, plan_, "minimax_h3.prompt_embedding");
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("MiniMax-H3 prompt embedding graph compute failed");
        }
        return core::read_tensor_f32(output_.tensor);
    }

private:
    core::ExecutionContext & execution_;
    int64_t tokens_ = 0;
    MiniMaxH3Config cfg_;
    std::unique_ptr<ggml_context, PromptGgmlContextDeleter> ctx_;
    std::unique_ptr<ggml_context, PromptGgmlContextDeleter> input_ctx_;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    ggml_backend_buffer_t input_buffer_ = nullptr;
    core::HostGraphPlan plan_;
    core::TensorValue ids_;
    core::TensorValue output_;
};

core::TensorValue build_prompt_layer_group(
    core::ModuleBuildContext & ctx,
    const PromptEncoderWeightStore & weights,
    const MiniMaxH3Config & cfg,
    const core::TensorValue & input,
    const core::TensorValue & cos,
    const core::TensorValue & sin,
    int64_t layer_begin,
    int64_t layer_end) {
    auto x = input;
    for (int64_t layer = layer_begin; layer < layer_end; ++layer) {
        const std::string p = "model.language_model.layers." + std::to_string(layer) + ".";
        auto h = modules::RMSNormModule({cfg.prompt_hidden, cfg.prompt_eps, true, false}).build(
            ctx, x, {weights.require(p + "input_layernorm.weight"), std::nullopt});
        auto attn = prompt_attention(ctx, weights, cfg, h, cos, sin, layer);
        x = modules::AddModule{}.build(
            ctx,
            core::ensure_backend_addressable_layout(ctx, x),
            core::ensure_backend_addressable_layout(ctx, attn));
        h = modules::RMSNormModule({cfg.prompt_hidden, cfg.prompt_eps, true, false}).build(
            ctx, x, {weights.require(p + "post_attention_layernorm.weight"), std::nullopt});
        auto mlp = prompt_mlp(ctx, weights, cfg, h, layer);
        x = modules::AddModule{}.build(
            ctx,
            core::ensure_backend_addressable_layout(ctx, x),
            core::ensure_backend_addressable_layout(ctx, mlp));
    }
    return x;
}

class PromptLayerGroupGraph {
public:
    PromptLayerGroupGraph(
        PromptEncoderWeightStore & weights,
        const MiniMaxH3Config & cfg,
        int64_t tokens,
        int64_t layer_begin,
        int64_t layer_end)
        : execution_(weights.execution),
          cfg_(cfg),
          tokens_(tokens) {
        ctx_.reset(ggml_init({512 * 1024 * 1024, nullptr, true}));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize MiniMax-H3 prompt layerwise graph context");
        }
        input_ctx_.reset(ggml_init({32 * 1024 * 1024, nullptr, true}));
        if (input_ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize MiniMax-H3 prompt layerwise input context");
        }
        core::ModuleBuildContext input_ctx{input_ctx_.get(), "minimax_h3.prompt_layerwise.inputs", execution_.backend_type()};
        hidden_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, tokens_, cfg_.prompt_hidden}));
        cos_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 1, tokens_, cfg_.prompt_head_dim}));
        sin_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 1, tokens_, cfg_.prompt_head_dim}));
        ggml_set_input(hidden_.tensor);
        ggml_set_input(cos_.tensor);
        ggml_set_input(sin_.tensor);
        core::ModuleBuildContext build_ctx{ctx_.get(), "minimax_h3.prompt_layerwise", execution_.backend_type()};
        output_ = build_prompt_layer_group(build_ctx, weights, cfg_, hidden_, cos_, sin_, layer_begin, layer_end);
        graph_ = ggml_new_graph_custom(ctx_.get(), 262144, false);
        ggml_set_output(output_.tensor);
        ggml_build_forward_expand(graph_, output_.tensor);
        input_buffer_ = ggml_backend_alloc_ctx_tensors(input_ctx_.get(), execution_.backend());
        if (input_buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate MiniMax-H3 prompt layerwise inputs");
        }
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend()));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("failed to allocate MiniMax-H3 prompt layerwise graph");
        }
        core::prepare_host_graph_plan(execution_, graph_, plan_);
    }

    ~PromptLayerGroupGraph() {
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

    std::vector<float> run(const std::vector<float> & hidden, const std::vector<float> & cos, const std::vector<float> & sin) {
        core::write_tensor_f32(hidden_, hidden);
        core::write_tensor_f32(cos_, cos);
        core::write_tensor_f32(sin_, sin);
        core::set_backend_threads(execution_.backend(), 8);
        const ggml_status status = core::compute_graph(execution_, graph_, plan_, "minimax_h3.prompt_layerwise");
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("MiniMax-H3 prompt layerwise graph compute failed");
        }
        return core::read_tensor_f32(output_.tensor);
    }

private:
    core::ExecutionContext & execution_;
    MiniMaxH3Config cfg_;
    int64_t tokens_ = 0;
    std::unique_ptr<ggml_context, PromptGgmlContextDeleter> ctx_;
    std::unique_ptr<ggml_context, PromptGgmlContextDeleter> input_ctx_;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    ggml_backend_buffer_t input_buffer_ = nullptr;
    core::HostGraphPlan plan_;
    core::TensorValue hidden_;
    core::TensorValue cos_;
    core::TensorValue sin_;
    core::TensorValue output_;
};

std::vector<float> run_prompt_graph(PromptEncoderWeightStore & weights, const MiniMaxH3Config & cfg, const std::vector<int32_t> & ids) {
    PromptEncoderGraph graph(weights, cfg, static_cast<int64_t>(ids.size()));
    return graph.run(cfg, ids);
}

std::vector<float> run_prompt_graph_layerwise(
    core::ExecutionContext & execution,
    std::shared_ptr<const assets::TensorSource> tensor_source,
    const MiniMaxH3Config & cfg,
    const std::vector<int32_t> & ids,
    size_t weight_context_bytes,
    int64_t layer_batch) {
    const int64_t batch = std::max<int64_t>(1, layer_batch);
    const int64_t tokens = static_cast<int64_t>(ids.size());
    std::vector<float> hidden;
    {
        PromptEncoderWeightStore weights(
            execution,
            tensor_source,
            weight_context_bytes,
            {"model.language_model.embed_tokens.weight"},
            {});
        PromptEmbeddingGraph graph(weights, cfg, tokens);
        hidden = graph.run(ids);
    }
    const auto cos = prompt_rope_values(cfg, tokens, true);
    const auto sin = prompt_rope_values(cfg, tokens, false);
    for (int64_t layer = 0; layer < cfg.prompt_layers; layer += batch) {
        const int64_t end = std::min<int64_t>(layer + batch, cfg.prompt_layers);
        std::vector<std::string> prefixes;
        for (int64_t i = layer; i < end; ++i) {
            prefixes.push_back("model.language_model.layers." + std::to_string(i) + ".");
        }
        PromptEncoderWeightStore weights(execution, tensor_source, weight_context_bytes, {}, prefixes);
        PromptLayerGroupGraph graph(weights, cfg, tokens, layer, end);
        hidden = graph.run(hidden, cos, sin);
    }
    std::vector<float> out(static_cast<size_t>(tokens * cfg.prompt_hidden));
    std::memcpy(out.data(), hidden.data(), out.size() * sizeof(float));
    return out;
}

}  // namespace engine::models::minimax_h3
