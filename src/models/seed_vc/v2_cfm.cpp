#include "engine/models/seed_vc/v2_cfm.h"

#include "engine/models/seed_vc/assets.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <cmath>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

namespace engine::models::seed_vc {
namespace {

using engine::core::TensorShape;
using engine::core::TensorValue;

TensorValue require_tensor(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & name,
    engine::assets::TensorStorageType storage_type) {
    return engine::modules::binding::tensor_from_named_source(
        store,
        source,
        name,
        seed_vc_component_storage_type(source, name, storage_type));
}

engine::modules::LinearWeights linear_weights(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    engine::assets::TensorStorageType storage_type) {
    return engine::modules::LinearWeights{
        require_tensor(store, source, prefix + ".weight", storage_type),
        require_tensor(store, source, prefix + ".bias", storage_type)};
}

engine::modules::LinearWeights linear_weights_no_bias(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    engine::assets::TensorStorageType storage_type) {
    return engine::modules::LinearWeights{
        require_tensor(store, source, prefix + ".weight", storage_type),
        std::nullopt};
}

engine::modules::NormWeights rms_weight(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & name,
    engine::assets::TensorStorageType storage_type) {
    return engine::modules::NormWeights{require_tensor(store, source, name, storage_type), std::nullopt};
}

TensorValue contiguous(engine::core::ModuleBuildContext & ctx, const TensorValue & value) {
    return engine::core::ensure_backend_addressable_layout(ctx, value);
}

TensorValue add(engine::core::ModuleBuildContext & ctx, const TensorValue & lhs, const TensorValue & rhs) {
    return engine::modules::AddModule{}.build(ctx, lhs, rhs);
}

TensorValue mul(engine::core::ModuleBuildContext & ctx, const TensorValue & lhs, const TensorValue & rhs) {
    return engine::modules::MulModule{}.build(ctx, lhs, rhs);
}

TensorValue add_one(engine::core::ModuleBuildContext & ctx, const TensorValue & value) {
    return engine::core::wrap_tensor(
        ggml_scale_bias(ctx.ggml, value.tensor, 1.0F, 1.0F),
        value.shape,
        GGML_TYPE_F32);
}

TensorValue slice_last(
    engine::core::ModuleBuildContext & ctx,
    const TensorValue & input,
    int64_t start,
    int64_t length) {
    return engine::modules::SliceModule({static_cast<int>(input.shape.rank - 1), start, length}).build(ctx, input);
}

TensorValue expand_batch_token(
    engine::core::ModuleBuildContext & ctx,
    const TensorValue & value,
    int64_t batch,
    int64_t tokens,
    int64_t channels) {
    auto reshaped = engine::core::reshape_tensor(ctx, contiguous(ctx, value), TensorShape::from_dims({batch, 1, channels}));
    return engine::modules::RepeatModule({TensorShape::from_dims({batch, tokens, channels})}).build(ctx, reshaped);
}

TensorValue timestep_frequency_embedding(
    engine::core::ModuleBuildContext & ctx,
    ggml_tensor * freqs_tensor,
    const TensorValue & timestep) {
    const int64_t batch = timestep.shape.dims[0];
    auto freqs = engine::core::wrap_tensor(freqs_tensor, TensorShape::from_dims({128}), GGML_TYPE_F32);
    auto freqs_b = engine::core::reshape_tensor(ctx, freqs, TensorShape::from_dims({1, 128}));
    freqs_b = engine::modules::RepeatModule({TensorShape::from_dims({batch, 128})}).build(ctx, freqs_b);
    auto t = engine::core::reshape_tensor(ctx, timestep, TensorShape::from_dims({batch, 1}));
    t = engine::modules::RepeatModule({TensorShape::from_dims({batch, 128})}).build(ctx, t);
    auto args = mul(ctx, engine::core::wrap_tensor(ggml_scale(ctx.ggml, t.tensor, 1000.0F), t.shape, GGML_TYPE_F32), freqs_b);
    auto cos_part = engine::core::wrap_tensor(ggml_cos(ctx.ggml, args.tensor), args.shape, GGML_TYPE_F32);
    auto sin_part = engine::core::wrap_tensor(ggml_sin(ctx.ggml, args.tensor), args.shape, GGML_TYPE_F32);
    return engine::modules::ConcatModule({1}).build(ctx, cos_part, sin_part);
}

std::vector<float> make_timestep_freqs() {
    std::vector<float> out(128, 0.0F);
    for (int64_t i = 0; i < 128; ++i) {
        out[static_cast<size_t>(i)] =
            std::exp(-std::log(10000.0F) * static_cast<float>(i) / 128.0F);
    }
    return out;
}

struct CfmLayerWeights {
    engine::modules::LinearWeights attention_wqkv;
    engine::modules::LinearWeights attention_wo;
    engine::modules::LinearWeights attention_norm_linear;
    engine::modules::NormWeights attention_norm;
    engine::modules::NormWeights ffn_norm;
    engine::modules::LinearWeights ff_w1;
    engine::modules::LinearWeights ff_w2;
    engine::modules::LinearWeights ff_w3;
};

struct CfmWeights {
    engine::modules::LinearWeights cond_projection;
    engine::modules::LinearWeights cond_x_merge_linear;
    engine::modules::LinearWeights style_in;
    engine::modules::LinearWeights t_embedder_0;
    engine::modules::LinearWeights t_embedder_2;
    engine::modules::LinearWeights final_mlp_0;
    engine::modules::LinearWeights final_mlp_2;
    std::vector<CfmLayerWeights> layers;
    engine::modules::LinearWeights final_norm_linear;
    engine::modules::NormWeights final_norm;
};

CfmWeights load_cfm_weights(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    int64_t layers,
    engine::assets::TensorStorageType storage_type) {
    const std::string root = "cfm.estimator.";
    CfmWeights out;
    out.cond_projection = linear_weights(store, source, root + "cond_projection", storage_type);
    out.cond_x_merge_linear = linear_weights(store, source, root + "cond_x_merge_linear", storage_type);
    out.style_in = linear_weights(store, source, root + "style_in", storage_type);
    out.t_embedder_0 = linear_weights(store, source, root + "t_embedder.mlp.0", storage_type);
    out.t_embedder_2 = linear_weights(store, source, root + "t_embedder.mlp.2", storage_type);
    out.final_mlp_0 = linear_weights(store, source, root + "final_mlp.0", storage_type);
    out.final_mlp_2 = linear_weights(store, source, root + "final_mlp.2", storage_type);
    out.layers.reserve(static_cast<size_t>(layers));
    for (int64_t layer = 0; layer < layers; ++layer) {
        const std::string prefix = root + "transformer.layers." + std::to_string(layer);
        CfmLayerWeights item;
        item.attention_wqkv = linear_weights_no_bias(store, source, prefix + ".attention.wqkv", storage_type);
        item.attention_wo = linear_weights_no_bias(store, source, prefix + ".attention.wo", storage_type);
        item.attention_norm_linear = linear_weights(store, source, prefix + ".attention_norm.linear", storage_type);
        item.attention_norm = rms_weight(store, source, prefix + ".attention_norm.norm.weight", storage_type);
        item.ffn_norm = rms_weight(store, source, prefix + ".ffn_norm.weight", storage_type);
        item.ff_w1 = linear_weights_no_bias(store, source, prefix + ".feed_forward.w1", storage_type);
        item.ff_w2 = linear_weights_no_bias(store, source, prefix + ".feed_forward.w2", storage_type);
        item.ff_w3 = linear_weights_no_bias(store, source, prefix + ".feed_forward.w3", storage_type);
        out.layers.push_back(item);
    }
    out.final_norm_linear = linear_weights(store, source, root + "transformer.norm.linear", storage_type);
    out.final_norm = rms_weight(store, source, root + "transformer.norm.norm.weight", storage_type);
    return out;
}

TensorValue build_timestep_embedding(
    engine::core::ModuleBuildContext & ctx,
    ggml_tensor * freqs_tensor,
    const TensorValue & timestep,
    const CfmWeights & weights,
    const SeedVcV2DitConfig & config) {
    auto hidden = timestep_frequency_embedding(ctx, freqs_tensor, timestep);
    hidden = engine::modules::LinearModule({256, config.hidden_dim, true, GGML_PREC_F32}).build(ctx, hidden, weights.t_embedder_0);
    hidden = engine::modules::SiluModule{}.build(ctx, hidden);
    return engine::modules::LinearModule({config.hidden_dim, config.hidden_dim, true, GGML_PREC_F32})
        .build(ctx, hidden, weights.t_embedder_2);
}

struct AdaptiveNormParts {
    TensorValue normalized;
    TensorValue gate_msa;
    TensorValue shift_mlp;
    TensorValue scale_mlp;
    TensorValue gate_mlp;
};

AdaptiveNormParts adaptive_attention_norm(
    engine::core::ModuleBuildContext & ctx,
    const TensorValue & input,
    const TensorValue & timestep_embedding,
    const CfmLayerWeights & weights,
    const SeedVcV2DitConfig & config) {
    const int64_t batch = input.shape.dims[0];
    const int64_t tokens = input.shape.dims[1];
    auto emb = engine::modules::SiluModule{}.build(ctx, timestep_embedding);
    emb = engine::modules::LinearModule({config.hidden_dim, config.hidden_dim * 6, true, GGML_PREC_F32})
              .build(ctx, emb, weights.attention_norm_linear);
    auto shift_msa = expand_batch_token(ctx, slice_last(ctx, emb, 0, config.hidden_dim), batch, tokens, config.hidden_dim);
    auto scale_msa = expand_batch_token(ctx, slice_last(ctx, emb, config.hidden_dim, config.hidden_dim), batch, tokens, config.hidden_dim);
    auto gate_msa = expand_batch_token(ctx, slice_last(ctx, emb, config.hidden_dim * 2, config.hidden_dim), batch, tokens, config.hidden_dim);
    auto shift_mlp = expand_batch_token(ctx, slice_last(ctx, emb, config.hidden_dim * 3, config.hidden_dim), batch, tokens, config.hidden_dim);
    auto scale_mlp = expand_batch_token(ctx, slice_last(ctx, emb, config.hidden_dim * 4, config.hidden_dim), batch, tokens, config.hidden_dim);
    auto gate_mlp = expand_batch_token(ctx, slice_last(ctx, emb, config.hidden_dim * 5, config.hidden_dim), batch, tokens, config.hidden_dim);
    auto normalized = engine::modules::RMSNormModule({config.hidden_dim, 1.0e-5F, true, false})
                          .build(ctx, input, weights.attention_norm);
    normalized = add(ctx, mul(ctx, normalized, add_one(ctx, scale_msa)), shift_msa);
    return {normalized, gate_msa, shift_mlp, scale_mlp, gate_mlp};
}

TensorValue build_attention(
    engine::core::ModuleBuildContext & ctx,
    const TensorValue & input,
    const TensorValue & positions,
    const CfmLayerWeights & weights,
    const SeedVcV2DitConfig & config) {
    const int64_t batch = input.shape.dims[0];
    const int64_t seq_len = input.shape.dims[1];
    const int64_t head_dim = config.hidden_dim / config.num_heads;
    auto qkv = engine::modules::LinearModule({config.hidden_dim, config.hidden_dim * 3, false, GGML_PREC_F32})
                   .build(ctx, input, weights.attention_wqkv);
    auto q = slice_last(ctx, qkv, 0, config.hidden_dim);
    auto k = slice_last(ctx, qkv, config.hidden_dim, config.hidden_dim);
    auto v = slice_last(ctx, qkv, config.hidden_dim * 2, config.hidden_dim);
    q = engine::core::reshape_tensor(ctx, contiguous(ctx, q), TensorShape::from_dims({batch, seq_len, config.num_heads, head_dim}));
    k = engine::core::reshape_tensor(ctx, contiguous(ctx, k), TensorShape::from_dims({batch, seq_len, config.num_heads, head_dim}));
    v = engine::core::reshape_tensor(ctx, contiguous(ctx, v), TensorShape::from_dims({batch, seq_len, config.num_heads, head_dim}));
    q = engine::modules::RoPEModule({head_dim, GGML_ROPE_TYPE_NORMAL, 10000.0F}).build(ctx, q, positions);
    k = engine::modules::RoPEModule({head_dim, GGML_ROPE_TYPE_NORMAL, 10000.0F}).build(ctx, k, positions);
    q = engine::modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, q);
    k = engine::modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, k);
    v = engine::modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, v);
    q = contiguous(ctx, q);
    k = contiguous(ctx, k);
    v = contiguous(ctx, v);
    auto * flash = ggml_flash_attn_ext(
        ctx.ggml,
        q.tensor,
        k.tensor,
        v.tensor,
        nullptr,
        1.0F / std::sqrt(static_cast<float>(head_dim)),
        0.0F,
        0.0F);
    ggml_flash_attn_ext_set_prec(flash, GGML_PREC_F32);
    auto context = engine::core::wrap_tensor(
        flash,
        TensorShape::from_dims({batch, seq_len, config.num_heads, head_dim}),
        GGML_TYPE_F32);
    context = engine::core::reshape_tensor(ctx, contiguous(ctx, context), TensorShape::from_dims({batch, seq_len, config.hidden_dim}));
    return engine::modules::LinearModule({config.hidden_dim, config.hidden_dim, false, GGML_PREC_F32})
        .build(ctx, context, weights.attention_wo);
}

TensorValue build_feed_forward(
    engine::core::ModuleBuildContext & ctx,
    const TensorValue & input,
    const CfmLayerWeights & weights,
    const SeedVcV2DitConfig & config) {
    auto gate = engine::modules::LinearModule({config.hidden_dim, config.hidden_dim * 3, false, GGML_PREC_F32})
                    .build(ctx, input, weights.ff_w1);
    gate = engine::modules::SiluModule{}.build(ctx, gate);
    auto up = engine::modules::LinearModule({config.hidden_dim, config.hidden_dim * 3, false, GGML_PREC_F32})
                  .build(ctx, input, weights.ff_w3);
    auto hidden = mul(ctx, gate, up);
    return engine::modules::LinearModule({config.hidden_dim * 3, config.hidden_dim, false, GGML_PREC_F32})
        .build(ctx, hidden, weights.ff_w2);
}

TensorValue build_transformer_layer(
    engine::core::ModuleBuildContext & ctx,
    const TensorValue & input,
    const TensorValue & timestep_embedding,
    const TensorValue & positions,
    const CfmLayerWeights & weights,
    const SeedVcV2DitConfig & config) {
    auto norm = adaptive_attention_norm(ctx, input, timestep_embedding, weights, config);
    auto attn = build_attention(ctx, norm.normalized, positions, weights, config);
    auto hidden = add(ctx, input, mul(ctx, norm.gate_msa, attn));
    auto ff_in = engine::modules::RMSNormModule({config.hidden_dim, 1.0e-5F, true, false})
                     .build(ctx, hidden, weights.ffn_norm);
    ff_in = add(ctx, mul(ctx, ff_in, add_one(ctx, norm.scale_mlp)), norm.shift_mlp);
    auto ff = build_feed_forward(ctx, ff_in, weights, config);
    return add(ctx, hidden, mul(ctx, norm.gate_mlp, ff));
}

TensorValue build_final_norm(
    engine::core::ModuleBuildContext & ctx,
    const TensorValue & input,
    const TensorValue & timestep_embedding,
    const CfmWeights & weights,
    const SeedVcV2DitConfig & config) {
    const int64_t batch = input.shape.dims[0];
    const int64_t tokens = input.shape.dims[1];
    auto emb = engine::modules::SiluModule{}.build(ctx, timestep_embedding);
    emb = engine::modules::LinearModule({config.hidden_dim, config.hidden_dim * 2, true, GGML_PREC_F32})
              .build(ctx, emb, weights.final_norm_linear);
    auto scale = expand_batch_token(ctx, slice_last(ctx, emb, 0, config.hidden_dim), batch, tokens, config.hidden_dim);
    auto shift = expand_batch_token(ctx, slice_last(ctx, emb, config.hidden_dim, config.hidden_dim), batch, tokens, config.hidden_dim);
    auto normalized = engine::modules::RMSNormModule({config.hidden_dim, 1.0e-5F, true, false})
                          .build(ctx, input, weights.final_norm);
    return add(ctx, mul(ctx, normalized, add_one(ctx, scale)), shift);
}

TensorValue build_estimator_graph(
    engine::core::ModuleBuildContext & ctx,
    const TensorValue & x_bct,
    const TensorValue & prompt_bct,
    const TensorValue & cond_btc,
    const TensorValue & style_bc,
    const TensorValue & timestep_b,
    const TensorValue & positions,
    ggml_tensor * freqs_tensor,
    const CfmWeights & weights,
    const SeedVcV2DitConfig & config) {
    auto t_emb = build_timestep_embedding(ctx, freqs_tensor, timestep_b, weights, config);
    auto cond = engine::modules::LinearModule({config.content_dim, config.hidden_dim, true, GGML_PREC_F32})
                    .build(ctx, cond_btc, weights.cond_projection);
    auto x = engine::modules::TransposeModule({{0, 2, 1}, 3}).build(ctx, x_bct);
    auto prompt = engine::modules::TransposeModule({{0, 2, 1}, 3}).build(ctx, prompt_bct);
    auto merged = engine::modules::ConcatModule({2}).build(ctx, x, prompt);
    merged = engine::modules::ConcatModule({2}).build(ctx, merged, cond);
    auto hidden = engine::modules::LinearModule({config.in_channels * 2 + config.hidden_dim, config.hidden_dim, true, GGML_PREC_F32})
                      .build(ctx, merged, weights.cond_x_merge_linear);
    auto style = engine::modules::LinearModule({config.style_encoder_dim, config.hidden_dim, true, GGML_PREC_F32})
                     .build(ctx, style_bc, weights.style_in);
    style = engine::core::reshape_tensor(ctx, contiguous(ctx, style), TensorShape::from_dims({style.shape.dims[0], 1, config.hidden_dim}));
    auto t_token = engine::core::reshape_tensor(ctx, contiguous(ctx, t_emb), TensorShape::from_dims({t_emb.shape.dims[0], 1, config.hidden_dim}));
    if (config.style_as_token) {
        hidden = engine::modules::ConcatModule({1}).build(ctx, style, hidden);
    }
    if (config.time_as_token) {
        hidden = engine::modules::ConcatModule({1}).build(ctx, t_token, hidden);
    }
    for (const auto & layer : weights.layers) {
        hidden = build_transformer_layer(ctx, hidden, t_token, positions, layer, config);
    }
    hidden = build_final_norm(ctx, hidden, t_token, weights, config);
    if (config.time_as_token) {
        hidden = engine::modules::SliceModule({1, 1, hidden.shape.dims[1] - 1}).build(ctx, hidden);
    }
    if (config.style_as_token) {
        hidden = engine::modules::SliceModule({1, 1, hidden.shape.dims[1] - 1}).build(ctx, hidden);
    }
    hidden = engine::modules::LinearModule({config.hidden_dim, config.hidden_dim, true, GGML_PREC_F32})
                 .build(ctx, hidden, weights.final_mlp_0);
    hidden = engine::modules::SiluModule{}.build(ctx, hidden);
    hidden = engine::modules::LinearModule({config.hidden_dim, config.in_channels, true, GGML_PREC_F32})
                 .build(ctx, hidden, weights.final_mlp_2);
    return engine::modules::TransposeModule({{0, 2, 1}, 3}).build(ctx, hidden);
}

class CfmEstimatorRunner {
public:
    CfmEstimatorRunner(
        engine::core::ExecutionContext & execution_context,
        CfmWeights weights,
        SeedVcV2DitConfig config)
        : execution_context_(execution_context),
          config_(std::move(config)),
          weights_(std::move(weights)) {
    }

    ~CfmEstimatorRunner() {
        release_graph();
    }

    SeedVcV2CfmEstimatorOutput run(const SeedVcV2CfmEstimatorInput & input) {
        std::lock_guard<std::mutex> lock(mutex_);
        validate_input(input);
        ensure_graph(input.batch, input.frames);
        engine::core::write_tensor_f32(x_, input.x);
        engine::core::write_tensor_f32(prompt_, input.prompt);
        engine::core::write_tensor_f32(cond_, input.cond);
        engine::core::write_tensor_f32(style_, input.style);
        engine::core::write_tensor_f32(timestep_, input.timestep);
        ggml_backend_tensor_set(freqs_, freqs_values_.data(), 0, freqs_values_.size() * sizeof(float));
        ggml_backend_tensor_set(positions_, position_values_.data(), 0, position_values_.size() * sizeof(int32_t));
        if (engine::core::compute_backend_graph(execution_context_.backend(), graph_) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("ggml_backend_graph_compute failed for Seed-VC V2 CFM estimator");
        }
        SeedVcV2CfmEstimatorOutput out;
        out.velocity = engine::core::read_tensor_f32(output_);
        out.batch = input.batch;
        out.channels = config_.in_channels;
        out.frames = input.frames;
        return out;
    }

private:
    void validate_input(const SeedVcV2CfmEstimatorInput & input) const {
        if (input.batch <= 0 || input.frames <= 0) {
            throw std::runtime_error("Seed-VC V2 CFM estimator requires positive batch and frame count");
        }
        if (static_cast<int64_t>(input.x.size()) != input.batch * config_.in_channels * input.frames ||
            static_cast<int64_t>(input.prompt.size()) != input.batch * config_.in_channels * input.frames ||
            static_cast<int64_t>(input.cond.size()) != input.batch * input.frames * config_.content_dim ||
            static_cast<int64_t>(input.style.size()) != input.batch * config_.style_encoder_dim ||
            static_cast<int64_t>(input.timestep.size()) != input.batch) {
            throw std::runtime_error("Seed-VC V2 CFM estimator input shape mismatch");
        }
    }

    void release_graph() {
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
            gallocr_ = nullptr;
        }
        if (ctx_ != nullptr) {
            ggml_free(ctx_);
            ctx_ = nullptr;
        }
        graph_ = nullptr;
        output_ = nullptr;
        x_ = {};
        prompt_ = {};
        cond_ = {};
        style_ = {};
        timestep_ = {};
        freqs_ = nullptr;
        positions_ = nullptr;
        batch_ = 0;
        frames_ = 0;
    }

    void ensure_graph(int64_t batch, int64_t frames) {
        if (ctx_ != nullptr && batch_ == batch && frames_ == frames) {
            return;
        }
        release_graph();
        ggml_init_params params{1024ull * 1024ull * 1024ull, nullptr, true};
        ctx_ = ggml_init(params);
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize Seed-VC V2 CFM estimator graph context");
        }
        engine::core::ModuleBuildContext ctx{ctx_, "seed_vc.v2_cfm.estimator", execution_context_.backend_type()};
        x_ = engine::core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({batch, config_.in_channels, frames}));
        prompt_ = engine::core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({batch, config_.in_channels, frames}));
        cond_ = engine::core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({batch, frames, config_.content_dim}));
        style_ = engine::core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({batch, config_.style_encoder_dim}));
        timestep_ = engine::core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({batch}));
        const int64_t seq_len = frames + (config_.time_as_token ? 1 : 0) + (config_.style_as_token ? 1 : 0);
        positions_ = ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, seq_len);
        freqs_ = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, 128);
        ggml_set_input(x_.tensor);
        ggml_set_input(prompt_.tensor);
        ggml_set_input(cond_.tensor);
        ggml_set_input(style_.tensor);
        ggml_set_input(timestep_.tensor);
        ggml_set_input(positions_);
        ggml_set_input(freqs_);
        auto output = build_estimator_graph(
            ctx,
            x_,
            prompt_,
            cond_,
            style_,
            timestep_,
            engine::core::wrap_tensor(positions_, TensorShape::from_dims({seq_len}), GGML_TYPE_I32),
            freqs_,
            weights_,
            config_);
        output_ = output.tensor;
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(ctx_, 131072, false);
        ggml_build_forward_expand(graph_, output_);
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_context_.backend()));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            release_graph();
            throw std::runtime_error("failed to allocate Seed-VC V2 CFM estimator graph memory");
        }
        freqs_values_ = make_timestep_freqs();
        position_values_.assign(static_cast<size_t>(seq_len), 0);
        for (int64_t i = 0; i < seq_len; ++i) {
            position_values_[static_cast<size_t>(i)] = static_cast<int32_t>(i);
        }
        batch_ = batch;
        frames_ = frames;
    }

    engine::core::ExecutionContext & execution_context_;
    SeedVcV2DitConfig config_;
    CfmWeights weights_;
    std::mutex mutex_;
    ggml_context * ctx_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    TensorValue x_;
    TensorValue prompt_;
    TensorValue cond_;
    TensorValue style_;
    TensorValue timestep_;
    ggml_tensor * freqs_ = nullptr;
    ggml_tensor * positions_ = nullptr;
    ggml_tensor * output_ = nullptr;
    std::vector<float> freqs_values_;
    std::vector<int32_t> position_values_;
    int64_t batch_ = 0;
    int64_t frames_ = 0;
};

}  // namespace

std::vector<float> build_seed_vc_t_span(int steps) {
    if (steps <= 0) {
        throw std::runtime_error("Seed-VC V2 CFM diffusion steps must be positive");
    }
    std::vector<float> out(static_cast<size_t>(steps + 1), 0.0F);
    for (int i = 0; i <= steps; ++i) {
        const float base = static_cast<float>(i) / static_cast<float>(steps);
        out[static_cast<size_t>(i)] =
            base - (std::cos(static_cast<float>(M_PI) * 0.5F * base) - 1.0F + base);
    }
    return out;
}

std::vector<float> zeros_like(size_t count) {
    return std::vector<float>(count, 0.0F);
}

void zero_prompt_region(std::vector<float> & values, int64_t batch, int64_t channels, int64_t frames, int64_t prompt_frames) {
    if (prompt_frames <= 0) {
        return;
    }
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t c = 0; c < channels; ++c) {
            const size_t base = static_cast<size_t>((b * channels + c) * frames);
            std::fill(
                values.begin() + static_cast<std::ptrdiff_t>(base),
                values.begin() + static_cast<std::ptrdiff_t>(base + static_cast<size_t>(prompt_frames)),
                0.0F);
        }
    }
}

std::vector<float> make_prompt_x(
    const std::vector<float> & prompt,
    int64_t batch,
    int64_t channels,
    int64_t frames,
    int64_t prompt_frames) {
    if (prompt_frames < 0 || prompt_frames > frames) {
        throw std::runtime_error("Seed-VC V2 CFM prompt length is out of range");
    }
    std::vector<float> out(static_cast<size_t>(batch * channels * frames), 0.0F);
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t c = 0; c < channels; ++c) {
            const size_t base = static_cast<size_t>((b * channels + c) * frames);
            std::copy(
                prompt.begin() + static_cast<std::ptrdiff_t>(base),
                prompt.begin() + static_cast<std::ptrdiff_t>(base + static_cast<size_t>(prompt_frames)),
                out.begin() + static_cast<std::ptrdiff_t>(base));
        }
    }
    return out;
}

void copy_batch_row(
    const std::vector<float> & src,
    int64_t src_batch,
    std::vector<float> & dst,
    int64_t dst_batch,
    int64_t row_values) {
    std::copy(
        src.begin() + static_cast<std::ptrdiff_t>(src_batch * row_values),
        src.begin() + static_cast<std::ptrdiff_t>((src_batch + 1) * row_values),
        dst.begin() + static_cast<std::ptrdiff_t>(dst_batch * row_values));
}

std::vector<float> repeat_or_zero_batch(
    const std::vector<float> & values,
    int64_t row_values,
    const std::vector<int> & modes) {
    std::vector<float> out(static_cast<size_t>(static_cast<int64_t>(modes.size()) * row_values), 0.0F);
    for (size_t row = 0; row < modes.size(); ++row) {
        if (modes[row] != 0) {
            copy_batch_row(values, 0, out, static_cast<int64_t>(row), row_values);
        }
    }
    return out;
}

std::vector<float> combine_cfg(
    const std::vector<float> & velocity,
    int64_t channels,
    int64_t frames,
    float intelligibility_cfg_rate,
    float similarity_cfg_rate,
    bool random_voice) {
    const int64_t row_values = channels * frames;
    std::vector<float> out(static_cast<size_t>(row_values), 0.0F);
    if (random_voice) {
        for (int64_t i = 0; i < row_values; ++i) {
            out[static_cast<size_t>(i)] =
                (1.0F + intelligibility_cfg_rate) * velocity[static_cast<size_t>(i)] -
                intelligibility_cfg_rate * velocity[static_cast<size_t>(row_values + i)];
        }
        return out;
    }
    if (intelligibility_cfg_rate == 0.0F && similarity_cfg_rate == 0.0F) {
        std::copy(velocity.begin(), velocity.begin() + static_cast<std::ptrdiff_t>(row_values), out.begin());
        return out;
    }
    if (intelligibility_cfg_rate == 0.0F) {
        for (int64_t i = 0; i < row_values; ++i) {
            out[static_cast<size_t>(i)] =
                (1.0F + similarity_cfg_rate) * velocity[static_cast<size_t>(i)] -
                similarity_cfg_rate * velocity[static_cast<size_t>(row_values + i)];
        }
        return out;
    }
    if (similarity_cfg_rate == 0.0F) {
        for (int64_t i = 0; i < row_values; ++i) {
            out[static_cast<size_t>(i)] =
                (1.0F + intelligibility_cfg_rate) * velocity[static_cast<size_t>(i)] -
                intelligibility_cfg_rate * velocity[static_cast<size_t>(row_values + i)];
        }
        return out;
    }
    for (int64_t i = 0; i < row_values; ++i) {
        out[static_cast<size_t>(i)] =
            (1.0F + intelligibility_cfg_rate + similarity_cfg_rate) * velocity[static_cast<size_t>(i)] -
            similarity_cfg_rate * velocity[static_cast<size_t>(row_values + i)] -
            intelligibility_cfg_rate * velocity[static_cast<size_t>(2 * row_values + i)];
    }
    return out;
}

struct SeedVcV2CfmEstimator::State {
    std::shared_ptr<engine::core::ExecutionContext> execution_context;
    std::shared_ptr<engine::core::BackendWeightStore> store;
    std::unique_ptr<CfmEstimatorRunner> runner;
};

SeedVcV2CfmEstimator::SeedVcV2CfmEstimator(
    std::shared_ptr<const engine::assets::TensorSource> source,
    engine::core::BackendConfig backend,
    engine::assets::TensorStorageType storage_type,
    SeedVcV2DitConfig config)
    : config_(std::move(config)),
      state_(std::make_shared<State>()) {
    if (source == nullptr) {
        throw std::runtime_error("Seed-VC V2 CFM estimator requires weights");
    }
    if (config_.hidden_dim <= 0 || config_.depth <= 0 || config_.num_heads <= 0 ||
        config_.in_channels <= 0 || config_.content_dim <= 0 || config_.style_encoder_dim <= 0 ||
        config_.hidden_dim % config_.num_heads != 0) {
        throw std::runtime_error("Seed-VC V2 CFM estimator config is invalid");
    }
    state_->execution_context = std::make_shared<engine::core::ExecutionContext>(backend);
    state_->store = std::make_shared<engine::core::BackendWeightStore>(
        state_->execution_context->backend(),
        state_->execution_context->backend_type(),
        "seed_vc.v2_cfm.estimator.weights",
        256ull * 1024ull * 1024ull);
    auto weights = load_cfm_weights(*state_->store, *source, config_.depth, storage_type);
    state_->store->upload();
    state_->runner = std::make_unique<CfmEstimatorRunner>(*state_->execution_context, std::move(weights), config_);
}

SeedVcV2CfmEstimator::~SeedVcV2CfmEstimator() = default;
SeedVcV2CfmEstimator::SeedVcV2CfmEstimator(SeedVcV2CfmEstimator &&) noexcept = default;
SeedVcV2CfmEstimator & SeedVcV2CfmEstimator::operator=(SeedVcV2CfmEstimator &&) noexcept = default;

SeedVcV2CfmEstimatorOutput SeedVcV2CfmEstimator::run(const SeedVcV2CfmEstimatorInput & input) const {
    if (state_ == nullptr || state_->runner == nullptr) {
        throw std::runtime_error("Seed-VC V2 CFM estimator is not initialized");
    }
    return state_->runner->run(input);
}

SeedVcV2CfmEstimatorOutput SeedVcV2CfmEstimator::infer(const SeedVcV2CfmInferenceInput & input) const {
    if (state_ == nullptr || state_->runner == nullptr) {
        throw std::runtime_error("Seed-VC V2 CFM estimator is not initialized");
    }
    if (input.batch != 1 || input.frames <= 0 || input.prompt_frames < 0 || input.prompt_frames > input.frames) {
        throw std::runtime_error("Seed-VC V2 CFM inference currently expects batch 1 and valid frame lengths");
    }
    const int64_t mel_values = input.batch * config_.in_channels * input.frames;
    const int64_t cond_values = input.batch * input.frames * config_.content_dim;
    if (static_cast<int64_t>(input.mu.size()) != cond_values ||
        static_cast<int64_t>(input.prompt.size()) != mel_values ||
        static_cast<int64_t>(input.style.size()) != input.batch * config_.style_encoder_dim ||
        static_cast<int64_t>(input.initial_noise.size()) != mel_values) {
        throw std::runtime_error("Seed-VC V2 CFM inference input shape mismatch");
    }
    auto x = input.initial_noise;
    for (float & value : x) {
        value *= input.temperature;
    }
    auto prompt_x = make_prompt_x(input.prompt, input.batch, config_.in_channels, input.frames, input.prompt_frames);
    zero_prompt_region(x, input.batch, config_.in_channels, input.frames, input.prompt_frames);
    const auto t_span = build_seed_vc_t_span(input.num_inference_steps);
    float t = t_span.front();
    float dt = t_span[1] - t_span[0];
    for (int step = 1; step <= input.num_inference_steps; ++step) {
        std::vector<int> prompt_modes;
        std::vector<int> style_modes;
        std::vector<int> mu_modes;
        if (input.random_voice) {
            prompt_modes = {0, 0};
            style_modes = {0, 0};
            mu_modes = {1, 0};
        } else if (input.intelligibility_cfg_rate == 0.0F && input.similarity_cfg_rate == 0.0F) {
            prompt_modes = {1};
            style_modes = {1};
            mu_modes = {1};
        } else if (input.intelligibility_cfg_rate == 0.0F) {
            prompt_modes = {1, 0};
            style_modes = {1, 0};
            mu_modes = {1, 1};
        } else if (input.similarity_cfg_rate == 0.0F) {
            prompt_modes = {1, 0};
            style_modes = {1, 0};
            mu_modes = {1, 0};
        } else {
            prompt_modes = {1, 0, 0};
            style_modes = {1, 0, 0};
            mu_modes = {1, 1, 0};
        }
        const int64_t cfg_batch = static_cast<int64_t>(prompt_modes.size());
        SeedVcV2CfmEstimatorInput estimator_input;
        estimator_input.batch = cfg_batch;
        estimator_input.frames = input.frames;
        estimator_input.x = repeat_or_zero_batch(x, config_.in_channels * input.frames, std::vector<int>(static_cast<size_t>(cfg_batch), 1));
        estimator_input.prompt = repeat_or_zero_batch(prompt_x, config_.in_channels * input.frames, prompt_modes);
        estimator_input.cond = repeat_or_zero_batch(input.mu, input.frames * config_.content_dim, mu_modes);
        estimator_input.style = repeat_or_zero_batch(input.style, config_.style_encoder_dim, style_modes);
        estimator_input.timestep.assign(static_cast<size_t>(cfg_batch), t);
        const auto velocity = state_->runner->run(estimator_input).velocity;
        const auto dphi_dt = combine_cfg(
            velocity,
            config_.in_channels,
            input.frames,
            input.intelligibility_cfg_rate,
            input.similarity_cfg_rate,
            input.random_voice);
        for (size_t index = 0; index < x.size(); ++index) {
            x[index] += dt * dphi_dt[index];
        }
        t += dt;
        if (step < input.num_inference_steps) {
            dt = t_span[static_cast<size_t>(step + 1)] - t;
        }
        zero_prompt_region(x, input.batch, config_.in_channels, input.frames, input.prompt_frames);
    }
    SeedVcV2CfmEstimatorOutput out;
    out.velocity = std::move(x);
    out.batch = input.batch;
    out.channels = config_.in_channels;
    out.frames = input.frames;
    return out;
}

}  // namespace engine::models::seed_vc
