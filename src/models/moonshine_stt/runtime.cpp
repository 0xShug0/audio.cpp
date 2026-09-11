#include "engine/models/moonshine_stt/runtime.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/core/module.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/feed_forward.h"
#include "engine/framework/modules/attention/grouped_query_attention.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/optimizations/fast_kv_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/streaming_conv_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/transformers/qwen_causal_decoder.h"
#include "engine/framework/runtime/kv_cache.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/sampling/hf_sampler.h"
#include "engine/models/moonshine_stt/assets.h"
#include "engine/models/moonshine_stt/weights.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace engine::models::moonshine_stt {
namespace {

using Clock = std::chrono::steady_clock;

constexpr size_t kDefaultWeightContextBytes = 256ull * 1024ull * 1024ull;
constexpr size_t kDefaultGraphArenaBytes = 512ull * 1024ull * 1024ull;

engine::modules::GeluApproximation parse_encoder_gelu_approximation(
    const std::unordered_map<std::string, std::string> & options) {
    const auto value = runtime::find_option(options, {"moonshine_stt.encoder_gelu"});
    if (!value.has_value() || *value == "quick") {
        return engine::modules::GeluApproximation::Quick;
    }
    if (*value == "erf" || *value == "exact") {
        return engine::modules::GeluApproximation::ExactErf;
    }
    if (*value == "tanh") {
        return engine::modules::GeluApproximation::Tanh;
    }
    throw std::runtime_error("moonshine_stt.encoder_gelu must be erf, exact, tanh, or quick");
}

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct GgmlGallocrDeleter {
    void operator()(ggml_gallocr * alloc) const noexcept {
        if (alloc != nullptr) {
            ggml_gallocr_free(alloc);
        }
    }
};

struct GgmlBackendDeleter {
    void operator()(ggml_backend * backend) const noexcept {
        if (backend != nullptr) {
            ggml_backend_free(backend);
        }
    }
};

struct GgmlBackendSchedDeleter {
    void operator()(ggml_backend_sched * sched) const noexcept {
        if (sched != nullptr) {
            ggml_backend_sched_free(sched);
        }
    }
};

ggml_backend_t init_blas_backend_if_available(int threads) {
    for (size_t reg_index = 0; reg_index < ggml_backend_reg_count(); ++reg_index) {
        ggml_backend_reg_t reg = ggml_backend_reg_get(reg_index);
        const char * reg_name = reg != nullptr ? ggml_backend_reg_name(reg) : nullptr;
        if (reg_name == nullptr || std::strcmp(reg_name, "BLAS") != 0) {
            continue;
        }
        if (ggml_backend_reg_dev_count(reg) == 0) {
            return nullptr;
        }
        ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, 0);
        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        if (backend != nullptr && threads > 0) {
            engine::core::set_backend_threads(backend, threads);
        }
        return backend;
    }
    return nullptr;
}

struct MoonshineAttentionConfig {
    int64_t hidden_size = 0;
    int64_t heads = 0;
    int64_t kv_heads = 0;
    int64_t head_dim = 0;
    int64_t rotary_dim = 0;
    float rope_theta = 10000.0F;
};

MoonshineAttentionConfig attention_config(const MoonshineEncoderConfig & config) {
    return {
        config.hidden_size,
        config.heads,
        config.kv_heads,
        config.head_dim,
        0,
        10000.0F,
    };
}

MoonshineAttentionConfig attention_config(const MoonshineDecoderConfig & config) {
    return {
        config.hidden_size,
        config.heads,
        config.kv_heads,
        config.head_dim,
        config.rotary_dim,
        config.rope_theta,
    };
}

engine::core::TensorValue build_attention(
    engine::core::ModuleBuildContext & ctx,
    const engine::modules::AttentionWeights & weights,
    const engine::core::TensorValue & query_input,
    const engine::core::TensorValue & key_value_input,
    const MoonshineAttentionConfig & config,
    const std::optional<engine::core::TensorValue> & mask,
    bool causal,
    const engine::core::TensorValue * positions) {
    const bool apply_rope = positions != nullptr;
    engine::core::TensorValue q;
    engine::core::TensorValue k;
    engine::core::TensorValue v;
    if (weights.qkv_weight.has_value()) {
        if (query_input.tensor != key_value_input.tensor) {
            throw std::runtime_error("Moonshine packed QKV attention requires shared query/key/value input");
        }
        auto qkv = engine::modules::LinearModule({
            config.hidden_size,
            config.heads * config.head_dim + 2 * config.kv_heads * config.head_dim,
            weights.qkv_bias.has_value(),
        }).build(ctx, query_input, {*weights.qkv_weight, weights.qkv_bias});
        const int last_axis = static_cast<int>(qkv.shape.rank - 1);
        q = engine::modules::SliceModule({last_axis, 0, config.heads * config.head_dim}).build(ctx, qkv);
        k = engine::modules::SliceModule({last_axis, config.heads * config.head_dim, config.kv_heads * config.head_dim}).build(ctx, qkv);
        v = engine::modules::SliceModule({last_axis, config.heads * config.head_dim + config.kv_heads * config.head_dim, config.kv_heads * config.head_dim}).build(ctx, qkv);
    } else {
        q = engine::modules::LinearModule({config.hidden_size, config.heads * config.head_dim, false})
                .build(ctx, query_input, {weights.q_weight, weights.q_bias});
        k = engine::modules::LinearModule({config.hidden_size, config.kv_heads * config.head_dim, false})
                .build(ctx, key_value_input, {weights.k_weight, weights.k_bias});
        v = engine::modules::LinearModule({config.hidden_size, config.kv_heads * config.head_dim, false})
                .build(ctx, key_value_input, {weights.v_weight, weights.v_bias});
    }
    q = engine::core::reshape_tensor(
        ctx,
        engine::core::ensure_backend_addressable_layout(ctx, q),
        engine::core::TensorShape::from_dims({q.shape.dims[0], q.shape.dims[1], config.heads, config.head_dim}));
    k = engine::core::reshape_tensor(
        ctx,
        engine::core::ensure_backend_addressable_layout(ctx, k),
        engine::core::TensorShape::from_dims({k.shape.dims[0], k.shape.dims[1], config.kv_heads, config.head_dim}));
    v = engine::core::reshape_tensor(
        ctx,
        engine::core::ensure_backend_addressable_layout(ctx, v),
        engine::core::TensorShape::from_dims({v.shape.dims[0], v.shape.dims[1], config.kv_heads, config.head_dim}));
    if (apply_rope) {
        const engine::modules::RoPEModule rope({config.rotary_dim, GGML_ROPE_TYPE_NORMAL, config.rope_theta});
        q = rope.build(ctx, q, *positions);
        k = rope.build(ctx, k, *positions);
    }
    q = engine::modules::TransposeModule({{0, 2, 1, 3}, q.shape.rank}).build(ctx, q);
    k = engine::modules::TransposeModule({{0, 2, 1, 3}, k.shape.rank}).build(ctx, k);
    v = engine::modules::TransposeModule({{0, 2, 1, 3}, v.shape.rank}).build(ctx, v);
    auto context = engine::modules::GroupedQueryAttentionModule({
        config.head_dim,
        engine::modules::GroupedQueryAttentionLowering::FlashGrouped,
        GGML_PREC_DEFAULT,
        causal ? engine::modules::AttentionCausality::Causal : engine::modules::AttentionCausality::NonCausal,
    }).build(ctx, q, k, v, mask);
    context = engine::core::ensure_backend_addressable_layout(ctx, context);
    context = engine::core::reshape_tensor(
        ctx,
        context,
        engine::core::TensorShape::from_dims(
            {query_input.shape.dims[0], query_input.shape.dims[1], config.heads * config.head_dim}));
    return engine::modules::LinearModule({config.heads * config.head_dim, config.hidden_size, false})
        .build(ctx, context, {weights.out_weight, weights.out_bias});
}

struct EncoderMaskBinding {
    engine::core::TensorValue tensor;
    std::vector<float> values;
};

engine::core::TensorValue reshape_heads_bthd(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    int64_t heads,
    int64_t head_dim) {
    return engine::core::reshape_tensor(
        ctx,
        engine::core::ensure_backend_addressable_layout(ctx, input),
        engine::core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], heads, head_dim}));
}

engine::core::TensorValue bthd_to_bhtd(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & value) {
    return engine::modules::TransposeModule({{0, 2, 1, 3}, value.shape.rank}).build(ctx, value);
}

struct MoonshineStaticAttentionOutput {
    engine::core::TensorValue output;
    engine::core::TensorValue key;
    engine::core::TensorValue value;
};

MoonshineStaticAttentionOutput build_self_attention_static(
    engine::core::ModuleBuildContext & ctx,
    const MoonshineDecoderConfig & config,
    const engine::modules::AttentionWeights & weights,
    const engine::core::TensorValue & input,
    const engine::core::TensorValue & positions,
    const engine::core::TensorValue & attention_mask,
    const engine::core::TensorValue & cache_key,
    const engine::core::TensorValue & cache_value,
    const engine::core::TensorValue & cache_slot) {
    auto q = engine::modules::LinearModule({config.hidden_size, config.heads * config.head_dim, false})
                 .build(ctx, input, {weights.q_weight, weights.q_bias});
    auto k = engine::modules::LinearModule({config.hidden_size, config.kv_heads * config.head_dim, false})
                 .build(ctx, input, {weights.k_weight, weights.k_bias});
    auto v = engine::modules::LinearModule({config.hidden_size, config.kv_heads * config.head_dim, false})
                 .build(ctx, input, {weights.v_weight, weights.v_bias});
    q = reshape_heads_bthd(ctx, q, config.heads, config.head_dim);
    k = reshape_heads_bthd(ctx, k, config.kv_heads, config.head_dim);
    v = reshape_heads_bthd(ctx, v, config.kv_heads, config.head_dim);
    const engine::modules::RoPEModule rope({config.rotary_dim, GGML_ROPE_TYPE_NORMAL, config.rope_theta});
    q = rope.build(ctx, q, positions);
    k = rope.build(ctx, k, positions);
    k = engine::core::ensure_backend_addressable_layout(ctx, k);
    v = engine::core::ensure_backend_addressable_layout(ctx, v);

    const engine::modules::FastKVSetRowsModule set_rows;
    auto updated_key = set_rows.build(ctx, cache_key, k, cache_slot);
    auto updated_value = set_rows.build(ctx, cache_value, v, cache_slot);

    auto q_heads = bthd_to_bhtd(ctx, q);
    q_heads = engine::core::wrap_tensor(ggml_cont(ctx.ggml, q_heads.tensor), q_heads.shape, q_heads.type);
    const auto k_heads = bthd_to_bhtd(ctx, updated_key);
    const auto v_heads = bthd_to_bhtd(ctx, updated_value);
    auto context = engine::modules::GroupedQueryAttentionModule({
        config.head_dim,
        engine::modules::GroupedQueryAttentionLowering::FlashGroupedViewKV,
        GGML_PREC_DEFAULT,
        engine::modules::AttentionCausality::NonCausal,
    }).build(ctx, q_heads, k_heads, v_heads, attention_mask);
    context = engine::core::ensure_backend_addressable_layout(ctx, context);
    context = engine::core::reshape_tensor(
        ctx,
        context,
        engine::core::TensorShape::from_dims({1, 1, config.heads * config.head_dim}));
    return {
        engine::modules::LinearModule({config.heads * config.head_dim, config.hidden_size, false})
            .build(ctx, context, {weights.out_weight, weights.out_bias}),
        k,
        v,
    };
}

engine::core::TensorValue build_cross_attention_cached(
    engine::core::ModuleBuildContext & ctx,
    const MoonshineDecoderConfig & config,
    const engine::modules::AttentionWeights & weights,
    const engine::core::TensorValue & input,
    const engine::core::TensorValue & cross_key,
    const engine::core::TensorValue & cross_value) {
    auto q = engine::modules::LinearModule({config.hidden_size, config.heads * config.head_dim, false})
                 .build(ctx, input, {weights.q_weight, weights.q_bias});
    q = reshape_heads_bthd(ctx, q, config.heads, config.head_dim);
    auto q_heads = bthd_to_bhtd(ctx, q);
    q_heads = engine::core::wrap_tensor(ggml_cont(ctx.ggml, q_heads.tensor), q_heads.shape, q_heads.type);
    const auto k_heads = bthd_to_bhtd(ctx, cross_key);
    const auto v_heads = bthd_to_bhtd(ctx, cross_value);
    auto context = engine::modules::GroupedQueryAttentionModule({
        config.head_dim,
        engine::modules::GroupedQueryAttentionLowering::FlashGroupedViewKV,
        GGML_PREC_DEFAULT,
        engine::modules::AttentionCausality::NonCausal,
    }).build(ctx, q_heads, k_heads, v_heads);
    context = engine::core::ensure_backend_addressable_layout(ctx, context);
    context = engine::core::reshape_tensor(
        ctx,
        context,
        engine::core::TensorShape::from_dims({1, 1, config.heads * config.head_dim}));
    return engine::modules::LinearModule({config.heads * config.head_dim, config.hidden_size, false})
        .build(ctx, context, {weights.out_weight, weights.out_bias});
}

engine::core::TensorValue build_encoder_layer(
    engine::core::ModuleBuildContext & ctx,
    const MoonshineEncoderConfig & config,
    const MoonshineEncoderLayerWeights & weights,
    const engine::core::TensorValue & input,
    const MoonshineSlidingWindow & window,
    engine::modules::GeluApproximation gelu_approximation,
    std::vector<EncoderMaskBinding> & masks) {
    auto hidden = engine::modules::LayerNormModule({config.hidden_size, 1.0e-5F, true, false})
                      .build(ctx, input, weights.input_norm);
    const int64_t frames = hidden.shape.dims[1];
    auto mask = engine::core::make_tensor(
        ctx,
        GGML_TYPE_F16,
        engine::core::TensorShape::from_dims({1, 1, frames, frames}));
    ggml_set_input(mask.tensor);
    std::vector<float> mask_values(static_cast<size_t>(frames * frames), 0.0F);
    const float neg = -std::numeric_limits<float>::infinity();
    for (int64_t q = 0; q < frames; ++q) {
        for (int64_t k = 0; k < frames; ++k) {
            const int64_t distance = q - k;
            const bool left = distance >= 0 && distance < window.past;
            const bool right = distance < 0 && -distance < window.future;
            if (!left && !right) {
                mask_values[static_cast<size_t>(q * frames + k)] = neg;
            }
        }
    }
    masks.push_back({mask, std::move(mask_values)});
    hidden = build_attention(
        ctx,
        weights.self_attn,
        hidden,
        hidden,
        attention_config(config),
        mask,
        false,
        nullptr);
    auto x = engine::modules::AddModule().build(ctx, input, hidden);

    hidden = engine::modules::LayerNormModule({config.hidden_size, 1.0e-5F, true, false})
                 .build(ctx, x, weights.post_attention_norm);
    hidden = engine::modules::FeedForwardModule({
        config.hidden_size,
        config.intermediate_size,
        true,
        gelu_approximation,
    }).build(ctx, hidden, weights.feed_forward);
    return engine::modules::AddModule().build(ctx, x, hidden);
}

engine::core::TensorValue build_decoder_mlp(
    engine::core::ModuleBuildContext & ctx,
    const MoonshineDecoderConfig & config,
    const MoonshineDecoderLayerWeights & weights,
    const engine::core::TensorValue & input) {
    auto projected = engine::modules::LinearModule({config.hidden_size, config.intermediate_size * 2, true})
                         .build(ctx, input, weights.mlp_fc1);
    const int last_axis = static_cast<int>(projected.shape.rank - 1);
    auto hidden = engine::modules::SliceModule({last_axis, 0, config.intermediate_size}).build(ctx, projected);
    auto gate = engine::modules::SliceModule({last_axis, config.intermediate_size, config.intermediate_size}).build(ctx, projected);
    gate = engine::modules::SiluModule().build(ctx, gate);
    hidden = engine::modules::MulModule().build(ctx, hidden, gate);
    return engine::modules::LinearModule({config.intermediate_size, config.hidden_size, true})
        .build(ctx, hidden, weights.mlp_fc2);
}

engine::core::TensorValue build_decoder_layer(
    engine::core::ModuleBuildContext & ctx,
    const MoonshineDecoderConfig & config,
    const MoonshineDecoderLayerWeights & weights,
    const engine::core::TensorValue & input,
    const engine::core::TensorValue & memory,
    const engine::core::TensorValue & positions) {
    auto hidden = engine::modules::LayerNormModule({config.hidden_size, 1.0e-5F, true, false})
                      .build(ctx, input, weights.input_norm);
    hidden = build_attention(
        ctx,
        weights.self_attn,
        hidden,
        hidden,
        attention_config(config),
        std::nullopt,
        true,
        &positions);
    auto x = engine::modules::AddModule().build(ctx, input, hidden);

    hidden = engine::modules::LayerNormModule({config.hidden_size, 1.0e-5F, true, false})
                 .build(ctx, x, weights.post_attention_norm);
    hidden = build_attention(
        ctx,
        weights.cross_attn,
        hidden,
        memory,
        attention_config(config),
        std::nullopt,
        false,
        nullptr);
    x = engine::modules::AddModule().build(ctx, x, hidden);

    hidden = engine::modules::LayerNormModule({config.hidden_size, 1.0e-5F, true, false})
                 .build(ctx, x, weights.final_norm);
    hidden = build_decoder_mlp(ctx, config, weights, hidden);
    return engine::modules::AddModule().build(ctx, x, hidden);
}

std::vector<float> prepare_compressed_frames(
    const std::vector<float> & audio,
    int64_t frame_length,
    float log_k) {
    const int64_t frames = static_cast<int64_t>(audio.size()) / frame_length;
    if (frames <= 0) {
        return {};
    }
    std::vector<float> out(static_cast<size_t>(frames * frame_length), 0.0F);
    const float scale = std::exp(log_k);
    for (int64_t frame = 0; frame < frames; ++frame) {
        const float * in = audio.data() + frame * frame_length;
        double mean = 0.0;
        for (int64_t i = 0; i < frame_length; ++i) {
            mean += static_cast<double>(in[i]);
        }
        mean /= static_cast<double>(frame_length);
        double sq = 0.0;
        for (int64_t i = 0; i < frame_length; ++i) {
            const double centered = static_cast<double>(in[i]) - mean;
            sq += centered * centered;
        }
        const float rms = static_cast<float>(std::sqrt(sq / static_cast<double>(frame_length) + 1.0e-6));
        for (int64_t i = 0; i < frame_length; ++i) {
            const float normalized = (in[i] - static_cast<float>(mean)) / rms;
            out[static_cast<size_t>(frame * frame_length + i)] = std::asinh(scale * normalized);
        }
    }
    return out;
}

class EncoderGraph {
public:
    EncoderGraph(
        const MoonshineAssets & assets,
        const MoonshineWeights & weights,
        const engine::core::ExecutionContext & execution_context,
        int64_t input_frames,
        engine::modules::GeluApproximation encoder_gelu,
        bool cpu_blas_scheduler,
        size_t graph_arena_bytes)
        : backend_(execution_context.backend()),
          frames_(input_frames),
          hidden_(assets.config.decoder.hidden_size) {
        const auto build_start = Clock::now();
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize Moonshine encoder graph context");
        }
        engine::core::ModuleBuildContext ctx{ctx_.get(), "moonshine_stt.encoder", execution_context.backend_type()};
        input_ = engine::core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({1, input_frames, assets.config.encoder.frame_length}));
        ggml_set_input(input_.tensor);

        auto x = engine::modules::LinearModule({
            assets.config.encoder.frame_length,
            assets.config.encoder.hidden_size,
            false,
        }).build(ctx, input_, weights.encoder.frontend.linear);
        x = engine::modules::SiluModule().build(ctx, x);
        x = engine::modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
        x = engine::modules::StreamingConv1dModule({
            assets.config.encoder.hidden_size,
            assets.config.encoder.hidden_size * 2,
            5,
            2,
            1,
            true,
            engine::modules::StreamingPadMode::Constant,
            engine::modules::StreamingConv1dPaddingMode::StrictCausal,
        }).build(ctx, x, weights.encoder.frontend.conv1);
        x = engine::modules::SiluModule().build(ctx, x);
        x = engine::modules::StreamingConv1dModule({
            assets.config.encoder.hidden_size * 2,
            assets.config.encoder.hidden_size,
            5,
            2,
            1,
            true,
            engine::modules::StreamingPadMode::Constant,
            engine::modules::StreamingConv1dPaddingMode::StrictCausal,
        }).build(ctx, x, weights.encoder.frontend.conv2);
        x = engine::modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
        memory_frames_ = x.shape.dims[1];

        masks_.reserve(weights.encoder.layers.size());
        for (size_t i = 0; i < weights.encoder.layers.size(); ++i) {
            x = build_encoder_layer(
                ctx,
                assets.config.encoder,
                weights.encoder.layers[i],
                x,
                assets.config.encoder.sliding_windows[i],
                encoder_gelu,
                masks_);
        }
        x = engine::modules::LayerNormModule({assets.config.encoder.hidden_size, 1.0e-5F, true, false})
                .build(ctx, x, weights.encoder.final_norm);

        positions_ = engine::core::make_tensor(
            ctx,
            GGML_TYPE_I32,
            engine::core::TensorShape::from_dims({memory_frames_}));
        ggml_set_input(positions_.tensor);
        auto pos = engine::modules::EmbeddingModule({
            assets.config.decoder.max_position_embeddings,
            assets.config.encoder.hidden_size,
        }).build(ctx, positions_, weights.decoder.position_embedding);
        pos = engine::core::reshape_tensor(
            ctx,
            pos,
            engine::core::TensorShape::from_dims({1, memory_frames_, assets.config.encoder.hidden_size}));
        x = engine::modules::AddModule().build(ctx, x, pos);
        if (weights.decoder.adapter_proj.has_value()) {
            x = engine::modules::LinearModule({
                assets.config.encoder.hidden_size,
                assets.config.decoder.hidden_size,
                false,
            }).build(ctx, x, *weights.decoder.adapter_proj);
        }
        output_ = x.tensor;
        hidden_ = x.shape.dims[2];
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 65536, false);
        ggml_build_forward_expand(graph_, output_);
        if (cpu_blas_scheduler && execution_context.backend_type() == engine::core::BackendType::Cpu) {
            blas_backend_.reset(init_blas_backend_if_available(execution_context.config().threads));
            if (blas_backend_ != nullptr) {
                ggml_backend_t backends[] = {blas_backend_.get(), backend_};
                ggml_backend_buffer_type_t bufts[] = {
                    ggml_backend_get_default_buffer_type(blas_backend_.get()),
                    ggml_backend_get_default_buffer_type(backend_),
                };
                sched_.reset(ggml_backend_sched_new(backends, bufts, 2, ggml_graph_size(graph_), false, false));
                if (sched_ != nullptr &&
                    ggml_backend_sched_reserve(sched_.get(), graph_) &&
                    ggml_backend_sched_alloc_graph(sched_.get(), graph_)) {
                    use_scheduler_ = true;
                } else {
                    sched_.reset();
                    blas_backend_.reset();
                }
            }
        }
        if (!use_scheduler_) {
            gallocr_.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_)));
            if (gallocr_ == nullptr ||
                !ggml_gallocr_reserve(gallocr_.get(), graph_) ||
                !ggml_gallocr_alloc_graph(gallocr_.get(), graph_)) {
                throw std::runtime_error("failed to allocate Moonshine encoder graph");
            }
            engine::core::prepare_host_graph_plan(execution_context, graph_, plan_);
        }
        const auto pos_values = engine::modules::qwen_position_ids(memory_frames_);
        engine::core::write_tensor_i32(positions_, pos_values);
        for (const auto & mask : masks_) {
            engine::core::write_tensor_f16(mask.tensor, mask.values);
        }
        debug::trace_log_scalar("moonshine_stt.encoder_attention_masks", static_cast<int64_t>(masks_.size()));
        debug::trace_log_scalar("moonshine_stt.encoder_blas_scheduler", use_scheduler_);
        debug::timing_log_scalar("moonshine_stt.encoder_graph_build_ms", engine::debug::elapsed_ms(build_start));
    }

    ~EncoderGraph() {
        engine::core::release_backend_graph_resources(backend_, graph_);
    }

    int64_t memory_frames() const noexcept {
        return memory_frames_;
    }

    int64_t hidden_size() const noexcept {
        return hidden_;
    }

    std::vector<float> run(const std::vector<float> & frames, const engine::core::ExecutionContext & execution_context) {
        if (frames.size() != static_cast<size_t>(frames_ * input_.shape.dims[2])) {
            throw std::runtime_error("Moonshine encoder input frame shape mismatch");
        }
        const auto compute_start = Clock::now();
        engine::core::write_tensor_f32(input_, frames);
        const ggml_status status = use_scheduler_
            ? ggml_backend_sched_graph_compute(sched_.get(), graph_)
            : engine::core::compute_graph(execution_context, graph_, plan_, "moonshine_stt.encoder");
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Moonshine encoder graph compute failed");
        }
        if (use_scheduler_) {
            ggml_backend_sched_synchronize(sched_.get());
        } else {
            ggml_backend_synchronize(backend_);
        }
        debug::timing_log_scalar("moonshine_stt.encoder_compute_ms", engine::debug::elapsed_ms(compute_start));
        return engine::core::read_tensor_f32(output_);
    }

private:
    ggml_backend_t backend_ = nullptr;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    std::unique_ptr<ggml_gallocr, GgmlGallocrDeleter> gallocr_;
    std::unique_ptr<ggml_backend, GgmlBackendDeleter> blas_backend_;
    std::unique_ptr<ggml_backend_sched, GgmlBackendSchedDeleter> sched_;
    engine::core::HostGraphPlan plan_;
    ggml_cgraph * graph_ = nullptr;
    engine::core::TensorValue input_;
    std::vector<EncoderMaskBinding> masks_;
    engine::core::TensorValue positions_;
    ggml_tensor * output_ = nullptr;
    int64_t frames_ = 0;
    int64_t memory_frames_ = 0;
    int64_t hidden_ = 0;
    bool use_scheduler_ = false;
};

class CachedDecoderGraph {
public:
    CachedDecoderGraph(
        const MoonshineAssets & assets,
        const MoonshineWeights & weights,
        const engine::core::ExecutionContext & execution_context,
        int64_t cache_steps,
        int64_t memory_frames,
        size_t graph_arena_bytes)
        : backend_(execution_context.backend()),
          cache_steps_(cache_steps),
          memory_frames_(memory_frames),
          vocab_(assets.config.decoder.vocab_size) {
        if (cache_steps_ <= 0 || memory_frames_ <= 0) {
            throw std::runtime_error("Moonshine decoder cached graph requires positive cache and memory sizes");
        }
        ggml_init_params state_params{64ull * 1024ull * 1024ull, nullptr, true};
        state_ctx_.reset(ggml_init(state_params));
        if (state_ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize Moonshine decoder state context");
        }
        ggml_init_params graph_params{graph_arena_bytes, nullptr, true};
        cross_ctx_.reset(ggml_init(graph_params));
        step_ctx_.reset(ggml_init(graph_params));
        if (cross_ctx_ == nullptr || step_ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize Moonshine cached decoder graph context");
        }

        const auto & config = assets.config.decoder;
        token_id_ = ggml_new_tensor_1d(state_ctx_.get(), GGML_TYPE_I32, 1);
        position_ = ggml_new_tensor_1d(state_ctx_.get(), GGML_TYPE_I32, 1);
        cache_slot_ = ggml_new_tensor_1d(state_ctx_.get(), GGML_TYPE_I32, 1);
        attention_mask_ = ggml_new_tensor_1d(state_ctx_.get(), GGML_TYPE_F16, cache_steps_);
        memory_ = ggml_new_tensor_3d(state_ctx_.get(), GGML_TYPE_F32, config.hidden_size, memory_frames_, 1);

        std::vector<engine::core::TensorValue> self_keys;
        std::vector<engine::core::TensorValue> self_values;
        self_keys.reserve(weights.decoder.layers.size());
        self_values.reserve(weights.decoder.layers.size());
        cross_keys_.reserve(weights.decoder.layers.size());
        cross_values_.reserve(weights.decoder.layers.size());
        for (size_t layer = 0; layer < weights.decoder.layers.size(); ++layer) {
            self_keys.push_back(engine::core::wrap_tensor(
                ggml_new_tensor_4d(state_ctx_.get(), GGML_TYPE_F32, config.head_dim, config.kv_heads, cache_steps_, 1),
                engine::core::TensorShape::from_dims({1, cache_steps_, config.kv_heads, config.head_dim}),
                GGML_TYPE_F32));
            self_values.push_back(engine::core::wrap_tensor(
                ggml_new_tensor_4d(state_ctx_.get(), GGML_TYPE_F32, config.head_dim, config.kv_heads, cache_steps_, 1),
                engine::core::TensorShape::from_dims({1, cache_steps_, config.kv_heads, config.head_dim}),
                GGML_TYPE_F32));
            cross_keys_.push_back(engine::core::wrap_tensor(
                ggml_new_tensor_4d(state_ctx_.get(), GGML_TYPE_F32, config.head_dim, config.kv_heads, memory_frames_, 1),
                engine::core::TensorShape::from_dims({1, memory_frames_, config.kv_heads, config.head_dim}),
                GGML_TYPE_F32));
            cross_values_.push_back(engine::core::wrap_tensor(
                ggml_new_tensor_4d(state_ctx_.get(), GGML_TYPE_F32, config.head_dim, config.kv_heads, memory_frames_, 1),
                engine::core::TensorShape::from_dims({1, memory_frames_, config.kv_heads, config.head_dim}),
                GGML_TYPE_F32));
        }
        state_buffer_ = ggml_backend_alloc_ctx_tensors(state_ctx_.get(), backend_);
        if (state_buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate Moonshine decoder state tensors");
        }

        build_cross_graph(assets, weights, execution_context, graph_arena_bytes);
        build_step_graph(assets, weights, execution_context, graph_arena_bytes, std::move(self_keys), std::move(self_values));
    }

    ~CachedDecoderGraph() {
        engine::core::release_backend_graph_resources(backend_, cross_graph_);
        engine::core::release_backend_graph_resources(backend_, step_graph_);
        cross_gallocr_.reset();
        step_gallocr_.reset();
        if (state_buffer_ != nullptr) {
            ggml_backend_buffer_free(state_buffer_);
            state_buffer_ = nullptr;
        }
    }

    void prepare_memory(
        const std::vector<float> & memory,
        const engine::core::ExecutionContext & execution_context) {
        if (memory.size() != static_cast<size_t>(memory_frames_ * memory_hidden_)) {
            throw std::runtime_error("Moonshine decoder memory shape mismatch");
        }
        const auto start = Clock::now();
        ggml_backend_tensor_set(memory_, memory.data(), 0, memory.size() * sizeof(float));
        if (engine::core::compute_graph(execution_context, cross_graph_, cross_plan_, "moonshine_stt.decoder.cross_kv") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Moonshine decoder cross-KV graph compute failed");
        }
        ggml_backend_synchronize(backend_);
        reset_state();
        debug::timing_log_scalar("moonshine_stt.decode_cross_kv_ms", engine::debug::elapsed_ms(start));
    }

    std::vector<float> run_step(
        int32_t token,
        int32_t position,
        const engine::core::ExecutionContext & execution_context) {
        if (position < 0 || position >= cache_steps_) {
            throw std::runtime_error("Moonshine decoder position exceeds cache capacity");
        }
        const auto upload_start = Clock::now();
        ggml_backend_tensor_set(token_id_, &token, 0, sizeof(token));
        ggml_backend_tensor_set(position_, &position, 0, sizeof(position));
        ggml_backend_tensor_set(cache_slot_, &position, 0, sizeof(position));
        engine::modules::write_qwen_cached_step_mask(
            attention_mask_,
            attention_mask_values_,
            cache_steps_,
            position,
            position);
        step_upload_ms_ += engine::debug::elapsed_ms(upload_start);

        const auto compute_start = Clock::now();
        if (engine::core::compute_graph(execution_context, step_graph_, step_plan_, "moonshine_stt.decoder.step") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Moonshine decoder step graph compute failed");
        }
        ggml_backend_synchronize(backend_);
        step_compute_ms_ += engine::debug::elapsed_ms(compute_start);

        self_cache_.advance_after_direct_append(1);
        const auto read_start = Clock::now();
        engine::core::read_tensor_f32_into(logits_, logits_values_);
        step_read_ms_ += engine::debug::elapsed_ms(read_start);
        ++step_count_;
        return logits_values_;
    }

    int64_t vocab_size() const noexcept {
        return vocab_;
    }

    void log_step_timings() const {
        debug::timing_log_scalar("moonshine_stt.decode_step_count", step_count_);
        debug::timing_log_scalar("moonshine_stt.decode_step_upload_ms", step_upload_ms_);
        debug::timing_log_scalar("moonshine_stt.decode_step_compute_ms", step_compute_ms_);
        debug::timing_log_scalar("moonshine_stt.decode_step_read_ms", step_read_ms_);
    }

private:
    void build_cross_graph(
        const MoonshineAssets & assets,
        const MoonshineWeights & weights,
        const engine::core::ExecutionContext & execution_context,
        size_t /*graph_arena_bytes*/) {
        const auto & config = assets.config.decoder;
        memory_hidden_ = config.hidden_size;
        engine::core::ModuleBuildContext ctx{cross_ctx_.get(), "moonshine_stt.decoder.cross_kv", execution_context.backend_type()};
        auto memory = engine::core::wrap_tensor(
            memory_,
            engine::core::TensorShape::from_dims({1, memory_frames_, config.hidden_size}),
            GGML_TYPE_F32);
        cross_graph_ = ggml_new_graph_custom(cross_ctx_.get(), 65536, false);
        for (size_t layer_index = 0; layer_index < weights.decoder.layers.size(); ++layer_index) {
            const auto & layer = weights.decoder.layers[layer_index];
            auto k = engine::modules::LinearModule({config.hidden_size, config.kv_heads * config.head_dim, false})
                         .build(ctx, memory, {layer.cross_attn.k_weight, layer.cross_attn.k_bias});
            auto v = engine::modules::LinearModule({config.hidden_size, config.kv_heads * config.head_dim, false})
                         .build(ctx, memory, {layer.cross_attn.v_weight, layer.cross_attn.v_bias});
            k = reshape_heads_bthd(ctx, k, config.kv_heads, config.head_dim);
            v = reshape_heads_bthd(ctx, v, config.kv_heads, config.head_dim);
            auto * key_copy = ggml_cpy(cross_ctx_.get(), engine::core::ensure_backend_addressable_layout(ctx, k).tensor, cross_keys_[layer_index].tensor);
            auto * value_copy = ggml_cpy(cross_ctx_.get(), engine::core::ensure_backend_addressable_layout(ctx, v).tensor, cross_values_[layer_index].tensor);
            ggml_set_output(key_copy);
            ggml_set_output(value_copy);
            ggml_build_forward_expand(cross_graph_, key_copy);
            ggml_build_forward_expand(cross_graph_, value_copy);
        }
        cross_gallocr_.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_)));
        if (cross_gallocr_ == nullptr ||
            !ggml_gallocr_reserve(cross_gallocr_.get(), cross_graph_) ||
            !ggml_gallocr_alloc_graph(cross_gallocr_.get(), cross_graph_)) {
            throw std::runtime_error("failed to allocate Moonshine decoder cross-KV graph");
        }
        engine::core::prepare_host_graph_plan(execution_context, cross_graph_, cross_plan_);
    }

    void build_step_graph(
        const MoonshineAssets & assets,
        const MoonshineWeights & weights,
        const engine::core::ExecutionContext & execution_context,
        size_t /*graph_arena_bytes*/,
        std::vector<engine::core::TensorValue> self_keys,
        std::vector<engine::core::TensorValue> self_values) {
        const auto & config = assets.config.decoder;
        engine::core::ModuleBuildContext ctx{step_ctx_.get(), "moonshine_stt.decoder.step", execution_context.backend_type()};
        auto ids = engine::core::wrap_tensor(token_id_, engine::core::TensorShape::from_dims({1}), GGML_TYPE_I32);
        auto positions = engine::core::wrap_tensor(position_, engine::core::TensorShape::from_dims({1}), GGML_TYPE_I32);
        auto cache_slot = engine::core::wrap_tensor(cache_slot_, engine::core::TensorShape::from_dims({1}), GGML_TYPE_I32);
        auto attention_mask = engine::core::wrap_tensor(
            attention_mask_,
            engine::core::TensorShape::from_dims({1, 1, 1, cache_steps_}),
            GGML_TYPE_F16);
        auto x = engine::modules::EmbeddingModule({
            config.vocab_size,
            config.hidden_size,
        }).build(ctx, ids, weights.decoder.token_embedding);
        x = engine::core::reshape_tensor(ctx, x, engine::core::TensorShape::from_dims({1, 1, config.hidden_size}));
        for (size_t layer_index = 0; layer_index < weights.decoder.layers.size(); ++layer_index) {
            const auto & layer = weights.decoder.layers[layer_index];
            auto hidden = engine::modules::LayerNormModule({config.hidden_size, 1.0e-5F, true, false})
                              .build(ctx, x, layer.input_norm);
            auto self_attn = build_self_attention_static(
                ctx,
                config,
                layer.self_attn,
                hidden,
                positions,
                attention_mask,
                self_keys[layer_index],
                self_values[layer_index],
                cache_slot);
            x = engine::modules::AddModule().build(ctx, x, self_attn.output);

            hidden = engine::modules::LayerNormModule({config.hidden_size, 1.0e-5F, true, false})
                         .build(ctx, x, layer.post_attention_norm);
            hidden = build_cross_attention_cached(
                ctx,
                config,
                layer.cross_attn,
                hidden,
                cross_keys_[layer_index],
                cross_values_[layer_index]);
            x = engine::modules::AddModule().build(ctx, x, hidden);

            hidden = engine::modules::LayerNormModule({config.hidden_size, 1.0e-5F, true, false})
                         .build(ctx, x, layer.final_norm);
            hidden = build_decoder_mlp(ctx, config, layer, hidden);
            x = engine::modules::AddModule().build(ctx, x, hidden);
        }
        x = engine::modules::LayerNormModule({config.hidden_size, 1.0e-5F, true, false})
                .build(ctx, x, weights.decoder.norm);
        x = engine::modules::LinearModule({
            config.hidden_size,
            config.vocab_size,
            false,
        }).build(ctx, x, weights.decoder.output_projection);
        logits_ = x.tensor;
        ggml_set_output(logits_);
        step_graph_ = ggml_new_graph_custom(step_ctx_.get(), 65536, false);
        ggml_build_forward_expand(step_graph_, logits_);
        step_gallocr_.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_)));
        if (step_gallocr_ == nullptr ||
            !ggml_gallocr_reserve(step_gallocr_.get(), step_graph_) ||
            !ggml_gallocr_alloc_graph(step_gallocr_.get(), step_graph_)) {
            throw std::runtime_error("failed to allocate Moonshine decoder step graph");
        }
        engine::core::prepare_host_graph_plan(execution_context, step_graph_, step_plan_);
        self_cache_ = engine::runtime::TransformerKVCache(
            cache_steps_,
            config.kv_heads * config.head_dim,
            std::move(self_keys),
            std::move(self_values));
        self_cache_layers_ = weights.decoder.layers.size();
        attention_mask_values_.assign(
            static_cast<size_t>(cache_steps_),
            ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity()));
    }

    void reset_state() {
        engine::runtime::TransformerKVState empty;
        empty.current_end = 0;
        empty.layers.resize(self_cache_layers_);
        self_cache_.import_state(empty);
        std::fill(
            attention_mask_values_.begin(),
            attention_mask_values_.end(),
            ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity()));
        ggml_backend_tensor_set(
            attention_mask_,
            attention_mask_values_.data(),
            0,
            attention_mask_values_.size() * sizeof(ggml_fp16_t));
        step_upload_ms_ = 0.0;
        step_compute_ms_ = 0.0;
        step_read_ms_ = 0.0;
        step_count_ = 0;
    }

    ggml_backend_t backend_ = nullptr;
    std::unique_ptr<ggml_context, GgmlContextDeleter> state_ctx_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> cross_ctx_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> step_ctx_;
    std::unique_ptr<ggml_gallocr, GgmlGallocrDeleter> cross_gallocr_;
    std::unique_ptr<ggml_gallocr, GgmlGallocrDeleter> step_gallocr_;
    engine::core::HostGraphPlan cross_plan_;
    engine::core::HostGraphPlan step_plan_;
    ggml_backend_buffer_t state_buffer_ = nullptr;
    ggml_cgraph * cross_graph_ = nullptr;
    ggml_cgraph * step_graph_ = nullptr;
    ggml_tensor * token_id_ = nullptr;
    ggml_tensor * position_ = nullptr;
    ggml_tensor * cache_slot_ = nullptr;
    ggml_tensor * attention_mask_ = nullptr;
    ggml_tensor * memory_ = nullptr;
    ggml_tensor * logits_ = nullptr;
    int64_t cache_steps_ = 0;
    int64_t memory_frames_ = 0;
    int64_t memory_hidden_ = 0;
    int64_t vocab_ = 0;
    size_t self_cache_layers_ = 0;
    std::vector<engine::core::TensorValue> cross_keys_;
    std::vector<engine::core::TensorValue> cross_values_;
    std::vector<ggml_fp16_t> attention_mask_values_;
    std::vector<float> logits_values_;
    engine::runtime::TransformerKVCache self_cache_;
    double step_upload_ms_ = 0.0;
    double step_compute_ms_ = 0.0;
    double step_read_ms_ = 0.0;
    int64_t step_count_ = 0;
};


}  // namespace

MoonshineRuntimeConfig make_moonshine_runtime_config(
    const MoonshineConfig & config,
    engine::core::BackendType backend_type,
    const std::unordered_map<std::string, std::string> & options) {
    MoonshineRuntimeConfig runtime_config;
    runtime_config.weight_context_bytes =
        runtime::parse_size_mb_option(options, {"moonshine_stt.weight_context_mb"}, kDefaultWeightContextBytes);
    runtime_config.graph_arena_bytes =
        runtime::parse_size_mb_option(options, {"moonshine_stt.graph_arena_mb"}, kDefaultGraphArenaBytes);
    runtime_config.encoder_gelu = parse_encoder_gelu_approximation(options);
    const auto default_matmul_storage =
        backend_type == engine::core::BackendType::Cpu &&
            config.encoder.hidden_size == 620 &&
            config.encoder.layers == 10 &&
            config.decoder.hidden_size == 512
        ? engine::assets::TensorStorageType::F16
        : engine::assets::TensorStorageType::Native;
    runtime_config.encoder_weight_storage_type = runtime::parse_tensor_storage_option(
        options,
        "moonshine_stt.weight_type",
        default_matmul_storage,
        {
            engine::assets::TensorStorageType::Native,
            engine::assets::TensorStorageType::F32,
            engine::assets::TensorStorageType::F16,
        });
    runtime_config.decoder_weight_storage_type = runtime::parse_tensor_storage_option(
        options,
        "moonshine_stt.decoder_weight_type",
        "moonshine_stt.weight_type",
        engine::assets::TensorStorageType::Native,
        {
            engine::assets::TensorStorageType::Native,
            engine::assets::TensorStorageType::F32,
            engine::assets::TensorStorageType::F16,
        });
    runtime_config.conv_weight_storage_type = runtime::parse_tensor_storage_option(
        options,
        "moonshine_stt.conv_weight_type",
        engine::assets::TensorStorageType::Native,
        {
            engine::assets::TensorStorageType::Native,
            engine::assets::TensorStorageType::F32,
            engine::assets::TensorStorageType::F16,
        });
    const auto cpu_blas_scheduler = runtime::find_option(options, {"moonshine_stt.cpu_blas_scheduler"});
    runtime_config.cpu_blas_scheduler = !cpu_blas_scheduler.has_value() ||
        runtime::parse_bool_option(*cpu_blas_scheduler, "moonshine_stt.cpu_blas_scheduler");
    return runtime_config;
}

runtime::TaskResult transcribe_moonshine_stt(
    const MoonshineAssets & assets,
    const MoonshineWeights & weights,
    const engine::core::ExecutionContext & execution_context,
    const runtime::AudioBuffer & audio,
    const std::unordered_map<std::string, std::string> & options,
    const MoonshineRuntimeConfig & config) {
    const auto start = Clock::now();

    const auto resample_start = Clock::now();
    const auto mono = engine::audio::convert_interleaved_audio_to_mono_linear_resampled(
        audio.samples,
        audio.sample_rate,
        audio.channels,
        static_cast<int>(assets.config.encoder.sample_rate));
    debug::timing_log_scalar("moonshine_stt.resample_ms", engine::debug::elapsed_ms(resample_start));

    const auto frontend_start = Clock::now();
    const auto frames = prepare_compressed_frames(
        mono,
        assets.config.encoder.frame_length,
        weights.encoder.frontend.log_k);
    if (frames.empty()) {
        runtime::TaskResult result;
        result.text_output = runtime::Transcript{"", "en"};
        return result;
    }
    const int64_t frame_count = static_cast<int64_t>(frames.size()) / assets.config.encoder.frame_length;
    debug::timing_log_scalar("moonshine_stt.frontend_cpu_ms", engine::debug::elapsed_ms(frontend_start));
    debug::trace_log_scalar("moonshine_stt.input_samples", mono.size());
    debug::trace_log_scalar("moonshine_stt.input_frames", frame_count);

    EncoderGraph encoder_graph(
        assets,
        weights,
        execution_context,
        frame_count,
        config.encoder_gelu,
        config.cpu_blas_scheduler,
        config.graph_arena_bytes);
    const auto memory = encoder_graph.run(frames, execution_context);
    const int64_t memory_frames = encoder_graph.memory_frames();
    debug::trace_log_scalar("moonshine_stt.memory_frames", memory_frames);

    const int64_t audio_max_tokens = static_cast<int64_t>(
        std::ceil(static_cast<double>(memory_frames) * 0.020 * static_cast<double>(assets.config.max_tokens_per_second)));
    int64_t max_tokens = std::min<int64_t>(audio_max_tokens, assets.config.decoder.max_position_embeddings);
    if (const auto requested = runtime::parse_i64_option(options, {"max_tokens"});
        requested.has_value() && *requested > 0) {
        max_tokens = std::min(max_tokens, *requested);
    }
    if (max_tokens <= 0) {
        runtime::TaskResult result;
        result.text_output = runtime::Transcript{"", "en"};
        return result;
    }

    std::vector<int32_t> prefix = {static_cast<int32_t>(assets.config.decoder.decoder_start_token_id)};
    std::vector<int32_t> decoded;
    decoded.reserve(static_cast<size_t>(max_tokens));
    const auto decode_start = Clock::now();
    CachedDecoderGraph decoder_graph(
        assets,
        weights,
        execution_context,
        max_tokens,
        memory_frames,
        config.graph_arena_bytes);
    decoder_graph.prepare_memory(memory, execution_context);
    double decode_step_select_ms = 0.0;
    for (int64_t step = 0; step < max_tokens; ++step) {
        const auto logits = decoder_graph.run_step(prefix.back(), static_cast<int32_t>(step), execution_context);
        const auto select_start = Clock::now();
        if (logits.size() != static_cast<size_t>(decoder_graph.vocab_size())) {
            throw std::runtime_error("Moonshine STT logits shape mismatch");
        }
        const int32_t next = sampling::HfLogitsProcessor::argmax(
            logits.data(),
            logits.size(),
            "Moonshine STT greedy decode");
        decode_step_select_ms += engine::debug::elapsed_ms(select_start);
        if (next == assets.config.decoder.eos_token_id) {
            break;
        }
        decoded.push_back(next);
        prefix.push_back(next);
    }
    decoder_graph.log_step_timings();
    debug::timing_log_scalar("moonshine_stt.decode_step_select_ms", decode_step_select_ms);
    debug::timing_log_scalar("moonshine_stt.decode_ms", engine::debug::elapsed_ms(decode_start));
    debug::trace_log_scalar("moonshine_stt.output_tokens", decoded.size());

    runtime::TaskResult result;
    result.text_output = runtime::Transcript{decode_moonshine_tokens(assets, decoded), "en"};
    debug::timing_log_scalar("moonshine_stt.run_ms", engine::debug::elapsed_ms(start, Clock::now()));
    return result;
}


}  // namespace engine::models::moonshine_stt
