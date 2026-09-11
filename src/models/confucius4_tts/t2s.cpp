#include "engine/models/confucius4_tts/t2s.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/optimizations/fast_kv_modules.h"
#include "engine/framework/modules/optimizations/fast_projection_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"
#include "engine/framework/sampling/torch_random.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::confucius4_tts {
namespace {

namespace binding = engine::modules::binding;
namespace core = engine::core;
namespace modules = engine::modules;
using Clock = std::chrono::steady_clock;

constexpr float kLayerNormEps = 1.0e-5F;
constexpr int64_t kSpeakerHidden = 512;
constexpr int64_t kSpeakerScale = 8;
constexpr int64_t kSpeakerWidth = kSpeakerHidden / kSpeakerScale;
constexpr int64_t kInitialDecodeSemanticCapacity = 256;
constexpr float kStatsEps = 1.0e-12F;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

ConfuciusT2SConvWeights load_speaker_conv(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    int64_t out_channels,
    int64_t in_channels,
    int64_t kernel,
    int64_t dilation = 1) {
    ConfuciusT2SConvWeights conv;
    conv.in_channels = in_channels;
    conv.out_channels = out_channels;
    conv.kernel = kernel;
    conv.dilation = dilation;
    conv.weights = binding::conv1d_from_source(
        store,
        source,
        prefix,
        storage_type,
        out_channels,
        in_channels,
        kernel,
        true);
    return conv;
}

ConfuciusT2SSpeakerWeights load_speaker_weights(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    assets::TensorStorageType conv_storage_type,
    int64_t input_dim,
    int64_t output_dim) {
    ConfuciusT2SSpeakerWeights weights;
    weights.block0 = load_speaker_conv(store, source, "speaker_encoder.blocks.0.conv", conv_storage_type, kSpeakerHidden, input_dim, 5);
    for (int64_t block = 1; block <= 3; ++block) {
        ConfuciusT2SSERes2NetWeights layer;
        const int64_t dilation = block + 1;
        const std::string prefix = "speaker_encoder.blocks." + std::to_string(block);
        layer.tdnn1 = load_speaker_conv(store, source, prefix + ".tdnn1.conv", conv_storage_type, kSpeakerHidden, kSpeakerHidden, 1);
        for (int64_t i = 0; i < kSpeakerScale - 1; ++i) {
            layer.res2net.push_back(load_speaker_conv(
                store,
                source,
                prefix + ".res2net_block.blocks." + std::to_string(i) + ".conv",
                conv_storage_type,
                kSpeakerWidth,
                kSpeakerWidth,
                3,
                dilation));
        }
        layer.tdnn2 = load_speaker_conv(store, source, prefix + ".tdnn2.conv", conv_storage_type, kSpeakerHidden, kSpeakerHidden, 1);
        layer.se_conv1 = load_speaker_conv(store, source, prefix + ".se_block.conv1", conv_storage_type, 128, kSpeakerHidden, 1);
        layer.se_conv2 = load_speaker_conv(store, source, prefix + ".se_block.conv2", conv_storage_type, kSpeakerHidden, 128, 1);
        weights.blocks.push_back(std::move(layer));
    }
    weights.mfa = load_speaker_conv(store, source, "speaker_encoder.mfa.conv", conv_storage_type, 1536, 1536, 1);
    weights.asp_tdnn = load_speaker_conv(store, source, "speaker_encoder.asp.tdnn.conv", conv_storage_type, 128, 4608, 1);
    weights.asp_conv = load_speaker_conv(store, source, "speaker_encoder.asp.conv", conv_storage_type, 1536, 128, 1);
    weights.fc = load_speaker_conv(store, source, "speaker_encoder.fc", conv_storage_type, output_dim, 3072, 1);
    return weights;
}

core::TensorValue build_linear(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t in_features,
    int64_t out_features,
    const modules::LinearWeights & weights,
    ggml_prec precision = GGML_PREC_DEFAULT) {
    if (ctx.backend_type != core::BackendType::Cuda || !weights.bias.has_value() || out_features % 4 != 0) {
        return modules::LinearModule({in_features, out_features, weights.bias.has_value(), precision}).build(ctx, input, weights);
    }
    auto projected = modules::FastPackedProjection4Module({in_features, out_features, precision})
                         .build(ctx, input, {weights.weight, std::nullopt});
    const auto matrix_shape = core::TensorShape::from_dims({projected.shape.prefix_elements(), out_features});
    auto matrix = core::reshape_tensor(ctx, projected, matrix_shape);
    matrix = core::wrap_tensor(ggml_add(ctx.ggml, matrix.tensor, weights.bias->tensor), matrix_shape, GGML_TYPE_F32);
    return core::reshape_tensor(ctx, matrix, input.shape.with_last_dim(out_features));
}

core::TensorValue conv1d_same(
    core::ModuleBuildContext & ctx,
    core::TensorValue input,
    const ConfuciusT2SConvWeights & conv) {
    const int64_t padding = conv.dilation * (conv.kernel - 1) / 2;
    if (padding > 0) {
        input = modules::ReflectPad1dModule({padding, padding}).build(ctx, input);
    }
    return modules::Conv1dModule({
        conv.in_channels,
        conv.out_channels,
        conv.kernel,
        1,
        0,
        static_cast<int>(conv.dilation),
        true,
    }).build(ctx, input, conv.weights);
}

core::TensorValue tdnn(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const ConfuciusT2SConvWeights & conv) {
    return modules::ReluModule{}.build(ctx, conv1d_same(ctx, input, conv));
}

core::TensorValue se_res2net(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const ConfuciusT2SSERes2NetWeights & weights) {
    auto hidden = tdnn(ctx, input, weights.tdnn1);
    core::TensorValue merged;
    core::TensorValue previous;
    for (int64_t i = 0; i < kSpeakerScale; ++i) {
        auto chunk = modules::SliceModule({1, i * kSpeakerWidth, kSpeakerWidth}).build(ctx, hidden);
        core::TensorValue out;
        if (i == 0) {
            out = chunk;
        } else if (i == 1) {
            out = tdnn(ctx, chunk, weights.res2net[0]);
        } else {
            out = tdnn(ctx, modules::AddModule{}.build(ctx, chunk, previous), weights.res2net[static_cast<size_t>(i - 1)]);
        }
        previous = out;
        merged = merged.valid() ? modules::ConcatModule({1}).build(ctx, merged, out) : out;
    }
    hidden = tdnn(ctx, merged, weights.tdnn2);
    hidden = modules::SqueezeExcite1dModule({kSpeakerHidden, 128, true}).build(
        ctx,
        hidden,
        {weights.se_conv1.weights, weights.se_conv2.weights});
    return modules::AddModule{}.build(ctx, hidden, input);
}

core::TensorValue attentive_statistics_pool(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const ConfuciusT2SConvWeights & tdnn_conv,
    const ConfuciusT2SConvWeights & attention_conv,
    const core::TensorValue & eps) {
    auto mean = modules::ReduceMeanModule({2}).build(ctx, input);
    auto mean_rep = modules::RepeatModule({input.shape}).build(ctx, mean);
    auto centered = core::wrap_tensor(ggml_sub(ctx.ggml, input.tensor, mean_rep.tensor), input.shape, GGML_TYPE_F32);
    auto variance = modules::ReduceMeanModule({2}).build(ctx, modules::MulModule{}.build(ctx, centered, centered));
    auto eps_stats = modules::RepeatModule({variance.shape}).build(ctx, eps);
    auto std = modules::SqrtModule{}.build(ctx, modules::AddModule{}.build(ctx, variance, eps_stats));
    auto std_rep = modules::RepeatModule({input.shape}).build(ctx, std);
    auto attention = modules::ConcatModule({1}).build(ctx, modules::ConcatModule({1}).build(ctx, input, mean_rep), std_rep);
    attention = tdnn(ctx, attention, tdnn_conv);
    attention = modules::TanhModule{}.build(ctx, attention);
    attention = conv1d_same(ctx, attention, attention_conv);
    auto weights = modules::SoftmaxModule{}.build(ctx, attention);
    auto weighted = modules::MulModule{}.build(ctx, input, weights);
    auto mean_att = modules::ReduceSumModule({2}).build(ctx, weighted);
    auto mean_att_rep = modules::RepeatModule({input.shape}).build(ctx, mean_att);
    auto diff = core::wrap_tensor(ggml_sub(ctx.ggml, input.tensor, mean_att_rep.tensor), input.shape, GGML_TYPE_F32);
    auto var_att = modules::ReduceSumModule({2}).build(ctx, modules::MulModule{}.build(ctx, modules::MulModule{}.build(ctx, diff, diff), weights));
    eps_stats = modules::RepeatModule({var_att.shape}).build(ctx, eps);
    auto std_att = modules::SqrtModule{}.build(ctx, modules::AddModule{}.build(ctx, var_att, eps_stats));
    return modules::ConcatModule({1}).build(ctx, mean_att, std_att);
}

core::TensorValue speaker_encoder(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & semantic,
    const ConfuciusT2SSpeakerWeights & weights,
    const core::TensorValue & eps) {
    auto hidden = tdnn(ctx, semantic, weights.block0);
    std::vector<core::TensorValue> layer_outputs;
    for (const auto & block : weights.blocks) {
        hidden = se_res2net(ctx, hidden, block);
        layer_outputs.push_back(hidden);
    }
    hidden = modules::ConcatModule({1}).build(
        ctx,
        modules::ConcatModule({1}).build(ctx, layer_outputs[0], layer_outputs[1]),
        layer_outputs[2]);
    hidden = tdnn(ctx, hidden, weights.mfa);
    hidden = attentive_statistics_pool(ctx, hidden, weights.asp_tdnn, weights.asp_conv, eps);
    hidden = conv1d_same(ctx, hidden, weights.fc);
    return core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, hidden),
        core::TensorShape::from_dims({1, 1, hidden.shape.dims[1]}));
}

core::TensorValue reshape_heads(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t heads,
    int64_t head_dim) {
    return core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, input),
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], heads, head_dim}));
}

core::TensorValue gpt_attention(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & q_heads,
    const core::TensorValue & k_heads,
    const core::TensorValue & v_heads,
    int64_t head_dim,
    const std::optional<core::TensorValue> & attention_mask) {
    if (attention_mask.has_value()) {
        auto scores = modules::MatMulModule{}.build(ctx, q_heads, modules::TransposeModule({{0, 1, 3, 2}, 4}).build(ctx, k_heads));
        scores = core::ensure_backend_addressable_layout(ctx, scores);
        scores = core::wrap_tensor(
            ggml_soft_max_ext(
                ctx.ggml,
                scores.tensor,
                core::ensure_backend_addressable_layout(ctx, *attention_mask).tensor,
                1.0F / std::sqrt(static_cast<float>(head_dim)),
                0.0F),
            scores.shape,
            GGML_TYPE_F32);
        return modules::MatMulModule{}.build(ctx, scores, v_heads);
    }
    auto scores = modules::MatMulModule{}.build(ctx, q_heads, modules::TransposeModule({{0, 1, 3, 2}, 4}).build(ctx, k_heads));
    scores = core::wrap_tensor(
        ggml_scale(ctx.ggml, scores.tensor, 1.0F / std::sqrt(static_cast<float>(head_dim))),
        scores.shape,
        GGML_TYPE_F32);
    scores = core::wrap_tensor(ggml_diag_mask_inf(ctx.ggml, scores.tensor, 0), scores.shape, GGML_TYPE_F32);
    scores = core::wrap_tensor(ggml_soft_max(ctx.ggml, core::ensure_backend_addressable_layout(ctx, scores).tensor), scores.shape, GGML_TYPE_F32);
    return modules::MatMulModule{}.build(ctx, scores, v_heads);
}

core::TensorValue gpt_mlp(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const ConfuciusT2SLayerWeights & weights,
    const ConfuciusT2SConfig & config) {
    auto hidden = build_linear(ctx, input, config.model_dim, config.model_dim * 4, weights.mlp_in, GGML_PREC_F32);
    hidden = modules::GeluModule({modules::GeluApproximation::Tanh}).build(ctx, hidden);
    return build_linear(ctx, hidden, config.model_dim * 4, config.model_dim, weights.mlp_out, GGML_PREC_F32);
}

struct GptLayerOutput {
    core::TensorValue output;
    core::TensorValue key;
    core::TensorValue value;
};

GptLayerOutput gpt_layer_full(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const ConfuciusT2SLayerWeights & weights,
    const ConfuciusT2SConfig & config,
    const std::optional<core::TensorValue> & attention_mask = std::nullopt) {
    const int64_t head_dim = config.model_dim / config.num_heads;
    auto normed = modules::LayerNormModule({config.model_dim, kLayerNormEps, true, true}).build(ctx, input, weights.attn_norm);
    auto qkv = build_linear(ctx, normed, config.model_dim, 3 * config.model_dim, weights.qkv, GGML_PREC_F32);
    auto q = modules::SliceModule({2, 0, config.model_dim}).build(ctx, qkv);
    auto k = modules::SliceModule({2, config.model_dim, config.model_dim}).build(ctx, qkv);
    auto v = modules::SliceModule({2, 2 * config.model_dim, config.model_dim}).build(ctx, qkv);
    auto key_cache = reshape_heads(ctx, k, config.num_heads, head_dim);
    auto value_cache = reshape_heads(ctx, v, config.num_heads, head_dim);
    auto q_heads = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, reshape_heads(ctx, q, config.num_heads, head_dim));
    auto k_heads = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, key_cache);
    auto v_heads = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, value_cache);
    auto context = gpt_attention(ctx, q_heads, k_heads, v_heads, head_dim, attention_mask);
    context = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, context);
    context = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, context), input.shape);
    auto hidden = modules::AddModule{}.build(ctx, input, build_linear(ctx, context, config.model_dim, config.model_dim, weights.attn_out, GGML_PREC_F32));
    auto mlp_in = modules::LayerNormModule({config.model_dim, kLayerNormEps, true, true}).build(ctx, hidden, weights.mlp_norm);
    return {modules::AddModule{}.build(ctx, hidden, gpt_mlp(ctx, mlp_in, weights, config)), key_cache, value_cache};
}

GptLayerOutput gpt_layer_cached_tail(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const ConfuciusT2SLayerWeights & weights,
    const ConfuciusT2SConfig & config,
    const core::TensorValue & cache_key,
    const core::TensorValue & cache_value,
    const core::TensorValue & cache_slots,
    const core::TensorValue & attention_mask) {
    const int64_t head_dim = config.model_dim / config.num_heads;
    if (cache_key.shape.dims[0] != input.shape.dims[0] ||
        cache_value.shape.dims[0] != input.shape.dims[0] ||
        cache_key.shape.dims[1] != cache_value.shape.dims[1] ||
        cache_key.shape.dims[2] != cache_value.shape.dims[2] ||
        cache_key.shape.dims[3] != cache_value.shape.dims[3] ||
        cache_key.shape.dims[2] != config.num_heads ||
        cache_key.shape.dims[3] != head_dim ||
        cache_slots.shape.dims[0] != input.shape.dims[0]) {
        throw std::runtime_error("Confucius4-TTS T2S cached layer shape mismatch");
    }
    auto normed = modules::LayerNormModule({config.model_dim, kLayerNormEps, true, true}).build(ctx, input, weights.attn_norm);
    auto qkv = build_linear(ctx, normed, config.model_dim, 3 * config.model_dim, weights.qkv, GGML_PREC_F32);
    auto q = modules::SliceModule({2, 0, config.model_dim}).build(ctx, qkv);
    auto k = modules::SliceModule({2, config.model_dim, config.model_dim}).build(ctx, qkv);
    auto v = modules::SliceModule({2, 2 * config.model_dim, config.model_dim}).build(ctx, qkv);
    q = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, reshape_heads(ctx, q, config.num_heads, head_dim));
    k = reshape_heads(ctx, k, config.num_heads, head_dim);
    v = reshape_heads(ctx, v, config.num_heads, head_dim);

    const modules::FastKVSetRowsModule set_rows;
    auto updated_key = set_rows.build(ctx, cache_key, k, cache_slots);
    auto updated_value = set_rows.build(ctx, cache_value, v, cache_slots);
    auto context = gpt_attention(
        ctx,
        q,
        modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, updated_key),
        modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, updated_value),
        head_dim,
        attention_mask);
    context = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, context), input.shape);
    auto hidden = modules::AddModule{}.build(ctx, input, build_linear(ctx, context, config.model_dim, config.model_dim, weights.attn_out, GGML_PREC_F32));
    auto mlp_in = modules::LayerNormModule({config.model_dim, kLayerNormEps, true, true}).build(ctx, hidden, weights.mlp_norm);
    return {modules::AddModule{}.build(ctx, hidden, gpt_mlp(ctx, mlp_in, weights, config)), k, v};
}

struct TopPItem {
    size_t index = 0;
    float score = 0.0F;
    float weight = 0.0F;
};

struct SampleScore {
    size_t flat_index = 0;
    float score = 0.0F;
};

struct RankedSample {
    size_t score_index = 0;
    size_t flat_index = 0;
    double rank = 0.0;
};

struct RunningCandidate {
    size_t topk_rank = 0;
    size_t parent = 0;
    int32_t token = 0;
    float score = 0.0F;
};

struct SamplerWorkspace {
    std::vector<float> scores;
    std::vector<TopPItem> top_p_items;
    std::vector<uint32_t> seen_tokens;
    uint32_t seen_generation = 1;
    std::vector<size_t> finite_score_indices;
    std::vector<size_t> top_k_indices;
    std::vector<SampleScore> sample_scores;
    std::vector<RankedSample> ranked_samples;
    std::vector<size_t> selected_scores;
    std::vector<RunningCandidate> running_candidates;
};

void apply_repetition_penalty(
    std::vector<float> & logits,
    const std::vector<int32_t> & codes,
    int32_t bos_token,
    float penalty,
    SamplerWorkspace & workspace) {
    if (penalty == 1.0F) {
        return;
    }
    if (!(penalty > 0.0F)) {
        throw std::runtime_error("Confucius4-TTS T2S repetition_penalty must be positive");
    }
    if (workspace.seen_tokens.size() != logits.size()) {
        workspace.seen_tokens.assign(logits.size(), 0);
        workspace.seen_generation = 1;
    } else if (workspace.seen_generation == 0) {
        std::fill(workspace.seen_tokens.begin(), workspace.seen_tokens.end(), 0);
        workspace.seen_generation = 1;
    }
    const uint32_t generation = workspace.seen_generation++;
    const auto apply = [&](int32_t token) {
        if (token < 0 || static_cast<size_t>(token) >= logits.size() || workspace.seen_tokens[static_cast<size_t>(token)] == generation) {
            return;
        }
        workspace.seen_tokens[static_cast<size_t>(token)] = generation;
        float & value = logits[static_cast<size_t>(token)];
        value = value < 0.0F ? value * penalty : value / penalty;
    };
    apply(bos_token);
    for (const int32_t token : codes) {
        apply(token);
    }
}

void log_probs(
    const std::vector<float> & logits,
    const std::vector<int32_t> & codes,
    const ConfuciusGenerationOptions & options,
    int32_t bos_token,
    SamplerWorkspace & workspace) {
    if (!(options.temperature > 0.0F)) {
        throw std::runtime_error("Confucius4-TTS T2S temperature must be positive");
    }
    auto & scores = workspace.scores;
    if (options.num_beams == 1) {
        scores = logits;
    } else {
        float max_logit = -std::numeric_limits<float>::infinity();
        for (float logit : logits) {
            max_logit = std::max(max_logit, logit);
        }
        float total = 0.0F;
        for (float logit : logits) {
            total += std::exp(logit - max_logit);
        }
        if (!(total > 0.0F)) {
            throw std::runtime_error("Confucius4-TTS T2S sampler invalid logit mass");
        }
        const float log_total = std::log(total);
        scores.resize(logits.size());
        for (size_t i = 0; i < logits.size(); ++i) {
            scores[i] = logits[i] - max_logit - log_total;
        }
    }
    apply_repetition_penalty(scores, codes, bos_token, options.repetition_penalty, workspace);
    for (float & score : scores) {
        score /= options.temperature;
    }
    const size_t min_tokens_to_keep = options.num_beams > 1 ? 2 : 1;
    auto & finite_indices = workspace.finite_score_indices;
    finite_indices.clear();
    for (size_t i = 0; i < scores.size(); ++i) {
        if (std::isfinite(scores[i])) {
            finite_indices.push_back(i);
        }
    }
    if (options.top_k > 0 && static_cast<size_t>(options.top_k) < finite_indices.size()) {
        const size_t keep_count = std::max(static_cast<size_t>(options.top_k), min_tokens_to_keep);
        auto & top_k_indices = workspace.top_k_indices;
        top_k_indices = finite_indices;
        auto keep_end = top_k_indices.begin() + static_cast<std::ptrdiff_t>(keep_count - 1);
        std::nth_element(
            top_k_indices.begin(),
            keep_end,
            top_k_indices.end(),
            [&](size_t lhs, size_t rhs) {
                return scores[lhs] == scores[rhs] ? lhs < rhs : scores[lhs] > scores[rhs];
            });
        const float threshold = scores[*keep_end];
        finite_indices.clear();
        for (size_t i = 0; i < scores.size(); ++i) {
            if (std::isfinite(scores[i]) && scores[i] >= threshold) {
                finite_indices.push_back(i);
            }
        }
        for (size_t i = 0; i < scores.size(); ++i) {
            if (!std::isfinite(scores[i]) || scores[i] < threshold) {
                scores[i] = -std::numeric_limits<float>::infinity();
            }
        }
    }
    if (finite_indices.empty()) {
        throw std::runtime_error("Confucius4-TTS T2S sampler has no finite score");
    }
    if (options.top_p > 0.0F && options.top_p < 1.0F) {
        auto & sorted = workspace.top_p_items;
        sorted.clear();
        sorted.reserve(finite_indices.size());
        float max_score = -std::numeric_limits<float>::infinity();
        for (const size_t i : finite_indices) {
            max_score = std::max(max_score, scores[i]);
        }
        float total_weight = 0.0F;
        for (const size_t i : finite_indices) {
            const float weight = std::exp(scores[i] - max_score);
            sorted.push_back({i, scores[i], weight});
            total_weight += weight;
        }
        std::sort(sorted.begin(), sorted.end(), [](const TopPItem & lhs, const TopPItem & rhs) {
            return lhs.score == rhs.score ? lhs.index < rhs.index : lhs.score < rhs.score;
        });
        const float remove_mass = 1.0F - options.top_p;
        const size_t keep_from = sorted.size() > min_tokens_to_keep ? sorted.size() - min_tokens_to_keep : 0;
        float cumulative = 0.0F;
        finite_indices.clear();
        for (size_t i = 0; i < sorted.size(); ++i) {
            cumulative += sorted[i].weight / total_weight;
            if (i < keep_from && cumulative <= remove_mass) {
                scores[sorted[i].index] = -std::numeric_limits<float>::infinity();
            } else {
                finite_indices.push_back(sorted[i].index);
            }
        }
    }
}

void sample_indices(
    const std::vector<SampleScore> & scores,
    size_t total_score_count,
    size_t count,
    uint64_t seed,
    uint64_t step,
    const sampling::TorchCudaSamplingPolicy & policy,
    std::vector<RankedSample> & ranked,
    std::vector<size_t> & selected) {
    float max_score = -std::numeric_limits<float>::infinity();
    for (const auto & score : scores) {
        max_score = std::max(max_score, score.score);
    }
    if (!std::isfinite(max_score)) {
        throw std::runtime_error("Confucius4-TTS T2S sampler has no finite beam score");
    }
    ranked.clear();
    ranked.reserve(scores.size());
    for (size_t i = 0; i < scores.size(); ++i) {
        const auto & score = scores[i];
        const float probability = std::exp(score.score - max_score);
        const float exponential = sampling::torch_cuda_tensor_iterator_exponential_element(
            seed,
            static_cast<uint64_t>(total_score_count),
            static_cast<uint64_t>(score.flat_index),
            step,
            policy.multiprocessor_count,
            policy.max_threads_per_multiprocessor);
        ranked.push_back({i, score.flat_index, static_cast<double>(probability) / static_cast<double>(exponential)});
    }
    const size_t keep = std::min(count, ranked.size());
    std::partial_sort(
        ranked.begin(),
        ranked.begin() + static_cast<std::ptrdiff_t>(keep),
        ranked.end(),
        [](const RankedSample & lhs, const RankedSample & rhs) {
            return lhs.rank == rhs.rank ? lhs.flat_index < rhs.flat_index : lhs.rank > rhs.rank;
        });
    selected.clear();
    selected.reserve(keep);
    for (size_t i = 0; i < keep; ++i) {
        selected.push_back(ranked[i].score_index);
    }
}

}  // namespace

std::shared_ptr<const ConfuciusT2SWeights> load_confucius_t2s_weights(
    const ConfuciusAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    assets::TensorStorageType matmul_storage_type,
    assets::TensorStorageType conv_storage_type,
    size_t weight_context_bytes) {
    auto weights = std::make_shared<ConfuciusT2SWeights>();
    weights->store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "confucius4_tts.t2s.weights",
        weight_context_bytes);
    const auto & source = *assets.t2s_weights;
    auto & store = *weights->store;
    const auto & config = assets.config.t2s;
    weights->text_embedding = store.load_tensor(
        source,
        "text_projector.embed.weight",
        matmul_storage_type,
        {config.vocab_size, config.text_embedding_dim});
    weights->text_fc1 = binding::linear_from_source(
        store,
        source,
        "text_projector.text_projection_fc1",
        matmul_storage_type,
        config.text_embedding_dim,
        config.text_embedding_dim,
        true);
    weights->text_fc2 = binding::linear_from_source(
        store,
        source,
        "text_projector.text_projection_fc2",
        matmul_storage_type,
        config.model_dim,
        config.text_embedding_dim,
        true);
    weights->semantic_embedding = store.load_tensor(
        source,
        "semantic_embedding.weight",
        matmul_storage_type,
        {config.semantic_vocab_size, config.model_dim});
    weights->text_pos_embedding = store.load_f32_tensor(
        source,
        "text_position_embedding.embedding.weight",
        {config.max_text_seq_lens, config.model_dim});
    weights->semantic_pos_embedding = store.load_f32_tensor(
        source,
        "semantic_position_embedding.embedding.weight",
        {config.max_semantic_seq_lens, config.model_dim});
    weights->layers.reserve(static_cast<size_t>(config.num_layers));
    for (int64_t layer = 0; layer < config.num_layers; ++layer) {
        const std::string prefix = "transformer.h." + std::to_string(layer);
        ConfuciusT2SLayerWeights layer_weights;
        layer_weights.attn_norm = binding::norm_from_source(
            store,
            source,
            prefix + ".ln_1",
            config.model_dim);
        layer_weights.qkv = binding::hf_conv1d_linear_from_source(
            store,
            source,
            prefix + ".attn.c_attn",
            matmul_storage_type,
            config.model_dim,
            3 * config.model_dim,
            true);
        layer_weights.attn_out = binding::hf_conv1d_linear_from_source(
            store,
            source,
            prefix + ".attn.c_proj",
            matmul_storage_type,
            config.model_dim,
            config.model_dim,
            true);
        layer_weights.mlp_norm = binding::norm_from_source(
            store,
            source,
            prefix + ".ln_2",
            config.model_dim);
        layer_weights.mlp_in = binding::hf_conv1d_linear_from_source(
            store,
            source,
            prefix + ".mlp.c_fc",
            matmul_storage_type,
            config.model_dim,
            4 * config.model_dim,
            true);
        layer_weights.mlp_out = binding::hf_conv1d_linear_from_source(
            store,
            source,
            prefix + ".mlp.c_proj",
            matmul_storage_type,
            4 * config.model_dim,
            config.model_dim,
            true);
        weights->layers.push_back(std::move(layer_weights));
    }
    weights->gpt_final_norm = binding::norm_from_source(
        store,
        source,
        "transformer.ln_f",
        config.model_dim);
    weights->final_norm = binding::norm_from_source(store, source, "final_norm", config.model_dim);
    weights->semantic_head = binding::linear_from_source(
        store,
        source,
        "semantic_head",
        matmul_storage_type,
        config.semantic_vocab_size,
        config.model_dim,
        true);
    weights->speaker = load_speaker_weights(
        store,
        source,
        conv_storage_type,
        config.speaker_embedding_dim,
        config.model_dim);
    store.upload();
    assets.t2s_weights->release_storage();
    return weights;
}

class ConfuciusT2SRuntime::SpeakerConditionGraph {
public:
    SpeakerConditionGraph(
        core::ExecutionContext & execution,
        std::shared_ptr<const ConfuciusT2SWeights> weights,
        const ConfuciusT2SConfig & config,
        int64_t frames,
        size_t graph_arena_bytes)
        : execution_(execution),
          weights_(std::move(weights)),
          frames_(frames) {
        if (weights_ == nullptr || frames_ <= 0) {
            throw std::runtime_error("Confucius4-TTS T2S speaker condition graph requires weights and frames");
        }
        const auto build_start = Clock::now();
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize Confucius4-TTS T2S speaker condition graph context");
        }
        ggml_init_params input_params{32ull * 1024ull * 1024ull, nullptr, true};
        input_ctx_.reset(ggml_init(input_params));
        if (input_ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize Confucius4-TTS T2S speaker condition input context");
        }
        core::ModuleBuildContext ctx{ctx_.get(), "confucius4_tts.t2s.speaker", execution_.backend_type()};
        core::ModuleBuildContext input_ctx{input_ctx_.get(), "confucius4_tts.t2s.speaker.inputs", execution_.backend_type()};
        semantic_ = core::make_tensor(
                        input_ctx,
                        GGML_TYPE_F32,
                        core::TensorShape::from_dims({1, config.speaker_embedding_dim, frames_}))
                        .tensor;
        eps_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 1, 1})).tensor;
        ggml_set_input(semantic_);
        ggml_set_input(eps_);
        auto semantic = core::wrap_tensor(
            semantic_,
            core::TensorShape::from_dims({1, config.speaker_embedding_dim, frames_}),
            GGML_TYPE_F32);
        auto eps = core::wrap_tensor(eps_, core::TensorShape::from_dims({1, 1, 1}), GGML_TYPE_F32);
        auto output = speaker_encoder(ctx, semantic, weights_->speaker, eps);
        output_ = core::ensure_backend_addressable_layout(ctx, output).tensor;
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 65536, false);
        ggml_build_forward_expand(graph_, output_);
        debug::trace_log_scalar("confucius4_tts.t2s.speaker.graph_nodes", ggml_graph_n_nodes(graph_));
        input_buffer_ = ggml_backend_alloc_ctx_tensors(input_ctx_.get(), execution_.backend());
        if (input_buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate Confucius4-TTS T2S speaker input buffer");
        }
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend()));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            clear_graph();
            throw std::runtime_error("failed to allocate Confucius4-TTS T2S speaker graph");
        }
        debug::timing_log_scalar("confucius4_tts.t2s.speaker.graph.build_ms", debug::elapsed_ms(build_start));
    }

    ~SpeakerConditionGraph() {
        clear_graph();
    }

    bool matches(int64_t frames) const noexcept {
        return frames_ == frames;
    }

    std::vector<float> run(const ConfuciusSemanticEmbedding & semantic) {
        if (semantic.frames != frames_ ||
            semantic.dims != weights_->speaker.block0.in_channels ||
            static_cast<int64_t>(semantic.values.size()) != frames_ * semantic.dims) {
            throw std::runtime_error("Confucius4-TTS T2S speaker semantic shape mismatch");
        }
        std::vector<float> bct(static_cast<size_t>(semantic.dims * frames_));
        for (int64_t t = 0; t < frames_; ++t) {
            for (int64_t c = 0; c < semantic.dims; ++c) {
                bct[static_cast<size_t>(c * frames_ + t)] =
                    semantic.values[static_cast<size_t>(t * semantic.dims + c)];
            }
        }
        auto start = Clock::now();
        ggml_backend_tensor_set(semantic_, bct.data(), 0, bct.size() * sizeof(float));
        const float eps = kStatsEps;
        ggml_backend_tensor_set(eps_, &eps, 0, sizeof(float));
        debug::timing_log_scalar("confucius4_tts.t2s.speaker.input_upload_ms", debug::elapsed_ms(start));
        core::set_backend_threads(execution_.backend(), execution_.config().threads);
        start = Clock::now();
        const ggml_status status = core::compute_backend_graph(execution_.backend(), graph_);
        ggml_backend_synchronize(execution_.backend());
        debug::timing_log_scalar("confucius4_tts.t2s.speaker.graph.compute_ms", debug::elapsed_ms(start));
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Confucius4-TTS T2S speaker graph compute failed");
        }
        std::vector<float> output(static_cast<size_t>(weights_->semantic_head.weight.shape.dims[1]));
        start = Clock::now();
        ggml_backend_tensor_get(output_, output.data(), 0, output.size() * sizeof(float));
        debug::timing_log_scalar("confucius4_tts.t2s.speaker.output_read_ms", debug::elapsed_ms(start));
        return output;
    }

private:
    void clear_graph() {
        if (graph_ != nullptr) {
            core::release_backend_graph_resources(execution_.backend(), graph_);
            graph_ = nullptr;
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
            gallocr_ = nullptr;
        }
        if (input_buffer_ != nullptr) {
            ggml_backend_buffer_free(input_buffer_);
            input_buffer_ = nullptr;
        }
    }

    core::ExecutionContext & execution_;
    std::shared_ptr<const ConfuciusT2SWeights> weights_;
    int64_t frames_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> input_ctx_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * semantic_ = nullptr;
    ggml_tensor * eps_ = nullptr;
    ggml_tensor * output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    ggml_backend_buffer_t input_buffer_ = nullptr;
};

struct T2SPrefillOutput {
    std::vector<float> logits;
    runtime::TransformerKVState kv_state;
};

int64_t initial_decode_cache_steps(int64_t prompt_steps, int64_t max_semantic_tokens) {
    const int64_t semantic_capacity = std::min<int64_t>(
        std::max<int64_t>(1, max_semantic_tokens + 1),
        kInitialDecodeSemanticCapacity);
    return prompt_steps + semantic_capacity;
}

int64_t grown_decode_cache_steps(
    int64_t prompt_steps,
    int64_t current_cache_steps,
    int64_t required_cache_steps,
    int64_t max_semantic_tokens) {
    const int64_t max_cache_steps = prompt_steps + std::max<int64_t>(1, max_semantic_tokens + 1);
    int64_t semantic_capacity = std::max<int64_t>(1, current_cache_steps - prompt_steps);
    const int64_t required_semantic_capacity = std::max<int64_t>(1, required_cache_steps - prompt_steps);
    while (semantic_capacity < required_semantic_capacity && prompt_steps + semantic_capacity < max_cache_steps) {
        semantic_capacity *= 2;
    }
    return std::min<int64_t>(max_cache_steps, prompt_steps + std::max(semantic_capacity, required_semantic_capacity));
}

class ConfuciusT2SRuntime::PrefillGraph {
public:
    PrefillGraph(
        core::ExecutionContext & execution,
        std::shared_ptr<const ConfuciusT2SWeights> weights,
        const ConfuciusT2SConfig & config,
        int64_t text_tokens,
        size_t graph_arena_bytes)
        : execution_(execution),
          weights_(std::move(weights)),
          config_(config),
          text_tokens_(text_tokens),
          prompt_steps_(text_tokens + 2) {
        if (weights_ == nullptr || text_tokens_ <= 0) {
            throw std::runtime_error("Confucius4-TTS T2S prefill graph requires weights and text tokens");
        }
        const auto build_start = Clock::now();
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize Confucius4-TTS T2S prefill graph context");
        }
        ggml_init_params input_params{32ull * 1024ull * 1024ull, nullptr, true};
        input_ctx_.reset(ggml_init(input_params));
        if (input_ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize Confucius4-TTS T2S prefill input context");
        }
        ggml_init_params output_params{64ull * 1024ull * 1024ull, nullptr, true};
        output_ctx_.reset(ggml_init(output_params));
        if (output_ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize Confucius4-TTS T2S prefill output context");
        }
        core::ModuleBuildContext ctx{ctx_.get(), "confucius4_tts.t2s.prefill", execution_.backend_type()};
        core::ModuleBuildContext input_ctx{input_ctx_.get(), "confucius4_tts.t2s.prefill.inputs", execution_.backend_type()};
        core::ModuleBuildContext output_ctx{output_ctx_.get(), "confucius4_tts.t2s.prefill.outputs", execution_.backend_type()};
        condition_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 1, config_.model_dim})).tensor;
        text_ids_ = core::make_tensor(input_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({text_tokens_})).tensor;
        bos_id_ = core::make_tensor(input_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({1})).tensor;
        ggml_set_input(condition_);
        ggml_set_input(text_ids_);
        ggml_set_input(bos_id_);

        auto condition = core::wrap_tensor(condition_, core::TensorShape::from_dims({1, 1, config_.model_dim}), GGML_TYPE_F32);
        auto text_ids = core::wrap_tensor(text_ids_, core::TensorShape::from_dims({text_tokens_}), GGML_TYPE_I32);
        auto text = modules::EmbeddingModule({config_.vocab_size, config_.text_embedding_dim}).build(ctx, text_ids, weights_->text_embedding);
        text = build_linear(ctx, text, config_.text_embedding_dim, config_.text_embedding_dim, weights_->text_fc1);
        text = modules::SiluModule{}.build(ctx, text);
        text = build_linear(ctx, text, config_.text_embedding_dim, config_.model_dim, weights_->text_fc2);
        auto text_pos = modules::SliceModule({0, 0, text_tokens_}).build(ctx, weights_->text_pos_embedding);
        text = modules::AddModule{}.build(ctx, text, text_pos);
        text = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, text), core::TensorShape::from_dims({1, text_tokens_, config_.model_dim}));

        auto bos_id = core::wrap_tensor(bos_id_, core::TensorShape::from_dims({1}), GGML_TYPE_I32);
        auto semantic = modules::EmbeddingModule({config_.semantic_vocab_size, config_.model_dim}).build(ctx, bos_id, weights_->semantic_embedding);
        auto semantic_pos = modules::SliceModule({0, 0, 1}).build(ctx, weights_->semantic_pos_embedding);
        semantic = modules::AddModule{}.build(ctx, semantic, semantic_pos);
        semantic = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, semantic), core::TensorShape::from_dims({1, 1, config_.model_dim}));

        auto hidden = modules::ConcatModule({1}).build(ctx, modules::ConcatModule({1}).build(ctx, condition, text), semantic);
        graph_ = ggml_new_graph_custom(ctx_.get(), 131072, false);
        keys_.reserve(weights_->layers.size());
        values_.reserve(weights_->layers.size());
        for (const auto & layer : weights_->layers) {
            auto out = gpt_layer_full(ctx, hidden, layer, config_);
            hidden = out.output;
            auto key = core::ensure_backend_addressable_layout(ctx, out.key);
            auto value = core::ensure_backend_addressable_layout(ctx, out.value);
            auto * key_output = core::make_tensor(output_ctx, GGML_TYPE_F32, key.shape).tensor;
            auto * value_output = core::make_tensor(output_ctx, GGML_TYPE_F32, value.shape).tensor;
            keys_.push_back(key_output);
            values_.push_back(value_output);
            ggml_build_forward_expand(graph_, ggml_cpy(ctx_.get(), key.tensor, key_output));
            ggml_build_forward_expand(graph_, ggml_cpy(ctx_.get(), value.tensor, value_output));
        }
        hidden = modules::LayerNormModule({config_.model_dim, kLayerNormEps, true, true}).build(ctx, hidden, weights_->gpt_final_norm);
        hidden = modules::SliceModule({1, prompt_steps_ - 1, 1}).build(ctx, hidden);
        hidden = modules::LayerNormModule({config_.model_dim, kLayerNormEps, true, true}).build(ctx, hidden, weights_->final_norm);
        auto logits = core::ensure_backend_addressable_layout(
            ctx,
            build_linear(ctx, hidden, config_.model_dim, config_.semantic_vocab_size, weights_->semantic_head, GGML_PREC_F32));
        logits_ = core::make_tensor(output_ctx, GGML_TYPE_F32, logits.shape).tensor;
        ggml_set_output(logits_);
        ggml_build_forward_expand(graph_, ggml_cpy(ctx_.get(), logits.tensor, logits_));
        debug::trace_log_scalar("confucius4_tts.t2s.prefill.graph_nodes", ggml_graph_n_nodes(graph_));

        input_buffer_ = ggml_backend_alloc_ctx_tensors(input_ctx_.get(), execution_.backend());
        if (input_buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate Confucius4-TTS T2S prefill input buffer");
        }
        output_buffer_ = ggml_backend_alloc_ctx_tensors(output_ctx_.get(), execution_.backend());
        if (output_buffer_ == nullptr) {
            clear_graph();
            throw std::runtime_error("failed to allocate Confucius4-TTS T2S prefill output buffer");
        }
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend()));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            clear_graph();
            throw std::runtime_error("failed to allocate Confucius4-TTS T2S prefill graph");
        }
        const int32_t bos = static_cast<int32_t>(config_.start_semantic_token);
        ggml_backend_tensor_set(bos_id_, &bos, 0, sizeof(int32_t));
        debug::timing_log_scalar("confucius4_tts.t2s.prefill.graph.build_ms", debug::elapsed_ms(build_start));
        debug::trace_log_scalar("confucius4_tts.t2s.prefill.prompt_steps", prompt_steps_);
    }

    ~PrefillGraph() {
        clear_graph();
    }

    bool matches(int64_t text_tokens) const noexcept {
        return text_tokens_ == text_tokens;
    }

    int64_t prompt_steps() const noexcept {
        return prompt_steps_;
    }

    T2SPrefillOutput run(const std::vector<float> & condition, const std::vector<int32_t> & text_tokens) {
        if (static_cast<int64_t>(condition.size()) != config_.model_dim ||
            static_cast<int64_t>(text_tokens.size()) != text_tokens_) {
            throw std::runtime_error("Confucius4-TTS T2S prefill input shape mismatch");
        }
        auto start = Clock::now();
        ggml_backend_tensor_set(condition_, condition.data(), 0, condition.size() * sizeof(float));
        ggml_backend_tensor_set(text_ids_, text_tokens.data(), 0, text_tokens.size() * sizeof(int32_t));
        debug::timing_log_scalar("confucius4_tts.t2s.prefill.input_upload_ms", debug::elapsed_ms(start));
        core::set_backend_threads(execution_.backend(), execution_.config().threads);
        start = Clock::now();
        const ggml_status status = core::compute_backend_graph(execution_.backend(), graph_);
        ggml_backend_synchronize(execution_.backend());
        debug::timing_log_scalar("confucius4_tts.t2s.prefill.graph.compute_ms", debug::elapsed_ms(start));
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Confucius4-TTS T2S prefill graph compute failed");
        }
        T2SPrefillOutput out;
        out.logits.resize(static_cast<size_t>(config_.semantic_vocab_size));
        ggml_backend_tensor_get(logits_, out.logits.data(), 0, out.logits.size() * sizeof(float));
        out.kv_state.current_end = prompt_steps_;
        out.kv_state.layers.resize(keys_.size());
        const size_t layer_values = static_cast<size_t>(prompt_steps_ * config_.num_heads * (config_.model_dim / config_.num_heads));
        for (size_t layer = 0; layer < keys_.size(); ++layer) {
            auto & state = out.kv_state.layers[layer];
            state.valid_steps = prompt_steps_;
            state.key.resize(layer_values);
            state.value.resize(layer_values);
            ggml_backend_tensor_get(keys_[layer], state.key.data(), 0, state.key.size() * sizeof(float));
            ggml_backend_tensor_get(values_[layer], state.value.data(), 0, state.value.size() * sizeof(float));
        }
        return out;
    }

private:
    void clear_graph() {
        if (graph_ != nullptr) {
            core::release_backend_graph_resources(execution_.backend(), graph_);
            graph_ = nullptr;
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
            gallocr_ = nullptr;
        }
        if (input_buffer_ != nullptr) {
            ggml_backend_buffer_free(input_buffer_);
            input_buffer_ = nullptr;
        }
        if (output_buffer_ != nullptr) {
            ggml_backend_buffer_free(output_buffer_);
            output_buffer_ = nullptr;
        }
    }

    core::ExecutionContext & execution_;
    std::shared_ptr<const ConfuciusT2SWeights> weights_;
    ConfuciusT2SConfig config_;
    int64_t text_tokens_ = 0;
    int64_t prompt_steps_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> input_ctx_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> output_ctx_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * condition_ = nullptr;
    ggml_tensor * text_ids_ = nullptr;
    ggml_tensor * bos_id_ = nullptr;
    ggml_tensor * logits_ = nullptr;
    std::vector<ggml_tensor *> keys_;
    std::vector<ggml_tensor *> values_;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    ggml_backend_buffer_t input_buffer_ = nullptr;
    ggml_backend_buffer_t output_buffer_ = nullptr;
};

class ConfuciusT2SRuntime::DecodeGraph {
public:
    struct StepOutput {
        std::vector<float> logits;
    };

    struct BatchOutput {
        std::vector<StepOutput> steps;
        double input_upload_ms = 0.0;
        double graph_compute_ms = 0.0;
        double output_read_ms = 0.0;
    };

    DecodeGraph(
        core::ExecutionContext & execution,
        std::shared_ptr<const ConfuciusT2SWeights> weights,
        const ConfuciusT2SConfig & config,
        int64_t prompt_steps,
        int64_t cache_steps,
        int64_t beam_count,
        size_t graph_arena_bytes)
        : execution_(execution),
          weights_(std::move(weights)),
          config_(config),
          cache_steps_(cache_steps),
          beam_count_(beam_count),
          beam_slots_(2 * beam_count) {
        if (weights_ == nullptr || prompt_steps <= 0 || cache_steps_ <= 0 || beam_count_ <= 0) {
            throw std::runtime_error("Confucius4-TTS T2S decode graph requires weights and cache steps");
        }
        const auto build_start = Clock::now();
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize Confucius4-TTS T2S decode graph context");
        }
        ggml_init_params state_params{512ull * 1024ull * 1024ull, nullptr, true};
        state_ctx_.reset(ggml_init(state_params));
        if (state_ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize Confucius4-TTS T2S decode state context");
        }
        core::ModuleBuildContext ctx{ctx_.get(), "confucius4_tts.t2s.decode", execution_.backend_type()};
        core::ModuleBuildContext state_ctx{state_ctx_.get(), "confucius4_tts.t2s.decode.state", execution_.backend_type()};
        token_ids_ = core::make_tensor(state_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({beam_count_})).tensor;
        semantic_positions_ = core::make_tensor(state_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({beam_count_})).tensor;
        cache_slots_ = core::make_tensor(state_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({beam_count_})).tensor;
        parent_rows_ = core::make_tensor(state_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({beam_count_})).tensor;
        semantic_position_lookup_ = core::make_tensor(state_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({cache_steps_, 1})).tensor;
        attention_mask_ = core::make_tensor(
                              state_ctx,
                              GGML_TYPE_F16,
                              core::TensorShape::from_dims({1, 1, 1, cache_steps_}))
                              .tensor;
        ggml_set_input(token_ids_);
        ggml_set_input(semantic_positions_);
        ggml_set_input(cache_slots_);
        ggml_set_input(parent_rows_);
        ggml_set_input(attention_mask_);
        const int64_t head_dim = config_.model_dim / config_.num_heads;
        for (int64_t bank = 0; bank < 2; ++bank) {
            auto & keys = bank_keys_[static_cast<size_t>(bank)];
            auto & values = bank_values_[static_cast<size_t>(bank)];
            keys.reserve(weights_->layers.size());
            values.reserve(weights_->layers.size());
            for (size_t layer = 0; layer < weights_->layers.size(); ++layer) {
                keys.push_back(core::make_tensor(
                    state_ctx,
                    GGML_TYPE_F32,
                    core::TensorShape::from_dims({beam_count_, cache_steps_, config_.num_heads, head_dim})));
                values.push_back(core::make_tensor(
                    state_ctx,
                    GGML_TYPE_F32,
                    core::TensorShape::from_dims({beam_count_, cache_steps_, config_.num_heads, head_dim})));
            }
        }
        build_bank_graph(ctx, 0);
        if (beam_count_ > 1) {
            build_bank_graph(ctx, 1);
        }
        state_buffer_ = ggml_backend_alloc_ctx_tensors(state_ctx_.get(), execution_.backend());
        if (state_buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate Confucius4-TTS T2S decode state buffer");
        }
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend()));
        bool allocated = gallocr_ != nullptr &&
                         ggml_gallocr_reserve(gallocr_, bank_graphs_[0].graph) &&
                         ggml_gallocr_alloc_graph(gallocr_, bank_graphs_[0].graph);
        if (allocated && beam_count_ > 1) {
            allocated = ggml_gallocr_reserve(gallocr_, bank_graphs_[1].graph) &&
                        ggml_gallocr_alloc_graph(gallocr_, bank_graphs_[1].graph);
        }
        if (!allocated) {
            clear_graph();
            throw std::runtime_error("failed to allocate Confucius4-TTS T2S decode graph");
        }
        attention_mask_values_.assign(static_cast<size_t>(cache_steps_), ggml_fp32_to_fp16(-INFINITY));
        ggml_backend_tensor_set(
            attention_mask_,
            attention_mask_values_.data(),
            0,
            attention_mask_values_.size() * sizeof(ggml_fp16_t));
        semantic_position_lookup_values_.assign(static_cast<size_t>(cache_steps_), 0);
        for (int64_t step = 0; step < cache_steps_; ++step) {
            semantic_position_lookup_values_[static_cast<size_t>(step)] =
                static_cast<int32_t>(std::max<int64_t>(0, step - prompt_steps + 1));
        }
        ggml_backend_tensor_set(
            semantic_position_lookup_,
            semantic_position_lookup_values_.data(),
            0,
            semantic_position_lookup_values_.size() * sizeof(int32_t));
        token_values_.assign(static_cast<size_t>(beam_count_), 0);
        position_values_.assign(static_cast<size_t>(beam_count_), 0);
        cache_slot_values_.assign(static_cast<size_t>(beam_count_), 0);
        parent_row_values_.assign(static_cast<size_t>(beam_count_), 0);
        debug::timing_log_scalar("confucius4_tts.t2s.decode.graph.build_ms", debug::elapsed_ms(build_start));
        debug::trace_log_scalar("confucius4_tts.t2s.decode.cache_steps", cache_steps_);
        debug::trace_log_scalar("confucius4_tts.t2s.decode.beam_batch", beam_count_);
    }

    ~DecodeGraph() {
        clear_graph();
    }

    bool can_run(int64_t required_steps, int64_t required_beam_slots) const noexcept {
        return cache_steps_ >= required_steps && beam_slots_ >= required_beam_slots && beam_count_ * 2 == required_beam_slots;
    }

    int64_t cache_steps() const noexcept {
        return cache_steps_;
    }

    void initialize_beam_slot(int64_t slot, const runtime::TransformerKVState & state) {
        if (slot < 0 || slot >= beam_slots_) {
            throw std::runtime_error("Confucius4-TTS T2S beam slot is out of range");
        }
        if (state.layers.size() != weights_->layers.size()) {
            throw std::runtime_error("Confucius4-TTS T2S beam state layer count mismatch");
        }
        for (size_t layer = 0; layer < state.layers.size(); ++layer) {
            const auto & layer_state = state.layers[layer];
            if (layer_state.valid_steps > cache_steps_) {
                throw std::runtime_error("Confucius4-TTS T2S beam state exceeds cache capacity");
            }
            write_beam_cache_prefix(slot, layer, layer_state.valid_steps, true, layer_state.key);
            write_beam_cache_prefix(slot, layer, layer_state.valid_steps, false, layer_state.value);
        }
    }

    runtime::TransformerKVState export_beam_slot(int64_t slot, int64_t valid_steps) {
        if (slot < 0 || slot >= beam_slots_ || valid_steps <= 0 || valid_steps > cache_steps_) {
            throw std::runtime_error("Confucius4-TTS T2S beam export shape mismatch");
        }
        runtime::TransformerKVState state;
        state.current_end = valid_steps;
        state.layers.resize(weights_->layers.size());
        const int64_t head_dim = config_.model_dim / config_.num_heads;
        const size_t layer_values = static_cast<size_t>(valid_steps * config_.num_heads * head_dim);
        for (size_t layer = 0; layer < weights_->layers.size(); ++layer) {
            auto & layer_state = state.layers[layer];
            layer_state.valid_steps = valid_steps;
            layer_state.key.resize(layer_values);
            layer_state.value.resize(layer_values);
            read_beam_cache_prefix(slot, layer, valid_steps, true, layer_state.key);
            read_beam_cache_prefix(slot, layer, valid_steps, false, layer_state.value);
        }
        return state;
    }

    void copy_beam_slot_to(DecodeGraph & target, int64_t source_slot, int64_t target_slot, int64_t valid_steps) const {
        if (source_slot < 0 || source_slot >= beam_slots_ ||
            target_slot < 0 || target_slot >= target.beam_slots_ ||
            valid_steps <= 0 || valid_steps > cache_steps_ || valid_steps > target.cache_steps_ ||
            weights_->layers.size() != target.weights_->layers.size()) {
            throw std::runtime_error("Confucius4-TTS T2S beam cache copy shape mismatch");
        }
        for (size_t layer = 0; layer < weights_->layers.size(); ++layer) {
            copy_beam_cache_prefix_to(target, source_slot, target_slot, layer, valid_steps, true);
            copy_beam_cache_prefix_to(target, source_slot, target_slot, layer, valid_steps, false);
        }
    }

    BatchOutput run_batch_from_beams(
        const std::vector<int64_t> & parent_slots,
        const std::vector<int64_t> & child_slots,
        int64_t valid_steps,
        const std::vector<int32_t> & tokens,
        int32_t semantic_position) {
        const size_t active = parent_slots.size();
        if (active == 0 || child_slots.size() != active || tokens.size() != active) {
            throw std::runtime_error("Confucius4-TTS T2S decode input shape mismatch");
        }
        if (active > static_cast<size_t>(beam_count_) || valid_steps >= cache_steps_) {
            throw std::runtime_error("Confucius4-TTS T2S decode active batch exceeds graph capacity");
        }
        const int64_t child_bank = child_slots.front() / beam_count_;
        if (child_bank < 0 || child_bank > 1 || bank_graphs_[static_cast<size_t>(child_bank)].graph == nullptr) {
            throw std::runtime_error("Confucius4-TTS T2S child beam bank is out of range");
        }
        for (size_t row = 0; row < active; ++row) {
            if (parent_slots[row] < 0 || parent_slots[row] >= beam_slots_ ||
                child_slots[row] < 0 || child_slots[row] >= beam_slots_ ||
                child_slots[row] / beam_count_ != child_bank ||
                child_slots[row] % beam_count_ != static_cast<int64_t>(row)) {
                throw std::runtime_error("Confucius4-TTS T2S beam slot layout mismatch");
            }
            parent_row_values_[row] = static_cast<int32_t>(parent_slots[row] % beam_count_);
            token_values_[row] = tokens[row];
            position_values_[row] = semantic_position;
        }
        for (size_t row = active; row < static_cast<size_t>(beam_count_); ++row) {
            parent_row_values_[row] = static_cast<int32_t>(parent_slots.front() % beam_count_);
            token_values_[row] = tokens.front();
            position_values_[row] = semantic_position;
        }

        auto timing_start = Clock::now();
        if (valid_steps < visible_attention_steps_) {
            std::fill(attention_mask_values_.begin(), attention_mask_values_.end(), ggml_fp32_to_fp16(-INFINITY));
            ggml_backend_tensor_set(
                attention_mask_,
                attention_mask_values_.data(),
                0,
                attention_mask_values_.size() * sizeof(ggml_fp16_t));
            visible_attention_steps_ = -1;
        }
        if (valid_steps > visible_attention_steps_) {
            const int64_t first = visible_attention_steps_ + 1;
            const int64_t count = valid_steps - visible_attention_steps_;
            std::fill(
                attention_mask_values_.begin() + static_cast<std::ptrdiff_t>(first),
                attention_mask_values_.begin() + static_cast<std::ptrdiff_t>(first + count),
                ggml_fp32_to_fp16(0.0F));
            ggml_backend_tensor_set(
                attention_mask_,
                attention_mask_values_.data() + static_cast<std::ptrdiff_t>(first),
                static_cast<size_t>(first) * sizeof(ggml_fp16_t),
                static_cast<size_t>(count) * sizeof(ggml_fp16_t));
            visible_attention_steps_ = valid_steps;
        }
        for (int64_t row = 0; row < beam_count_; ++row) {
            cache_slot_values_[static_cast<size_t>(row)] = static_cast<int32_t>(row * cache_steps_ + valid_steps);
        }
        ggml_backend_tensor_set(token_ids_, token_values_.data(), 0, token_values_.size() * sizeof(int32_t));
        if (beam_count_ > 1) {
            ggml_backend_tensor_set(semantic_positions_, position_values_.data(), 0, position_values_.size() * sizeof(int32_t));
        }
        ggml_backend_tensor_set(cache_slots_, cache_slot_values_.data(), 0, cache_slot_values_.size() * sizeof(int32_t));
        if (beam_count_ > 1) {
            ggml_backend_tensor_set(parent_rows_, parent_row_values_.data(), 0, parent_row_values_.size() * sizeof(int32_t));
        }

        auto & graph = bank_graphs_[static_cast<size_t>(child_bank)];
        core::set_backend_threads(execution_.backend(), execution_.config().threads);
        BatchOutput out;
        out.input_upload_ms = debug::elapsed_ms(timing_start);
        timing_start = Clock::now();
        const ggml_status status = core::compute_backend_graph(execution_.backend(), graph.graph);
        ggml_backend_synchronize(execution_.backend());
        out.graph_compute_ms = debug::elapsed_ms(timing_start);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Confucius4-TTS T2S decode graph compute failed");
        }

        std::vector<float> logits(static_cast<size_t>(beam_count_ * config_.semantic_vocab_size));
        timing_start = Clock::now();
        ggml_backend_tensor_get(graph.logits, logits.data(), 0, logits.size() * sizeof(float));
        out.output_read_ms = debug::elapsed_ms(timing_start);
        out.steps.reserve(active);
        for (size_t row = 0; row < active; ++row) {
            StepOutput step;
            step.logits.assign(
                logits.begin() + static_cast<std::ptrdiff_t>(row * static_cast<size_t>(config_.semantic_vocab_size)),
                logits.begin() + static_cast<std::ptrdiff_t>((row + 1) * static_cast<size_t>(config_.semantic_vocab_size)));
            out.steps.push_back(std::move(step));
        }
        return out;
    }

private:
    struct BankGraph {
        ggml_cgraph * graph = nullptr;
        ggml_tensor * logits = nullptr;
    };

    void clear_graph() {
        for (auto & graph : bank_graphs_) {
            if (graph.graph != nullptr) {
                core::release_backend_graph_resources(execution_.backend(), graph.graph);
                graph.graph = nullptr;
            }
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
            gallocr_ = nullptr;
        }
        if (state_buffer_ != nullptr) {
            ggml_backend_buffer_free(state_buffer_);
            state_buffer_ = nullptr;
        }
    }

    void validate_beam_cache_prefix(int64_t slot, size_t layer, int64_t steps) const {
        if (slot < 0 || slot >= beam_slots_ || layer >= weights_->layers.size() || steps < 0 || steps > cache_steps_) {
            throw std::runtime_error("Confucius4-TTS T2S beam cache prefix view shape mismatch");
        }
    }

    size_t beam_cache_prefix_bytes(int64_t steps) const {
        const int64_t head_dim = config_.model_dim / config_.num_heads;
        return static_cast<size_t>(steps * config_.num_heads * head_dim) * sizeof(float);
    }

    size_t beam_cache_prefix_offset(int64_t slot) const {
        const int64_t head_dim = config_.model_dim / config_.num_heads;
        const int64_t row = slot % beam_count_;
        return static_cast<size_t>(row * cache_steps_ * config_.num_heads * head_dim) * sizeof(float);
    }

    ggml_tensor * beam_cache_tensor(int64_t slot, size_t layer, bool key) const {
        const int64_t bank = slot / beam_count_;
        const auto & tensor = key ? bank_keys_[static_cast<size_t>(bank)][layer] : bank_values_[static_cast<size_t>(bank)][layer];
        return tensor.tensor;
    }

    void write_beam_cache_prefix(
        int64_t slot,
        size_t layer,
        int64_t steps,
        bool key,
        const std::vector<float> & values) const {
        validate_beam_cache_prefix(slot, layer, steps);
        const size_t bytes = beam_cache_prefix_bytes(steps);
        if (values.size() * sizeof(float) != bytes) {
            throw std::runtime_error("Confucius4-TTS T2S beam cache write size mismatch");
        }
        ggml_backend_tensor_set(beam_cache_tensor(slot, layer, key), values.data(), beam_cache_prefix_offset(slot), bytes);
    }

    void read_beam_cache_prefix(
        int64_t slot,
        size_t layer,
        int64_t steps,
        bool key,
        std::vector<float> & values) const {
        validate_beam_cache_prefix(slot, layer, steps);
        const size_t bytes = beam_cache_prefix_bytes(steps);
        if (values.size() * sizeof(float) != bytes) {
            throw std::runtime_error("Confucius4-TTS T2S beam cache read size mismatch");
        }
        ggml_backend_tensor_get(beam_cache_tensor(slot, layer, key), values.data(), beam_cache_prefix_offset(slot), bytes);
    }

    void copy_beam_cache_prefix_to(
        DecodeGraph & target,
        int64_t source_slot,
        int64_t target_slot,
        size_t layer,
        int64_t steps,
        bool key) const {
        validate_beam_cache_prefix(source_slot, layer, steps);
        target.validate_beam_cache_prefix(target_slot, layer, steps);
        const size_t bytes = beam_cache_prefix_bytes(steps);
        cache_copy_scratch_.resize(bytes / sizeof(float));
        ggml_backend_tensor_get(beam_cache_tensor(source_slot, layer, key), cache_copy_scratch_.data(), beam_cache_prefix_offset(source_slot), bytes);
        ggml_backend_tensor_set(target.beam_cache_tensor(target_slot, layer, key), cache_copy_scratch_.data(), target.beam_cache_prefix_offset(target_slot), bytes);
    }

    void build_bank_graph(core::ModuleBuildContext & ctx, int64_t bank) {
        BankGraph graph;
        graph.graph = ggml_new_graph_custom(ctx_.get(), 65536, false);
        const int64_t source_bank = 1 - bank;
        const int64_t head_dim = config_.model_dim / config_.num_heads;
        const int64_t row_elems = cache_steps_ * config_.num_heads * head_dim;
        auto parent_rows = core::wrap_tensor(parent_rows_, core::TensorShape::from_dims({beam_count_}), GGML_TYPE_I32);
        if (beam_count_ > 1) {
            for (size_t layer = 0; layer < weights_->layers.size(); ++layer) {
                auto source_key = core::reshape_tensor(
                    ctx,
                    bank_keys_[static_cast<size_t>(source_bank)][layer],
                    core::TensorShape::from_dims({beam_count_, row_elems}));
                auto source_value = core::reshape_tensor(
                    ctx,
                    bank_values_[static_cast<size_t>(source_bank)][layer],
                    core::TensorShape::from_dims({beam_count_, row_elems}));
                auto child_key = core::reshape_tensor(
                    ctx,
                    bank_keys_[static_cast<size_t>(bank)][layer],
                    core::TensorShape::from_dims({beam_count_, row_elems}));
                auto child_value = core::reshape_tensor(
                    ctx,
                    bank_values_[static_cast<size_t>(bank)][layer],
                    core::TensorShape::from_dims({beam_count_, row_elems}));
                auto gathered_key = core::wrap_tensor(
                    ggml_get_rows(ctx.ggml, source_key.tensor, parent_rows.tensor),
                    child_key.shape,
                    GGML_TYPE_F32);
                auto gathered_value = core::wrap_tensor(
                    ggml_get_rows(ctx.ggml, source_value.tensor, parent_rows.tensor),
                    child_value.shape,
                    GGML_TYPE_F32);
                ggml_build_forward_expand(graph.graph, ggml_cpy(ctx.ggml, gathered_key.tensor, child_key.tensor));
                ggml_build_forward_expand(graph.graph, ggml_cpy(ctx.ggml, gathered_value.tensor, child_value.tensor));
            }
        }
        auto token = core::wrap_tensor(token_ids_, core::TensorShape::from_dims({beam_count_}), GGML_TYPE_I32);
        auto hidden = modules::EmbeddingModule({config_.semantic_vocab_size, config_.model_dim}).build(ctx, token, weights_->semantic_embedding);
        auto cache_slots = core::wrap_tensor(cache_slots_, core::TensorShape::from_dims({beam_count_}), GGML_TYPE_I32);
        auto positions = beam_count_ == 1 ?
                             core::reshape_tensor(
                                 ctx,
                                 core::wrap_tensor(
                                     ggml_get_rows(ctx.ggml, semantic_position_lookup_, cache_slots.tensor),
                                     core::TensorShape::from_dims({beam_count_, 1}),
                                     GGML_TYPE_I32),
                                 core::TensorShape::from_dims({beam_count_})) :
                             core::wrap_tensor(semantic_positions_, core::TensorShape::from_dims({beam_count_}), GGML_TYPE_I32);
        auto pos = modules::EmbeddingModule({config_.max_semantic_seq_lens, config_.model_dim}).build(ctx, positions, weights_->semantic_pos_embedding);
        hidden = modules::AddModule{}.build(ctx, hidden, pos);
        hidden = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, hidden), core::TensorShape::from_dims({beam_count_, 1, config_.model_dim}));
        auto mask = core::wrap_tensor(
            attention_mask_,
            core::TensorShape::from_dims({1, 1, 1, cache_steps_}),
            GGML_TYPE_F16);
        for (size_t layer = 0; layer < weights_->layers.size(); ++layer) {
            auto out = gpt_layer_cached_tail(
                ctx,
                hidden,
                weights_->layers[layer],
                config_,
                bank_keys_[static_cast<size_t>(bank)][layer],
                bank_values_[static_cast<size_t>(bank)][layer],
                cache_slots,
                mask);
            hidden = out.output;
        }
        hidden = modules::LayerNormModule({config_.model_dim, kLayerNormEps, true, true}).build(ctx, hidden, weights_->gpt_final_norm);
        hidden = modules::LayerNormModule({config_.model_dim, kLayerNormEps, true, true}).build(ctx, hidden, weights_->final_norm);
        graph.logits = build_linear(ctx, hidden, config_.model_dim, config_.semantic_vocab_size, weights_->semantic_head, GGML_PREC_F32).tensor;
        ggml_set_output(graph.logits);
        ggml_build_forward_expand(graph.graph, graph.logits);
        bank_graphs_[static_cast<size_t>(bank)] = graph;
    }

    core::ExecutionContext & execution_;
    std::shared_ptr<const ConfuciusT2SWeights> weights_;
    ConfuciusT2SConfig config_;
    int64_t cache_steps_ = 0;
    int64_t beam_count_ = 0;
    int64_t beam_slots_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> state_ctx_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * token_ids_ = nullptr;
    ggml_tensor * semantic_positions_ = nullptr;
    ggml_tensor * cache_slots_ = nullptr;
    ggml_tensor * parent_rows_ = nullptr;
    ggml_tensor * semantic_position_lookup_ = nullptr;
    ggml_tensor * attention_mask_ = nullptr;
    std::array<BankGraph, 2> bank_graphs_;
    std::array<std::vector<core::TensorValue>, 2> bank_keys_;
    std::array<std::vector<core::TensorValue>, 2> bank_values_;
    std::vector<ggml_fp16_t> attention_mask_values_;
    std::vector<int32_t> token_values_;
    std::vector<int32_t> position_values_;
    std::vector<int32_t> cache_slot_values_;
    std::vector<int32_t> parent_row_values_;
    std::vector<int32_t> semantic_position_lookup_values_;
    mutable std::vector<float> cache_copy_scratch_;
    int64_t visible_attention_steps_ = -1;
    ggml_gallocr_t gallocr_ = nullptr;
    ggml_backend_buffer_t state_buffer_ = nullptr;
};

class ConfuciusT2SRuntime::ForwardGraph {
public:
    ForwardGraph(
        core::ExecutionContext & execution,
        std::shared_ptr<const ConfuciusT2SWeights> weights,
        const ConfuciusT2SConfig & config,
        int64_t batch,
        int64_t text_tokens,
        int64_t semantic_steps,
        bool return_sequence,
        size_t graph_arena_bytes)
        : execution_(execution),
          weights_(std::move(weights)),
          config_(config),
          batch_(batch),
          text_tokens_(text_tokens),
          semantic_steps_(semantic_steps),
          return_sequence_(return_sequence) {
        if (weights_ == nullptr || batch_ <= 0 || text_tokens_ <= 0 || semantic_steps_ <= 0) {
            throw std::runtime_error("Confucius4-TTS T2S forward graph requires positive shapes");
        }
        if (return_sequence_ && semantic_steps_ <= 2) {
            throw std::runtime_error("Confucius4-TTS T2S latent forward requires semantic code tokens");
        }
        const auto build_start = Clock::now();
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize Confucius4-TTS T2S forward graph context");
        }
        ggml_init_params input_params{32ull * 1024ull * 1024ull, nullptr, true};
        input_ctx_.reset(ggml_init(input_params));
        if (input_ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize Confucius4-TTS T2S forward input context");
        }
        core::ModuleBuildContext ctx{ctx_.get(), "confucius4_tts.t2s.forward", execution_.backend_type()};
        core::ModuleBuildContext input_ctx{input_ctx_.get(), "confucius4_tts.t2s.forward.inputs", execution_.backend_type()};
        condition_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch_, 1, config_.model_dim})).tensor;
        text_ids_ = core::make_tensor(input_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({text_tokens_})).tensor;
        semantic_ids_ = core::make_tensor(input_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({batch_, semantic_steps_})).tensor;
        ggml_set_input(condition_);
        ggml_set_input(text_ids_);
        ggml_set_input(semantic_ids_);

        auto condition = core::wrap_tensor(condition_, core::TensorShape::from_dims({batch_, 1, config_.model_dim}), GGML_TYPE_F32);
        auto text_ids = core::wrap_tensor(text_ids_, core::TensorShape::from_dims({text_tokens_}), GGML_TYPE_I32);
        auto text = modules::EmbeddingModule({config_.vocab_size, config_.text_embedding_dim}).build(ctx, text_ids, weights_->text_embedding);
        text = build_linear(ctx, text, config_.text_embedding_dim, config_.text_embedding_dim, weights_->text_fc1);
        text = modules::SiluModule{}.build(ctx, text);
        text = build_linear(ctx, text, config_.text_embedding_dim, config_.model_dim, weights_->text_fc2);
        auto text_pos = modules::SliceModule({0, 0, text_tokens_}).build(ctx, weights_->text_pos_embedding);
        text = modules::AddModule{}.build(ctx, text, text_pos);
        text = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, text), core::TensorShape::from_dims({1, text_tokens_, config_.model_dim}));
        text = modules::RepeatModule({core::TensorShape::from_dims({batch_, text_tokens_, config_.model_dim})}).build(ctx, text);

        auto semantic_ids = core::wrap_tensor(semantic_ids_, core::TensorShape::from_dims({batch_, semantic_steps_}), GGML_TYPE_I32);
        auto semantic = modules::EmbeddingModule({config_.semantic_vocab_size, config_.model_dim}).build(ctx, semantic_ids, weights_->semantic_embedding);
        auto semantic_pos = modules::SliceModule({0, 0, semantic_steps_}).build(ctx, weights_->semantic_pos_embedding);
        semantic_pos = core::reshape_tensor(
            ctx,
            core::ensure_backend_addressable_layout(ctx, semantic_pos),
            core::TensorShape::from_dims({1, semantic_steps_, config_.model_dim}));
        semantic_pos = modules::RepeatModule({core::TensorShape::from_dims({batch_, semantic_steps_, config_.model_dim})}).build(ctx, semantic_pos);
        semantic = modules::AddModule{}.build(ctx, semantic, semantic_pos);

        auto hidden = modules::ConcatModule({1}).build(ctx, modules::ConcatModule({1}).build(ctx, condition, text), semantic);
        for (const auto & layer : weights_->layers) {
            hidden = gpt_layer_full(ctx, hidden, layer, config_).output;
        }
        hidden = modules::LayerNormModule({config_.model_dim, kLayerNormEps, true, true}).build(ctx, hidden, weights_->gpt_final_norm);
        const int64_t last_index = hidden.shape.dims[1] - 1;
        auto last = modules::SliceModule({1, last_index, 1}).build(ctx, hidden);
        last = modules::LayerNormModule({config_.model_dim, kLayerNormEps, true, true}).build(ctx, last, weights_->final_norm);
        logits_ = core::ensure_backend_addressable_layout(
                      ctx,
                      build_linear(ctx, last, config_.model_dim, config_.semantic_vocab_size, weights_->semantic_head, GGML_PREC_F32))
                      .tensor;
        ggml_set_output(logits_);
        if (return_sequence_) {
            auto sequence = modules::SliceModule({1, 1 + text_tokens_, semantic_steps_ - 2}).build(ctx, hidden);
            sequence = core::ensure_backend_addressable_layout(ctx, sequence);
            sequence_ = sequence.tensor;
            ggml_set_output(sequence_);
        }
        graph_ = ggml_new_graph_custom(ctx_.get(), 131072, false);
        ggml_build_forward_expand(graph_, logits_);
        if (return_sequence_) {
            ggml_build_forward_expand(graph_, sequence_);
        }
        debug::trace_log_scalar("confucius4_tts.t2s.forward.graph_nodes", ggml_graph_n_nodes(graph_));
        input_buffer_ = ggml_backend_alloc_ctx_tensors(input_ctx_.get(), execution_.backend());
        if (input_buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate Confucius4-TTS T2S forward input buffer");
        }
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend()));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            clear_graph();
            throw std::runtime_error("failed to allocate Confucius4-TTS T2S forward graph");
        }
        debug::timing_log_scalar("confucius4_tts.t2s.forward.graph.build_ms", debug::elapsed_ms(build_start));
    }

    ~ForwardGraph() {
        clear_graph();
    }

    bool matches(int64_t batch, int64_t text_tokens, int64_t semantic_steps, bool return_sequence) const noexcept {
        return batch_ == batch &&
               text_tokens_ == text_tokens &&
               semantic_steps_ == semantic_steps &&
               return_sequence_ == return_sequence;
    }

    struct Output {
        std::vector<float> logits;
        std::vector<float> sequence;
    };

    Output run(
        const std::vector<float> & condition,
        const std::vector<int32_t> & text_tokens,
        const std::vector<int32_t> & semantic_ids) {
        if (static_cast<int64_t>(condition.size()) != batch_ * config_.model_dim ||
            static_cast<int64_t>(text_tokens.size()) != text_tokens_ ||
            static_cast<int64_t>(semantic_ids.size()) != batch_ * semantic_steps_) {
            throw std::runtime_error("Confucius4-TTS T2S forward input shape mismatch");
        }
        auto start = Clock::now();
        ggml_backend_tensor_set(condition_, condition.data(), 0, condition.size() * sizeof(float));
        ggml_backend_tensor_set(text_ids_, text_tokens.data(), 0, text_tokens.size() * sizeof(int32_t));
        ggml_backend_tensor_set(semantic_ids_, semantic_ids.data(), 0, semantic_ids.size() * sizeof(int32_t));
        debug::timing_log_scalar("confucius4_tts.t2s.forward.input_upload_ms", debug::elapsed_ms(start));
        core::set_backend_threads(execution_.backend(), execution_.config().threads);
        start = Clock::now();
        const ggml_status status = core::compute_backend_graph(execution_.backend(), graph_);
        ggml_backend_synchronize(execution_.backend());
        debug::timing_log_scalar("confucius4_tts.t2s.forward.graph.compute_ms", debug::elapsed_ms(start));
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Confucius4-TTS T2S forward graph compute failed");
        }
        Output output;
        output.logits.resize(static_cast<size_t>(batch_ * config_.semantic_vocab_size));
        ggml_backend_tensor_get(logits_, output.logits.data(), 0, output.logits.size() * sizeof(float));
        if (return_sequence_) {
            output.sequence.resize(static_cast<size_t>(batch_ * (semantic_steps_ - 2) * config_.model_dim));
            ggml_backend_tensor_get(sequence_, output.sequence.data(), 0, output.sequence.size() * sizeof(float));
        }
        return output;
    }

private:
    void clear_graph() {
        if (graph_ != nullptr) {
            core::release_backend_graph_resources(execution_.backend(), graph_);
            graph_ = nullptr;
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
            gallocr_ = nullptr;
        }
        if (input_buffer_ != nullptr) {
            ggml_backend_buffer_free(input_buffer_);
            input_buffer_ = nullptr;
        }
    }

    core::ExecutionContext & execution_;
    std::shared_ptr<const ConfuciusT2SWeights> weights_;
    ConfuciusT2SConfig config_;
    int64_t batch_ = 0;
    int64_t text_tokens_ = 0;
    int64_t semantic_steps_ = 0;
    bool return_sequence_ = false;
    std::unique_ptr<ggml_context, GgmlContextDeleter> input_ctx_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * condition_ = nullptr;
    ggml_tensor * text_ids_ = nullptr;
    ggml_tensor * semantic_ids_ = nullptr;
    ggml_tensor * logits_ = nullptr;
    ggml_tensor * sequence_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    ggml_backend_buffer_t input_buffer_ = nullptr;
};

ConfuciusT2SRuntime::ConfuciusT2SRuntime(
    std::shared_ptr<const ConfuciusAssets> assets,
    core::ExecutionContext & execution,
    size_t graph_arena_bytes,
    size_t weight_context_bytes,
    assets::TensorStorageType matmul_storage_type,
    assets::TensorStorageType conv_storage_type)
    : assets_(std::move(assets)),
      execution_(&execution),
      graph_arena_bytes_(graph_arena_bytes) {
    if (assets_ == nullptr || graph_arena_bytes_ == 0) {
        throw std::runtime_error("Confucius4-TTS T2S runtime requires assets and graph arena");
    }
    weights_ = load_confucius_t2s_weights(
        *assets_,
        execution.backend(),
        execution.backend_type(),
        matmul_storage_type,
        conv_storage_type,
        weight_context_bytes);
}

ConfuciusT2SRuntime::~ConfuciusT2SRuntime() = default;

void ConfuciusT2SRuntime::prepare_generation(int64_t text_tokens, int64_t max_semantic_tokens, int64_t num_beams) {
    if (execution_ == nullptr || weights_ == nullptr) {
        throw std::runtime_error("Confucius4-TTS T2S runtime is not initialized");
    }
    if (text_tokens <= 0 || max_semantic_tokens <= 0 || num_beams <= 0) {
        throw std::runtime_error("Confucius4-TTS T2S generation prepare requires positive shapes");
    }
    const auto prepare_start = Clock::now();
    const bool rebuild_prefill = prefill_graph_ == nullptr || !prefill_graph_->matches(text_tokens);
    if (prefill_graph_ == nullptr || !prefill_graph_->matches(text_tokens)) {
        prefill_graph_.reset();
        prefill_graph_ = std::make_unique<PrefillGraph>(
            *execution_,
            weights_,
            assets_->config.t2s,
            text_tokens,
            graph_arena_bytes_);
    }
    const int64_t required_cache_steps = initial_decode_cache_steps(prefill_graph_->prompt_steps(), max_semantic_tokens);
    const int64_t required_beam_slots = 2 * std::max<int64_t>(1, num_beams);
    const bool oversized =
        decode_graph_ != nullptr &&
        decode_graph_->cache_steps() > required_cache_steps * 2;
    const bool rebuild_decode =
        decode_graph_ == nullptr || oversized || !decode_graph_->can_run(required_cache_steps, required_beam_slots);
    if (decode_graph_ == nullptr || oversized || !decode_graph_->can_run(required_cache_steps, required_beam_slots)) {
        decode_graph_.reset();
        decode_graph_ = std::make_unique<DecodeGraph>(
            *execution_,
            weights_,
            assets_->config.t2s,
            prefill_graph_->prompt_steps(),
            required_cache_steps,
            std::max<int64_t>(1, num_beams),
            graph_arena_bytes_);
    }
    debug::timing_log_scalar("confucius4_tts.t2s.prepare_generation_ms", debug::elapsed_ms(prepare_start));
    debug::timing_log_scalar("confucius4_tts.t2s.prefill.graph.rebuilt", rebuild_prefill);
    debug::timing_log_scalar("confucius4_tts.t2s.prefill.graph.reused", !rebuild_prefill);
    debug::timing_log_scalar("confucius4_tts.t2s.decode.graph.rebuilt", rebuild_decode);
    debug::timing_log_scalar("confucius4_tts.t2s.decode.graph.reused", !rebuild_decode);
}

ConfuciusT2SSemanticGeneration ConfuciusT2SRuntime::generate(const ConfuciusT2SGenerationRequest & request) {
    if (execution_ == nullptr || weights_ == nullptr) {
        throw std::runtime_error("Confucius4-TTS T2S runtime is not initialized");
    }
    if (request.text_tokens.empty() ||
        request.semantic_condition.frames <= 0 ||
        request.semantic_condition.dims != assets_->config.t2s.speaker_embedding_dim) {
        throw std::runtime_error("Confucius4-TTS T2S generation input shape mismatch");
    }
    if (speaker_graph_ == nullptr || !speaker_graph_->matches(request.semantic_condition.frames)) {
        speaker_graph_.reset();
        speaker_graph_ = std::make_unique<SpeakerConditionGraph>(
            *execution_,
            weights_,
            assets_->config.t2s,
            request.semantic_condition.frames,
            graph_arena_bytes_);
    }
    auto timing_start = Clock::now();
    const auto condition = speaker_graph_->run(request.semantic_condition);
    debug::timing_log_scalar("confucius4_tts.t2s.speaker.total_ms", debug::elapsed_ms(timing_start));
    const int beam_count = std::max(1, request.options.num_beams);
    const int64_t text_tokens = static_cast<int64_t>(request.text_tokens.size());
    const int64_t max_new = std::max<int64_t>(1, request.options.max_tokens - text_tokens - 2);
    timing_start = Clock::now();
    prepare_generation(text_tokens, max_new, beam_count);
    debug::timing_log_scalar("confucius4_tts.t2s.prepare_generation.total_ms", debug::elapsed_ms(timing_start));
    timing_start = Clock::now();
    auto prefill = prefill_graph_->run(condition, request.text_tokens);
    debug::timing_log_scalar("confucius4_tts.t2s.prefill.total_ms", debug::elapsed_ms(timing_start));

    struct Beam {
        std::vector<int32_t> codes;
        std::vector<float> logits;
        int64_t slot = 0;
        int64_t valid_steps = 0;
        float score = 0.0F;
    };
    struct FinishedBeam {
        Beam beam;
        float score = -1.0e9F;
        bool finished = false;
    };
    const int64_t prefill_valid_steps = prefill.kv_state.layers.empty() ? 0 : prefill.kv_state.layers.front().valid_steps;
    std::vector<Beam> beams;
    beams.reserve(static_cast<size_t>(beam_count));
    for (int beam = 0; beam < beam_count; ++beam) {
        decode_graph_->initialize_beam_slot(beam, prefill.kv_state);
        Beam initial;
        initial.logits = prefill.logits;
        initial.slot = beam;
        initial.valid_steps = prefill_valid_steps;
        initial.score = beam == 0 ? 0.0F : -1.0e9F;
        beams.push_back(std::move(initial));
    }
    std::vector<FinishedBeam> finished(static_cast<size_t>(beam_count));
    const auto update_finished = [&](Beam beam, float score) {
        const float normalized_score = score / static_cast<float>(std::max<size_t>(1, beam.codes.size() + 1));
        auto slot = std::min_element(finished.begin(), finished.end(), [](const FinishedBeam & lhs, const FinishedBeam & rhs) {
            return lhs.score < rhs.score;
        });
        if (slot != finished.end() && normalized_score > slot->score) {
            beam.score = score;
            slot->beam = std::move(beam);
            slot->score = normalized_score;
            slot->finished = true;
        }
    };
    const auto all_finished = [&]() {
        return std::all_of(finished.begin(), finished.end(), [](const FinishedBeam & beam) {
            return beam.finished;
        });
    };
    const auto worst_finished_score = [&]() {
        if (!all_finished()) {
            return -1.0e9F;
        }
        const auto slot = std::min_element(finished.begin(), finished.end(), [](const FinishedBeam & lhs, const FinishedBeam & rhs) {
            return lhs.score < rhs.score;
        });
        return slot == finished.end() ? -1.0e9F : slot->score;
    };
    SamplerWorkspace sampler_workspace;
    const auto sampling_policy = sampling::resolve_torch_cuda_sampling_policy(
        execution_->backend_type(),
        execution_->config().device,
        "confucius4_tts.t2s.cuda_sampling_policy",
        "Confucius4-TTS",
        sampling::TorchCudaSamplingPolicyFailureMode::FallbackToDefault);
    uint64_t sample_call_index = request.rng_offset_blocks;
    uint64_t rng_offset_blocks = request.rng_offset_blocks;
    double sampling_ms = 0.0;
    double decode_input_upload_ms = 0.0;
    double decode_graph_compute_ms = 0.0;
    double decode_output_read_ms = 0.0;
    double decode_grow_copy_ms = 0.0;
    int64_t decode_grow_count = 0;
    const int32_t bos = static_cast<int32_t>(assets_->config.t2s.start_semantic_token);
    const int32_t eos = static_cast<int32_t>(assets_->config.t2s.stop_semantic_token);
    int active_bank = 0;
    double decode_run_ms = 0.0;
    std::vector<int64_t> parent_slots;
    std::vector<int64_t> child_slots;
    std::vector<int32_t> next_tokens;
    parent_slots.reserve(static_cast<size_t>(beam_count));
    child_slots.reserve(static_cast<size_t>(beam_count));
    next_tokens.reserve(static_cast<size_t>(beam_count));
    const auto grow_decode_graph = [&](int64_t required_cache_steps) {
        if (decode_graph_ != nullptr &&
            decode_graph_->can_run(required_cache_steps, 2 * static_cast<int64_t>(beam_count))) {
            return;
        }
        const int64_t new_cache_steps = grown_decode_cache_steps(
            prefill_graph_->prompt_steps(),
            decode_graph_->cache_steps(),
            required_cache_steps,
            max_new);
        auto next_graph = std::make_unique<DecodeGraph>(
            *execution_,
            weights_,
            assets_->config.t2s,
            prefill_graph_->prompt_steps(),
            new_cache_steps,
            beam_count,
            graph_arena_bytes_);
        const auto copy_start = Clock::now();
        for (const auto & beam : beams) {
            decode_graph_->copy_beam_slot_to(*next_graph, beam.slot, beam.slot, beam.valid_steps);
        }
        decode_grow_copy_ms += debug::elapsed_ms(copy_start);
        decode_grow_count += 1;
        decode_graph_ = std::move(next_graph);
    };
    for (int64_t step = 0; step < max_new && !beams.empty(); ++step) {
        const auto sampling_start = Clock::now();
        std::vector<SampleScore> sample_scores;
        for (size_t beam = 0; beam < beams.size(); ++beam) {
            log_probs(beams[beam].logits, beams[beam].codes, request.options, bos, sampler_workspace);
            const size_t beam_offset = beam * static_cast<size_t>(assets_->config.t2s.semantic_vocab_size);
            for (const size_t token : sampler_workspace.finite_score_indices) {
                sample_scores.push_back({beam_offset + token, beams[beam].score + sampler_workspace.scores[token]});
            }
        }
        const size_t keep = std::min<size_t>(static_cast<size_t>(2 * beam_count), sample_scores.size());
        rng_offset_blocks += sampling::torch_cuda_tensor_iterator_offset_blocks(
            static_cast<uint64_t>(std::max<size_t>(1, beams.size() * static_cast<size_t>(assets_->config.t2s.semantic_vocab_size))),
            sampling_policy);
        sample_indices(
            sample_scores,
            beams.size() * static_cast<size_t>(assets_->config.t2s.semantic_vocab_size),
            keep,
            request.options.seed,
            sample_call_index++,
            sampling_policy,
            sampler_workspace.ranked_samples,
            sampler_workspace.selected_scores);
        std::vector<Beam> next;
        next.reserve(static_cast<size_t>(beam_count));
        const int next_bank = beam_count == 1 ? active_bank : 1 - active_bank;
        parent_slots.clear();
        child_slots.clear();
        next_tokens.clear();
        auto & running_candidates = sampler_workspace.running_candidates;
        running_candidates.clear();
        running_candidates.reserve(sampler_workspace.selected_scores.size());
        for (size_t rank = 0; rank < sampler_workspace.selected_scores.size(); ++rank) {
            const size_t selected = sampler_workspace.selected_scores[rank];
            const auto & score = sample_scores[selected];
            const size_t parent = score.flat_index / static_cast<size_t>(assets_->config.t2s.semantic_vocab_size);
            const int32_t token = static_cast<int32_t>(score.flat_index % static_cast<size_t>(assets_->config.t2s.semantic_vocab_size));
            if (token == eos) {
                if (rank < static_cast<size_t>(beam_count)) {
                    Beam beam = beams[parent];
                    update_finished(std::move(beam), score.score);
                }
            } else {
                running_candidates.push_back({rank, parent, token, score.score});
            }
        }
        std::sort(running_candidates.begin(), running_candidates.end(), [](const RunningCandidate & lhs, const RunningCandidate & rhs) {
            const auto lhs_key = std::make_pair(lhs.score, lhs.topk_rank);
            const auto rhs_key = std::make_pair(rhs.score, rhs.topk_rank);
            return lhs_key.first == rhs_key.first ? lhs_key.second > rhs_key.second : lhs_key.first > rhs_key.first;
        });
        const size_t running_count = std::min(static_cast<size_t>(beam_count), running_candidates.size());
        for (size_t i = 0; i < running_count; ++i) {
            const auto & candidate = running_candidates[i];
            Beam beam = beams[candidate.parent];
            beam.score = candidate.score;
            beam.codes.push_back(candidate.token);
            beam.slot = beam_count == 1 ?
                            beams[candidate.parent].slot :
                            static_cast<int64_t>(next_bank * beam_count + static_cast<int>(next.size()));
            beam.valid_steps += 1;
            parent_slots.push_back(beams[candidate.parent].slot);
            child_slots.push_back(beam.slot);
            next_tokens.push_back(candidate.token);
            next.push_back(std::move(beam));
        }
        sampling_ms += debug::elapsed_ms(sampling_start);
        if (!next.empty()) {
            const auto decode_start = Clock::now();
            const int64_t parent_valid_steps = next.front().valid_steps - 1;
            grow_decode_graph(parent_valid_steps + 1);
            const int32_t semantic_position = static_cast<int32_t>(parent_valid_steps - (text_tokens + 1));
            auto decoded = decode_graph_->run_batch_from_beams(
                parent_slots,
                child_slots,
                parent_valid_steps,
                next_tokens,
                semantic_position);
            decode_input_upload_ms += decoded.input_upload_ms;
            decode_graph_compute_ms += decoded.graph_compute_ms;
            decode_output_read_ms += decoded.output_read_ms;
            decode_run_ms += debug::elapsed_ms(decode_start);
            if (decoded.steps.size() != next.size()) {
                throw std::runtime_error("Confucius4-TTS T2S decode output size mismatch");
            }
            for (size_t beam = 0; beam < next.size(); ++beam) {
                next[beam].logits = std::move(decoded.steps[beam].logits);
            }
        }
        beams.swap(next);
        active_bank = next_bank;
        if (beams.empty()) {
            break;
        }
        const float generated_len = static_cast<float>(std::max<int64_t>(1, step + 1));
        const float best_running_score = beams.front().score / generated_len;
        if (all_finished() || best_running_score <= worst_finished_score()) {
            break;
        }
    }
    debug::timing_log_scalar("confucius4_tts.t2s.sampling_ms", sampling_ms);
    debug::timing_log_scalar("confucius4_tts.t2s.decode.run_ms", decode_run_ms);
    debug::timing_log_scalar("confucius4_tts.t2s.decode.input_upload_ms", decode_input_upload_ms);
    debug::timing_log_scalar("confucius4_tts.t2s.decode.graph.compute_ms", decode_graph_compute_ms);
    debug::timing_log_scalar("confucius4_tts.t2s.decode.output_read_ms", decode_output_read_ms);
    debug::timing_log_scalar("confucius4_tts.t2s.decode.grow.copy_ms", decode_grow_copy_ms);
    debug::timing_log_scalar("confucius4_tts.t2s.decode.grow.count", decode_grow_count);
    if (!all_finished()) {
        for (auto & beam : beams) {
            const float normalized_score = beam.score / static_cast<float>(std::max<size_t>(1, beam.codes.size()));
            auto slot = std::min_element(finished.begin(), finished.end(), [](const FinishedBeam & lhs, const FinishedBeam & rhs) {
                return lhs.score < rhs.score;
            });
            if (slot != finished.end() && normalized_score > slot->score) {
                slot->beam = std::move(beam);
                slot->score = normalized_score;
                slot->finished = true;
            }
        }
    }
    if (!std::any_of(finished.begin(), finished.end(), [](const FinishedBeam & beam) { return beam.finished; })) {
        throw std::runtime_error("Confucius4-TTS T2S generation produced no semantic tokens");
    }
    const auto best = std::max_element(finished.begin(), finished.end(), [](const FinishedBeam & lhs, const FinishedBeam & rhs) {
        return lhs.score < rhs.score;
    });
    std::vector<int32_t> semantic_ids;
    semantic_ids.reserve(best->beam.codes.size() + 2);
    semantic_ids.push_back(bos);
    semantic_ids.insert(semantic_ids.end(), best->beam.codes.begin(), best->beam.codes.end());
    semantic_ids.push_back(eos);
    const bool return_sequence = true;
    timing_start = Clock::now();
    const bool rebuild_forward =
        forward_graph_ == nullptr ||
        !forward_graph_->matches(1, text_tokens, static_cast<int64_t>(semantic_ids.size()), return_sequence);
    if (rebuild_forward) {
        forward_graph_.reset();
        forward_graph_ = std::make_unique<ForwardGraph>(
            *execution_,
            weights_,
            assets_->config.t2s,
            1,
            text_tokens,
            static_cast<int64_t>(semantic_ids.size()),
            return_sequence,
            graph_arena_bytes_);
    }
    debug::timing_log_scalar("confucius4_tts.t2s.forward.prepare_ms", debug::elapsed_ms(timing_start));
    debug::timing_log_scalar("confucius4_tts.t2s.forward.graph.rebuilt", rebuild_forward);
    debug::timing_log_scalar("confucius4_tts.t2s.forward.graph.reused", !rebuild_forward);
    timing_start = Clock::now();
    auto latent_output = forward_graph_->run(condition, request.text_tokens, semantic_ids);
    debug::timing_log_scalar("confucius4_tts.t2s.forward.total_ms", debug::elapsed_ms(timing_start));
    ConfuciusT2SSemanticGeneration out;
    out.semantic_codes = best->beam.codes;
    out.latent = std::move(latent_output.sequence);
    out.frames = static_cast<int64_t>(out.semantic_codes.size());
    out.dims = assets_->config.t2s.model_dim;
    out.rng_offset_blocks = rng_offset_blocks;
    debug::trace_log_scalar("confucius4_tts.t2s.generated_code_count", static_cast<int64_t>(out.semantic_codes.size()));
    debug::trace_log_i32(
        "confucius4_tts.t2s.semantic_codes",
        {static_cast<int64_t>(out.semantic_codes.size())},
        out.semantic_codes);
    return out;
}

void ConfuciusT2SRuntime::release_graphs() {
    speaker_graph_.reset();
    prefill_graph_.reset();
    decode_graph_.reset();
    forward_graph_.reset();
}

}  // namespace engine::models::confucius4_tts
