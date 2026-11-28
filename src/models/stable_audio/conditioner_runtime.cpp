#include "engine/models/stable_audio/conditioner_runtime.h"

#include "engine/framework/debug/profiler.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace engine::models::stable_audio {
namespace {

constexpr int64_t kSecondsFourierDim = 256;

std::shared_ptr<const StableAudioAssets> require_assets(std::shared_ptr<const StableAudioAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("Stable Audio conditioner runtime requires assets");
    }
    return assets;
}

void validate_batch_seconds(const StableAudioTokenBatch & tokens, const std::vector<float> & normalized_seconds) {
    if (tokens.batch <= 0) {
        throw std::runtime_error("Stable Audio conditioner runtime requires a positive batch");
    }
    if (static_cast<int64_t>(normalized_seconds.size()) != tokens.batch) {
        throw std::runtime_error("Stable Audio conditioner seconds batch does not match prompt batch");
    }
}

}  // namespace

StableAudioConditionerRuntime::StableAudioConditionerRuntime(
    core::ExecutionContext & execution,
    std::shared_ptr<const StableAudioAssets> assets,
    assets::TensorStorageType weight_storage_type)
    : assets_(require_assets(std::move(assets))),
      t5_(execution, assets_, weight_storage_type),
      prompt_padding_embedding_(assets_->model_weights->require_f32(
          "conditioner.conditioners.prompt.padding_embedding",
          {assets_->config.cond_dim})),
      seconds_projection_weight_(assets_->model_weights->require_f32(
          "conditioner.conditioners.seconds_total.embedder.embedding.1.weight",
          {assets_->config.cond_dim, kSecondsFourierDim})),
      seconds_projection_bias_(assets_->model_weights->require_f32(
          "conditioner.conditioners.seconds_total.embedder.embedding.1.bias",
          {assets_->config.cond_dim})) {
    if (assets_->config.cond_dim != assets_->config.t5_hidden_size) {
        throw std::runtime_error("Stable Audio conditioner expects T5 hidden size to match conditioning dim");
    }
}

StableAudioConditionerRuntime::~StableAudioConditionerRuntime() = default;

void StableAudioConditionerRuntime::prepare() const {
    t5_.prepare();
}

StableAudioConditioningInputs StableAudioConditionerRuntime::encode(const StableAudioConditioningBatch & inputs) const {
    StableAudioConditioningInputs out;
    out.positive = encode_one(inputs.prompt_tokens, inputs.normalized_seconds, inputs.seconds_fourier_features);
    if (inputs.negative_prompt_tokens.batch > 0) {
        out.negative = encode_one(
            inputs.negative_prompt_tokens,
            inputs.normalized_seconds,
            inputs.seconds_fourier_features);
        out.has_negative = true;
    }
    return out;
}

void StableAudioConditionerRuntime::release_runtime_graphs() const {
    t5_.release_runtime_graphs();
}

StableAudioConditionedInput StableAudioConditionerRuntime::encode_one(
    const StableAudioTokenBatch & tokens,
    const std::vector<float> & normalized_seconds,
    const std::vector<float> & seconds_fourier_features) const {
    (void) normalized_seconds;
    validate_batch_seconds(tokens, normalized_seconds);
    const auto & config = assets_->config;
    if (tokens.tokens != config.prompt_max_length) {
        throw std::runtime_error("Stable Audio conditioner token length mismatch");
    }
    if (static_cast<int64_t>(seconds_fourier_features.size()) != tokens.batch * kSecondsFourierDim) {
        throw std::runtime_error("Stable Audio conditioner seconds Fourier shape mismatch");
    }

    const StableAudioT5Encoding prompt = t5_.encode(tokens);
    if (prompt.batch != tokens.batch || prompt.tokens != tokens.tokens || prompt.hidden_size != config.cond_dim) {
        throw std::runtime_error("Stable Audio T5 output shape does not match conditioner");
    }
    for (const float value : prompt.values) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("Stable Audio T5 conditioner produced non-finite output");
        }
    }

    StableAudioConditionedInput out;
    out.batch = tokens.batch;
    out.cross_tokens = tokens.tokens + 1;
    out.cond_dim = config.cond_dim;
    out.cross_attention.assign(static_cast<size_t>(out.batch * out.cross_tokens * out.cond_dim), 0.0F);
    out.global.assign(static_cast<size_t>(out.batch * out.cond_dim), 0.0F);
    out.cross_attention_mask.assign(static_cast<size_t>(out.batch * out.cross_tokens), 0);

    for (int64_t b = 0; b < out.batch; ++b) {
        for (int64_t t = 0; t < tokens.tokens; ++t) {
            const bool is_valid = tokens.attention_mask[static_cast<size_t>(b * tokens.tokens + t)] != 0;
            out.cross_attention_mask[static_cast<size_t>(b * out.cross_tokens + t)] = is_valid ? 1 : 0;
            const float * src = is_valid
                ? prompt.values.data() + static_cast<std::ptrdiff_t>((b * tokens.tokens + t) * out.cond_dim)
                : prompt_padding_embedding_.data();
            float * dst = out.cross_attention.data() + static_cast<std::ptrdiff_t>((b * out.cross_tokens + t) * out.cond_dim);
            std::copy_n(src, static_cast<size_t>(out.cond_dim), dst);
        }

        float * seconds_embedding = out.cross_attention.data() +
            static_cast<std::ptrdiff_t>((b * out.cross_tokens + tokens.tokens) * out.cond_dim);
        out.cross_attention_mask[static_cast<size_t>(b * out.cross_tokens + tokens.tokens)] = 1;
        for (int64_t o = 0; o < out.cond_dim; ++o) {
            float sum = seconds_projection_bias_[static_cast<size_t>(o)];
            for (int64_t i = 0; i < kSecondsFourierDim; ++i) {
                sum += seconds_projection_weight_[static_cast<size_t>(o * kSecondsFourierDim + i)] *
                       seconds_fourier_features[static_cast<size_t>(b * kSecondsFourierDim + i)];
            }
            seconds_embedding[o] = sum;
            out.global[static_cast<size_t>(b * out.cond_dim + o)] = sum;
            if (!std::isfinite(sum)) {
                throw std::runtime_error("Stable Audio seconds conditioner produced non-finite output");
            }
        }
    }
    return out;
}

}  // namespace engine::models::stable_audio
