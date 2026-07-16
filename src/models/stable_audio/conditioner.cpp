#include "engine/models/stable_audio/conditioner.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace engine::models::stable_audio {
namespace {

constexpr int64_t kSecondsFourierDim = 256;
constexpr float kSecondsMinFreq = 0.5F;
constexpr float kSecondsMaxFreq = 10000.0F;
constexpr float kPi = 3.14159265358979323846F;

std::shared_ptr<const StableAudioAssets> require_assets(std::shared_ptr<const StableAudioAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("Stable Audio conditioner requires assets");
    }
    return assets;
}

}  // namespace

StableAudioConditionerInputs::StableAudioConditionerInputs(std::shared_ptr<const StableAudioAssets> assets)
    : assets_(require_assets(std::move(assets))),
      t5_pieces_(engine::tokenizers::load_sentencepiece_model(assets_->t5_tokenizer_model_path)) {}

StableAudioConditioningBatch StableAudioConditionerInputs::build(const StableAudioRequest & request) const {
    StableAudioConditioningBatch out;
    out.prompt_tokens = tokenize_prompts(request.prompts);
    if (!request.negative_prompts.empty()) {
        out.negative_prompt_tokens = tokenize_prompts(request.negative_prompts);
    }
    out.normalized_seconds = normalized_seconds(request.durations_seconds);
    out.seconds_fourier_features = seconds_fourier_features(out.normalized_seconds);
    return out;
}

StableAudioTokenBatch StableAudioConditionerInputs::tokenize_prompts(const std::vector<std::string> & prompts) const {
    const int64_t batch = static_cast<int64_t>(prompts.size());
    const int64_t tokens = assets_->config.prompt_max_length;
    if (batch <= 0 || tokens <= 0) {
        throw std::runtime_error("Stable Audio tokenizer requires positive batch and token length");
    }
    StableAudioTokenBatch out;
    out.batch = batch;
    out.tokens = tokens;
    out.input_ids.assign(static_cast<size_t>(batch * tokens), static_cast<int32_t>(assets_->config.t5_pad_token_id));
    out.attention_mask.assign(static_cast<size_t>(batch * tokens), 0);
    for (int64_t b = 0; b < batch; ++b) {
        auto ids = engine::tokenizers::tokenize_sentencepiece(t5_pieces_, prompts[static_cast<size_t>(b)]);
        if (static_cast<int64_t>(ids.size()) > tokens) {
            ids.resize(static_cast<size_t>(tokens));
        }
        for (size_t i = 0; i < ids.size(); ++i) {
            out.input_ids[static_cast<size_t>(b * tokens) + i] = ids[i];
            out.attention_mask[static_cast<size_t>(b * tokens) + i] = 1;
        }
    }
    return out;
}

std::vector<float> StableAudioConditionerInputs::normalized_seconds(const std::vector<float> & durations) const {
    std::vector<float> out;
    out.reserve(durations.size());
    const float min_value = assets_->config.seconds_min;
    const float max_value = assets_->config.seconds_max;
    if (!(max_value > min_value)) {
        throw std::runtime_error("Stable Audio seconds_total conditioner range is invalid");
    }
    for (const float duration : durations) {
        const float clamped = std::min(std::max(duration, min_value), max_value);
        out.push_back((clamped - min_value) / (max_value - min_value));
    }
    return out;
}

std::vector<float> StableAudioConditionerInputs::seconds_fourier_features(const std::vector<float> & normalized) const {
    const int64_t half = kSecondsFourierDim / 2;
    std::vector<float> out(static_cast<size_t>(static_cast<int64_t>(normalized.size()) * kSecondsFourierDim), 0.0F);
    for (size_t b = 0; b < normalized.size(); ++b) {
        for (int64_t i = 0; i < half; ++i) {
            const float ramp = static_cast<float>(i) / static_cast<float>(half - 1);
            const float freq = std::exp(ramp * (std::log(kSecondsMaxFreq) - std::log(kSecondsMinFreq)) + std::log(kSecondsMinFreq));
            const float arg = normalized[b] * freq * 2.0F * kPi;
            out[b * static_cast<size_t>(kSecondsFourierDim) + static_cast<size_t>(i)] = std::cos(arg);
            out[b * static_cast<size_t>(kSecondsFourierDim) + static_cast<size_t>(half + i)] = std::sin(arg);
        }
    }
    return out;
}

}  // namespace engine::models::stable_audio
