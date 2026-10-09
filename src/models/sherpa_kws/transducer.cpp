#include "engine/models/sherpa_kws/transducer.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace engine::models::sherpa_kws {
namespace {

constexpr int64_t kInputsPerGroup = 4;

}  // namespace

TransducerScorer::TransducerScorer(
    std::shared_ptr<const SherpaKwsAssets> assets)
    : assets_(std::move(assets)) {
    if (!assets_) {
        throw std::runtime_error("sherpa KWS transducer scorer requires assets");
    }
    const auto & source = *assets_->weights;
    const int64_t vocabulary = assets_->config.vocab_size;
    const int64_t hidden = assets_->config.decoder_dim;
    embedding_ = source.require_f32(
        "decoder.decoder.embedding.weight", {vocabulary, hidden});
    convolution_ = source.require_f32(
        "decoder.decoder.conv.weight", {hidden, kInputsPerGroup, 2});
    decoder_projection_ = source.require_f32(
        "decoder.decoder_proj.weight", {hidden, hidden});
    decoder_bias_ = source.require_f32(
        "decoder.decoder_proj.bias", {hidden});
    joiner_projection_ = source.require_f32(
        "joiner.output_linear.weight", {vocabulary, hidden});
    joiner_bias_ = source.require_f32(
        "joiner.output_linear.bias", {vocabulary});
    activated_.resize(static_cast<size_t>(hidden));
}

std::vector<float> TransducerScorer::predictor(
    const std::array<int32_t, 2> & context) const {
    const int64_t hidden = assets_->config.decoder_dim;
    const int64_t vocabulary = assets_->config.vocab_size;
    const int64_t groups = hidden / kInputsPerGroup;
    std::vector<float> embedded(static_cast<size_t>(hidden * 2), 0.0F);
    for (int64_t position = 0; position < 2; ++position) {
        const int32_t token = context[static_cast<size_t>(position)];
        if (token < 0) continue;
        if (token >= vocabulary) {
            throw std::runtime_error("sherpa KWS predictor token is outside the vocabulary");
        }
        std::copy_n(
            embedding_.data() + static_cast<int64_t>(token) * hidden,
            hidden,
            embedded.data() + position * hidden);
    }

    std::vector<float> convolved(static_cast<size_t>(hidden), 0.0F);
    for (int64_t output = 0; output < hidden; ++output) {
        const int64_t input_start = (output / (hidden / groups)) * kInputsPerGroup;
        float value = 0.0F;
        for (int64_t input = 0; input < kInputsPerGroup; ++input) {
            for (int64_t position = 0; position < 2; ++position) {
                value += embedded[static_cast<size_t>(position * hidden + input_start + input)] *
                         convolution_[static_cast<size_t>((output * kInputsPerGroup + input) * 2 + position)];
            }
        }
        convolved[static_cast<size_t>(output)] = std::max(value, 0.0F);
    }

    std::vector<float> result(static_cast<size_t>(hidden), 0.0F);
    for (int64_t output = 0; output < hidden; ++output) {
        double value = decoder_bias_[static_cast<size_t>(output)];
        const float * weight = decoder_projection_.data() + output * hidden;
        for (int64_t input = 0; input < hidden; ++input) {
            value += static_cast<double>(weight[input]) *
                     static_cast<double>(convolved[static_cast<size_t>(input)]);
        }
        result[static_cast<size_t>(output)] = static_cast<float>(value);
    }
    return result;
}

std::vector<float> TransducerScorer::score(
    const float * encoder_frame,
    const std::array<int32_t, 2> & context) const {
    if (!encoder_frame) {
        throw std::runtime_error("sherpa KWS transducer scorer requires an encoder frame");
    }
    const int64_t hidden = assets_->config.decoder_dim;
    const int64_t vocabulary = assets_->config.vocab_size;
    const auto decoder = predictor(context);
    for (int64_t index = 0; index < hidden; ++index) {
        activated_[static_cast<size_t>(index)] =
            std::tanh(encoder_frame[index] + decoder[static_cast<size_t>(index)]);
    }
    std::vector<float> scores(static_cast<size_t>(vocabulary));
#pragma omp parallel for schedule(static) if(vocabulary >= 256)
    for (int64_t token = 0; token < vocabulary; ++token) {
        double value = joiner_bias_[static_cast<size_t>(token)];
        const float * weight = joiner_projection_.data() + token * hidden;
        for (int64_t index = 0; index < hidden; ++index) {
            value += static_cast<double>(weight[index]) *
                     static_cast<double>(activated_[static_cast<size_t>(index)]);
        }
        scores[static_cast<size_t>(token)] = static_cast<float>(value);
    }
    return scores;
}

}  // namespace engine::models::sherpa_kws
