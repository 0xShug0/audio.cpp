#include "engine/community_models/kroko_asr/decoder.h"

#include "engine/framework/debug/profiler.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace engine::models::kroko_asr {
namespace {

constexpr int64_t kHidden = 512;
constexpr int64_t kGroups = 128;
constexpr int64_t kInputsPerGroup = 4;

}  // namespace

KrokoGreedyDecoder::KrokoGreedyDecoder(
    std::shared_ptr<const KrokoASRAssets> assets)
    : assets_(std::move(assets)) {
    if (assets_ == nullptr) {
        throw std::runtime_error("Kroko greedy decoder requires assets");
    }
    const auto & source = *assets_->weights;
    const int64_t vocab = assets_->config.vocab_size;
    embedding_ = source.require_f32(
        "decoder.decoder.embedding.weight",
        {vocab, kHidden});
    conv_ = source.require_f32(
        "decoder.decoder.conv.weight",
        {kHidden, kInputsPerGroup, 2});
    decoder_projection_ = source.require_f32(
        "decoder.decoder_proj.weight",
        {kHidden, kHidden});
    decoder_bias_ = source.require_f32(
        "decoder.decoder_proj.bias",
        {kHidden});
    joiner_projection_ = source.require_f32(
        "joiner.output_linear.weight",
        {vocab, kHidden});
    joiner_bias_ = source.require_f32(
        "joiner.output_linear.bias",
        {vocab});
    reset();
}

std::array<float, 512> KrokoGreedyDecoder::predictor(
    const std::array<int32_t, 2> & context) const {
    std::array<float, kHidden * 2> embedded{};
    const int64_t vocab = assets_->config.vocab_size;
    for (int64_t position = 0; position < 2; ++position) {
        const int32_t token = context[static_cast<size_t>(position)];
        if (token < 0) {
            continue;
        }
        if (token >= vocab) {
            throw std::runtime_error("Kroko predictor token is outside the vocabulary");
        }
        std::copy_n(
            embedding_.data() + static_cast<int64_t>(token) * kHidden,
            kHidden,
            embedded.data() + position * kHidden);
    }

    std::array<float, kHidden> convolved{};
    for (int64_t output = 0; output < kHidden; ++output) {
        const int64_t group = output / (kHidden / kGroups);
        const int64_t input_start = group * kInputsPerGroup;
        float value = 0.0F;
        for (int64_t input = 0; input < kInputsPerGroup; ++input) {
            for (int64_t position = 0; position < 2; ++position) {
                value +=
                    embedded[static_cast<size_t>(position * kHidden + input_start + input)] *
                    conv_[static_cast<size_t>(
                        (output * kInputsPerGroup + input) * 2 + position)];
            }
        }
        convolved[static_cast<size_t>(output)] = std::max(value, 0.0F);
    }

    std::array<float, kHidden> result{};
    for (int64_t output = 0; output < kHidden; ++output) {
        double value = decoder_bias_[static_cast<size_t>(output)];
        const float * weight =
            decoder_projection_.data() + output * kHidden;
        for (int64_t input = 0; input < kHidden; ++input) {
            value += static_cast<double>(weight[input]) *
                static_cast<double>(convolved[static_cast<size_t>(input)]);
        }
        result[static_cast<size_t>(output)] = static_cast<float>(value);
    }
    return result;
}

int32_t KrokoGreedyDecoder::join(
    const float * encoder_frame,
    const std::array<float, 512> & decoder_output) const {
    const int64_t vocab = assets_->config.vocab_size;
    int32_t best_id = 0;
    float best_value = -std::numeric_limits<float>::infinity();
    for (int64_t token = 0; token < vocab; ++token) {
        double value = joiner_bias_[static_cast<size_t>(token)];
        const float * weight =
            joiner_projection_.data() + token * kHidden;
        for (int64_t hidden = 0; hidden < kHidden; ++hidden) {
            const float activated = std::tanh(
                encoder_frame[hidden] +
                decoder_output[static_cast<size_t>(hidden)]);
            value += static_cast<double>(weight[hidden]) *
                static_cast<double>(activated);
        }
        if (static_cast<float>(value) > best_value) {
            best_value = static_cast<float>(value);
            best_id = static_cast<int32_t>(token);
        }
    }
    return best_id;
}

void KrokoGreedyDecoder::reset() {
    context_ = {
        -1,
        static_cast<int32_t>(assets_->config.blank_id),
    };
    decoder_output_ = predictor(context_);
    decoded_ = KrokoDecodedTokens{};
    decoded_frames_ = 0;
}

const KrokoDecodedTokens & KrokoGreedyDecoder::append(
    const std::vector<float> & encoder_output,
    int64_t frames,
    int64_t hidden_size) {
    if (frames <= 0 || hidden_size != kHidden ||
        static_cast<int64_t>(encoder_output.size()) != frames * hidden_size) {
        throw std::runtime_error("Kroko greedy decoder received invalid encoder output");
    }
    const auto start = std::chrono::steady_clock::now();
    for (int64_t frame = 0; frame < frames; ++frame) {
        const int32_t token = join(
            encoder_output.data() + frame * hidden_size,
            decoder_output_);
        if (token == assets_->config.blank_id || token == assets_->config.unk_id) {
            continue;
        }
        decoded_.ids.push_back(token);
        decoded_.frame_indices.push_back(decoded_frames_ + frame);
        context_[0] = context_[1];
        context_[1] = token;
        decoder_output_ = predictor(context_);
    }
    decoded_frames_ += frames;
    engine::debug::timing_log_scalar(
        "kroko_asr.decoder_ms",
        engine::debug::elapsed_ms(start));
    engine::debug::trace_log_scalar(
        "kroko_asr.decoder.tokens",
        decoded_.ids.size());
    return decoded_;
}

const KrokoDecodedTokens & KrokoGreedyDecoder::decoded() const noexcept {
    return decoded_;
}

KrokoDecodedTokens KrokoGreedyDecoder::decode(
    const std::vector<float> & encoder_output,
    int64_t frames,
    int64_t hidden_size) {
    reset();
    append(encoder_output, frames, hidden_size);
    return decoded_;
}

}  // namespace engine::models::kroko_asr
