#include "engine/community_models/whistle_asr/runtime.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::community_models::whistle_asr {
namespace {

constexpr int kTextVocabulary = 8192;
constexpr std::array<const char *, 7> kLanguages = {"en", "de", "fr", "es", "it", "nl", "pl"};

void require_f32_checkpoint(const assets::TensorSource & source) {
    for (const auto & metadata : source.tensors()) {
        if (metadata.dtype != "f32" && metadata.dtype != "F32" && metadata.dtype != "float32") {
            throw std::runtime_error("Whistle currently requires FP32 checkpoint tensors");
        }
    }
}

std::shared_ptr<const WhistleAssets> checked(
    std::shared_ptr<const WhistleAssets> assets, const core::ExecutionContext & execution_context) {
    if (execution_context.config().threads < 1 || execution_context.config().threads > 64) {
        throw std::invalid_argument("Whistle CPU thread count must be between 1 and 64");
    }
    if (!assets || !assets->weights) {
        throw std::invalid_argument("Whistle needs verified model assets");
    }
    require_f32_checkpoint(*assets->weights);
    return assets;
}

}  // namespace

WhistleRuntime::WhistleRuntime(
    std::shared_ptr<const WhistleAssets> assets, core::ExecutionContext & execution_context)
    : assets_(checked(std::move(assets), execution_context)),
      encoder_(assets_, execution_context),
      decoder_(assets_, execution_context),
      frontend_(assets_->mel_filterbank) {}

WhistleRuntime::~WhistleRuntime() = default;

WhistleTranscript WhistleRuntime::transcribe(
    const runtime::AudioBuffer & audio, const std::string & language, const WhistleDecodeObserver & observer) {
    if (audio.sample_rate != 16000 || audio.channels != 1) {
        throw std::invalid_argument("Whistle requires 16 kHz mono audio");
    }
    if (audio.samples.size() > 30 * 16000) {
        throw std::invalid_argument("Whistle audio exceeds the 30-second input limit");
    }
    if (!language.empty() &&
        std::find_if(kLanguages.begin(), kLanguages.end(),
            [&](const char * candidate) { return language == candidate; }) == kLanguages.end()) {
        throw std::invalid_argument("Whistle does not support the requested language");
    }
    double energy = 0.0;
    for (float sample : audio.samples) {
        if (!std::isfinite(sample)) {
            throw std::invalid_argument("Whistle audio contains a non-finite sample");
        }
        energy += static_cast<double>(sample) * sample;
    }
    if (audio.samples.size() < 640) {
        return {"", ""};
    }
    if (energy == 0.0) {
        return {"", ""};
    }
    const auto encode_started = std::chrono::steady_clock::now();
    const auto mel = frontend_.extract(audio.samples);
    const WhistleEncoderOutput encoder = encoder_.encode(mel);
    debug::timing_log_scalar("whistle_asr.encode_ms", debug::elapsed_ms(encode_started));
    const auto decode_started = std::chrono::steady_clock::now();
    decoder_.start(encoder);
    std::vector<int32_t> tokens{2};
    std::vector<int32_t> text;
    std::string detected_language;
    bool ended = false;
    for (size_t position = 0; position <= kWhistleMaximumTextTokens + 1; ++position) {
        const std::vector<float> & logits = decoder_.step(tokens);
        if (observer) {
            observer(WhistleDecodeStep{position, tokens.back(), logits});
        }
        if (position == 0) {
            const size_t index = language.empty()
                ? static_cast<size_t>(std::max_element(logits.begin() + kTextVocabulary,
                    logits.begin() + kWhistleVocabulary) - (logits.begin() + kTextVocabulary))
                : static_cast<size_t>(std::find_if(kLanguages.begin(), kLanguages.end(),
                    [&](const char * candidate) { return language == candidate; }) - kLanguages.begin());
            detected_language = kLanguages[index];
            tokens.push_back(static_cast<int32_t>(kTextVocabulary + index));
            continue;
        }
        const int32_t next = static_cast<int32_t>(
            std::max_element(logits.begin(), logits.begin() + kTextVocabulary) - logits.begin());
        if (next == 1) {
            ended = true;
            break;
        }
        if (text.size() >= kWhistleMaximumTextTokens) {
            break;
        }
        text.push_back(next);
        tokens.push_back(next);
    }
    decoder_.log_timings();
    debug::timing_log_scalar("whistle_asr.decode_ms", debug::elapsed_ms(decode_started));
    if (!ended) {
        // The C API has no partial-result status; do not present capped text as complete speech.
        throw std::runtime_error("Whistle decoding reached the token limit without an end token");
    }
    std::string transcript = decode_whistle_tokens(assets_->tokenizer_pieces, text);
    const auto first = transcript.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        transcript.clear();
    } else {
        const auto last = transcript.find_last_not_of(" \t\r\n");
        transcript = transcript.substr(first, last + 1 - first);
    }
    if (transcript.empty()) {
        detected_language.clear();
    }
    return {std::move(transcript), std::move(detected_language)};
}

}  // namespace engine::community_models::whistle_asr
