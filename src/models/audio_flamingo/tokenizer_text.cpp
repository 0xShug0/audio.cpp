#include "engine/models/audio_flamingo/tokenizer_text.h"

#include "engine/framework/tokenizers/llama_bpe.h"
#include "engine/framework/runtime/errors.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::audio_flamingo {
struct AudioFlamingoTextTokenizer::Impl {
    explicit Impl(std::shared_ptr<const AudioFlamingoAssets> input_assets)
        : assets(std::move(input_assets)),
          tokenizer(tokenizers::LlamaBpeTokenizerSpec{
              {}, {}, assets->resources.require_file("tokenizer_config"),
              assets->resources.require_file("tokenizer_json"), tokenizers::LlamaBpePreTokenizer::Qwen2}) {
        const auto token = tokenizer.find_token_id("<sound>");
        if (!token || *token != assets->config.text_decoder.audio_token_id) {
            throw std::runtime_error("Audio Flamingo tokenizer audio token does not match the model config");
        }
        audio_id = *token;
        const auto chat_template = assets->resources.read_text("chat_template");
        const std::string system_start = "<|im_start|>system\n";
        const auto start = chat_template.find(system_start);
        const auto end = chat_template.find("<|im_end|>", start);
        if (start == std::string::npos || end == std::string::npos) {
            throw std::runtime_error("Audio Flamingo chat template requires a static system message");
        }
        system_prompt = chat_template.substr(start + system_start.size(), end - start - system_start.size());
        if (assets->config.variant == AudioFlamingoVariant::Next) {
            const auto bos = tokenizer.find_token_id("<|sound_bos|>");
            const auto eos = tokenizer.find_token_id("<|sound_eos|>");
            if (!bos || !eos || *bos != assets->config.text_decoder.audio_bos_token_id ||
                *eos != assets->config.text_decoder.audio_eos_token_id) {
                throw std::runtime_error("Audio Flamingo Next tokenizer framing tokens do not match the model config");
            }
        }
    }

    std::shared_ptr<const AudioFlamingoAssets> assets;
    tokenizers::LlamaBpeTokenizer tokenizer;
    int32_t audio_id = 0;
    std::string system_prompt;
};

AudioFlamingoTextTokenizer::AudioFlamingoTextTokenizer(std::shared_ptr<const AudioFlamingoAssets> assets)
    : impl_(std::make_shared<Impl>(std::move(assets))) {}

std::string AudioFlamingoTextTokenizer::decode(const std::vector<int32_t> & token_ids, bool skip_special_tokens) const {
    return impl_->tokenizer.decode(token_ids, skip_special_tokens);
}

AudioFlamingoPrompt AudioFlamingoTextTokenizer::build_prompt(const std::string & prompt, const AudioFlamingoAudioFeatures & features, bool temporal_grounding) const {
    if (features.post_lengths.empty()) {
        throw std::runtime_error("Audio Flamingo prompt requires audio post lengths");
    }
    std::string expanded;
    const int64_t audio_token_count = std::accumulate(features.post_lengths.begin(), features.post_lengths.end(), int64_t{0});
    if (audio_token_count <= 0) {
        throw std::runtime_error("Audio Flamingo prompt requires positive audio token length");
    }
    expanded.reserve(prompt.size() + 512 + static_cast<size_t>(audio_token_count) * 7);
    expanded += "<|im_start|>system\n";
    expanded += impl_->system_prompt;
    expanded += "<|im_end|>\n";
    expanded += "<|im_start|>user\n";
    if (impl_->assets->config.variant == AudioFlamingoVariant::Next) {
        expanded += "<|sound_bos|>";
    }
    if (temporal_grounding) {
        const int64_t seconds = audio_token_count / 25;
        if (seconds == 0) {
            throw runtime::InvalidRequestError("SpotSound requires at least one second of audio");
        }
        for (int64_t second = 0; second < seconds; ++second) {
            expanded += "timestamp: " + std::to_string(second) + " seconds; feature: ";
            for (int64_t i = 0; i < 25; ++i) expanded += "<sound>";
        }
    } else {
        for (int64_t i = 0; i < audio_token_count; ++i) expanded += "<sound>";
    }
    if (impl_->assets->config.variant == AudioFlamingoVariant::Next) {
        expanded += "<|sound_eos|>";
    }
    expanded += prompt;
    expanded += "<|im_end|>\n<|im_start|>assistant\n";

    AudioFlamingoPrompt out;
    out.input_ids = impl_->tokenizer.encode(expanded, true);
    for (size_t i = 0; i < out.input_ids.size(); ++i) {
        if (out.input_ids[i] == impl_->audio_id) {
            out.audio_token_positions.push_back(static_cast<int32_t>(i));
        }
    }
    if (out.audio_token_positions.size() != static_cast<size_t>(temporal_grounding ? audio_token_count / 25 * 25 : audio_token_count)) {
        throw std::runtime_error("Audio Flamingo prompt audio token count does not match feature count");
    }
    return out;
}

}  // namespace engine::models::audio_flamingo
