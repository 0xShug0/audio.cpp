#include "engine/community_models/lfm2_audio/tts.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/sampling/hf_sampler.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::community_models::lfm2_audio {
namespace {

// generate_sequential's text phase before <|audio_start|>. The published
// checkpoints emit it first; a few tokens of slack before giving up.
constexpr int64_t kMaxTextTokens = 16;

struct Voice {
    const char * id;
    const char * description;
};

// liquid-audio README, "TTS": the four voices of the English checkpoint.
constexpr Voice kEnglishVoices[] = {
    {"us_male", "US male"},
    {"us_female", "US female"},
    {"uk_male", "UK male"},
    {"uk_female", "UK female"},
};

void append(std::vector<int32_t> & out, const std::vector<int32_t> & ids) {
    out.insert(out.end(), ids.begin(), ids.end());
}

int32_t greedy(const std::vector<float> & logits) {
    if (!std::all_of(logits.begin(), logits.end(), [](float value) { return std::isfinite(value); })) {
        throw std::runtime_error("LFM2-Audio backbone produced non-finite logits");
    }

    return static_cast<int32_t>(std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
}

}  // namespace

std::vector<std::string> lfm2_tts_voices(const std::string & language) {
    if (language == "ja") {
        return {};
    }

    std::vector<std::string> out;
    for (const auto & voice : kEnglishVoices) {
        out.emplace_back(voice.id);
    }

    return out;
}

std::string lfm2_tts_system_prompt(const std::string & language, const std::string & voice) {
    // LFM2.5-Audio-1.5B-JP model card, "TTS".
    if (language == "ja") {
        if (!voice.empty()) {
            throw std::runtime_error("the Japanese LFM2-Audio checkpoint has one voice; leave the voice unset, not " + voice);
        }

        return "Perform TTS in japanese.";
    }

    if (language != "en") {
        throw std::runtime_error("LFM2-Audio has no TTS prompt for language " + language);
    }

    const std::string wanted = voice.empty() ? kEnglishVoices[0].id : voice;
    std::string known;
    for (const auto & entry : kEnglishVoices) {
        if (wanted == entry.id) {
            return std::string("Perform TTS. Use the ") + entry.description + " voice.";
        }

        known += (known.empty() ? "" : ", ") + std::string(entry.id);
    }

    throw std::runtime_error("unknown LFM2-Audio voice " + wanted + "; the voices are " + known);
}

Lfm2Prompt make_lfm2_tts_prompt(const Lfm2TextTokenizer & tokenizer, const std::string & system_prompt, const std::string & text) {
    for (const char * token : {"<|startoftext|>", "<|im_start|>", "<|im_end|>"}) {
        (void)tokenizer.require_token_id(token);
    }

    // ChatState (processor.py) encodes each piece on its own.
    Lfm2Prompt out;
    for (const std::string & piece : {std::string("<|startoftext|>"), std::string("<|im_start|>system\n"), system_prompt,
                                      std::string("<|im_end|>\n"), std::string("<|im_start|>user\n"), text,
                                      std::string("<|im_end|>\n"), std::string("<|im_start|>assistant\n")}) {
        append(out.input_ids, tokenizer.encode(piece));
    }

    return out;
}

Lfm2Speech generate_lfm2_speech(
    Lfm2BackboneRuntime & backbone,
    Lfm2DepthformerRuntime & depthformer,
    const Lfm2TextTokenizer & tokenizer,
    const Lfm2Prompt & prompt,
    int32_t end_of_audio,
    const Lfm2SpeechOptions & options) {
    if (options.max_frames <= 0) {
        throw std::runtime_error("LFM2-Audio speech needs a positive frame budget");
    }

    const int32_t audio_start = tokenizer.require_token_id("<|audio_start|>");
    const int32_t end_of_turn = tokenizer.require_token_id("<|im_end|>");
    const auto & sampling = options.sampling;
    const bool greedy_audio = sampling.temperature <= 0.0f || sampling.top_k == 1;

    sampling::HfSamplingOptions sampler_options;
    sampler_options.do_sample = !greedy_audio;
    sampler_options.temperature = sampling.temperature;
    sampler_options.top_k = sampling.top_k;
    const sampling::HfSampler sampler;
    sampling::HfSamplerScratch scratch;
    std::mt19937 rng(static_cast<uint32_t>(sampling.seed));
    const std::vector<int32_t> no_history;
    const auto pick = [&](int64_t, std::vector<float> & logits) {
        if (greedy_audio) {
            return greedy(logits);
        }

        scratch.reserve_vocab(logits.size());
        return sampler.sample(logits, no_history, sampler_options, scratch, rng, nullptr, "lfm2_audio audio code");
    };

    Lfm2Speech out;
    const auto start_time = std::chrono::steady_clock::now();
    auto logits = backbone.start(prompt, {}, kMaxTextTokens + options.max_frames);
    std::vector<float> hidden;
    while (hidden.empty()) {
        const int32_t token = greedy(logits);
        if (token == end_of_turn) {
            throw std::runtime_error("LFM2-Audio ended the turn without speech");
        }

        if (token == audio_start) {
            hidden = backbone.step_text(token, Lfm2StepOutput::Hidden);
            break;
        }

        if (static_cast<int64_t>(out.text_tokens.size()) == kMaxTextTokens) {
            throw std::runtime_error("LFM2-Audio wrote text instead of starting speech");
        }

        out.text_tokens.push_back(token);
        logits = backbone.step_text(token, Lfm2StepOutput::Logits);
    }

    // A frame whose first code is end-of-audio ends the speech; the reference
    // sets its other codes to end-of-audio and does not decode it.
    while (static_cast<int64_t>(out.frames.size()) < options.max_frames) {
        auto codes = depthformer.frame(hidden, pick);
        if (codes.front() == end_of_audio) {
            out.ended = true;
            break;
        }

        out.frames.push_back(codes);
        if (static_cast<int64_t>(out.frames.size()) < options.max_frames) {
            hidden = backbone.step_audio(codes, Lfm2StepOutput::Hidden);
        }
    }

    debug::timing_log_scalar("lfm2_audio.speech.ms", engine::debug::elapsed_ms(start_time));
    debug::timing_log_scalar("lfm2_audio.speech.frames", static_cast<double>(out.frames.size()));
    return out;
}

}  // namespace engine::community_models::lfm2_audio
