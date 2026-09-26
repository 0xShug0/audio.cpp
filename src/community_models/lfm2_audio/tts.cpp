#include "engine/community_models/lfm2_audio/tts.h"

#include "engine/framework/debug/profiler.h"

#include <algorithm>
#include <chrono>
#include <cmath>
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

}  // namespace

bool lfm2_speaks(const std::vector<int32_t> & codes, int32_t end_of_audio) {
    return std::find(codes.begin(), codes.end(), end_of_audio) == codes.end();
}

int32_t lfm2_greedy(const std::vector<float> & logits) {
    if (!std::all_of(logits.begin(), logits.end(), [](float value) { return std::isfinite(value); })) {
        throw std::runtime_error("LFM2-Audio backbone produced non-finite logits");
    }

    return static_cast<int32_t>(std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
}

Lfm2CodeSampler::Lfm2CodeSampler(const Lfm2AudioSampling & sampling)
    : greedy_(sampling.temperature <= 0.0f || sampling.top_k == 1), rng_(static_cast<uint32_t>(sampling.seed)) {
    options_.do_sample = !greedy_;
    options_.temperature = sampling.temperature;
    options_.top_k = sampling.top_k;
}

int32_t Lfm2CodeSampler::pick(std::vector<float> & logits) {
    if (greedy_) {
        return lfm2_greedy(logits);
    }

    scratch_.reserve_vocab(logits.size());
    return sampler_.sample(logits, {}, options_, scratch_, rng_, nullptr, "lfm2_audio audio code");
}

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

struct Lfm2SpeechGenerator::Impl {
    Impl(Lfm2BackboneRuntime & backbone_in,
         Lfm2DepthformerRuntime & depthformer_in,
         const Lfm2TextTokenizer & tokenizer,
         Lfm2Prompt prompt_in,
         int32_t end_of_audio_in,
         const Lfm2SpeechOptions & options_in)
        : backbone(backbone_in),
          depthformer(depthformer_in),
          prompt(std::move(prompt_in)),
          end_of_audio(end_of_audio_in),
          audio_start(tokenizer.require_token_id("<|audio_start|>")),
          end_of_turn(tokenizer.require_token_id("<|im_end|>")),
          options(options_in),
          sampler(options.sampling) {
        if (options.max_frames <= 0) {
            throw std::runtime_error("LFM2-Audio speech needs a positive frame budget");
        }
    }

    // The prompt and any text before <|audio_start|>; leaves the backbone
    // output that the first frame comes from.
    void start() {
        auto logits = backbone.start(prompt, {}, kMaxTextTokens + options.max_frames);
        while (true) {
            const int32_t token = lfm2_greedy(logits);
            if (token == end_of_turn) {
                throw std::runtime_error("LFM2-Audio ended the turn without speech");
            }

            if (token == audio_start) {
                hidden = backbone.step_text(token, Lfm2StepOutput::Hidden);
                return;
            }

            if (static_cast<int64_t>(text_tokens.size()) == kMaxTextTokens) {
                throw std::runtime_error("LFM2-Audio wrote text instead of starting speech");
            }

            text_tokens.push_back(token);
            logits = backbone.step_text(token, Lfm2StepOutput::Logits);
        }
    }

    Lfm2BackboneRuntime & backbone;
    Lfm2DepthformerRuntime & depthformer;
    Lfm2Prompt prompt;
    int32_t end_of_audio;
    int32_t audio_start;
    int32_t end_of_turn;
    Lfm2SpeechOptions options;
    Lfm2CodeSampler sampler;

    bool started = false;
    bool finished = false;
    bool ended = false;
    int64_t frames = 0;
    std::vector<float> hidden;
    std::vector<int32_t> text_tokens;
};

Lfm2SpeechGenerator::Lfm2SpeechGenerator(
    Lfm2BackboneRuntime & backbone,
    Lfm2DepthformerRuntime & depthformer,
    const Lfm2TextTokenizer & tokenizer,
    Lfm2Prompt prompt,
    int32_t end_of_audio,
    const Lfm2SpeechOptions & options)
    : impl_(std::make_unique<Impl>(backbone, depthformer, tokenizer, std::move(prompt), end_of_audio, options)) {}

Lfm2SpeechGenerator::~Lfm2SpeechGenerator() = default;

std::optional<std::vector<int32_t>> Lfm2SpeechGenerator::next_frame() {
    auto & s = *impl_;
    if (s.finished) {
        return std::nullopt;
    }

    if (!s.started) {
        s.started = true;
        s.start();
    }

    while (true) {
        // The frame after the last one never needs the backbone step it
        // would take to feed it back.
        if (s.frames == s.options.max_frames) {
            s.finished = true;
            return std::nullopt;
        }

        auto codes = s.depthformer.frame(s.hidden, [&](int64_t, std::vector<float> & logits) { return s.sampler.pick(logits); });

        // A frame whose first code is end-of-audio ends the speech; the
        // reference sets its other codes to end-of-audio and does not decode it.
        if (codes.front() == s.end_of_audio) {
            s.finished = true;
            s.ended = true;
            return std::nullopt;
        }

        ++s.frames;
        if (s.frames < s.options.max_frames) {
            s.hidden = s.backbone.step_audio(codes, Lfm2StepOutput::Hidden);
        }

        // End-of-audio picked for another codebook has no sound: the
        // reference feeds the frame back like any other, and its demo skips
        // it when decoding (demo/chat.py).
        if (lfm2_speaks(codes, s.end_of_audio)) {
            return codes;
        }
    }
}

bool Lfm2SpeechGenerator::ended() const {
    return impl_->ended;
}

const std::vector<int32_t> & Lfm2SpeechGenerator::text_tokens() const {
    return impl_->text_tokens;
}

Lfm2Speech generate_lfm2_speech(
    Lfm2BackboneRuntime & backbone,
    Lfm2DepthformerRuntime & depthformer,
    const Lfm2TextTokenizer & tokenizer,
    const Lfm2Prompt & prompt,
    int32_t end_of_audio,
    const Lfm2SpeechOptions & options) {
    const auto start_time = std::chrono::steady_clock::now();
    Lfm2SpeechGenerator generator(backbone, depthformer, tokenizer, prompt, end_of_audio, options);
    Lfm2Speech out;
    while (auto frame = generator.next_frame()) {
        out.frames.push_back(std::move(*frame));
    }

    out.ended = generator.ended();
    out.text_tokens = generator.text_tokens();
    debug::timing_log_scalar("lfm2_audio.speech.ms", engine::debug::elapsed_ms(start_time));
    debug::timing_log_scalar("lfm2_audio.speech.frames", static_cast<double>(out.frames.size()));
    return out;
}

}  // namespace engine::community_models::lfm2_audio
