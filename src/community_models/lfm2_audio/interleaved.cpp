#include "engine/community_models/lfm2_audio/interleaved.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace engine::community_models::lfm2_audio {

struct Lfm2InterleavedGenerator::Impl {
    enum class Modality { Text, Audio };

    Impl(Lfm2BackboneRuntime & backbone_in,
         Lfm2DepthformerRuntime & depthformer_in,
         const Lfm2TextTokenizer & tokenizer,
         Lfm2Prompt prompt_in,
         Lfm2AudioEmbeddings audio_in,
         int32_t end_of_audio_in,
         const Lfm2InterleavedOptions & options_in)
        : backbone(backbone_in),
          depthformer(depthformer_in),
          prompt(std::move(prompt_in)),
          audio(std::move(audio_in)),
          end_of_audio(end_of_audio_in),
          end_of_turn(tokenizer.require_token_id("<|im_end|>")),
          text_end(tokenizer.require_token_id("<|text_end|>")),
          options(options_in),
          sampler(options.sampling) {
        if (options.text_steps <= 0 || options.audio_steps <= 0) {
            throw std::runtime_error("LFM2-Audio interleaved generation needs positive text and audio block sizes");
        }

        if (options.max_steps <= 0) {
            throw std::runtime_error("LFM2-Audio interleaved generation needs a positive step budget");
        }
    }

    // Text blocks read the text logits, audio blocks the hidden state the
    // depthformer turns into a frame.
    [[nodiscard]] Lfm2StepOutput wanted() const {
        return modality == Modality::Text ? Lfm2StepOutput::Logits : Lfm2StepOutput::Hidden;
    }

    Lfm2BackboneRuntime & backbone;
    Lfm2DepthformerRuntime & depthformer;
    Lfm2Prompt prompt;
    Lfm2AudioEmbeddings audio;
    int32_t end_of_audio;
    int32_t end_of_turn;
    int32_t text_end;
    Lfm2InterleavedOptions options;
    Lfm2CodeSampler sampler;

    Modality modality = Modality::Text;
    int64_t left = 0;  // steps left in the current block
    bool text_done = false;
    bool started = false;
    bool finished = false;
    bool ended = false;
    int64_t steps = 0;
    std::vector<float> output;  // what the backbone gave for the next step
};

Lfm2InterleavedGenerator::Lfm2InterleavedGenerator(
    Lfm2BackboneRuntime & backbone,
    Lfm2DepthformerRuntime & depthformer,
    const Lfm2TextTokenizer & tokenizer,
    Lfm2Prompt prompt,
    Lfm2AudioEmbeddings audio,
    int32_t end_of_audio,
    const Lfm2InterleavedOptions & options)
    : impl_(std::make_unique<Impl>(backbone, depthformer, tokenizer, std::move(prompt), std::move(audio), end_of_audio, options)) {}

Lfm2InterleavedGenerator::~Lfm2InterleavedGenerator() = default;

std::optional<Lfm2ReplyStep> Lfm2InterleavedGenerator::next() {
    auto & s = *impl_;
    if (s.finished) {
        return std::nullopt;
    }

    if (!s.started) {
        s.started = true;
        s.left = s.options.text_steps;
        // A reply samples its audio, so it takes the speech sizing, as TTS
        // does: a seeded reply must not depend on the requests before.
        s.output = s.backbone.start(s.prompt, s.audio, s.options.max_steps, Lfm2DecodeCache::Speech);
    }

    if (s.steps == s.options.max_steps) {
        s.finished = true;
        return std::nullopt;
    }

    // generate_interleaved counts each step against the current block before
    // it runs.
    ++s.steps;
    --s.left;

    Lfm2ReplyStep step;
    if (s.modality == Impl::Modality::Text) {
        step.token = lfm2_greedy(s.output);
        if (step.token == s.end_of_turn) {
            s.finished = true;
            s.ended = true;
            return std::nullopt;
        }

        if (step.token == s.text_end) {
            s.text_done = true;
        }

        if (s.left == 0 || s.text_done) {
            s.modality = Impl::Modality::Audio;
            s.left = s.options.audio_steps;
        }
    } else {
        step.codes = s.depthformer.frame(s.output, [&](int64_t, std::vector<float> & logits) { return s.sampler.pick(logits); });
        if (s.left == 0 && !s.text_done) {
            s.modality = Impl::Modality::Text;
            s.left = s.options.text_steps;
        }

        if (step.codes.front() == s.end_of_audio) {
            std::fill(step.codes.begin(), step.codes.end(), s.end_of_audio);
            s.modality = Impl::Modality::Text;
        }
    }

    // The last step of the budget is never fed back.
    if (s.steps < s.options.max_steps) {
        s.output = step.codes.empty() ? s.backbone.step_text(step.token, s.wanted()) : s.backbone.step_audio(step.codes, s.wanted());
    }

    return step;
}

bool Lfm2InterleavedGenerator::ended() const {
    return impl_->ended;
}

}  // namespace engine::community_models::lfm2_audio
