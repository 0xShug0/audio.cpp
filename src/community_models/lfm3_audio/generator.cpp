#include "engine/community_models/lfm3_audio/generator.h"

#include "engine/community_models/lfm3_audio/debug.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace engine::community_models::lfm3_audio {

Lfm3ModeMachine::Lfm3ModeMachine(Lfm3Mode initial, const Lfm3TokenIds & ids, std::vector<int32_t> eos_ids, int64_t n_text,
                                 int64_t n_audio, int32_t end_of_audio)
    : mode_(initial),
      text_end_(ids.text_end),
      audio_start_(ids.audio_start),
      mixed_start_(ids.mixed_start),
      eos_ids_(std::move(eos_ids)),
      n_text_(n_text),
      n_audio_(n_audio),
      end_of_audio_(end_of_audio),
      slots_left_(n_text) {
    if (n_text <= 0 || n_audio <= 0) {
        throw std::runtime_error("LFM3-Audio generation needs positive interleaved text and audio block sizes");
    }
}

Lfm3SlotKind Lfm3ModeMachine::next_slot() const {
    return mode_ == Lfm3Mode::Audio || (mode_ == Lfm3Mode::Interleaved && speaking_) ? Lfm3SlotKind::Audio : Lfm3SlotKind::Text;
}

void Lfm3ModeMachine::advance(const Lfm3Step & step) {
    if (finished_) {
        throw std::runtime_error("LFM3-Audio generation advanced past the end of the turn");
    }

    const bool frame = !step.codes.empty();
    if (frame != (next_slot() == Lfm3SlotKind::Audio)) {
        throw std::runtime_error(frame ? "LFM3-Audio generation got a frame for a text slot"
                                       : "LFM3-Audio generation got a text token for an audio slot");
    }

    // The block counter below follows the mode this slot started in.
    const bool interleaving = mode_ == Lfm3Mode::Interleaved;
    if (frame) {
        if (std::find(step.codes.begin(), step.codes.end(), end_of_audio_) != step.codes.end()) {
            mode_ = Lfm3Mode::Text;
        } else if (interleaving && text_done_) {
            ++tail_frames_;
        }
    } else if (step.token == text_end_) {
        // Checked before the eos ids, which list <|text_end|> too: inside a block it only ends the text.
        if (interleaving) {
            text_done_ = speaking_ = true;
        } else {
            finished_ = true;
        }
    } else if (std::find(eos_ids_.begin(), eos_ids_.end(), step.token) != eos_ids_.end()) {
        finished_ = true;
    } else if (interleaving) {
        // The mode markers only open a block; inside one they are ordinary text.
    } else if (step.token == audio_start_) {
        mode_ = Lfm3Mode::Audio;
    } else if (step.token == mixed_start_) {
        mode_ = Lfm3Mode::Interleaved;
        slots_left_ = n_text_;
        speaking_ = text_done_ = false;
        tail_frames_ = 0;
    }

    if (interleaving && !text_done_ && --slots_left_ == 0) {
        speaking_ = !speaking_;
        slots_left_ = speaking_ ? n_audio_ : n_text_;
    }
}

bool Lfm3ModeMachine::finished() const noexcept {
    return finished_;
}

Lfm3Mode Lfm3ModeMachine::mode() const noexcept {
    return mode_;
}

bool Lfm3ModeMachine::force_audio_eos(int64_t max_audio_tail_frames) const noexcept {
    return max_audio_tail_frames > 0 && mode_ == Lfm3Mode::Interleaved && text_done_ && tail_frames_ >= max_audio_tail_frames;
}

std::vector<int32_t> lfm3_text_suppress(const Lfm3TokenIds & ids) {
    return {ids.audio, ids.audio_out};
}

struct Lfm3Generator::Impl {
    Impl(Lfm3BackboneRuntime & backbone_in, Lfm3DepthDecoderRuntime * depth_in, const Lfm3Tokenizer & tokenizer,
         const Lfm3GenerationConfig & config, const Lfm3DepthConfig & depth_config, Lfm3Prompt prompt_in,
         const Lfm3AudioEmbeddings & audio_in, Lfm3GenerationOptions options_in)
        : backbone(backbone_in),
          depth(depth_in),
          ids(tokenizer.ids()),
          end_of_audio(depth_config.end_of_audio()),
          codebooks(depth_config.codebooks),
          schedule(depth_config.maskgit_schedule),
          prompt(std::move(prompt_in)),
          audio(audio_in),
          options(std::move(options_in)),
          machine(lfm3_initial_mode(prompt, ids, end_of_audio), ids, config.specials.eos_ids, config.interleaved_text,
                  config.interleaved_audio, end_of_audio),
          audio_sampler(options.audio, options.seed),
          text_sampler(options.text, options.seed),
          suppress(lfm3_text_suppress(ids)),
          dump(lfm3_dump_enabled()),
          dump_frames(dump ? lfm3_dump_frames() : 0) {
        if (options.max_steps <= 0) {
            throw std::runtime_error("LFM3-Audio generation needs max_steps > 0");
        }
        if (codebooks != static_cast<int64_t>(Lfm3Codes{}.size())) {
            throw std::runtime_error("LFM3-Audio generation expects 12 codebooks");
        }
        suppress.insert(suppress.end(), options.suppress.begin(), options.suppress.end());
    }

    [[nodiscard]] Lfm3Output output_for_next() const {
        return machine.next_slot() == Lfm3SlotKind::Text ? Lfm3Output::Logits : Lfm3Output::Hidden;
    }

    Lfm3Codes next_frame() {
        if (depth == nullptr) {
            throw std::runtime_error("LFM3-Audio generation reached an audio slot without a depth decoder");
        }
        Lfm3Codes codes;
        if (machine.force_audio_eos(options.max_audio_tail_frames)) {
            codes.fill(end_of_audio);
            return codes;
        }
        const auto order = frames_sampled < options.maskgit_orders.size() ? options.maskgit_orders[frames_sampled] : audio_sampler.next_order();
        Lfm3FrameTaps taps;
        const bool tap = dump && static_cast<int64_t>(frames_sampled) < dump_frames;
        codes = lfm3_maskgit_frame(*depth, out, order, schedule, audio_sampler, end_of_audio, tap ? &taps : nullptr);
        if (tap) {
            for (size_t p = 0; p < taps.pass_logits.size(); ++p) {
                const auto & logits = taps.pass_logits[p];
                lfm3_dump("depth_logits_f" + std::to_string(frames_sampled) + "_p" + std::to_string(p), logits.data(),
                          {codebooks, static_cast<int64_t>(logits.size()) / codebooks});
            }
        }
        if (dump) {
            orders.insert(orders.end(), order.begin(), order.end());
        }
        ++frames_sampled;
        if (codes[0] != end_of_audio) {  // lfm3_maskgit_frame makes an EOS frame all end_of_audio
            audio_sampler.commit(codes);
        }
        return codes;
    }

    void write_dumps() const {
        if (!dump) {
            return;
        }
        std::vector<int32_t> rows;
        for (const auto & step : steps) {
            rows.push_back(step.token);
            for (int64_t k = 0; k < codebooks; ++k) {
                rows.push_back(step.codes.empty() ? -1 : step.codes[static_cast<size_t>(k)]);
            }
        }
        lfm3_dump("steps", rows.data(), {static_cast<int64_t>(steps.size()), codebooks + 1});
        lfm3_dump("maskgit_orders", orders.data(), {static_cast<int64_t>(frames_sampled), codebooks});
    }

    Lfm3BackboneRuntime & backbone;
    Lfm3DepthDecoderRuntime * depth;
    Lfm3TokenIds ids;
    int32_t end_of_audio;
    int64_t codebooks;
    std::vector<int64_t> schedule;
    Lfm3Prompt prompt;
    Lfm3AudioEmbeddings audio;  // until the prefill
    Lfm3GenerationOptions options;
    Lfm3ModeMachine machine;
    Lfm3AudioSampler audio_sampler;
    Lfm3TextSampler text_sampler;
    std::vector<int32_t> suppress;
    bool dump;
    int64_t dump_frames;
    std::vector<float> out;  // the backbone's output for the next slot: logits (text) or hidden (audio)
    std::vector<Lfm3Step> steps;
    std::vector<int32_t> orders;  // dumps: every sampled frame's order
    size_t frames_sampled = 0;  // frames drawn by MaskGIT (not forced EOS frames)
    bool started = false;
    bool ended = false;
    bool done = false;
};

Lfm3Generator::Lfm3Generator(Lfm3BackboneRuntime & backbone, Lfm3DepthDecoderRuntime * depth, const Lfm3Tokenizer & tokenizer,
                             const Lfm3GenerationConfig & config, const Lfm3DepthConfig & depth_config, Lfm3Prompt prompt,
                             const Lfm3AudioEmbeddings & audio, Lfm3GenerationOptions options)
    : impl_(std::make_unique<Impl>(backbone, depth, tokenizer, config, depth_config, std::move(prompt), audio, std::move(options))) {}

Lfm3Generator::~Lfm3Generator() = default;

// One slot per call, as upstream's generate() loop.
std::optional<Lfm3Step> Lfm3Generator::next() {
    auto & g = *impl_;
    if (g.done) {
        return std::nullopt;
    }
    if (!g.started) {
        g.out = g.backbone.prefill(g.prompt, g.ids.audio, g.ids.audio_out, g.audio, g.options.max_steps, g.output_for_next());
        g.audio = {};
        g.started = true;
    }

    Lfm3Step step;
    if (g.machine.next_slot() == Lfm3SlotKind::Audio) {
        const auto codes = g.next_frame();
        step.codes.assign(codes.begin(), codes.end());
    } else {
        step.token = g.text_sampler.pick(g.out, g.suppress);
    }
    g.machine.advance(step);
    g.steps.push_back(step);

    const bool eos_frame = !step.codes.empty() && step.codes[0] == g.end_of_audio;
    g.ended = g.machine.finished() || (g.options.stop_after_audio_eos && eos_frame);
    g.done = g.ended || static_cast<int64_t>(g.steps.size()) >= g.options.max_steps;
    if (!g.done) {
        const auto output = g.output_for_next();
        g.out = step.codes.empty() ? g.backbone.step_text(step.token, output) : g.backbone.step_frame(step.codes, output);
    }
    if (g.done) {
        g.write_dumps();
    }
    return step;
}

bool Lfm3Generator::ended() const noexcept {
    return impl_->ended;
}

const std::vector<Lfm3Step> & Lfm3Generator::steps() const noexcept {
    return impl_->steps;
}

}  // namespace engine::community_models::lfm3_audio
