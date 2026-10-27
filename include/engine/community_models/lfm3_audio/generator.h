#pragma once

// LFM3-Audio generation: the text/audio/interleaved mode machine and the slot
// loop that drives the backbone, the depth decoder and the samplers with it.

#include "engine/community_models/lfm3_audio/assets.h"
#include "engine/community_models/lfm3_audio/backbone.h"
#include "engine/community_models/lfm3_audio/chat.h"
#include "engine/community_models/lfm3_audio/depth.h"
#include "engine/community_models/lfm3_audio/encoder.h"
#include "engine/community_models/lfm3_audio/prompt.h"
#include "engine/community_models/lfm3_audio/sampling.h"
#include "engine/community_models/lfm3_audio/tokenizer.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace engine::community_models::lfm3_audio {

enum class Lfm3SlotKind { Text, Audio };

// Follows the upstream generate() loop slot for slot:
// - Text: <|audio_start|> opens Audio, <|mixed_start|> opens an interleaved
//   block; <|text_end|> or an eos id ends the turn.
// - Audio: frames until an EOS frame (any code == end_of_audio), then Text.
// - Interleaved: n_text text slots and n_audio frames alternate until
//   <|text_end|>, which starts the audio tail; an EOS frame goes back to Text;
//   an eos id (not <|text_end|>) ends the turn; the mode markers are plain text.
class Lfm3ModeMachine {
public:
    Lfm3ModeMachine(Lfm3Mode initial, const Lfm3TokenIds & ids, std::vector<int32_t> eos_ids, int64_t n_text, int64_t n_audio,
                    int32_t end_of_audio = 2016);
    [[nodiscard]] Lfm3SlotKind next_slot() const;
    void advance(const Lfm3Step & step);
    [[nodiscard]] bool finished() const noexcept;
    [[nodiscard]] Lfm3Mode mode() const noexcept;
    // True when the next slot must be the EOS frame: max_audio_tail_frames > 0 and that many frames followed
    // <|text_end|> in the current interleaved block.
    [[nodiscard]] bool force_audio_eos(int64_t max_audio_tail_frames) const noexcept;

private:
    Lfm3Mode mode_;
    int32_t text_end_, audio_start_, mixed_start_;
    std::vector<int32_t> eos_ids_;
    int64_t n_text_, n_audio_;
    int32_t end_of_audio_;
    int64_t slots_left_;  // slots left in the current interleaved block
    bool speaking_ = false;
    bool text_done_ = false;
    bool finished_ = false;
    int64_t tail_frames_ = 0;  // frames after <|text_end|> in this interleaved block
};

// Ids the text sampler never picks: the <|audio|> and <|audio_out|> placeholders.
std::vector<int32_t> lfm3_text_suppress(const Lfm3TokenIds & ids);

struct Lfm3GenerationOptions {
    int64_t max_steps = 0;
    Lfm3TextSampling text;
    Lfm3AudioSampling audio;
    uint64_t seed = 0;
    std::vector<int32_t> suppress;      // always {audio, audio_out}; chat adds {mixed_start, audio_start}
    bool stop_after_audio_eos = false;  // TTS: finish at the first EOS frame
    // At most this many tail frames are sampled after <|text_end|> in an interleaved block; the next audio slot then
    // emits the EOS frame (2016x12) unsampled and goes on as for a sampled EOS. 0 = no cap.
    int64_t max_audio_tail_frames = 0;      // S2S session default 375 (30 s)
    std::vector<Lfm3Codes> maskgit_orders;  // test/debug override, consumed one per frame, then RNG
};

class Lfm3Generator {
public:
    Lfm3Generator(Lfm3BackboneRuntime & backbone, Lfm3DepthDecoderRuntime * depth /*null: text only*/,
                  const Lfm3Tokenizer & tokenizer, const Lfm3GenerationConfig & config, const Lfm3DepthConfig & depth_config,
                  Lfm3Prompt prompt, const Lfm3AudioEmbeddings & audio, Lfm3GenerationOptions options);
    ~Lfm3Generator();
    // The first call prefills; nullopt when finished or max_steps used. With LFM3_AUDIO_DUMP_DIR set it writes
    // the steps, maskgit_orders and depth_logits_f{f}_p{p} dumps (the backbone writes the prefill ones).
    std::optional<Lfm3Step> next();
    [[nodiscard]] bool ended() const noexcept;  // finished by the mode machine / stop rule, not by max_steps
    [[nodiscard]] const std::vector<Lfm3Step> & steps() const noexcept;  // all emitted, EOS frame included (history artifact)

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::community_models::lfm3_audio
