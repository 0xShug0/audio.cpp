#pragma once

#include "engine/community_models/lfm3_audio/assets.h"
#include "engine/community_models/lfm3_audio/frontend.h"
#include "engine/community_models/lfm3_audio/prompt.h"
#include "engine/community_models/lfm3_audio/weights.h"
#include "engine/framework/core/execution_context.h"
#include <cstdint>
#include <memory>
#include <vector>
#include "engine/community_models/lfm3_audio/depth.h"
#include "engine/framework/sampling/hf_sampler.h"
#include <array>
#include <optional>
#include <random>

namespace engine::community_models::lfm3_audio {

enum class Lfm3Output { Logits, Hidden };  // Logits = [vocab_size], Hidden = final-normed [2048]

class Lfm3BackboneRuntime {
public:
    static constexpr int64_t kPrefillBlock = 256;

    Lfm3BackboneRuntime(Lfm3BackboneWeights weights, const Lfm3BackboneConfig & config,
                        int64_t codebooks, int64_t audio_vocab_size, core::ExecutionContext & execution,
                        const Lfm3WeightOptions & options);
    ~Lfm3BackboneRuntime();

    // Resets state, ensures cache capacity L + max_new_steps rounded up to 256, fills <|audio|> rows with `audio` and
    // <|audio_out|> rows with prompt.frames' embeddings in order, and returns the output for the last position.
    std::vector<float> prefill(const Lfm3Prompt & prompt, int32_t audio_id, int32_t audio_out_id,
                               const Lfm3AudioEmbeddings & audio, int64_t max_new_steps, Lfm3Output output);
    std::vector<float> step_text(int32_t token, Lfm3Output output);
    std::vector<float> step_frame(const std::vector<int32_t> & codes, Lfm3Output output);  // sum_k codebook_embedding[k*2017+c_k]
    std::vector<float> logits(const std::vector<float> & hidden);  // tied head on a given hidden
    // Input embeddings of audio frames, row-major [frames][hidden], summed exactly as prefill and step_frame sum them.
    std::vector<float> embed_frames(const std::vector<std::vector<int32_t>> & frames);
    [[nodiscard]] int64_t position() const noexcept;  // next position
    [[nodiscard]] int64_t capacity() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

struct Lfm3AudioSampling {
    std::optional<float> temperature = 0.4f;  // nullopt: unset (greedy unless top_k is set)
    int64_t top_k = 0;                        // 0: unset
    float repetition_penalty = 1.2f;          // 1: off
    int64_t penalty_window = 64;              // frames of history per codebook
};

struct Lfm3TextSampling {
    std::optional<float> temperature = 0.0f;  // nullopt: unset (greedy unless top_k is set)
    int64_t top_k = 0;                        // 0: unset
};

class Lfm3AudioSampler {
public:
    Lfm3AudioSampler(const Lfm3AudioSampling & sampling, uint64_t seed);
    Lfm3Codes next_order();  // Fisher-Yates with j = rng() % (i+1): portable, unlike std::shuffle
    int32_t pick(int64_t codebook, std::vector<float> & logits);
    void commit(const Lfm3Codes & frame);  // non-EOS frames only; keeps last penalty_window per codebook

private:
    Lfm3AudioSampling sampling_;
    sampling::HfSamplingOptions options_;
    std::mt19937 rng_;  // one stream for orders and draws (HF uses one global RNG)
    sampling::HfSampler sampler_;
    sampling::HfSamplerScratch scratch_;
    std::array<std::vector<int32_t>, 12> history_;  // oldest first
};

class Lfm3TextSampler {
public:
    Lfm3TextSampler(const Lfm3TextSampling & sampling, uint64_t seed);  // own stream, seed mixed as Lfm2CodeSampler
    int32_t pick(std::vector<float> & logits, const std::vector<int32_t> & suppress);

private:
    sampling::HfSamplingOptions options_;
    std::mt19937 rng_;
    sampling::HfSampler sampler_;
    sampling::HfSamplerScratch scratch_;
};

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
};

class Lfm3Generator {
public:
    Lfm3Generator(Lfm3BackboneRuntime & backbone, Lfm3DepthDecoderRuntime * depth /*null: text only*/,
                  const Lfm3Tokenizer & tokenizer, const Lfm3GenerationConfig & config, const Lfm3DepthConfig & depth_config,
                  Lfm3Prompt prompt, const Lfm3AudioEmbeddings & audio, Lfm3GenerationOptions options);
    ~Lfm3Generator();
    // The first call prefills; nullopt when finished or max_steps used.
    std::optional<Lfm3Step> next();
    [[nodiscard]] bool ended() const noexcept;  // finished by the mode machine / stop rule, not by max_steps
    [[nodiscard]] const std::vector<Lfm3Step> & steps() const noexcept;  // all emitted, EOS frame included (history artifact)

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::community_models::lfm3_audio
