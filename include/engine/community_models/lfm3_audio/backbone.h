#pragma once

// LFM3-Audio backbone: chunked prefill, one decode step per text token or
// audio frame, the KV and conv state between them, the tied text head and the
// audio-frame input embedding.

#include "engine/community_models/lfm3_audio/assets.h"
#include "engine/community_models/lfm3_audio/encoder.h"
#include "engine/community_models/lfm3_audio/prompt.h"
#include "engine/community_models/lfm3_audio/weights.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace engine::community_models::lfm3_audio {

enum class Lfm3Output { Logits, Hidden };  // Logits = [vocab_size], Hidden = final-normed [2048]

class Lfm3BackboneRuntime {
public:
    static constexpr int64_t kPrefillBlock = 256;

    Lfm3BackboneRuntime(std::shared_ptr<const assets::TensorSource> model, const Lfm3BackboneConfig & config,
                        int64_t codebooks, int64_t audio_vocab_size, core::ExecutionContext & execution,
                        const Lfm3WeightOptions & options);
    ~Lfm3BackboneRuntime();

    // Resets state, ensures cache capacity L + max_new_steps rounded up to 256, fills <|audio|> rows with `audio` and
    // <|audio_out|> rows with prompt.frames' embeddings in order, and returns the output for the last position.
    // `taps` (tests/dumps only): final-normed hidden rows at the given absolute positions, row-major [P][2048].
    struct Taps {
        std::vector<int64_t> positions;
        std::vector<float> rows;
    };
    std::vector<float> prefill(const Lfm3Prompt & prompt, int32_t audio_id, int32_t audio_out_id,
                               const Lfm3AudioEmbeddings & audio, int64_t max_new_steps, Lfm3Output output,
                               Taps * taps = nullptr);
    std::vector<float> step_text(int32_t token, Lfm3Output output);
    std::vector<float> step_frame(const std::vector<int32_t> & codes, Lfm3Output output);  // sum_k codebook_embedding[k*2017+c_k]
    std::vector<float> logits(const std::vector<float> & hidden);  // tied head on a given hidden (prefill dumps / tests)
    // Input embeddings of audio frames, row-major [frames][hidden], summed exactly as prefill and step_frame sum them.
    std::vector<float> embed_frames(const std::vector<std::vector<int32_t>> & frames);
    [[nodiscard]] int64_t position() const noexcept;  // next position
    [[nodiscard]] int64_t capacity() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::community_models::lfm3_audio
