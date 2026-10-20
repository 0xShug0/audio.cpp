#pragma once

// GPT-2 autoregressive mel-code generator for xTTS v2.
//
// Pipeline:
//   1. Reference audio -> 80-band log-mel -> ConditioningEncoder -> Perceiver -> 32 cond latents
//   2. Text -> BPE tokenize -> text tokens
//   3. GPT-2 AR: [cond_latents | text_embs | start_mel] -> mel codes
//   4. Reuse the final normalized AR hidden states for the vocoder.

#include "engine/framework/core/backend.h"
#include "engine/framework/core/execution_context.h"
#include "engine/community_models/xtts_v2/assets.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace engine::models::xtts_v2 {

// GPT-2 inference engine for xTTS v2.
class XttSv2GptEngine {
public:
    XttSv2GptEngine();
    ~XttSv2GptEngine();

    XttSv2GptEngine(const XttSv2GptEngine &) = delete;
    XttSv2GptEngine & operator=(const XttSv2GptEngine &) = delete;

    // Load from GGUF. Returns true on success.
    bool load(const XttSv2Assets & assets, core::ExecutionContext & execution);

    // Compute GPT conditioning latents from reference audio.
    // ref_pcm: mono 22050 Hz float32 PCM.
    // Returns true on success. cond_latents: [n_latents, d_model].
    bool compute_conditioning(const float * ref_pcm, int ref_n_samples,
                              std::vector<float> & cond_latents);

    // Autoregressive mel-code generation.
    // text_tokens: BPE token IDs.
    // cond_latents: [n_latents, d_model] conditioning from compute_conditioning.
    // Returns generated mel codes, including EOS when sampled.
    std::vector<int32_t> generate_mel_codes(
        const std::vector<int32_t> & text_tokens,
        const std::vector<float> & cond_latents,
        float temperature,
        int top_k,
        float top_p,
        float repetition_penalty,
        int max_mel_tokens,
        std::mt19937 & rng,
        std::vector<float> * decode_hiddens = nullptr);


private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::xtts_v2
