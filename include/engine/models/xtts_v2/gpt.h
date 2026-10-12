#pragma once

// GPT-2 autoregressive mel-code generator for xTTS v2.
//
// Pipeline:
//   1. Reference audio -> 80-band log-mel -> ConditioningEncoder -> Perceiver -> 32 cond latents
//   2. Text -> BPE tokenize -> text tokens
//   3. GPT-2 AR: [cond_latents | text_embs | start_mel] -> mel codes
//   4. GPT-2 forward (return_latent): text + mel codes -> hidden states [d_model, T_mel]

#include "engine/framework/core/backend.h"
#include "engine/models/xtts_v2/assets.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::xtts_v2 {

// GPT-2 inference engine for xTTS v2.
class XttSv2GptEngine {
public:
    XttSv2GptEngine() = default;
    ~XttSv2GptEngine();

    XttSv2GptEngine(const XttSv2GptEngine &) = delete;
    XttSv2GptEngine & operator=(const XttSv2GptEngine &) = delete;

    // Load from GGUF. Returns true on success.
    bool load(const std::filesystem::path & gpt_path,
              const std::filesystem::path & vocoder_path,
              int n_threads, bool use_gpu);

    // Compute conditioning latents and speaker embedding from reference audio.
    // ref_pcm: mono 22050 Hz float32 PCM.
    // Returns true on success. cond_latents: [d_model, n_latents].
    bool compute_conditioning(const float * ref_pcm, int ref_n_samples,
                              std::vector<float> & cond_latents);

    // Autoregressive mel-code generation.
    // text_tokens: BPE token IDs.
    // cond_latents: [d_model, n_latents] conditioning from compute_conditioning.
    // Returns mel code tokens (each in [0, 1024]).
    std::vector<int32_t> generate_mel_codes(
        const std::vector<int32_t> & text_tokens,
        const std::vector<float> & cond_latents,
        float temperature = 0.75f,
        int top_k = 50,
        float top_p = 0.85f,
        float repetition_penalty = 5.0f,
        int max_mel_tokens = 1500);

    // Extract GPT hidden latents for vocoding.
    // text_tokens: BPE token IDs.
    // mel_codes: generated mel code tokens.
    // Returns [d_model, T_mel] hidden states.
    std::vector<float> extract_latents(
        const std::vector<int32_t> & text_tokens,
        const std::vector<int32_t> & mel_codes,
        const std::vector<float> & cond_latents = {});

    const XttSv2HyperParams & hyper_params() const;
    const XttSv2BpeVocab & vocab() const;

private:
    struct Impl;
    Impl* impl_ = nullptr;
};

}  // namespace engine::models::xtts_v2

