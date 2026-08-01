#pragma once

#include "engine/framework/sampling/torch_random.h"
#include "engine/models/higgs_audio_tts/codebooks.h"

#include <cstdint>
#include <optional>
#include <random>
#include <vector>

namespace engine::models::higgs_audio_tts {

constexpr int64_t kHiggsMaxTopK = 1026;

using HiggsCudaSamplingPolicy = engine::sampling::TorchCudaSamplingPolicy;

struct HiggsSamplingOptions {
    float temperature = 1.0F;
    std::optional<float> top_p;
    std::optional<int64_t> top_k;
    bool has_seed = false;
    uint64_t seed = 0;
    HiggsCudaSamplingPolicy cuda_policy;
    std::mt19937 * fallback_rng = nullptr;
};

struct HiggsSamplerState {
    int64_t num_codebooks = 0;
    int64_t delay_count = 0;
    std::optional<int64_t> eoc_countdown;
    bool generation_done = false;
    int64_t step_count = 0;
    std::vector<int32_t> last_codes;
};

class HiggsCodebookSampler {
public:
    explicit HiggsCodebookSampler(int64_t num_codebooks, int64_t codebook_vocab_size);

    HiggsSamplerState make_state() const;
    const std::vector<int32_t> & step(const float * logits,
                                      int64_t logits_count,
                                      HiggsSamplerState & state,
                                      HiggsSamplingOptions & options);

private:
    int64_t num_codebooks_ = 0;
    int64_t codebook_vocab_size_ = 0;
    std::vector<float> scratch_scores_;
    std::vector<float> scratch_probs_;
    std::vector<int64_t> scratch_order_;
    std::vector<int64_t> scratch_kept_;
    std::vector<int32_t> scratch_codes_;
};

} // namespace engine::models::higgs_audio_tts
