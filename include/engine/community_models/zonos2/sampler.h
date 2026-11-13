#pragma once

#include <cstdint>
#include <deque>
#include <random>
#include <vector>

namespace engine::models::zonos2 {

// Defaults of ZONOS2's TTS request (python/zonos2/message/tts.py).
struct Zonos2SamplingOptions {
    float temperature = 1.15f;  // <= 0 decodes greedily
    int top_k = 106;
    float top_p = 0.0f;  // 0 disables
    float min_p = 0.18f;
    float repetition_penalty = 1.2f;
    int repetition_window = 50;    // generated frames of history; 0 disables
    int repetition_codebooks = 8;  // -1 = every codebook
    uint32_t seed = 0;
};

// ZONOS2's per-codebook sampler (python/zonos2/tts/sampler.py sample_tts): the repetition penalty applies once per
// distinct ordinary code in the codebook's recent generated history, then greedy argmax or temperature, top-k (ties
// at the threshold kept), softmax, top-p on the mass before each token, min-p, and an inverse-CDF draw from one
// std::mt19937 per generation. No id is masked.
class Zonos2Sampler {
public:
    Zonos2Sampler(const Zonos2SamplingOptions & options, int codebooks, int vocab, int codebook_size);
    int sample(const float * logits, int codebook);  // logits has `vocab` entries
    void accept(const std::vector<int32_t> & frame);

private:
    Zonos2SamplingOptions options_;
    std::mt19937 rng_;
    int vocab_;
    int codebook_size_;
    int penalized_codebooks_;
    std::vector<std::deque<int32_t>> history_;
    std::vector<float> logit_, scratch_;
    std::vector<double> p_;
    std::vector<int> candidates_, order_;
    std::vector<uint32_t> seen_;
    uint32_t seen_generation_ = 0;
};

}  // namespace engine::models::zonos2
