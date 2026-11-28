#pragma once

#include <cstdint>
#include <deque>
#include <random>
#include <vector>

namespace engine::models::zonos3 {

// Registry defaults (ZONOS3 model/src/zonos3/registries/generation.json, sampling v2).
struct Zonos3SamplingOptions {
    float temperature = 0.75f;  // <= 0 decodes greedily
    float top_p = 0.69f;
    float min_p = 0.0f;
    float repetition_penalty = 2.4f;
    int top_k = 80;
    int repetition_window = 96;
    int repetition_codebooks = 6;  // -1 = every codebook
    uint32_t seed = 0;
};

// ZONOS3's per-codebook sampler: pad and foreign special ids masked, repetition penalty over each codebook's recent
// ordinary codes, temperature, top-k with ties kept, softmax, top-p on the mass before each token, min-p, then an
// inverse-CDF draw from one shared std::mt19937. Seeded outputs match the ZONOS3 runtime.
class Zonos3Sampler {
public:
    Zonos3Sampler(const Zonos3SamplingOptions & options, int codebooks, int vocab, int codebook_size, int pad_id,
                  int stop_id);
    int sample(const float * logits, int codebook);  // logits has `vocab` entries
    void accept(const std::vector<int32_t> & frame);

private:
    Zonos3SamplingOptions options_;
    std::mt19937 rng_;
    int codebooks_;
    int vocab_;
    int codebook_size_;
    int pad_id_;
    int stop_id_;
    std::vector<std::deque<int32_t>> history_;
    int generated_ = 0;  // frames accepted so far
    std::vector<float> logit_, scratch_, p_;
    std::vector<int> candidates_, order_;
    std::vector<char> keep_;
    std::vector<uint32_t> seen_;
    uint32_t seen_generation_ = 0;
};

}  // namespace engine::models::zonos3
