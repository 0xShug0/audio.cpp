#pragma once

#include "engine/community_models/kitten_tts/predictor.h"

#include <cstdint>
#include <vector>

typedef struct ggml_backend *ggml_backend_t;

namespace engine::models::kitten_tts {

struct KittenStyleTTS2DecoderCapacityContract {
    int64_t decoder_frames = 0;
    int64_t conditioning_frames = 0;
};

class KittenStyleTTS2DecoderRuntime {
  public:
    KittenStyleTTS2DecoderRuntime(std::shared_ptr<const KittenWeights> weights, ggml_backend_t backend, int n_threads,
                                bool use_device_backend, uint64_t rng_seed, KittenStyleTTS2DecoderCapacityContract contract);
    ~KittenStyleTTS2DecoderRuntime();

    KittenStyleTTS2DecoderRuntime(const KittenStyleTTS2DecoderRuntime &) = delete;
    KittenStyleTTS2DecoderRuntime &operator=(const KittenStyleTTS2DecoderRuntime &) = delete;

    void prepare(KittenStyleTTS2DecoderCapacityContract contract);

    std::vector<float> decode(const PredictorOutputs &predictor, const std::vector<float> &ref_s);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace engine::models::kitten_tts
