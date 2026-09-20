#pragma once

#include "engine/community_models/kitten_tts/predictor.h"

#include <cstdint>
#include <vector>

typedef struct ggml_backend *ggml_backend_t;

namespace engine::models::kitten_tts {

struct KittenDecoderCapacityContract {
    int64_t decoder_frames = 0;
    int64_t conditioning_frames = 0;
};

class KittenDecoderRuntime {
  public:
    KittenDecoderRuntime(std::shared_ptr<const KittenWeights> weights, ggml_backend_t backend, int n_threads,
                         bool use_device_backend, uint64_t rng_seed, KittenDecoderCapacityContract contract);
    ~KittenDecoderRuntime();

    KittenDecoderRuntime(const KittenDecoderRuntime &) = delete;
    KittenDecoderRuntime &operator=(const KittenDecoderRuntime &) = delete;

    void prepare(KittenDecoderCapacityContract contract);

    std::vector<float> decode(const PredictorOutputs &predictor, const std::vector<float> &ref_s);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace engine::models::kitten_tts
