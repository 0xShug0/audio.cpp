#pragma once

#include "engine/models/kokoro_tts/predictor.h"

#include <cstdint>
#include <vector>

typedef struct ggml_backend * ggml_backend_t;

namespace kokoro_ggml {

struct KokoroIstftNetDecoderCapacityContract {
    int64_t decoder_frames = 0;
    int64_t conditioning_frames = 0;
};

class KokoroIstftNetDecoderRuntime {
public:
    KokoroIstftNetDecoderRuntime(
        std::shared_ptr<const KokoroWeights> weights,
        ggml_backend_t backend,
        int n_threads,
        bool use_device_backend,
        uint64_t rng_seed,
        KokoroIstftNetDecoderCapacityContract contract);
    ~KokoroIstftNetDecoderRuntime();

    KokoroIstftNetDecoderRuntime(const KokoroIstftNetDecoderRuntime &) = delete;
    KokoroIstftNetDecoderRuntime & operator=(const KokoroIstftNetDecoderRuntime &) = delete;

    void prepare(KokoroIstftNetDecoderCapacityContract contract);

    std::vector<float> decode(
        const PredictorOutputs & predictor,
        const std::vector<float> & ref_s);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace kokoro_ggml
