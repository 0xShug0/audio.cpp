#pragma once

#include "engine/models/kokoro_tts/predictor.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

typedef struct ggml_backend * ggml_backend_t;

namespace kokoro_ggml {

struct KokoroIstftNetDecoderCapacityContract {
    int64_t decoder_frames = 0;
    int64_t conditioning_frames = 0;
};

class KokoroIstftNetDecoderRuntime {
public:
    // `rknn_generator_path` (optional) routes the generator stage to a pre-compiled
    // RKNN engine; see engine/models/kokoro_tts/rknn_generator.h.
    // `rknn_generator_overlap` / `rknn_generator_gain` come from exec_plan.params on that
    // sidecar's window walk (overlap-blend / per-window loudness correction); both
    // default to the shipping walk.
    KokoroIstftNetDecoderRuntime(
        std::shared_ptr<const KokoroWeights> weights,
        ggml_backend_t backend,
        int n_threads,
        bool use_device_backend,
        uint64_t rng_seed,
        KokoroIstftNetDecoderCapacityContract contract,
        const std::string & rknn_generator_path = {},
        int64_t rknn_generator_overlap = 0,
        bool rknn_generator_gain = false,
        bool neon_generator = false);
    ~KokoroIstftNetDecoderRuntime();

    KokoroIstftNetDecoderRuntime(const KokoroIstftNetDecoderRuntime &) = delete;
    KokoroIstftNetDecoderRuntime & operator=(const KokoroIstftNetDecoderRuntime &) = delete;

    void prepare(KokoroIstftNetDecoderCapacityContract contract);
    void set_seed(uint64_t seed);

    std::vector<float> decode(
        const PredictorOutputs & predictor,
        const std::vector<float> & ref_s);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace kokoro_ggml
