#pragma once

#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/transformers/lfm_decoder_runtime.h"

namespace engine::modules {

struct Lfm3AudioEncoderConfig {
    LfmDecoderConfig decoder;
    int64_t mel_bins = 128;
    int64_t conv_channels = 0;
    int64_t output_size = 0;
    int64_t sliding_window = 0;
    float projector_norm_eps = 1e-5f;
};

struct Lfm3AudioEncoderWeights {
    Conv2dWeights conv_in, conv_mid, lin_mid, conv_end, lin_end;
    LinearWeights lin_out;
    LfmDecoderWeights decoder;
    NormWeights projector_norm;
    LinearWeights projector_1, projector_2;
};

struct Lfm3AudioEncoding {
    core::TensorValue projected; // borrowed F32 [1, tokens, output_size]
    int64_t feature_frames = 0;
    int64_t tokens = 0;
};

// Resolved weights and execution context remain caller-owned. Feature extraction
// and checkpoint interpretation belong to the model, not this numerical runtime.
class Lfm3AudioEncoderRuntime {
public:
    Lfm3AudioEncoderRuntime(Lfm3AudioEncoderConfig config, Lfm3AudioEncoderWeights weights,
                           core::ExecutionContext & execution, size_t graph_cache_slots = 4);
    ~Lfm3AudioEncoderRuntime();
    Lfm3AudioEncoderRuntime(const Lfm3AudioEncoderRuntime &) = delete;
    Lfm3AudioEncoderRuntime & operator=(const Lfm3AudioEncoderRuntime &) = delete;

    // Feature-major [mel_bins, frames]. Output survives until the next encode(),
    // release_workspace(), or destruction. Calls and consumers must be sequential.
    Lfm3AudioEncoding encode(const std::vector<float> & features, int64_t frames);
    void release_workspace();
    size_t graphs_built() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace engine::modules
