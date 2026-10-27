#pragma once

// LFM3-Audio ASR encoder: a conv downsampler (x8 over time and mel), an LFM2
// stack with sliding-window attention, and an MLP projector to the backbone
// width.

#include "engine/community_models/lfm3_audio/assets.h"
#include "engine/community_models/lfm3_audio/features.h"
#include "engine/community_models/lfm3_audio/weights.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace engine::community_models::lfm3_audio {

struct Lfm3AudioEmbeddings {  // row-major [tokens][2048]
    std::vector<float> values;
    int64_t tokens = 0;
    int64_t hidden = 0;
};

struct Lfm3EncoderTaps {
    std::vector<float> downsample /*[N,768]*/, encoder /*[N,768]*/;
};

class Lfm3AudioEncoderRuntime {
public:
    Lfm3AudioEncoderRuntime(std::shared_ptr<const assets::TensorSource> model, const Lfm3EncoderConfig & config,
                            core::ExecutionContext & execution, const Lfm3WeightOptions & options);
    ~Lfm3AudioEncoderRuntime();
    // One embedding per 8 feature frames, rounded up. The graph stays for the next call, which reuses it for the
    // same frame count; a pass longer than 30 s frees its compute buffer afterwards. With LFM3_AUDIO_DUMP_DIR set
    // it writes the downsample, encoder and projector dumps.
    Lfm3AudioEmbeddings encode(const Lfm3Features & features, Lfm3EncoderTaps * taps = nullptr);
    [[nodiscard]] int64_t graphs_built() const noexcept;  // lets tests check graph reuse

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::community_models::lfm3_audio
