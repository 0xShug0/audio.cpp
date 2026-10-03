#pragma once

// LFM2.5-Audio audio input path: NeMo log-mel features, the FastConformer
// encoder and the adapter MLP that maps encoder frames into the backbone's
// embedding space.

#include "engine/community_models/lfm2_audio/assets.h"
#include "engine/framework/audio/nemo_mel_frontend.h"
#include "engine/framework/core/execution_context.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace engine::community_models::lfm2_audio {

// Feature-major log-mel features, [n_mels][frames].
struct Lfm2AudioFeatures {
    int64_t n_mels = 0;
    int64_t frames = 0;
    std::vector<float> values;
};

// Adapter output, row-major [tokens][hidden_size].
struct Lfm2AudioEmbeddings {
    int64_t tokens = 0;
    int64_t hidden_size = 0;
    std::vector<float> values;
};

// The checkpoint's preprocessor, liquid-audio's copy of NeMo's
// AudioToMelSpectrogramPreprocessor (model/conformer/processor.py). It has no
// weights.
class Lfm2AudioFeatureExtractor {
public:
    Lfm2AudioFeatureExtractor(int64_t n_mels, int threads);

    // Mono 16 kHz samples in. Like the reference, the frame count is
    // samples / hop + 1 and the last frame is zero.
    [[nodiscard]] Lfm2AudioFeatures extract(const std::vector<float> & samples) const;

private:
    int64_t n_mels_;
    size_t threads_;
    audio::NemoMelFrontend frontend_;
};

class Lfm2FastConformerEncoderRuntime {
public:
    Lfm2FastConformerEncoderRuntime(
        std::shared_ptr<const assets::TensorSource> source,
        const Lfm2FastConformerEncoderConfig & config,
        core::ExecutionContext & execution);
    ~Lfm2FastConformerEncoderRuntime();

    Lfm2FastConformerEncoderRuntime(const Lfm2FastConformerEncoderRuntime &) = delete;
    Lfm2FastConformerEncoderRuntime & operator=(const Lfm2FastConformerEncoderRuntime &) = delete;

    // One embedding per 8 feature frames, rounded up. The graph and its
    // compute buffer stay for the next call, which reuses the graph when the
    // frame count is the same. A chunk longer than 30 s frees its buffer
    // afterwards, and so does a call that fails.
    Lfm2AudioEmbeddings encode(const Lfm2AudioFeatures & features);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::community_models::lfm2_audio
