#pragma once

#include "engine/framework/runtime/session.h"
#include <cstddef>
#include <cstdint>
#include <vector>
#include "engine/community_models/lfm3_audio/assets.h"
#include "engine/community_models/lfm3_audio/weights.h"
#include "engine/framework/core/execution_context.h"
#include <memory>

namespace engine::community_models::lfm3_audio {

// Mixdown and resampling with libsoxr HQ (librosa soxr_hq, as HF loads audio); the torchaudio sinc resampler
// only when libsoxr is absent.
std::vector<float> lfm3_audio_mono_16k(const runtime::AudioBuffer & audio);
int64_t lfm3_feature_frames(int64_t samples);  // 1 + ceil(max(n-400,0)/160)
int64_t lfm3_audio_tokens(int64_t frames);     // ceil(frames/8)

struct Lfm3Features {  // feature-major [128][frames]
    std::vector<float> values;
    int64_t frames = 0;
};

class Lfm3FeatureExtractor {
public:
    Lfm3FeatureExtractor();
    // Right-pads to whole frames (at least one).
    [[nodiscard]] Lfm3Features extract(const std::vector<float> & mono16k, size_t threads = 0) const;
};

struct Lfm3AudioEmbeddings {  // row-major [tokens][2048]
    std::vector<float> values;
    int64_t tokens = 0;
    int64_t hidden = 0;
};

class Lfm3AudioFrontend {
public:
    Lfm3AudioFrontend(Lfm3EncoderWeights weights, const Lfm3EncoderConfig & config,
                            core::ExecutionContext & execution, const Lfm3WeightOptions & options);
    ~Lfm3AudioFrontend();
    // One embedding per 8 feature frames, rounded up. The graph stays for the next call, which reuses it for the
    // same frame count; a pass longer than 30 s frees its compute buffer afterwards.
    Lfm3AudioEmbeddings encode(const Lfm3Features & features);
    Lfm3AudioEmbeddings encode_audio(const std::vector<float> & mono16k);
    [[nodiscard]] int64_t graphs_built() const noexcept;  // lets tests check graph reuse

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::community_models::lfm3_audio
