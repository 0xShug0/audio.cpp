#pragma once

#include "engine/models/voxmel_tts/assets.h"
#include "engine/framework/core/attention_fallback.h"
#include "engine/framework/core/execution_context.h"

#include <memory>
#include <vector>

namespace engine::models::voxmel_tts {

struct SynthesisOptions {
    int64_t steps = 32, up_steps = 32, min_frames = 15, max_frames = 1407;
    float text_guidance = 3.5f, speaker_guidance = 3.0f;
    uint64_t seed = 0;
};

// Torchaudio-compatible magnitude/HTK frontend. Filterbank and window are
// exported constants; FFT and projection use the existing framework DSP.
std::vector<float> reference_mel(const ModelAssets & assets, const std::vector<float> & mono, size_t threads);

// Continuous velocity head. Conditioning and timestep projections are prepared
// once per sequence; device sampling retains the latent across all Euler steps.
// Explicit noise is also the independent replay boundary used by parity tests.
class VoxMelFlowRuntime {
public:
    VoxMelFlowRuntime(const ModelAssets & assets, core::ExecutionContext & execution,
        assets::TensorStorageType storage, const std::string & prefix, int64_t condition_size);
    ~VoxMelFlowRuntime();
    std::vector<float> sample(const std::vector<float> & conditioning, int64_t rows, int64_t branches,
        const std::vector<float> & noise, int64_t steps, float text_guidance = 0, float speaker_guidance = 0);
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

// Produces fixed Transformer conditioning for a temporal/frequency residual.
// Positions within a tier are evaluated together; the two tiers remain ordered.
class VoxMelTierRuntime {
public:
    VoxMelTierRuntime(const ModelAssets & assets, core::ExecutionContext & execution,
        assets::TensorStorageType storage, core::AttentionPreference attention, size_t tier);
    ~VoxMelTierRuntime();
    std::vector<float> condition(const std::vector<float> & known);
    std::vector<float> denormalize(const std::vector<float> & detail) const;
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

// Model orchestration; the Qwen backbone, flow-loop lifecycle and Vocos stay
// with their framework runtimes. Calls are serialized by the owning session.
class VoxMelRuntime {
public:
    VoxMelRuntime(std::shared_ptr<const ModelAssets> assets, core::ExecutionContext & execution,
        assets::TensorStorageType storage, core::AttentionPreference attention);
    ~VoxMelRuntime();

    std::vector<float> synthesize(const std::vector<int32_t> & text,
        const std::vector<int32_t> & reference_text, const std::vector<float> & reference_mel,
        const SynthesisOptions & options);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace engine::models::voxmel_tts
