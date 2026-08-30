#pragma once

#include "engine/framework/core/backend.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace engine::audio {

struct RnnoiseWeights;

struct RnnoiseConfig {
    int64_t feature_size = 65;
    int64_t conv1_channels = 128;
    int64_t conv2_channels = 384;
    int64_t gru_size = 384;
    int64_t gain_bands = 32;
};

struct RnnoiseFrameOutput {
    std::vector<float> gains;
    float vad = 0.0f;
};

struct RnnoiseSequenceOutput {
    std::vector<float> gains;
    std::vector<float> vad;
    int64_t frames = 0;
    int64_t gain_bands = 0;
};

struct RnnoiseWaveformOutput {
    int sample_rate = 48000;
    std::vector<float> samples;
    std::vector<float> vad;
};

// Group delay of the RNNoise analysis/synthesis pipeline, in samples at 48 kHz.
// The synthesis stage inverse-transforms the *previous* frame's spectrum, so an
// output sample at index n reconstructs the input sample at index n - 960
// (20 ms). DeepFilterNet2 crops its own 480-sample delay the same way.
inline constexpr int64_t kRnnoiseOutputDelaySamples = 960;

struct RnnoiseProcessOptions {
    // Pad the input tail by kRnnoiseOutputDelaySamples and crop the same amount
    // off the front of the synthesis output, so that output[n] lines up with
    // input[n] and the last 20 ms of input still reaches the output. Set false
    // for bit-exact parity with the upstream reference implementation, which
    // leaves the delay in — that is what the reference fixtures under
    // tests/unittests/assets were captured with.
    bool compensate_output_delay = true;
};

// Frame, padding and crop arithmetic process_mono_48k uses for a given input
// length. Exposed so the alignment can be exercised without model weights.
struct RnnoiseAlignmentPlan {
    int64_t frames = 0;
    int64_t padded_samples = 0;
    int64_t crop_offset = 0;
    int64_t output_samples = 0;
    int64_t vad_frames = 0;
};

RnnoiseAlignmentPlan rnnoise_alignment_plan(int64_t input_samples, const RnnoiseProcessOptions & options) noexcept;

class RnnoiseModel {
public:
    static RnnoiseModel load_from_safetensors(const std::filesystem::path & checkpoint_path);
    static RnnoiseModel load_from_safetensors(const std::filesystem::path & checkpoint_path, const core::BackendConfig & backend_config);

    RnnoiseModel();
    ~RnnoiseModel();
    RnnoiseModel(RnnoiseModel &&) noexcept;
    RnnoiseModel & operator=(RnnoiseModel &&) noexcept;
    RnnoiseModel(const RnnoiseModel &) = delete;
    RnnoiseModel & operator=(const RnnoiseModel &) = delete;

    const RnnoiseConfig & config() const noexcept;
    const std::filesystem::path & source_path() const noexcept;

    RnnoiseSequenceOutput infer_features(
        const std::vector<float> & features,
        int64_t frames,
        int64_t feature_size) const;
    RnnoiseWaveformOutput process_mono_48k(
        const std::vector<float> & waveform,
        const RnnoiseProcessOptions & options = RnnoiseProcessOptions{}) const;

    std::unique_ptr<class RnnoiseStreamingSession> create_streaming_session() const;

private:
    explicit RnnoiseModel(std::shared_ptr<const RnnoiseWeights> weights);

    std::shared_ptr<const RnnoiseWeights> weights_;
};

class RnnoiseStreamingSession {
public:
    explicit RnnoiseStreamingSession(std::shared_ptr<const RnnoiseWeights> weights);
    ~RnnoiseStreamingSession();
    RnnoiseStreamingSession(RnnoiseStreamingSession &&) noexcept;
    RnnoiseStreamingSession & operator=(RnnoiseStreamingSession &&) noexcept;
    RnnoiseStreamingSession(const RnnoiseStreamingSession &) = delete;
    RnnoiseStreamingSession & operator=(const RnnoiseStreamingSession &) = delete;

    void reset();
    RnnoiseFrameOutput process_frame(const float * features, int64_t feature_size);
    float process_audio_frame(const float * input, float * output, int64_t samples);

private:
    struct State;

    std::shared_ptr<const RnnoiseWeights> weights_;
    std::unique_ptr<State> state_;
};

}  // namespace engine::audio
