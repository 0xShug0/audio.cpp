#pragma once

#include "engine/framework/core/backend.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

namespace engine::audio {

struct FlashSrWeights;
class FlashSrGraph;

struct FlashSrOutput {
    int sample_rate = 48000;
    std::vector<float> samples;
};

// Absolute peak the FlashSR output is never allowed to exceed.
inline constexpr float kFlashSrPeakCeiling = 0.9990000128746033f;

struct FlashSrOptions {
    // Restore the input waveform's peak on the output instead of normalising
    // every file to `peak_ceiling`. With this off a single loud sample sets the
    // level of the whole file and the input's own level is discarded — a -40 dBFS
    // whisper and a -3 dBFS shout both come back at -0.009 dBFS. Set false for
    // bit-exact parity with the upstream reference implementation, which is what
    // the reference fixtures under tests/unittests/assets were captured with.
    bool preserve_input_level = true;
    // Safety limit, not a target. The output is scaled down only if restoring
    // the input peak would push it above this value.
    float peak_ceiling = kFlashSrPeakCeiling;
};

// Gain applied to a FlashSR output whose absolute peak is `output_peak`, given
// an input whose absolute peak is `input_peak`. Returns 0 for a silent output
// (silence in, silence out) and never lets the result exceed
// `options.peak_ceiling`.
float flashsr_output_gain(float input_peak, float output_peak, const FlashSrOptions & options) noexcept;

class FlashSrModel {
public:
    static FlashSrModel load_from_directory(const std::filesystem::path & model_dir);
    static FlashSrModel load_from_directory(const std::filesystem::path & model_dir, const core::BackendConfig & backend_config);

    FlashSrModel();
    ~FlashSrModel();
    FlashSrModel(FlashSrModel &&) noexcept;
    FlashSrModel & operator=(FlashSrModel &&) noexcept;
    FlashSrModel(const FlashSrModel &) = delete;
    FlashSrModel & operator=(const FlashSrModel &) = delete;

    FlashSrOutput super_resolve_mono_16k(
        const std::vector<float> & waveform,
        const FlashSrOptions & options = FlashSrOptions{}) const;

private:
    explicit FlashSrModel(std::shared_ptr<FlashSrWeights> weights);

    std::shared_ptr<FlashSrWeights> weights_;
    mutable std::unique_ptr<FlashSrGraph> graph_;
};

}  // namespace engine::audio
