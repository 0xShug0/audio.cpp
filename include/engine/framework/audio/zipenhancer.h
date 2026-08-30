#pragma once

#include "engine/framework/core/backend.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

namespace engine::audio {

struct ZipEnhancerModelState;

struct ZipEnhancerWaveformOutput {
    int sample_rate = 16000;
    std::vector<float> samples;
};

// Chunk-join and output-length policy for ZipEnhancerModel::denoise_mono_16k.
struct ZipEnhancerOptions {
    // Length in samples (at 16 kHz) of the linear crossfade applied where two
    // consecutive 2 s analysis windows meet. Clamped to the 8000-sample (500 ms)
    // overlap between windows. 0 restores the legacy hard splice, where the
    // first half of the overlap comes from the earlier chunk and the second half
    // from the later one with no fade between them.
    int64_t chunk_crossfade_samples = 8000;
    // When true denoise_mono_16k returns exactly as many samples as it was
    // given. When false the un-segmented path returns floor(n / 100) * 100
    // samples, which is what the upstream reference implementation emits and
    // what the reference fixtures under tests/unittests/assets were captured
    // with.
    bool match_input_length = true;
};

// Segmentation geometry denoise_mono_16k uses for a given input length. Exposed
// so the length and coverage arithmetic can be exercised without model weights.
struct ZipEnhancerChunkPlan {
    int64_t window_samples = 0;
    int64_t stride_samples = 0;
    int64_t padded_samples = 0;
    int64_t output_samples = 0;
    bool segmented = false;
};

ZipEnhancerChunkPlan zipenhancer_chunk_plan(int64_t input_samples, const ZipEnhancerOptions & options) noexcept;

// Rising crossfade weight at `position` inside an `overlap_samples`-long chunk
// join, using a linear ramp of `fade_samples` centred in the overlap. The pair
// (position, overlap_samples - 1 - position) always sums to exactly 1.
float zipenhancer_chunk_fade_weight(int64_t position, int64_t overlap_samples, int64_t fade_samples) noexcept;

// Weight the chunk starting at `segment_start` contributes to the output sample
// at `segment_start + offset_in_segment`.
float zipenhancer_segment_weight(
    int64_t offset_in_segment,
    int64_t segment_start,
    const ZipEnhancerChunkPlan & plan,
    const ZipEnhancerOptions & options) noexcept;

// Total overlap-add weight every padded output sample receives. Every entry
// below plan.output_samples must be strictly positive or the join leaves a hole.
std::vector<float> zipenhancer_chunk_weights(const ZipEnhancerChunkPlan & plan, const ZipEnhancerOptions & options);

class ZipEnhancerModel {
public:
    static ZipEnhancerModel load_from_directory(const std::filesystem::path & model_dir);
    static ZipEnhancerModel load_from_directory(const std::filesystem::path & model_dir, const core::BackendConfig & backend_config);

    ZipEnhancerModel();
    ~ZipEnhancerModel();
    ZipEnhancerModel(ZipEnhancerModel &&) noexcept;
    ZipEnhancerModel & operator=(ZipEnhancerModel &&) noexcept;
    ZipEnhancerModel(const ZipEnhancerModel &) = delete;
    ZipEnhancerModel & operator=(const ZipEnhancerModel &) = delete;

    ZipEnhancerWaveformOutput denoise_mono_16k(
        const std::vector<float> & waveform,
        const ZipEnhancerOptions & options = ZipEnhancerOptions{}) const;

private:
    explicit ZipEnhancerModel(std::shared_ptr<const ZipEnhancerModelState> state);

    std::shared_ptr<const ZipEnhancerModelState> state_;
};

}  // namespace engine::audio
