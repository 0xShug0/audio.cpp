#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace engine::audio {

enum class Pcm16QuantizeMode {
    TruncateTowardZero,
    RoundToNearest,
};

void apply_preemphasis_in_place(float * mono_samples, size_t sample_count, float coefficient);
void apply_preemphasis_in_place(std::vector<float> & mono_samples, float coefficient);
std::vector<float> apply_preemphasis(std::vector<float> mono_samples, float coefficient);
std::vector<float> copy_or_zero_pad_samples_to_count(
    const std::vector<float> & samples,
    size_t output_sample_count);
std::vector<float> zero_pad_samples_to_multiple(
    const std::vector<float> & samples,
    int64_t sample_multiple);
void truncate_samples_to_count(std::vector<float> & samples, size_t max_sample_count);
void normalize_peak_to_unit_range_and_clamp_in_place(std::vector<float> & samples);
std::vector<float> reflect_pad_samples(
    const std::vector<float> & samples,
    int64_t left_pad_samples,
    int64_t right_pad_samples);
float root_mean_square_or_throw(const std::vector<float> & samples);
std::vector<int16_t> float_to_pcm16_clipped(
    const std::vector<float> & samples,
    Pcm16QuantizeMode quantize_mode);
std::vector<float> pcm16_to_float_unit_range(const std::vector<int16_t> & pcm_samples);

}  // namespace engine::audio
