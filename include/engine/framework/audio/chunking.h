#pragma once

#include "engine/framework/runtime/session.h"

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace engine::audio {

enum class AudioChunkPadMode {
    Zero,
    Reflect,
};

enum class AudioChunkTailAlignment {
    Start,
    Center,
};

enum class AudioChunkCounterMode {
    SharedAcrossLanes,
    PerLane,
};

enum class AudioChunkMode {
    Auto,
    Fixed,
    QuietEnergy,
    Vad,
    None,
};

struct AudioChunkSpec {
    int64_t chunk_samples = 0;
    int64_t hop_samples = 0;
    AudioChunkPadMode pad_mode = AudioChunkPadMode::Zero;
    AudioChunkTailAlignment tail_alignment = AudioChunkTailAlignment::Start;
    int64_t reflect_min_valid_samples = 0;
};

struct AudioChunkSpan {
    int64_t index = 0;
    int64_t output_start_sample = 0;
    int64_t valid_samples = 0;
    int64_t copy_start_sample = 0;
    int64_t valid_start_in_chunk = 0;
};

struct VadAudioChunkOptions {
    int64_t max_chunk_samples = 0;
    int64_t merge_gap_samples = 0;
    int64_t padding_samples = 0;
};

struct QuietEnergyAudioChunkOptions {
    int64_t chunk_samples = 0;
    int64_t boundary_context_samples = 0;
    int64_t min_energy_window_samples = 0;
};

// How two consecutive output chunks are joined into one waveform.
//
// Concat      — raw splice, the historical behaviour. Bit-exact, but a step at the
//               seam is a broadband impulse at 20*log10(step) dBFS.
// EqualPower  — sin/cos pair. cos^2 + sin^2 == 1, so it preserves *power* through
//               the overlap. Correct for the uncorrelated content either side of a
//               text-chunk seam; +3.01 dB at the midpoint if the two sides happen
//               to be identical.
// CosSquared  — cos^2/sin^2 pair. The gains sum to exactly 1, so it preserves
//               *amplitude* for correlated content and dips ~1.25 dB for
//               uncorrelated content.
// Auto        — measure the seam first and only crossfade a real discontinuity.
//               A join that continues the waveform, or that lands in silence, is
//               spliced and is bit-exact with Concat.
enum class AudioChunkJoinMode {
    Concat,
    EqualPower,
    CosSquared,
    Auto,
};

struct AudioChunkJoinSpec {
    AudioChunkJoinMode mode = AudioChunkJoinMode::Auto;
    // 10 ms. F4.15 puts the practical knee at 5-20 ms: a crossfade drops the peak
    // spectral energy of a seam step by roughly 20*log10(1/L), so 10 ms at 24 kHz
    // buys about -47 dB while costing at most 10 ms of duration per seam.
    float cross_fade_seconds = 0.010F;
    // Auto declines to crossfade at or below this step. 0.002 is -54 dBFS, which
    // is already quieter than a 0.1 step (-20 dBFS) crossfaded over 20 ms. A seam
    // landing in the inter-sentence silence a text-chunk split produces is far
    // below it, so Auto leaves such joins untouched.
    float silent_seam_step = 0.002F;
    // Auto also declines when the step is within reach of the ordinary
    // sample-to-sample slew either side of the join, because that is what a join
    // continuing the same waveform looks like — an accumulating streaming buffer,
    // or a vocoder emitting consecutive frames. Only a step that stands out from
    // its neighbourhood by this factor is treated as a discontinuity.
    float seam_slew_ratio = 2.0F;
    // Samples inspected either side of the seam to establish that slew. Never
    // spans the seam itself.
    int64_t seam_slew_window_frames = 64;
};

std::vector<AudioChunkSpan> plan_audio_chunks(int64_t input_samples, const AudioChunkSpec & spec);

AudioChunkMode parse_audio_chunk_mode(
    const std::unordered_map<std::string, std::string> & options);

std::optional<float> parse_audio_chunk_seconds_override(
    const std::unordered_map<std::string, std::string> & options);

std::vector<runtime::TimeSpan> plan_vad_audio_chunks(
    const std::vector<runtime::SpeechSegment> & segments,
    int64_t audio_samples,
    const VadAudioChunkOptions & options);

std::vector<runtime::TimeSpan> plan_vad_audio_chunks(
    const runtime::AudioBuffer & audio,
    runtime::IOfflineVoiceTaskSession & vad_session,
    const VadAudioChunkOptions & options);

std::vector<runtime::TimeSpan> plan_quiet_energy_audio_chunks(
    const std::vector<float> & mono_samples,
    const QuietEnergyAudioChunkOptions & options);

runtime::AudioBuffer slice_audio_buffer(
    const runtime::AudioBuffer & audio,
    const runtime::TimeSpan & span);

AudioChunkJoinMode parse_audio_chunk_join_mode(const std::string & value);

AudioChunkJoinSpec parse_audio_chunk_join_spec(
    const std::unordered_map<std::string, std::string> & options);

float measure_chunk_seam_step(
    const runtime::AudioBuffer & previous,
    const runtime::AudioBuffer & next);

float measure_chunk_edge_slew(
    const runtime::AudioBuffer & previous,
    const runtime::AudioBuffer & next,
    int64_t window_frames);

bool chunk_seam_is_discontinuous(
    const runtime::AudioBuffer & previous,
    const runtime::AudioBuffer & next,
    const AudioChunkJoinSpec & spec);

int64_t resolve_chunk_cross_fade_frames(
    int64_t previous_frames,
    int64_t next_frames,
    int sample_rate,
    float cross_fade_seconds);

void append_audio_chunk(
    runtime::AudioBuffer & dst,
    const runtime::AudioBuffer & src,
    const AudioChunkJoinSpec & spec);

std::vector<float> make_triangular_overlap_window(int64_t chunk_samples);
std::vector<float> make_linear_fade_window(int64_t chunk_samples, int64_t fade_samples);

void copy_interleaved_chunk_to_planar(
    std::vector<float> & output_planar,
    const std::vector<float> & input_interleaved,
    int64_t channels,
    int64_t input_frames,
    const AudioChunkSpan & span,
    const AudioChunkSpec & spec);

void copy_planar_chunk(
    std::vector<float> & output_planar,
    const std::vector<float> & input_planar,
    int64_t lanes,
    int64_t input_frames,
    const AudioChunkSpan & span,
    const AudioChunkSpec & spec);

void overlap_add_planar_chunk(
    std::vector<float> & output_planar,
    std::vector<float> & weights,
    const std::vector<float> & chunk_planar,
    int64_t lanes,
    int64_t output_frames,
    const AudioChunkSpan & span,
    const std::vector<float> & window,
    AudioChunkCounterMode counter_mode);

void normalize_overlap_added_planar(
    std::vector<float> & output_planar,
    const std::vector<float> & weights,
    int64_t lanes,
    int64_t output_frames,
    AudioChunkCounterMode counter_mode);

void append_chunk_word_timestamps(
    std::vector<runtime::WordTimestamp> & output,
    const std::vector<runtime::WordTimestamp> & chunk_words,
    const runtime::TimeSpan & chunk_span);

void append_chunk_word_timestamps(
    std::vector<runtime::WordTimestamp> & output,
    const std::vector<runtime::WordTimestamp> & chunk_words,
    const runtime::TimeSpan & source_span,
    const runtime::TimeSpan & keep_span);

void append_chunk_word_timestamps(
    std::vector<runtime::WordTimestamp> & output,
    const std::vector<runtime::WordTimestamp> & chunk_words,
    const runtime::TimeSpan & source_span,
    const runtime::TimeSpan & keep_span,
    int64_t source_sample_rate,
    int64_t timestamp_sample_rate);

void append_chunk_speech_metadata(
    runtime::TaskResult & output,
    const runtime::TaskResult & chunk_result,
    const runtime::TimeSpan & source_span,
    const runtime::TimeSpan & keep_span,
    int64_t source_sample_rate,
    int64_t timestamp_sample_rate);

}  // namespace engine::audio
