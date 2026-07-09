#include "engine/framework/audio/chunking.h"

#include "engine/framework/runtime/options.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::audio {
namespace {

void require_positive(int64_t value, const char * name) {
    if (value <= 0) {
        throw std::runtime_error(std::string("Audio chunker requires positive ") + name);
    }
}

int64_t reflect_index(int64_t index, int64_t length) {
    require_positive(length, "reflect length");
    if (length == 1) {
        return 0;
    }
    while (index < 0 || index >= length) {
        if (index < 0) {
            index = -index;
        } else {
            index = 2 * length - index - 2;
        }
    }
    return index;
}

bool should_reflect_pad(const AudioChunkSpan & span, const AudioChunkSpec & spec) {
    return spec.pad_mode == AudioChunkPadMode::Reflect &&
        (spec.reflect_min_valid_samples <= 0 ||
         span.valid_samples >= spec.reflect_min_valid_samples);
}

size_t planar_index(int64_t lane, int64_t frame, int64_t frames) {
    return static_cast<size_t>(lane * frames + frame);
}

size_t weight_index(AudioChunkCounterMode mode, int64_t lane, int64_t frame, int64_t output_frames) {
    return mode == AudioChunkCounterMode::SharedAcrossLanes
        ? static_cast<size_t>(frame)
        : planar_index(lane, frame, output_frames);
}

void validate_copy_shape(
    const std::vector<float> & input,
    int64_t lanes,
    int64_t input_frames,
    const std::vector<float> & output,
    const AudioChunkSpec & spec) {
    require_positive(lanes, "lanes");
    require_positive(input_frames, "input frames");
    require_positive(spec.chunk_samples, "chunk samples");
    if (static_cast<int64_t>(input.size()) != lanes * input_frames) {
        throw std::runtime_error("Audio chunker input size mismatch");
    }
    if (static_cast<int64_t>(output.size()) != lanes * spec.chunk_samples) {
        throw std::runtime_error("Audio chunker output size mismatch");
    }
}

void require_valid_time_span(
    const runtime::TimeSpan & span,
    int64_t audio_samples,
    const char * label) {
    if (span.start_sample < 0 || span.end_sample <= span.start_sample || span.end_sample > audio_samples) {
        throw std::runtime_error(std::string(label) + " span is outside audio bounds");
    }
}

runtime::TimeSpan padded_span(
    const runtime::TimeSpan & span,
    int64_t audio_samples,
    int64_t padding_samples) {
    runtime::TimeSpan out;
    out.start_sample = std::max<int64_t>(0, span.start_sample - padding_samples);
    out.end_sample = std::min<int64_t>(audio_samples, span.end_sample + padding_samples);
    return out;
}

}  // namespace

std::vector<AudioChunkSpan> plan_audio_chunks(int64_t input_samples, const AudioChunkSpec & spec) {
    require_positive(input_samples, "input samples");
    require_positive(spec.chunk_samples, "chunk samples");
    require_positive(spec.hop_samples, "hop samples");
    if (spec.hop_samples > spec.chunk_samples) {
        throw std::runtime_error("Audio chunker hop_samples must be <= chunk_samples");
    }

    std::vector<AudioChunkSpan> spans;
    int64_t start = 0;
    int64_t index = 0;
    while (start < input_samples) {
        const int64_t valid = std::min(spec.chunk_samples, input_samples - start);
        AudioChunkSpan span;
        span.index = index++;
        span.output_start_sample = start;
        span.valid_samples = valid;
        span.copy_start_sample = start;
        if (valid < spec.chunk_samples &&
            spec.tail_alignment == AudioChunkTailAlignment::Center) {
            span.copy_start_sample -= (spec.chunk_samples - valid) / 2;
        }
        span.valid_start_in_chunk = start - span.copy_start_sample;
        if (span.valid_start_in_chunk < 0 ||
            span.valid_start_in_chunk + span.valid_samples > spec.chunk_samples) {
            throw std::runtime_error("Audio chunker planned invalid chunk span");
        }
        spans.push_back(span);
        start += spec.hop_samples;
    }
    return spans;
}

AudioChunkMode parse_audio_chunk_mode(
    const std::unordered_map<std::string, std::string> & options) {
    const auto mode = runtime::find_option(options, {"audio_chunk_mode"});
    if (!mode.has_value() || *mode == "auto") {
        return AudioChunkMode::Auto;
    }
    if (*mode == "fixed") {
        return AudioChunkMode::Fixed;
    }
    if (*mode == "quiet_energy") {
        return AudioChunkMode::QuietEnergy;
    }
    if (*mode == "vad") {
        return AudioChunkMode::Vad;
    }
    if (*mode == "none") {
        return AudioChunkMode::None;
    }
    throw std::runtime_error("audio_chunk_mode must be auto, fixed, quiet_energy, vad, or none");
}

std::optional<float> parse_audio_chunk_seconds_override(
    const std::unordered_map<std::string, std::string> & options) {
    return runtime::parse_float_option(
        options,
        {"audio_chunk_seconds", "audio_chunk_duration_seconds", "audio_chunk_duration"});
}

std::vector<runtime::TimeSpan> plan_vad_audio_chunks(
    const std::vector<runtime::SpeechSegment> & segments,
    int64_t audio_samples,
    const VadAudioChunkOptions & options) {
    require_positive(audio_samples, "audio samples");
    require_positive(options.max_chunk_samples, "max VAD chunk samples");
    if (options.merge_gap_samples < 0) {
        throw std::runtime_error("Audio VAD chunker merge_gap_samples must be non-negative");
    }
    if (options.padding_samples < 0) {
        throw std::runtime_error("Audio VAD chunker padding_samples must be non-negative");
    }
    if (segments.empty()) {
        return {};
    }

    struct WorkSpan {
        runtime::TimeSpan padded;
        runtime::TimeSpan speech;
    };
    struct ChunkState {
        runtime::TimeSpan span;
        int64_t speech_end_sample = 0;
    };

    std::vector<WorkSpan> spans;
    spans.reserve(segments.size());
    for (const auto & segment : segments) {
        require_valid_time_span(segment.span, audio_samples, "Audio VAD chunker speech segment");
        spans.push_back(WorkSpan{
            padded_span(segment.span, audio_samples, options.padding_samples),
            segment.span,
        });
    }
    std::sort(spans.begin(), spans.end(), [](const WorkSpan & a, const WorkSpan & b) {
        if (a.padded.start_sample != b.padded.start_sample) {
            return a.padded.start_sample < b.padded.start_sample;
        }
        return a.padded.end_sample < b.padded.end_sample;
    });

    std::vector<ChunkState> states;
    for (const auto & item : spans) {
        auto span = item.padded;
        const auto speech = item.speech;
        while (span.start_sample < span.end_sample) {
            const auto start_chunk = [&]() {
                runtime::TimeSpan chunk;
                chunk.start_sample = span.start_sample;
                chunk.end_sample = std::min<int64_t>(span.end_sample, span.start_sample + options.max_chunk_samples);
                const int64_t speech_end = chunk.end_sample > speech.start_sample
                    ? std::min<int64_t>(speech.end_sample, chunk.end_sample)
                    : chunk.start_sample;
                states.push_back(ChunkState{chunk, speech_end});
                span.start_sample = chunk.end_sample;
            };
            if (!states.empty()) {
                auto & current = states.back();
                if (span.end_sample <= current.span.end_sample) {
                    current.speech_end_sample = std::max(current.speech_end_sample, speech.end_sample);
                    break;
                }
                if (span.start_sample <= current.span.end_sample) {
                    const int64_t capacity_end = current.span.start_sample + options.max_chunk_samples;
                    if (span.end_sample <= capacity_end) {
                        current.span.end_sample = span.end_sample;
                        current.speech_end_sample = std::max(current.speech_end_sample, speech.end_sample);
                        break;
                    }
                    if (speech.start_sample > current.speech_end_sample &&
                        current.speech_end_sample > current.span.start_sample) {
                        const int64_t boundary = std::min(current.span.end_sample, speech.start_sample);
                        if (boundary >= current.speech_end_sample && boundary > current.span.start_sample) {
                            current.span.end_sample = boundary;
                            span.start_sample = boundary;
                            start_chunk();
                            continue;
                        }
                    }
                    if (current.span.end_sample < capacity_end) {
                        current.span.end_sample = std::min<int64_t>(span.end_sample, capacity_end);
                        if (current.span.end_sample > speech.start_sample) {
                            current.speech_end_sample = std::max(
                                current.speech_end_sample,
                                std::min<int64_t>(speech.end_sample, current.span.end_sample));
                        }
                        span.start_sample = current.span.end_sample;
                        continue;
                    }
                } else {
                    const int64_t gap = span.start_sample - current.span.end_sample;
                    if (gap <= options.merge_gap_samples &&
                        span.end_sample - current.span.start_sample <= options.max_chunk_samples) {
                        current.span.end_sample = span.end_sample;
                        current.speech_end_sample = std::max(current.speech_end_sample, speech.end_sample);
                        break;
                    }
                }
            }
            start_chunk();
        }
    }

    std::vector<runtime::TimeSpan> chunks;
    chunks.reserve(states.size());
    for (const auto & state : states) {
        chunks.push_back(state.span);
    }
    return chunks;
}

std::vector<runtime::TimeSpan> plan_vad_audio_chunks(
    const runtime::AudioBuffer & audio,
    runtime::IOfflineVoiceTaskSession & vad_session,
    const VadAudioChunkOptions & options) {
    if (audio.channels <= 0) {
        throw std::runtime_error("Audio VAD chunker requires positive audio channels");
    }
    if (audio.samples.size() % static_cast<size_t>(audio.channels) != 0) {
        throw std::runtime_error("Audio VAD chunker input size is not divisible by channel count");
    }
    runtime::TaskRequest vad_request;
    vad_request.audio_input = audio;
    vad_session.prepare(runtime::build_preparation_request(vad_request));
    const auto vad_result = vad_session.run(vad_request);
    return plan_vad_audio_chunks(
        vad_result.speech_segments,
        static_cast<int64_t>(audio.samples.size() / static_cast<size_t>(audio.channels)),
        options);
}

std::vector<runtime::TimeSpan> plan_quiet_energy_audio_chunks(
    const std::vector<float> & mono_samples,
    const QuietEnergyAudioChunkOptions & options) {
    require_positive(static_cast<int64_t>(mono_samples.size()), "quiet-energy input samples");
    require_positive(options.chunk_samples, "quiet-energy chunk samples");
    require_positive(options.boundary_context_samples, "quiet-energy boundary context samples");
    require_positive(options.min_energy_window_samples, "quiet-energy min energy window samples");

    const int64_t total = static_cast<int64_t>(mono_samples.size());
    std::vector<runtime::TimeSpan> chunks;
    int64_t index = 0;
    while (index < total) {
        if (index + options.chunk_samples >= total) {
            chunks.push_back({index, total});
            break;
        }
        const int64_t search_start = std::max(index, index + options.chunk_samples - options.boundary_context_samples);
        const int64_t search_end = std::min(index + options.chunk_samples, total);
        int64_t split = index + options.chunk_samples;
        if (search_end > search_start) {
            if (search_end - search_start <= options.min_energy_window_samples) {
                split = (search_start + search_end) / 2;
            } else {
                float min_energy = std::numeric_limits<float>::infinity();
                const int64_t upper = search_end - search_start - options.min_energy_window_samples;
                for (int64_t i = 0; i < upper; i += options.min_energy_window_samples) {
                    double sum = 0.0;
                    for (int64_t j = 0; j < options.min_energy_window_samples; ++j) {
                        const float value = mono_samples[static_cast<size_t>(search_start + i + j)];
                        sum += static_cast<double>(value) * static_cast<double>(value);
                    }
                    const float energy =
                        std::sqrt(static_cast<float>(sum / static_cast<double>(options.min_energy_window_samples)));
                    if (energy < min_energy) {
                        min_energy = energy;
                        split = search_start + i;
                    }
                }
            }
        }
        split = std::max<int64_t>(index + 1, std::min<int64_t>(split, total));
        chunks.push_back({index, split});
        index = split;
    }
    return chunks;
}

runtime::AudioBuffer slice_audio_buffer(
    const runtime::AudioBuffer & audio,
    const runtime::TimeSpan & span) {
    if (audio.sample_rate <= 0) {
        throw std::runtime_error("Audio chunker slice requires a positive sample rate");
    }
    require_positive(audio.channels, "audio channels");
    if (audio.samples.empty()) {
        throw std::runtime_error("Audio chunker slice requires non-empty audio");
    }
    if (audio.samples.size() % static_cast<size_t>(audio.channels) != 0) {
        throw std::runtime_error("Audio chunker slice input size is not divisible by channel count");
    }
    const int64_t frames = static_cast<int64_t>(audio.samples.size() / static_cast<size_t>(audio.channels));
    require_valid_time_span(span, frames, "Audio chunker slice");

    runtime::AudioBuffer out;
    out.sample_rate = audio.sample_rate;
    out.channels = audio.channels;
    const size_t begin = static_cast<size_t>(span.start_sample * audio.channels);
    const size_t end = static_cast<size_t>(span.end_sample * audio.channels);
    out.samples.assign(
        audio.samples.begin() + static_cast<std::ptrdiff_t>(begin),
        audio.samples.begin() + static_cast<std::ptrdiff_t>(end));
    return out;
}

std::vector<float> make_triangular_overlap_window(int64_t chunk_samples) {
    require_positive(chunk_samples, "chunk samples");
    std::vector<float> window(static_cast<size_t>(chunk_samples), 1.0F);
    const int64_t rising = chunk_samples / 2;
    const int64_t falling = chunk_samples - rising;
    for (int64_t i = 0; i < rising; ++i) {
        window[static_cast<size_t>(i)] = static_cast<float>(i + 1);
    }
    for (int64_t i = 0; i < falling; ++i) {
        window[static_cast<size_t>(rising + i)] = static_cast<float>(falling - i);
    }
    const float max_value = *std::max_element(window.begin(), window.end());
    for (float & value : window) {
        value /= max_value;
    }
    return window;
}

std::vector<float> make_linear_fade_window(int64_t chunk_samples, int64_t fade_samples) {
    require_positive(chunk_samples, "chunk samples");
    if (fade_samples < 0 || fade_samples * 2 > chunk_samples) {
        throw std::runtime_error("Audio chunker fade_samples is invalid");
    }
    std::vector<float> window(static_cast<size_t>(chunk_samples), 1.0F);
    if (fade_samples <= 0) {
        return window;
    }
    for (int64_t i = 0; i < fade_samples; ++i) {
        const float alpha = fade_samples == 1
            ? 1.0F
            : static_cast<float>(i) / static_cast<float>(fade_samples - 1);
        window[static_cast<size_t>(i)] = alpha;
        window[static_cast<size_t>(chunk_samples - fade_samples + i)] = 1.0F - alpha;
    }
    return window;
}

void copy_interleaved_chunk_to_planar(
    std::vector<float> & output_planar,
    const std::vector<float> & input_interleaved,
    int64_t channels,
    int64_t input_frames,
    const AudioChunkSpan & span,
    const AudioChunkSpec & spec) {
    validate_copy_shape(input_interleaved, channels, input_frames, output_planar, spec);
    std::fill(output_planar.begin(), output_planar.end(), 0.0F);
    const bool reflect = should_reflect_pad(span, spec);
#ifdef _OPENMP
    #pragma omp parallel for if(channels >= 2)
#endif
    for (int64_t ch = 0; ch < channels; ++ch) {
        float * dst = output_planar.data() + static_cast<size_t>(ch * spec.chunk_samples);
        for (int64_t i = 0; i < spec.chunk_samples; ++i) {
            int64_t src_frame = span.copy_start_sample + i;
            if (src_frame < 0 || src_frame >= input_frames) {
                if (!reflect) {
                    continue;
                }
                src_frame = reflect_index(src_frame, input_frames);
            }
            dst[i] = input_interleaved[static_cast<size_t>(src_frame * channels + ch)];
        }
    }
}

void copy_planar_chunk(
    std::vector<float> & output_planar,
    const std::vector<float> & input_planar,
    int64_t lanes,
    int64_t input_frames,
    const AudioChunkSpan & span,
    const AudioChunkSpec & spec) {
    validate_copy_shape(input_planar, lanes, input_frames, output_planar, spec);
    std::fill(output_planar.begin(), output_planar.end(), 0.0F);
    const bool reflect = should_reflect_pad(span, spec);
#ifdef _OPENMP
    #pragma omp parallel for if(lanes >= 2)
#endif
    for (int64_t lane = 0; lane < lanes; ++lane) {
        float * dst = output_planar.data() + static_cast<size_t>(lane * spec.chunk_samples);
        const float * src = input_planar.data() + static_cast<size_t>(lane * input_frames);
        for (int64_t i = 0; i < spec.chunk_samples; ++i) {
            int64_t src_frame = span.copy_start_sample + i;
            if (src_frame < 0 || src_frame >= input_frames) {
                if (!reflect) {
                    continue;
                }
                src_frame = reflect_index(src_frame, input_frames);
            }
            dst[i] = src[src_frame];
        }
    }
}

void overlap_add_planar_chunk(
    std::vector<float> & output_planar,
    std::vector<float> & weights,
    const std::vector<float> & chunk_planar,
    int64_t lanes,
    int64_t output_frames,
    const AudioChunkSpan & span,
    const std::vector<float> & window,
    AudioChunkCounterMode counter_mode) {
    require_positive(lanes, "lanes");
    require_positive(output_frames, "output frames");
    const int64_t chunk_samples = static_cast<int64_t>(window.size());
    require_positive(chunk_samples, "window samples");
    if (static_cast<int64_t>(chunk_planar.size()) != lanes * chunk_samples ||
        static_cast<int64_t>(output_planar.size()) != lanes * output_frames) {
        throw std::runtime_error("Audio chunker overlap-add size mismatch");
    }
    const int64_t expected_weights = counter_mode == AudioChunkCounterMode::SharedAcrossLanes
        ? output_frames
        : lanes * output_frames;
    if (static_cast<int64_t>(weights.size()) != expected_weights) {
        throw std::runtime_error("Audio chunker overlap-add weight size mismatch");
    }
    if (span.valid_start_in_chunk < 0 ||
        span.valid_start_in_chunk + span.valid_samples > chunk_samples ||
        span.output_start_sample < 0 ||
        span.output_start_sample + span.valid_samples > output_frames) {
        throw std::runtime_error("Audio chunker overlap-add span mismatch");
    }

#ifdef _OPENMP
    #pragma omp parallel for if(lanes >= 2)
#endif
    for (int64_t lane = 0; lane < lanes; ++lane) {
        for (int64_t i = 0; i < span.valid_samples; ++i) {
            const int64_t chunk_frame = span.valid_start_in_chunk + i;
            const int64_t output_frame = span.output_start_sample + i;
            const float gain = window[static_cast<size_t>(chunk_frame)];
            output_planar[planar_index(lane, output_frame, output_frames)] +=
                chunk_planar[planar_index(lane, chunk_frame, chunk_samples)] * gain;
            if (counter_mode == AudioChunkCounterMode::PerLane || lane == 0) {
                weights[weight_index(counter_mode, lane, output_frame, output_frames)] += gain;
            }
        }
    }
}

void normalize_overlap_added_planar(
    std::vector<float> & output_planar,
    const std::vector<float> & weights,
    int64_t lanes,
    int64_t output_frames,
    AudioChunkCounterMode counter_mode) {
    require_positive(lanes, "lanes");
    require_positive(output_frames, "output frames");
    if (static_cast<int64_t>(output_planar.size()) != lanes * output_frames) {
        throw std::runtime_error("Audio chunker normalize output size mismatch");
    }
    const int64_t expected_weights = counter_mode == AudioChunkCounterMode::SharedAcrossLanes
        ? output_frames
        : lanes * output_frames;
    if (static_cast<int64_t>(weights.size()) != expected_weights) {
        throw std::runtime_error("Audio chunker normalize weight size mismatch");
    }
#ifdef _OPENMP
    #pragma omp parallel for if(lanes >= 2)
#endif
    for (int64_t lane = 0; lane < lanes; ++lane) {
        for (int64_t frame = 0; frame < output_frames; ++frame) {
            const float denom = weights[weight_index(counter_mode, lane, frame, output_frames)] > 1.0e-8F
                ? weights[weight_index(counter_mode, lane, frame, output_frames)]
                : 1.0F;
            output_planar[planar_index(lane, frame, output_frames)] /= denom;
        }
    }
}

void append_chunk_word_timestamps(
    std::vector<runtime::WordTimestamp> & output,
    const std::vector<runtime::WordTimestamp> & chunk_words,
    const runtime::TimeSpan & chunk_span) {
    append_chunk_word_timestamps(output, chunk_words, chunk_span, chunk_span);
}

void append_chunk_word_timestamps(
    std::vector<runtime::WordTimestamp> & output,
    const std::vector<runtime::WordTimestamp> & chunk_words,
    const runtime::TimeSpan & source_span,
    const runtime::TimeSpan & keep_span) {
    const auto valid_span = [](const runtime::TimeSpan & span) {
        return span.start_sample >= 0 && span.end_sample >= span.start_sample;
    };
    if (!valid_span(source_span)) {
        throw std::runtime_error("Audio chunker word merge requires a valid source span");
    }
    if (!valid_span(keep_span) ||
        keep_span.start_sample < source_span.start_sample ||
        keep_span.end_sample > source_span.end_sample) {
        throw std::runtime_error("Audio chunker word merge requires keep span inside source span");
    }
    const int64_t source_samples = source_span.end_sample - source_span.start_sample;
    for (const auto & word : chunk_words) {
        if (word.span.end_sample < word.span.start_sample) {
            throw std::runtime_error("Audio chunker word merge requires ordered word timestamps");
        }
        const int64_t local_start = std::max<int64_t>(word.span.start_sample, 0);
        const int64_t local_end = std::min<int64_t>(word.span.end_sample, source_samples);
        if (local_start >= local_end) {
            throw std::runtime_error("Audio chunker word merge received a timestamp outside the chunk span");
        }
        const int64_t global_start = source_span.start_sample + local_start;
        if (global_start < keep_span.start_sample || global_start >= keep_span.end_sample) {
            continue;
        }
        auto merged = word;
        merged.span.start_sample = global_start;
        merged.span.end_sample = source_span.start_sample + local_end;
        output.push_back(std::move(merged));
    }
}

}  // namespace engine::audio
