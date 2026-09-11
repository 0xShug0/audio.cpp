#include "engine/models/sortformer_diar/postprocess.h"

#include "engine/framework/debug/trace.h"
#include "engine/framework/runtime/options.h"
#include "engine/models/sortformer_diar/graph.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace engine::models::sortformer_diar {

namespace {

constexpr double kDefaultSessionLenSec = 20.0;

std::string speaker_id_for_index(int64_t index) {
    std::ostringstream oss;
    oss << "SPEAKER_" << std::setw(2) << std::setfill('0') << index;
    return oss.str();
}

struct SecondSpan {
    double start = 0.0;
    double end = 0.0;
};

std::vector<SecondSpan> merge_overlap_segments(std::vector<SecondSpan> segments) {
    if (segments.size() <= 1) {
        return segments;
    }
    std::sort(segments.begin(), segments.end(), [](const SecondSpan & lhs, const SecondSpan & rhs) {
        if (lhs.start != rhs.start) {
            return lhs.start < rhs.start;
        }
        return lhs.end < rhs.end;
    });
    std::vector<SecondSpan> merged;
    merged.reserve(segments.size());
    for (const auto & segment : segments) {
        if (merged.empty() || merged.back().end < segment.start) {
            merged.push_back(segment);
        } else {
            merged.back().end = std::max(merged.back().end, segment.end);
        }
    }
    return merged;
}

std::vector<SecondSpan> filter_short_segments(const std::vector<SecondSpan> & segments, double threshold_sec) {
    std::vector<SecondSpan> kept;
    kept.reserve(segments.size());
    for (const auto & segment : segments) {
        if (segment.end - segment.start >= threshold_sec) {
            kept.push_back(segment);
        }
    }
    return kept;
}

std::vector<SecondSpan> get_gap_segments(const std::vector<SecondSpan> & segments) {
    if (segments.size() <= 1) {
        return {};
    }
    std::vector<SecondSpan> sorted = segments;
    std::sort(sorted.begin(), sorted.end(), [](const SecondSpan & lhs, const SecondSpan & rhs) {
        if (lhs.start != rhs.start) {
            return lhs.start < rhs.start;
        }
        return lhs.end < rhs.end;
    });
    std::vector<SecondSpan> gaps;
    gaps.reserve(sorted.size() - 1);
    for (size_t i = 0; i + 1 < sorted.size(); ++i) {
        gaps.push_back({sorted[i].end, sorted[i + 1].start});
    }
    return gaps;
}

std::vector<SecondSpan> subtract_segments(
    const std::vector<SecondSpan> & original,
    const std::vector<SecondSpan> & to_remove) {
    std::vector<SecondSpan> kept;
    kept.reserve(original.size());
    for (const auto & candidate : original) {
        bool remove = false;
        for (const auto & forbidden : to_remove) {
            if (candidate.start == forbidden.start && candidate.end == forbidden.end) {
                remove = true;
                break;
            }
        }
        if (!remove) {
            kept.push_back(candidate);
        }
    }
    return kept;
}

std::vector<SecondSpan> binarize_sequence(
    const std::vector<float> & sequence,
    float onset,
    float offset,
    double frame_length_sec,
    double pad_onset_sec,
    double pad_offset_sec) {
    bool speech = false;
    double start = 0.0;
    size_t i = 0;
    std::vector<SecondSpan> segments;
    for (i = 0; i < sequence.size(); ++i) {
        const double time_sec = static_cast<double>(i) * frame_length_sec;
        if (speech) {
            if (sequence[i] < offset) {
                const double seg_start = std::max(0.0, start - pad_onset_sec);
                const double seg_end = time_sec + pad_offset_sec;
                if (seg_end > seg_start) {
                    segments.push_back({seg_start, seg_end});
                }
                start = time_sec;
                speech = false;
            }
        } else if (sequence[i] > onset) {
            start = time_sec;
            speech = true;
        }
    }
    if (speech && !sequence.empty()) {
        const double seg_start = std::max(0.0, start - pad_onset_sec);
        const double seg_end = static_cast<double>(i - 1) * frame_length_sec + pad_offset_sec;
        if (seg_end > seg_start) {
            segments.push_back({seg_start, seg_end});
        }
    }
    return merge_overlap_segments(std::move(segments));
}

std::vector<SecondSpan> apply_vad_filtering(
    std::vector<SecondSpan> speech_segments,
    double min_duration_on_sec,
    double min_duration_off_sec,
    bool filter_speech_first) {
    if (speech_segments.empty()) {
        return speech_segments;
    }
    if (filter_speech_first) {
        if (min_duration_on_sec > 0.0) {
            speech_segments = filter_short_segments(speech_segments, min_duration_on_sec);
        }
        if (min_duration_off_sec > 0.0 && !speech_segments.empty()) {
            const auto non_speech_segments = get_gap_segments(speech_segments);
            const auto long_non_speech_segments = filter_short_segments(non_speech_segments, min_duration_off_sec);
            const auto short_non_speech_segments = subtract_segments(non_speech_segments, long_non_speech_segments);
            speech_segments.insert(
                speech_segments.end(),
                short_non_speech_segments.begin(),
                short_non_speech_segments.end());
            speech_segments = merge_overlap_segments(std::move(speech_segments));
        }
    } else {
        if (min_duration_off_sec > 0.0) {
            const auto non_speech_segments = get_gap_segments(speech_segments);
            const auto long_non_speech_segments = filter_short_segments(non_speech_segments, min_duration_off_sec);
            const auto short_non_speech_segments = subtract_segments(non_speech_segments, long_non_speech_segments);
            speech_segments.insert(
                speech_segments.end(),
                short_non_speech_segments.begin(),
                short_non_speech_segments.end());
            speech_segments = merge_overlap_segments(std::move(speech_segments));
        }
        if (min_duration_on_sec > 0.0) {
            speech_segments = filter_short_segments(speech_segments, min_duration_on_sec);
        }
    }
    return speech_segments;
}

int64_t fl2int(double value, int decimals) {
    return static_cast<int64_t>(std::llround(value * std::pow(10.0, decimals)));
}

double int2fl(int64_t value, int decimals) {
    return static_cast<double>(value) / std::pow(10.0, decimals);
}

std::vector<SecondSpan> merge_float_intervals_with_margin(
    const std::vector<SecondSpan> & ranges,
    int decimals = 5,
    int margin = 2) {
    std::vector<std::pair<int64_t, int64_t>> ranges_int;
    ranges_int.reserve(ranges.size());
    for (const auto & range : ranges) {
        const int64_t start = fl2int(range.start, decimals) + margin;
        const int64_t end = fl2int(range.end, decimals);
        if (start < end) {
            ranges_int.emplace_back(start, end);
        }
    }
    std::sort(ranges_int.begin(), ranges_int.end());
    std::vector<std::pair<int64_t, int64_t>> merged_int;
    for (const auto & range : ranges_int) {
        if (merged_int.empty() || merged_int.back().second < range.first) {
            merged_int.push_back(range);
        } else {
            merged_int.back().second = std::max(merged_int.back().second, range.second);
        }
    }
    std::vector<SecondSpan> merged;
    merged.reserve(merged_int.size());
    for (const auto & range : merged_int) {
        merged.push_back({int2fl(range.first - margin, decimals), int2fl(range.second, decimals)});
    }
    return merged;
}

}  // namespace

SortformerPostprocessConfig parse_sortformer_postprocess_config(const runtime::SessionOptions & options) {
    SortformerPostprocessConfig config;
    if (const auto value = runtime::parse_finite_float_option(
            options.options,
            {"sortformer_diar.speaker_threshold", "speaker_threshold"})) {
        if (*value < 0.0F || *value > 1.0F) {
            throw std::runtime_error("Sortformer diar speaker_threshold must be between 0 and 1");
        }
        config.threshold = *value;
    }
    if (const auto value = runtime::parse_i64_option(
            options.options,
            {"sortformer_diar.speaker_min_frames", "speaker_min_frames"})) {
        if (*value < 0) {
            throw std::runtime_error("Sortformer diar speaker_min_frames must be non-negative");
        }
        config.min_frames = *value;
    }
    if (const auto value = runtime::parse_i64_option(
            options.options,
            {"sortformer_diar.speaker_pad_frames", "speaker_pad_frames"})) {
        if (*value < 0) {
            throw std::runtime_error("Sortformer diar speaker_pad_frames must be non-negative");
        }
        config.pad_frames = *value;
    }
    return config;
}

SortformerFixedContextContract make_sortformer_fixed_context_contract_for_samples(
    int64_t sample_count,
    const SortformerAssets & assets) {
    SortformerFixedContextContract contract;
    const int64_t sample_rate = assets.feature_config.sample_rate;
    const int64_t hop_length = assets.feature_config.hop_length;
    const int64_t kernel = assets.model_config.fc_encoder.subsampling_conv_kernel_size;
    const int64_t stride = assets.model_config.fc_encoder.subsampling_conv_stride;
    const int64_t padding = (kernel - 1) / 2;
    contract.session_len_sec = static_cast<double>(sample_count) / static_cast<double>(sample_rate);
    const int64_t valid_feature_frames = std::max<int64_t>(0, sample_count / hop_length);
    contract.feature_frames = ((valid_feature_frames + 15) / 16) * 16;

    const int64_t valid1 = sortformer_conv_valid_length(valid_feature_frames, kernel, stride, padding);
    const int64_t valid2 = sortformer_conv_valid_length(valid1, kernel, stride, padding);
    contract.encoder_frames = sortformer_conv_valid_length(valid2, kernel, stride, padding);

    if (contract.encoder_frames <= 0 || contract.feature_frames <= 0) {
        throw std::runtime_error("Sortformer diar session_len_sec produced an empty graph contract");
    }
    if (contract.encoder_frames > assets.model_config.fc_encoder.max_position_embeddings) {
        throw std::runtime_error("Sortformer diar session_len_sec exceeds fc_encoder.max_position_embeddings");
    }
    if (contract.encoder_frames > assets.model_config.tf_encoder.max_source_positions) {
        throw std::runtime_error("Sortformer diar session_len_sec exceeds tf_encoder.max_source_positions");
    }
    return contract;
}

SortformerFixedContextContract parse_sortformer_fixed_context_contract(
    const runtime::SessionOptions & options,
    const SortformerAssets & assets) {
    double session_len_sec = kDefaultSessionLenSec;
    if (const auto value = runtime::parse_positive_finite_float_option(
            options.options,
            {"sortformer_diar.session_len_sec", "session_len_sec"})) {
        session_len_sec = *value;
    }
    if (!(session_len_sec > 0.0)) {
        throw std::runtime_error("Sortformer diar session_len_sec must be > 0");
    }
    const int64_t sample_count = static_cast<int64_t>(
        std::llround(session_len_sec * static_cast<double>(assets.feature_config.sample_rate)));
    return make_sortformer_fixed_context_contract_for_samples(sample_count, assets);
}

void emit_sortformer_timings(const SortformerRunTimings & timings) {
    if (!debug::timing_log_enabled()) {
        return;
    }
    debug::timing_log_scalar("sortformer.frontend_ms", timings.frontend_ms);
    debug::timing_log_scalar("sortformer.log_mel_ms", timings.log_mel_ms);
    debug::timing_log_scalar("sortformer.feature_normalizer_ms", timings.feature_normalizer_ms);
    debug::timing_log_scalar("sortformer.graph.ensure_ms", timings.graph_ensure_ms);
    debug::timing_log_scalar("sortformer.graph.prepare_ms", timings.graph_prepare_ms);
    debug::timing_log_scalar("sortformer.encoder_compute_ms", timings.encoder_compute_ms);
    debug::timing_log_scalar("sortformer.encoder_readback_ms", timings.encoder_readback_ms);
    debug::timing_log_scalar("sortformer.encoder_ms", timings.encoder_ms);
    debug::timing_log_scalar("sortformer.postprocess_ms", timings.postprocess_ms);
    debug::timing_log_scalar("session.wall_ms", timings.wall_ms);
}

std::vector<runtime::SpeakerTurn> decode_sortformer_speaker_turns(
    const std::vector<float> & probabilities,
    int64_t,
    int64_t valid_frames,
    int64_t num_speakers,
    int64_t frame_step_samples,
    const SortformerPostprocessConfig & config) {
    std::vector<runtime::SpeakerTurn> turns;
    const double step_sec =
        static_cast<double>(frame_step_samples) / static_cast<double>(16000);
    constexpr double kVADFrameSec = 0.01;
    const int64_t repeat_count = std::max<int64_t>(1, static_cast<int64_t>(std::llround(step_sec / kVADFrameSec)));
    for (int64_t speaker = 0; speaker < num_speakers; ++speaker) {
        std::vector<float> repeated_scores;
        repeated_scores.reserve(static_cast<size_t>(valid_frames * repeat_count));
        for (int64_t frame = 0; frame < valid_frames; ++frame) {
            const float score = probabilities[static_cast<size_t>(frame * num_speakers + speaker)];
            for (int64_t i = 0; i < repeat_count; ++i) {
                repeated_scores.push_back(score);
            }
        }
        auto segments = binarize_sequence(
            repeated_scores,
            config.threshold,
            config.threshold,
            kVADFrameSec,
            static_cast<double>(config.pad_frames) * step_sec,
            static_cast<double>(config.pad_frames) * step_sec);
        segments = apply_vad_filtering(
            std::move(segments),
            static_cast<double>(config.min_frames) * step_sec,
            0.0,
            true);
        for (auto & segment : segments) {
            segment.start = std::round(segment.start * 100.0) / 100.0;
            segment.end = std::round(segment.end * 100.0) / 100.0;
        }
        segments = merge_float_intervals_with_margin(segments);
        for (const auto & segment : segments) {
            turns.push_back({
                {
                    static_cast<int64_t>(std::llround(segment.start * 16000.0)),
                    static_cast<int64_t>(std::llround(segment.end * 16000.0)),
                },
                speaker_id_for_index(speaker),
                0.0f,
                "",
            });
        }
    }
    std::sort(turns.begin(), turns.end(), [](const runtime::SpeakerTurn & lhs, const runtime::SpeakerTurn & rhs) {
        if (lhs.span.start_sample != rhs.span.start_sample) {
            return lhs.span.start_sample < rhs.span.start_sample;
        }
        return lhs.speaker_id < rhs.speaker_id;
    });
    return turns;
}

}  // namespace engine::models::sortformer_diar
