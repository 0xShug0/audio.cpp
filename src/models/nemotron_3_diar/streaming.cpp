#include "engine/models/nemotron_3_diar/streaming.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/runtime/options.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace engine::models::nemotron_3_diar {
namespace {

constexpr float kNegativeInfinity = -std::numeric_limits<float>::infinity();
constexpr float kPositiveInfinity = std::numeric_limits<float>::infinity();

std::vector<int> top_indices(
    const std::vector<float> & scores,
    int frames,
    int speakers,
    int speaker,
    int count) {
    if (frames <= 0 || count <= 0) return {};
    std::vector<int> indices(static_cast<size_t>(frames));
    std::iota(indices.begin(), indices.end(), 0);
    count = std::min(count, frames);
    std::partial_sort(indices.begin(), indices.begin() + count, indices.end(), [&](int left, int right) {
        const float lhs = scores[static_cast<size_t>(left) * speakers + speaker];
        const float rhs = scores[static_cast<size_t>(right) * speakers + speaker];
        return lhs != rhs ? lhs > rhs : left < right;
    });
    indices.resize(static_cast<size_t>(count));
    return indices;
}

int64_t option_or(
    const std::unordered_map<std::string, std::string> & options,
    const char * name,
    int64_t fallback) {
    const auto value = runtime::parse_i64_option(options, {name});
    return value.value_or(fallback);
}

}  // namespace

StreamingConfig streaming_profile(
    const StreamingConfig & model_defaults,
    const std::unordered_map<std::string, std::string> & options) {
    StreamingConfig result = model_defaults;
    const auto profile = runtime::find_option(
        options, {"nemotron_3_diar.latency_profile"}).value_or("very_high");
    if (profile == "very_high") {
        result.spkcache_len = 264;
        result.fifo_len = 40;
        result.chunk_len = 340;
        result.chunk_right_context = 40;
        result.spkcache_update_period = 300;
    } else if (profile == "low") {
        result.spkcache_len = 264;
        result.fifo_len = 264;
        result.chunk_len = 9;
        result.chunk_right_context = 4;
        result.spkcache_update_period = 222;
    } else if (profile == "very_low") {
        result.spkcache_len = 264;
        result.fifo_len = 264;
        result.chunk_len = 6;
        result.chunk_right_context = 2;
        result.spkcache_update_period = 222;
    } else if (profile == "ultra_low") {
        result.spkcache_len = 264;
        result.fifo_len = 264;
        result.chunk_len = 3;
        result.chunk_right_context = 1;
        result.spkcache_update_period = 222;
    } else if (profile == "custom") {
        result.spkcache_len = option_or(
            options, "nemotron_3_diar.spkcache_len", result.spkcache_len);
        result.fifo_len = option_or(
            options, "nemotron_3_diar.fifo_len", result.fifo_len);
        result.chunk_len = option_or(
            options, "nemotron_3_diar.chunk_len", result.chunk_len);
        result.chunk_right_context = option_or(
            options, "nemotron_3_diar.chunk_right_context", result.chunk_right_context);
        result.spkcache_update_period = option_or(
            options, "nemotron_3_diar.spkcache_update_period", result.spkcache_update_period);
    } else {
        throw std::runtime_error("unknown Nemotron 3 diarization latency profile: " + profile);
    }
    if (result.spkcache_len < 16 || result.fifo_len < 0 || result.chunk_len <= 0 ||
        result.chunk_left_context < 0 || result.chunk_right_context < 0 ||
        result.spkcache_update_period <= 0) {
        throw std::runtime_error("invalid Nemotron 3 diarization streaming geometry");
    }
    return result;
}

StreamScheduler::StreamScheduler(
    const FeatureConfig & frontend,
    const StreamingConfig & streaming,
    int64_t subsampling_factor)
    : frontend_(frontend), streaming_(streaming), subsampling_factor_(subsampling_factor) {
    if (frontend_.sample_rate <= 0 || frontend_.hop_length <= 0 || frontend_.n_fft <= 0 ||
        subsampling_factor_ <= 0) {
        throw std::invalid_argument("invalid Nemotron 3 diarization frontend geometry");
    }
    reset();
}

void StreamScheduler::reset() {
    audio_.clear();
    audio_base_sample_ = 0;
    audio_received_samples_ = 0;
    logical_audio_end_samples_ = 0;
    mel_produced_ = 0;
    mel_consumed_ = 0;
    started_ = false;
    finished_ = false;
}

void StreamScheduler::append_audio(const runtime::AudioChunk & chunk) {
    if (chunk.sample_rate != frontend_.sample_rate || chunk.channels <= 0 ||
        chunk.samples.size() % static_cast<size_t>(chunk.channels) != 0) {
        throw std::invalid_argument("invalid Nemotron 3 diarization audio chunk");
    }
    if (!started_) {
        if (chunk.start_sample < 0) throw std::invalid_argument("audio chunk starts before sample zero");
        audio_received_samples_ = chunk.start_sample;
        audio_base_sample_ = chunk.start_sample;
        started_ = true;
    }
    if (chunk.start_sample != audio_received_samples_) {
        throw std::invalid_argument("Nemotron 3 diarization audio chunks must be contiguous");
    }
    const auto mono = audio::mixdown_interleaved_to_mono_average(chunk.samples, chunk.channels);
    audio_.insert(audio_.end(), mono.begin(), mono.end());
    audio_received_samples_ += static_cast<int64_t>(mono.size());
    logical_audio_end_samples_ = audio_received_samples_;
}

int64_t StreamScheduler::available_mel_frames(bool final_flush) const noexcept {
    if (logical_audio_end_samples_ < frontend_.n_fft / 2) return 0;
    const int64_t samples = logical_audio_end_samples_ - frontend_.n_fft / 2;
    return final_flush
        ? (samples + frontend_.hop_length - 1) / frontend_.hop_length
        : samples / frontend_.hop_length + 1;
}

StreamWindow StreamScheduler::make_window(int64_t end_mel, bool final_flush) const {
    const int64_t sub = subsampling_factor_;
    const int64_t left_limit = streaming_.chunk_left_context * sub;
    const int64_t right_limit = streaming_.chunk_right_context * sub;
    const int64_t start_mel = mel_consumed_;
    const int64_t left_mel = std::min(left_limit, start_mel);
    const int64_t right_mel = std::min(right_limit, mel_produced_ - end_mel);
    const int64_t window_start = start_mel - left_mel;
    const int64_t window_frames = end_mel + right_mel - window_start;
    const int64_t sample_start = window_start * frontend_.hop_length;
    const int64_t pre_roll = frontend_.n_fft / 2;
    const int64_t feature_start = window_start == 0 ? 0 : sample_start - pre_roll;
    const int64_t feature_end =
        sample_start + (window_frames - 1) * frontend_.hop_length + pre_roll;

    StreamWindow window;
    window.mel_start = window_start;
    window.mel_frames = window_frames;
    const int64_t new_mel_frames = end_mel - start_mel;
    window.encoder_frames = final_flush
        ? (new_mel_frames + sub - 1) / sub
        : new_mel_frames / sub;
    window.left_context_frames = left_mel / sub;
    window.right_context_frames = (right_mel + sub - 1) / sub;
    window.mono_samples.assign(static_cast<size_t>(std::max<int64_t>(feature_end - feature_start, 0)), 0.0F);
    const int64_t copy_start = std::max(feature_start, audio_base_sample_);
    const int64_t copy_end = std::min(feature_end, audio_received_samples_);
    if (copy_end > copy_start) {
        std::copy_n(
            audio_.data() + static_cast<size_t>(copy_start - audio_base_sample_),
            static_cast<size_t>(copy_end - copy_start),
            window.mono_samples.data() + static_cast<size_t>(copy_start - feature_start));
    }
    return window;
}

void StreamScheduler::trim_audio() {
    const int64_t keep_mel = std::max<int64_t>(
        mel_consumed_ - streaming_.chunk_left_context * subsampling_factor_, 0);
    const int64_t keep_sample = std::max<int64_t>(
        keep_mel * frontend_.hop_length - frontend_.n_fft / 2, 0);
    if (keep_sample <= audio_base_sample_) return;
    const int64_t count = std::min<int64_t>(
        keep_sample - audio_base_sample_, static_cast<int64_t>(audio_.size()));
    audio_.erase(audio_.begin(), audio_.begin() + count);
    audio_base_sample_ += count;
}

std::vector<StreamWindow> StreamScheduler::drain(bool final_flush) {
    const int64_t chunk_mel = streaming_.chunk_len * subsampling_factor_;
    const int64_t right_mel = streaming_.chunk_right_context * subsampling_factor_;
    std::vector<StreamWindow> windows;
    mel_produced_ = available_mel_frames(final_flush);
    while (true) {
        const int64_t start = mel_consumed_;
        int64_t end = start + chunk_mel;
        if (!final_flush) {
            if (end + right_mel > mel_produced_) break;
        } else {
            end = std::min(end, mel_produced_);
            if (end <= start) break;
        }
        windows.push_back(make_window(end, final_flush));
        mel_consumed_ = end;
        trim_audio();
    }
    return windows;
}

std::vector<StreamWindow> StreamScheduler::push_audio(const runtime::AudioChunk & chunk) {
    if (finished_) return {};
    append_audio(chunk);
    return drain(false);
}

std::vector<StreamWindow> StreamScheduler::finalize() {
    if (finished_) return {};
    finished_ = true;
    if (!started_ || audio_received_samples_ == 0) return {};
    const int64_t actual_end = audio_received_samples_;
    const int64_t tail = frontend_.n_fft / 2;
    audio_.insert(audio_.end(), static_cast<size_t>(tail), 0.0F);
    audio_received_samples_ += tail;
    logical_audio_end_samples_ = actual_end + tail;
    return drain(true);
}

AoscState::AoscState(
    const StreamingConfig & config,
    int speakers,
    int embedding_dim,
    std::vector<float> silence_embedding)
    : config_(config),
      speakers_(speakers),
      embedding_dim_(embedding_dim),
      silence_embedding_(std::move(silence_embedding)) {
    if (speakers_ <= 0 || embedding_dim_ <= 0 ||
        silence_embedding_.size() != static_cast<size_t>(embedding_dim_)) {
        throw std::invalid_argument("invalid Nemotron 3 diarization AOSC state");
    }
}

void AoscState::update(
    const float * chunk_embeddings,
    int window_frames,
    const float * probabilities,
    int left_context,
    int right_context) {
    const int valid_chunk = window_frames - left_context - right_context;
    if (valid_chunk <= 0) return;
    const int old_cache = speaker_cache_frames_;
    const int old_fifo = fifo_frames_;
    const float * fifo_probabilities = probabilities + static_cast<size_t>(old_cache) * speakers_;
    const float * chunk_probabilities =
        probabilities + static_cast<size_t>(old_cache + old_fifo + left_context) * speakers_;
    const float * valid_embeddings =
        chunk_embeddings + static_cast<size_t>(left_context) * embedding_dim_;

    fifo_.insert(
        fifo_.end(), valid_embeddings,
        valid_embeddings + static_cast<size_t>(valid_chunk) * embedding_dim_);
    std::vector<float> combined_probabilities(static_cast<size_t>(old_fifo + valid_chunk) * speakers_);
    std::memcpy(
        combined_probabilities.data(), fifo_probabilities,
        static_cast<size_t>(old_fifo) * speakers_ * sizeof(float));
    std::memcpy(
        combined_probabilities.data() + static_cast<size_t>(old_fifo) * speakers_,
        chunk_probabilities,
        static_cast<size_t>(valid_chunk) * speakers_ * sizeof(float));
    fifo_frames_ = old_fifo + valid_chunk;

    if (fifo_frames_ <= config_.fifo_len) return;
    int pop = static_cast<int>(config_.spkcache_update_period);
    pop = std::max(pop, valid_chunk - static_cast<int>(config_.fifo_len) + old_fifo);
    pop = std::min(pop, fifo_frames_);
    speaker_cache_.insert(
        speaker_cache_.end(), fifo_.begin(),
        fifo_.begin() + static_cast<size_t>(pop) * embedding_dim_);
    if (!speaker_cache_probabilities_.empty()) {
        speaker_cache_probabilities_.insert(
            speaker_cache_probabilities_.end(), combined_probabilities.begin(),
            combined_probabilities.begin() + static_cast<size_t>(pop) * speakers_);
    }
    speaker_cache_frames_ += pop;
    if (speaker_cache_frames_ > config_.spkcache_len && speaker_cache_probabilities_.empty()) {
        speaker_cache_probabilities_.resize(static_cast<size_t>(speaker_cache_frames_) * speakers_);
        std::memcpy(
            speaker_cache_probabilities_.data(), probabilities,
            static_cast<size_t>(old_cache) * speakers_ * sizeof(float));
        std::memcpy(
            speaker_cache_probabilities_.data() + static_cast<size_t>(old_cache) * speakers_,
            combined_probabilities.data(), static_cast<size_t>(pop) * speakers_ * sizeof(float));
    }
    fifo_.erase(fifo_.begin(), fifo_.begin() + static_cast<size_t>(pop) * embedding_dim_);
    fifo_frames_ -= pop;
    if (speaker_cache_frames_ > config_.spkcache_len) compress(speaker_cache_probabilities_);
}

void AoscState::compress(const std::vector<float> & probabilities) {
    const int frames = speaker_cache_frames_;
    const int capacity = static_cast<int>(config_.spkcache_len);
    const int per_speaker = capacity / speakers_ - static_cast<int>(config_.spkcache_sil_frames_per_spk);
    const int strong_count = static_cast<int>(std::floor(per_speaker * config_.strong_boost_rate));
    const int weak_count = static_cast<int>(std::floor(per_speaker * config_.weak_boost_rate));
    const int minimum_positive = static_cast<int>(std::floor(per_speaker * config_.min_pos_scores_rate));
    const float log_half = std::log(0.5F);
    std::vector<float> scores(static_cast<size_t>(frames) * speakers_);
    for (int frame = 0; frame < frames; ++frame) {
        float inactive_sum = 0.0F;
        for (int speaker = 0; speaker < speakers_; ++speaker) {
            inactive_sum += std::log(std::max(
                1.0F - probabilities[static_cast<size_t>(frame) * speakers_ + speaker],
                config_.pred_score_threshold));
        }
        for (int speaker = 0; speaker < speakers_; ++speaker) {
            const size_t index = static_cast<size_t>(frame) * speakers_ + speaker;
            const float probability = probabilities[index];
            scores[index] = std::log(std::max(probability, config_.pred_score_threshold)) -
                std::log(std::max(1.0F - probability, config_.pred_score_threshold)) + inactive_sum - log_half;
        }
    }
    std::vector<int> positives(static_cast<size_t>(speakers_), 0);
    for (int frame = 0; frame < frames; ++frame) {
        for (int speaker = 0; speaker < speakers_; ++speaker) {
            const size_t index = static_cast<size_t>(frame) * speakers_ + speaker;
            if (!(probabilities[index] > 0.5F)) scores[index] = kNegativeInfinity;
            if (scores[index] > 0.0F) ++positives[static_cast<size_t>(speaker)];
        }
    }
    for (int speaker = 0; speaker < speakers_; ++speaker) {
        if (positives[static_cast<size_t>(speaker)] < minimum_positive) continue;
        for (int frame = 0; frame < frames; ++frame) {
            const size_t index = static_cast<size_t>(frame) * speakers_ + speaker;
            if (probabilities[index] > 0.5F && !(scores[index] > 0.0F)) scores[index] = kNegativeInfinity;
        }
    }
    if (config_.scores_boost_latest > 0.0F) {
        for (int frame = capacity; frame < frames; ++frame) {
            for (int speaker = 0; speaker < speakers_; ++speaker) {
                scores[static_cast<size_t>(frame) * speakers_ + speaker] += config_.scores_boost_latest;
            }
        }
    }
    for (int speaker = 0; speaker < speakers_; ++speaker) {
        for (int frame : top_indices(scores, frames, speakers_, speaker, strong_count)) {
            scores[static_cast<size_t>(frame) * speakers_ + speaker] -= 2.0F * log_half;
        }
        for (int frame : top_indices(scores, frames, speakers_, speaker, weak_count)) {
            scores[static_cast<size_t>(frame) * speakers_ + speaker] -= log_half;
        }
    }

    const int padded_frames = frames + static_cast<int>(config_.spkcache_sil_frames_per_spk);
    std::vector<int64_t> flat(static_cast<size_t>(speakers_) * padded_frames);
    std::iota(flat.begin(), flat.end(), 0);
    auto flat_score = [&](int64_t index) {
        const int frame = static_cast<int>(index % padded_frames);
        if (frame >= frames) return kPositiveInfinity;
        return scores[static_cast<size_t>(frame) * speakers_ + static_cast<int>(index / padded_frames)];
    };
    std::partial_sort(flat.begin(), flat.begin() + capacity, flat.end(), [&](int64_t left, int64_t right) {
        const float lhs = flat_score(left);
        const float rhs = flat_score(right);
        return lhs != rhs ? lhs > rhs : left < right;
    });
    flat.resize(static_cast<size_t>(capacity));
    for (auto & index : flat) {
        if (flat_score(index) == kNegativeInfinity) {
            index = config_.max_index * padded_frames + config_.max_index;
        }
    }
    std::sort(flat.begin(), flat.end());
    std::vector<float> next_cache(static_cast<size_t>(capacity) * embedding_dim_);
    std::vector<float> next_probabilities(static_cast<size_t>(capacity) * speakers_, 0.0F);
    for (int output = 0; output < capacity; ++output) {
        const int64_t index = flat[static_cast<size_t>(output)];
        const int frame = static_cast<int>(index % padded_frames);
        const bool disabled = index >= static_cast<int64_t>(speakers_) * padded_frames || frame >= frames;
        if (disabled) {
            std::copy(
                silence_embedding_.begin(), silence_embedding_.end(),
                next_cache.begin() + static_cast<size_t>(output) * embedding_dim_);
        } else {
            std::copy_n(
                speaker_cache_.begin() + static_cast<size_t>(frame) * embedding_dim_, embedding_dim_,
                next_cache.begin() + static_cast<size_t>(output) * embedding_dim_);
            std::copy_n(
                probabilities.begin() + static_cast<size_t>(frame) * speakers_, speakers_,
                next_probabilities.begin() + static_cast<size_t>(output) * speakers_);
        }
    }
    speaker_cache_ = std::move(next_cache);
    speaker_cache_probabilities_ = std::move(next_probabilities);
    speaker_cache_frames_ = capacity;
}

}  // namespace engine::models::nemotron_3_diar
