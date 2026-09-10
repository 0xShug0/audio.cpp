#include "engine/community_models/sortformer_diar/stream_schedule.h"

#include "engine/framework/audio/conversion.h"

#include <algorithm>
#include <stdexcept>

namespace engine::community_models::sortformer_diar {

SortformerV2StreamScheduler::SortformerV2StreamScheduler(
    const SortformerV2FeatureExtractorConfig & frontend,
    const SortformerV2DiarGeometry & geometry,
    int64_t subsampling_factor)
    : frontend_(frontend), geometry_(geometry), subsampling_factor_(subsampling_factor) {
    if (frontend_.sample_rate <= 0 || frontend_.hop_length <= 0 || frontend_.n_fft <= 0 ||
        subsampling_factor_ <= 0) {
        throw std::invalid_argument("Sortformer v2 stream scheduler requires positive frontend geometry");
    }
    if (geometry_.chunk_left_context < 0 || geometry_.chunk_right_context < 0 || geometry_.chunk_len <= 0) {
        throw std::invalid_argument("Sortformer v2 stream scheduler received invalid chunk geometry");
    }
    reset();
}

void SortformerV2StreamScheduler::reset() {
    audio_.clear();
    audio_base_sample_ = 0;
    audio_received_samples_ = 0;
    logical_audio_end_samples_ = 0;
    mel_produced_ = 0;
    mel_consumed_ = 0;
    emitted_frames_ = 0;
    started_ = false;
    finished_ = false;
}

void SortformerV2StreamScheduler::append_audio(const runtime::AudioChunk & chunk) {
    if (chunk.sample_rate != frontend_.sample_rate) {
        throw std::invalid_argument("Sortformer v2 stream scheduler requires the model sample rate");
    }
    if (chunk.channels <= 0 || chunk.samples.size() % static_cast<size_t>(chunk.channels) != 0) {
        throw std::invalid_argument("Sortformer v2 audio chunk has invalid channel layout");
    }
    if (!started_) {
        if (chunk.start_sample < 0) {
            throw std::invalid_argument("Sortformer v2 audio chunk starts before sample zero");
        }
        audio_received_samples_ = chunk.start_sample;
        audio_base_sample_ = chunk.start_sample;
        started_ = true;
    }
    if (chunk.start_sample != audio_received_samples_) {
        throw std::invalid_argument("Sortformer v2 audio chunks must be contiguous");
    }
    const auto mono = audio::mixdown_interleaved_to_mono_average(chunk.samples, chunk.channels);
    audio_.insert(audio_.end(), mono.begin(), mono.end());
    audio_received_samples_ += static_cast<int64_t>(mono.size());
    logical_audio_end_samples_ = audio_received_samples_;
}

int64_t SortformerV2StreamScheduler::available_mel_frames(bool final_flush) const noexcept {
    if (logical_audio_end_samples_ < frontend_.n_fft / 2) {
        return 0;
    }
    const int64_t signal_samples = logical_audio_end_samples_ - frontend_.n_fft / 2;
    if (final_flush) {
        return (signal_samples + frontend_.hop_length - 1) / frontend_.hop_length;
    }
    return signal_samples / frontend_.hop_length + 1;
}

SortformerV2StreamWindow SortformerV2StreamScheduler::make_window(int64_t end_mel, bool final_flush) const {
    const int64_t sub = subsampling_factor_;
    const int64_t left_mel_max = geometry_.chunk_left_context * sub;
    const int64_t right_mel_max = geometry_.chunk_right_context * sub;
    const int64_t start_mel = mel_consumed_;
    const int64_t left_mel = std::min(left_mel_max, start_mel);
    const int64_t right_mel = std::min(right_mel_max, mel_produced_ - end_mel);
    const int64_t window_mel_start = start_mel - left_mel;
    const int64_t window_mel_frames = end_mel + right_mel - window_mel_start;
    const int64_t sample_start = window_mel_start * frontend_.hop_length;
    const int64_t stft_pre_roll = frontend_.n_fft / 2;
    const bool initial_window = window_mel_start == 0;
    // NeMo's mid-stream frontend receives n_fft/2 real samples before the
    // first requested mel center and uses center=false. That makes local
    // frame zero equal to the global mel frame at window_mel_start. The
    // initial window instead uses the centered, constant-padded frontend.
    const int64_t feature_sample_start = initial_window
        ? 0
        : sample_start - stft_pre_roll;
    const int64_t feature_sample_end =
        sample_start + (window_mel_frames - 1) * frontend_.hop_length + stft_pre_roll;

    SortformerV2StreamWindow window;
    window.sample_start = sample_start;
    window.feature_sample_start = feature_sample_start;
    window.mel_start = window_mel_start;
    window.mel_frames = window_mel_frames;
    window.new_mel_frames = end_mel - start_mel;
    window.encoder_frame_start = start_mel / sub;
    window.encoder_frames = final_flush
        ? (window.new_mel_frames + sub - 1) / sub
        : window.new_mel_frames / sub;
    window.left_context_frames = left_mel / sub;
    window.right_context_frames = (right_mel + sub - 1) / sub;
    window.left_reflect_samples = initial_window ? stft_pre_roll : 0;
    window.right_zero_pad_samples = std::max<int64_t>(feature_sample_end - audio_received_samples_, 0);
    window.final_flush = final_flush;

    const int64_t sample_count = std::max<int64_t>(feature_sample_end - feature_sample_start, 0);
    window.mono_samples.assign(static_cast<size_t>(sample_count), 0.0f);
    const int64_t copy_start = std::max(feature_sample_start, audio_base_sample_);
    const int64_t copy_end = std::min(feature_sample_end, audio_received_samples_);
    if (copy_end > copy_start) {
        const auto src_offset = static_cast<size_t>(copy_start - audio_base_sample_);
        const auto dst_offset = static_cast<size_t>(copy_start - feature_sample_start);
        const auto count = static_cast<size_t>(copy_end - copy_start);
        std::copy_n(audio_.data() + src_offset, count, window.mono_samples.data() + dst_offset);
    }
    return window;
}

void SortformerV2StreamScheduler::trim_audio() {
    const int64_t keep_mel = std::max<int64_t>(
        mel_consumed_ - geometry_.chunk_left_context * subsampling_factor_, 0);
    const int64_t keep_sample = std::max<int64_t>(
        keep_mel * frontend_.hop_length - frontend_.n_fft / 2, 0);
    if (keep_sample <= audio_base_sample_) {
        return;
    }
    const int64_t erase_count = std::min<int64_t>(keep_sample - audio_base_sample_, static_cast<int64_t>(audio_.size()));
    audio_.erase(audio_.begin(), audio_.begin() + erase_count);
    audio_base_sample_ += erase_count;
}

std::vector<SortformerV2StreamWindow> SortformerV2StreamScheduler::drain(bool final_flush) {
    const int64_t chunk_mel = geometry_.chunk_len * subsampling_factor_;
    const int64_t right_mel = geometry_.chunk_right_context * subsampling_factor_;
    std::vector<SortformerV2StreamWindow> windows;
    mel_produced_ = available_mel_frames(final_flush);
    while (true) {
        const int64_t start = mel_consumed_;
        int64_t end = start + chunk_mel;
        if (!final_flush) {
            if (end + right_mel > mel_produced_) {
                break;
            }
        } else {
            end = std::min(end, mel_produced_);
            if (end <= start) {
                break;
            }
        }
        windows.push_back(make_window(end, final_flush));
        mel_consumed_ = end;
        emitted_frames_ += windows.back().encoder_frames;
        trim_audio();
    }
    return windows;
}

std::vector<SortformerV2StreamWindow> SortformerV2StreamScheduler::push_audio(const runtime::AudioChunk & chunk) {
    if (finished_) {
        return {};
    }
    append_audio(chunk);
    return drain(false);
}

std::vector<SortformerV2StreamWindow> SortformerV2StreamScheduler::finalize() {
    if (finished_) {
        return {};
    }
    finished_ = true;
    if (!started_ || audio_received_samples_ == 0) {
        return {};
    }
    const int64_t actual_audio_end = audio_received_samples_;
    const int64_t tail = frontend_.n_fft / 2;
    if (!audio_.empty() && tail > 0) {
        const float preemphasis = frontend_.preemphasis;
        float sample = audio_.back();
        for (int64_t i = 0; i < tail; ++i) {
            sample *= preemphasis;
            audio_.push_back(sample);
        }
        audio_received_samples_ += tail;
    }
    // The decay tail makes pre-emphasis of the synthetic samples exactly zero,
    // matching NeMo's post-pre-emphasis right padding. Keep the logical end
    // based on real audio so ceil(signal/hop) remains the emitted mel count.
    logical_audio_end_samples_ = actual_audio_end + tail;
    return drain(true);
}

}  // namespace engine::community_models::sortformer_diar
