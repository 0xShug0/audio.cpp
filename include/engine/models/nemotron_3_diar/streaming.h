#pragma once

#include "engine/models/nemotron_3_diar/assets.h"
#include "engine/framework/runtime/session.h"

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace engine::models::nemotron_3_diar {

struct StreamWindow {
    std::vector<float> mono_samples;
    int64_t mel_start = 0;
    int64_t mel_frames = 0;
    int64_t encoder_frames = 0;
    int64_t left_context_frames = 0;
    int64_t right_context_frames = 0;
};

class StreamScheduler {
public:
    StreamScheduler(const FeatureConfig & frontend, const StreamingConfig & streaming, int64_t subsampling_factor);

    std::vector<StreamWindow> push_audio(const runtime::AudioChunk & chunk);
    std::vector<StreamWindow> finalize();
    void reset();

private:
    std::vector<StreamWindow> drain(bool final_flush);
    void append_audio(const runtime::AudioChunk & chunk);
    int64_t available_mel_frames(bool final_flush) const noexcept;
    StreamWindow make_window(int64_t end_mel, bool final_flush) const;
    void trim_audio();

    FeatureConfig frontend_;
    StreamingConfig streaming_;
    int64_t subsampling_factor_ = 0;
    std::vector<float> audio_;
    int64_t audio_base_sample_ = 0;
    int64_t audio_received_samples_ = 0;
    int64_t logical_audio_end_samples_ = 0;
    int64_t mel_produced_ = 0;
    int64_t mel_consumed_ = 0;
    bool started_ = false;
    bool finished_ = false;
};

class AoscState {
public:
    AoscState(
        const StreamingConfig & config,
        int speakers,
        int embedding_dim,
        std::vector<float> silence_embedding);

    void update(
        const float * chunk_embeddings,
        int window_frames,
        const float * probabilities,
        int left_context,
        int right_context);

    int speaker_cache_frames() const noexcept { return speaker_cache_frames_; }
    int fifo_frames() const noexcept { return fifo_frames_; }
    const std::vector<float> & speaker_cache() const noexcept { return speaker_cache_; }
    const std::vector<float> & fifo() const noexcept { return fifo_; }

private:
    void compress(const std::vector<float> & probabilities);

    StreamingConfig config_;
    int speakers_ = 0;
    int embedding_dim_ = 0;
    std::vector<float> speaker_cache_;
    std::vector<float> speaker_cache_probabilities_;
    int speaker_cache_frames_ = 0;
    std::vector<float> fifo_;
    int fifo_frames_ = 0;
    std::vector<float> silence_embedding_;
};

StreamingConfig streaming_profile(
    const StreamingConfig & model_defaults,
    const std::unordered_map<std::string, std::string> & options);

}  // namespace engine::models::nemotron_3_diar
