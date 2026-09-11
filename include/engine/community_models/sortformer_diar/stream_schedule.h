#pragma once

#include "engine/community_models/sortformer_diar/aosc_state.h"
#include "engine/framework/runtime/session.h"

#include <cstdint>
#include <vector>

namespace engine::community_models::sortformer_diar {

struct SortformerV2StreamWindow {
    std::vector<float> mono_samples;
    int64_t sample_start = 0;
    int64_t feature_sample_start = 0;
    int64_t mel_start = 0;
    int64_t mel_frames = 0;
    int64_t new_mel_frames = 0;
    int64_t encoder_frame_start = 0;
    int64_t encoder_frames = 0;
    int64_t left_context_frames = 0;
    int64_t right_context_frames = 0;
    int64_t left_reflect_samples = 0;
    int64_t right_zero_pad_samples = 0;
    bool final_flush = false;
};

class SortformerV2StreamScheduler {
public:
    SortformerV2StreamScheduler(
        const SortformerV2FeatureExtractorConfig & frontend,
        const SortformerV2DiarGeometry & geometry,
        int64_t subsampling_factor);

    std::vector<SortformerV2StreamWindow> push_audio(const runtime::AudioChunk & chunk);
    std::vector<SortformerV2StreamWindow> finalize();
    void reset();

    int64_t emitted_frames() const noexcept { return emitted_frames_; }
    int64_t consumed_mel_frames() const noexcept { return mel_consumed_; }
    int64_t produced_mel_frames() const noexcept { return mel_produced_; }
    bool finished() const noexcept { return finished_; }

private:
    std::vector<SortformerV2StreamWindow> drain(bool final_flush);
    void append_audio(const runtime::AudioChunk & chunk);
    int64_t available_mel_frames(bool final_flush) const noexcept;
    SortformerV2StreamWindow make_window(int64_t end_mel, bool final_flush) const;
    void trim_audio();

    SortformerV2FeatureExtractorConfig frontend_;
    SortformerV2DiarGeometry geometry_;
    int64_t subsampling_factor_ = 0;
    std::vector<float> audio_;
    int64_t audio_base_sample_ = 0;
    int64_t audio_received_samples_ = 0;
    int64_t logical_audio_end_samples_ = 0;
    int64_t mel_produced_ = 0;
    int64_t mel_consumed_ = 0;
    int64_t emitted_frames_ = 0;
    bool started_ = false;
    bool finished_ = false;
};

}  // namespace engine::community_models::sortformer_diar
