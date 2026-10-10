#pragma once
#include "engine/community_models/parakeet_tdt/session.h"

#include <chrono>

namespace engine::community_models::parakeet_tdt {

// Original Phonon live policy: re-decode the current phrase, publish revisable
// snapshots, and commit once at silence/end/cap. No encoder-cache claim.
class PhononStreamingSession final : public ParakeetTDTSessionBase,
                                     public runtime::IStreamingVoiceTaskSession {
public:
    using ParakeetTDTSessionBase::ParakeetTDTSessionBase;
    std::string family() const override { return family_impl(); }
    runtime::VoiceTaskKind task_kind() const override { return task_kind_impl(); }
    runtime::RunMode run_mode() const override { return run_mode_impl(); }
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::StreamingPolicy streaming_policy() const override;
    void start_stream(const runtime::TaskRequest & request) override;
    void set_stream_event_sink(runtime::StreamEventCallback sink) override { sink_ = std::move(sink); }
    void reset() override;
    runtime::StreamEvent process_audio_chunk(const runtime::AudioChunk & chunk) override;
    runtime::TaskResult finalize() override;

private:
    void reset_segment();
    runtime::StreamEvent feed_block(const std::vector<float> & block);
    runtime::StreamEvent finish_segment();
    ParakeetDecodedText decode_segment();
    void publish(const runtime::StreamEvent & event);
    void abort();

    runtime::StreamEventCallback sink_;
    ParakeetDecodeOptions options_;
    runtime::AudioBuffer segment_{16000, 1, {}};
    std::vector<float> pending_;
    int64_t received_ = 0, segment_start_ = 0, last_voice_ = 0, next_partial_ = 5600;
    int64_t committed_tokens_ = 0;
    float peak_rms_ = 0;
    bool voiced_ = false, active_ = false;
    std::chrono::steady_clock::time_point wall_start_;
    std::string text_;
    std::vector<runtime::WordTimestamp> words_;
};
} // namespace engine::community_models::parakeet_tdt
