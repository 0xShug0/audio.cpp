#pragma once
#include "engine/community_models/parakeet_tdt/session.h"

namespace engine::community_models::parakeet_tdt {

// Original Phonon live policy: re-decode the current phrase, publish revisable
// snapshots, and commit once at silence/end/cap. No encoder-cache claim.
class PhononStreamingSession final : public ParakeetTDTSessionBase,
                                     public runtime::StreamingAsrSessionBase {
public:
    using ParakeetTDTSessionBase::ParakeetTDTSessionBase;
    std::string family() const override { return family_impl(); }
    runtime::VoiceTaskKind task_kind() const override { return task_kind_impl(); }
    runtime::RunMode run_mode() const override { return run_mode_impl(); }
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::StreamingPolicy streaming_policy() const override;

private:
    void begin_stream(const runtime::TaskRequest & request) override;
    void reset_stream_state() override;
    void consume_audio(const runtime::AudioChunk & chunk) override;
    void complete_stream(runtime::TaskResult & result) override;
    void reset_segment();
    void feed_block(const std::vector<float> & block);
    void finish_segment();
    ParakeetDecodedText decode_segment();

    ParakeetDecodeOptions options_;
    runtime::AudioBuffer segment_{16000, 1, {}};
    std::vector<float> pending_;
    int64_t received_ = 0, segment_start_ = 0, last_voice_ = 0, next_partial_ = 5600;
    int64_t committed_tokens_ = 0;
    float peak_rms_ = 0;
    bool voiced_ = false;
};
} // namespace engine::community_models::parakeet_tdt
