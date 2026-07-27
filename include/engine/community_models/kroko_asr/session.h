#pragma once

#include "engine/community_models/kroko_asr/assets.h"
#include "engine/community_models/kroko_asr/decoder.h"
#include "engine/community_models/kroko_asr/encoder.h"
#include "engine/community_models/kroko_asr/tokenizer.h"
#include "engine/community_models/kroko_asr/zipformer.h"
#include "engine/framework/runtime/session_base.h"

#include <chrono>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::kroko_asr {

class KrokoASRSession final
    : public runtime::RuntimeSessionBase,
      public runtime::IOfflineVoiceTaskSession,
      public runtime::IStreamingVoiceTaskSession {
public:
    KrokoASRSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const KrokoASRAssets> assets);
    ~KrokoASRSession() override;

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(
        const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(
        const runtime::TaskRequest & request) override;
    runtime::StreamingPolicy streaming_policy() const override;
    void start_stream(const runtime::TaskRequest & request) override;
    void set_stream_event_sink(runtime::StreamEventCallback sink) override;
    void reset() override;
    runtime::StreamEvent process_audio_chunk(
        const runtime::AudioChunk & chunk) override;
    runtime::TaskResult finish_stream() override;
    runtime::TaskResult finalize() override;

private:
    runtime::StreamEvent process_streaming_audio(bool final);
    runtime::TaskResult make_result(
        const KrokoDecodedTokens & decoded,
        int64_t audio_samples,
        const std::string & language) const;
    std::string request_language(
        const runtime::TaskRequest & request) const;

    runtime::TaskSpec task_;
    std::shared_ptr<const KrokoASRAssets> assets_;
    KrokoTokenizer tokenizer_;
    KrokoGreedyDecoder decoder_;
    KrokoEncoderRuntime subsampling_;
    KrokoZipformerRuntime zipformer_;
    std::vector<float> chunk_scratch_;
    runtime::AudioBuffer streaming_audio_;
    runtime::StreamEventCallback stream_event_sink_;
    std::string streaming_language_;
    int64_t processed_feature_offset_ = 0;
    int64_t streaming_total_samples_ = 0;
    int64_t streaming_encoder_chunks_ = 0;
    size_t streaming_peak_buffer_values_ = 0;
    bool stream_started_ = false;
    std::chrono::steady_clock::time_point stream_start_{};
};

}  // namespace engine::models::kroko_asr
