#pragma once

#include "engine/community_models/kroko_asr/assets.h"
#include "engine/community_models/kroko_asr/decoder.h"
#include "engine/community_models/kroko_asr/encoder.h"
#include "engine/community_models/kroko_asr/tokenizer.h"
#include "engine/community_models/kroko_asr/zipformer.h"
#include "engine/framework/model_spec/metadata.h"
#include "engine/framework/runtime/session_base.h"

#include <chrono>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace engine::models::kroko_asr {

std::shared_ptr<runtime::IVoiceModelLoader> make_kroko_asr_loader();

class KrokoASRSession final
    : public runtime::RuntimeSessionBase,
      public runtime::IOfflineVoiceTaskSession,
      public runtime::IStreamingVoiceTaskSession {
public:
    KrokoASRSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const KrokoASRAssets> assets,
        std::shared_ptr<const engine::model_spec::ModelContract> contract);
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
    KrokoDecodedTokens combined_decoded() const;
    void configure_request(
        const runtime::TaskRequest & request);
    bool endpoint_detected() const;
    std::optional<runtime::SpeechSegment> close_endpoint_segment(
        int64_t audio_samples,
        bool final);
    void archive_decoder_and_reset(int64_t frame_offset);
    void append_streaming_chunk(
        const runtime::AudioChunk & chunk);
    void flush_streaming_resampler();
    std::string request_language(
        const runtime::TaskRequest & request) const;

    runtime::TaskSpec task_;
    std::shared_ptr<const KrokoASRAssets> assets_;
    std::shared_ptr<const engine::model_spec::ModelContract> contract_;
    KrokoTokenizer tokenizer_;
    KrokoTransducerDecoder decoder_;
    KrokoEncoderRuntime subsampling_;
    KrokoZipformerRuntime zipformer_;
    std::vector<float> chunk_scratch_;
    KrokoDecodedTokens completed_decoded_;
    std::vector<runtime::SpeechSegment> endpoint_segments_;
    runtime::AudioBuffer streaming_audio_;
    runtime::StreamEventCallback stream_event_sink_;
    std::string streaming_language_;
    std::vector<float> streaming_resampler_source_;
    int64_t processed_feature_offset_ = 0;
    int64_t streaming_total_samples_ = 0;
    int64_t streaming_source_offset_ = 0;
    int64_t streaming_source_frames_ = 0;
    int64_t streaming_next_output_sample_ = 0;
    int64_t streaming_encoder_chunks_ = 0;
    int64_t endpoint_frame_offset_ = 0;
    int64_t endpoint_segment_start_sample_ = 0;
    size_t streaming_peak_buffer_values_ = 0;
    size_t streaming_peak_source_values_ = 0;
    int streaming_source_sample_rate_ = 0;
    int streaming_source_channels_ = 0;
    float endpoint_rule1_silence_ = 2.4F;
    float endpoint_rule2_silence_ = 1.2F;
    float endpoint_rule3_utterance_ = 20.0F;
    bool endpoint_enabled_ = false;
    bool streaming_received_audio_ = false;
    bool stream_started_ = false;
    std::chrono::steady_clock::time_point stream_start_{};
};

}  // namespace engine::models::kroko_asr
