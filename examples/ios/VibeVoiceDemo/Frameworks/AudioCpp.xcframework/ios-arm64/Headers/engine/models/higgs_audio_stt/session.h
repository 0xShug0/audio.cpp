#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/models/higgs_audio_stt/assets.h"
#include "engine/models/higgs_audio_stt/audio_encoder.h"
#include "engine/models/higgs_audio_stt/frontend_whisper.h"
#include "engine/models/higgs_audio_stt/postprocess.h"
#include "engine/models/higgs_audio_stt/prompt_asr.h"
#include "engine/models/higgs_audio_stt/text_decoder.h"
#include "engine/models/higgs_audio_stt/tokenizer_text.h"

#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace engine::runtime {
class ILoadedVoiceModel;
}

namespace engine::models::higgs_audio_stt {

class HiggsAudioSTTSession final
    : public runtime::RuntimeSessionBase
    , public runtime::IOfflineVoiceTaskSession
    , public runtime::IStreamingVoiceTaskSession {
public:
    HiggsAudioSTTSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const HiggsAudioSTTAssets> assets);
    ~HiggsAudioSTTSession() override;

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;
    runtime::StreamingPolicy streaming_policy() const override;
    void start_stream(const runtime::TaskRequest & request) override;
    void set_stream_event_sink(runtime::StreamEventCallback sink) override;
    void reset() override;
    runtime::StreamEvent process_audio_chunk(const runtime::AudioChunk & chunk) override;
    runtime::TaskResult finish_stream() override;
    runtime::TaskResult finalize() override;

private:
    struct AudioChunkPlan {
        runtime::TimeSpan source_span;
    };

    HiggsAudioSTTRequest make_request(const runtime::TaskRequest & request) const;
    std::vector<AudioChunkPlan> audio_chunk_plan(const runtime::TaskRequest & request) const;
    runtime::TaskResult run_single(const HiggsAudioSTTRequest & request);

    runtime::TaskSpec task_;
    std::shared_ptr<const HiggsAudioSTTAssets> assets_;
    size_t audio_encoder_graph_arena_bytes_ = 128ull * 1024ull * 1024ull;
    size_t text_decoder_prefill_graph_arena_bytes_ = 256ull * 1024ull * 1024ull;
    size_t text_decoder_decode_graph_arena_bytes_ = 256ull * 1024ull * 1024ull;
    size_t text_decoder_weight_context_bytes_ = 64ull * 1024ull * 1024ull;
    engine::assets::TensorStorageType audio_encoder_weight_storage_type_ = engine::assets::TensorStorageType::Native;
    engine::assets::TensorStorageType text_decoder_weight_storage_type_ = engine::assets::TensorStorageType::Native;
    HiggsAudioSTTTextTokenizer tokenizer_;
    HiggsAudioSTTWhisperFrontend frontend_;
    HiggsAudioSTTAudioEncoderRuntime audio_encoder_;
    HiggsAudioSTTTextDecoderRuntime text_decoder_;
    HiggsAudioSTTPromptBuilder prompt_builder_;
    HiggsAudioSTTPostprocessor postprocessor_;
    runtime::TaskRequest streaming_request_;
    runtime::TaskResult streaming_result_;
    runtime::StreamEventCallback stream_event_sink_;
    bool stream_started_ = false;
    int64_t streaming_chunks_processed_ = 0;
};

}  // namespace engine::models::higgs_audio_stt
