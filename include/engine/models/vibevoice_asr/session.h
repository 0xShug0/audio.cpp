#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/framework/sampling/torch_random.h"
#include "engine/models/vibevoice_asr/assets.h"
#include "engine/models/vibevoice_asr/frontend.h"
#include "engine/models/vibevoice_asr/postprocess.h"
#include "engine/models/vibevoice_asr/speech_encoder.h"
#include "engine/models/vibevoice_asr/text_decoder.h"
#include "engine/models/vibevoice_asr/tokenizer_text.h"

#include <cstddef>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace engine::runtime {
class ILoadedVoiceModel;
}

namespace engine::models::vibevoice_asr {

class VibeVoiceASRSession final
    : public runtime::RuntimeSessionBase
    , public runtime::IOfflineVoiceTaskSession
    , public runtime::IStreamingVoiceTaskSession {
public:
    VibeVoiceASRSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const VibeVoiceAssets> assets);
    ~VibeVoiceASRSession() override;

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
        runtime::TimeSpan keep_span;
    };

    VibeVoiceASRRequest make_request(const runtime::TaskRequest & request) const;
    std::vector<AudioChunkPlan> audio_chunk_plan(const runtime::TaskRequest & request);
    runtime::IOfflineVoiceTaskSession & vad_session();
    runtime::TaskResult run_single(const VibeVoiceASRRequest & request);
    std::vector<int32_t> generate_tokens(
        const VibeVoiceASRRequest & request,
        const VibeVoiceASRPrompt & prompt,
        VibeVoiceDecoderPrefillOutput prefill,
        uint64_t rng_call_offset,
        const std::function<void(const std::vector<int32_t> &)> & token_callback);
    std::vector<int32_t> generate_greedy_or_sample(
        const VibeVoiceASRRequest & request,
        const VibeVoiceASRPrompt & prompt,
        VibeVoiceDecoderPrefillOutput prefill,
        uint64_t rng_call_offset,
        const std::function<void(const std::vector<int32_t> &)> & token_callback);
    std::vector<int32_t> generate_beam(
        const VibeVoiceASRRequest & request,
        const VibeVoiceASRPrompt & prompt,
        VibeVoiceDecoderPrefillOutput prefill,
        const std::function<void(const std::vector<int32_t> &)> & token_callback);

    runtime::TaskSpec task_;
    std::shared_ptr<const VibeVoiceAssets> assets_;
    size_t tokenizer_weight_context_bytes_ = 512ull * 1024ull * 1024ull;
    size_t connector_weight_context_bytes_ = 128ull * 1024ull * 1024ull;
    size_t decoder_weight_context_bytes_ = 4096ull * 1024ull * 1024ull;
    assets::TensorStorageType tokenizer_weight_storage_type_ = assets::TensorStorageType::Native;
    assets::TensorStorageType connector_weight_storage_type_ = assets::TensorStorageType::Native;
    assets::TensorStorageType decoder_weight_storage_type_ = assets::TensorStorageType::Native;
    bool greedy_compare_bf16_ = false;
    sampling::TorchCudaSamplingPolicy sampling_policy_;
    VibeVoiceASRTextTokenizer tokenizer_;
    VibeVoiceASRFrontend frontend_;
    VibeVoiceASRSpeechEncoder speech_encoder_;
    VibeVoiceDecoderWeightsRuntime text_decoder_;
    VibeVoiceASRPostprocessor postprocessor_;
    std::filesystem::path vad_model_path_;
    std::unique_ptr<runtime::ILoadedVoiceModel> vad_model_;
    std::unique_ptr<runtime::IOfflineVoiceTaskSession> vad_session_;
    runtime::TaskRequest streaming_request_;
    runtime::TaskResult streaming_result_;
    runtime::StreamEventCallback stream_event_sink_;
    bool stream_started_ = false;
    int64_t streaming_chunks_processed_ = 0;
};

}  // namespace engine::models::vibevoice_asr
