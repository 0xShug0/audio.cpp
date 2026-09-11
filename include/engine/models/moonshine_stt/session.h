#pragma once

#include "engine/framework/runtime/session_base.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/models/moonshine_stt/assets.h"
#include "engine/models/moonshine_stt/weights.h"

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>

namespace engine::models::moonshine_stt {

class MoonshineSTTSession final
    : public runtime::RuntimeSessionBase
    , public runtime::IOfflineVoiceTaskSession
    , public runtime::IStreamingVoiceTaskSession {
public:
    MoonshineSTTSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const MoonshineAssets> assets);
    ~MoonshineSTTSession() override;

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
    runtime::TaskResult finalize() override;
    runtime::TaskResult finish_stream() override;

private:
    runtime::TaskResult transcribe(
        const runtime::AudioBuffer & audio,
        const std::unordered_map<std::string, std::string> & options);

    runtime::TaskSpec task_;
    std::shared_ptr<const MoonshineAssets> assets_;
    std::shared_ptr<const MoonshineWeights> weights_;
    runtime::StreamEventCallback stream_event_sink_;
    runtime::AudioBuffer streaming_audio_;
    runtime::TaskRequest streaming_request_;
    size_t weight_context_bytes_ = 256ull * 1024ull * 1024ull;
    size_t graph_arena_bytes_ = 512ull * 1024ull * 1024ull;
    engine::modules::GeluApproximation encoder_gelu_ = engine::modules::GeluApproximation::Quick;
    engine::assets::TensorStorageType encoder_weight_storage_type_ = engine::assets::TensorStorageType::Native;
    engine::assets::TensorStorageType decoder_weight_storage_type_ = engine::assets::TensorStorageType::Native;
    engine::assets::TensorStorageType conv_weight_storage_type_ = engine::assets::TensorStorageType::Native;
    bool cpu_blas_scheduler_ = true;
    bool stream_started_ = false;
};

}  // namespace engine::models::moonshine_stt
