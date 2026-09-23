#pragma once

#include "engine/framework/model_spec/metadata.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/models/moonshine_asr/runtime.h"

#include <filesystem>
#include <memory>
#include <string>

namespace engine::runtime {
class ILoadedVoiceModel;
}

namespace engine::models::moonshine_asr {

struct MoonshineAssets;
struct MoonshineWeights;

std::shared_ptr<runtime::IVoiceModelLoader> make_moonshine_asr_loader();

class MoonshineSTTSession final
    : public runtime::RuntimeSessionBase
    , public runtime::IOfflineVoiceTaskSession
    , public runtime::IStreamingVoiceTaskSession {
public:
    MoonshineSTTSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const MoonshineAssets> assets,
        std::shared_ptr<const engine::model_spec::ModelContract> contract);
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
    runtime::IOfflineVoiceTaskSession & vad_session();

    runtime::TaskSpec task_;
    std::shared_ptr<const MoonshineAssets> assets_;
    std::shared_ptr<const engine::model_spec::ModelContract> contract_;
    MoonshineRuntimeConfig runtime_config_;
    std::shared_ptr<const MoonshineWeights> weights_;
    std::filesystem::path vad_model_path_;
    std::unique_ptr<runtime::ILoadedVoiceModel> vad_model_;
    std::unique_ptr<runtime::IOfflineVoiceTaskSession> vad_session_;
    runtime::StreamEventCallback stream_event_sink_;
    runtime::AudioBuffer streaming_audio_;
    runtime::TaskRequest streaming_request_;
    bool stream_started_ = false;
};

}  // namespace engine::models::moonshine_asr
