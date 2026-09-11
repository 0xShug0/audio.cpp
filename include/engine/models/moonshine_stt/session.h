#pragma once

#include "engine/framework/runtime/session_base.h"
#include "engine/models/moonshine_stt/runtime.h"

#include <memory>
#include <string>

namespace engine::models::moonshine_stt {

struct MoonshineAssets;
struct MoonshineWeights;

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
    runtime::TaskSpec task_;
    std::shared_ptr<const MoonshineAssets> assets_;
    MoonshineRuntimeConfig runtime_config_;
    std::shared_ptr<const MoonshineWeights> weights_;
    runtime::StreamEventCallback stream_event_sink_;
    runtime::AudioBuffer streaming_audio_;
    runtime::TaskRequest streaming_request_;
    bool stream_started_ = false;
};

}  // namespace engine::models::moonshine_stt
