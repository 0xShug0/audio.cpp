#pragma once

#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/community_models/liveavatar/assets.h"
#include "engine/community_models/liveavatar/pipeline.h"

#include <memory>
#include <string>

namespace engine::community_models::liveavatar {

class LiveAvatarSession final
    : public engine::runtime::RuntimeSessionBase
    , public engine::runtime::IOfflineVoiceTaskSession
    , public engine::runtime::IStreamingVoiceTaskSession {
public:
    LiveAvatarSession(
        engine::runtime::TaskSpec task,
        engine::runtime::SessionOptions options,
        std::shared_ptr<const LiveAvatarAssets> assets);
    ~LiveAvatarSession() override;

    std::string family() const override;
    engine::runtime::VoiceTaskKind task_kind() const override;
    engine::runtime::RunMode run_mode() const override;
    void prepare(const engine::runtime::SessionPreparationRequest & request) override;
    engine::runtime::TaskResult run(const engine::runtime::TaskRequest & request) override;
    engine::runtime::StreamingPolicy streaming_policy() const override;
    void start_stream(const engine::runtime::TaskRequest & request) override;
    void set_stream_event_sink(engine::runtime::StreamEventCallback sink) override;
    engine::runtime::TaskResult finish_stream() override;
    void reset() override;
    engine::runtime::StreamEvent process_audio_chunk(const engine::runtime::AudioChunk & chunk) override;
    engine::runtime::TaskResult finalize() override;

private:
    LiveAvatarGenerateRequest make_request(const engine::runtime::TaskRequest & request) const;

    engine::runtime::TaskSpec task_;
    std::shared_ptr<const LiveAvatarAssets> assets_;
    std::unique_ptr<LiveAvatarPipelineRuntime> pipeline_;
    engine::runtime::StreamEventCallback stream_event_sink_;
    engine::runtime::TaskResult stream_result_;
    bool stream_started_ = false;
};

class LiveAvatarLoadedModel final : public engine::runtime::ILoadedVoiceModel {
public:
    explicit LiveAvatarLoadedModel(std::shared_ptr<const LiveAvatarAssets> assets);

    const engine::runtime::ModelMetadata & metadata() const noexcept override;
    const engine::runtime::CapabilitySet & capabilities() const noexcept override;
    std::unique_ptr<engine::runtime::IVoiceTaskSession> create_task_session(
        const engine::runtime::TaskSpec & task,
        const engine::runtime::SessionOptions & options) const override;

private:
    engine::runtime::ModelMetadata metadata_;
    engine::runtime::CapabilitySet capabilities_;
    std::shared_ptr<const LiveAvatarAssets> assets_;
};

std::shared_ptr<engine::runtime::IVoiceModelLoader> make_liveavatar_loader();

}  // namespace engine::community_models::liveavatar
