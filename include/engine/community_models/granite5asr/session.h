#pragma once

#include "engine/framework/audio/chunking.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/community_models/granite5asr/assets.h"
#include "engine/community_models/granite5asr/encoder.h"
#include "engine/community_models/granite5asr/frontend.h"

#include <memory>
#include <string>
#include <vector>

namespace engine::community_models::granite5asr {

class Granite5ASRSessionBase : public runtime::RuntimeSessionBase {
public:
    Granite5ASRSessionBase(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const Granite5ASRAssets> assets);
    ~Granite5ASRSessionBase() override;

protected:
    std::string family_impl() const;
    runtime::VoiceTaskKind task_kind_impl() const;
    runtime::RunMode run_mode_impl() const;

    runtime::Transcript transcribe_audio(
        const runtime::AudioBuffer & audio,
        const std::unordered_map<std::string, std::string> & options);

    runtime::IOfflineVoiceTaskSession & vad_session();

    runtime::TaskSpec task_;
    std::shared_ptr<const Granite5ASRAssets> assets_;
    Granite5Frontend frontend_;
    std::unique_ptr<Granite5EncoderRuntime> encoder_;
    std::string vad_model_path_;
    std::unique_ptr<runtime::ILoadedVoiceModel> vad_model_;
    std::unique_ptr<runtime::IOfflineVoiceTaskSession> vad_session_;
};

class Granite5ASROfflineSession final
    : public Granite5ASRSessionBase
    , public runtime::IOfflineVoiceTaskSession {
public:
    Granite5ASROfflineSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const Granite5ASRAssets> assets);

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;
};

class Granite5ASRStreamingSession final
    : public Granite5ASRSessionBase
    , public runtime::IStreamingVoiceTaskSession {
public:
    Granite5ASRStreamingSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const Granite5ASRAssets> assets);

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::StreamingPolicy streaming_policy() const override;
    void start_stream(const runtime::TaskRequest & request) override;
    void set_stream_event_sink(runtime::StreamEventCallback sink) override;
    void reset() override;
    runtime::StreamEvent process_audio_chunk(const runtime::AudioChunk & chunk) override;
    runtime::TaskResult finish_stream() override;
    runtime::TaskResult finalize() override;

private:
    runtime::StreamEventCallback stream_event_sink_;
    runtime::AudioBuffer streaming_audio_;
    runtime::TaskRequest streaming_request_;
};

class Granite5ASRLoadedModel final : public runtime::ILoadedVoiceModel {
public:
    Granite5ASRLoadedModel(
        runtime::ModelMetadata metadata,
        runtime::CapabilitySet capabilities,
        std::shared_ptr<const Granite5ASRAssets> assets);

    const runtime::ModelMetadata & metadata() const noexcept override;
    const runtime::CapabilitySet & capabilities() const noexcept override;
    std::unique_ptr<runtime::IVoiceTaskSession> create_task_session(
        const runtime::TaskSpec & task,
        const runtime::SessionOptions & options) const override;

private:
    runtime::ModelMetadata metadata_;
    runtime::CapabilitySet capabilities_;
    std::shared_ptr<const Granite5ASRAssets> assets_;
};

std::unique_ptr<Granite5ASRLoadedModel> load_granite5asr_model(
    const runtime::ModelLoadRequest & request);

std::shared_ptr<runtime::IVoiceModelLoader> make_granite5asr_loader();

}  // namespace engine::community_models::granite5asr
