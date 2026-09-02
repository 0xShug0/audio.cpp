#pragma once

#include "engine/community_models/mira_tts/assets.h"
#include "engine/framework/model_spec/metadata.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/session_base.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace engine::community_models::mira_tts {

std::shared_ptr<runtime::IVoiceModelLoader> make_mira_tts_loader();

class MiraPromptBuilder;
class MiraSpeakerEncoder;
class MiraGenerator;
class MiraAcousticProcessor;
class MiraDecoder;

class MiraTTSOfflineSession final : public runtime::RuntimeSessionBase,
                                    public runtime::IOfflineVoiceTaskSession,
                                    public runtime::IStreamingVoiceTaskSession {
public:
    MiraTTSOfflineSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const MiraTTSAssets> assets,
        std::shared_ptr<const engine::model_spec::ModelContract> contract);
    ~MiraTTSOfflineSession() override;

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;
    runtime::StreamingPolicy streaming_policy() const override;
    void start_stream(const runtime::TaskRequest & request) override;
    std::optional<runtime::StreamEvent> next_stream_event() override;
    void set_stream_event_sink(runtime::StreamEventCallback sink) override;
    runtime::TaskResult finish_stream() override;
    void reset() override;
    runtime::StreamEvent process_audio_chunk(const runtime::AudioChunk & chunk) override;
    runtime::TaskResult finalize() override;

private:
    const runtime::AudioBuffer & reference_audio(
        const runtime::TaskRequest & request) const;
    runtime::AudioBuffer synthesize_text(
        const std::string & text,
        const std::vector<int32_t> & context_codes,
        const MiraGenerationOptions & options);

    runtime::TaskSpec task_;
    std::shared_ptr<const MiraTTSAssets> assets_;
    std::shared_ptr<const engine::model_spec::ModelContract> contract_;
    std::optional<runtime::AudioBuffer> prepared_reference_;
    std::unique_ptr<MiraPromptBuilder> prompt_;
    std::unique_ptr<MiraSpeakerEncoder> speaker_encoder_;
    std::unique_ptr<MiraGenerator> generator_;
    std::unique_ptr<MiraAcousticProcessor> processor_;
    std::unique_ptr<MiraDecoder> decoder_;
    std::vector<int32_t> streaming_context_codes_;
    std::vector<std::string> streaming_text_chunks_;
    std::vector<runtime::AudioBuffer> streaming_audio_chunks_;
    std::optional<MiraGenerationOptions> streaming_generation_;
    runtime::StreamEventCallback stream_sink_;
    size_t streaming_chunk_index_ = 0;
    bool streaming_started_ = false;
};

}  // namespace engine::community_models::mira_tts
