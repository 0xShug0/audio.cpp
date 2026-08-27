#pragma once

#include "engine/framework/model_spec/metadata.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/community_models/soprano_tts/assets.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace engine::community_models::soprano_tts {

std::shared_ptr<runtime::IVoiceModelLoader> make_soprano_tts_loader();

struct SopranoRequest {
    std::string text;
    SopranoGenerationOptions generation;
};

class SopranoTTSGenerator;
class SopranoDecoderRuntime;

class SopranoTTSOfflineSession final : public runtime::RuntimeSessionBase,
                                       public runtime::IOfflineVoiceTaskSession,
                                       public runtime::IStreamingVoiceTaskSession {
public:
    SopranoTTSOfflineSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const SopranoTTSAssets> assets,
        std::shared_ptr<const engine::model_spec::ModelContract> contract);
    ~SopranoTTSOfflineSession() override;

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
    SopranoRequest make_request(const runtime::TaskRequest & request) const;
    runtime::AudioBuffer synthesize(const SopranoRequest & request);
    // Streaming state
    std::optional<std::vector<std::string>> streaming_chunks_;
    size_t streaming_chunk_index_ = 0;
    std::vector<runtime::AudioBuffer> streaming_audio_;
    runtime::StreamEventCallback stream_sink_;

    runtime::TaskSpec task_;
    std::shared_ptr<const SopranoTTSAssets> assets_;
    std::shared_ptr<const engine::model_spec::ModelContract> contract_;
    std::unique_ptr<SopranoTTSGenerator> generator_;
    std::unique_ptr<SopranoDecoderRuntime> decoder_;
};

}  // namespace engine::community_models::soprano_tts
