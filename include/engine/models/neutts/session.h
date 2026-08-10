#pragma once

#include "engine/framework/model_spec/metadata.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/models/neutts/ar.h"
#include "engine/models/neutts/assets.h"
#include "engine/models/neutts/codec.h"
#include "engine/models/neutts/prompt.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace engine::models::neutts {

std::shared_ptr<runtime::IVoiceModelLoader> make_neutts_loader();

struct NeuTTSRequest {
    std::string text;
    std::string speaker = "emily";
    std::string emotion = "neutral";
    NeuTTSGenerationOptions generation;
};

class NeuTTSSession final
    : public runtime::RuntimeSessionBase,
      public runtime::IOfflineVoiceTaskSession,
      public runtime::IStreamingVoiceTaskSession {
public:
    NeuTTSSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const NeuTTSAssets> assets,
        std::shared_ptr<const engine::model_spec::ModelContract> contract);
    ~NeuTTSSession() override;

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
    NeuTTSRequest parse_request(const runtime::TaskRequest & request) const;
    runtime::AudioBuffer synthesize(const NeuTTSRequest & request);

    runtime::TaskSpec task_;
    std::shared_ptr<const NeuTTSAssets> assets_;
    std::shared_ptr<const engine::model_spec::ModelContract> contract_;
    NeuTTSPromptBuilder prompt_builder_;
    std::unique_ptr<NeuTTSARRuntime> ar_;
    std::unique_ptr<NeuTTSCodecDecoderRuntime> codec_;
    std::vector<NeuTTSRequest> streaming_requests_;
    std::vector<runtime::AudioBuffer> streaming_chunks_;
    size_t streaming_index_ = 0;
};

}  // namespace engine::models::neutts
