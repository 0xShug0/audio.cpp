#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/framework/model_spec/metadata.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/models/meanvc2/assets.h"
#include "engine/models/meanvc2/asr_encoder.h"
#include "engine/models/meanvc2/audio_pipeline.h"
#include "engine/models/meanvc2/speaker_encoder.h"
#include "engine/models/meanvc2/flow_sampler_runtime.h"
#include "engine/models/meanvc2/vocoder.h"

#include <memory>
#include <string>
#include <cstdint>
#include <vector>

namespace engine::models::meanvc2 {

std::shared_ptr<runtime::IVoiceModelLoader> make_meanvc2_loader();

class MeanVC2Session final
    : public runtime::RuntimeSessionBase,
      public runtime::IOfflineVoiceTaskSession,
      public runtime::IStreamingVoiceTaskSession {
public:
    MeanVC2Session(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const MeanVC2Assets> assets,
        std::shared_ptr<const engine::model_spec::ModelContract> contract);
    ~MeanVC2Session() override;

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
    runtime::TaskResult convert(const runtime::TaskRequest & request, const runtime::AudioBuffer & source_audio);
    runtime::AudioBuffer process_streaming_audio(const runtime::AudioBuffer & audio);
    runtime::AudioBuffer drain_streaming_conditions(bool finish);

    runtime::TaskSpec task_;
    std::shared_ptr<const MeanVC2Assets> assets_;
    std::shared_ptr<const engine::model_spec::ModelContract> contract_;
    engine::core::ExecutionContext execution_context_;
    engine::core::ExecutionContext vocoder_execution_context_;
    MeanVC2StreamingFrontend frontend_;
    MeanVC2BnStreamAdapter bn_adapter_;
    std::unique_ptr<MeanVC2AsrEncoderRuntime> asr_encoder_;
    std::unique_ptr<MeanVC2SpeakerEncoderRuntime> speaker_encoder_;
    std::unique_ptr<MeanVC2FlowSamplerRuntime> flow_;
    std::unique_ptr<MeanVC2VocoderRuntime> vocoder_;
    runtime::TaskRequest streaming_request_;
    runtime::AudioBuffer streaming_output_;
    std::vector<float> streaming_bn_buffer_;
    std::vector<float> streaming_speaker_embedding_;
    MeanVC2GtmMemory streaming_speaker_memory_;
    uint64_t streaming_seed_ = 42;
    runtime::StreamEventCallback stream_event_sink_;
    bool stream_started_ = false;
};

}  // namespace engine::models::meanvc2
