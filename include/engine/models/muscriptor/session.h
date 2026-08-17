#pragma once

#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/framework/model_spec/metadata.h"
#include "engine/models/muscriptor/decoder.h"
#include "engine/models/muscriptor/frontend.h"
#include "engine/models/muscriptor/tokenizer.h"

#include <memory>

namespace engine::models::muscriptor {

std::shared_ptr<runtime::IVoiceModelLoader> make_muscriptor_loader();

class MuScriptorSession final
    : public runtime::RuntimeSessionBase
    , public runtime::IOfflineVoiceTaskSession
    , public runtime::IStreamingVoiceTaskSession {
public:
    MuScriptorSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const MuScriptorAssets> assets,
        std::shared_ptr<const engine::model_spec::ModelContract> contract);
    ~MuScriptorSession() override;

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;
    runtime::StreamingPolicy streaming_policy() const override;
    void start_stream(const runtime::TaskRequest & request) override;
    void set_stream_event_sink(runtime::StreamEventCallback sink) override;
    runtime::TaskResult finish_stream() override;
    void reset() override;
    runtime::StreamEvent process_audio_chunk(const runtime::AudioChunk & chunk) override;
    runtime::TaskResult finalize() override;

private:
    runtime::TaskResult transcribe(
        const runtime::TaskRequest & request,
        const runtime::StreamEventCallback & stream_sink);
    void emit_stream_events(
        const std::vector<MuScriptorEvent> & decoded,
        size_t & published_events,
        size_t completed_chunks,
        size_t total_chunks,
        const runtime::StreamEventCallback & stream_sink) const;

    runtime::TaskSpec task_;
    std::shared_ptr<const MuScriptorAssets> assets_;
    std::shared_ptr<const engine::model_spec::ModelContract> contract_;
    MuScriptorTokenizer tokenizer_;
    MuScriptorFrontend frontend_;
    MuScriptorDecoderRuntime decoder_;
    runtime::TaskRequest streaming_request_;
    runtime::StreamEventCallback stream_sink_;
    runtime::AudioBuffer streaming_audio_;
    bool stream_started_ = false;
};

}  // namespace engine::models::muscriptor
