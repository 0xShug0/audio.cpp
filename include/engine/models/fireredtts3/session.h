#pragma once

#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/framework/model_spec/metadata.h"
#include "engine/models/fireredtts3/assets.h"
#include "engine/models/fireredtts3/tokenizer_text.h"

#include <memory>

namespace engine::models::fireredtts3 {

class FireRedTTS3BaseRuntime;
class FireRedTTS3InstructRuntime;
class FireRedTTS3StreamSession;

std::shared_ptr<engine::runtime::IVoiceModelLoader> make_fireredtts3_loader();

class FireRedTTS3Session final
    : public engine::runtime::RuntimeSessionBase
    , public engine::runtime::IOfflineVoiceTaskSession
    , public engine::runtime::IStreamingVoiceTaskSession {
public:
    FireRedTTS3Session(
        engine::runtime::TaskSpec task,
        engine::runtime::SessionOptions options,
        std::shared_ptr<const FireRedTTS3Assets> assets,
        std::shared_ptr<const engine::model_spec::ModelContract> contract);
    ~FireRedTTS3Session() override;

    std::string family() const override;
    engine::runtime::VoiceTaskKind task_kind() const override;
    engine::runtime::RunMode run_mode() const override;
    void prepare(const engine::runtime::SessionPreparationRequest & request) override;
    engine::runtime::TaskResult run(const engine::runtime::TaskRequest & request) override;

    // 流式（增量）接口
    engine::runtime::StreamingPolicy streaming_policy() const override;
    void start_stream(const engine::runtime::TaskRequest & request) override;
    std::optional<engine::runtime::StreamEvent> next_stream_event() override;
    void set_stream_event_sink(engine::runtime::StreamEventCallback sink) override;
    engine::runtime::TaskResult finish_stream() override;
    void reset() override;
    engine::runtime::StreamEvent process_audio_chunk(const engine::runtime::AudioChunk & chunk) override;
    engine::runtime::TaskResult finalize() override;

private:
    void initialize_stream_request(const engine::runtime::TaskRequest & request);

    engine::runtime::TaskSpec task_;
    std::shared_ptr<const FireRedTTS3Assets> assets_;
    std::shared_ptr<const engine::model_spec::ModelContract> contract_;
    std::unique_ptr<FireRedTTS3TextTokenizer> tokenizer_;
    std::unique_ptr<FireRedTTS3BaseRuntime> runtime_;
    std::unique_ptr<FireRedTTS3InstructRuntime> instruct_runtime_;
    bool mem_saver_ = false;

    // 流式状态
    bool stream_started_ = false;
    engine::runtime::TaskRequest stream_request_;
    std::vector<int64_t> stream_chunk_patches_;
    size_t stream_chunk_index_ = 0;
    engine::runtime::AudioBuffer stream_merged_audio_;
    std::string stream_generated_text_;
    std::unique_ptr<FireRedTTS3StreamSession> stream_session_;
};

}  // namespace engine::models::fireredtts3
