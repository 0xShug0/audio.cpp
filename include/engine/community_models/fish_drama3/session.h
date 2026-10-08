#pragma once

#include "engine/community_models/fish_drama3/generator.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/framework/runtime/streaming_tts_session.h"

namespace engine::models::fish_drama3 {

class Drama3Session final : public runtime::RuntimeSessionBase,
                            public runtime::IOfflineVoiceTaskSession,
                            public runtime::StreamingTtsSessionBase {
public:
    Drama3Session(runtime::TaskSpec task, const runtime::SessionOptions & options,
                  std::shared_ptr<const Drama3Assets> assets);
    ~Drama3Session() override;
    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;

private:
    runtime::TaskResult generate_stream(const runtime::TaskRequest & request) override;
    void reset_stream_state() override {}
    runtime::TaskResult synthesize(const runtime::TaskRequest & request, bool streaming);
    Drama3Take conditioning_take(const engine::io::json::Value & value);
    runtime::TaskSpec task_;
    std::shared_ptr<const Drama3Assets> assets_;
    std::unique_ptr<Drama3Generator> generator_;
    runtime::TaskRequest defaults_;
};

}  // namespace engine::models::fish_drama3
