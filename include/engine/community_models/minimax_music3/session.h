#pragma once

#include "engine/community_models/minimax_music3/assets.h"
#include "engine/community_models/minimax_music3/pipeline.h"
#include "engine/framework/runtime/session_base.h"

#include <memory>
#include <string>

namespace engine::models::minimax_music3 {

class MiniMaxMusic3Session final
    : public runtime::RuntimeSessionBase,
      public runtime::IOfflineVoiceTaskSession {
public:
    MiniMaxMusic3Session(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const MiniMaxMusic3Assets> assets);
    ~MiniMaxMusic3Session() override;

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;

private:
    MiniMaxMusic3Request parse_request(const runtime::TaskRequest & request) const;

    runtime::TaskSpec task_;
    std::shared_ptr<const MiniMaxMusic3Assets> assets_;
    std::unique_ptr<MiniMaxMusic3PipelineRuntime> pipeline_;
};

std::shared_ptr<runtime::IVoiceModelLoader> make_minimax_music3_loader();

}  // namespace engine::models::minimax_music3
