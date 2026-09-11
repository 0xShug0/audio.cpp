#pragma once

#include "engine/framework/model_spec/metadata.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/community_models/minimax_h3/assets.h"
#include "engine/community_models/minimax_h3/pipeline.h"

#include <memory>
#include <string>

namespace engine::models::minimax_h3 {

class MiniMaxH3Session final
    : public runtime::RuntimeSessionBase
    , public runtime::IOfflineVoiceTaskSession {
public:
    MiniMaxH3Session(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const MiniMaxH3Assets> assets,
        std::shared_ptr<const engine::model_spec::ModelContract> contract);

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;

private:
    MiniMaxH3GenerateRequest make_request(const runtime::TaskRequest & request) const;

    runtime::TaskSpec task_;
    std::shared_ptr<const MiniMaxH3Assets> assets_;
    std::shared_ptr<const engine::model_spec::ModelContract> contract_;
    size_t weight_context_bytes_ = 256ull * 1024ull * 1024ull;
    std::unique_ptr<MiniMaxH3PipelineRuntime> runtime_;
};

std::shared_ptr<runtime::IVoiceModelLoader> make_minimax_h3_loader();

}  // namespace engine::models::minimax_h3
