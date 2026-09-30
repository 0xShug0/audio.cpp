#pragma once

#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/community_models/reuse/runtime.h"

namespace engine::model_spec { struct ModelContract; }

namespace engine::models::reuse {

class ReuseSession final
    : public runtime::RuntimeSessionBase
    , public runtime::IOfflineVoiceTaskSession
    , public runtime::IBatchedOfflineVoiceTaskSession {
public:
    ReuseSession(runtime::TaskSpec task, runtime::SessionOptions options,
                 std::shared_ptr<const ReuseAssets> assets,
                 std::shared_ptr<const model_spec::ModelContract> contract);
    ~ReuseSession() override;
    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;
    std::vector<runtime::TaskResult> run_batch(
        const std::vector<runtime::TaskRequest> & requests) override;
    void run_batch(
        const std::vector<runtime::TaskRequest> & requests,
        const runtime::IBatchedOfflineVoiceTaskSession::ResultCallback & on_result) override;

private:
    runtime::TaskSpec task_;
    std::shared_ptr<const model_spec::ModelContract> contract_;
    std::unique_ptr<ReuseRuntime> runtime_;
};

std::shared_ptr<runtime::IVoiceModelLoader> make_reuse_loader();

}  // namespace engine::models::reuse
