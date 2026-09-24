#pragma once

#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/models/apollo/assets.h"
#include "engine/models/apollo/runtime.h"

namespace engine::model_spec {
struct ModelContract;
}

namespace engine::models::apollo {

class ApolloSession final : public runtime::RuntimeSessionBase, public runtime::IOfflineVoiceTaskSession {
public:
    ApolloSession(runtime::TaskSpec task, runtime::SessionOptions options,
                  std::shared_ptr<const ApolloAssets> assets,
                  std::shared_ptr<const model_spec::ModelContract> contract);
    ~ApolloSession() override;

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;

private:
    runtime::TaskSpec task_;
    std::shared_ptr<const ApolloAssets> assets_;
    std::shared_ptr<const model_spec::ModelContract> contract_;
    std::unique_ptr<ApolloRuntime> runtime_;
};

std::shared_ptr<runtime::IVoiceModelLoader> make_apollo_loader();

}  // namespace engine::models::apollo
