#pragma once

#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/models/tf_gridnet/runtime.h"

namespace engine::model_spec { struct ModelContract; }

namespace engine::models::tf_gridnet {

class TFGridNetSession final : public runtime::RuntimeSessionBase, public runtime::IOfflineVoiceTaskSession {
public:
    TFGridNetSession(runtime::TaskSpec task, runtime::SessionOptions options,
        std::shared_ptr<const TFGridNetAssets> assets,
        std::shared_ptr<const model_spec::ModelContract> contract);
    ~TFGridNetSession() override;
    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;

private:
    runtime::TaskSpec task_;
    std::shared_ptr<const TFGridNetAssets> assets_;
    std::shared_ptr<const model_spec::ModelContract> contract_;
    std::unique_ptr<TFGridNetBiLSTMAttentionRuntime> runtime_;
};

std::shared_ptr<runtime::IVoiceModelLoader> make_tf_gridnet_loader();

}  // namespace engine::models::tf_gridnet
