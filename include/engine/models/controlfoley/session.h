#pragma once

#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/models/controlfoley/assets.h"

#include <memory>

namespace engine::models::controlfoley {

class ControlFoleyPipelineRuntime;

std::shared_ptr<engine::runtime::IVoiceModelLoader> make_controlfoley_loader();

class ControlFoleySession final
    : public engine::runtime::RuntimeSessionBase
    , public engine::runtime::IOfflineVoiceTaskSession {
public:
    ControlFoleySession(
        engine::runtime::TaskSpec task,
        engine::runtime::SessionOptions options,
        std::shared_ptr<const ControlFoleyAssets> assets);
    ~ControlFoleySession() override;

    std::string family() const override;
    engine::runtime::VoiceTaskKind task_kind() const override;
    engine::runtime::RunMode run_mode() const override;
    void prepare(const engine::runtime::SessionPreparationRequest & request) override;
    engine::runtime::TaskResult run(const engine::runtime::TaskRequest & request) override;

private:
    engine::runtime::TaskSpec task_;
    engine::runtime::SessionOptions options_;
    std::shared_ptr<const ControlFoleyAssets> assets_;
    std::unique_ptr<engine::core::ExecutionContext> execution_;
    std::unique_ptr<ControlFoleyPipelineRuntime> pipeline_;
};

}  // namespace engine::models::controlfoley
