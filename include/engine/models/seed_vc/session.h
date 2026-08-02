#pragma once

#include "engine/framework/model_spec/metadata.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/models/seed_vc/assets.h"

#include <memory>
#include <optional>
#include <string>

namespace engine::models::seed_vc {

std::shared_ptr<runtime::IVoiceModelLoader> make_seed_vc_loader();

struct SeedVcRouteRuntime;

class SeedVcSession final
    : public runtime::RuntimeSessionBase
    , public runtime::IOfflineVoiceTaskSession {
public:
    SeedVcSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const SeedVcAssets> assets,
        std::shared_ptr<const engine::model_spec::ModelContract> contract);

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;

private:
    runtime::TaskSpec task_;
    std::shared_ptr<const SeedVcAssets> assets_;
    std::shared_ptr<const engine::model_spec::ModelContract> contract_;
    std::shared_ptr<SeedVcRouteRuntime> route_runtime_;
    std::optional<engine::assets::TensorStorageType> weight_storage_type_;
};

}  // namespace engine::models::seed_vc
