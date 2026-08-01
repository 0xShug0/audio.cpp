#pragma once

#include "engine/framework/runtime/cache_slots.h"
#include "engine/framework/model_spec/metadata.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/models/rvc/assets.h"
#include "engine/models/rvc/native_pipeline.h"

#include <memory>

namespace engine::models::rvc {

std::shared_ptr<runtime::IVoiceModelLoader> make_rvc_loader();

class RvcSession final
    : public runtime::RuntimeSessionBase
    , public runtime::IOfflineVoiceTaskSession {
public:
    RvcSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const RvcAssets> assets,
        std::shared_ptr<const engine::model_spec::ModelContract> contract);

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;

private:
    runtime::TaskSpec task_;
    std::shared_ptr<const RvcAssets> assets_;
    std::shared_ptr<const engine::model_spec::ModelContract> contract_;
    engine::assets::TensorStorageType weight_storage_type_;
    RvcNativePipeline pipeline_;
    runtime::CacheSlots<std::string, RvcVoiceModel> user_voice_cache_;
};

}  // namespace engine::models::rvc
