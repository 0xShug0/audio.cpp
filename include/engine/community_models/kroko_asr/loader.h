#pragma once

#include "engine/community_models/kroko_asr/assets.h"
#include "engine/framework/runtime/model.h"

#include <memory>

namespace engine::models::kroko_asr {

class KrokoASRLoadedModel final
    : public runtime::ILoadedVoiceModel {
public:
    KrokoASRLoadedModel(
        runtime::ModelMetadata metadata,
        runtime::CapabilitySet capabilities,
        std::shared_ptr<const KrokoASRAssets> assets);

    const runtime::ModelMetadata & metadata() const noexcept override;
    const runtime::CapabilitySet & capabilities() const noexcept override;
    std::unique_ptr<runtime::IVoiceTaskSession>
    create_task_session(
        const runtime::TaskSpec & task,
        const runtime::SessionOptions & options) const override;

private:
    runtime::ModelMetadata metadata_;
    runtime::CapabilitySet capabilities_;
    std::shared_ptr<const KrokoASRAssets> assets_;
};

std::unique_ptr<KrokoASRLoadedModel> load_kroko_asr_model(
    const std::filesystem::path & model_path);
std::shared_ptr<runtime::IVoiceModelLoader>
make_kroko_asr_loader();

}  // namespace engine::models::kroko_asr
