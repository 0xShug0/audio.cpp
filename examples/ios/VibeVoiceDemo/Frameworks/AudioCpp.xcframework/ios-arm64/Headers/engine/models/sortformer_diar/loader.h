#pragma once

#include "engine/framework/runtime/model.h"
#include "engine/models/sortformer_diar/assets.h"

#include <filesystem>
#include <memory>

namespace engine::models::sortformer_diar {

class SortformerDiarSession;

class SortformerDiarLoadedModel final : public runtime::ILoadedVoiceModel {
public:
    SortformerDiarLoadedModel(
        runtime::ModelMetadata metadata,
        runtime::CapabilitySet capabilities,
        std::shared_ptr<const SortformerAssets> assets);

    const runtime::ModelMetadata & metadata() const noexcept override;
    const runtime::CapabilitySet & capabilities() const noexcept override;
    std::unique_ptr<runtime::IVoiceTaskSession> create_task_session(
        const runtime::TaskSpec & task,
        const runtime::SessionOptions & options) const override;

private:
    runtime::ModelMetadata metadata_;
    runtime::CapabilitySet capabilities_;
    std::shared_ptr<const SortformerAssets> assets_;
};

std::unique_ptr<SortformerDiarLoadedModel> load_sortformer_diar_model(const std::filesystem::path & model_path);
std::shared_ptr<runtime::IVoiceModelLoader> make_sortformer_diar_loader();

}  // namespace engine::models::sortformer_diar
