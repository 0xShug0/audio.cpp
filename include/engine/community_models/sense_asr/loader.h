#pragma once

#include "engine/framework/runtime/model.h"
#include "engine/community_models/sense_asr/assets.h"

#include <memory>

namespace engine::community_models::sense_asr {

class SenseAsrLoadedModel final : public runtime::ILoadedVoiceModel {
public:
  SenseAsrLoadedModel(runtime::ModelMetadata metadata,
                      runtime::CapabilitySet capabilities,
                      std::shared_ptr<const SenseAsrAssets> assets);

  const runtime::ModelMetadata &metadata() const noexcept override;
  const runtime::CapabilitySet &capabilities() const noexcept override;
  std::unique_ptr<runtime::IVoiceTaskSession>
  create_task_session(const runtime::TaskSpec &task,
                      const runtime::SessionOptions &options) const override;

private:
  runtime::ModelMetadata metadata_;
  runtime::CapabilitySet capabilities_;
  std::shared_ptr<const SenseAsrAssets> assets_;
};

std::unique_ptr<SenseAsrLoadedModel>
load_sense_asr_model(const std::filesystem::path &model_path);
std::shared_ptr<runtime::IVoiceModelLoader> make_sense_asr_loader();

} // namespace engine::community_models::sense_asr
