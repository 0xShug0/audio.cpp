#pragma once

#include "engine/framework/runtime/model.h"
#include "engine/models/fun_asr_nano/assets.h"

#include <memory>

namespace engine::models::fun_asr_nano {

class FunAsrNanoLoadedModel final : public runtime::ILoadedVoiceModel {
public:
  FunAsrNanoLoadedModel(runtime::ModelMetadata metadata,
                        runtime::CapabilitySet capabilities,
                        std::shared_ptr<const FunAsrNanoAssets> assets);

  const runtime::ModelMetadata &metadata() const noexcept override;
  const runtime::CapabilitySet &capabilities() const noexcept override;
  std::unique_ptr<runtime::IVoiceTaskSession>
  create_task_session(const runtime::TaskSpec &task,
                      const runtime::SessionOptions &options) const override;

private:
  runtime::ModelMetadata metadata_;
  runtime::CapabilitySet capabilities_;
  std::shared_ptr<const FunAsrNanoAssets> assets_;
};

std::unique_ptr<FunAsrNanoLoadedModel>
load_fun_asr_nano_model(const std::filesystem::path &model_path);
std::shared_ptr<runtime::IVoiceModelLoader> make_fun_asr_nano_loader();

} // namespace engine::models::fun_asr_nano
