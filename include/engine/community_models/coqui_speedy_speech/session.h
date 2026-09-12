#pragma once

#include "engine/community_models/coqui_speedy_speech/assets.h"
#include "engine/framework/model_spec/metadata.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/session_base.h"

#include <memory>

namespace engine::community_models::coqui_speedy_speech {

std::shared_ptr<engine::runtime::IVoiceModelLoader>
make_coqui_speedy_speech_loader();

class NativeRuntime;
class Session final : public engine::runtime::RuntimeSessionBase,
                      public engine::runtime::IOfflineVoiceTaskSession {
public:
  Session(engine::runtime::TaskSpec task,
          engine::runtime::SessionOptions options,
          std::shared_ptr<const Assets> assets,
          std::shared_ptr<const engine::model_spec::ModelContract> contract);
  ~Session() override;
  std::string family() const override;
  engine::runtime::VoiceTaskKind task_kind() const override;
  engine::runtime::RunMode run_mode() const override;
  void
  prepare(const engine::runtime::SessionPreparationRequest &request) override;
  engine::runtime::TaskResult
  run(const engine::runtime::TaskRequest &request) override;

private:
  engine::runtime::TaskSpec task_;
  std::shared_ptr<const Assets> assets_;
  std::shared_ptr<const engine::model_spec::ModelContract> contract_;
  std::unique_ptr<NativeRuntime> runtime_;
};

} // namespace engine::community_models::coqui_speedy_speech
