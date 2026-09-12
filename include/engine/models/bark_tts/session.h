#pragma once

#include "engine/framework/model_spec/metadata.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/models/bark_tts/assets.h"

#include <memory>

namespace engine::core { class ExecutionContext; }
namespace engine::models::bark_tts {
class BarkGenerator;
std::shared_ptr<engine::runtime::IVoiceModelLoader> make_bark_tts_loader();

class BarkSession final : public engine::runtime::RuntimeSessionBase,
                          public engine::runtime::IOfflineVoiceTaskSession {
public:
    BarkSession(engine::runtime::TaskSpec task, engine::runtime::SessionOptions options,
                std::shared_ptr<const BarkAssets> assets,
                std::shared_ptr<const engine::model_spec::ModelContract> contract);
    ~BarkSession() override;
    std::string family() const override;
    engine::runtime::VoiceTaskKind task_kind() const override;
    engine::runtime::RunMode run_mode() const override;
    void prepare(const engine::runtime::SessionPreparationRequest & request) override;
    engine::runtime::TaskResult run(const engine::runtime::TaskRequest & request) override;
private:
    engine::runtime::TaskSpec task_;
    engine::runtime::SessionOptions options_;
    std::shared_ptr<const BarkAssets> assets_;
    std::shared_ptr<const engine::model_spec::ModelContract> contract_;
    std::unique_ptr<engine::core::ExecutionContext> execution_;
    std::unique_ptr<BarkGenerator> generator_;
};
}  // namespace engine::models::bark_tts
