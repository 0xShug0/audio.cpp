#pragma once

#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/framework/model_spec/metadata.h"
#include "engine/models/fireredtts3/assets.h"
#include "engine/models/fireredtts3/tokenizer_text.h"

#include <memory>

namespace engine::models::fireredtts3 {

class FireRedTTS3BaseRuntime;
class FireRedTTS3InstructRuntime;

std::shared_ptr<engine::runtime::IVoiceModelLoader> make_fireredtts3_loader();

class FireRedTTS3Session final
    : public engine::runtime::RuntimeSessionBase
    , public engine::runtime::IOfflineVoiceTaskSession {
public:
    FireRedTTS3Session(
        engine::runtime::TaskSpec task,
        engine::runtime::SessionOptions options,
        std::shared_ptr<const FireRedTTS3Assets> assets,
        std::shared_ptr<const engine::model_spec::ModelContract> contract);
    ~FireRedTTS3Session() override;

    std::string family() const override;
    engine::runtime::VoiceTaskKind task_kind() const override;
    engine::runtime::RunMode run_mode() const override;
    void prepare(const engine::runtime::SessionPreparationRequest & request) override;
    engine::runtime::TaskResult run(const engine::runtime::TaskRequest & request) override;

private:
    engine::runtime::TaskSpec task_;
    std::shared_ptr<const FireRedTTS3Assets> assets_;
    std::shared_ptr<const engine::model_spec::ModelContract> contract_;
    std::unique_ptr<FireRedTTS3TextTokenizer> tokenizer_;
    std::unique_ptr<FireRedTTS3BaseRuntime> runtime_;
    std::unique_ptr<FireRedTTS3InstructRuntime> instruct_runtime_;
    bool mem_saver_ = false;
};

}  // namespace engine::models::fireredtts3
