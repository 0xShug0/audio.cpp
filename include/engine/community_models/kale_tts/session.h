#pragma once

#include "engine/community_models/kale_tts/assets.h"
#include "engine/community_models/kale_tts/frontend.h"
#include "engine/community_models/kale_tts/runtime.h"
#include "engine/framework/model_spec/metadata.h"
#include "engine/framework/runtime/session_base.h"

#include <memory>
#include <string>
#include <vector>

namespace engine::models::kale_tts {

std::shared_ptr<runtime::IVoiceModelLoader> make_kale_tts_loader();

class KaleTtsSession final
    : public runtime::RuntimeSessionBase
    , public runtime::IOfflineVoiceTaskSession {
public:
    KaleTtsSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const KaleAssets> assets,
        std::shared_ptr<const engine::model_spec::ModelContract> contract);
    ~KaleTtsSession() override;

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;

private:
    runtime::TaskSpec task_;
    std::shared_ptr<const KaleAssets> assets_;
    std::shared_ptr<const engine::model_spec::ModelContract> contract_;
    std::unique_ptr<KaleFrontend> frontend_;
    std::unique_ptr<KaleRuntime> runtime_;
    std::vector<float> parity_noise_;   // kale_tts.parity_noise_file: the same noise as a reference run (tests only)
};

}  // namespace engine::models::kale_tts
