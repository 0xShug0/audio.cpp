#pragma once

#include "engine/models/midashenglm_gen/ar_runtime.h"
#include "engine/models/midashenglm_gen/audio_tokenizer.h"
#include "engine/models/midashenglm_gen/tokenizer_text.h"
#include "engine/framework/model_spec/metadata.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/session_base.h"

#include <memory>

namespace engine::models::midashenglm_gen {

std::shared_ptr<engine::runtime::IVoiceModelLoader> make_midashenglm_gen_loader();

class MiDashengLmGenSession final
    : public engine::runtime::RuntimeSessionBase
    , public engine::runtime::IOfflineVoiceTaskSession {
public:
    MiDashengLmGenSession(
        engine::runtime::TaskSpec task,
        engine::runtime::SessionOptions options,
        std::shared_ptr<const MiDashengLmGenAssets> assets,
        std::shared_ptr<const engine::model_spec::ModelContract> contract);
    ~MiDashengLmGenSession() override;

    std::string family() const override;
    engine::runtime::VoiceTaskKind task_kind() const override;
    engine::runtime::RunMode run_mode() const override;
    void prepare(const engine::runtime::SessionPreparationRequest & request) override;
    engine::runtime::TaskResult run(const engine::runtime::TaskRequest & request) override;

private:
    MiDashengLmGenGenerationOptions generation_options(
        const engine::runtime::TaskRequest & request) const;

    engine::runtime::TaskSpec task_;
    engine::runtime::SessionOptions options_;
    std::shared_ptr<const MiDashengLmGenAssets> assets_;
    std::shared_ptr<const engine::model_spec::ModelContract> contract_;
    std::unique_ptr<engine::core::ExecutionContext> execution_;
    std::unique_ptr<MiDashengLmGenTextTokenizer> tokenizer_;
    std::unique_ptr<MiDashengLmGenPromptEncoderRuntime> prompt_encoder_;
    std::unique_ptr<MiDashengLmGenFlowRuntime> flow_;
    std::unique_ptr<MiDashengLmGenARRuntime> ar_;
    std::unique_ptr<MiDashengLmGenAudioTokenizerRuntime> audio_tokenizer_;
};

}  // namespace engine::models::midashenglm_gen
