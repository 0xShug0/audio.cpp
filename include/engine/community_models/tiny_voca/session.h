#pragma once

#include "engine/community_models/tiny_voca/assets.h"
#include "engine/community_models/tiny_voca/frontend.h"
#include "engine/community_models/tiny_voca/runtime.h"
#include "engine/framework/model_spec/metadata.h"
#include "engine/framework/runtime/session_base.h"

#include <memory>
#include <string>
#include <vector>

namespace engine::models::tiny_voca {

std::shared_ptr<runtime::IVoiceModelLoader> make_tiny_voca_loader();

class TinyVocaSession final
    : public runtime::RuntimeSessionBase
    , public runtime::IOfflineVoiceTaskSession {
public:
    TinyVocaSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const TinyVocaAssets> assets,
        std::shared_ptr<const engine::model_spec::ModelContract> contract);
    ~TinyVocaSession() override;

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;

private:
    TinyVocaGenerationOptions generation_options(const runtime::TaskRequest & request) const;
    // Speaks one piece of text, cutting it smaller when it is too long for one pass.
    void speak(const std::string & text, const TinyVocaGenerationOptions & options, std::vector<std::vector<float>> & waves, int64_t & spoken);

    runtime::TaskSpec task_;
    std::shared_ptr<const TinyVocaAssets> assets_;
    std::shared_ptr<const engine::model_spec::ModelContract> contract_;
    std::unique_ptr<TinyVocaFrontend> frontend_;
    std::unique_ptr<TinyVocaRuntime> runtime_;
};

}  // namespace engine::models::tiny_voca
