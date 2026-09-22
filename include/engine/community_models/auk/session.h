#pragma once

#include "engine/framework/runtime/session_base.h"
#include "engine/framework/model_spec/metadata.h"
#include "engine/community_models/auk/conditioning.h"
#include "engine/community_models/auk/flow.h"
#include "engine/community_models/auk/vae.h"

namespace engine::models::auk {

struct AukAssets {
    std::filesystem::path model_root;
    bool flash = false;
    std::shared_ptr<const assets::TensorSource> model;
    std::shared_ptr<const assets::TensorSource> qwen;
    std::shared_ptr<const assets::TensorSource> vae;
    std::shared_ptr<tokenizers::LlamaBpeTokenizer> tokenizer;
};

class AukSession final : public runtime::RuntimeSessionBase, public runtime::IOfflineVoiceTaskSession {
public:
    AukSession(runtime::TaskSpec task, runtime::SessionOptions options, std::shared_ptr<const AukAssets> assets,
               std::shared_ptr<const model_spec::ModelContract> contract);
    ~AukSession() override;
    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;

private:
    runtime::TaskSpec task_;
    std::shared_ptr<const AukAssets> assets_;
    std::shared_ptr<const model_spec::ModelContract> contract_;
    bool mem_saver_ = false;
    std::unique_ptr<ConditioningRuntime> conditioning_;
    std::unique_ptr<FlowRuntime> flow_;
    std::unique_ptr<VaeDecoderRuntime> decoder_;
    std::unique_ptr<AudioConditioningRuntime> audio_conditioning_;
    std::unique_ptr<VaeEncoderRuntime> reference_encoder_;
};

std::shared_ptr<runtime::IVoiceModelLoader> make_auk_loader();

}  // namespace engine::models::auk
