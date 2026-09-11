#pragma once

#include "engine/framework/model_spec/metadata.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/models/magpie_tts/assets.h"
#include "engine/models/magpie_tts/runtime.h"
#include "engine/models/magpie_tts/tokenizer_text.h"
#include "engine/models/magpie_tts/types.h"

#include <memory>
#include <optional>
#include <string>

namespace engine::models::magpie_tts {

std::shared_ptr<runtime::IVoiceModelLoader> make_magpie_tts_loader();

class MagpieTTSSession final
    : public runtime::RuntimeSessionBase,
      public runtime::IOfflineVoiceTaskSession {
public:
    MagpieTTSSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const MagpieTTSAssets> assets,
        std::shared_ptr<const engine::model_spec::ModelContract> contract);
    ~MagpieTTSSession() override;

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;

private:
    runtime::TaskSpec task_;
    std::shared_ptr<const MagpieTTSAssets> assets_;
    std::shared_ptr<const engine::model_spec::ModelContract> contract_;
    MagpieTextTokenizer tokenizer_;
    std::unique_ptr<MagpieTTSRuntime> runtime_;
    std::optional<MagpieTTSRequest> prepared_defaults_;
    size_t graph_arena_bytes_ = 1024ull * 1024ull * 1024ull;
    size_t weight_context_bytes_ = 2048ull * 1024ull * 1024ull;
    engine::assets::TensorStorageType matmul_weight_storage_type_ = engine::assets::TensorStorageType::Native;
    engine::assets::TensorStorageType conv_weight_storage_type_ = engine::assets::TensorStorageType::Native;
};

}  // namespace engine::models::magpie_tts
