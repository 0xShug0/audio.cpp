#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/models/supertonic/assets.h"
#include "engine/models/supertonic/tokenizer_text.h"

#include <memory>
#include <string>

namespace engine::models::supertonic {

class SupertonicNativeRuntime;

struct SupertonicGenerationOptions {
    int num_inference_steps = 8;
    float speaking_rate = 1.05F;
    uint32_t seed = 1234U;
    std::string voice = "M1";
    std::string language = "en";
};

class SupertonicSession final
    : public runtime::RuntimeSessionBase
    , public runtime::IOfflineVoiceTaskSession {
public:
    SupertonicSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const SupertonicAssets> assets);
    ~SupertonicSession() override;

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;

private:
    SupertonicGenerationOptions generation_options_from_request(const runtime::TaskRequest & request) const;
    void validate_request(const runtime::TaskRequest & request) const;

    runtime::TaskSpec task_;
    std::shared_ptr<const SupertonicAssets> assets_;
    SupertonicTextTokenizer tokenizer_;
    assets::TensorStorageType weight_storage_type_ = assets::TensorStorageType::Native;
    std::unique_ptr<SupertonicNativeRuntime> runtime_;
};

}  // namespace engine::models::supertonic
