#pragma once

#include "engine/models/foundation_1/assets.h"
#include "engine/framework/runtime/model.h"

namespace engine::models::foundation_1 {

struct GenerationOptions {
    float duration_sec = 10;
    int64_t num_steps = 100;
    float guidance_scale = 7;
    uint64_t seed = 42;
    std::string negative_prompt;
};

class FoundationPipeline {
public:
    FoundationPipeline(std::shared_ptr<const FoundationAssets> assets,
        core::ExecutionContext & execution, assets::TensorStorageType storage);
    ~FoundationPipeline();
    runtime::AudioBuffer generate(const std::string & prompt, const GenerationOptions & options);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

std::shared_ptr<runtime::IVoiceModelLoader> make_foundation_1_loader();

} // namespace engine::models::foundation_1
