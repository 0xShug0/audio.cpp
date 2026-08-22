#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/framework/runtime/session.h"
#include "engine/models/controlfoley/assets.h"
#include "engine/models/controlfoley/conditioning.h"

#include <filesystem>
#include <memory>
#include <optional>
#include <unordered_map>

namespace engine::models::controlfoley {

struct ControlFoleyOptions {
    float duration_sec = 8.0F;
    int64_t num_inference_steps = 25;
    float cfg_strength = 4.5F;
    uint32_t seed = 42;
    std::string negative_text;
    std::optional<std::filesystem::path> video;
    bool mask_away_clip = false;
};

ControlFoleyOptions parse_controlfoley_options(
    const std::unordered_map<std::string, std::string> & options);

class ControlFoleyPipelineRuntime {
public:
    ControlFoleyPipelineRuntime(
        std::shared_ptr<const ControlFoleyAssets> assets,
        engine::core::ExecutionContext & execution);
    ~ControlFoleyPipelineRuntime();

    engine::runtime::AudioBuffer run(const engine::runtime::TaskRequest & request);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::controlfoley
