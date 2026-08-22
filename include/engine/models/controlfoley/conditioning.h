#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/models/controlfoley/flow_denoiser.h"
#include "engine/framework/runtime/session.h"
#include "engine/models/controlfoley/assets.h"

#include <filesystem>
#include <memory>
#include <optional>
#include <string>

namespace engine::models::controlfoley {

struct ControlFoleyTemporalShape {
    int64_t latent = 0;
    int64_t clip = 0;
    int64_t visual = 0;
    int64_t sync = 0;
};

struct ControlFoleyConditioningRequest {
    std::optional<std::string> text;
    std::string negative_text;
    std::optional<engine::runtime::AudioBuffer> audio;
    std::optional<std::filesystem::path> video;
    bool mask_away_clip = false;
};

struct ControlFoleyConditioningBatch {
    ControlFoleyFlowConditionInput condition;
    ControlFoleyFlowConditionInput empty;
};

class ControlFoleyConditionerRuntime {
public:
    ControlFoleyConditionerRuntime(
        std::shared_ptr<const ControlFoleyAssets> assets,
        engine::core::ExecutionContext & execution);
    ~ControlFoleyConditionerRuntime();

    ControlFoleyConditioningBatch build(
        const ControlFoleyConditioningRequest & request,
        const ControlFoleyTemporalShape & shape) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::controlfoley
