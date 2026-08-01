#pragma once

#include "engine/framework/core/module.h"
#include "engine/framework/modules/attention/types.h"

#include <optional>

namespace engine::modules {

class SelfAttentionModule {
public:
    explicit SelfAttentionModule(AttentionConfig config);

    const AttentionConfig & config() const noexcept;
    const core::ModuleSchema & schema() const noexcept;

    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const AttentionWeights & weights) const;

    static const core::ModuleSchema & static_schema() noexcept;

private:
    AttentionConfig config_;
};

class StreamingSelfAttentionModule {
public:
    explicit StreamingSelfAttentionModule(AttentionConfig config);

    const AttentionConfig & config() const noexcept;

    StreamingAttentionOutputs build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const core::TensorValue & positions,
        const AttentionWeights & weights,
        const std::optional<core::TensorValue> & prefix_key = std::nullopt,
        const std::optional<core::TensorValue> & prefix_value = std::nullopt,
        const std::optional<core::TensorValue> & attention_mask = std::nullopt) const;

private:
    AttentionConfig config_;
};

}  // namespace engine::modules
