#pragma once

#include "engine/framework/core/module.h"
#include "engine/framework/modules/attention/types.h"

namespace engine::modules {

class CrossAttentionModule {
public:
    explicit CrossAttentionModule(AttentionConfig config);

    const AttentionConfig & config() const noexcept;
    const core::ModuleSchema & schema() const noexcept;

    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & query,
        const core::TensorValue & memory,
        const AttentionWeights & weights) const;

    static const core::ModuleSchema & static_schema() noexcept;

private:
    AttentionConfig config_;
};

}  // namespace engine::modules
