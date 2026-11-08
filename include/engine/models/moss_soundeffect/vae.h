#pragma once

#include "engine/models/moss_soundeffect/assets.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/runtime/session.h"

#include <cstddef>

namespace engine::models::moss_soundeffect {

class VaeRuntime {
public:
    VaeRuntime(core::ExecutionContext & execution, std::shared_ptr<const Assets> assets);
    ~VaeRuntime();
    runtime::AudioBuffer decode(const std::vector<float> & latents);
    runtime::AudioBuffer decode(const std::vector<float> & latents, size_t output_samples);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::moss_soundeffect
