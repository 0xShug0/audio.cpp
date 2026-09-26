#pragma once

#include "engine/models/samsone/assets.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/runtime/session.h"

#include <cstdint>
#include <memory>
#include <string>

namespace engine::models::samsone {

class SamsoneRuntime {
public:
    SamsoneRuntime(
        std::shared_ptr<const SamsoneAssets> assets,
        core::ExecutionContext & execution);
    ~SamsoneRuntime();

    SamsoneRuntime(const SamsoneRuntime &) = delete;
    SamsoneRuntime & operator=(const SamsoneRuntime &) = delete;

    std::string generate(
        const runtime::AudioBuffer & audio,
        const std::string & prompt,
        int64_t max_tokens,
        size_t threads);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::samsone
