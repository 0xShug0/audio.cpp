#pragma once

#include "engine/models/index_echo/assets.h"

#include "engine/framework/core/execution_context.h"
#include "engine/framework/runtime/session.h"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

namespace engine::models::index_echo {

class IndexEchoQwen3OmniAuTQwen35TranslationRuntime {
public:
    IndexEchoQwen3OmniAuTQwen35TranslationRuntime(
        std::shared_ptr<const IndexEchoAssets> assets,
        core::ExecutionContext & execution,
        assets::TensorStorageType weight_storage_type);
    ~IndexEchoQwen3OmniAuTQwen35TranslationRuntime();

    IndexEchoQwen3OmniAuTQwen35TranslationRuntime(const IndexEchoQwen3OmniAuTQwen35TranslationRuntime &) = delete;
    IndexEchoQwen3OmniAuTQwen35TranslationRuntime & operator=(const IndexEchoQwen3OmniAuTQwen35TranslationRuntime &) = delete;

    std::string translate_window(
        const runtime::AudioBuffer & audio,
        std::string_view target_language,
        std::string_view context,
        std::string_view glossary,
        int64_t max_new_tokens,
        float temperature,
        uint32_t seed);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::index_echo
