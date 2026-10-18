#pragma once

#include "engine/models/index_echo/assets.h"

#include "engine/framework/core/execution_context.h"

#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

namespace engine::models::index_echo {

class IndexEchoHidden2CVRuntime {
public:
    IndexEchoHidden2CVRuntime(
        std::shared_ptr<const IndexEchoAssets> assets,
        core::ExecutionContext & execution,
        assets::TensorStorageType storage_type);
    ~IndexEchoHidden2CVRuntime();

    IndexEchoHidden2CVRuntime(const IndexEchoHidden2CVRuntime &) = delete;
    IndexEchoHidden2CVRuntime & operator=(const IndexEchoHidden2CVRuntime &) = delete;

    std::vector<float> map(
        const std::vector<float> & text_hidden,
        const std::vector<float> & cosyvoice_embeddings,
        int64_t tokens,
        std::string_view target_language);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::index_echo
