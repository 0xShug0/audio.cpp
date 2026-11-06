#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::mulacover {

// One instance per session; retains weights and the framework's cached graphs.
// Sources use the Qwen3 HF tensor names (also preserved by framework GGUF conversion).
class StyleEncoder {
public:
    StyleEncoder(
        std::shared_ptr<const assets::TensorSource> source,
        const std::filesystem::path & tokenizer_directory,
        const core::BackendConfig & backend,
        assets::TensorStorageType storage_type = assets::TensorStorageType::Native);
    ~StyleEncoder();

    StyleEncoder(const StyleEncoder &) = delete;
    StyleEncoder & operator=(const StyleEncoder &) = delete;

    // Qwen3-Embedding-0.6B: 1024 F32 values, L2 normalized for nonempty tags.
    std::vector<float> encode(const std::string & tags);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::mulacover
