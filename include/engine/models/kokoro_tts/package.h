#pragma once

#include "engine/framework/assets/tensor_source.h"
#include <filesystem>
#include <memory>

namespace engine::models::kokoro_tts {
// Owns materialized resources for the lifetime of a standalone model.
struct KokoroPackage {
    std::filesystem::path root;
    std::shared_ptr<const assets::TensorSource> weights;
    bool temporary = false;
    ~KokoroPackage();
};
std::shared_ptr<KokoroPackage> open_kokoro_package(const std::filesystem::path & path);
bool is_kokoro_gguf(const std::filesystem::path & path) noexcept;
}
