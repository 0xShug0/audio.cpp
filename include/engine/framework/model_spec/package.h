#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/io/json.h"

#include <filesystem>
#include <optional>
#include <string_view>
#include <vector>

namespace engine::model_spec {

enum class ResourceKind {
    Files,
    Tensors,
};

class ScopedSpecOverride {
public:
    explicit ScopedSpecOverride(const std::optional<std::filesystem::path> & path,
        const std::filesystem::path & model_path = {});
    ~ScopedSpecOverride();

    ScopedSpecOverride(const ScopedSpecOverride &) = delete;
    ScopedSpecOverride & operator=(const ScopedSpecOverride &) = delete;

private:
    std::optional<std::filesystem::path> previous_;
    std::optional<std::filesystem::path> previous_model_path_;
};

[[nodiscard]] std::filesystem::path default_spec_path(std::string_view family);

[[nodiscard]] engine::io::json::Value load_spec(const std::filesystem::path & spec_path);

[[nodiscard]] assets::ResourceBundle load_resource_bundle(const std::filesystem::path & model_path,
    const std::filesystem::path & spec_path);

[[nodiscard]] std::vector<assets::ResourceFile> discover_resources(const std::filesystem::path & model_path,
    const std::filesystem::path & spec_path,
    ResourceKind kind);

}  // namespace engine::model_spec
