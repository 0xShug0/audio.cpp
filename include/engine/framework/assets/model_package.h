#pragma once

#include "engine/framework/assets/resource_bundle.h"

#include <filesystem>
#include <optional>
#include <string_view>

namespace engine::assets {

enum class ModelPackageResourceKind {
    Files,
    Tensors,
};

class ScopedModelPackageSpecOverride {
public:
    explicit ScopedModelPackageSpecOverride(const std::optional<std::filesystem::path> & path,
                                            const std::filesystem::path & model_path = {});
    ~ScopedModelPackageSpecOverride();

    ScopedModelPackageSpecOverride(const ScopedModelPackageSpecOverride &) = delete;
    ScopedModelPackageSpecOverride & operator=(const ScopedModelPackageSpecOverride &) = delete;

private:
    std::optional<std::filesystem::path> previous_;
    std::optional<std::filesystem::path> previous_model_path_;
};

[[nodiscard]] std::filesystem::path default_model_package_spec_path(std::string_view family);

[[nodiscard]] ResourceBundle load_resource_bundle_from_package_spec(const std::filesystem::path & model_path,
    const std::filesystem::path & spec_path);

[[nodiscard]] std::vector<ResourceFile> discover_resources_from_package_spec(const std::filesystem::path & model_path,
    const std::filesystem::path & spec_path,
    ModelPackageResourceKind kind);

}  // namespace engine::assets
