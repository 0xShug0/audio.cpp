#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/io/json.h"
#include "engine/framework/io/yaml.h"

#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace engine::assets {

struct ResourceFile {
    std::string id;
    std::filesystem::path path;
};

struct ResourceSpec {
    std::string id;
    std::filesystem::path relative_path;
    bool required = true;
};

struct TensorResource {
    std::filesystem::path path;
    std::string prefix;
};

[[nodiscard]] inline std::filesystem::path checkpoint_sidecar_config_path(const std::filesystem::path & checkpoint_path) {
    return checkpoint_path.parent_path() / (checkpoint_path.stem().string() + "_config.json");
}

class ResourceBundle {
public:
    explicit ResourceBundle(std::filesystem::path model_root = {});

    void add_file(std::string id, const std::filesystem::path & path);
    void add_tensor_source(std::string id, const std::filesystem::path & path, std::string tensor_prefix = {});
    void add_model_file(std::string id, const std::filesystem::path & relative_path);
    bool add_optional_model_file(std::string id, const std::filesystem::path & relative_path);
    void add_model_files(std::initializer_list<ResourceSpec> specs);

    [[nodiscard]] const std::filesystem::path & model_root() const noexcept;
    [[nodiscard]] bool has_file(std::string_view id) const noexcept;
    [[nodiscard]] const std::filesystem::path * find_file(std::string_view id) const noexcept;
    [[nodiscard]] const std::filesystem::path & require_file(std::string_view id) const;
    [[nodiscard]] std::vector<ResourceFile> files() const;

    [[nodiscard]] std::string read_text(std::string_view id) const;
    [[nodiscard]] engine::io::json::Value parse_json(std::string_view id) const;
    [[nodiscard]] engine::io::json::Value parse_jsonc(std::string_view id) const;
    [[nodiscard]] engine::io::yaml::FlattenedDocument parse_flattened_yaml(std::string_view id) const;
    [[nodiscard]] std::shared_ptr<const TensorSource> open_tensor_source(std::string_view id) const;

private:
    std::filesystem::path model_root_;
    std::unordered_map<std::string, std::filesystem::path> files_;
    std::unordered_map<std::string, TensorResource> tensor_resources_;
    mutable std::unordered_map<std::string, std::shared_ptr<const TensorSource>> tensor_sources_;
};

}  // namespace engine::assets
