#pragma once

#include "engine/framework/runtime/model.h"

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace engine::runtime {

struct RegistryConfig {
    std::vector<std::string> enabled_families;
};

class ModelRegistry {
public:
    void register_loader(std::shared_ptr<IVoiceModelLoader> loader);

    bool empty() const noexcept;
    size_t size() const noexcept;
    std::vector<std::string> families() const;
    bool supports_family(const std::string & family) const noexcept;
    /** Path-free loader catalog for ``--list-loaders --json``. */
    std::vector<LoaderAdvertisement> advertise_loaders() const;
    ModelInspection inspect(const ModelLoadRequest & request) const;
    ModelInspection inspect(const std::filesystem::path & model_path) const;
    std::unique_ptr<ILoadedVoiceModel> load(const ModelLoadRequest & request) const;
    std::unique_ptr<ILoadedVoiceModel> load(const std::filesystem::path & model_path) const;

private:
    void validate_request(const ModelLoadRequest & request) const;
    const IVoiceModelLoader * find_loader(const ModelLoadRequest & request) const;

    std::vector<std::shared_ptr<IVoiceModelLoader>> loaders_;
};

RegistryConfig load_registry_config(const std::filesystem::path & path);
ModelRegistry make_registry_from_config(
    const RegistryConfig & config,
    const std::vector<std::shared_ptr<IVoiceModelLoader>> & available_loaders);
ModelRegistry make_default_registry(const std::optional<std::filesystem::path> & config_path = std::nullopt);

}  // namespace engine::runtime
