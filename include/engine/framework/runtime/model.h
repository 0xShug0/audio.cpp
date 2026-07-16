#pragma once

#include "engine/framework/runtime/session.h"

#include <filesystem>
#include <optional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace engine::assets {
enum class ModelPackageResourceKind;
}

namespace engine::runtime {

struct NamedAsset {
    std::string id;
    std::filesystem::path path;
};

struct TaskCapability {
    VoiceTaskKind task = VoiceTaskKind::Vad;
    std::vector<RunMode> modes;
};

struct CapabilitySet {
    std::vector<TaskCapability> supported_tasks;
    std::vector<std::string> languages;
    bool supports_speaker_reference = false;
    bool supports_style_condition = false;
    bool supports_timestamps = false;
};

struct ModelMetadata {
    std::string family;
    std::string variant;
    std::string description;
    std::vector<std::string> config_candidates;
    std::vector<std::string> weight_candidates;
};

struct CliOptionInfo {
    std::string name;
    std::string value_name;
    std::string description;
};

struct ModelCliInterface {
    std::vector<CliOptionInfo> request_options;
    std::vector<CliOptionInfo> session_options;
    std::vector<CliOptionInfo> load_options;
};

struct ModelInspection {
    ModelMetadata metadata;
    CapabilitySet capabilities;
    ModelCliInterface cli;
    std::filesystem::path model_root;
    std::vector<NamedAsset> discovered_configs;
    std::vector<NamedAsset> discovered_weights;
};

struct ModelLoadRequest {
    std::filesystem::path model_path;
    std::optional<std::filesystem::path> model_spec_override = std::nullopt;
    std::optional<std::string> family_hint = std::nullopt;
    std::optional<std::string> config_id = std::nullopt;
    std::optional<std::string> weight_id = std::nullopt;
    std::unordered_map<std::string, std::string> options;
};

std::vector<NamedAsset> discover_named_assets(
    const std::filesystem::path & root,
    const std::vector<std::string> & relative_candidates);

std::vector<NamedAsset> discover_named_assets_from_package_spec(
    const std::filesystem::path & model_path,
    const std::filesystem::path & spec_path,
    engine::assets::ModelPackageResourceKind kind);

const NamedAsset * find_named_asset(
    const std::vector<NamedAsset> & assets,
    const std::string & id) noexcept;

const NamedAsset & require_named_asset(
    const std::vector<NamedAsset> & assets,
    const std::string & id,
    const std::string & role);

const NamedAsset * select_named_asset(
    const std::vector<NamedAsset> & assets,
    const std::optional<std::string> & id,
    const std::string & role) noexcept(false);

class ILoadedVoiceModel {
public:
    virtual ~ILoadedVoiceModel() = default;

    virtual const ModelMetadata & metadata() const noexcept = 0;
    virtual const CapabilitySet & capabilities() const noexcept = 0;
    virtual std::unique_ptr<IVoiceTaskSession> create_task_session(
        const TaskSpec & task,
        const SessionOptions & options) const = 0;
};

class IVoiceModelLoader {
public:
    virtual ~IVoiceModelLoader() = default;

    virtual std::string family() const = 0;
    virtual bool can_load(const ModelLoadRequest & request) const = 0;
    virtual ModelInspection inspect(const ModelLoadRequest & request) const = 0;
    virtual std::unique_ptr<ILoadedVoiceModel> load(const ModelLoadRequest & request) const = 0;
};

}  // namespace engine::runtime
