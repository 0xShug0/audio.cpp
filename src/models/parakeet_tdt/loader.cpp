#include "engine/models/parakeet_tdt/loader.h"
#include "engine/models/parakeet_tdt/session.h"

#include "engine/framework/io/filesystem.h"

#include <stdexcept>

namespace engine::models::parakeet_tdt {

namespace {

std::filesystem::path resolve_model_root(const std::filesystem::path & model_path) {
    if (engine::io::is_existing_directory(model_path)) {
        return std::filesystem::weakly_canonical(model_path);
    }
    throw std::runtime_error("Parakeet TDT expects a model directory: " + model_path.string());
}

std::vector<runtime::NamedAsset> discover_config_assets(const runtime::ModelLoadRequest & request) {
    return runtime::discover_named_assets(
        resolve_model_root(request.model_path),
        {"config.json", "processor_config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json"});
}

std::vector<runtime::NamedAsset> discover_weight_assets(const runtime::ModelLoadRequest & request) {
    return runtime::discover_named_assets(resolve_model_root(request.model_path), {"model.safetensors"});
}

class ParakeetTDTLoader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return "parakeet_tdt";
    }

    bool can_load(const runtime::ModelLoadRequest & request) const override {
        try {
            const auto root = resolve_model_root(request.model_path);
            return engine::io::is_existing_file(root / "config.json")
                && engine::io::is_existing_file(root / "processor_config.json")
                && engine::io::is_existing_file(root / "tokenizer.json")
                && engine::io::is_existing_file(root / "model.safetensors")
                && (!request.family_hint.has_value() || *request.family_hint == family());
        } catch (...) {
            return false;
        }
    }

    runtime::ModelInspection inspect(const runtime::ModelLoadRequest & request) const override {
        const auto root = resolve_model_root(request.model_path);
        runtime::ModelInspection inspection;
        inspection.model_root = root;
        inspection.metadata.family = family();
        inspection.metadata.variant = root.filename().string();
        inspection.metadata.description = "Parakeet TDT loaded from local HF-style assets.";
        inspection.metadata.config_candidates = {
            "config.json",
            "processor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "generation_config.json",
        };
        inspection.metadata.weight_candidates = {"model.safetensors"};
        inspection.capabilities.supported_tasks = {
            {runtime::VoiceTaskKind::Asr, {runtime::RunMode::Offline, runtime::RunMode::Streaming}},
        };
        inspection.capabilities.supports_timestamps = true;
        inspection.discovered_configs = discover_config_assets(request);
        inspection.discovered_weights = discover_weight_assets(request);
        return inspection;
    }

    std::unique_ptr<runtime::ILoadedVoiceModel> load(const runtime::ModelLoadRequest & request) const override {
        return load_parakeet_tdt_model(resolve_model_root(request.model_path));
    }
};

}  // namespace

ParakeetTDTLoadedModel::ParakeetTDTLoadedModel(
    runtime::ModelMetadata metadata,
    runtime::CapabilitySet capabilities,
    std::shared_ptr<const ParakeetAssets> assets)
    : metadata_(std::move(metadata)),
      capabilities_(std::move(capabilities)),
      assets_(std::move(assets)) {}

const runtime::ModelMetadata & ParakeetTDTLoadedModel::metadata() const noexcept {
    return metadata_;
}

const runtime::CapabilitySet & ParakeetTDTLoadedModel::capabilities() const noexcept {
    return capabilities_;
}

std::unique_ptr<runtime::IVoiceTaskSession> ParakeetTDTLoadedModel::create_task_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options) const {
    if (task.task != runtime::VoiceTaskKind::Asr) {
        throw std::runtime_error("Parakeet TDT only supports VoiceTaskKind::Asr");
    }
    return std::make_unique<ParakeetTDTSession>(task, options, assets_);
}

std::unique_ptr<ParakeetTDTLoadedModel> load_parakeet_tdt_model(const std::filesystem::path & model_path) {
    const auto root = resolve_model_root(model_path);
    auto assets = load_parakeet_assets(root);

    runtime::ModelMetadata metadata;
    metadata.family = "parakeet_tdt";
    metadata.variant = root.filename().string();
    metadata.description = "Parakeet TDT loaded from local HF-style assets.";
    metadata.config_candidates = {
        "config.json",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
    };
    metadata.weight_candidates = {"model.safetensors"};

    runtime::CapabilitySet capabilities;
    capabilities.supported_tasks = {
        {runtime::VoiceTaskKind::Asr, {runtime::RunMode::Offline, runtime::RunMode::Streaming}},
    };
    capabilities.supports_timestamps = true;

    return std::make_unique<ParakeetTDTLoadedModel>(std::move(metadata), std::move(capabilities), std::move(assets));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_parakeet_tdt_loader() {
    return std::make_shared<ParakeetTDTLoader>();
}

}  // namespace engine::models::parakeet_tdt
