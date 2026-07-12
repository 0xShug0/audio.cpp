#include "engine/models/qwen3_asr/loader.h"

#include "engine/framework/io/filesystem.h"
#include "engine/models/qwen3_asr/session.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace engine::models::qwen3_asr {
namespace {

std::filesystem::path resolve_model_root(const std::filesystem::path & model_path) {
    if (engine::io::is_existing_directory(model_path)) {
        return std::filesystem::weakly_canonical(model_path);
    }
    if (engine::io::is_existing_file(model_path)) {
        return std::filesystem::weakly_canonical(model_path.parent_path());
    }
    throw std::runtime_error("Qwen3 ASR model path does not exist: " + model_path.string());
}

bool has_qwen3_asr_assets(const std::filesystem::path & root) {
    const bool has_frontend = engine::io::is_existing_file(root / "preprocessor_config.json")
        || engine::io::is_existing_file(root / "processor_config.json");
    const bool has_tokenizer = engine::io::is_existing_file(root / "tokenizer.json")
        || (engine::io::is_existing_file(root / "vocab.json")
            && engine::io::is_existing_file(root / "merges.txt"));
    return engine::io::is_existing_file(root / "config.json")
        && engine::io::is_existing_file(root / "model.safetensors")
        && engine::io::is_existing_file(root / "tokenizer_config.json")
        && engine::io::is_existing_file(root / "generation_config.json")
        && has_frontend
        && has_tokenizer;
}

std::vector<runtime::NamedAsset> discover_config_assets(const runtime::ModelLoadRequest & request) {
    const auto root = resolve_model_root(request.model_path);
    return runtime::discover_named_assets(
        root,
        {"config.json", "generation_config.json", "preprocessor_config.json", "processor_config.json",
         "tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt"});
}

std::vector<runtime::NamedAsset> discover_weight_assets(const runtime::ModelLoadRequest & request) {
    const auto root = resolve_model_root(request.model_path);
    return runtime::discover_named_assets(root, {"model.safetensors"});
}

class Qwen3ASRLoader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return "qwen3_asr";
    }

    bool can_load(const runtime::ModelLoadRequest & request) const override {
        try {
            const auto root = resolve_model_root(request.model_path);
            return has_qwen3_asr_assets(root)
                && (!request.family_hint.has_value() || *request.family_hint == family());
        } catch (...) {
            return false;
        }
    }

    runtime::ModelInspection inspect(const runtime::ModelLoadRequest & request) const override {
        const auto assets = load_qwen3_asr_assets(resolve_model_root(request.model_path));
        runtime::ModelInspection inspection;
        inspection.model_root = assets->paths.model_root;
        inspection.metadata.family = family();
        inspection.metadata.variant = assets->config.model_size.empty() ? assets->config.model_type : assets->config.model_size;
        inspection.metadata.description = "Qwen3 ASR loaded from local extracted assets.";
        inspection.metadata.config_candidates = {"config.json", "generation_config.json", "tokenizer_config.json"};
        inspection.metadata.weight_candidates = {"model.safetensors"};
        inspection.capabilities.supported_tasks = {
            {runtime::VoiceTaskKind::Asr, {runtime::RunMode::Offline}},
        };
        inspection.capabilities.supports_timestamps = false;
        inspection.capabilities.languages = assets->config.supported_languages;
        inspection.discovered_configs = discover_config_assets(request);
        inspection.discovered_weights = discover_weight_assets(request);
        return inspection;
    }

    std::unique_ptr<runtime::ILoadedVoiceModel> load(const runtime::ModelLoadRequest & request) const override {
        return load_qwen3_asr_model(resolve_model_root(request.model_path));
    }
};

}  // namespace

Qwen3ASRLoadedModel::Qwen3ASRLoadedModel(
    runtime::ModelMetadata metadata,
    runtime::CapabilitySet capabilities,
    std::shared_ptr<const Qwen3ASRAssets> assets)
    : metadata_(std::move(metadata)),
      capabilities_(std::move(capabilities)),
      assets_(std::move(assets)) {}

const runtime::ModelMetadata & Qwen3ASRLoadedModel::metadata() const noexcept {
    return metadata_;
}

const runtime::CapabilitySet & Qwen3ASRLoadedModel::capabilities() const noexcept {
    return capabilities_;
}

std::unique_ptr<runtime::IVoiceTaskSession> Qwen3ASRLoadedModel::create_task_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options) const {
    if (task.task != runtime::VoiceTaskKind::Asr) {
        throw std::runtime_error("Qwen3 ASR only supports the Asr task");
    }
    if (task.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("Qwen3 ASR currently supports offline sessions");
    }
    return std::make_unique<Qwen3ASRSession>(task, options, assets_);
}

std::unique_ptr<Qwen3ASRLoadedModel> load_qwen3_asr_model(const std::filesystem::path & model_path) {
    auto assets = load_qwen3_asr_assets(model_path);

    runtime::ModelMetadata metadata;
    metadata.family = "qwen3_asr";
    metadata.variant = assets->config.model_size.empty() ? assets->config.model_type : assets->config.model_size;
    metadata.description = "Qwen3 ASR loaded from local extracted assets.";
    metadata.config_candidates = {"config.json", "generation_config.json", "tokenizer_config.json"};
    metadata.weight_candidates = {"model.safetensors"};

    runtime::CapabilitySet capabilities;
    capabilities.supported_tasks = {
        {runtime::VoiceTaskKind::Asr, {runtime::RunMode::Offline}},
    };
    capabilities.languages = assets->config.supported_languages;
    capabilities.supports_timestamps = false;

    return std::make_unique<Qwen3ASRLoadedModel>(std::move(metadata), std::move(capabilities), std::move(assets));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_qwen3_asr_loader() {
    return std::make_shared<Qwen3ASRLoader>();
}

}  // namespace engine::models::qwen3_asr
