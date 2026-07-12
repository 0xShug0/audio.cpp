#include "engine/models/qwen3_forced_aligner/loader.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/io/filesystem.h"
#include "engine/framework/io/json.h"
#include "engine/models/qwen3_asr/assets.h"
#include "engine/models/qwen3_forced_aligner/session.h"

#include <stdexcept>
#include <utility>

namespace engine::models::qwen3_forced_aligner {
namespace {

std::filesystem::path resolve_model_root(const std::filesystem::path & model_path) {
    if (engine::io::is_existing_directory(model_path)) {
        return std::filesystem::weakly_canonical(model_path);
    }
    if (engine::io::is_existing_file(model_path)) {
        return std::filesystem::weakly_canonical(model_path.parent_path());
    }
    throw std::runtime_error("Qwen3 forced aligner model path does not exist: " + model_path.string());
}

bool has_forced_aligner_assets(const std::filesystem::path & root) {
    if (!engine::io::is_existing_file(root / "config.json") ||
        (!engine::io::is_existing_file(root / "model.gguf") &&
         !engine::io::is_existing_file(root / "model.safetensors")) ||
        !engine::io::is_existing_file(root / "tokenizer_config.json") ||
        !engine::io::is_existing_file(root / "vocab.json") ||
        !engine::io::is_existing_file(root / "merges.txt")) {
        return false;
    }
    const auto config = engine::io::json::parse_file(root / "config.json");
    return config.require("thinker_config").require("model_type").as_string() == "qwen3_forced_aligner";
}

std::vector<runtime::NamedAsset> discover_config_assets(const runtime::ModelLoadRequest & request) {
    const auto root = resolve_model_root(request.model_path);
    return runtime::discover_named_assets(
        root,
        {"config.json", "generation_config.json", "tokenizer_config.json"});
}

std::vector<runtime::NamedAsset> discover_weight_assets(const runtime::ModelLoadRequest & request) {
    const auto root = resolve_model_root(request.model_path);
    return runtime::discover_named_assets(root, {"model.gguf", "model.safetensors"});
}

class Qwen3ForcedAlignerLoader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return "qwen3_forced_aligner";
    }

    bool can_load(const runtime::ModelLoadRequest & request) const override {
        try {
            const auto gguf_path = engine::io::is_existing_directory(request.model_path)
                ? request.model_path / "model.gguf"
                : request.model_path;
            if (engine::io::is_existing_file(gguf_path) &&
                gguf_path.extension() == ".gguf" &&
                assets::gguf_has_embedded_sidecars(gguf_path)) {
                return !request.family_hint.has_value() || *request.family_hint == family();
            }
            const auto root = resolve_model_root(request.model_path);
            return has_forced_aligner_assets(root)
                && (!request.family_hint.has_value() || *request.family_hint == family());
        } catch (...) {
            return false;
        }
    }

    runtime::ModelInspection inspect(const runtime::ModelLoadRequest & request) const override {
        const auto assets = engine::models::qwen3_asr::load_qwen3_asr_assets(request.model_path);
        runtime::ModelInspection inspection;
        inspection.model_root = assets->paths.model_root;
        inspection.metadata.family = family();
        inspection.metadata.variant = assets->config.model_size.empty() ? assets->config.thinker_model_type : assets->config.model_size;
        inspection.metadata.description = "Qwen3 forced aligner loaded from local extracted assets.";
        inspection.metadata.config_candidates = {"config.json", "generation_config.json", "tokenizer_config.json"};
        inspection.metadata.weight_candidates = {"model.gguf", "model.safetensors"};
        inspection.capabilities.supported_tasks = {
            {runtime::VoiceTaskKind::Alignment, {runtime::RunMode::Offline}},
        };
        inspection.capabilities.supports_timestamps = true;
        inspection.capabilities.languages = assets->config.supported_languages;
        inspection.discovered_configs = discover_config_assets(request);
        inspection.discovered_weights = discover_weight_assets(request);
        return inspection;
    }

    std::unique_ptr<runtime::ILoadedVoiceModel> load(const runtime::ModelLoadRequest & request) const override {
        return load_qwen3_forced_aligner_model(request.model_path);
    }
};

}  // namespace

Qwen3ForcedAlignerLoadedModel::Qwen3ForcedAlignerLoadedModel(
    runtime::ModelMetadata metadata,
    runtime::CapabilitySet capabilities,
    std::shared_ptr<const engine::models::qwen3_asr::Qwen3ASRAssets> assets)
    : metadata_(std::move(metadata)),
      capabilities_(std::move(capabilities)),
      assets_(std::move(assets)) {}

const runtime::ModelMetadata & Qwen3ForcedAlignerLoadedModel::metadata() const noexcept {
    return metadata_;
}

const runtime::CapabilitySet & Qwen3ForcedAlignerLoadedModel::capabilities() const noexcept {
    return capabilities_;
}

std::unique_ptr<runtime::IVoiceTaskSession> Qwen3ForcedAlignerLoadedModel::create_task_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options) const {
    if (task.task != runtime::VoiceTaskKind::Alignment) {
        throw std::runtime_error("Qwen3 forced aligner only supports the Alignment task");
    }
    if (task.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("Qwen3 forced aligner currently supports offline sessions");
    }
    return std::make_unique<Qwen3ForcedAlignerSession>(task, options, assets_);
}

std::unique_ptr<Qwen3ForcedAlignerLoadedModel> load_qwen3_forced_aligner_model(const std::filesystem::path & model_path) {
    auto assets = engine::models::qwen3_asr::load_qwen3_asr_assets(model_path);
    if (assets->config.thinker_model_type != "qwen3_forced_aligner") {
        throw std::runtime_error("Qwen3 forced aligner loader received non-aligner assets");
    }

    runtime::ModelMetadata metadata;
    metadata.family = "qwen3_forced_aligner";
    metadata.variant = assets->config.model_size.empty() ? assets->config.thinker_model_type : assets->config.model_size;
    metadata.description = "Qwen3 forced aligner loaded from local extracted assets.";
    metadata.config_candidates = {"config.json", "generation_config.json", "tokenizer_config.json"};
    metadata.weight_candidates = {"model.gguf", "model.safetensors"};

    runtime::CapabilitySet capabilities;
    capabilities.supported_tasks = {
        {runtime::VoiceTaskKind::Alignment, {runtime::RunMode::Offline}},
    };
    capabilities.languages = assets->config.supported_languages;
    capabilities.supports_timestamps = true;

    return std::make_unique<Qwen3ForcedAlignerLoadedModel>(std::move(metadata), std::move(capabilities), std::move(assets));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_qwen3_forced_aligner_loader() {
    return std::make_shared<Qwen3ForcedAlignerLoader>();
}

}  // namespace engine::models::qwen3_forced_aligner
