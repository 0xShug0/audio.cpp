#include "engine/models/kokoro_tts/loader.h"
#include "engine/models/kokoro_tts/session.h"

#include "engine/framework/io/filesystem.h"

#include <stdexcept>

namespace engine::models::kokoro_tts {

namespace {

std::filesystem::path resolve_model_root(const std::filesystem::path & model_path) {
    if (engine::io::is_existing_directory(model_path)) {
        return std::filesystem::weakly_canonical(model_path);
    }
    throw std::runtime_error("Kokoro TTS expects a model directory: " + model_path.string());
}

std::vector<runtime::NamedAsset> discover_config_assets(const runtime::ModelLoadRequest & request) {
    return runtime::discover_named_assets(resolve_model_root(request.model_path), {"config.json", "voices.json", "vocab.tsv"});
}

std::vector<runtime::NamedAsset> discover_weight_assets(const runtime::ModelLoadRequest & request) {
    return runtime::discover_named_assets(resolve_model_root(request.model_path), {"kokoro-v1_0.safetensors"});
}

class KokoroTTSLoader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return "kokoro_tts";
    }

    bool can_load(const runtime::ModelLoadRequest & request) const override {
        try {
            const auto root = resolve_model_root(request.model_path);
            return engine::io::is_existing_file(root / "config.json")
                && engine::io::is_existing_file(root / "voices.json")
                && engine::io::is_existing_file(root / "kokoro-v1_0.safetensors")
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
        inspection.metadata.description = "Kokoro TTS loaded from local extracted assets.";
        inspection.metadata.config_candidates = {"config.json", "voices.json", "vocab.tsv"};
        inspection.metadata.weight_candidates = {"kokoro-v1_0.safetensors"};
        inspection.capabilities.supported_tasks = {
            {runtime::VoiceTaskKind::Tts, {runtime::RunMode::Offline}},
        };
        inspection.capabilities.languages = {"a", "b"};
        inspection.capabilities.supports_style_condition = true;
        inspection.discovered_configs = discover_config_assets(request);
        inspection.discovered_weights = discover_weight_assets(request);
        return inspection;
    }

    std::unique_ptr<runtime::ILoadedVoiceModel> load(const runtime::ModelLoadRequest & request) const override {
        return load_kokoro_tts_model(resolve_model_root(request.model_path));
    }
};

}  // namespace

KokoroTTSLoadedModel::KokoroTTSLoadedModel(
    runtime::ModelMetadata metadata,
    runtime::CapabilitySet capabilities,
    std::shared_ptr<const KokoroAssets> assets)
    : metadata_(std::move(metadata)),
      capabilities_(std::move(capabilities)),
      assets_(std::move(assets)) {}

const runtime::ModelMetadata & KokoroTTSLoadedModel::metadata() const noexcept {
    return metadata_;
}

const runtime::CapabilitySet & KokoroTTSLoadedModel::capabilities() const noexcept {
    return capabilities_;
}

std::unique_ptr<runtime::IVoiceTaskSession> KokoroTTSLoadedModel::create_task_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options) const {
    if (task.task != runtime::VoiceTaskKind::Tts) {
        throw std::runtime_error("Kokoro TTS only supports VoiceTaskKind::Tts");
    }
    return std::make_unique<KokoroTTSSession>(task, options, assets_);
}

std::unique_ptr<KokoroTTSLoadedModel> load_kokoro_tts_model(const std::filesystem::path & model_path) {
    const auto root = resolve_model_root(model_path);
    auto assets = load_kokoro_assets(root);

    runtime::ModelMetadata metadata;
    metadata.family = "kokoro_tts";
    metadata.variant = root.filename().string();
    metadata.description = "Kokoro TTS loaded from local extracted assets.";
    metadata.config_candidates = {"config.json", "voices.json", "vocab.tsv"};
    metadata.weight_candidates = {"kokoro-v1_0.safetensors"};

    runtime::CapabilitySet capabilities;
    capabilities.supported_tasks = {
        {runtime::VoiceTaskKind::Tts, {runtime::RunMode::Offline}},
    };
    capabilities.languages = {"a", "b"};
    capabilities.supports_style_condition = true;

    return std::make_unique<KokoroTTSLoadedModel>(std::move(metadata), std::move(capabilities), std::move(assets));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_kokoro_tts_loader() {
    return std::make_shared<KokoroTTSLoader>();
}

}  // namespace engine::models::kokoro_tts
