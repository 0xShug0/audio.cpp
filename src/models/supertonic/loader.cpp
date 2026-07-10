#include "engine/models/supertonic/loader.h"

#include "engine/framework/io/filesystem.h"
#include "engine/models/supertonic/session.h"

#include <stdexcept>
#include <utility>

namespace engine::models::supertonic {
namespace {

std::filesystem::path resolve_model_root(const std::filesystem::path & model_path) {
    if (engine::io::is_existing_directory(model_path)) {
        return std::filesystem::weakly_canonical(model_path);
    }
    if (engine::io::is_existing_file(model_path)) {
        return std::filesystem::weakly_canonical(model_path.parent_path());
    }
    throw std::runtime_error("Supertonic model path does not exist: " + model_path.string());
}

bool has_supertonic_assets(const std::filesystem::path & root) {
    return engine::io::is_existing_file(root / "config" / "tts.json") &&
        engine::io::is_existing_file(root / "config" / "unicode_indexer.json") &&
        engine::io::is_existing_file(root / "ggml" / "supertonic.safetensors") &&
        engine::io::is_existing_directory(root / "voice_styles");
}

runtime::CapabilitySet supertonic_capabilities() {
    runtime::CapabilitySet capabilities;
    capabilities.supported_tasks = {
        {runtime::VoiceTaskKind::Tts, {runtime::RunMode::Offline}},
    };
    capabilities.languages = {
        "en", "ko", "ja", "ar", "bg", "cs", "da", "de", "el", "es", "et", "fi", "fr", "hi", "hr", "hu",
        "id", "it", "lt", "lv", "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "tr", "uk", "vi", "na",
    };
    capabilities.supports_style_condition = true;
    return capabilities;
}

runtime::ModelMetadata supertonic_metadata(const SupertonicAssets & assets) {
    runtime::ModelMetadata metadata;
    metadata.family = "supertonic";
    metadata.variant = "supertonic-3";
    metadata.description = "Supertonic 3 loaded from GGML safetensors assets.";
    metadata.config_candidates = {
        "config/tts.json",
        "config/unicode_indexer.json",
    };
    metadata.weight_candidates = {"ggml/supertonic.safetensors"};
    (void)assets;
    return metadata;
}

runtime::ModelCliInterface supertonic_cli() {
    runtime::ModelCliInterface cli;
    cli.request_options = {
        {"voice", "M1|F1", "Preset voice style id, default M1."},
        {"num_inference_steps", "n", "Flow denoising steps, default 8."},
        {"speaking_rate", "float", "Speech speed multiplier, default 1.05."},
        {"seed", "n", "Noise seed, default 1234."},
    };
    cli.session_options = {
        {"supertonic.weight_type", "native|f32|f16|bf16|q8_0", "Supertonic weight storage type."},
    };
    return cli;
}

std::vector<runtime::NamedAsset> discover_config_assets(const runtime::ModelLoadRequest & request) {
    const auto root = resolve_model_root(request.model_path);
    return runtime::discover_named_assets(root, {
        "config/tts.json",
        "config/unicode_indexer.json",
    });
}

std::vector<runtime::NamedAsset> discover_weight_assets(const runtime::ModelLoadRequest & request) {
    const auto root = resolve_model_root(request.model_path);
    return runtime::discover_named_assets(root, {"ggml/supertonic.safetensors"});
}

class SupertonicLoader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override { return "supertonic"; }

    bool can_load(const runtime::ModelLoadRequest & request) const override {
        try {
            const auto root = resolve_model_root(request.model_path);
            return has_supertonic_assets(root) && (!request.family_hint.has_value() || *request.family_hint == family());
        } catch (...) {
            return false;
        }
    }

    runtime::ModelInspection inspect(const runtime::ModelLoadRequest & request) const override {
        const auto assets = load_supertonic_assets(resolve_model_root(request.model_path));
        runtime::ModelInspection inspection;
        inspection.model_root = assets->paths.model_root;
        inspection.metadata = supertonic_metadata(*assets);
        inspection.capabilities = supertonic_capabilities();
        inspection.cli = supertonic_cli();
        inspection.discovered_configs = discover_config_assets(request);
        inspection.discovered_weights = discover_weight_assets(request);
        return inspection;
    }

    std::unique_ptr<runtime::ILoadedVoiceModel> load(const runtime::ModelLoadRequest & request) const override {
        return load_supertonic_model(resolve_model_root(request.model_path));
    }
};

}  // namespace

SupertonicLoadedModel::SupertonicLoadedModel(
    runtime::ModelMetadata metadata,
    runtime::CapabilitySet capabilities,
    std::shared_ptr<const SupertonicAssets> assets)
    : metadata_(std::move(metadata)),
      capabilities_(std::move(capabilities)),
      assets_(std::move(assets)) {}

const runtime::ModelMetadata & SupertonicLoadedModel::metadata() const noexcept {
    return metadata_;
}

const runtime::CapabilitySet & SupertonicLoadedModel::capabilities() const noexcept {
    return capabilities_;
}

std::unique_ptr<runtime::IVoiceTaskSession> SupertonicLoadedModel::create_task_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options) const {
    if (task.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("Supertonic only supports offline sessions");
    }
    if (task.task != runtime::VoiceTaskKind::Tts) {
        throw std::runtime_error("Supertonic only supports the Tts task");
    }
    return std::make_unique<SupertonicSession>(task, options, assets_);
}

std::unique_ptr<SupertonicLoadedModel> load_supertonic_model(const std::filesystem::path & model_path) {
    auto assets = load_supertonic_assets(model_path);
    return std::make_unique<SupertonicLoadedModel>(
        supertonic_metadata(*assets),
        supertonic_capabilities(),
        std::move(assets));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_supertonic_loader() {
    return std::make_shared<SupertonicLoader>();
}

}  // namespace engine::models::supertonic
