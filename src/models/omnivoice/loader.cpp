#include "engine/models/omnivoice/loader.h"

#include "engine/framework/io/filesystem.h"
#include "engine/models/omnivoice/session.h"

#include <stdexcept>
#include <utility>

namespace engine::models::omnivoice {
namespace {

std::filesystem::path resolve_model_root(const std::filesystem::path & model_path) {
    if (engine::io::is_existing_directory(model_path)) {
        return std::filesystem::weakly_canonical(model_path);
    }
    if (engine::io::is_existing_file(model_path)) {
        return std::filesystem::weakly_canonical(model_path.parent_path());
    }
    throw std::runtime_error("OmniVoice model path does not exist: " + model_path.string());
}

bool has_omnivoice_assets(const std::filesystem::path & root) {
    return engine::io::is_existing_file(root / "config.json") &&
        engine::io::is_existing_file(root / "model.safetensors") &&
        engine::io::is_existing_file(root / "tokenizer.json") &&
        engine::io::is_existing_file(root / "tokenizer_config.json") &&
        engine::io::is_existing_file(root / "audio_tokenizer" / "config.json") &&
        engine::io::is_existing_file(root / "audio_tokenizer" / "model.safetensors") &&
        engine::io::is_existing_file(root / "audio_tokenizer" / "preprocessor_config.json");
}

std::vector<runtime::NamedAsset> discover_config_assets(const runtime::ModelLoadRequest & request) {
    const auto root = resolve_model_root(request.model_path);
    return runtime::discover_named_assets(
        root,
        {
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
            "audio_tokenizer/config.json",
            "audio_tokenizer/preprocessor_config.json",
        });
}

std::vector<runtime::NamedAsset> discover_weight_assets(const runtime::ModelLoadRequest & request) {
    const auto root = resolve_model_root(request.model_path);
    return runtime::discover_named_assets(root, {"model.safetensors", "audio_tokenizer/model.safetensors"});
}

class OmniVoiceLoader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return "omnivoice";
    }

    bool can_load(const runtime::ModelLoadRequest & request) const override {
        try {
            const auto root = resolve_model_root(request.model_path);
            return has_omnivoice_assets(root) &&
                (!request.family_hint.has_value() || *request.family_hint == family());
        } catch (...) {
            return false;
        }
    }

    runtime::ModelInspection inspect(const runtime::ModelLoadRequest & request) const override {
        const auto assets = load_omnivoice_assets(resolve_model_root(request.model_path));
        runtime::ModelInspection inspection;
        inspection.model_root = assets->paths.model_root;
        inspection.metadata.family = family();
        inspection.metadata.variant = assets->config.model_type;
        inspection.metadata.description = "OmniVoice multilingual TTS with voice clone and voice design pipelines.";
        inspection.metadata.config_candidates = {
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
            "audio_tokenizer/config.json",
            "audio_tokenizer/preprocessor_config.json",
        };
        inspection.metadata.weight_candidates = {"model.safetensors", "audio_tokenizer/model.safetensors"};
        inspection.cli.session_options = {
            {"omnivoice.mem_saver", "true|false", "Release staged runtime graphs after request phases; default false."},
        };
        inspection.capabilities.supported_tasks = {
            {runtime::VoiceTaskKind::Tts, {runtime::RunMode::Offline}},
        };
        inspection.capabilities.supports_speaker_reference = true;
        inspection.capabilities.supports_style_condition = true;
        inspection.capabilities.languages = assets->config.supported_languages;
        inspection.discovered_configs = discover_config_assets(request);
        inspection.discovered_weights = discover_weight_assets(request);
        return inspection;
    }

    std::unique_ptr<runtime::ILoadedVoiceModel> load(const runtime::ModelLoadRequest & request) const override {
        return load_omnivoice_model(resolve_model_root(request.model_path));
    }
};

}  // namespace

OmniVoiceLoadedModel::OmniVoiceLoadedModel(
    runtime::ModelMetadata metadata,
    runtime::CapabilitySet capabilities,
    std::shared_ptr<const OmniVoiceAssets> assets)
    : metadata_(std::move(metadata)),
      capabilities_(std::move(capabilities)),
      assets_(std::move(assets)) {}

const runtime::ModelMetadata & OmniVoiceLoadedModel::metadata() const noexcept {
    return metadata_;
}

const runtime::CapabilitySet & OmniVoiceLoadedModel::capabilities() const noexcept {
    return capabilities_;
}

std::unique_ptr<runtime::IVoiceTaskSession> OmniVoiceLoadedModel::create_task_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options) const {
    return std::make_unique<OmniVoiceSession>(task, options, assets_);
}

std::unique_ptr<OmniVoiceLoadedModel> load_omnivoice_model(const std::filesystem::path & model_path) {
    auto assets = load_omnivoice_assets(model_path);

    runtime::ModelMetadata metadata;
    metadata.family = "omnivoice";
    metadata.variant = assets->config.model_type;
    metadata.description = "OmniVoice multilingual TTS with voice clone and voice design pipelines.";
    metadata.config_candidates = {
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "audio_tokenizer/config.json",
        "audio_tokenizer/preprocessor_config.json",
    };
    metadata.weight_candidates = {"model.safetensors", "audio_tokenizer/model.safetensors"};
    runtime::CapabilitySet capabilities;
    capabilities.supported_tasks = {
        {runtime::VoiceTaskKind::Tts, {runtime::RunMode::Offline}},
    };
    capabilities.supports_speaker_reference = true;
    capabilities.supports_style_condition = true;
    capabilities.languages = assets->config.supported_languages;

    return std::make_unique<OmniVoiceLoadedModel>(std::move(metadata), std::move(capabilities), std::move(assets));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_omnivoice_loader() {
    return std::make_shared<OmniVoiceLoader>();
}

}  // namespace engine::models::omnivoice
