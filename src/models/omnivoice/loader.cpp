#include "engine/models/omnivoice/loader.h"

#include "engine/framework/assets/model_package.h"
#include "engine/models/omnivoice/session.h"

#include <stdexcept>
#include <utility>

namespace engine::models::omnivoice {
namespace {

runtime::ModelMetadata metadata(const OmniVoiceAssets & assets) {
    runtime::ModelMetadata out;
    out.family = "omnivoice";
    out.variant = assets.config.model_type;
    out.description = "OmniVoice multilingual TTS with voice clone and voice design pipelines.";
    return out;
}

runtime::CapabilitySet capabilities(const OmniVoiceAssets & assets) {
    runtime::CapabilitySet out;
    out.supported_tasks = {
        {runtime::VoiceTaskKind::Tts, {runtime::RunMode::Offline}},
    };
    out.supports_speaker_reference = true;
    out.supports_style_condition = true;
    out.languages = assets.config.supported_languages;
    return out;
}

runtime::ModelCliInterface cli(const OmniVoiceAssets &) {
    runtime::ModelCliInterface out;
    out.request_options = {
        {"text_chunk_mode", "default|tag_aware|japanese|endline", "Text chunking mode; default tag_aware."},
    };
    out.session_options = {
        {"omnivoice.mem_saver", "true|false", "Release staged runtime graphs after request phases; default false."},
    };
    return out;
}

class OmniVoiceLoader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return "omnivoice";
    }

    bool can_load(const runtime::ModelLoadRequest & request) const override {
        try {
            const auto package_spec = engine::assets::default_model_package_spec_path(family());
            (void) engine::assets::load_resource_bundle_from_package_spec(request.model_path, package_spec);
            return !request.family_hint.has_value() || *request.family_hint == family();
        } catch (...) {
            return false;
        }
    }

    runtime::ModelInspection inspect(const runtime::ModelLoadRequest & request) const override {
        const auto assets = load_omnivoice_assets(request.model_path);
        runtime::ModelInspection inspection;
        inspection.model_root = assets->resources.model_root();
        inspection.metadata = metadata(*assets);
        inspection.capabilities = capabilities(*assets);
        inspection.cli = cli(*assets);
        const auto package_spec = engine::assets::default_model_package_spec_path(family());
        inspection.discovered_configs = runtime::discover_named_assets_from_package_spec(
            request.model_path,
            package_spec,
            engine::assets::ModelPackageResourceKind::Files);
        inspection.discovered_weights = runtime::discover_named_assets_from_package_spec(
            request.model_path,
            package_spec,
            engine::assets::ModelPackageResourceKind::Tensors);
        return inspection;
    }

    std::unique_ptr<runtime::ILoadedVoiceModel> load(const runtime::ModelLoadRequest & request) const override {
        return load_omnivoice_model(request.model_path);
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
    return std::make_unique<OmniVoiceLoadedModel>(metadata(*assets), capabilities(*assets), std::move(assets));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_omnivoice_loader() {
    return std::make_shared<OmniVoiceLoader>();
}

}  // namespace engine::models::omnivoice
