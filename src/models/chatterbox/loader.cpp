#include "engine/models/chatterbox/loader.h"

#include "engine/models/chatterbox/session.h"
#include "engine/models/chatterbox/text_tokenizer.h"

#include "engine/framework/model_spec/package.h"

#include <memory>
#include <stdexcept>

namespace engine::models::chatterbox {

namespace {

runtime::CapabilitySet capabilities(const ChatterboxAssets &) {
    runtime::CapabilitySet capabilities;
    capabilities.supported_tasks.push_back({runtime::VoiceTaskKind::VoiceCloning, {runtime::RunMode::Offline}});
    capabilities.supported_tasks.push_back({runtime::VoiceTaskKind::VoiceConversion, {runtime::RunMode::Offline}});
    capabilities.languages = supported_chatterbox_language_codes();
    capabilities.supports_speaker_reference = true;
    capabilities.supports_style_condition = true;
    return capabilities;
}

runtime::ModelMetadata metadata(const ChatterboxAssets & assets) {
    runtime::ModelMetadata out;
    out.family = "chatterbox";
    out.variant = assets.resources.model_root().filename().string();
    out.description = "Chatterbox voice cloning and voice conversion loaded from local assets.";
    return out;
}

runtime::ModelCliInterface cli(const ChatterboxAssets &) {
    runtime::ModelCliInterface out;
    out.session_options = {
        {
            "chatterbox.conditionals_cache_slots",
            "n",
            "Prepared voice-condition cache slots; default 1, set 0 to disable.",
        },
        {
            "chatterbox.mem_saver",
            "true|false",
            "Free non-conditional runtime graphs after each request chunk; default false.",
        },
    };
    return out;
}

class ChatterboxLoader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return "chatterbox";
    }

    runtime::CapabilitySet advertised_capabilities() const override {
        runtime::CapabilitySet out;
        out.supported_tasks.push_back({runtime::VoiceTaskKind::VoiceCloning, {runtime::RunMode::Offline}});
        out.supported_tasks.push_back({runtime::VoiceTaskKind::VoiceConversion, {runtime::RunMode::Offline}});
        out.languages = supported_chatterbox_language_codes();
        out.supports_speaker_reference = true;
        out.supports_style_condition = true;
        return out;
    }

    bool can_load(const runtime::ModelLoadRequest & request) const override {
        try {
            (void) engine::model_spec::load_resource_bundle(
                request.model_path,
                engine::model_spec::default_spec_path(family()));
            return !request.family_hint.has_value() || *request.family_hint == family();
        } catch (...) {
            return false;
        }
    }

    runtime::ModelInspection inspect(const runtime::ModelLoadRequest & request) const override {
        const auto assets = load_chatterbox_assets(request.model_path);
        runtime::ModelInspection inspection;
        inspection.model_root = assets->resources.model_root();
        inspection.metadata = metadata(*assets);
        inspection.capabilities = capabilities(*assets);
        inspection.cli = cli(*assets);
        const auto spec_path = engine::model_spec::default_spec_path(family());
        inspection.discovered_configs = runtime::discover_named_assets_from_package_spec(
            request.model_path,
            spec_path,
            engine::model_spec::ResourceKind::Files);
        inspection.discovered_weights = runtime::discover_named_assets_from_package_spec(
            request.model_path,
            spec_path,
            engine::model_spec::ResourceKind::Tensors);
        return inspection;
    }

    std::unique_ptr<runtime::ILoadedVoiceModel> load(const runtime::ModelLoadRequest & request) const override {
        return load_chatterbox_model(request.model_path);
    }
};

}  // namespace

ChatterboxLoadedModel::ChatterboxLoadedModel(
    runtime::ModelMetadata metadata,
    runtime::CapabilitySet capabilities,
    std::shared_ptr<const ChatterboxAssets> assets)
    : metadata_(std::move(metadata)),
      capabilities_(std::move(capabilities)),
      assets_(std::move(assets)) {
    if (assets_ == nullptr) {
        throw std::runtime_error("Chatterbox loaded model requires assets");
    }
}

const runtime::ModelMetadata & ChatterboxLoadedModel::metadata() const noexcept {
    return metadata_;
}

const runtime::CapabilitySet & ChatterboxLoadedModel::capabilities() const noexcept {
    return capabilities_;
}

std::unique_ptr<runtime::IVoiceTaskSession> ChatterboxLoadedModel::create_task_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options) const {
    if (task.task != runtime::VoiceTaskKind::VoiceCloning &&
        task.task != runtime::VoiceTaskKind::VoiceConversion) {
        throw std::runtime_error("Chatterbox supports VoiceCloning and VoiceConversion");
    }
    if (task.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("Chatterbox only supports offline mode");
    }
    return std::make_unique<ChatterboxSession>(task, options, assets_);
}

std::unique_ptr<ChatterboxLoadedModel> load_chatterbox_model(const std::filesystem::path & model_root) {
    auto assets = load_chatterbox_assets(model_root);

    return std::make_unique<ChatterboxLoadedModel>(
        metadata(*assets),
        capabilities(*assets),
        std::move(assets));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_chatterbox_loader() {
    return std::make_shared<ChatterboxLoader>();
}

}  // namespace engine::models::chatterbox
