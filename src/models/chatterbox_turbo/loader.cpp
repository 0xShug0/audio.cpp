#include "engine/models/chatterbox_turbo/loader.h"

#include "engine/models/chatterbox_turbo/session.h"

#include "engine/framework/model_spec/package.h"

#include <memory>
#include <stdexcept>

namespace engine::models::chatterbox_turbo {

namespace {

runtime::CapabilitySet capabilities(const ChatterboxTurboAssets &) {
    runtime::CapabilitySet out;
    out.supported_tasks.push_back({runtime::VoiceTaskKind::VoiceCloning, {runtime::RunMode::Offline}});
    out.languages = {"en"};
    // Custom voice cloning (a caller-supplied speaker reference) is not implemented yet -- only
    // the built-in default voice baked into the GGUF's `conds.*` tensors is supported. Advertise
    // this honestly rather than claiming speaker-reference support the loader would reject.
    out.supports_speaker_reference = false;
    out.supports_style_condition = false;
    return out;
}

runtime::ModelMetadata metadata(const ChatterboxTurboAssets & assets) {
    runtime::ModelMetadata out;
    out.family = "chatterbox_turbo";
    out.variant = assets.resources.model_root().filename().string();
    out.description =
        "Chatterbox Turbo (distilled GPT2 T3 backbone + meanflow S3Gen decoder) loaded from local assets. "
        "Built-in default voice only; custom voice cloning is not yet implemented.";
    return out;
}

class ChatterboxTurboLoader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return "chatterbox_turbo";
    }

    runtime::CapabilitySet advertised_capabilities() const override {
        runtime::CapabilitySet out;
        out.supported_tasks.push_back({runtime::VoiceTaskKind::VoiceCloning, {runtime::RunMode::Offline}});
        out.languages = {"en"};
        out.supports_speaker_reference = false;
        out.supports_style_condition = false;
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
        const auto assets = load_chatterbox_turbo_assets(request.model_path);
        runtime::ModelInspection inspection;
        inspection.model_root = assets->resources.model_root();
        inspection.metadata = metadata(*assets);
        inspection.capabilities = capabilities(*assets);
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
        return load_chatterbox_turbo_model(request.model_path);
    }
};

}  // namespace

ChatterboxTurboLoadedModel::ChatterboxTurboLoadedModel(
    runtime::ModelMetadata metadata,
    runtime::CapabilitySet capabilities,
    std::shared_ptr<const ChatterboxTurboAssets> assets)
    : metadata_(std::move(metadata)),
      capabilities_(std::move(capabilities)),
      assets_(std::move(assets)) {
    if (assets_ == nullptr) {
        throw std::runtime_error("Chatterbox Turbo loaded model requires assets");
    }
}

const runtime::ModelMetadata & ChatterboxTurboLoadedModel::metadata() const noexcept {
    return metadata_;
}

const runtime::CapabilitySet & ChatterboxTurboLoadedModel::capabilities() const noexcept {
    return capabilities_;
}

std::unique_ptr<runtime::IVoiceTaskSession> ChatterboxTurboLoadedModel::create_task_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options) const {
    if (task.task != runtime::VoiceTaskKind::VoiceCloning) {
        throw std::runtime_error("Chatterbox Turbo supports VoiceCloning only");
    }
    if (task.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("Chatterbox Turbo only supports offline mode");
    }
    return std::make_unique<ChatterboxTurboSession>(task, options, assets_);
}

std::unique_ptr<ChatterboxTurboLoadedModel> load_chatterbox_turbo_model(const std::filesystem::path & model_root) {
    auto assets = load_chatterbox_turbo_assets(model_root);
    return std::make_unique<ChatterboxTurboLoadedModel>(metadata(*assets), capabilities(*assets), std::move(assets));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_chatterbox_turbo_loader() {
    return std::make_shared<ChatterboxTurboLoader>();
}

}  // namespace engine::models::chatterbox_turbo
