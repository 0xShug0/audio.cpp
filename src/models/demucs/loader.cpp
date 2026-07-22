#include "engine/models/demucs/loader.h"

#include "engine/framework/model_spec/package.h"
#include "engine/models/demucs/session.h"

#include <stdexcept>
#include <utility>

namespace engine::models::demucs {
namespace {

runtime::ModelCliInterface cli() {
    runtime::ModelCliInterface out;
    out.session_options = {
        {"htdemucs.weight_type", "native|f32|f16|bf16|q8_0", "HTDemucs weight storage type."},
    };
    return out;
}

runtime::ModelMetadata metadata(const HTDemucsAssets & assets) {
    runtime::ModelMetadata out;
    out.family = "htdemucs";
    out.variant = assets.manifest.name;
    out.description = "HTDemucs source separation model converted from a Demucs reference checkpoint.";
    return out;
}

runtime::CapabilitySet capabilities(const HTDemucsAssets &) {
    runtime::CapabilitySet out;
    out.supported_tasks = {
        {runtime::VoiceTaskKind::SourceSeparation, {runtime::RunMode::Offline}},
    };
    out.languages = {"N/A"};
    return out;
}

class HTDemucsLoader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return "htdemucs";
    }

    runtime::CapabilitySet advertised_capabilities() const override {
        runtime::CapabilitySet out;
        out.supported_tasks = {
            {runtime::VoiceTaskKind::SourceSeparation, {runtime::RunMode::Offline}},
        };
        return out;
    }

    bool can_load(const runtime::ModelLoadRequest & request) const override {
        try {
            const auto package_spec = engine::model_spec::default_spec_path(family());
            (void) engine::model_spec::load_resource_bundle(request.model_path, package_spec);
            return !request.family_hint.has_value() || *request.family_hint == family();
        } catch (...) {
            return false;
        }
    }

    runtime::ModelInspection inspect(const runtime::ModelLoadRequest & request) const override {
        const auto model_assets = load_htdemucs_assets(request);
        const auto package_spec = engine::model_spec::default_spec_path(family());
        runtime::ModelInspection inspection;
        inspection.model_root = model_assets->resources.model_root();
        inspection.metadata = metadata(*model_assets);
        inspection.capabilities = capabilities(*model_assets);
        inspection.cli = cli();
        inspection.discovered_configs = runtime::discover_named_assets_from_package_spec(
            request.model_path,
            package_spec,
            engine::model_spec::ResourceKind::Files);
        inspection.discovered_weights = runtime::discover_named_assets_from_package_spec(
            request.model_path,
            package_spec,
            engine::model_spec::ResourceKind::Tensors);
        return inspection;
    }

    std::unique_ptr<runtime::ILoadedVoiceModel> load(const runtime::ModelLoadRequest & request) const override {
        return load_htdemucs_model(request);
    }
};

}  // namespace

HTDemucsLoadedModel::HTDemucsLoadedModel(
    runtime::ModelMetadata metadata,
    runtime::CapabilitySet capabilities,
    std::shared_ptr<const HTDemucsAssets> assets)
    : metadata_(std::move(metadata)),
      capabilities_(std::move(capabilities)),
      assets_(std::move(assets)) {
    if (assets_ == nullptr) {
        throw std::runtime_error("HTDemucs loaded model requires assets");
    }
}

const runtime::ModelMetadata & HTDemucsLoadedModel::metadata() const noexcept {
    return metadata_;
}

const runtime::CapabilitySet & HTDemucsLoadedModel::capabilities() const noexcept {
    return capabilities_;
}

std::unique_ptr<runtime::IVoiceTaskSession> HTDemucsLoadedModel::create_task_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options) const {
    return std::make_unique<HTDemucsSession>(task, options, assets_);
}

std::unique_ptr<runtime::ILoadedVoiceModel> load_htdemucs_model(const runtime::ModelLoadRequest & request) {
    auto assets = load_htdemucs_assets(request);
    return std::make_unique<HTDemucsLoadedModel>(
        metadata(*assets),
        capabilities(*assets),
        std::move(assets));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_htdemucs_loader() {
    return std::make_shared<HTDemucsLoader>();
}

}  // namespace engine::models::demucs
