#include "engine/models/miotts/loader.h"

#include "engine/framework/assets/model_package.h"
#include "engine/models/miotts/session.h"

#include <stdexcept>
#include <utility>

namespace engine::models::miotts {
namespace {

runtime::ModelMetadata metadata(const MioTTSAssets & assets) {
    runtime::ModelMetadata out;
    out.family = "miotts";
    out.variant = assets.config.model_type;
    out.description = "MioTTS loaded from local Qwen3 assets with MioCodec decoding.";
    return out;
}

runtime::CapabilitySet capabilities(const MioTTSAssets &) {
    runtime::CapabilitySet out;
    out.supported_tasks = {
        {runtime::VoiceTaskKind::Tts, {runtime::RunMode::Offline}},
    };
    out.supports_speaker_reference = true;
    out.supports_timestamps = false;
    return out;
}

class MioTTSLoader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return "miotts";
    }

    bool can_load(const runtime::ModelLoadRequest & request) const override {
        if (request.family_hint.has_value() && *request.family_hint != family()) {
            return false;
        }
        try {
            (void) engine::assets::load_resource_bundle_from_package_spec(
                request.model_path,
                engine::assets::default_model_package_spec_path(family()));
            return true;
        } catch (...) {
            return false;
        }
    }

    runtime::ModelInspection inspect(const runtime::ModelLoadRequest & request) const override {
        const auto assets = load_miotts_assets(request.model_path);
        runtime::ModelInspection inspection;
        inspection.model_root = assets->resources.model_root();
        inspection.metadata = metadata(*assets);
        inspection.capabilities = capabilities(*assets);
        const auto spec_path = engine::assets::default_model_package_spec_path(family());
        inspection.discovered_configs = runtime::discover_named_assets_from_package_spec(
            request.model_path,
            spec_path,
            engine::assets::ModelPackageResourceKind::Files);
        inspection.discovered_weights = runtime::discover_named_assets_from_package_spec(
            request.model_path,
            spec_path,
            engine::assets::ModelPackageResourceKind::Tensors);
        return inspection;
    }

    std::unique_ptr<runtime::ILoadedVoiceModel> load(const runtime::ModelLoadRequest & request) const override {
        return load_miotts_model(request.model_path);
    }
};

}  // namespace

MioTTSLoadedModel::MioTTSLoadedModel(
    runtime::ModelMetadata metadata,
    runtime::CapabilitySet capabilities,
    std::shared_ptr<const MioTTSAssets> assets)
    : metadata_(std::move(metadata)),
      capabilities_(std::move(capabilities)),
      assets_(std::move(assets)) {}

const runtime::ModelMetadata & MioTTSLoadedModel::metadata() const noexcept {
    return metadata_;
}

const runtime::CapabilitySet & MioTTSLoadedModel::capabilities() const noexcept {
    return capabilities_;
}

std::unique_ptr<runtime::IVoiceTaskSession> MioTTSLoadedModel::create_task_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options) const {
    if (task.task != runtime::VoiceTaskKind::Tts) {
        throw std::runtime_error("MioTTS only supports the Tts task");
    }
    if (task.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("MioTTS currently supports offline sessions");
    }
    return std::make_unique<MioTTSSession>(task, options, assets_);
}

std::unique_ptr<MioTTSLoadedModel> load_miotts_model(const std::filesystem::path & model_path) {
    auto assets = load_miotts_assets(model_path);
    return std::make_unique<MioTTSLoadedModel>(metadata(*assets), capabilities(*assets), std::move(assets));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_miotts_loader() {
    return std::make_shared<MioTTSLoader>();
}

}  // namespace engine::models::miotts
