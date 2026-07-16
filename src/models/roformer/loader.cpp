#include "engine/models/roformer/loader.h"

#include "engine/framework/assets/model_package.h"
#include "engine/models/roformer/session.h"

#include <stdexcept>
#include <utility>

namespace engine::models::roformer {
namespace {

runtime::ModelMetadata metadata(const RoformerAssets & assets) {
    runtime::ModelMetadata out;
    out.family = std::string(kMelBandRoformerFamily);
    out.variant = assets.resources.model_root().filename().string();
    out.description = "Mel-band RoFormer music source separation model.";
    return out;
}

runtime::CapabilitySet capabilities(const RoformerAssets &) {
    runtime::CapabilitySet out;
    out.supported_tasks = {
        {runtime::VoiceTaskKind::SourceSeparation, {runtime::RunMode::Offline}},
    };
    return out;
}

runtime::ModelCliInterface cli(const RoformerAssets &) {
    runtime::ModelCliInterface out;
    out.session_options = {
        {
            std::string(kMelBandRoformerFamily) + ".weight_type",
            "native|f32|f16|bf16|q8_0",
            "RoFormer weight storage type.",
        },
    };
    return out;
}

runtime::ModelInspection inspect_model(const runtime::ModelLoadRequest & request) {
    const auto assets = load_mel_band_roformer_assets(request);
    const auto package_spec = engine::assets::default_model_package_spec_path(std::string(kMelBandRoformerFamily));
    runtime::ModelInspection inspection;
    inspection.model_root = assets->resources.model_root();
    inspection.metadata = metadata(*assets);
    inspection.capabilities = capabilities(*assets);
    inspection.cli = cli(*assets);
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

class RoformerLoader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return std::string(kMelBandRoformerFamily);
    }

    bool can_load(const runtime::ModelLoadRequest & request) const override {
        if (request.family_hint.has_value() && *request.family_hint != family()) {
            return false;
        }
        try {
            (void) load_mel_band_roformer_assets(request);
            return true;
        } catch (...) {
            if (request.family_hint.has_value() && *request.family_hint == family()) {
                throw;
            }
            return false;
        }
    }

    runtime::ModelInspection inspect(const runtime::ModelLoadRequest & request) const override {
        return inspect_model(request);
    }

    std::unique_ptr<runtime::ILoadedVoiceModel> load(const runtime::ModelLoadRequest & request) const override {
        return load_roformer_model(request);
    }
};

}  // namespace

RoformerLoadedModel::RoformerLoadedModel(
    runtime::ModelMetadata metadata,
    runtime::CapabilitySet capabilities,
    std::shared_ptr<const RoformerAssets> assets)
    : metadata_(std::move(metadata)),
      capabilities_(std::move(capabilities)),
      assets_(std::move(assets)) {
    if (assets_ == nullptr) {
        throw std::runtime_error("RoFormer loaded model requires assets");
    }
}

const runtime::ModelMetadata & RoformerLoadedModel::metadata() const noexcept {
    return metadata_;
}

const runtime::CapabilitySet & RoformerLoadedModel::capabilities() const noexcept {
    return capabilities_;
}

std::unique_ptr<runtime::IVoiceTaskSession> RoformerLoadedModel::create_task_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options) const {
    return std::make_unique<RoformerSession>(task, options, assets_);
}

std::unique_ptr<runtime::ILoadedVoiceModel> load_roformer_model(
    const runtime::ModelLoadRequest & request) {
    auto assets = load_mel_band_roformer_assets(request);
    return std::make_unique<RoformerLoadedModel>(
        metadata(*assets),
        capabilities(*assets),
        std::move(assets));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_mel_band_roformer_loader() {
    return std::make_shared<RoformerLoader>();
}

}  // namespace engine::models::roformer
