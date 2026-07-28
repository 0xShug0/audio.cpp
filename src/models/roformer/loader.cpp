#include "engine/models/roformer/loader.h"

#include "engine/framework/model_spec/package.h"
#include "engine/models/roformer/session.h"

#include <stdexcept>
#include <utility>

namespace engine::models::roformer {
namespace {

runtime::ModelMetadata metadata(const RoformerAssets & assets) {
    runtime::ModelMetadata out;
    out.family = assets.config.family;
    out.variant = assets.resources.model_root().filename().string();
    out.description = assets.config.family == kBsRoformerFamily
        ? "Band-Split RoFormer music source separation model."
        : "Mel-band RoFormer music source separation model.";
    return out;
}

runtime::CapabilitySet capabilities(const RoformerAssets &) {
    runtime::CapabilitySet out;
    out.supported_tasks = {
        {runtime::VoiceTaskKind::SourceSeparation, {runtime::RunMode::Offline}},
    };
    return out;
}

runtime::ModelCliInterface cli(const RoformerAssets & assets) {
    runtime::ModelCliInterface out;
    out.session_options = {
        {
            assets.config.family + ".weight_type",
            "native|f32|f16|bf16|q8_0",
            "RoFormer weight storage type.",
        },
    };
    if (assets.config.family == kBsRoformerFamily) {
        out.session_options.push_back({
            assets.config.family + ".num_overlap",
            "n",
            "Number of overlapping inference windows; defaults to the package "
            "configuration. Lower values improve throughput but can reduce "
            "boundary quality.",
        });
    }
    return out;
}

runtime::ModelInspection inspect_model(
    const runtime::ModelLoadRequest & request,
    std::string_view family) {
    const auto assets = load_roformer_assets(request, family);
    const auto package_spec =
        engine::model_spec::default_spec_path(std::string(family));
    runtime::ModelInspection inspection;
    inspection.model_root = assets->resources.model_root();
    inspection.metadata = metadata(*assets);
    inspection.capabilities = capabilities(*assets);
    inspection.cli = cli(*assets);
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

class RoformerLoader final : public runtime::IVoiceModelLoader {
public:
    explicit RoformerLoader(std::string family)
        : family_(std::move(family)) {}

    std::string family() const override {
        return family_;
    }

    runtime::CapabilitySet advertised_capabilities() const override {
        runtime::CapabilitySet out;
        out.supported_tasks = {
            {runtime::VoiceTaskKind::SourceSeparation, {runtime::RunMode::Offline}},
        };
        return out;
    }

    bool can_load(const runtime::ModelLoadRequest & request) const override {
        if (request.family_hint.has_value() && *request.family_hint != family()) {
            return false;
        }
        try {
            (void) load_roformer_assets(request, family_);
            return true;
        } catch (...) {
            if (request.family_hint.has_value() && *request.family_hint == family()) {
                throw;
            }
            return false;
        }
    }

    runtime::ModelInspection inspect(const runtime::ModelLoadRequest & request) const override {
        return inspect_model(request, family_);
    }

    std::unique_ptr<runtime::ILoadedVoiceModel> load(const runtime::ModelLoadRequest & request) const override {
        return load_roformer_model(request, family_);
    }

private:
    std::string family_;
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
    const runtime::ModelLoadRequest & request,
    std::string_view family) {
    auto assets = load_roformer_assets(request, family);
    return std::make_unique<RoformerLoadedModel>(
        metadata(*assets),
        capabilities(*assets),
        std::move(assets));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_mel_band_roformer_loader() {
    return std::make_shared<RoformerLoader>(
        std::string(kMelBandRoformerFamily));
}

}  // namespace engine::models::roformer
