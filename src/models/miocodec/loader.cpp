#include "engine/models/miocodec/loader.h"

#include "engine/framework/assets/model_package.h"
#include "engine/models/miocodec/session.h"

#include <stdexcept>
#include <utility>

namespace engine::models::miocodec {
namespace {

runtime::CapabilitySet capabilities(const MioCodecAssets &) {
    runtime::CapabilitySet out;
    out.supported_tasks = {
        {runtime::VoiceTaskKind::VoiceConversion, {runtime::RunMode::Offline}},
        {runtime::VoiceTaskKind::SpeechToSpeech, {runtime::RunMode::Offline}},
    };
    out.supports_speaker_reference = true;
    out.supports_style_condition = false;
    out.supports_timestamps = false;
    return out;
}

runtime::ModelMetadata metadata(const MioCodecAssets & assets) {
    runtime::ModelMetadata out;
    out.family = "miocodec";
    out.variant = std::to_string(assets.config.sample_rate) + "hz-" + std::to_string(assets.config.codebook_size);
    out.description = "MioCodec loaded from local extracted assets.";
    return out;
}

class MioCodecLoader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return "miocodec";
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
        const auto assets = load_miocodec_assets(request.model_path);
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
        return load_miocodec_model(request.model_path);
    }
};

}  // namespace

MioCodecLoadedModel::MioCodecLoadedModel(
    runtime::ModelMetadata metadata,
    runtime::CapabilitySet capabilities,
    std::shared_ptr<const MioCodecAssets> assets)
    : metadata_(std::move(metadata)),
      capabilities_(std::move(capabilities)),
      assets_(std::move(assets)) {}

const runtime::ModelMetadata & MioCodecLoadedModel::metadata() const noexcept {
    return metadata_;
}

const runtime::CapabilitySet & MioCodecLoadedModel::capabilities() const noexcept {
    return capabilities_;
}

std::unique_ptr<runtime::IVoiceTaskSession> MioCodecLoadedModel::create_task_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options) const {
    if (task.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("MioCodec only supports offline sessions");
    }
    if (task.task != runtime::VoiceTaskKind::VoiceConversion
        && task.task != runtime::VoiceTaskKind::SpeechToSpeech) {
        throw std::runtime_error("MioCodec supports VoiceConversion or SpeechToSpeech tasks");
    }
    return std::make_unique<MioCodecSession>(task, options, assets_);
}

std::unique_ptr<MioCodecLoadedModel> load_miocodec_model(const std::filesystem::path & model_path) {
    auto assets = load_miocodec_assets(model_path);
    return std::make_unique<MioCodecLoadedModel>(metadata(*assets), capabilities(*assets), std::move(assets));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_miocodec_loader() {
    return std::make_shared<MioCodecLoader>();
}

}  // namespace engine::models::miocodec
