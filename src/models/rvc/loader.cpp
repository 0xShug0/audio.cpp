#include "engine/models/rvc/loader.h"

#include "engine/framework/model_spec/package.h"
#include "engine/models/rvc/assets.h"
#include "engine/models/rvc/session.h"

#include <memory>
#include <stdexcept>
#include <utility>

namespace engine::models::rvc {
namespace {

runtime::CapabilitySet capabilities() {
    runtime::CapabilitySet out;
    out.supported_tasks = {
        {runtime::VoiceTaskKind::VoiceConversion, {runtime::RunMode::Offline}},
    };
    return out;
}

runtime::ModelMetadata metadata() {
    runtime::ModelMetadata out;
    out.family = "rvc";
    out.variant = "v1-v2-rmvpe-native";
    out.description = "RVC v1/v2 offline voice conversion with native RMVPE and user-supplied voice/index paths.";
    return out;
}

runtime::ModelCliInterface cli() {
    runtime::ModelCliInterface out;
    out.request_options = {
        {"rvc.voice", "default|manthos|chocola|fraise", "RVC voice model id. Defaults to default."},
        {"rvc.voice_model", "path", "Optional user RVC .pth/.pt voice checkpoint path."},
        {"rvc.f0_method", "rmvpe", "Native pitch extractor. Defaults to rmvpe."},
        {"rvc.f0_file", "path", "Optional CSV F0 override file with time,Hz rows."},
        {"rvc.file_index", "path", "Optional user FAISS .index retrieval path for index_rate."},
        {"rvc.index_rate", "0..1", "IVF retrieval blend rate. Defaults to 0."},
        {"rvc.f0_up_key", "n", "Semitone pitch shift. Defaults to 0."},
        {"rvc.filter_radius", "n", "Median filter radius for F0. Defaults to 3."},
        {"rvc.resample_sr", "hz", "Output resample rate; 0 keeps model sample rate."},
        {"rvc.rms_mix_rate", "float", "RMS mix rate. Defaults to 0.25."},
        {"rvc.protect", "float", "Unvoiced consonant protection. Defaults to 0.33."},
        {"rvc.speaker_id", "n", "Speaker embedding id for multi-speaker RVC checkpoints. Defaults to 0."},
        {"rvc.x_pad", "seconds", "Long-audio chunk pad seconds. Defaults to 1."},
        {"rvc.x_query", "seconds", "Long-audio quiet-point query window seconds. Defaults to 5."},
        {"rvc.x_center", "seconds", "Long-audio split center stride seconds. Defaults to 30."},
        {"rvc.x_max", "seconds", "Input seconds before quiet-point splitting. Defaults to 32."},
    };
    out.session_options = {
        {"rvc.weight_type", "f32|native|f16|bf16|q8_0", "Tensor storage type for native RVC, HuBERT, and RMVPE weights. Defaults to f32."},
        {"rvc.voice_model_cache_slots", "n", "User voice model cache slots; default 4, set 0 to disable."},
    };
    return out;
}

class RvcLoadedModel final : public runtime::ILoadedVoiceModel {
public:
    RvcLoadedModel(
        runtime::ModelMetadata metadata,
        runtime::CapabilitySet capabilities,
        std::shared_ptr<const RvcAssets> assets)
        : metadata_(std::move(metadata)),
          capabilities_(std::move(capabilities)),
          assets_(std::move(assets)) {
        if (assets_ == nullptr) {
            throw std::runtime_error("RVC loaded model requires assets");
        }
    }

    const runtime::ModelMetadata & metadata() const noexcept override {
        return metadata_;
    }

    const runtime::CapabilitySet & capabilities() const noexcept override {
        return capabilities_;
    }

    std::unique_ptr<runtime::IVoiceTaskSession> create_task_session(
        const runtime::TaskSpec & task,
        const runtime::SessionOptions & options) const override {
        if (task.mode != runtime::RunMode::Offline) {
            throw std::runtime_error("RVC supports offline sessions");
        }
        if (task.task != runtime::VoiceTaskKind::VoiceConversion) {
            throw std::runtime_error("RVC supports VoiceConversion tasks");
        }
        return std::make_unique<RvcSession>(task, options, assets_);
    }

private:
    runtime::ModelMetadata metadata_;
    runtime::CapabilitySet capabilities_;
    std::shared_ptr<const RvcAssets> assets_;
};

class RvcLoader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return "rvc";
    }

    runtime::CapabilitySet advertised_capabilities() const override {
        return capabilities();
    }

    bool can_load(const runtime::ModelLoadRequest & request) const override {
        if (request.family_hint.has_value() && *request.family_hint != family()) {
            return false;
        }
        try {
            (void)engine::model_spec::load_resource_bundle(
                request.model_path,
                engine::model_spec::default_spec_path(family()));
            return true;
        } catch (...) {
            return false;
        }
    }

    runtime::ModelInspection inspect(const runtime::ModelLoadRequest & request) const override {
        const auto assets = load_rvc_assets(request.model_path);
        runtime::ModelInspection inspection;
        inspection.model_root = assets->resources.model_root();
        inspection.metadata = metadata();
        inspection.capabilities = capabilities();
        inspection.cli = cli();
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
        auto assets = load_rvc_assets(request.model_path);
        return std::make_unique<RvcLoadedModel>(metadata(), capabilities(), std::move(assets));
    }
};

}  // namespace

std::shared_ptr<runtime::IVoiceModelLoader> make_rvc_loader() {
    return std::make_shared<RvcLoader>();
}

}  // namespace engine::models::rvc
