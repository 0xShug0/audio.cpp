#include "engine/models/seed_vc/loader.h"

#include "engine/framework/model_spec/package.h"
#include "engine/models/seed_vc/assets.h"
#include "engine/models/seed_vc/session.h"

#include <memory>
#include <stdexcept>
#include <utility>

namespace engine::models::seed_vc {
namespace {

runtime::CapabilitySet capabilities(const SeedVcAssets &) {
    runtime::CapabilitySet out;
    out.supported_tasks = {
        {runtime::VoiceTaskKind::VoiceConversion, {runtime::RunMode::Offline}},
        {runtime::VoiceTaskKind::Svc, {runtime::RunMode::Offline}},
    };
    out.supports_speaker_reference = true;
    out.supports_style_condition = true;
    return out;
}

runtime::ModelMetadata metadata(const SeedVcAssets &) {
    runtime::ModelMetadata out;
    out.family = "seed_vc";
    out.variant = "v2-vc-v1-svc";
    out.description = "SeedVC-MLX asset bundle for V2 voice conversion and V1 voice conversion/SVC variants.";
    return out;
}

runtime::ModelCliInterface cli(const SeedVcAssets &) {
    runtime::ModelCliInterface out;
    out.request_options = {
        {"route", "v2_vc|v1_svc|v1_whisper_bigvgan_vc|v1_xlsr_hift_vc",
         "Seed-VC route. Defaults to v2_vc for VC and v1_svc for SVC."},
        {"length_adjust", "float", "Duration multiplier; default 1.0."},
        {"num_inference_steps", "n", "Diffusion steps; default 30."},
        {"inference_cfg_rate", "float", "V1 CFG rate; default 0.7."},
        {"intelligibility_cfg_rate", "float", "V2 intelligibility CFG rate; default 0.7."},
        {"similarity_cfg_rate", "float", "V2 similarity CFG rate; default 0.7."},
        {"f0_condition", "true|false", "Enable V1 F0 conditioning."},
        {"auto_f0_adjust", "true|false", "Enable automatic V1 F0 adjustment."},
        {"semi_tone_shift", "n", "V1 semitone shift; default 0."},
    };
    out.session_options = {
        {"seed_vc.weight_type", "native|f32|f16|bf16|q8_0", "Seed-VC weight storage type."},
    };
    return out;
}

class SeedVcLoadedModel final : public runtime::ILoadedVoiceModel {
public:
    SeedVcLoadedModel(
        runtime::ModelMetadata metadata,
        runtime::CapabilitySet capabilities,
        std::shared_ptr<const SeedVcAssets> assets)
        : metadata_(std::move(metadata)),
          capabilities_(std::move(capabilities)),
          assets_(std::move(assets)) {
        if (assets_ == nullptr) {
            throw std::runtime_error("Seed-VC loaded model requires assets");
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
            throw std::runtime_error("Seed-VC currently supports offline sessions");
        }
        if (task.task != runtime::VoiceTaskKind::VoiceConversion && task.task != runtime::VoiceTaskKind::Svc) {
            throw std::runtime_error("Seed-VC supports VoiceConversion and Svc tasks");
        }
        return std::make_unique<SeedVcSession>(task, options, assets_);
    }

private:
    runtime::ModelMetadata metadata_;
    runtime::CapabilitySet capabilities_;
    std::shared_ptr<const SeedVcAssets> assets_;
};

class SeedVcLoader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return "seed_vc";
    }

    runtime::CapabilitySet advertised_capabilities() const override {
        runtime::CapabilitySet out;
        out.supported_tasks = {
            {runtime::VoiceTaskKind::VoiceConversion, {runtime::RunMode::Offline}},
            {runtime::VoiceTaskKind::Svc, {runtime::RunMode::Offline}},
        };
        out.supports_speaker_reference = true;
        out.supports_style_condition = true;
        return out;
    }

    bool can_load(const runtime::ModelLoadRequest & request) const override {
        if (request.family_hint.has_value() && *request.family_hint != family()) {
            return false;
        }
        try {
            (void) engine::model_spec::load_resource_bundle(
                request.model_path,
                engine::model_spec::default_spec_path(family()));
            return true;
        } catch (...) {
            return false;
        }
    }

    runtime::ModelInspection inspect(const runtime::ModelLoadRequest & request) const override {
        const auto assets = load_seed_vc_assets(request.model_path);
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
        auto assets = load_seed_vc_assets(request.model_path);
        return std::make_unique<SeedVcLoadedModel>(
            metadata(*assets),
            capabilities(*assets),
            std::move(assets));
    }
};

}  // namespace

std::shared_ptr<runtime::IVoiceModelLoader> make_seed_vc_loader() {
    return std::make_shared<SeedVcLoader>();
}

}  // namespace engine::models::seed_vc
