#include "engine/models/dramabox/loader.h"

#include "engine/framework/model_spec/package.h"
#include "engine/models/dramabox/session.h"

#include <stdexcept>
#include <utility>

namespace engine::models::dramabox {
namespace {

runtime::ModelMetadata metadata() {
    runtime::ModelMetadata out;
    out.family = "dramabox";
    out.variant = "dramabox-tts";
    out.description = "DramaBox TTS loaded from a native GGUF package.";
    out.config_candidates = {
        "config.json",
        "audio_components_config.json",
        "gemma-3-12b-it-bnb-4bit/config.json",
        "gemma-3-12b-it-bnb-4bit/tokenizer_config.json",
    };
    out.weight_candidates = {"model.gguf"};
    return out;
}

runtime::CapabilitySet capabilities() {
    runtime::CapabilitySet out;
    out.supported_tasks = {
        {runtime::VoiceTaskKind::Tts, {runtime::RunMode::Offline}},
        {runtime::VoiceTaskKind::AudioGeneration, {runtime::RunMode::Offline}},
    };
    out.languages = {"en"};
    out.supports_speaker_reference = true;
    out.supports_style_condition = true;
    return out;
}

runtime::ModelCliInterface cli() {
    runtime::ModelCliInterface out;
    out.request_options = {
        {"target_voice", "wav", "Reference voice audio path."},
        {"duration_sec", "seconds", "Explicit target duration; zero uses prompt duration estimate."},
        {"num_inference_steps", "n", "LTX Euler diffusion steps."},
        {"guidance_scale", "float", "Classifier-free guidance scale."},
        {"spatio_temporal_guidance_scale", "float", "Spatio-temporal guidance scale."},
        {"duration_scale", "float", "Prompt duration estimate scale."},
        {"reference_duration_sec", "seconds", "Reference voice crop/repeat duration."},
        {"guidance_rescale", "float|auto", "Guidance rescale; omit for official auto mode."},
        {"audio_chunk_threshold_sec", "seconds", "Long-form chunk route threshold."},
        {"audio_chunk_duration_sec", "seconds", "Long-form target chunk duration."},
        {"cross_fade_duration_sec", "seconds", "Long-form equal-power cross-fade duration."},
        {"seed", "n", "Torch RNG seed."},
    };
    out.session_options = {
        {"dramabox.perf_mode", "off|flash_attention", "Default off keeps the exact reference-query attention path."},
    };
    return out;
}

class DramaBoxLoader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return "dramabox";
    }

    runtime::CapabilitySet advertised_capabilities() const override {
        return capabilities();
    }

    bool can_load(const runtime::ModelLoadRequest & request) const override {
        if (request.family_hint.has_value() && *request.family_hint != family()) {
            return false;
        }
        try {
            const auto package_spec = engine::model_spec::default_spec_path(family());
            (void) engine::model_spec::load_resource_bundle(request.model_path, package_spec);
            return true;
        } catch (...) {
            return false;
        }
    }

    runtime::ModelInspection inspect(const runtime::ModelLoadRequest & request) const override {
        const auto assets = load_dramabox_assets(request.model_path);
        runtime::ModelInspection inspection;
        inspection.model_root = assets->resources.model_root();
        inspection.metadata = metadata();
        inspection.capabilities = capabilities();
        inspection.cli = cli();
        const auto package_spec = engine::model_spec::default_spec_path(family());
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
        return load_dramabox_model(request.model_path);
    }
};

}  // namespace

DramaBoxLoadedModel::DramaBoxLoadedModel(
    runtime::ModelMetadata metadata,
    runtime::CapabilitySet capabilities,
    std::shared_ptr<const DramaBoxAssets> assets)
    : metadata_(std::move(metadata)),
      capabilities_(std::move(capabilities)),
      assets_(std::move(assets)) {}

const runtime::ModelMetadata & DramaBoxLoadedModel::metadata() const noexcept {
    return metadata_;
}

const runtime::CapabilitySet & DramaBoxLoadedModel::capabilities() const noexcept {
    return capabilities_;
}

std::unique_ptr<runtime::IVoiceTaskSession> DramaBoxLoadedModel::create_task_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options) const {
    return std::make_unique<DramaBoxSession>(task, options, assets_);
}

std::unique_ptr<DramaBoxLoadedModel> load_dramabox_model(const std::filesystem::path & model_path) {
    auto assets = load_dramabox_assets(model_path);
    return std::make_unique<DramaBoxLoadedModel>(
        metadata(),
        capabilities(),
        std::move(assets));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_dramabox_loader() {
    return std::make_shared<DramaBoxLoader>();
}

}  // namespace engine::models::dramabox
