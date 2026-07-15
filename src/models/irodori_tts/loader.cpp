#include "engine/models/irodori_tts/loader.h"

#include "engine/framework/assets/model_package.h"
#include "engine/models/irodori_tts/session.h"

#include <stdexcept>
#include <utility>

namespace engine::models::irodori_tts {
namespace {

runtime::CapabilitySet capabilities_for_assets(const IrodoriAssets & assets) {
    runtime::CapabilitySet capabilities;
    capabilities.supported_tasks = {
        {runtime::VoiceTaskKind::Tts, {runtime::RunMode::Offline}},
        {runtime::VoiceTaskKind::VoiceCloning, {runtime::RunMode::Offline}},
        {runtime::VoiceTaskKind::VoiceDesign, {runtime::RunMode::Offline}},
    };
    capabilities.supports_style_condition = assets.config.use_caption_condition;
    capabilities.supports_speaker_reference = assets.config.use_speaker_condition;
    capabilities.languages = {"ja"};
    return capabilities;
}

std::string variant_for_assets(const IrodoriAssets & assets) {
    return assets.config.use_caption_condition ? "600M-v3-VoiceDesign" : "500M-v3";
}

class IrodoriTTSLoader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return "irodori_tts";
    }

    bool can_load(const runtime::ModelLoadRequest & request) const override {
        try {
            (void) engine::assets::load_resource_bundle_from_package_spec(
                request.model_path,
                engine::assets::default_model_package_spec_path(family()));
            return !request.family_hint.has_value() || *request.family_hint == family();
        } catch (...) {
            return false;
        }
    }

    runtime::ModelInspection inspect(const runtime::ModelLoadRequest & request) const override {
        const auto assets = load_irodori_assets(request.model_path);
        runtime::ModelInspection inspection;
        inspection.model_root = assets->resources.model_root();
        inspection.metadata.family = family();
        inspection.metadata.variant = variant_for_assets(*assets);
        inspection.metadata.description = "Irodori-TTS V3 loaded from local safetensors and DACVAE assets.";
        inspection.capabilities = capabilities_for_assets(*assets);
        inspection.cli.request_options = {
            {"num_inference_steps", "n", "RF diffusion steps."},
            {"duration_seconds", "seconds", "Explicit output duration."},
            {"duration_scale", "float", "Predicted-duration multiplier."},
            {"text_chunk_mode", "japanese|endline", "Text chunking mode; default endline."},
            {"min_seconds", "seconds", "Minimum generated duration."},
            {"max_seconds", "seconds", "Maximum generated duration."},
            {"text_guidance_scale", "float", "Text classifier-free guidance scale."},
            {"speaker_guidance_scale", "float", "Speaker classifier-free guidance scale."},
            {"caption_guidance_scale", "float", "Caption classifier-free guidance scale."},
            {"guidance_mode", "independent", "Classifier-free guidance combination mode."},
            {"guidance_min_t", "float", "Minimum diffusion time for guidance."},
            {"guidance_max_t", "float", "Maximum diffusion time for guidance."},
            {"seed", "n", "Torch RNG seed."},
            {"trim_tail", "bool", "Trim trailing silence-like samples."},
        };
        inspection.cli.session_options = {
            {"irodori_tts.weight_type", "native|f32|f16|bf16|q8_0", "Model weight storage type."},
            {"irodori_tts.codec_weight_type", "native|f32|f16|q8_0", "DACVAE codec weight storage type."},
            {"irodori_tts.condition_graph_arena_mb", "n", "Condition encoder graph arena size."},
            {"irodori_tts.rf_graph_arena_mb", "n", "RF sampler graph arena size."},
            {"irodori_tts.codec_graph_arena_mb", "n", "DACVAE codec graph arena size."},
            {"irodori_tts.condition_weight_context_mb", "n", "Condition encoder weight context size."},
            {"irodori_tts.rf_weight_context_mb", "n", "RF sampler weight context size."},
            {"irodori_tts.codec_weight_context_mb", "n", "DACVAE codec weight context size."},
            {"irodori_tts.mem_saver", "true|false", "Release staged runtime graphs after request phases; default true."},
            {"irodori_tts.reference_cache_slots", "n", "Prepared reference-speaker cache slots; default 1."},
        };
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
        return load_irodori_tts_model(request.model_path);
    }
};

}  // namespace

IrodoriTTSLoadedModel::IrodoriTTSLoadedModel(
    runtime::ModelMetadata metadata,
    runtime::CapabilitySet capabilities,
    std::shared_ptr<const IrodoriAssets> assets)
    : metadata_(std::move(metadata)),
      capabilities_(std::move(capabilities)),
      assets_(std::move(assets)) {}

const runtime::ModelMetadata & IrodoriTTSLoadedModel::metadata() const noexcept {
    return metadata_;
}

const runtime::CapabilitySet & IrodoriTTSLoadedModel::capabilities() const noexcept {
    return capabilities_;
}

std::unique_ptr<runtime::IVoiceTaskSession> IrodoriTTSLoadedModel::create_task_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options) const {
    if (task.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("Irodori-TTS only supports offline sessions");
    }
    if (task.task != runtime::VoiceTaskKind::Tts &&
        task.task != runtime::VoiceTaskKind::VoiceCloning &&
        task.task != runtime::VoiceTaskKind::VoiceDesign) {
        throw std::runtime_error("Irodori-TTS supports only TTS, voice-cloning, and voice-design offline tasks");
    }
    return std::make_unique<IrodoriTTSSession>(task, options, assets_);
}

std::unique_ptr<IrodoriTTSLoadedModel> load_irodori_tts_model(const std::filesystem::path & model_path) {
    auto assets = load_irodori_assets(model_path);

    runtime::ModelMetadata metadata;
    metadata.family = "irodori_tts";
    metadata.variant = variant_for_assets(*assets);
    metadata.description = "Irodori-TTS V3 loaded from local safetensors and DACVAE assets.";

    return std::make_unique<IrodoriTTSLoadedModel>(
        std::move(metadata),
        capabilities_for_assets(*assets),
        std::move(assets));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_irodori_tts_loader() {
    return std::make_shared<IrodoriTTSLoader>();
}

}  // namespace engine::models::irodori_tts
