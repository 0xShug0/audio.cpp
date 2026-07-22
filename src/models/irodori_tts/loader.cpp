#include "engine/models/irodori_tts/loader.h"

#include "engine/framework/model_spec/package.h"
#include "engine/models/irodori_tts/session.h"

#include <stdexcept>
#include <utility>

namespace engine::models::irodori_tts {
namespace {

runtime::CapabilitySet capabilities(const IrodoriTTSAssets & assets) {
    runtime::CapabilitySet out;
    out.supported_tasks = {
        {runtime::VoiceTaskKind::Tts, {runtime::RunMode::Offline}},
        {runtime::VoiceTaskKind::VoiceCloning, {runtime::RunMode::Offline}},
        {runtime::VoiceTaskKind::VoiceDesign, {runtime::RunMode::Offline}},
    };
    out.supports_style_condition = assets.config.use_caption_condition;
    out.supports_speaker_reference = assets.config.use_speaker_condition;
    out.languages = {"ja"};
    return out;
}

std::string variant(const IrodoriTTSAssets & assets) {
    return assets.config.use_caption_condition ? "600M-v3-VoiceDesign" : "500M-v3";
}

runtime::ModelMetadata metadata(const IrodoriTTSAssets & assets) {
    runtime::ModelMetadata out;
    out.family = "irodori_tts";
    out.variant = variant(assets);
    out.description = "Irodori-TTS V3 loaded from local safetensors and DACVAE assets.";
    return out;
}

runtime::ModelCliInterface cli(const IrodoriTTSAssets &) {
    runtime::ModelCliInterface out;
    out.request_options = {
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
    out.session_options = {
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
    return out;
}

class IrodoriTTSLoader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return "irodori_tts";
    }

    runtime::CapabilitySet advertised_capabilities() const override {
        runtime::CapabilitySet out;
        out.supported_tasks = {
            {runtime::VoiceTaskKind::Tts, {runtime::RunMode::Offline}},
            {runtime::VoiceTaskKind::VoiceCloning, {runtime::RunMode::Offline}},
            {runtime::VoiceTaskKind::VoiceDesign, {runtime::RunMode::Offline}},
        };
        out.supports_speaker_reference = true;
        out.supports_style_condition = true;
        return out;
    }

    std::string advertised_instructions_policy() const override {
        return "caption_option";
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
        const auto assets = load_irodori_tts_assets(request.model_path);
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
        return load_irodori_tts_model(request.model_path);
    }
};

}  // namespace

IrodoriTTSLoadedModel::IrodoriTTSLoadedModel(
    runtime::ModelMetadata metadata,
    runtime::CapabilitySet capabilities,
    std::shared_ptr<const IrodoriTTSAssets> assets)
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
    auto assets = load_irodori_tts_assets(model_path);
    return std::make_unique<IrodoriTTSLoadedModel>(
        metadata(*assets),
        capabilities(*assets),
        std::move(assets));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_irodori_tts_loader() {
    return std::make_shared<IrodoriTTSLoader>();
}

}  // namespace engine::models::irodori_tts
