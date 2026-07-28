#include "engine/community_models/glm_tts/loader.h"

#include "engine/community_models/glm_tts/session.h"
#include "engine/framework/model_spec/package.h"

#include <stdexcept>
#include <utility>

namespace engine::models::glm_tts {
namespace {

runtime::ModelMetadata metadata() {
    runtime::ModelMetadata out;
    out.family = "glm_tts";
    out.variant = "GLM-TTS";
    out.description =
        "Community GLM-TTS zero-shot speech synthesis with native "
        "Llama, Whisper-VQ, Flow/DiT, CAMPPlus, and HiFT execution.";
    return out;
}

runtime::CapabilitySet capabilities() {
    runtime::CapabilitySet out;
    out.supported_tasks = {
        {runtime::VoiceTaskKind::Tts, {runtime::RunMode::Offline}},
        {runtime::VoiceTaskKind::VoiceCloning,
         {runtime::RunMode::Offline}}};
    out.supports_speaker_reference = true;
    out.languages = {"Chinese", "English"};
    return out;
}

runtime::ModelCliInterface cli() {
    runtime::ModelCliInterface out;
    out.request_options = {
        {"reference_text",
         "text",
         "Transcript matching --voice-ref; required by GLM-TTS "
         "zero-shot synthesis."},
        {"max_tokens",
         "n",
         "Maximum generated speech tokens; otherwise the official "
         "2x-to-20x text-token bounds are used."},
        {"temperature",
         "float",
         "Speech-token temperature; official default 1.0."},
        {"top_k", "n", "Speech-token top-k; official default 25."},
        {"top_p",
         "float",
         "Speech-token nucleus threshold; official default 0.8."},
        {"seed", "n", "Speech-token, Flow-noise, and HiFT seed."},
        {"num_inference_steps",
         "n",
         "Flow Euler steps; official default 10."},
        {"flow_guidance_scale",
         "float",
         "Flow classifier-free guidance rate; official default 0.7."},
        {"flow_noise_path",
         "path",
         "Optional raw float32 initial Flow noise for parity tests."},
        {"hift_source_random_path",
         "path",
         "Optional raw float32 HiFT phase-uniform and Gaussian values "
         "for parity tests."},
        {"hift_prior_noise_count",
         "n",
         "Torch RNG value offset used before HiFT source generation."}};
    out.session_options = {
        {"glm_tts.weight_type",
         "native|f32|f16|bf16|q8_0",
         "Requested component weight storage type."},
        {"glm_tts.mem_saver",
         "true|false",
         "Release reference-only encoders after caching the voice while "
         "keeping the generation path warm; default false."},
        {"glm_tts.aggressive_mem_saver",
         "true|false",
         "Also release Llama, Flow, and HiFT after every stage. Minimizes "
         "VRAM but reloads the generation path on every request; default "
         "false."},
        {"glm_tts.reference_cache_slots",
         "n",
         "Prepared reference-audio cache slots; default 1. "
         "Use 0 to disable."},
        {"glm_tts.llama_weight_context_mb",
         "n",
         "Llama weight metadata context in MiB."},
        {"glm_tts.constant_context_mb",
         "n",
         "Llama constant tensor context in MiB."}};
    return out;
}

class GlmTTSLoader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return "glm_tts";
    }

    runtime::CapabilitySet advertised_capabilities() const override {
        return capabilities();
    }

    bool can_load(
        const runtime::ModelLoadRequest & request) const override {
        if (request.family_hint.has_value() &&
            *request.family_hint != family()) {
            return false;
        }
        try {
            (void)model_spec::load_resource_bundle(
                request.model_path,
                model_spec::default_spec_path(family()));
            return true;
        } catch (...) {
            return false;
        }
    }

    runtime::ModelInspection inspect(
        const runtime::ModelLoadRequest & request) const override {
        const auto assets = load_glm_tts_assets(request.model_path);
        runtime::ModelInspection out;
        out.model_root = assets->resources.model_root();
        out.metadata = metadata();
        out.capabilities = capabilities();
        out.cli = cli();
        const auto spec = model_spec::default_spec_path(family());
        out.discovered_configs =
            runtime::discover_named_assets_from_package_spec(
                request.model_path,
                spec,
                model_spec::ResourceKind::Files);
        out.discovered_weights =
            runtime::discover_named_assets_from_package_spec(
                request.model_path,
                spec,
                model_spec::ResourceKind::Tensors);
        return out;
    }

    std::unique_ptr<runtime::ILoadedVoiceModel> load(
        const runtime::ModelLoadRequest & request) const override {
        return load_glm_tts_model(request.model_path);
    }
};

}  // namespace

GlmTTSLoadedModel::GlmTTSLoadedModel(
    runtime::ModelMetadata metadata_in,
    runtime::CapabilitySet capabilities_in,
    std::shared_ptr<const GlmTTSAssets> assets)
    : metadata_(std::move(metadata_in)),
      capabilities_(std::move(capabilities_in)),
      assets_(std::move(assets)) {
    if (assets_ == nullptr) {
        throw std::runtime_error(
            "GLM-TTS loaded model requires assets");
    }
}

const runtime::ModelMetadata &
GlmTTSLoadedModel::metadata() const noexcept {
    return metadata_;
}

const runtime::CapabilitySet &
GlmTTSLoadedModel::capabilities() const noexcept {
    return capabilities_;
}

std::unique_ptr<runtime::IVoiceTaskSession>
GlmTTSLoadedModel::create_task_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options) const {
    return std::make_unique<GlmTTSSession>(
        task, options, assets_);
}

std::unique_ptr<GlmTTSLoadedModel> load_glm_tts_model(
    const std::filesystem::path & model_path) {
    return std::make_unique<GlmTTSLoadedModel>(
        metadata(), capabilities(), load_glm_tts_assets(model_path));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_glm_tts_loader() {
    return std::make_shared<GlmTTSLoader>();
}

}  // namespace engine::models::glm_tts
