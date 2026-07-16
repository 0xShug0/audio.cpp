#include "engine/models/stable_audio/loader.h"

#include "engine/framework/assets/model_package.h"
#include "engine/models/stable_audio/foundation/session.h"
#include "engine/models/stable_audio/session.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::stable_audio {
namespace {

std::string variant(const StableAudioAssets & assets) {
    const auto name = assets.resources.model_root().filename().string();
    if (assets.config.stable_audio_open_v1) {
        return name.empty() ? "foundation-1" : name;
    }
    if (name.rfind("stable-audio-3-", 0) == 0) {
        return name.substr(std::string("stable-audio-3-").size());
    }
    return assets.config.medium_architecture ? "medium" : "small";
}

runtime::ModelMetadata metadata(const StableAudioAssets & assets) {
    runtime::ModelMetadata out;
    out.family = "stable_audio";
    out.variant = variant(assets);
    out.description = assets.config.stable_audio_open_v1
        ? "Stable Audio Foundation loaded from local safetensors and config-declared T5Base prompt encoder assets."
        : "Stable Audio 3 loaded from local safetensors and config-declared prompt encoder assets.";
    return out;
}

runtime::CapabilitySet capabilities(const StableAudioAssets &) {
    runtime::CapabilitySet capabilities;
    capabilities.supported_tasks = {
        {runtime::VoiceTaskKind::AudioGeneration, {runtime::RunMode::Offline}},
    };
    capabilities.languages = {"en"};
    return capabilities;
}

runtime::ModelCliInterface cli(const StableAudioAssets &) {
    runtime::ModelCliInterface out;
    out.request_options = {
        {"duration_seconds", "seconds[,seconds...]", "Target duration per prompt."},
        {"num_inference_steps", "n", "RF diffusion steps."},
        {"guidance_scale", "float", "Classifier-free guidance scale."},
        {"sampler", "pingpong|euler|dpmpp-2m|dpmpp-3m-sde", "Diffusion sampler."},
        {"apg_scale", "float", "Adaptive projected guidance scale."},
        {"negative_prompt", "text", "Negative prompt text."},
        {"batch_size", "n", "Prompt batch size."},
        {"duration_padding_seconds", "seconds", "Extra generated padding before truncation."},
        {"truncate_output_to_duration", "bool", "Trim decoded audio to requested duration."},
        {"chunked_decode", "bool", "Decode the vocoder in chunks."},
        {"init_noise_level", "float", "Strength for audio-conditioned generation."},
        {"audio_input_kind", "init_audio|inpaint_audio", "How to use request audio input."},
        {"inpaint_mask_start_seconds", "seconds[,seconds...]", "Inpaint region start times."},
        {"inpaint_mask_end_seconds", "seconds[,seconds...]", "Inpaint region end times."},
        {"seed", "n", "Torch RNG seed."},
    };
    out.session_options = {
        {"stable_audio.max_batch", "n", "Maximum prompt batch size; default 1."},
        {"stable_audio.weight_type", "native|f32|f16|bf16|q8_0", "Stable Audio weight storage type."},
        {"stable_audio.mem_saver", "true|false", "Release staged runtime graphs after each request; default false."},
    };
    return out;
}

class StableAudioLoader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return "stable_audio";
    }

    bool can_load(const runtime::ModelLoadRequest & request) const override {
        try {
            const auto package_spec = engine::assets::default_model_package_spec_path(family());
            (void) engine::assets::load_resource_bundle_from_package_spec(request.model_path, package_spec);
            return !request.family_hint.has_value() || *request.family_hint == family();
        } catch (...) {
            if (request.family_hint.has_value() && *request.family_hint == family()) {
                throw;
            }
            return false;
        }
    }

    runtime::ModelInspection inspect(const runtime::ModelLoadRequest & request) const override {
        const auto assets = load_stable_audio_assets(request.model_path);
        runtime::ModelInspection inspection;
        inspection.model_root = assets->resources.model_root();
        inspection.metadata = metadata(*assets);
        inspection.capabilities = capabilities(*assets);
        inspection.cli = cli(*assets);
        const auto package_spec = engine::assets::default_model_package_spec_path(family());
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

    std::unique_ptr<runtime::ILoadedVoiceModel> load(const runtime::ModelLoadRequest & request) const override {
        return load_stable_audio_model(request.model_path);
    }
};

}  // namespace

StableAudioLoadedModel::StableAudioLoadedModel(
    runtime::ModelMetadata metadata,
    runtime::CapabilitySet capabilities,
    std::shared_ptr<const StableAudioAssets> assets)
    : metadata_(std::move(metadata)),
      capabilities_(std::move(capabilities)),
      assets_(std::move(assets)) {}

const runtime::ModelMetadata & StableAudioLoadedModel::metadata() const noexcept {
    return metadata_;
}

const runtime::CapabilitySet & StableAudioLoadedModel::capabilities() const noexcept {
    return capabilities_;
}

std::unique_ptr<runtime::IVoiceTaskSession> StableAudioLoadedModel::create_task_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options) const {
    if (task.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("Stable Audio only supports offline sessions");
    }
    if (task.task != runtime::VoiceTaskKind::AudioGeneration) {
        throw std::runtime_error("Stable Audio supports only the gen task");
    }
    if (assets_->config.stable_audio_open_v1) {
        return std::make_unique<foundation::FoundationSession>(task, options, assets_);
    }
    return std::make_unique<StableAudioSession>(task, options, assets_);
}

std::unique_ptr<StableAudioLoadedModel> load_stable_audio_model(const std::filesystem::path & model_path) {
    auto assets = load_stable_audio_assets(model_path);

    return std::make_unique<StableAudioLoadedModel>(
        metadata(*assets),
        capabilities(*assets),
        std::move(assets));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_stable_audio_loader() {
    return std::make_shared<StableAudioLoader>();
}

}  // namespace engine::models::stable_audio
