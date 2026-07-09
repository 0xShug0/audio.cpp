#include "engine/models/stable_audio/loader.h"

#include "engine/framework/io/filesystem.h"
#include "engine/models/stable_audio/foundation/session.h"
#include "engine/models/stable_audio/session.h"

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::stable_audio {
namespace {

std::filesystem::path resolve_model_root(const std::filesystem::path & model_path) {
    if (engine::io::is_existing_directory(model_path)) {
        return std::filesystem::weakly_canonical(model_path);
    }
    if (engine::io::is_existing_file(model_path)) {
        return std::filesystem::weakly_canonical(model_path.parent_path());
    }
    throw std::runtime_error("Stable Audio model path does not exist: " + model_path.string());
}

bool has_stable_audio_assets(const std::filesystem::path & root) {
    (void)resolve_stable_audio_assets(root);
    return true;
}

std::string relative_candidate(const std::filesystem::path & root, const std::filesystem::path & path) {
    const auto relative = path.lexically_relative(root);
    if (relative.empty()) {
        return path.filename().generic_string();
    }
    return relative.generic_string();
}

std::vector<std::string> config_candidates_for_paths(const StableAudioAssetPaths & paths) {
    return {
        relative_candidate(paths.model_root, paths.model_config_path),
        relative_candidate(paths.model_root, paths.t5_config_path),
        relative_candidate(paths.model_root, paths.t5_tokenizer_config_path),
    };
}

std::vector<std::string> weight_candidates_for_paths(const StableAudioAssetPaths & paths) {
    return {
        relative_candidate(paths.model_root, paths.model_safetensors_path),
        relative_candidate(paths.model_root, paths.t5_safetensors_path),
    };
}

std::vector<runtime::NamedAsset> discover_config_assets(const runtime::ModelLoadRequest & request) {
    const auto paths = resolve_stable_audio_assets(resolve_model_root(request.model_path));
    return runtime::discover_named_assets(paths.model_root, config_candidates_for_paths(paths));
}

std::vector<runtime::NamedAsset> discover_weight_assets(const runtime::ModelLoadRequest & request) {
    const auto paths = resolve_stable_audio_assets(resolve_model_root(request.model_path));
    return runtime::discover_named_assets(paths.model_root, weight_candidates_for_paths(paths));
}

runtime::CapabilitySet capabilities_for_assets(const StableAudioAssets &) {
    runtime::CapabilitySet capabilities;
    capabilities.supported_tasks = {
        {runtime::VoiceTaskKind::AudioGeneration, {runtime::RunMode::Offline}},
    };
    capabilities.languages = {"en"};
    return capabilities;
}

std::string variant_for_assets(const StableAudioAssets & assets) {
    const auto name = assets.paths.model_root.filename().string();
    if (assets.config.stable_audio_open_v1) {
        return name.empty() ? "foundation-1" : name;
    }
    if (name.rfind("stable-audio-3-", 0) == 0) {
        return name.substr(std::string("stable-audio-3-").size());
    }
    return assets.config.is_medium() ? "medium" : "small";
}

class StableAudioLoader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return "stable_audio";
    }

    bool can_load(const runtime::ModelLoadRequest & request) const override {
        try {
            const auto root = resolve_model_root(request.model_path);
            return has_stable_audio_assets(root) &&
                   (!request.family_hint.has_value() || *request.family_hint == family());
        } catch (...) {
            if (request.family_hint.has_value() && *request.family_hint == family()) {
                throw;
            }
            return false;
        }
    }

    runtime::ModelInspection inspect(const runtime::ModelLoadRequest & request) const override {
        const auto assets = load_stable_audio_assets(resolve_model_root(request.model_path));
        runtime::ModelInspection inspection;
        inspection.model_root = assets->paths.model_root;
        inspection.metadata.family = family();
        inspection.metadata.variant = variant_for_assets(*assets);
        inspection.metadata.description = "Stable Audio 3 loaded from local safetensors and config-declared prompt encoder assets.";
        inspection.metadata.config_candidates = config_candidates_for_paths(assets->paths);
        inspection.metadata.weight_candidates = weight_candidates_for_paths(assets->paths);
        inspection.capabilities = capabilities_for_assets(*assets);
        inspection.cli.request_options = {
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
        inspection.cli.session_options = {
            {"stable_audio.max_batch", "n", "Maximum prompt batch size; default 1."},
            {"stable_audio.weight_type", "native|f32|f16|bf16|q8_0", "Stable Audio weight storage type."},
            {"stable_audio.mem_saver", "true|false", "Release staged runtime graphs after each request; default false."},
        };
        inspection.discovered_configs = discover_config_assets(request);
        inspection.discovered_weights = discover_weight_assets(request);
        return inspection;
    }

    std::unique_ptr<runtime::ILoadedVoiceModel> load(const runtime::ModelLoadRequest & request) const override {
        return load_stable_audio_model(resolve_model_root(request.model_path));
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

    runtime::ModelMetadata metadata;
    metadata.family = "stable_audio";
    metadata.variant = variant_for_assets(*assets);
    metadata.description = assets->config.stable_audio_open_v1
        ? "Stable Audio Foundation loaded from local safetensors and config-declared T5Base prompt encoder assets."
        : "Stable Audio 3 loaded from local safetensors and config-declared prompt encoder assets.";
    metadata.config_candidates = config_candidates_for_paths(assets->paths);
    metadata.weight_candidates = weight_candidates_for_paths(assets->paths);

    return std::make_unique<StableAudioLoadedModel>(
        std::move(metadata),
        capabilities_for_assets(*assets),
        std::move(assets));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_stable_audio_loader() {
    return std::make_shared<StableAudioLoader>();
}

}  // namespace engine::models::stable_audio
