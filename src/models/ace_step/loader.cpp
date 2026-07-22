#include "engine/models/ace_step/loader.h"

#include "engine/framework/model_spec/package.h"
#include "engine/models/ace_step/session.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::ace_step {
namespace {

AceStepModelSelection selection_from_request(const runtime::ModelLoadRequest & request) {
    AceStepModelSelection selection;
    if (const auto it = request.options.find("ace_step.dit_model_path"); it != request.options.end()) {
        selection.dit_model_path = it->second;
        return selection;
    }
    std::string model_name = request.model_path.filename().generic_string();
    std::transform(model_name.begin(), model_name.end(), model_name.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    if (model_name.find("base") != std::string::npos) {
        selection.dit_model_path = "acestep-v15-base";
    } else if (model_name.find("turbo") != std::string::npos) {
        selection.dit_model_path = "acestep-v15-turbo";
    }
    return selection;
}

runtime::ModelMetadata metadata(const AceStepAssets & assets) {
    runtime::ModelMetadata out;
    out.family = "ace_step";
    out.variant = assets.selection.dit_model_path;
    out.description = "ACE-Step loaded from local extracted assets.";
    return out;
}

runtime::CapabilitySet capabilities(const AceStepAssets &) {
    runtime::CapabilitySet out;
    out.supported_tasks = {
        {runtime::VoiceTaskKind::AudioGeneration, {runtime::RunMode::Offline}},
    };
    return out;
}

runtime::ModelCliInterface cli(const AceStepAssets &) {
    runtime::ModelCliInterface out;
    out.request_options = {
        {"route", "text2music|complete|lego|extract|cover|cover-nofsq|repaint", "ACE-Step operation; also exposed as --task-route."},
        {"lyrics", "text", "Vocal lyrics."},
        {"duration_seconds", "seconds", "Target duration; also exposed as --duration-seconds."},
        {"language", "code", "Vocal language for lyrics; also exposed as --language."},
        {"track_name", "text", "Track name used by lego and extract routes; also exposed as --track-name."},
        {"complete_track_classes", "a,b", "Track classes for complete route."},
        {"repainting_start", "seconds", "Start time for repaint route; also exposed as --repaint-start."},
        {"repainting_end", "seconds", "End time for repaint route; also exposed as --repaint-end."},
        {"repaint_mode", "balanced|conservative|aggressive", "Preset repaint blending policy; also exposed as --repaint-mode."},
        {"repaint_strength", "0..1", "Repaint strength; also exposed as --repaint-strength."},
        {"num_inference_steps", "n", "Diffusion denoising steps; also exposed as --num-inference-steps."},
        {"guidance_scale", "float", "Diffusion guidance scale; also exposed as --guidance-scale."},
        {"seed", "n", "Generation seed; also exposed as --seed."},
        {"bpm", "n", "Force BPM metadata."},
        {"keyscale", "text", "Force key metadata."},
        {"timesignature", "text", "Force time signature metadata."},
        {"negative_prompt", "text", "Negative prompt."},
        {"audio_codes", "text", "Use supplied ACE semantic code text instead of planner generation."},
        {"audio_cover_strength", "float", "Cover strength for cover/edit-style conditioning."},
        {"cover_noise_strength", "float", "Noise strength for cover conditioning."},
        {"lm_temperature", "float", "Planner sampling temperature."},
        {"lm_cfg_scale", "float", "Planner CFG scale."},
        {"lm_top_k", "n", "Planner top-k; 0 disables top-k."},
        {"lm_top_p", "float", "Planner top-p."},
        {"lm_repetition_penalty", "float", "Planner repetition penalty."},
        {"sampler_mode", "euler|heun", "Diffusion sampler mode."},
        {"retake_seed", "n", "Optional retake noise seed; -1 clears it."},
        {"retake_variance", "float", "Retake noise mixing strength."},
        {"flow_edit_morph", "bool", "Status: parsed for text2music, but not usable because the flow-edit diffusion overlay is not implemented."},
        {"dcw_enabled", "bool", "Status: experimental dynamic-cfg wavelet path; keep disabled unless validating that path."},
    };
    out.session_options = {
        {"ace_step.weight_type", "native|f32|f16|bf16|q8_0", "Shared ACE-Step weight storage type."},
        {"ace_step.dit_weight_type", "native|f32|f16|bf16|q8_0", "DiT weight storage type; f16 is promoted to bf16."},
        {"ace_step.planner_weight_type", "native|f32|f16|bf16|q8_0", "Planner LM weight storage type."},
        {"ace_step.text_encoder_weight_type", "native|f32|f16|bf16|q8_0", "Text encoder weight storage type."},
        {"ace_step.vae_weight_type", "native|f32|f16|bf16|q8_0", "VAE weight storage type."},
        {"ace_step.mem_saver", "true|false", "Release staged runtime graphs after each request; default false."},
    };
    out.load_options = {
        {"ace_step.dit_model_path", "acestep-v15-turbo|acestep-v15-base", "DiT variant inside the model package."},
    };
    return out;
}

class AceStepLoader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return "ace_step";
    }

    runtime::CapabilitySet advertised_capabilities() const override {
        runtime::CapabilitySet out;
        out.supported_tasks = {
            {runtime::VoiceTaskKind::AudioGeneration, {runtime::RunMode::Offline}},
        };
        return out;
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
        const auto assets = load_ace_step_assets(request.model_path, selection_from_request(request));
        runtime::ModelInspection inspection;
        inspection.model_root = assets->resources.model_root();
        inspection.metadata = metadata(*assets);
        inspection.capabilities = capabilities(*assets);
        inspection.cli = cli(*assets);
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
        return load_ace_step_model(request.model_path, selection_from_request(request));
    }
};

}  // namespace

AceStepLoadedModel::AceStepLoadedModel(
    runtime::ModelMetadata metadata,
    runtime::CapabilitySet capabilities,
    std::shared_ptr<const AceStepAssets> assets)
    : metadata_(std::move(metadata)),
      capabilities_(std::move(capabilities)),
      assets_(std::move(assets)) {}

const runtime::ModelMetadata & AceStepLoadedModel::metadata() const noexcept {
    return metadata_;
}

const runtime::CapabilitySet & AceStepLoadedModel::capabilities() const noexcept {
    return capabilities_;
}

std::unique_ptr<runtime::IVoiceTaskSession> AceStepLoadedModel::create_task_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options) const {
    if (task.task != runtime::VoiceTaskKind::AudioGeneration) {
        throw std::runtime_error("ACE-Step supports only the gen task");
    }
    if (task.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("ACE-Step supports only offline mode");
    }
    return std::make_unique<AceStepSession>(task, options, assets_);
}

std::unique_ptr<AceStepLoadedModel> load_ace_step_model(
    const std::filesystem::path & model_path,
    const AceStepModelSelection & selection) {
    auto assets = load_ace_step_assets(model_path, selection);
    return std::make_unique<AceStepLoadedModel>(metadata(*assets), capabilities(*assets), std::move(assets));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_ace_step_loader() {
    return std::make_shared<AceStepLoader>();
}

}  // namespace engine::models::ace_step
