#include "engine/models/fish_audio/loader.h"

#include "engine/framework/assets/model_package.h"
#include "engine/models/fish_audio/session.h"

#include <stdexcept>
#include <utility>

namespace engine::models::fish_audio {
namespace {

runtime::ModelMetadata metadata(const FishAudioAssets &) {
    runtime::ModelMetadata out;
    out.family = "fish_audio";
    out.variant = "s2-pro";
    out.description = "Fish Audio S2-Pro loaded from prepared local assets.";
    out.config_candidates = {"config.json", "tokenizer_config.json", "tokenizer.json"};
    out.weight_candidates = {"model_audio_cpp.safetensors.index.json", "codec.safetensors", "model.gguf"};
    return out;
}

runtime::CapabilitySet capabilities(const FishAudioAssets &) {
    runtime::CapabilitySet out;
    out.supported_tasks = {
        {runtime::VoiceTaskKind::Tts, {runtime::RunMode::Offline}},
    };
    out.languages = {"en", "zh", "auto"};
    out.supports_speaker_reference = true;
    out.supports_style_condition = true;
    return out;
}

runtime::ModelCliInterface cli(const FishAudioAssets &) {
    runtime::ModelCliInterface out;
    out.request_options = {
        {"reference_text", "TEXT", "Reference transcript used with speaker reference audio."},
        {"max_new_tokens", "N", "Maximum Fish Audio semantic tokens to generate; default 1024, 0 uses the default."},
        {"text_chunk_size", "N", "Long-form text chunk size; default 200."},
        {"text_chunk_mode", "default|tag_aware|japanese|endline", "Framework text chunking mode."},
        {"top_p", "FLOAT", "Top-p sampling value."},
        {"top_k", "N", "Top-k sampling value."},
        {"temperature", "FLOAT", "Sampling temperature."},
        {"seed", "N", "Sampling seed for reproducible output; omitted uses a random seed."},
    };
    out.session_options = {
        {"fish_audio.mem_saver", "true|false", "Release cached AR runtime graphs after each request; default false."},
        {"fish_audio.reference_cache_slots", "n", "Prepared reference-audio cache slots; default 1."},
        {"fish_audio.weight_type", "native|f32|f16|bf16|q8_0", "AR matmul weight storage type; default native."},
        {"fish_audio.codec_weight_type", "native|f32|f16|q8_0", "Codec conv/matmul weight storage type; default native."},
    };
    return out;
}

class FishAudioLoader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return "fish_audio";
    }

    runtime::CapabilitySet advertised_capabilities() const override {
        runtime::CapabilitySet out;
        out.supported_tasks = {
            {runtime::VoiceTaskKind::Tts, {runtime::RunMode::Offline}},
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
            const auto package_spec = engine::assets::default_model_package_spec_path(family());
            (void) engine::assets::load_resource_bundle_from_package_spec(request.model_path, package_spec);
            return true;
        } catch (...) {
            return false;
        }
    }

    runtime::ModelInspection inspect(const runtime::ModelLoadRequest & request) const override {
        const auto assets = load_fish_audio_assets(request.model_path);
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
        return load_fish_audio_model(request.model_path);
    }
};

}  // namespace

FishAudioLoadedModel::FishAudioLoadedModel(
    runtime::ModelMetadata metadata,
    runtime::CapabilitySet capabilities,
    std::shared_ptr<const FishAudioAssets> assets)
    : metadata_(std::move(metadata)),
      capabilities_(std::move(capabilities)),
      assets_(std::move(assets)) {}

const runtime::ModelMetadata & FishAudioLoadedModel::metadata() const noexcept {
    return metadata_;
}

const runtime::CapabilitySet & FishAudioLoadedModel::capabilities() const noexcept {
    return capabilities_;
}

std::unique_ptr<runtime::IVoiceTaskSession> FishAudioLoadedModel::create_task_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options) const {
    if (task.task != runtime::VoiceTaskKind::Tts || task.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("Fish Audio S2-Pro supports offline TTS sessions");
    }
    return std::make_unique<FishAudioSession>(task, options, assets_);
}

std::unique_ptr<FishAudioLoadedModel> load_fish_audio_model(const std::filesystem::path & model_path) {
    auto assets = load_fish_audio_assets(model_path);
    return std::make_unique<FishAudioLoadedModel>(
        metadata(*assets),
        capabilities(*assets),
        std::move(assets));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_fish_audio_loader() {
    return std::make_shared<FishAudioLoader>();
}

}  // namespace engine::models::fish_audio
