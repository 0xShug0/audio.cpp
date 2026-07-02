#include "engine/models/vibevoice/loader.h"

#include "engine/framework/io/filesystem.h"
#include "engine/models/vibevoice/lora.h"
#include "engine/models/vibevoice/session.h"

#include <stdexcept>
#include <utility>

namespace engine::models::vibevoice {
namespace {

std::filesystem::path resolve_model_root(const std::filesystem::path & model_path) {
    if (engine::io::is_existing_directory(model_path)) {
        return std::filesystem::weakly_canonical(model_path);
    }
    if (engine::io::is_existing_file(model_path)) {
        return std::filesystem::weakly_canonical(model_path.parent_path());
    }
    throw std::runtime_error("VibeVoice model path does not exist: " + model_path.string());
}

bool has_vibevoice_assets(const std::filesystem::path & root) {
    return engine::io::is_existing_file(root / "config.json") &&
        engine::io::is_existing_file(root / "model.safetensors.index.json") &&
        engine::io::is_existing_file(root / "preprocessor_config.json") &&
        engine::io::is_existing_file(root / "tokenizer_config.json") &&
        engine::io::is_existing_file(root / "tokenizer.json") &&
        engine::io::is_existing_file(root / "vocab.json") &&
        engine::io::is_existing_file(root / "merges.txt");
}

std::vector<runtime::NamedAsset> discover_config_assets(const runtime::ModelLoadRequest & request) {
    const auto root = resolve_model_root(request.model_path);
    return runtime::discover_named_assets(
        root,
        {
            "config.json",
            "preprocessor_config.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "vocab.json",
            "merges.txt",
        });
}

std::vector<runtime::NamedAsset> discover_weight_assets(const runtime::ModelLoadRequest & request) {
    const auto root = resolve_model_root(request.model_path);
    return runtime::discover_named_assets(root, {"model.safetensors.index.json"});
}

runtime::CapabilitySet vibevoice_capabilities() {
    runtime::CapabilitySet capabilities;
    capabilities.supported_tasks = {
        {runtime::VoiceTaskKind::Tts, {runtime::RunMode::Offline}},
    };
    capabilities.languages = {"Auto"};
    capabilities.supports_speaker_reference = true;
    return capabilities;
}

runtime::ModelMetadata vibevoice_metadata(const VibeVoiceAssets & assets) {
    runtime::ModelMetadata metadata;
    metadata.family = "vibevoice";
    metadata.variant = assets.config.model_type + "-qwen2.5";
    metadata.description = "VibeVoice loaded from local safetensors shards.";
    metadata.config_candidates = {
        "config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
    };
    metadata.weight_candidates = {"model.safetensors.index.json"};
    return metadata;
}

class VibeVoiceLoader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return "vibevoice";
    }

    bool can_load(const runtime::ModelLoadRequest & request) const override {
        try {
            const auto root = resolve_model_root(request.model_path);
            return has_vibevoice_assets(root) &&
                (!request.family_hint.has_value() || *request.family_hint == family());
        } catch (...) {
            return false;
        }
    }

    runtime::ModelInspection inspect(const runtime::ModelLoadRequest & request) const override {
        const auto assets = load_vibevoice_assets(resolve_model_root(request.model_path));
        runtime::ModelInspection inspection;
        inspection.model_root = assets->paths.model_root;
        inspection.metadata = vibevoice_metadata(*assets);
        inspection.capabilities = vibevoice_capabilities();
        inspection.cli.request_options = {
            {"voice_samples", "path[,path...]", "Comma-separated speaker reference WAV paths for multi-speaker prompts."},
            {"num_inference_steps", "n", "Diffusion inference steps."},
            {"guidance_scale", "float", "Classifier-free guidance scale."},
            {"max_length_times", "float", "Generation length multiplier."},
            {"prompt_noise_file", "path", "F32 prompt acoustic noise file."},
            {"diffusion_noise_file", "path", "F32 initial diffusion noise file."},
        };
        inspection.cli.session_options = {
            {"vibevoice.weight_type", "native|f32|f16|bf16|q8_0", "Tokenizer, connector, decoder, and diffusion-head weight storage type."},
            {"vibevoice.tokenizer_weight_type", "native|f32|f16|bf16|q8_0", "Audio tokenizer weight storage type."},
            {"vibevoice.connector_weight_type", "native|f32|f16|bf16|q8_0", "Acoustic and semantic connector weight storage type."},
            {"vibevoice.decoder_weight_type", "native|f32|f16|bf16|q8_0", "Language decoder weight storage type."},
            {"vibevoice.diffusion_head_weight_type", "native|f32|f16|bf16|q8_0", "Diffusion prediction head weight storage type."},
        };
        inspection.cli.load_options = {
            {"vibevoice.lora", "path", "Fine-tune adapter dir (LM LoRA + diffusion head + acoustic/semantic connectors) merged at load time."},
            {"vibevoice.lora_scale", "float", "LoRA merge scale override; defaults to lora_alpha / r."},
        };
        inspection.discovered_configs = discover_config_assets(request);
        inspection.discovered_weights = discover_weight_assets(request);
        return inspection;
    }

    std::unique_ptr<runtime::ILoadedVoiceModel> load(const runtime::ModelLoadRequest & request) const override {
        auto assets = apply_vibevoice_finetune_options(
            load_vibevoice_assets(resolve_model_root(request.model_path)), request.options);
        return std::make_unique<VibeVoiceLoadedModel>(
            vibevoice_metadata(*assets), vibevoice_capabilities(), std::move(assets));
    }
};

}  // namespace

VibeVoiceLoadedModel::VibeVoiceLoadedModel(
    runtime::ModelMetadata metadata,
    runtime::CapabilitySet capabilities,
    std::shared_ptr<const VibeVoiceAssets> assets)
    : metadata_(std::move(metadata)),
      capabilities_(std::move(capabilities)),
      assets_(std::move(assets)) {}

const runtime::ModelMetadata & VibeVoiceLoadedModel::metadata() const noexcept {
    return metadata_;
}

const runtime::CapabilitySet & VibeVoiceLoadedModel::capabilities() const noexcept {
    return capabilities_;
}

std::unique_ptr<runtime::IVoiceTaskSession> VibeVoiceLoadedModel::create_task_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options) const {
    if (task.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("VibeVoice only supports offline sessions");
    }
    if (task.task != runtime::VoiceTaskKind::Tts) {
        throw std::runtime_error("VibeVoice only supports the Tts task");
    }
    return std::make_unique<VibeVoiceSession>(task, options, assets_);
}

std::unique_ptr<VibeVoiceLoadedModel> load_vibevoice_model(const std::filesystem::path & model_path) {
    auto assets = load_vibevoice_assets(model_path);
    return std::make_unique<VibeVoiceLoadedModel>(
        vibevoice_metadata(*assets),
        vibevoice_capabilities(),
        std::move(assets));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_vibevoice_loader() {
    return std::make_shared<VibeVoiceLoader>();
}

}  // namespace engine::models::vibevoice
