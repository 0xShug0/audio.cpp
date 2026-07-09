#include "engine/models/vibevoice_asr/loader.h"

#include "engine/framework/io/filesystem.h"
#include "engine/models/vibevoice_asr/session.h"

#include <stdexcept>
#include <utility>

namespace engine::models::vibevoice_asr {
namespace {

std::filesystem::path resolve_model_root(const std::filesystem::path & model_path) {
    if (engine::io::is_existing_directory(model_path)) {
        return std::filesystem::weakly_canonical(model_path);
    }
    if (engine::io::is_existing_file(model_path)) {
        return std::filesystem::weakly_canonical(model_path.parent_path());
    }
    throw std::runtime_error("VibeVoice-ASR model path does not exist: " + model_path.string());
}

bool has_vibevoice_asr_assets(const std::filesystem::path & root) {
    return engine::io::is_existing_file(root / "config.json") &&
        engine::io::is_existing_file(root / "model.safetensors.index.json") &&
        engine::io::is_existing_file(root / "tokenizer_config.json") &&
        engine::io::is_existing_file(root / "tokenizer.json") &&
        engine::io::is_existing_file(root / "vocab.json") &&
        engine::io::is_existing_file(root / "merges.txt");
}

std::vector<runtime::NamedAsset> discover_config_assets(const runtime::ModelLoadRequest & request) {
    const auto root = resolve_model_root(request.model_path);
    return runtime::discover_named_assets(
        root,
        {"config.json", "tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt"});
}

std::vector<runtime::NamedAsset> discover_weight_assets(const runtime::ModelLoadRequest & request) {
    const auto root = resolve_model_root(request.model_path);
    return runtime::discover_named_assets(root, {"model.safetensors.index.json"});
}

runtime::CapabilitySet vibevoice_asr_capabilities() {
    runtime::CapabilitySet capabilities;
    capabilities.supported_tasks = {
        {runtime::VoiceTaskKind::Asr, {runtime::RunMode::Offline}},
    };
    capabilities.languages = {"auto"};
    capabilities.supports_timestamps = true;
    return capabilities;
}

runtime::ModelMetadata vibevoice_asr_metadata(const VibeVoiceAssets & assets) {
    runtime::ModelMetadata metadata;
    metadata.family = "vibevoice_asr";
    metadata.variant = assets.config.model_type.empty() ? "vibevoice-asr" : assets.config.model_type;
    metadata.description = "VibeVoice-ASR loaded from local safetensors shards.";
    metadata.config_candidates = {
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
    };
    metadata.weight_candidates = {"model.safetensors.index.json"};
    return metadata;
}

class VibeVoiceASRLoader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return "vibevoice_asr";
    }

    bool can_load(const runtime::ModelLoadRequest & request) const override {
        try {
            const auto root = resolve_model_root(request.model_path);
            return has_vibevoice_asr_assets(root) &&
                (!request.family_hint.has_value() || *request.family_hint == family());
        } catch (...) {
            return false;
        }
    }

    runtime::ModelInspection inspect(const runtime::ModelLoadRequest & request) const override {
        const auto assets = load_vibevoice_assets(resolve_model_root(request.model_path));
        runtime::ModelInspection inspection;
        inspection.model_root = assets->paths.model_root;
        inspection.metadata = vibevoice_asr_metadata(*assets);
        inspection.capabilities = vibevoice_asr_capabilities();
        inspection.cli.request_options = {
            {"language", "code", "ASR language label."},
            {"max_tokens", "n", "Maximum generated text tokens."},
            {"temperature", "float", "Sampling temperature; 0 uses deterministic decoding."},
            {"top_p", "float", "Nucleus sampling probability."},
            {"top_k", "n", "Top-k sampling limit; 0 disables top-k filtering."},
            {"num_beams", "n", "Beam count for deterministic beam search."},
            {"repetition_penalty", "float", "Generation repetition penalty."},
            {"seed", "n", "Acoustic latent sampling seed."},
            {"audio_chunk_mode", "auto|fixed|vad|none", "Audio chunking mode; default auto uses fixed chunks."},
            {"audio_chunk_seconds", "seconds", "Audio chunk duration; default 1200."},
        };
        inspection.cli.session_options = {
            {"vibevoice_asr.weight_type", "native|f32|f16|bf16|q8_0", "Tokenizer, connector, and decoder weight storage type."},
            {"vibevoice_asr.tokenizer_weight_type", "native|f32|f16|bf16|q8_0", "Speech tokenizer weight storage type."},
            {"vibevoice_asr.connector_weight_type", "native|f32|f16|bf16|q8_0", "Acoustic and semantic connector weight storage type."},
            {"vibevoice_asr.decoder_weight_type", "native|f32|f16|bf16|q8_0", "Text decoder weight storage type."},
            {"vibevoice_asr.tokenizer_weight_context_mb", "mb", "Speech tokenizer weight context arena size."},
            {"vibevoice_asr.connector_weight_context_mb", "mb", "Acoustic and semantic connector weight context arena size."},
            {"vibevoice_asr.decoder_weight_context_mb", "mb", "Text decoder weight context arena size."},
            {"vibevoice_asr.vad_model_path", "path", "Silero VAD model path used by audio_chunk_mode=vad."},
        };
        inspection.discovered_configs = discover_config_assets(request);
        inspection.discovered_weights = discover_weight_assets(request);
        return inspection;
    }

    std::unique_ptr<runtime::ILoadedVoiceModel> load(const runtime::ModelLoadRequest & request) const override {
        return load_vibevoice_asr_model(resolve_model_root(request.model_path));
    }
};

}  // namespace

VibeVoiceASRLoadedModel::VibeVoiceASRLoadedModel(
    runtime::ModelMetadata metadata,
    runtime::CapabilitySet capabilities,
    std::shared_ptr<const VibeVoiceAssets> assets)
    : metadata_(std::move(metadata)),
      capabilities_(std::move(capabilities)),
      assets_(std::move(assets)) {}

const runtime::ModelMetadata & VibeVoiceASRLoadedModel::metadata() const noexcept {
    return metadata_;
}

const runtime::CapabilitySet & VibeVoiceASRLoadedModel::capabilities() const noexcept {
    return capabilities_;
}

std::unique_ptr<runtime::IVoiceTaskSession> VibeVoiceASRLoadedModel::create_task_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options) const {
    if (task.task != runtime::VoiceTaskKind::Asr) {
        throw std::runtime_error("VibeVoice-ASR only supports the Asr task");
    }
    if (task.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("VibeVoice-ASR streaming sessions are not supported");
    }
    return std::make_unique<VibeVoiceASRSession>(task, options, assets_);
}

std::unique_ptr<VibeVoiceASRLoadedModel> load_vibevoice_asr_model(const std::filesystem::path & model_path) {
    auto assets = load_vibevoice_assets(model_path);
    return std::make_unique<VibeVoiceASRLoadedModel>(
        vibevoice_asr_metadata(*assets),
        vibevoice_asr_capabilities(),
        std::move(assets));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_vibevoice_asr_loader() {
    return std::make_shared<VibeVoiceASRLoader>();
}

}  // namespace engine::models::vibevoice_asr
