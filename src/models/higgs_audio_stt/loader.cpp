#include "engine/models/higgs_audio_stt/loader.h"

#include "engine/framework/io/filesystem.h"
#include "engine/models/higgs_audio_stt/session.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace engine::models::higgs_audio_stt {
namespace {

std::filesystem::path resolve_model_root(const std::filesystem::path & model_path) {
    if (engine::io::is_existing_directory(model_path)) {
        return std::filesystem::weakly_canonical(model_path);
    }
    if (engine::io::is_existing_file(model_path)) {
        return std::filesystem::weakly_canonical(model_path.parent_path());
    }
    throw std::runtime_error("Higgs Audio STT model path does not exist: " + model_path.string());
}

bool has_higgs_audio_stt_assets(const std::filesystem::path & root) {
    return engine::io::is_existing_file(root / "config.json")
        && engine::io::is_existing_file(root / "model.safetensors.index.json")
        && engine::io::is_existing_file(root / "tokenizer_config.json")
        && engine::io::is_existing_file(root / "vocab.json")
        && engine::io::is_existing_file(root / "merges.txt");
}

std::vector<runtime::NamedAsset> discover_config_assets(const runtime::ModelLoadRequest & request) {
    const auto root = resolve_model_root(request.model_path);
    return runtime::discover_named_assets(
        root,
        {"config.json", "generation_config.json", "tokenizer_config.json"});
}

std::vector<runtime::NamedAsset> discover_weight_assets(const runtime::ModelLoadRequest & request) {
    const auto root = resolve_model_root(request.model_path);
    return runtime::discover_named_assets(root, {"model.safetensors.index.json"});
}

class HiggsAudioSTTLoader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return "higgs_audio_stt";
    }

    bool can_load(const runtime::ModelLoadRequest & request) const override {
        try {
            const auto root = resolve_model_root(request.model_path);
            return has_higgs_audio_stt_assets(root)
                && (!request.family_hint.has_value() || *request.family_hint == family());
        } catch (...) {
            return false;
        }
    }

    runtime::ModelInspection inspect(const runtime::ModelLoadRequest & request) const override {
        const auto assets = load_higgs_audio_stt_assets(resolve_model_root(request.model_path));
        runtime::ModelInspection inspection;
        inspection.model_root = assets->paths.model_root;
        inspection.metadata.family = family();
        inspection.metadata.variant = assets->config.model_size.empty() ? assets->config.model_type : assets->config.model_size;
        inspection.metadata.description = "Higgs Audio STT loaded from local extracted assets.";
        inspection.metadata.config_candidates = {"config.json", "generation_config.json", "tokenizer_config.json"};
        inspection.metadata.weight_candidates = {"model.safetensors.index.json"};
        inspection.capabilities.supported_tasks = {
            {runtime::VoiceTaskKind::Asr, {runtime::RunMode::Offline, runtime::RunMode::Streaming}},
        };
        inspection.capabilities.supports_timestamps = false;
        inspection.capabilities.languages = assets->config.supported_languages;
        inspection.discovered_configs = discover_config_assets(request);
        inspection.discovered_weights = discover_weight_assets(request);
        inspection.cli.request_options = {
            {"language", "code", "ASR language code."},
            {"max_tokens", "n", "Maximum generated transcript tokens."},
            {"enable_thinking", "true|false", "Enable the model thinking prompt; default true."},
            {"audio_chunk_mode", "auto|fixed|none", "Audio chunking mode; default auto uses fixed chunks."},
            {"audio_chunk_seconds", "seconds", "Fixed audio chunk duration; default 4."},
        };
        inspection.cli.session_options = {
            {"higgs_audio_stt.weight_type", "native|f32|f16|bf16|q8_0", "Shared text decoder weight storage type."},
            {"higgs_audio_stt.audio_encoder_weight_type", "native|f32|f16", "Audio encoder convolution weight storage type."},
            {"higgs_audio_stt.text_decoder_weight_type", "native|f32|f16|bf16|q8_0", "Text decoder matmul weight storage type."},
            {"higgs_audio_stt.audio_encoder_graph_arena_mb", "mb", "Audio encoder graph arena size."},
            {"higgs_audio_stt.text_decoder_prefill_graph_arena_mb", "mb", "Text decoder prefill graph arena size."},
            {"higgs_audio_stt.text_decoder_decode_graph_arena_mb", "mb", "Text decoder cached-step graph arena size."},
            {"higgs_audio_stt.text_decoder_weight_context_mb", "mb", "Text decoder weight context arena size."},
        };
        return inspection;
    }

    std::unique_ptr<runtime::ILoadedVoiceModel> load(const runtime::ModelLoadRequest & request) const override {
        return load_higgs_audio_stt_model(resolve_model_root(request.model_path));
    }
};

}  // namespace

HiggsAudioSTTLoadedModel::HiggsAudioSTTLoadedModel(
    runtime::ModelMetadata metadata,
    runtime::CapabilitySet capabilities,
    std::shared_ptr<const HiggsAudioSTTAssets> assets)
    : metadata_(std::move(metadata)),
      capabilities_(std::move(capabilities)),
      assets_(std::move(assets)) {}

const runtime::ModelMetadata & HiggsAudioSTTLoadedModel::metadata() const noexcept {
    return metadata_;
}

const runtime::CapabilitySet & HiggsAudioSTTLoadedModel::capabilities() const noexcept {
    return capabilities_;
}

std::unique_ptr<runtime::IVoiceTaskSession> HiggsAudioSTTLoadedModel::create_task_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options) const {
    if (task.task != runtime::VoiceTaskKind::Asr) {
        throw std::runtime_error("Higgs Audio STT only supports the Asr task");
    }
    if (task.mode != runtime::RunMode::Offline && task.mode != runtime::RunMode::Streaming) {
        throw std::runtime_error("Higgs Audio STT currently supports offline and streaming sessions");
    }
    return std::make_unique<HiggsAudioSTTSession>(task, options, assets_);
}

std::unique_ptr<HiggsAudioSTTLoadedModel> load_higgs_audio_stt_model(const std::filesystem::path & model_path) {
    auto assets = load_higgs_audio_stt_assets(model_path);

    runtime::ModelMetadata metadata;
    metadata.family = "higgs_audio_stt";
    metadata.variant = assets->config.model_size.empty() ? assets->config.model_type : assets->config.model_size;
    metadata.description = "Higgs Audio STT loaded from local extracted assets.";
    metadata.config_candidates = {"config.json", "generation_config.json", "tokenizer_config.json"};
    metadata.weight_candidates = {"model.safetensors.index.json"};

    runtime::CapabilitySet capabilities;
    capabilities.supported_tasks = {
        {runtime::VoiceTaskKind::Asr, {runtime::RunMode::Offline, runtime::RunMode::Streaming}},
    };
    capabilities.languages = assets->config.supported_languages;
    capabilities.supports_timestamps = false;

    return std::make_unique<HiggsAudioSTTLoadedModel>(std::move(metadata), std::move(capabilities), std::move(assets));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_higgs_audio_stt_loader() {
    return std::make_shared<HiggsAudioSTTLoader>();
}

}  // namespace engine::models::higgs_audio_stt
