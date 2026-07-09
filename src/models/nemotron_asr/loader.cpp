#include "engine/models/nemotron_asr/loader.h"

#include "engine/framework/io/filesystem.h"
#include "engine/models/nemotron_asr/session.h"

#include <stdexcept>
#include <utility>

namespace engine::models::nemotron_asr {
namespace {

std::filesystem::path resolve_model_root(const std::filesystem::path & model_path) {
    if (engine::io::is_existing_directory(model_path)) {
        return std::filesystem::weakly_canonical(model_path);
    }
    if (engine::io::is_existing_file(model_path)) {
        return std::filesystem::weakly_canonical(model_path.parent_path());
    }
    throw std::runtime_error("Nemotron ASR model path does not exist: " + model_path.string());
}

bool has_nemotron_assets(const std::filesystem::path & root) {
    return engine::io::is_existing_file(root / "config.json") &&
        engine::io::is_existing_file(root / "processor_config.json") &&
        engine::io::is_existing_file(root / "tokenizer.json") &&
        engine::io::is_existing_file(root / "model.safetensors");
}

std::vector<runtime::NamedAsset> discover_config_assets(const runtime::ModelLoadRequest & request) {
    return runtime::discover_named_assets(resolve_model_root(request.model_path), {"config.json", "processor_config.json"});
}

std::vector<runtime::NamedAsset> discover_weight_assets(const runtime::ModelLoadRequest & request) {
    return runtime::discover_named_assets(resolve_model_root(request.model_path), {"model.safetensors"});
}

class NemotronASRLoader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return "nemotron_asr";
    }

    bool can_load(const runtime::ModelLoadRequest & request) const override {
        try {
            const auto root = resolve_model_root(request.model_path);
            if (!has_nemotron_assets(root)) {
                return false;
            }
            const auto assets = load_nemotron_asr_assets(root);
            return assets->config.model_type == "nemotron3_5_asr" &&
                (!request.family_hint.has_value() || *request.family_hint == family());
        } catch (...) {
            return false;
        }
    }

    runtime::ModelInspection inspect(const runtime::ModelLoadRequest & request) const override {
        const auto assets = load_nemotron_asr_assets(resolve_model_root(request.model_path));
        runtime::ModelInspection inspection;
        inspection.model_root = assets->paths.model_root;
        inspection.metadata.family = family();
        inspection.metadata.variant = assets->config.model_type;
        inspection.metadata.description = "NVIDIA Nemotron 3.5 ASR streaming RNNT loaded from local safetensors assets.";
        inspection.metadata.config_candidates = {"config.json", "processor_config.json"};
        inspection.metadata.weight_candidates = {"model.safetensors"};
        inspection.capabilities.supported_tasks = {
            {runtime::VoiceTaskKind::Asr, {runtime::RunMode::Offline, runtime::RunMode::Streaming}},
        };
        for (const auto & item : assets->config.prompt_dictionary) {
            inspection.capabilities.languages.push_back(item.first);
        }
        inspection.capabilities.supports_timestamps = true;
        inspection.discovered_configs = discover_config_assets(request);
        inspection.discovered_weights = discover_weight_assets(request);
        inspection.cli.request_options = {
            {"language", "code", "ASR prompt language such as en-US, da-DK, or auto."},
            {"lookahead_tokens", "n", "Chunk-limited encoder right context; supported values come from processor_config."},
            {"max_tokens", "n", "Maximum RNNT generated tokens; 0 uses the model-derived limit."},
            {"keep_language_tags", "bool", "Keep language tag tokens in decoded text."},
        };
        inspection.cli.session_options = {
            {"nemotron_asr.weight_type", "native|f32|f16|bf16|q8_0", "Shared matmul weight storage type."},
            {"nemotron_asr.matmul_weight_type", "native|f32|f16|bf16|q8_0", "Encoder and decoder matmul weight storage type."},
            {"nemotron_asr.conv_weight_type", "native|f32|f16", "Convolution weight storage type."},
            {"nemotron_asr.weight_context_mb", "mb", "Weight context arena size."},
            {"nemotron_asr.encoder_graph_arena_mb", "mb", "Encoder graph arena size."},
            {"nemotron_asr.decoder_graph_arena_mb", "mb", "Decoder graph arena size."},
            {"nemotron_asr.mem_saver", "true|false", "Release the offline encoder graph after each offline request; default false."},
        };
        return inspection;
    }

    std::unique_ptr<runtime::ILoadedVoiceModel> load(const runtime::ModelLoadRequest & request) const override {
        return load_nemotron_asr_model(resolve_model_root(request.model_path));
    }
};

}  // namespace

NemotronASRLoadedModel::NemotronASRLoadedModel(
    runtime::ModelMetadata metadata,
    runtime::CapabilitySet capabilities,
    std::shared_ptr<const NemotronAssets> assets)
    : metadata_(std::move(metadata)),
      capabilities_(std::move(capabilities)),
      assets_(std::move(assets)) {}

const runtime::ModelMetadata & NemotronASRLoadedModel::metadata() const noexcept {
    return metadata_;
}

const runtime::CapabilitySet & NemotronASRLoadedModel::capabilities() const noexcept {
    return capabilities_;
}

std::unique_ptr<runtime::IVoiceTaskSession> NemotronASRLoadedModel::create_task_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options) const {
    if (task.task != runtime::VoiceTaskKind::Asr) {
        throw std::runtime_error("Nemotron ASR only supports the Asr task");
    }
    if (task.mode != runtime::RunMode::Offline && task.mode != runtime::RunMode::Streaming) {
        throw std::runtime_error("Nemotron ASR only supports offline and streaming sessions");
    }
    if (task.mode == runtime::RunMode::Streaming) {
        return std::make_unique<NemotronASRStreamingSession>(task, options, assets_);
    }
    return std::make_unique<NemotronASROfflineSession>(task, options, assets_);
}

std::unique_ptr<NemotronASRLoadedModel> load_nemotron_asr_model(const std::filesystem::path & model_path) {
    auto assets = load_nemotron_asr_assets(model_path);

    runtime::ModelMetadata metadata;
    metadata.family = "nemotron_asr";
    metadata.variant = assets->config.model_type;
    metadata.description = "NVIDIA Nemotron 3.5 ASR streaming RNNT loaded from local safetensors assets.";
    metadata.config_candidates = {"config.json", "processor_config.json"};
    metadata.weight_candidates = {"model.safetensors"};

    runtime::CapabilitySet capabilities;
    capabilities.supported_tasks = {
        {runtime::VoiceTaskKind::Asr, {runtime::RunMode::Offline, runtime::RunMode::Streaming}},
    };
    for (const auto & item : assets->config.prompt_dictionary) {
        capabilities.languages.push_back(item.first);
    }
    capabilities.supports_timestamps = true;

    return std::make_unique<NemotronASRLoadedModel>(std::move(metadata), std::move(capabilities), std::move(assets));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_nemotron_asr_loader() {
    return std::make_shared<NemotronASRLoader>();
}

}  // namespace engine::models::nemotron_asr
