#include "engine/models/hviske_asr/loader.h"

#include "engine/framework/io/filesystem.h"
#include "engine/models/hviske_asr/session.h"

#include <stdexcept>
#include <utility>

namespace engine::models::hviske_asr {
namespace {

std::filesystem::path resolve_model_root(const std::filesystem::path & model_path) {
    if (engine::io::is_existing_directory(model_path)) {
        return std::filesystem::weakly_canonical(model_path);
    }
    if (engine::io::is_existing_file(model_path)) {
        return std::filesystem::weakly_canonical(model_path.parent_path());
    }
    throw std::runtime_error("Hviske ASR model path does not exist: " + model_path.string());
}

bool has_hviske_assets(const std::filesystem::path & root) {
    return engine::io::is_existing_file(root / "config.json") &&
        engine::io::is_existing_file(root / "model.safetensors") &&
        engine::io::is_existing_file(root / "tokenizer.model");
}

std::vector<runtime::NamedAsset> discover_config_assets(const runtime::ModelLoadRequest & request) {
    return runtime::discover_named_assets(resolve_model_root(request.model_path), {"config.json", "generation_config.json"});
}

std::vector<runtime::NamedAsset> discover_weight_assets(const runtime::ModelLoadRequest & request) {
    return runtime::discover_named_assets(resolve_model_root(request.model_path), {"model.safetensors"});
}

class HviskeASRLoader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return "hviske_asr";
    }

    bool can_load(const runtime::ModelLoadRequest & request) const override {
        try {
            const auto root = resolve_model_root(request.model_path);
            if (!has_hviske_assets(root)) {
                return false;
            }
            const auto assets = load_hviske_assets(root);
            return assets->config.model_type == "cohere_asr" &&
                (!request.family_hint.has_value() || *request.family_hint == family());
        } catch (...) {
            return false;
        }
    }

    runtime::ModelInspection inspect(const runtime::ModelLoadRequest & request) const override {
        const auto assets = load_hviske_assets(resolve_model_root(request.model_path));
        runtime::ModelInspection inspection;
        inspection.model_root = assets->model_root;
        inspection.metadata.family = family();
        inspection.metadata.variant = assets->config.variant.empty() ? assets->config.model_type : assets->config.variant;
        inspection.metadata.description = "Hviske/Cohere ASR loaded from local safetensors assets.";
        inspection.metadata.config_candidates = {"config.json", "generation_config.json"};
        inspection.metadata.weight_candidates = {"model.safetensors"};
        inspection.capabilities.supported_tasks = {
            {runtime::VoiceTaskKind::Asr, {runtime::RunMode::Offline}},
        };
        inspection.capabilities.languages = assets->config.supported_languages;
        inspection.capabilities.supports_timestamps = false;
        inspection.discovered_configs = discover_config_assets(request);
        inspection.discovered_weights = discover_weight_assets(request);
        inspection.cli.request_options = {
            {"language", "code", "ASR language code; defaults to da."},
            {"punctuation", "bool", "Enable or disable punctuation tokens in the decoder prompt."},
            {"max_tokens", "n", "Maximum generated transcript tokens."},
            {"num_beams", "n", "Beam-search beam count; 1 uses greedy or sampling decode."},
            {"length_penalty", "float", "Beam-search length penalty."},
            {"do_sample", "bool", "Enable sampling instead of greedy decode when num_beams is 1."},
            {"temperature", "float", "Sampling temperature."},
            {"top_k", "n", "Top-k sampling limit; 0 disables top-k."},
            {"top_p", "float", "Nucleus sampling limit."},
            {"seed", "n", "Sampling seed."},
        };
        inspection.cli.session_options = {
            {"hviske_asr.weight_type", "native|f32|f16|bf16|q8_0", "Matmul weight storage type."},
            {"hviske_asr.conv_weight_type", "native|f32|f16", "Convolution weight storage type."},
            {"hviske_asr.weight_context_mb", "mb", "Weight context arena size."},
            {"hviske_asr.encoder_graph_arena_mb", "mb", "Encoder graph arena size."},
            {"hviske_asr.decoder_prefill_graph_arena_mb", "mb", "Decoder prefill graph arena size."},
            {"hviske_asr.decoder_decode_graph_arena_mb", "mb", "Decoder cached-step graph arena size."},
        };
        return inspection;
    }

    std::unique_ptr<runtime::ILoadedVoiceModel> load(const runtime::ModelLoadRequest & request) const override {
        return load_hviske_asr_model(resolve_model_root(request.model_path));
    }
};

}  // namespace

HviskeASRLoadedModel::HviskeASRLoadedModel(
    runtime::ModelMetadata metadata,
    runtime::CapabilitySet capabilities,
    std::shared_ptr<const HviskeAssets> assets)
    : metadata_(std::move(metadata)),
      capabilities_(std::move(capabilities)),
      assets_(std::move(assets)) {}

const runtime::ModelMetadata & HviskeASRLoadedModel::metadata() const noexcept {
    return metadata_;
}

const runtime::CapabilitySet & HviskeASRLoadedModel::capabilities() const noexcept {
    return capabilities_;
}

std::unique_ptr<runtime::IVoiceTaskSession> HviskeASRLoadedModel::create_task_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options) const {
    if (task.task != runtime::VoiceTaskKind::Asr) {
        throw std::runtime_error("Hviske ASR only supports the Asr task");
    }
    if (task.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("Hviske ASR currently supports offline sessions");
    }
    return std::make_unique<HviskeASRSession>(task, options, assets_);
}

std::unique_ptr<HviskeASRLoadedModel> load_hviske_asr_model(const std::filesystem::path & model_path) {
    auto assets = load_hviske_assets(model_path);

    runtime::ModelMetadata metadata;
    metadata.family = "hviske_asr";
    metadata.variant = assets->config.variant.empty() ? assets->config.model_type : assets->config.variant;
    metadata.description = "Hviske/Cohere ASR loaded from local safetensors assets.";
    metadata.config_candidates = {"config.json", "generation_config.json"};
    metadata.weight_candidates = {"model.safetensors"};

    runtime::CapabilitySet capabilities;
    capabilities.supported_tasks = {
        {runtime::VoiceTaskKind::Asr, {runtime::RunMode::Offline}},
    };
    capabilities.languages = assets->config.supported_languages;
    capabilities.supports_timestamps = false;

    return std::make_unique<HviskeASRLoadedModel>(std::move(metadata), std::move(capabilities), std::move(assets));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_hviske_asr_loader() {
    return std::make_shared<HviskeASRLoader>();
}

}  // namespace engine::models::hviske_asr
