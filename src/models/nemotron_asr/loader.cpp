#include "engine/models/nemotron_asr/loader.h"

#include "engine/framework/model_spec/package.h"
#include "engine/framework/model_spec/metadata.h"
#include "engine/models/nemotron_asr/session.h"

#include <stdexcept>
#include <utility>

namespace engine::models::nemotron_asr {
namespace {

// Compatibility for GGUFs that embed a pre-v1 spec when no v1 contract is available:
// describe the supported runtime options without relying on a v1 contract.
runtime::ModelCliInterface legacy_cli() {
    runtime::ModelCliInterface cli;
    cli.request_options = {
        {"language", "code", "ASR prompt language such as en-US, da-DK, or auto."},
        {"lookahead_tokens", "n", "Chunk-limited encoder right context; supported values come from processor_config.", false, std::nullopt, "0"},
        {"max_tokens", "n", "Maximum RNNT generated tokens; 0 uses the model-derived limit.", false, "0", "0"},
        {"keep_language_tags", "bool", "Keep language tag tokens in decoded text.", false, "false"},
        {"return_timestamps", "true|false", "Advisory compatibility option; timestamps remain present when false or omitted.", false, "false"},
        {"streaming", "true|false", "Compatibility flag; true requires a streaming session. Select --mode streaming rather than setting it on an offline request.", false, "false"},
        {"speaker_probabilities", "path", "NPZ speaker probability file for speaker tagging."},
        {"speaker_mode", "attribution|masked", "Speaker tagging mode; requires speaker_probabilities.", false, "masked"},
        {"speaker_mask", "mel|audio", "Speaker masking domain; requires speaker_probabilities.", false, "mel"},
        {"speaker_segment_gap_sec", "seconds", "Maximum gap for joining speaker segments; requires speaker_probabilities.", false, "1", "0"},
        {"speaker_segment_max_sec", "seconds", "Positive maximum speaker segment duration; requires speaker_probabilities.", false, "10"},
    };
    cli.session_options = {
        {"nemotron_asr.weight_type", "native|f32|f16|bf16|q8_0", "Shared matmul weight storage type.", false, "native"},
        {"nemotron_asr.matmul_weight_type", "native|f32|f16|bf16|q8_0", "Encoder and decoder matmul weight storage type."},
        {"nemotron_asr.conv_weight_type", "native|f32|f16", "Convolution weight storage type.", false, "native"},
        {"nemotron_asr.weight_context_mb", "mb", "Weight context arena size.", false, "3072", "1"},
        {"nemotron_asr.encoder_graph_arena_mb", "mb", "Encoder graph arena size.", false, "1024", "1"},
        {"nemotron_asr.decoder_graph_arena_mb", "mb", "Decoder graph arena size.", false, "256", "1"},
        {"nemotron_asr.mem_saver", "true|false", "Release the offline encoder graph after each offline request; default false.", false, "false"},
    };
    return cli;
}

// nullptr when only a legacy embedded spec exists; sessions then keep the legacy validation.
std::shared_ptr<const model_spec::ModelContract> find_contract() {
    auto contract = model_spec::find_model_contract("nemotron_asr");
    return contract ? std::make_shared<const model_spec::ModelContract>(std::move(*contract)) : nullptr;
}

runtime::ModelMetadata metadata(const NemotronASRAssets & assets) {
    runtime::ModelMetadata out;
    out.family = "nemotron_asr";
    out.variant = assets.config.model_type;
    out.description = "NVIDIA Nemotron 3.5 ASR streaming RNNT loaded from local assets.";
    return out;
}

runtime::CapabilitySet capabilities(const NemotronASRAssets & assets) {
    runtime::CapabilitySet out;
    out.supported_tasks = {
        {runtime::VoiceTaskKind::Asr, {runtime::RunMode::Offline, runtime::RunMode::Streaming}},
    };
    for (const auto & item : assets.config.prompt_dictionary) {
        out.languages.push_back(item.first);
    }
    out.supports_timestamps = true;
    return out;
}

class NemotronASRLoader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return "nemotron_asr";
    }

    runtime::CapabilitySet advertised_capabilities() const override {
        runtime::CapabilitySet out;
        out.supported_tasks = {
            {runtime::VoiceTaskKind::Asr, {runtime::RunMode::Offline, runtime::RunMode::Streaming}},
        };
        out.supports_timestamps = true;
        return out;
    }

    bool can_load(const runtime::ModelLoadRequest & request) const override {
        try {
            const auto package_spec = engine::model_spec::default_spec_path(family());
            (void) engine::model_spec::load_resource_bundle(request.model_path, package_spec);
            return !request.family_hint.has_value() || *request.family_hint == family();
        } catch (...) {
            return false;
        }
    }

    runtime::ModelInspection inspect(const runtime::ModelLoadRequest & request) const override {
        const auto assets = load_nemotron_asr_assets(request.model_path);
        runtime::ModelInspection inspection;
        inspection.model_root = assets->resources.model_root();
        inspection.metadata = metadata(*assets);
        inspection.capabilities = capabilities(*assets);
        const auto package_spec = engine::model_spec::default_spec_path(family());
        inspection.discovered_configs = runtime::discover_named_assets_from_package_spec(
            request.model_path,
            package_spec,
            engine::model_spec::ResourceKind::Files);
        inspection.discovered_weights = runtime::discover_named_assets_from_package_spec(
            request.model_path,
            package_spec,
            engine::model_spec::ResourceKind::Tensors);
        const auto contract = find_contract();
        inspection.cli = contract ? contract->cli : legacy_cli();
        return inspection;
    }

    std::unique_ptr<runtime::ILoadedVoiceModel> load(const runtime::ModelLoadRequest & request) const override {
        return load_nemotron_asr_model(request.model_path);
    }
};

}  // namespace

NemotronASRLoadedModel::NemotronASRLoadedModel(
    runtime::ModelMetadata metadata,
    runtime::CapabilitySet capabilities,
    std::shared_ptr<const NemotronASRAssets> assets,
    std::shared_ptr<const model_spec::ModelContract> contract)
    : metadata_(std::move(metadata)),
      capabilities_(std::move(capabilities)),
      assets_(std::move(assets)),
      contract_(std::move(contract)) {}

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
        return std::make_unique<NemotronASRStreamingSession>(task, options, assets_, contract_);
    }
    return std::make_unique<NemotronASROfflineSession>(task, options, assets_, contract_);
}

std::unique_ptr<NemotronASRLoadedModel> load_nemotron_asr_model(const std::filesystem::path & model_path) {
    auto assets = load_nemotron_asr_assets(model_path);
    return std::make_unique<NemotronASRLoadedModel>(
        metadata(*assets),
        capabilities(*assets),
        std::move(assets),
        find_contract());
}

std::shared_ptr<runtime::IVoiceModelLoader> make_nemotron_asr_loader() {
    return std::make_shared<NemotronASRLoader>();
}

}  // namespace engine::models::nemotron_asr
