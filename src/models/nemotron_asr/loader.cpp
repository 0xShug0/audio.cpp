#include "engine/models/nemotron_asr/loader.h"

#include "engine/framework/assets/model_package.h"
#include "engine/models/nemotron_asr/session.h"

#include <stdexcept>
#include <utility>

namespace engine::models::nemotron_asr {
namespace {

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

    bool can_load(const runtime::ModelLoadRequest & request) const override {
        try {
            const auto package_spec = engine::assets::default_model_package_spec_path(family());
            (void) engine::assets::load_resource_bundle_from_package_spec(request.model_path, package_spec);
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
        const auto package_spec = engine::assets::default_model_package_spec_path(family());
        inspection.discovered_configs = runtime::discover_named_assets_from_package_spec(
            request.model_path,
            package_spec,
            engine::assets::ModelPackageResourceKind::Files);
        inspection.discovered_weights = runtime::discover_named_assets_from_package_spec(
            request.model_path,
            package_spec,
            engine::assets::ModelPackageResourceKind::Tensors);
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
        return load_nemotron_asr_model(request.model_path);
    }
};

}  // namespace

NemotronASRLoadedModel::NemotronASRLoadedModel(
    runtime::ModelMetadata metadata,
    runtime::CapabilitySet capabilities,
    std::shared_ptr<const NemotronASRAssets> assets)
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
        return std::make_unique<NemotronASRLoadedModel>(
        metadata(*assets),
        capabilities(*assets),
        std::move(assets));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_nemotron_asr_loader() {
    return std::make_shared<NemotronASRLoader>();
}

}  // namespace engine::models::nemotron_asr
