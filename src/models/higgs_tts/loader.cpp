#include "engine/models/higgs_tts/loader.h"

#include "engine/framework/assets/model_package.h"
#include "engine/models/higgs_tts/session.h"

#include <stdexcept>
#include <utility>

namespace engine::models::higgs_tts {
namespace {

runtime::ModelMetadata metadata(const HiggsAssets &) {
    runtime::ModelMetadata out;
    out.family = "higgs_tts";
    out.variant = "v3-4b";
    out.description = "Higgs Audio v3 TTS loaded from local SGLang-Omni compatible assets.";
    out.config_candidates = {
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    };
    out.weight_candidates = {"model.safetensors.index.json", "model.gguf"};
    return out;
}

runtime::CapabilitySet capabilities(const HiggsAssets &) {
    runtime::CapabilitySet out;
    out.supported_tasks = {
        {runtime::VoiceTaskKind::Tts, {runtime::RunMode::Offline}},
    };
    out.supports_speaker_reference = true;
    out.languages = {"Auto"};
    return out;
}

runtime::ModelCliInterface cli(const HiggsAssets &) {
    runtime::ModelCliInterface out;
    out.request_options = {
        {"max_tokens", "n", "Maximum generated AR tokens; default 2048, 0 uses the default."},
        {"temperature", "float", "AR sampling temperature."},
        {"top_k", "n", "AR top-k sampling limit."},
        {"top_p", "float", "AR nucleus sampling probability."},
        {"repetition_penalty", "float", "Accepted for Python API compatibility; Higgs audio sampling does not consume it."},
        {"seed", "n", "Torch RNG seed."},
        {"text_chunk_size", "n", "Long-form text chunk size; default 1024."},
        {"text_chunk_mode", "default|tag_aware|japanese|endline", "Framework text chunking mode."},
    };
    out.session_options = {
        {"higgs_tts.weight_type", "native|f32|f16|bf16|q8_0", "AR and codec weight storage type."},
        {"higgs_tts.ar_weight_type", "native|f32|f16|bf16|q8_0", "Autoregressive decoder weight storage type."},
        {"higgs_tts.codec_weight_type", "native|f32|f16|bf16|q8_0", "Codec weight storage type."},
        {"higgs_tts.ar_weight_context_mb", "n", "AR weight context size."},
        {"higgs_tts.codec_weight_context_mb", "n", "Codec weight context size."},
        {"higgs_tts.ar_decode_graph_arena_mb", "n", "AR decode graph arena size."},
        {"higgs_tts.codec_decode_graph_arena_mb", "n", "Codec decode graph arena size."},
        {"higgs_tts.codec_encode_graph_arena_mb", "n", "Codec encode graph arena size."},
    };
    return out;
}

class HiggsTTSLoader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return "higgs_tts";
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
        const auto assets = load_higgs_assets(request.model_path);
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
        return load_higgs_tts_model(request.model_path);
    }
};

}  // namespace

HiggsTTSLoadedModel::HiggsTTSLoadedModel(
    runtime::ModelMetadata metadata,
    runtime::CapabilitySet capabilities,
    std::shared_ptr<const HiggsAssets> assets)
    : metadata_(std::move(metadata)),
      capabilities_(std::move(capabilities)),
      assets_(std::move(assets)) {}

const runtime::ModelMetadata & HiggsTTSLoadedModel::metadata() const noexcept {
    return metadata_;
}

const runtime::CapabilitySet & HiggsTTSLoadedModel::capabilities() const noexcept {
    return capabilities_;
}

std::unique_ptr<runtime::IVoiceTaskSession> HiggsTTSLoadedModel::create_task_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options) const {
    if (task.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("Higgs TTS only supports offline sessions");
    }
    if (task.task != runtime::VoiceTaskKind::Tts) {
        throw std::runtime_error("Higgs TTS only supports the Tts task");
    }
    return std::make_unique<HiggsTTSSession>(task, options, assets_);
}

std::unique_ptr<HiggsTTSLoadedModel> load_higgs_tts_model(const std::filesystem::path & model_path) {
    auto assets = load_higgs_assets(model_path);
    return std::make_unique<HiggsTTSLoadedModel>(
        metadata(*assets),
        capabilities(*assets),
        std::move(assets));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_higgs_tts_loader() {
    return std::make_shared<HiggsTTSLoader>();
}

}  // namespace engine::models::higgs_tts
