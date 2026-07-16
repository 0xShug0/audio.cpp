#include "engine/models/higgs_audio_stt/loader.h"

#include "engine/framework/assets/model_package.h"
#include "engine/models/higgs_audio_stt/session.h"

#include <stdexcept>
#include <utility>

namespace engine::models::higgs_audio_stt {
namespace {

runtime::ModelMetadata metadata(const HiggsAudioSTTAssets & assets) {
    runtime::ModelMetadata out;
    out.family = "higgs_audio_stt";
    out.variant = assets.config.model_size.empty() ? assets.config.model_type : assets.config.model_size;
    out.description = "Higgs Audio STT loaded from local extracted assets.";
    return out;
}

runtime::CapabilitySet capabilities(const HiggsAudioSTTAssets & assets) {
    runtime::CapabilitySet out;
    out.supported_tasks = {
        {runtime::VoiceTaskKind::Asr, {runtime::RunMode::Offline, runtime::RunMode::Streaming}},
    };
    out.languages = assets.config.supported_languages;
    out.supports_timestamps = false;
    return out;
}

runtime::ModelCliInterface cli(const HiggsAudioSTTAssets &) {
    runtime::ModelCliInterface out;
    out.request_options = {
        {"language", "code", "ASR language code."},
        {"max_tokens", "n", "Maximum generated transcript tokens."},
        {"enable_thinking", "true|false", "Enable the model thinking prompt; default true."},
        {"audio_chunk_mode", "auto|fixed|none", "Audio chunking mode; default auto uses fixed chunks."},
        {"audio_chunk_seconds", "seconds", "Fixed audio chunk duration; default 4."},
    };
    out.session_options = {
        {"higgs_audio_stt.weight_type", "native|f32|f16|bf16|q8_0", "Shared text decoder weight storage type."},
        {"higgs_audio_stt.audio_encoder_weight_type", "native|f32|f16", "Audio encoder convolution weight storage type."},
        {"higgs_audio_stt.text_decoder_weight_type", "native|f32|f16|bf16|q8_0", "Text decoder matmul weight storage type."},
        {"higgs_audio_stt.audio_encoder_graph_arena_mb", "mb", "Audio encoder graph arena size."},
        {"higgs_audio_stt.text_decoder_prefill_graph_arena_mb", "mb", "Text decoder prefill graph arena size."},
        {"higgs_audio_stt.text_decoder_decode_graph_arena_mb", "mb", "Text decoder cached-step graph arena size."},
        {"higgs_audio_stt.text_decoder_weight_context_mb", "mb", "Text decoder weight context arena size."},
    };
    return out;
}

class HiggsAudioSTTLoader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return "higgs_audio_stt";
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
        const auto assets = load_higgs_audio_stt_assets(request.model_path);
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
        inspection.cli = cli(*assets);
        return inspection;
    }

    std::unique_ptr<runtime::ILoadedVoiceModel> load(const runtime::ModelLoadRequest & request) const override {
        return load_higgs_audio_stt_model(request.model_path);
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
    return std::make_unique<HiggsAudioSTTLoadedModel>(
        metadata(*assets),
        capabilities(*assets),
        std::move(assets));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_higgs_audio_stt_loader() {
    return std::make_shared<HiggsAudioSTTLoader>();
}

}  // namespace engine::models::higgs_audio_stt
