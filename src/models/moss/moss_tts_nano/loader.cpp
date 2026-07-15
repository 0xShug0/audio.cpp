#include "engine/models/moss/moss_tts_nano/loader.h"

#include "engine/framework/assets/model_package.h"
#include "engine/models/moss/moss_tts_nano/session.h"

#include <stdexcept>
#include <utility>

namespace engine::models::moss_tts_nano {
namespace {

class MossTTSNanoLoader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return "moss_tts_nano";
    }

    bool can_load(const runtime::ModelLoadRequest & request) const override {
        try {
            (void) engine::assets::load_resource_bundle_from_package_spec(
                request.model_path,
                engine::assets::default_model_package_spec_path(family()));
            return !request.family_hint.has_value() || *request.family_hint == family();
        } catch (...) {
            return false;
        }
    }

    runtime::ModelInspection inspect(const runtime::ModelLoadRequest & request) const override {
        const auto assets = load_moss_tts_nano_assets(request.model_path);
        runtime::ModelInspection inspection;
        inspection.model_root = assets->resources.model_root();
        inspection.metadata.family = family();
        inspection.metadata.variant = assets->config.model_type;
        inspection.metadata.description = "MOSS-TTS-Nano loaded from local extracted assets.";
        inspection.capabilities.supported_tasks = {
            {runtime::VoiceTaskKind::Tts, {runtime::RunMode::Offline}},
            {runtime::VoiceTaskKind::VoiceCloning, {runtime::RunMode::Offline}},
        };
        inspection.capabilities.supports_speaker_reference = true;
        inspection.capabilities.languages = {"Auto"};
        inspection.cli.request_options = {
            {"max_tokens", "n", "Maximum generated audio frames."},
            {"do_sample", "true|false", "Enable stochastic audio-token sampling."},
            {"temperature", "float", "Audio-token sampling temperature."},
            {"top_p", "float", "Audio-token nucleus sampling limit."},
            {"top_k", "n", "Audio-token top-k sampling limit."},
            {"repetition_penalty", "float", "Audio-token repetition penalty."},
            {"text_temperature", "float", "Text-gate sampling temperature."},
            {"text_top_p", "float", "Text-gate nucleus sampling limit."},
            {"text_top_k", "n", "Text-gate top-k sampling limit."},
            {"text_chunk_mode", "default|tag_aware|japanese|endline", "Long-form text chunking mode; default splitter."},
        };
        inspection.cli.session_options = {
            {"moss_tts_nano.weight_type", "native|f32|f16|bf16|q8_0", "Global and local-frame weight storage type."},
            {"moss_tts_nano.global_weight_type", "native|f32|f16|bf16|q8_0", "Global transformer weight storage type."},
            {"moss_tts_nano.local_frame_weight_type", "native|f32|f16|bf16|q8_0", "Local frame decoder weight storage type."},
            {"moss_tts_nano.global_prefill_graph_arena_mb", "n", "Global transformer prefill graph arena size."},
            {"moss_tts_nano.global_decode_graph_arena_mb", "n", "Global transformer decode graph arena size."},
            {"moss_tts_nano.global_weight_context_mb", "n", "Global transformer weight context size."},
            {"moss_tts_nano.local_frame_graph_arena_mb", "n", "Local frame decoder graph arena size."},
            {"moss_tts_nano.local_frame_weight_context_mb", "n", "Local frame decoder weight context size."},
            {"moss_tts_nano.audio_tokenizer_encoder_graph_arena_mb", "n", "Audio tokenizer encoder graph arena size."},
            {"moss_tts_nano.audio_tokenizer_decoder_graph_arena_mb", "n", "Audio tokenizer decoder graph arena size."},
            {"moss_tts_nano.audio_tokenizer_weight_context_mb", "n", "Audio tokenizer weight context size."},
        };
        const auto spec_path = engine::assets::default_model_package_spec_path(family());
        inspection.discovered_configs = runtime::discover_named_assets_from_package_spec(
            request.model_path,
            spec_path,
            engine::assets::ModelPackageResourceKind::Files);
        inspection.discovered_weights = runtime::discover_named_assets_from_package_spec(
            request.model_path,
            spec_path,
            engine::assets::ModelPackageResourceKind::Tensors);
        return inspection;
    }

    std::unique_ptr<runtime::ILoadedVoiceModel> load(const runtime::ModelLoadRequest & request) const override {
        return load_moss_tts_nano_model(request.model_path);
    }
};

}  // namespace

MossTTSNanoLoadedModel::MossTTSNanoLoadedModel(
    runtime::ModelMetadata metadata,
    runtime::CapabilitySet capabilities,
    std::shared_ptr<const MossTTSNanoAssets> assets)
    : metadata_(std::move(metadata)),
      capabilities_(std::move(capabilities)),
      assets_(std::move(assets)) {}

const runtime::ModelMetadata & MossTTSNanoLoadedModel::metadata() const noexcept {
    return metadata_;
}

const runtime::CapabilitySet & MossTTSNanoLoadedModel::capabilities() const noexcept {
    return capabilities_;
}

std::unique_ptr<runtime::IVoiceTaskSession> MossTTSNanoLoadedModel::create_task_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options) const {
    if (task.task != runtime::VoiceTaskKind::Tts && task.task != runtime::VoiceTaskKind::VoiceCloning) {
        throw std::runtime_error("MOSS-TTS-Nano only supports the Tts and VoiceCloning tasks");
    }
    if (task.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("MOSS-TTS-Nano only supports offline sessions");
    }
    return std::make_unique<MossTTSNanoSession>(task, options, assets_);
}

std::unique_ptr<MossTTSNanoLoadedModel> load_moss_tts_nano_model(const std::filesystem::path & model_path) {
    auto assets = load_moss_tts_nano_assets(model_path);

    runtime::ModelMetadata metadata;
    metadata.family = "moss_tts_nano";
    metadata.variant = assets->config.model_type;
    metadata.description = "MOSS-TTS-Nano loaded from local extracted assets.";

    runtime::CapabilitySet capabilities;
    capabilities.supported_tasks = {
        {runtime::VoiceTaskKind::Tts, {runtime::RunMode::Offline}},
        {runtime::VoiceTaskKind::VoiceCloning, {runtime::RunMode::Offline}},
    };
    capabilities.supports_speaker_reference = true;
    capabilities.languages = {"Auto"};
    return std::make_unique<MossTTSNanoLoadedModel>(std::move(metadata), std::move(capabilities), std::move(assets));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_moss_tts_nano_loader() {
    return std::make_shared<MossTTSNanoLoader>();
}

}  // namespace engine::models::moss_tts_nano
