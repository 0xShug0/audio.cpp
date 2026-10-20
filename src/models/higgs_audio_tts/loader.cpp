#include "engine/models/higgs_audio_tts/loader.h"

#include "engine/framework/model_spec/package.h"
#include "engine/models/higgs_audio_tts/session.h"

#include <stdexcept>
#include <utility>

namespace engine::models::higgs_audio_tts {
namespace {

runtime::ModelMetadata metadata(const HiggsAssets &) {
    runtime::ModelMetadata out;
    out.family = "higgs_audio_tts";
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
        {runtime::VoiceTaskKind::Tts, {runtime::RunMode::Offline, runtime::RunMode::Streaming}},
        {runtime::VoiceTaskKind::VoiceCloning, {runtime::RunMode::Offline, runtime::RunMode::Streaming}},
    };
    out.supports_speaker_reference = true;
    out.languages = {"Auto"};
    return out;
}

runtime::ModelCliInterface cli(const HiggsAssets &) {
    runtime::ModelCliInterface out;
    out.request_options = {
        {"reference_text", "text", "Transcript for reference audio.", false, ""},
        {"max_tokens", "int", "Maximum generated AR tokens; zero uses the default. Must not exceed the configured context length.", false, "2048", "0"},
        {"allow_incomplete_generation", "bool", "Return generated audio with a warning instead of failing when max_tokens is reached before EOC. Audio may be truncated or contain silence/noise.", false, "false"},
        {"temperature", "float", "AR sampling temperature. Must be positive. Default: 0.8.", false, "0.8"},
        {"top_k", "int", "AR top-k sampling limit. Default: 30.", false, "30", "0"},
        {"top_p", "float", "AR nucleus sampling probability. Must be positive. Default: 0.8.", false, "0.8"},
        {"repetition_penalty", "float", "Accepted for Python API compatibility; audio sampling does not consume this value. Compatibility-only, but must be finite and positive. Default: 1.1.", false, "1.1"},
        {"seed", "n", "Torch RNG seed.", false, std::nullopt, "0"},
        {"text_chunk_size", "int", "Framework long-form text chunk size. Default: 1024.", false, "1024", "1"},
        {"text_chunk_mode", "default|tag_aware|japanese|endline", "Framework text chunking mode.", false, "default"},
        {"stream_frames_per_event", "n", "Streaming chunk frame cap; default 32.", false, "32", "1"},
        {"stream_chunk_policy", "grow|fixed", "Streaming chunk schedule; default grow.", false, "grow"},
    };
    out.session_options = {
        {"higgs_audio_tts.reference_prefill_mode", "auto|exact", "Cached-reference prefill attention; default auto preserves existing behavior. Try exact for excessive repetition or long, mostly silent continuations; may increase prefill time.", false, "auto"},
        {"higgs_audio_tts.weight_type", "native|f32|f16|bf16|q8_0", "AR and codec weight storage type.", false, "native"},
        {"higgs_audio_tts.ar_weight_type", "native|f32|f16|bf16|q8_0", "Autoregressive decoder weight storage type."},
        {"higgs_audio_tts.codec_weight_type", "native|f32|f16|bf16|q8_0", "Codec weight storage type."},
        {"higgs_audio_tts.ar_weight_context_mb", "n", "AR weight context in MiB; defaults to 4096 on 64-bit builds, 1024 on 32-bit builds.", false, std::nullopt, "1"},
        {"higgs_audio_tts.codec_weight_context_mb", "n", "Codec weight context size.", false, "1536", "1"},
        {"higgs_audio_tts.ar_decode_graph_arena_mb", "n", "AR decode graph arena size.", false, "512", "1"},
        {"higgs_audio_tts.codec_decode_graph_arena_mb", "n", "Codec decode graph arena size.", false, "128", "1"},
        {"higgs_audio_tts.codec_encode_graph_arena_mb", "n", "Codec encode graph arena size.", false, "256", "1"},
        {"higgs_audio_tts.reference_cache_slots", "n", "Encoded reference-audio cache slots; default 1.", false, "1", "0"},
        {"higgs_audio_tts.attention", "auto|flash|eager", "Attention lowering; auto probes the backend and falls back to eager on GPUs without a flash kernel (e.g. sm70); default auto.", false, "auto"},
    };
    return out;
}

class HiggsTTSLoader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return "higgs_audio_tts";
    }

    runtime::CapabilitySet advertised_capabilities() const override {
        runtime::CapabilitySet out;
        out.supported_tasks = {
            {runtime::VoiceTaskKind::Tts, {runtime::RunMode::Offline, runtime::RunMode::Streaming}},
            {runtime::VoiceTaskKind::VoiceCloning, {runtime::RunMode::Offline, runtime::RunMode::Streaming}},
        };
        out.supports_speaker_reference = true;
        return out;
    }

    bool can_load(const runtime::ModelLoadRequest & request) const override {
        if (request.family_hint.has_value() && *request.family_hint != family()) {
            return false;
        }
        try {
            const auto package_spec = engine::model_spec::default_spec_path(family());
            (void) engine::model_spec::load_resource_bundle(request.model_path, package_spec);
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
        const auto package_spec = engine::model_spec::default_spec_path(family());
        inspection.discovered_configs = runtime::discover_named_assets_from_package_spec(
            request.model_path,
            package_spec,
            engine::model_spec::ResourceKind::Files);
        inspection.discovered_weights = runtime::discover_named_assets_from_package_spec(
            request.model_path,
            package_spec,
            engine::model_spec::ResourceKind::Tensors);
        return inspection;
    }

    std::unique_ptr<runtime::ILoadedVoiceModel> load(const runtime::ModelLoadRequest & request) const override {
        return load_higgs_audio_tts_model(request.model_path);
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
    if (task.mode != runtime::RunMode::Offline && task.mode != runtime::RunMode::Streaming) {
        throw std::runtime_error("Higgs TTS supports offline and streaming sessions");
    }
    if (task.task != runtime::VoiceTaskKind::Tts &&
        task.task != runtime::VoiceTaskKind::VoiceCloning) {
        throw std::runtime_error("Higgs TTS supports Tts and VoiceCloning tasks");
    }
    return std::make_unique<HiggsTTSSession>(task, options, assets_);
}

std::unique_ptr<HiggsTTSLoadedModel> load_higgs_audio_tts_model(const std::filesystem::path & model_path) {
    auto assets = load_higgs_assets(model_path);
    return std::make_unique<HiggsTTSLoadedModel>(
        metadata(*assets),
        capabilities(*assets),
        std::move(assets));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_higgs_audio_tts_loader() {
    return std::make_shared<HiggsTTSLoader>();
}

}  // namespace engine::models::higgs_audio_tts
