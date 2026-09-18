#include "engine/models/vibevoice_asr/loader.h"

#include "engine/framework/model_spec/package.h"
#include "engine/models/vibevoice_asr/session.h"

#include <stdexcept>
#include <utility>

namespace engine::models::vibevoice_asr {
namespace {

runtime::CapabilitySet capabilities(const VibeVoiceASRAssets & assets) {
    runtime::CapabilitySet out;
    if (assets.family == "vibevoice_asr_streaming") {
        out.supported_tasks = {
            {runtime::VoiceTaskKind::Asr, {runtime::RunMode::Offline, runtime::RunMode::Streaming}},
        };
    } else {
        out.supported_tasks = {
            {runtime::VoiceTaskKind::Asr, {runtime::RunMode::Offline}},
        };
    }
    out.languages = {"auto"};
    out.supports_timestamps = true;
    return out;
}

runtime::ModelMetadata metadata(const VibeVoiceASRAssets & assets) {
    runtime::ModelMetadata out;
    out.family = assets.family;
    out.variant = assets.config.model_type.empty() ? "vibevoice-asr" : assets.config.model_type;
    out.description = "VibeVoice-ASR loaded from local assets.";
    return out;
}

runtime::ModelCliInterface cli(const VibeVoiceASRAssets &) {
    runtime::ModelCliInterface out;
    out.request_options = {
        {"language", "code", "ASR language label."},
        {"max_tokens", "n", "Maximum generated text tokens."},
        {"temperature", "float", "Sampling temperature; 0 uses deterministic decoding."},
        {"top_p", "float", "Nucleus sampling probability."},
        {"top_k", "n", "Top-k sampling limit; 0 disables top-k filtering."},
        {"num_beams", "n", "Beam count for deterministic beam search."},
        {"repetition_penalty", "float", "Generation repetition penalty."},
        {"seed", "n", "Acoustic latent sampling seed."},
        {"audio_chunk_mode", "auto|fixed|vad|none", "Audio chunking mode; default auto uses fixed chunks."},
        {"audio_chunk_duration_sec", "seconds", "Audio chunk duration; default 1200."},
    };
    out.session_options = {
        {"vibevoice_asr.weight_type", "native|f32|f16|bf16|q8_0", "Tokenizer, connector, and decoder weight storage type."},
        {"vibevoice_asr.tokenizer_weight_type", "native|f32|f16|bf16|q8_0", "Speech tokenizer weight storage type."},
        {"vibevoice_asr.connector_weight_type", "native|f32|f16|bf16|q8_0", "Acoustic and semantic connector weight storage type."},
        {"vibevoice_asr.decoder_weight_type", "native|f32|f16|bf16|q8_0", "Text decoder weight storage type."},
        {"vibevoice_asr.tokenizer_weight_context_mb", "mb", "Speech tokenizer weight context arena size."},
        {"vibevoice_asr.connector_weight_context_mb", "mb", "Acoustic and semantic connector weight context arena size."},
        {"vibevoice_asr.decoder_weight_context_mb", "mb", "Text decoder weight context arena size."},
        {"vibevoice_asr.vad_model_path", "path", "Silero VAD model path used by audio_chunk_mode=vad."},
        {"vibevoice_asr_streaming.weight_type", "native|f32|f16|bf16|q8_0", "Tokenizer, connector, and decoder weight storage type."},
        {"vibevoice_asr_streaming.tokenizer_weight_type", "native|f32|f16|bf16|q8_0", "Speech tokenizer weight storage type."},
        {"vibevoice_asr_streaming.connector_weight_type", "native|f32|f16|bf16|q8_0", "Acoustic and semantic connector weight storage type."},
        {"vibevoice_asr_streaming.decoder_weight_type", "native|f32|f16|bf16|q8_0", "Text decoder weight storage type."},
    };
    return out;
}

class VibeVoiceASRLoader final : public runtime::IVoiceModelLoader {
public:
    explicit VibeVoiceASRLoader(std::string family)
        : family_(std::move(family)) {}

    std::string family() const override {
        return family_;
    }

    runtime::CapabilitySet advertised_capabilities() const override {
        runtime::CapabilitySet out;
        if (family_ == "vibevoice_asr_streaming") {
            out.supported_tasks = {
                {runtime::VoiceTaskKind::Asr, {runtime::RunMode::Offline, runtime::RunMode::Streaming}},
            };
        } else {
            out.supported_tasks = {
                {runtime::VoiceTaskKind::Asr, {runtime::RunMode::Offline}},
            };
        }
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
        const auto assets = load_vibevoice_asr_assets(request.model_path, family_);
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
        return load_vibevoice_asr_model(request.model_path, family_);
    }

private:
    std::string family_;
};

}  // namespace

VibeVoiceASRLoadedModel::VibeVoiceASRLoadedModel(
    runtime::ModelMetadata metadata,
    runtime::CapabilitySet capabilities,
    std::shared_ptr<const VibeVoiceASRAssets> assets)
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
    if (metadata_.family != "vibevoice_asr_streaming" && task.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("VibeVoice-ASR streaming sessions are not supported");
    }
    return std::make_unique<VibeVoiceASRSession>(task, options, assets_);
}

std::unique_ptr<VibeVoiceASRLoadedModel> load_vibevoice_asr_model(
    const std::filesystem::path & model_path,
    const std::string & family) {
    auto assets = load_vibevoice_asr_assets(model_path, family);
    return std::make_unique<VibeVoiceASRLoadedModel>(
        metadata(*assets),
        capabilities(*assets),
        std::move(assets));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_vibevoice_asr_loader() {
    return std::make_shared<VibeVoiceASRLoader>("vibevoice_asr");
}

std::shared_ptr<runtime::IVoiceModelLoader> make_vibevoice_asr_streaming_loader() {
    return std::make_shared<VibeVoiceASRLoader>("vibevoice_asr_streaming");
}

}  // namespace engine::models::vibevoice_asr
