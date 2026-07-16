#include "engine/models/vevo2/loader.h"

#include "engine/framework/assets/model_package.h"
#include "engine/models/vevo2/assets.h"
#include "engine/models/vevo2/session.h"

#include <filesystem>
#include <optional>
#include <stdexcept>
#include <utility>

namespace engine::models::vevo2 {
namespace {

runtime::ModelMetadata metadata(const Vevo2Assets &) {
    runtime::ModelMetadata out;
    out.family = "vevo2";
    out.variant = "RMSnow/Vevo2";
    out.description = "Vevo2 singing voice conversion draft loaded from local extracted assets.";
    return out;
}

runtime::CapabilitySet capabilities(const Vevo2Assets &) {
    runtime::CapabilitySet out;
    out.supported_tasks = {
        {runtime::VoiceTaskKind::Tts, {runtime::RunMode::Offline}},
        {runtime::VoiceTaskKind::VoiceConversion, {runtime::RunMode::Offline}},
        {runtime::VoiceTaskKind::SpeechToSpeech, {runtime::RunMode::Offline}},
        {runtime::VoiceTaskKind::Svc, {runtime::RunMode::Offline}},
    };
    out.languages = {"en", "zh"};
    out.supports_speaker_reference = true;
    out.supports_style_condition = true;
    return out;
}

runtime::ModelCliInterface cli(const Vevo2Assets &) {
    runtime::ModelCliInterface out;
    out.request_options = {
        {
            "route",
            "route",
            "VeVo2 route: zero_shot_tts, text_to_singing, svs, style_preserved_vc, "
            "style_preserved_svc, style_converted_vc, style_converted_svc, editing, "
            "singing_style_conversion, humming_to_singing, or instrument_to_singing",
        },
        {"source_audio", "wav", "Source speech or singing audio for conversion/editing routes."},
        {"target_voice", "wav", "Target timbre reference audio."},
        {"prosody_ref", "wav", "Prosody or melody reference audio."},
        {"style_ref", "wav", "Style reference audio."},
        {"target_text", "text", "Text or lyrics to vocalize."},
        {"style_ref_text", "text", "Transcript for the style reference when text conditioning is used."},
        {"use_prosody_code", "true|false", "Enable explicit prosody-code conditioning."},
        {"predict_target_prosody", "true|false", "Predict target prosody during AR generation."},
        {"use_pitch_shift", "true|false", "Pitch-align source/prosody/style references to the target voice."},
        {"source_shift_steps", "n", "Manual pitch-shift semitone steps for source audio."},
        {"prosody_shift_steps", "n", "Manual pitch-shift semitone steps for prosody reference audio."},
        {"style_shift_steps", "n", "Manual pitch-shift semitone steps for style reference audio."},
        {"target_duration_seconds", "seconds", "Target output duration hint for flow matching."},
        {"reference_duration_seconds", "seconds", "Trim target voice reference duration before conditioning."},
        {"temperature", "value", "AR sampling temperature."},
        {"top_k", "n", "AR top-k sampling limit."},
        {"top_p", "value", "AR nucleus sampling threshold."},
        {"repetition_penalty", "value", "AR repetition penalty."},
        {"max_tokens", "n", "Maximum generated AR content/style tokens."},
        {"num_inference_steps", "n", "Flow-matching denoising steps."},
        {"seed", "n", "Request seed; omitted means random."},
        {"fm_noise_file", "path", "Optional flow-matching noise tensor file."},
    };
    out.session_options = {
        {"vevo2.weight_type", "native|f32|f16|bf16|q8_0", "Default matmul weight storage type."},
        {"vevo2.conv_weight_type", "native|f32|f16", "Default convolution weight storage type."},
        {"vevo2.ar_weight_type", "native|f32|f16|bf16|q8_0", "AR matmul weight storage type."},
        {"vevo2.ar_weight_context_mb", "n", "AR weight context size."},
        {"vevo2.ar_prefill_graph_context_mb", "n", "AR prefill graph context size."},
        {"vevo2.ar_decode_graph_context_mb", "n", "AR decode graph context size."},
        {"vevo2.whisper_weight_type", "native|f32|f16|bf16|q8_0", "Whisper matmul weight storage type."},
        {"vevo2.whisper_conv_weight_type", "native|f32|f16", "Whisper convolution weight storage type."},
        {"vevo2.whisper_weight_context_mb", "n", "Whisper weight context size."},
        {"vevo2.whisper_graph_context_mb", "n", "Whisper graph context size."},
        {"vevo2.tokenizer_weight_type", "native|f32|f16|bf16|q8_0", "Tokenizer matmul weight storage type."},
        {"vevo2.tokenizer_conv_weight_type", "native|f32|f16", "Tokenizer convolution weight storage type."},
        {"vevo2.tokenizer_weight_context_mb", "n", "Tokenizer weight context size."},
        {"vevo2.tokenizer_graph_context_mb", "n", "Tokenizer graph context size."},
        {"vevo2.fm_weight_type", "native|f32|f16|bf16|q8_0", "Flow-matching matmul weight storage type."},
        {"vevo2.fm_conv_weight_type", "native|f32|f16", "Flow-matching convolution weight storage type."},
        {"vevo2.fm_weight_context_mb", "n", "Flow-matching weight context size."},
        {"vevo2.fm_graph_context_mb", "n", "Flow-matching graph context size."},
        {"vevo2.vocoder_weight_type", "native|f32|f16|bf16|q8_0", "Vocoder matmul weight storage type."},
        {"vevo2.vocoder_conv_weight_type", "native|f32|f16", "Vocoder convolution weight storage type."},
        {"vevo2.vocoder_weight_context_mb", "n", "Vocoder weight context size."},
        {"vevo2.vocoder_graph_context_mb", "n", "Vocoder graph context size."},
    };
    out.load_options = {
        {"vevo2.whisper_model_path", "dir", "Local Whisper model directory used by VeVo2 feature extraction."},
    };
    return out;
}

class Vevo2LoadedModel final : public runtime::ILoadedVoiceModel {
public:
    Vevo2LoadedModel(
        runtime::ModelMetadata metadata,
        runtime::CapabilitySet capabilities,
        std::shared_ptr<const Vevo2Assets> assets)
        : metadata_(std::move(metadata)),
          capabilities_(std::move(capabilities)),
          assets_(std::move(assets)) {
        if (assets_ == nullptr) {
            throw std::runtime_error("Vevo2 loaded model requires assets");
        }
    }

    const runtime::ModelMetadata & metadata() const noexcept override {
        return metadata_;
    }

    const runtime::CapabilitySet & capabilities() const noexcept override {
        return capabilities_;
    }

    std::unique_ptr<runtime::IVoiceTaskSession> create_task_session(
        const runtime::TaskSpec & task,
        const runtime::SessionOptions & options) const override {
        if (task.task != runtime::VoiceTaskKind::Tts &&
            task.task != runtime::VoiceTaskKind::VoiceConversion &&
            task.task != runtime::VoiceTaskKind::SpeechToSpeech &&
            task.task != runtime::VoiceTaskKind::Svc) {
            throw std::runtime_error("Vevo2 supports tts, vc, s2s, and svc tasks");
        }
        if (task.mode != runtime::RunMode::Offline) {
            throw std::runtime_error("Vevo2 currently supports offline sessions");
        }
        return std::make_unique<Vevo2Session>(task, options, assets_);
    }

private:
    runtime::ModelMetadata metadata_;
    runtime::CapabilitySet capabilities_;
    std::shared_ptr<const Vevo2Assets> assets_;
};

class Vevo2Loader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return "vevo2";
    }

    bool can_load(const runtime::ModelLoadRequest & request) const override {
        if (request.family_hint.has_value() && *request.family_hint != family()) {
            return false;
        }
        try {
            (void)engine::assets::load_resource_bundle_from_package_spec(
                request.model_path,
                engine::assets::default_model_package_spec_path(family()));
            return true;
        } catch (...) {
            return false;
        }
    }

    runtime::ModelInspection inspect(const runtime::ModelLoadRequest & request) const override {
        const auto whisper_path = request.options.find("vevo2.whisper_model_path");
        const auto assets = load_vevo2_assets(
            request.model_path,
            whisper_path == request.options.end() || whisper_path->second.empty()
                ? std::nullopt
                : std::make_optional(std::filesystem::path(whisper_path->second)));
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
        return load_vevo2_model(request);
    }
};

}  // namespace

std::unique_ptr<runtime::ILoadedVoiceModel> load_vevo2_model(const runtime::ModelLoadRequest & request) {
    const auto whisper_path = request.options.find("vevo2.whisper_model_path");
    auto assets = load_vevo2_assets(
        request.model_path,
        whisper_path == request.options.end() || whisper_path->second.empty()
            ? std::nullopt
            : std::make_optional(std::filesystem::path(whisper_path->second)));
    return std::make_unique<Vevo2LoadedModel>(
        metadata(*assets),
        capabilities(*assets),
        std::move(assets));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_vevo2_loader() {
    return std::make_shared<Vevo2Loader>();
}

}  // namespace engine::models::vevo2
