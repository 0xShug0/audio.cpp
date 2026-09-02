#include "engine/community_models/mira_tts/session.h"

#include "engine/community_models/mira_tts/decoder.h"
#include "engine/community_models/mira_tts/generator.h"
#include "engine/community_models/mira_tts/processor.h"
#include "engine/community_models/mira_tts/prompt.h"
#include "engine/community_models/mira_tts/speaker_encoder.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/spec_backed_model.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace engine::community_models::mira_tts {
namespace {

constexpr const char * kFamily = "mira_tts";
constexpr size_t kGraphBytes = 512ull * 1024ull * 1024ull;
constexpr size_t kWeightBytes = 256ull * 1024ull * 1024ull;

std::shared_ptr<const MiraTTSAssets> require_assets(
    std::shared_ptr<const MiraTTSAssets> value) {
    if (value == nullptr) throw std::runtime_error("MiraTTS session requires assets");
    return value;
}

std::shared_ptr<const engine::model_spec::ModelContract> require_contract(
    std::shared_ptr<const engine::model_spec::ModelContract> value) {
    if (value == nullptr) throw std::runtime_error("MiraTTS session requires a model contract");
    return value;
}

MiraGenerationOptions generation_options(const runtime::TaskRequest & request) {
    MiraGenerationOptions out;
    if (const auto value = runtime::parse_i64_option(request.options, {"max_tokens"})) {
        out.max_new_tokens = *value;
    }
    if (const auto value = runtime::parse_i64_option(request.options, {"top_k"})) {
        out.top_k = *value;
    }
    if (const auto value = runtime::parse_finite_float_option(request.options, {"top_p"})) {
        out.top_p = *value;
    }
    if (const auto value = runtime::parse_finite_float_option(request.options, {"min_p"})) {
        out.min_p = *value;
    }
    if (const auto value = runtime::parse_finite_float_option(request.options, {"temperature"})) {
        out.temperature = *value;
    }
    if (const auto value = runtime::parse_finite_float_option(
            request.options, {"repetition_penalty"})) {
        out.repetition_penalty = *value;
    }
    if (const auto value = runtime::parse_u64_option(request.options, {"seed"})) {
        out.seed = *value;
        out.has_seed = true;
    }
    if (!out.has_seed) out.seed = runtime::random_u64_seed();
    if (out.max_new_tokens < 1 || out.top_k < 1 || out.temperature <= 0.0F ||
        out.top_p <= 0.0F || out.top_p > 1.0F || out.min_p < 0.0F ||
        out.min_p > 1.0F || out.repetition_penalty < 1.0F) {
        throw std::runtime_error("MiraTTS generation options are outside their valid ranges");
    }
    return out;
}

std::unique_ptr<runtime::IVoiceTaskSession> create_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options,
    std::shared_ptr<const MiraTTSAssets> assets,
    std::shared_ptr<const engine::model_spec::ModelContract> contract) {
    return std::make_unique<MiraTTSOfflineSession>(
        task, options, std::move(assets), std::move(contract));
}

}  // namespace

MiraTTSOfflineSession::MiraTTSOfflineSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const MiraTTSAssets> assets,
    std::shared_ptr<const engine::model_spec::ModelContract> contract)
    : RuntimeSessionBase(options),
      task_(task),
      assets_(require_assets(std::move(assets))),
      contract_(require_contract(std::move(contract))) {
    runtime::validate_spec_backed_session_options(
        options, *contract_, kFamily, "MiraTTS");
    if (task.mode != runtime::RunMode::Offline ||
        (task.task != runtime::VoiceTaskKind::Tts &&
         task.task != runtime::VoiceTaskKind::VoiceCloning)) {
        throw std::runtime_error("MiraTTS supports offline TTS/voice cloning only");
    }
    const auto lm_type = runtime::parse_tensor_storage_option(
        options.options, "backbone_weight_type", assets::TensorStorageType::Native,
        {assets::TensorStorageType::Native, assets::TensorStorageType::F32,
         assets::TensorStorageType::F16, assets::TensorStorageType::BF16,
         assets::TensorStorageType::Q8_0});
    const auto linear_type = runtime::parse_tensor_storage_option(
        options.options, "linear_weight_type", assets::TensorStorageType::Native,
        {assets::TensorStorageType::Native, assets::TensorStorageType::F32,
         assets::TensorStorageType::F16, assets::TensorStorageType::BF16,
         assets::TensorStorageType::Q8_0});
    const auto conv_type = runtime::parse_tensor_storage_option(
        options.options, "conv_weight_type", assets::TensorStorageType::F32,
        {assets::TensorStorageType::Native, assets::TensorStorageType::F32,
         assets::TensorStorageType::F16, assets::TensorStorageType::BF16});
    auto & execution = execution_context();
    prompt_ = std::make_unique<MiraPromptBuilder>(assets_);
    speaker_encoder_ = std::make_unique<MiraSpeakerEncoder>(
        *assets_, execution, kWeightBytes, kGraphBytes, linear_type, conv_type);
    generator_ = std::make_unique<MiraGenerator>(
        *assets_, execution, kGraphBytes, kGraphBytes, kWeightBytes, lm_type);
    processor_ = std::make_unique<MiraAcousticProcessor>(
        *assets_, execution, kWeightBytes, kGraphBytes, linear_type, conv_type);
    // ggml's current CUDA ConvTranspose1d kernel requires F32 weights.
    decoder_ = std::make_unique<MiraDecoder>(
        *assets_, execution, kWeightBytes, kGraphBytes,
        assets::TensorStorageType::F32);
}

MiraTTSOfflineSession::~MiraTTSOfflineSession() = default;

std::string MiraTTSOfflineSession::family() const { return kFamily; }

runtime::VoiceTaskKind MiraTTSOfflineSession::task_kind() const {
    return task_.task;
}

runtime::RunMode MiraTTSOfflineSession::run_mode() const { return task_.mode; }

void MiraTTSOfflineSession::prepare(
    const runtime::SessionPreparationRequest & request) {
    runtime::validate_spec_backed_request_options(
        request.options, *contract_, "MiraTTS");
    prepared_reference_.reset();
    if (request.voice.has_value() && request.voice->speaker.has_value() &&
        request.voice->speaker->audio.has_value()) {
        prepared_reference_ = *request.voice->speaker->audio;
    }
    mark_prepared();
}

const runtime::AudioBuffer & MiraTTSOfflineSession::reference_audio(
    const runtime::TaskRequest & request) const {
    if (request.voice.has_value() && request.voice->speaker.has_value() &&
        request.voice->speaker->audio.has_value()) {
        return *request.voice->speaker->audio;
    }
    if (request.audio_input.has_value()) return *request.audio_input;
    if (prepared_reference_.has_value()) return *prepared_reference_;
    throw std::runtime_error(
        "MiraTTS requires a reference voice in voice.speaker.audio or audio_input");
}

runtime::TaskResult MiraTTSOfflineSession::run(
    const runtime::TaskRequest & request) {
    require_prepared("MiraTTS run");
    runtime::validate_spec_backed_request_options(
        request.options, *contract_, "MiraTTS");
    if (!request.text_input.has_value() || request.text_input->text.empty()) {
        throw std::runtime_error("MiraTTS requires non-empty text input");
    }
    const auto context_codes = speaker_encoder_->encode(reference_audio(request));
    const auto prompt_ids = prompt_->build(request.text_input->text, context_codes);
    const auto speech_codes = generator_->generate(
        prompt_ids, generation_options(request));
    if (speech_codes.empty()) {
        throw std::runtime_error("MiraTTS generated no speech tokens");
    }
    const auto latents = processor_->process(speech_codes, context_codes);
    auto audio = decoder_->decode(
        latents, static_cast<int64_t>(speech_codes.size()));
    runtime::TaskResult result;
    result.audio_output = std::move(audio);
    return result;
}

std::shared_ptr<runtime::IVoiceModelLoader> make_mira_tts_loader() {
    runtime::SpecBackedVoiceModelConfig<MiraTTSAssets> config;
    config.family = kFamily;
    config.aliases = {"mira", "MiraTTS"};
    config.load_assets = load_mira_tts_assets;
    config.create_session = create_session;
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::community_models::mira_tts
