#include "engine/models/firered_audio/session.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/spec_backed_model.h"
#include "engine/models/firered_audio/pipeline.h"
#include "engine/models/firered_audio/redae.h"
#include "engine/models/firered_audio/understanding.h"

#include <stdexcept>
#include <utility>

namespace engine::models::firered_audio {
namespace {

constexpr const char * kFamily = "firered_audio";
constexpr const char * kModelName = "FireRedAudio";
constexpr size_t kDefaultReferenceCacheSlots = 2;

std::shared_ptr<const FireRedAudioAssets> require_assets(std::shared_ptr<const FireRedAudioAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("FireRedAudio session requires assets");
    }
    return assets;
}

std::shared_ptr<const engine::model_spec::ModelContract> require_contract(
    std::shared_ptr<const engine::model_spec::ModelContract> contract) {
    if (contract == nullptr) {
        throw std::runtime_error("FireRedAudio session requires a model contract");
    }
    return contract;
}

std::string request_language(const runtime::TaskRequest & request) {
    if (const auto language = runtime::find_option(request.options, {"language"})) {
        return *language;
    }
    if (request.text_input.has_value() && !request.text_input->language.empty()) {
        return request.text_input->language;
    }
    return "zh";
}

FireRedAudioGenerationRequest make_generation_request(
    const FireRedAudioAssets & assets,
    const FireRedAudioTokenizer & tokenizer,
    const runtime::TaskSpec & task,
    const runtime::TaskRequest & request) {
    FireRedAudioGenerationRequest out;
    const std::string mode = runtime::find_option(request.options, {"template_name"}).value_or(
        task.task == runtime::VoiceTaskKind::VoiceDesign ? "voice_design" :
            (request.audio_input.has_value() ? "acoustic_edit" : "tts_clone"));
    if (mode == "tts_clone") {
        if (!request.text_input.has_value() || request.text_input->text.empty()) {
            throw std::runtime_error("FireRedAudio clone requires text input");
        }
        if (!request.voice.has_value() ||
            !request.voice->speaker.has_value() ||
            !request.voice->speaker->audio.has_value()) {
            throw std::runtime_error("FireRedAudio clone requires reference audio");
        }
        out.prompt_audio = *request.voice->speaker->audio;
        out.prompt_audio_is_assistant = true;
        const auto audio_24k = prepare_firered_prompt_audio_24k(*out.prompt_audio, assets.redae, assets.patch_encoder.patch_size);
        const int64_t placeholder_count =
            static_cast<int64_t>(audio_24k.size()) /
            (assets.redae.audio_patch_size * assets.redae.enc_extra_downsample_rate * assets.patch_encoder.patch_size);
        const std::string language = request_language(request);
        const auto reference_text = runtime::find_option(request.options, {"reference_text"}).value_or("");
        auto encoded = tokenizer.encode_tts_clone_prompt(
            reference_text,
            request.text_input->text,
            language,
            placeholder_count);
        out.token_ids = std::move(encoded.token_ids);
        out.audio_no_latent_mask = std::move(encoded.audio_no_latent_mask);
        out.generated_text_language = language;
    } else if (mode == "voice_design") {
        if (!request.text_input.has_value() || request.text_input->text.empty()) {
            throw std::runtime_error("FireRedAudio voice design requires text input");
        }
        const auto instruction = runtime::find_option(request.options, {"instruction"}).value_or("");
        if (instruction.empty()) {
            throw std::runtime_error("FireRedAudio voice design requires instruction");
        }
        auto encoded = tokenizer.encode_voice_design_prompt(instruction, request.text_input->text);
        out.token_ids = std::move(encoded.token_ids);
        out.audio_no_latent_mask = std::move(encoded.audio_no_latent_mask);
        out.generated_text_language = request_language(request);
    } else if (mode == "semantic_edit" || mode == "acoustic_edit") {
        if (!request.audio_input.has_value()) {
            throw std::runtime_error("FireRedAudio edit requires input audio");
        }
        const auto instruction = runtime::find_option(request.options, {"instruction"}).value_or("");
        if (instruction.empty()) {
            throw std::runtime_error("FireRedAudio edit requires instruction");
        }
        out.prompt_audio = *request.audio_input;
        out.prompt_audio_is_assistant = false;
        const auto audio_24k = prepare_firered_prompt_audio_24k(*out.prompt_audio, assets.redae, assets.patch_encoder.patch_size);
        const int64_t placeholder_count =
            static_cast<int64_t>(audio_24k.size()) /
            (assets.redae.audio_patch_size * assets.redae.enc_extra_downsample_rate * assets.patch_encoder.patch_size);
        auto encoded = tokenizer.encode_edit_prompt(
            instruction,
            mode == "semantic_edit" ? "semantic" : "acoustic",
            placeholder_count);
        out.token_ids = std::move(encoded.token_ids);
        out.audio_no_latent_mask = std::move(encoded.audio_no_latent_mask);
        out.generated_text_language = request_language(request);
    } else {
        throw std::runtime_error("unknown FireRedAudio template_name: " + mode);
    }
    out.seed = runtime::parse_u32_option(request.options, {"seed"}).value_or(out.seed);
    out.num_inference_steps = runtime::parse_positive_i64_option(
        request.options,
        {"num_inference_steps"},
        out.num_inference_steps);
    if (const auto cfg = runtime::parse_positive_finite_float_option(request.options, {"guidance_scale"})) {
        out.guidance_scale = *cfg;
    }
    out.max_new_audio_steps = runtime::parse_positive_i64_option(
        request.options,
        {"max_new_audio_steps"},
        out.max_new_audio_steps);
    out.min_new_audio_steps = runtime::parse_i64_option(request.options, {"min_new_audio_steps"}).value_or(out.min_new_audio_steps);
    if (out.min_new_audio_steps < 0) {
        throw std::runtime_error("FireRedAudio min_new_audio_steps must be non-negative");
    }
    out.max_new_text_tokens = runtime::parse_positive_i64_option(
        request.options,
        {"max_new_text_tokens"},
        out.max_new_text_tokens);
    return out;
}

FireRedAudioUnderstandingRequest make_understanding_request(
    const runtime::TaskRequest & request) {
    if (!request.audio_input.has_value()) {
        throw std::runtime_error("FireRedAudio ASR/understanding requires input audio");
    }
    FireRedAudioUnderstandingRequest out;
    out.audio = *request.audio_input;
    out.prompt = request.text_input.has_value() && !request.text_input->text.empty()
        ? request.text_input->text
        : "Transcribe speech to text.";
    out.language = request_language(request);
    out.seed = runtime::parse_u32_option(request.options, {"seed"}).value_or(out.seed);
    out.max_new_tokens = runtime::parse_positive_i64_option(
        request.options,
        {"max_new_tokens"},
        out.max_new_tokens);
    out.enable_thinking = runtime::parse_bool_option(
        runtime::find_option(request.options, {"enable_thinking"}).value_or("false"),
        "enable_thinking");
    const std::string mode = runtime::find_option(request.options, {"template_name"}).value_or("asr");
    if (mode == "understand") {
        out.do_sample = true;
        out.repetition_penalty = 1.0F;
        if (out.enable_thinking && !runtime::find_option(request.options, {"max_new_tokens"}).has_value()) {
            out.max_new_tokens = 1024;
        }
    } else if (mode == "asr") {
        out.do_sample = false;
        out.repetition_penalty = 1.1F;
        if (out.enable_thinking) {
            throw std::runtime_error("FireRedAudio ASR does not support enable_thinking");
        }
    } else {
        throw std::runtime_error("unknown FireRedAudio ASR template_name: " + mode);
    }
    if (const auto top_k = runtime::parse_i64_option(request.options, {"top_k"})) {
        out.top_k = *top_k;
    }
    if (const auto top_p = runtime::parse_finite_float_option(request.options, {"top_p"})) {
        out.top_p = *top_p;
    }
    if (const auto temperature = runtime::parse_finite_float_option(request.options, {"temperature"})) {
        out.temperature = *temperature;
    }
    return out;
}

std::unique_ptr<runtime::IVoiceTaskSession> create_firered_audio_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options,
    std::shared_ptr<const FireRedAudioAssets> assets,
    std::shared_ptr<const engine::model_spec::ModelContract> contract) {
    return std::make_unique<FireRedAudioSession>(
        task,
        options,
        std::move(assets),
        std::move(contract));
}

}  // namespace

FireRedAudioSession::FireRedAudioSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const FireRedAudioAssets> assets,
    std::shared_ptr<const engine::model_spec::ModelContract> contract)
    : runtime::RuntimeSessionBase(options),
      task_(task),
      assets_(require_assets(std::move(assets))),
      contract_(require_contract(std::move(contract))),
      tokenizer_(std::make_unique<FireRedAudioTokenizer>(assets_)) {
    runtime::validate_spec_backed_session_options(options, *contract_, kFamily, kModelName);
    if (task_.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("FireRedAudio supports offline sessions");
    }
    if (task_.task != runtime::VoiceTaskKind::Tts &&
        task_.task != runtime::VoiceTaskKind::VoiceCloning &&
        task_.task != runtime::VoiceTaskKind::VoiceDesign &&
        task_.task != runtime::VoiceTaskKind::Asr) {
        throw std::runtime_error("FireRedAudio supports tts, clone, voice design, and asr tasks");
    }
    using T = engine::assets::TensorStorageType;
    const auto storage_type = runtime::parse_tensor_storage_option(
        options.options,
        "firered_audio.weight_type",
        T::Native,
        {T::Native, T::F32, T::F16, T::BF16, T::Q8_0});
    const auto graph_arena_bytes = runtime::parse_size_mb_option(
        options.options,
        {"firered_audio.graph_arena_mb"},
        1024ull * 1024ull * 1024ull);
    const auto weight_context_bytes = runtime::parse_size_mb_option(
        options.options,
        {"firered_audio.weight_context_mb"},
        1024ull * 1024ull * 1024ull);
    if (const auto mem_saver = runtime::find_option(options.options, {"firered_audio.mem_saver"})) {
        mem_saver_ = runtime::parse_bool_option(*mem_saver, "firered_audio.mem_saver");
    }
    const int64_t reference_cache_slots = runtime::parse_i64_option(
        options.options,
        {"firered_audio.reference_cache_slots"})
        .value_or(static_cast<int64_t>(kDefaultReferenceCacheSlots));
    if (reference_cache_slots < 0) {
        throw std::runtime_error("firered_audio.reference_cache_slots must be non-negative");
    }
    if (task_.task == runtime::VoiceTaskKind::Asr) {
        understanding_runtime_ = std::make_unique<FireRedAudioUnderstandingRuntime>(
            assets_,
            execution_context(),
            graph_arena_bytes,
            weight_context_bytes,
            storage_type,
            mem_saver_);
    } else {
        runtime_ = std::make_unique<FireRedAudioGenerationRuntime>(
            assets_,
            execution_context(),
            graph_arena_bytes,
            weight_context_bytes,
            storage_type,
            static_cast<size_t>(reference_cache_slots),
            mem_saver_);
    }
}

FireRedAudioSession::~FireRedAudioSession() = default;

std::string FireRedAudioSession::family() const {
    return kFamily;
}

runtime::VoiceTaskKind FireRedAudioSession::task_kind() const {
    return task_.task;
}

runtime::RunMode FireRedAudioSession::run_mode() const {
    return task_.mode;
}

void FireRedAudioSession::prepare(const runtime::SessionPreparationRequest & request) {
    runtime::validate_spec_backed_request_options(
        request.options,
        *contract_,
        kModelName);
    mark_prepared();
}

runtime::TaskResult FireRedAudioSession::run(const runtime::TaskRequest & request) {
    const auto wall_start = std::chrono::steady_clock::now();
    runtime::validate_spec_backed_request_options(
        request.options,
        *contract_,
        kModelName);
    require_prepared("FireRedAudio run");
    runtime::TaskResult result;
    if (task_.task == runtime::VoiceTaskKind::Asr) {
        auto parsed = make_understanding_request(request);
        auto generated = understanding_runtime_->generate(parsed);
        result.text_output = runtime::Transcript{std::move(generated.text), generated.language};
    } else {
        auto parsed = make_generation_request(*assets_, *tokenizer_, task_, request);
        auto generated = runtime_->generate(parsed);
        result.audio_output = std::move(generated.audio);
        if (!generated.generated_text.empty()) {
            result.text_output = runtime::Transcript{std::move(generated.generated_text), parsed.generated_text_language};
        }
    }
    engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start));
    return result;
}

std::shared_ptr<runtime::IVoiceModelLoader> make_firered_audio_loader() {
    runtime::SpecBackedVoiceModelConfig<FireRedAudioAssets> config;
    config.family = kFamily;
    config.load_assets = load_firered_audio_assets;
    config.create_session = create_firered_audio_session;
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::models::firered_audio
