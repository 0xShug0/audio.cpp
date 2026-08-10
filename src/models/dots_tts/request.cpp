#include "engine/models/dots_tts/request.h"

#include "engine/framework/runtime/options.h"
#include "engine/framework/text/chunking.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace engine::models::dots_tts {
namespace {

DotsTemplateName parse_template_name(const std::string & value) {
    if (value == "tts") {
        return DotsTemplateName::Tts;
    }
    if (value == "instruction_tts") {
        return DotsTemplateName::InstructionTts;
    }
    if (value == "text_to_audio") {
        return DotsTemplateName::TextToAudio;
    }
    if (value == "tts_interleave") {
        return DotsTemplateName::TtsInterleave;
    }
    throw std::runtime_error("DotTTS template_name must be tts, instruction_tts, text_to_audio, or tts_interleave");
}

DotsOdeMethod parse_sampler_mode(const std::string & value) {
    if (value == "euler") {
        return DotsOdeMethod::Euler;
    }
    if (value == "midpoint") {
        return DotsOdeMethod::Midpoint;
    }
    if (value == "rk4") {
        return DotsOdeMethod::Rk4;
    }
    throw std::runtime_error("DotTTS sampler_mode must be euler, midpoint, or rk4");
}

DotsGenerationOptions generation_options(
    const DotsConfig & config,
    const std::unordered_map<std::string, std::string> & options,
    DotsGenerationOptions defaults) {
    (void)config;
    if (const auto value = runtime::find_option(options, {"template_name"})) {
        defaults.template_name = parse_template_name(*value);
    }
    defaults.language = runtime::find_option(options, {"language"}).value_or(defaults.language);
    defaults.num_inference_steps =
        runtime::parse_i64_option(options, {"num_inference_steps"}).value_or(defaults.num_inference_steps);
    defaults.guidance_scale =
        runtime::parse_float_option(options, {"guidance_scale"}).value_or(defaults.guidance_scale);
    defaults.speaker_scale =
        runtime::parse_float_option(options, {"speaker_scale"}).value_or(defaults.speaker_scale);
    if (const auto value = runtime::find_option(options, {"sampler_mode"})) {
        defaults.ode_method = parse_sampler_mode(*value);
    }
    defaults.max_tokens = runtime::parse_i64_option(options, {"max_tokens"}).value_or(defaults.max_tokens);
    defaults.text_chunk_size =
        engine::text::parse_text_chunk_size_override(options).value_or(defaults.text_chunk_size);
    defaults.text_chunk_mode =
        engine::text::parse_text_chunk_mode_override(options).value_or(defaults.text_chunk_mode);
    defaults.vocoder_merge_steps =
        runtime::parse_i64_option(options, {"vocoder_merge_steps"}).value_or(defaults.vocoder_merge_steps);
    defaults.seed = runtime::parse_u64_option(options, {"seed"}).value_or(defaults.seed);
    if (defaults.num_inference_steps <= 0 || defaults.max_tokens <= 0 ||
        defaults.text_chunk_size <= 0 || defaults.vocoder_merge_steps <= 0) {
        throw std::runtime_error("DotTTS length and step options must be positive");
    }
    if (defaults.guidance_scale < 0.0F || !std::isfinite(defaults.guidance_scale)) {
        throw std::runtime_error("DotTTS guidance_scale must be finite and non-negative");
    }
    if (defaults.speaker_scale < 0.0F || !std::isfinite(defaults.speaker_scale)) {
        throw std::runtime_error("DotTTS speaker_scale must be finite and non-negative");
    }
    return defaults;
}

std::optional<DotsPromptReference> voice_reference(const std::optional<runtime::VoiceCondition> & voice) {
    if (!voice.has_value() || !voice->speaker.has_value()) {
        return std::nullopt;
    }
    if (voice->speaker->cached_voice_id.has_value() && !voice->speaker->cached_voice_id->empty()) {
        throw std::runtime_error("DotTTS cached_voice_id is not supported; pass reference audio for prompt prefill");
    }
    DotsPromptReference reference;
    if (voice->speaker->audio.has_value()) {
        reference.audio = voice->speaker->audio;
    }
    if (!reference.audio.has_value()) {
        return std::nullopt;
    }
    return reference;
}

void apply_reference_text(DotsPromptReference & reference, const std::unordered_map<std::string, std::string> & options) {
    reference.reference_text = runtime::find_option(options, {"reference_text"}).value_or(reference.reference_text);
}

void apply_reference_duration(DotsPromptReference & reference, const std::unordered_map<std::string, std::string> & options) {
    if (const auto value = runtime::parse_float_option(options, {"reference_duration_sec"})) {
        if (!std::isfinite(*value) || *value < 0.0F) {
            throw std::runtime_error("DotTTS reference_duration_sec must be finite and non-negative");
        }
        reference.duration_seconds = *value;
    }
}

}  // namespace

std::optional<DotsRequest> make_dots_prepare_defaults(
    const DotsAssets & assets,
    const runtime::SessionPreparationRequest & request) {
    DotsRequest defaults;
    bool has_defaults = false;
    if (request.text.has_value()) {
        defaults.text = request.text->text;
        has_defaults = true;
    }
    defaults.generation = generation_options(assets.config, request.options, defaults.generation);
    if (auto reference = voice_reference(request.voice)) {
        defaults.reference = std::move(*reference);
        apply_reference_text(defaults.reference, request.options);
        apply_reference_duration(defaults.reference, request.options);
        has_defaults = true;
    }
    return has_defaults ? std::optional<DotsRequest>(std::move(defaults)) : std::nullopt;
}

DotsRequest make_dots_request(
    const DotsAssets & assets,
    const runtime::TaskRequest & request,
    const std::optional<DotsRequest> & defaults) {
    DotsRequest out = defaults.value_or(DotsRequest{});
    if (request.text_input.has_value()) {
        out.text = request.text_input->text;
        if (!request.text_input->language.empty() &&
            runtime::find_option(request.options, {"language"}) == std::nullopt) {
            out.generation.language = request.text_input->language;
        }
    }
    out.generation = generation_options(assets.config, request.options, out.generation);
    if (auto reference = voice_reference(request.voice)) {
        out.reference = std::move(*reference);
    }
    apply_reference_text(out.reference, request.options);
    apply_reference_duration(out.reference, request.options);
    if (out.text.empty()) {
        throw std::runtime_error("DotTTS request text must not be empty");
    }
    if (!out.reference.reference_text.empty() && !out.reference.audio.has_value()) {
        throw std::runtime_error("DotTTS reference_text requires prompt reference audio");
    }
    return out;
}

}  // namespace engine::models::dots_tts
