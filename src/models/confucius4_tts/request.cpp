#include "engine/models/confucius4_tts/request.h"

#include "engine/framework/runtime/options.h"
#include "engine/framework/text/chunking.h"

#include <stdexcept>
#include <string>
#include <unordered_map>

namespace engine::models::confucius4_tts {
namespace {

constexpr const char * kDefaultLanguage = "zh";

std::string request_language(
    const std::optional<runtime::Transcript> & text,
    const std::optional<runtime::VoiceCondition> & voice,
    const std::unordered_map<std::string, std::string> & options,
    const std::string & fallback) {
    if (text.has_value() && !text->language.empty()) {
        return text->language;
    }
    if (voice.has_value() &&
        voice->style.has_value() &&
        voice->style->language.has_value() &&
        !voice->style->language->empty()) {
        return *voice->style->language;
    }
    if (const auto value = runtime::find_option(options, {"language"})) {
        if (!value->empty()) {
            return *value;
        }
    }
    return fallback.empty() ? kDefaultLanguage : fallback;
}

ConfuciusGenerationOptions generation_options(
    const std::unordered_map<std::string, std::string> & options,
    ConfuciusGenerationOptions defaults) {
    defaults.temperature = runtime::parse_float_option(options, {"temperature"}).value_or(defaults.temperature);
    defaults.top_p = runtime::parse_float_option(options, {"top_p"}).value_or(defaults.top_p);
    defaults.top_k = runtime::parse_int_option(options, {"top_k"}).value_or(defaults.top_k);
    defaults.num_beams = runtime::parse_int_option(options, {"num_beams"}).value_or(defaults.num_beams);
    defaults.repetition_penalty =
        runtime::parse_float_option(options, {"repetition_penalty"}).value_or(defaults.repetition_penalty);
    defaults.max_tokens = runtime::parse_i64_option(options, {"max_tokens"}).value_or(defaults.max_tokens);
    defaults.num_inference_steps =
        runtime::parse_i64_option(options, {"num_inference_steps"}).value_or(defaults.num_inference_steps);
    defaults.guidance_scale =
        runtime::parse_float_option(options, {"guidance_scale"}).value_or(defaults.guidance_scale);
    defaults.max_text_tokens_per_segment =
        engine::text::parse_text_chunk_size_override(options).value_or(defaults.max_text_tokens_per_segment);
    defaults.cross_fade_duration_sec =
        runtime::parse_float_option(options, {"cross_fade_duration_sec"}).value_or(defaults.cross_fade_duration_sec);
    defaults.edge_fade_duration_sec =
        runtime::parse_float_option(options, {"edge_fade_duration_sec"}).value_or(defaults.edge_fade_duration_sec);
    defaults.edge_pad_duration_sec =
        runtime::parse_float_option(options, {"edge_pad_duration_sec"}).value_or(defaults.edge_pad_duration_sec);
    defaults.seed = runtime::parse_u32_option(options, {"seed"}).value_or(defaults.seed);
    defaults.text_chunk_mode =
        engine::text::parse_text_chunk_mode_override(options).value_or(defaults.text_chunk_mode);
    if (!(defaults.temperature > 0.0F)) {
        throw std::runtime_error("Confucius4-TTS temperature must be positive");
    }
    if (!(defaults.top_p > 0.0F && defaults.top_p <= 1.0F)) {
        throw std::runtime_error("Confucius4-TTS top_p must be in (0, 1]");
    }
    if (defaults.top_k <= 0) {
        throw std::runtime_error("Confucius4-TTS top_k must be positive");
    }
    if (defaults.num_beams <= 0) {
        throw std::runtime_error("Confucius4-TTS num_beams must be positive");
    }
    if (defaults.max_tokens <= 0 || defaults.num_inference_steps <= 0 || defaults.max_text_tokens_per_segment <= 0) {
        throw std::runtime_error("Confucius4-TTS length and step options must be positive");
    }
    if (defaults.cross_fade_duration_sec < 0.0F || defaults.edge_fade_duration_sec < 0.0F ||
        defaults.edge_pad_duration_sec < 0.0F) {
        throw std::runtime_error("Confucius4-TTS fade durations must be non-negative");
    }
    return defaults;
}

std::optional<ConfuciusVoiceReference> voice_reference(const std::optional<runtime::VoiceCondition> & voice) {
    if (!voice.has_value() || !voice->speaker.has_value()) {
        return std::nullopt;
    }
    ConfuciusVoiceReference reference;
    if (voice->speaker->audio.has_value()) {
        reference.audio = voice->speaker->audio;
    }
    if (voice->speaker->cached_voice_id.has_value()) {
        reference.cache_id = *voice->speaker->cached_voice_id;
    }
    if (!reference.audio.has_value() && reference.cache_id.empty()) {
        return std::nullopt;
    }
    return reference;
}

}  // namespace

std::optional<ConfuciusRequest> make_confucius_prepare_defaults(
    const ConfuciusAssets &,
    const runtime::SessionPreparationRequest & request) {
    ConfuciusRequest defaults;
    bool has_defaults = false;
    if (request.text.has_value()) {
        defaults.text = request.text->text;
        defaults.language = request_language(request.text, request.voice, request.options, defaults.language);
        has_defaults = true;
    } else {
        defaults.language = request_language(std::nullopt, request.voice, request.options, defaults.language);
        has_defaults = defaults.language != kDefaultLanguage;
    }
    defaults.generation = generation_options(request.options, defaults.generation);
    if (auto reference = voice_reference(request.voice)) {
        defaults.reference = std::move(*reference);
        has_defaults = true;
    }
    return has_defaults ? std::optional<ConfuciusRequest>(std::move(defaults)) : std::nullopt;
}

ConfuciusRequest make_confucius_request(
    const ConfuciusAssets & assets,
    const runtime::TaskRequest & request,
    const std::optional<ConfuciusRequest> & defaults) {
    (void)assets;
    ConfuciusRequest out = defaults.value_or(ConfuciusRequest{});
    if (request.text_input.has_value()) {
        out.text = request.text_input->text;
    }
    out.language = request_language(request.text_input, request.voice, request.options, out.language);
    out.generation = generation_options(request.options, out.generation);
    if (auto reference = voice_reference(request.voice)) {
        out.reference = std::move(*reference);
    }
    if (out.text.empty()) {
        throw std::runtime_error("Confucius4-TTS request text must not be empty");
    }
    if (!out.reference.audio.has_value() && out.reference.cache_id.empty()) {
        throw std::runtime_error("Confucius4-TTS voice cloning requires speaker reference audio or cached_voice_id");
    }
    return out;
}

}  // namespace engine::models::confucius4_tts
