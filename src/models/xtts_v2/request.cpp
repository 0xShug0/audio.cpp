#include "engine/models/xtts_v2/request.h"

#include "engine/framework/io/text.h"
#include "engine/framework/runtime/options.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <unordered_set>

namespace engine::models::xtts_v2 {
namespace {

const std::unordered_set<std::string> kLanguages = {
    "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl",
    "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi",
};

std::string normalize_language(std::string language) {
    language = engine::io::trim_ascii_whitespace(language);
    std::transform(language.begin(), language.end(), language.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    if (language == "zh") {
        language = "zh-cn";
    }
    if (kLanguages.count(language) == 0) {
        throw std::runtime_error("unsupported XTTS v2 language: " + language);
    }
    return language;
}

}  // namespace

XttsV2Request parse_xtts_v2_request(const runtime::TaskRequest & request) {
    XttsV2Request out;
    if (request.text_input.has_value()) {
        out.text = engine::io::trim_ascii_whitespace(request.text_input->text);
    } else if (const auto text = runtime::find_option(request.options, {"text", "prompt"})) {
        out.text = engine::io::trim_ascii_whitespace(*text);
    }
    if (out.text.empty()) {
        throw std::runtime_error("XTTS v2 requires text input");
    }
    if (!request.voice.has_value() || !request.voice->speaker.has_value() ||
        !request.voice->speaker->audio.has_value()) {
        throw std::runtime_error("XTTS v2 requires --voice-ref or voice.speaker.audio");
    }
    out.speaker_audio = *request.voice->speaker->audio;
    if (out.speaker_audio.sample_rate <= 0 || out.speaker_audio.channels <= 0 ||
        out.speaker_audio.samples.empty()) {
        throw std::runtime_error("XTTS v2 speaker reference is empty or invalid");
    }
    if (const auto language = runtime::find_option(request.options, {"language"})) {
        out.language = normalize_language(*language);
    }
    if (const auto value = runtime::parse_finite_float_option(request.options, {"temperature"})) out.generation.temperature = *value;
    if (const auto value = runtime::parse_finite_float_option(request.options, {"top_p"})) out.generation.top_p = *value;
    if (const auto value = runtime::parse_int_option(request.options, {"top_k"})) out.generation.top_k = *value;
    if (const auto value = runtime::parse_finite_float_option(request.options, {"repetition_penalty"})) out.generation.repetition_penalty = *value;
    if (const auto value = runtime::parse_finite_float_option(request.options, {"speed"})) out.generation.speed = *value;
    if (const auto value = runtime::parse_int_option(request.options, {"max_tokens"})) out.generation.max_tokens = *value;
    if (const auto value = runtime::parse_u32_option(request.options, {"seed"})) {
        out.generation.seed = *value;
    } else {
        out.generation.seed = runtime::random_u32_seed();
    }
    if (!(out.generation.temperature > 0.0F) || !(out.generation.top_p > 0.0F && out.generation.top_p <= 1.0F) ||
        out.generation.top_k <= 0 || out.generation.repetition_penalty < 1.0F ||
        !(out.generation.speed > 0.0F) || out.generation.max_tokens <= 0 || out.generation.max_tokens > 603) {
        throw std::runtime_error("invalid XTTS v2 generation options");
    }
    return out;
}

}  // namespace engine::models::xtts_v2
