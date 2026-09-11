#include "engine/models/magpie_tts/request.h"

#include "engine/framework/runtime/options.h"
#include "engine/framework/text/chunking.h"
#include "engine/models/magpie_tts/tokenizer_text.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace engine::models::magpie_tts {
namespace {

std::string request_language(
    const std::optional<runtime::Transcript> & text,
    const std::unordered_map<std::string, std::string> & options,
    const std::string & fallback) {
    if (text.has_value() && !text->language.empty()) {
        return text->language;
    }
    if (const auto value = runtime::find_option(options, {"language"})) {
        if (!value->empty()) {
            return *value;
        }
    }
    return fallback.empty() ? "en" : fallback;
}

int32_t parse_voice_id(
    const MagpieTTSAssets & assets,
    const std::unordered_map<std::string, std::string> & options,
    int32_t fallback) {
    const auto value = runtime::find_option(options, {"voice_id"});
    if (!value.has_value() || value->empty()) {
        return fallback;
    }
    for (size_t index = 0; index < assets.config.speaker_names.size(); ++index) {
        if (assets.config.speaker_names[index] == *value) {
            return static_cast<int32_t>(index);
        }
    }
    size_t parsed = 0;
    const int64_t speaker = std::stoll(*value, &parsed);
    if (parsed != value->size()) {
        throw std::runtime_error("MagpieTTS voice_id must be a baked voice index or name");
    }
    if (speaker < 0 || speaker >= assets.config.speakers) {
        throw std::runtime_error("MagpieTTS voice_id is outside the baked voice range");
    }
    return static_cast<int32_t>(speaker);
}

MagpieTTSGenerationOptions generation_options(
    const MagpieTTSAssets & assets,
    const std::unordered_map<std::string, std::string> & options,
    MagpieTTSGenerationOptions defaults) {
    defaults.language = MagpieTextTokenizer::normalize_language(
        runtime::find_option(options, {"language"}).value_or(defaults.language));
    defaults.speaker = parse_voice_id(assets, options, defaults.speaker);
    defaults.temperature = runtime::parse_float_option(options, {"temperature"}).value_or(defaults.temperature);
    defaults.top_k = runtime::parse_int_option(options, {"top_k"}).value_or(defaults.top_k);
    defaults.guidance_scale =
        runtime::parse_float_option(options, {"guidance_scale"}).value_or(defaults.guidance_scale);
    defaults.max_tokens = runtime::parse_i64_option(options, {"max_tokens"}).value_or(defaults.max_tokens);
    defaults.text_chunk_size =
        engine::text::parse_text_chunk_size_override(options).value_or(defaults.text_chunk_size);
    defaults.text_chunk_mode =
        engine::text::parse_text_chunk_mode_override(options).value_or(defaults.text_chunk_mode);
    defaults.seed = runtime::parse_u64_option(options, {"seed"}).value_or(defaults.seed);
    if (!(defaults.temperature >= 0.0F) || !std::isfinite(defaults.temperature)) {
        throw std::runtime_error("MagpieTTS temperature must be finite and non-negative");
    }
    if (defaults.top_k <= 0) {
        throw std::runtime_error("MagpieTTS top_k must be positive");
    }
    if (!(defaults.guidance_scale >= 0.0F) || !std::isfinite(defaults.guidance_scale)) {
        throw std::runtime_error("MagpieTTS guidance_scale must be finite and non-negative");
    }
    if (defaults.max_tokens <= 0 || defaults.text_chunk_size <= 0) {
        throw std::runtime_error("MagpieTTS length options must be positive");
    }
    return defaults;
}

}  // namespace

std::optional<MagpieTTSRequest> make_magpie_tts_prepare_defaults(
    const MagpieTTSAssets & assets,
    const runtime::SessionPreparationRequest & request) {
    MagpieTTSRequest defaults;
    bool has_defaults = false;
    if (request.text.has_value()) {
        defaults.text = request.text->text;
        defaults.generation.language =
            request_language(request.text, request.options, defaults.generation.language);
        has_defaults = true;
    } else {
        defaults.generation.language =
            request_language(std::nullopt, request.options, defaults.generation.language);
        has_defaults = defaults.generation.language != "en";
    }
    defaults.generation = generation_options(assets, request.options, defaults.generation);
    return has_defaults ? std::optional<MagpieTTSRequest>(std::move(defaults)) : std::nullopt;
}

MagpieTTSRequest make_magpie_tts_request(
    const MagpieTTSAssets & assets,
    const runtime::TaskRequest & request,
    const std::optional<MagpieTTSRequest> & defaults) {
    MagpieTTSRequest out = defaults.value_or(MagpieTTSRequest{});
    if (request.text_input.has_value()) {
        out.text = request.text_input->text;
    }
    out.generation.language = request_language(request.text_input, request.options, out.generation.language);
    out.generation = generation_options(assets, request.options, out.generation);
    if (out.text.empty()) {
        throw std::runtime_error("MagpieTTS request text must not be empty");
    }
    return out;
}

}  // namespace engine::models::magpie_tts
