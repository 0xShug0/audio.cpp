#include "engine/models/personaplex/request.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/runtime/options.h"

#include <cstring>
#include <stdexcept>
#include <unordered_map>

namespace engine::models::personaplex {
namespace {

std::string wrapped_system_prompt(std::string text) {
    if (text.empty()) {
        return text;
    }
    const std::string prefix = "<system>";
    if (text.rfind(prefix, 0) == 0 && text.size() >= prefix.size() &&
        text.rfind(prefix) == text.size() - prefix.size()) {
        return text;
    }
    return "<system> " + text + " <system>";
}

float parse_nonnegative_float(
    const std::unordered_map<std::string, std::string> & options,
    std::initializer_list<std::string_view> keys,
    float fallback,
    const char * name) {
    const auto value = runtime::parse_finite_float_option(options, keys);
    if (!value.has_value()) {
        return fallback;
    }
    if (*value < 0.0F) {
        throw std::runtime_error(std::string("PersonaPlex ") + name + " must be non-negative");
    }
    return *value;
}

int64_t parse_nonnegative_i64(
    const std::unordered_map<std::string, std::string> & options,
    std::initializer_list<std::string_view> keys,
    int64_t fallback,
    const char * name) {
    const auto value = runtime::parse_i64_option(options, keys);
    if (!value.has_value()) {
        return fallback;
    }
    if (*value < 0) {
        throw std::runtime_error(std::string("PersonaPlex ") + name + " must be non-negative");
    }
    return *value;
}

}  // namespace

PersonaPlexRequest make_personaplex_request(
    const runtime::TaskRequest & request,
    const PersonaPlexAssets & assets) {
    if (!request.audio_input.has_value()) {
        throw std::runtime_error("PersonaPlex requires audio input");
    }
    PersonaPlexRequest out;
    out.audio = *request.audio_input;
    if (out.audio.sample_rate <= 0 || out.audio.channels <= 0 || out.audio.samples.empty()) {
        throw std::runtime_error("PersonaPlex audio input is empty");
    }

    out.generation = make_personaplex_generation_options(request, assets);
    return out;
}

PersonaPlexGenerationOptions make_personaplex_generation_options(
    const runtime::TaskRequest & request,
    const PersonaPlexAssets & assets) {
    PersonaPlexGenerationOptions out;
    if (request.voice.has_value() && request.voice->speaker.has_value()) {
        if (request.voice->speaker->audio.has_value()) {
            out.voice_prompt_audio = *request.voice->speaker->audio;
        }
        if (request.voice->speaker->cached_voice_id.has_value()) {
            out.voice_id = *request.voice->speaker->cached_voice_id;
        }
    }
    out.voice_id = runtime::find_option(request.options, {"voice_id"}).value_or(out.voice_id);
    if (!out.voice_prompt_audio.has_value() &&
        assets.voice_prompt_paths.find(out.voice_id) == assets.voice_prompt_paths.end()) {
        throw std::runtime_error("unknown PersonaPlex voice_id: " + out.voice_id);
    }
    const std::string request_text = request.text_input.has_value() ? request.text_input->text : "";
    out.system_prompt = wrapped_system_prompt(
        runtime::find_option(request.options, {"system_prompt"}).value_or(request_text));
    if (const auto value = runtime::find_option(request.options, {"do_sample"})) {
        out.do_sample = runtime::parse_bool_option(*value, "do_sample");
    }
    out.temperature = parse_nonnegative_float(
        request.options,
        {"temperature"},
        out.temperature,
        "temperature");
    out.text_temperature = parse_nonnegative_float(
        request.options,
        {"text_temperature"},
        out.temperature,
        "text_temperature");
    if (out.do_sample &&
        (out.temperature <= 0.0F || out.text_temperature <= 0.0F)) {
        throw std::runtime_error("PersonaPlex sampled decoding requires positive temperature and text_temperature");
    }
    out.top_k = parse_nonnegative_i64(request.options, {"top_k"}, out.top_k, "top_k");
    out.text_top_k = parse_nonnegative_i64(request.options, {"text_top_k"}, out.top_k, "text_top_k");
    out.seed = runtime::parse_u32_option(request.options, {"seed"});
    return out;
}

PersonaPlexVoicePromptState load_personaplex_voice_prompt(
    const PersonaPlexAssets & assets,
    const std::string & voice_id) {
    const auto it = assets.voice_prompt_paths.find(voice_id);
    if (it == assets.voice_prompt_paths.end()) {
        throw std::runtime_error("unknown PersonaPlex voice_id: " + voice_id);
    }
    const auto source = engine::assets::open_tensor_source(it->second);
    const auto embeddings_meta = source->require_metadata("embeddings");
    if (embeddings_meta.shape.size() != 4 ||
        embeddings_meta.shape[1] != 1 ||
        embeddings_meta.shape[2] != 1 ||
        embeddings_meta.shape[3] != assets.config.lm.hidden_size) {
        throw std::runtime_error("PersonaPlex voice prompt embeddings shape mismatch for " + voice_id);
    }
    const auto cache_meta = source->require_metadata("cache");
    if (cache_meta.shape != std::vector<int64_t>{1, assets.config.lm.lm_codebooks + 1, 4}) {
        throw std::runtime_error("PersonaPlex voice prompt cache shape mismatch for " + voice_id);
    }
    PersonaPlexVoicePromptState out;
    out.frames = embeddings_meta.shape[0];
    out.embeddings = source->require_f32("embeddings", std::optional<std::vector<int64_t>>(embeddings_meta.shape));
    const auto raw_cache = source->require_tensor_data("cache");
    if (raw_cache.metadata.dtype != "I64" ||
        raw_cache.bytes.size() != static_cast<size_t>(1 * (assets.config.lm.lm_codebooks + 1) * 4) * sizeof(int64_t)) {
        throw std::runtime_error("PersonaPlex voice prompt cache dtype mismatch for " + voice_id);
    }
    out.cache.resize(raw_cache.bytes.size() / sizeof(int64_t));
    std::memcpy(out.cache.data(), raw_cache.bytes.data(), raw_cache.bytes.size());
    return out;
}

}  // namespace engine::models::personaplex
