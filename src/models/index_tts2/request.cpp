#include "engine/models/index_tts2/request.h"

#include "engine/framework/io/text.h"
#include "engine/framework/runtime/options.h"

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>

namespace engine::models::index_tts2 {
namespace {

std::vector<float> parse_emotion_vector(const std::string & value) {
    std::string normalized = engine::io::trim_ascii_whitespace(value);
    if (!normalized.empty() && normalized.front() == '[') {
        normalized.erase(normalized.begin());
    }
    if (!normalized.empty() && normalized.back() == ']') {
        normalized.pop_back();
    }
    std::vector<float> values;
    std::stringstream stream(normalized);
    std::string item;
    while (std::getline(stream, item, ',')) {
        item = engine::io::trim_ascii_whitespace(item);
        if (item.empty()) {
            throw std::runtime_error("IndexTTS2 emotion_vector contains an empty item");
        }
        size_t parsed = 0;
        const float parsed_value = std::stof(item, &parsed);
        if (parsed != item.size() || !std::isfinite(parsed_value)) {
            throw std::runtime_error("IndexTTS2 emotion_vector must contain finite floats");
        }
        values.push_back(parsed_value);
    }
    if (values.size() != 8) {
        throw std::runtime_error("IndexTTS2 emotion_vector must contain exactly 8 values");
    }
    return values;
}

const runtime::AudioBuffer * speaker_audio_from_request(const runtime::TaskRequest & request) {
    if (request.voice.has_value() &&
        request.voice->speaker.has_value() &&
        request.voice->speaker->audio.has_value()) {
        return &*request.voice->speaker->audio;
    }
    return nullptr;
}

void require_valid_audio(const runtime::AudioBuffer & audio, const char * label) {
    if (audio.sample_rate <= 0 || audio.channels <= 0 || audio.samples.empty()) {
        throw std::runtime_error(std::string("IndexTTS2 ") + label + " audio is empty or invalid");
    }
}

}  // namespace

IndexTTS2Request parse_index_tts2_request(const runtime::TaskRequest & request) {
    IndexTTS2Request out;
    if (request.text_input.has_value()) {
        out.text = engine::io::trim_ascii_whitespace(request.text_input->text);
    } else if (const auto value = runtime::find_option(request.options, {"text", "prompt"})) {
        out.text = engine::io::trim_ascii_whitespace(*value);
    }
    if (out.text.empty()) {
        throw std::runtime_error("IndexTTS2 request requires text_input or text option");
    }

    if (const auto * speaker = speaker_audio_from_request(request)) {
        require_valid_audio(*speaker, "speaker reference");
        out.speaker_audio = *speaker;
    } else {
        throw std::runtime_error("IndexTTS2 request requires --voice-ref or voice.speaker.audio");
    }

    if (const auto value = runtime::parse_finite_float_option(request.options, {"emotion_alpha"})) {
        if (*value < 0.0F || *value > 1.0F) {
            throw std::runtime_error("IndexTTS2 emotion_alpha must be in [0, 1]");
        }
        out.emotion_alpha = *value;
    }
    if (const auto value = runtime::find_option(request.options, {"emotion_vector"})) {
        out.emotion_vector = parse_emotion_vector(*value);
    }
    const auto use_emotion_text_value = runtime::find_option(request.options, {"use_emotion_text"});
    if (use_emotion_text_value.has_value()) {
        out.use_emotion_text = runtime::parse_bool_option(*use_emotion_text_value, "use_emotion_text");
    }
    if (const auto value = runtime::find_option(request.options, {"emotion_text"})) {
        if (!engine::io::trim_ascii_whitespace(*value).empty()) {
            out.emotion_text = *value;
        }
    }
    if (request.voice.has_value() &&
        request.voice->style.has_value() &&
        request.voice->style->emotion.has_value()) {
        const auto & text = *request.voice->style->emotion;
        if (!engine::io::trim_ascii_whitespace(text).empty()) {
            if (use_emotion_text_value.has_value() && !out.use_emotion_text) {
                throw std::runtime_error("IndexTTS2 --emotion conflicts with use_emotion_text=false");
            }
            if (out.emotion_text.has_value() && *out.emotion_text != text) {
                throw std::runtime_error("IndexTTS2 --emotion conflicts with emotion_text");
            }
            out.use_emotion_text = true;
            out.emotion_text = text;
        }
    }
    if (const auto value = runtime::find_option(request.options, {"use_random_emotion"})) {
        out.use_random_emotion = runtime::parse_bool_option(*value, "use_random_emotion");
    }
    if (const auto value = runtime::parse_int_option(request.options, {"interval_silence_ms"})) {
        if (*value < 0) {
            throw std::runtime_error("IndexTTS2 interval_silence_ms must be non-negative");
        }
        out.interval_silence_ms = *value;
    }
    if (request.audio_input.has_value()) {
        require_valid_audio(*request.audio_input, "emotion reference");
        out.emotion_audio = request.audio_input;
    }
    if (out.emotion_vector.has_value() || out.use_emotion_text) {
        out.emotion_audio = std::nullopt;
    }

    if (const auto value = runtime::find_option(request.options, {"do_sample"})) {
        out.generation.do_sample = runtime::parse_bool_option(*value, "do_sample");
    }
    if (const auto value = runtime::parse_finite_float_option(request.options, {"top_p"})) {
        out.generation.top_p = *value;
    }
    if (const auto value = runtime::parse_int_option(request.options, {"top_k"})) {
        out.generation.top_k = *value;
    }
    if (const auto value = runtime::parse_finite_float_option(request.options, {"temperature"})) {
        out.generation.temperature = *value;
    }
    if (const auto value = runtime::parse_finite_float_option(request.options, {"length_penalty"})) {
        out.generation.length_penalty = *value;
    }
    if (const auto value = runtime::parse_int_option(request.options, {"num_beams"})) {
        out.generation.num_beams = *value;
    }
    if (const auto value = runtime::parse_finite_float_option(request.options, {"repetition_penalty"})) {
        out.generation.repetition_penalty = *value;
    }
    if (const auto value = runtime::parse_int_option(request.options, {"max_tokens"})) {
        if (*value <= 0) {
            throw std::runtime_error("IndexTTS2 max_tokens must be positive");
        }
        out.generation.max_mel_tokens = *value;
    }
    if (const auto value = runtime::parse_u32_option(request.options, {"seed"})) {
        out.generation.seed = *value;
    } else {
        out.generation.seed = runtime::random_u32_seed();
    }

    if (out.generation.top_k <= 0) {
        throw std::runtime_error("IndexTTS2 top_k must be positive");
    }
    if (!(out.generation.top_p > 0.0F && out.generation.top_p <= 1.0F)) {
        throw std::runtime_error("IndexTTS2 top_p must be in (0, 1]");
    }
    if (!(out.generation.temperature > 0.0F)) {
        throw std::runtime_error("IndexTTS2 temperature must be positive");
    }
    if (out.generation.num_beams <= 0) {
        throw std::runtime_error("IndexTTS2 num_beams must be positive");
    }
    return out;
}

}  // namespace engine::models::index_tts2
