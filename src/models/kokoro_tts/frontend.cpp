#include "engine/models/kokoro_tts/frontend.h"

#include "engine/models/kokoro_tts/g2p_en.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <stdexcept>

namespace engine::models::kokoro_tts {

namespace {

std::string lower_ascii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

std::string trim_ascii(std::string value) {
    const auto is_space = [](unsigned char ch) { return std::isspace(ch) != 0; };
    auto begin = value.begin();
    while (begin != value.end() && is_space(static_cast<unsigned char>(*begin))) {
        ++begin;
    }
    auto end = value.end();
    while (end != begin && is_space(static_cast<unsigned char>(*(end - 1)))) {
        --end;
    }
    return std::string(begin, end);
}

std::string default_voice_id() {
    return "af_heart";
}

std::string resolve_language_code_alias(const std::string & value) {
    const std::string normalized = lower_ascii(trim_ascii(value));
    if (normalized.empty()) {
        return {};
    }
    if (normalized == "a" || normalized == "en" || normalized == "en-us" || normalized == "us" || normalized == "american" || normalized == "american english") {
        return "a";
    }
    if (normalized == "b" || normalized == "en-gb" || normalized == "gb" || normalized == "uk" || normalized == "british" || normalized == "british english") {
        return "b";
    }
    throw std::runtime_error("unsupported Kokoro language: " + value + " (English only: en-us/a or en-gb/b)");
}

std::string voice_language_code(const std::string & voice_id) {
    if (voice_id.size() < 2 || (voice_id[1] != 'f' && voice_id[1] != 'm')) {
        throw std::runtime_error("invalid Kokoro voice id: " + voice_id);
    }
    return std::string(1, voice_id[0]);
}

std::string resolve_voice_id(
    const std::optional<runtime::VoiceCondition> & voice,
    const KokoroAssets & assets) {
    std::string voice_id = default_voice_id();
    if (voice.has_value() && voice->speaker.has_value() && voice->speaker->cached_voice_id.has_value()) {
        voice_id = *voice->speaker->cached_voice_id;
    }
    if (assets.voices.find(voice_id) == assets.voices.end()) {
        throw std::runtime_error("unknown Kokoro voice id: " + voice_id);
    }
    const std::string language_code = voice_language_code(voice_id);
    if (language_code != "a" && language_code != "b") {
        throw std::runtime_error(
            "Kokoro currently supports only English voices; got voice id " + voice_id +
            " with lang_code=" + language_code);
    }
    return voice_id;
}

std::string resolve_language_code(
    const runtime::Transcript & text,
    const std::optional<runtime::VoiceCondition> & voice,
    const std::string & voice_id) {
    std::string language_code;
    if (voice.has_value() && voice->style.has_value() && voice->style->language.has_value()) {
        language_code = resolve_language_code_alias(*voice->style->language);
    } else if (!text.language.empty()) {
        language_code = resolve_language_code_alias(text.language);
    } else {
        language_code = voice_language_code(voice_id);
    }
    const std::string expected = voice_language_code(voice_id);
    if (language_code != expected) {
        throw std::runtime_error(
            "Kokoro voice/language mismatch: voice " + voice_id +
            " requires lang_code=" + expected +
            " but request resolved to " + language_code);
    }
    return language_code;
}

std::string phonemize_text(
    const runtime::Transcript & text,
    const std::string & language_code,
    const KokoroAssets & assets) {
    if (text.text.empty()) {
        throw std::runtime_error("Kokoro TTS requires non-empty text");
    }
    if (language_code == "a" || language_code == "b") {
        const auto & g2p = language_code == "b" ? assets.english_g2p_gb : assets.english_g2p_us;
        if (!g2p) {
            throw std::runtime_error("Kokoro English G2P assets were not prepared");
        }
        return (*g2p)(text.text).first;
    }
    throw std::runtime_error(
        "unsupported Kokoro language code: " + language_code +
        " (English only: a/en-us or b/en-gb)");
}

struct EncodedInputIds {
    std::vector<int32_t> ids;
    size_t phoneme_count = 0;
};

EncodedInputIds encode_input_ids_and_count(
    const std::string & phonemes,
    const KokoroAssets & assets) {
    EncodedInputIds encoded;
    encoded.ids.reserve(phonemes.size() + 2);
    encoded.ids.push_back(0);
    for (size_t i = 0; i < phonemes.size();) {
        const unsigned char lead = static_cast<unsigned char>(phonemes[i]);
        size_t width = 0;
        if ((lead & 0x80u) == 0) {
            width = 1;
        } else if ((lead & 0xE0u) == 0xC0u) {
            width = 2;
        } else if ((lead & 0xF0u) == 0xE0u) {
            width = 3;
        } else if ((lead & 0xF8u) == 0xF0u) {
            width = 4;
        } else {
            throw std::runtime_error("invalid UTF-8 lead byte in Kokoro phoneme string");
        }
        if (i + width > phonemes.size()) {
            throw std::runtime_error("truncated UTF-8 codepoint in Kokoro phoneme string");
        }
        for (size_t j = 1; j < width; ++j) {
            const unsigned char byte = static_cast<unsigned char>(phonemes[i + j]);
            if ((byte & 0xC0u) != 0x80u) {
                throw std::runtime_error("invalid UTF-8 continuation byte in Kokoro phoneme string");
            }
        }
        const auto it = assets.vocab.find(phonemes.substr(i, width));
        if (it == assets.vocab.end()) {
            throw std::runtime_error("Kokoro vocab is missing phoneme symbol: " + phonemes.substr(i, width));
        }
        encoded.ids.push_back(it->second);
        ++encoded.phoneme_count;
        i += width;
    }
    encoded.ids.push_back(0);
    return encoded;
}

std::vector<float> style_for_phoneme_count(
    const KokoroVoicePack & pack,
    size_t phoneme_count) {
    if (pack.cols != 256) {
        throw std::runtime_error("Kokoro voice pack must have 256 columns: " + pack.id);
    }
    if (phoneme_count == 0) {
        throw std::runtime_error("Kokoro phoneme string must not be empty");
    }
    if (phoneme_count > static_cast<size_t>(pack.rows)) {
        throw std::runtime_error(
            "Kokoro phoneme count exceeds voice style rows: " + std::to_string(phoneme_count));
    }
    const size_t row = phoneme_count - 1;
    const size_t offset = row * static_cast<size_t>(pack.cols);
    std::vector<float> style(static_cast<size_t>(pack.cols));
    std::memcpy(
        style.data(),
        pack.values.data() + offset,
        static_cast<size_t>(pack.cols) * sizeof(float));
    return style;
}

float resolve_speaking_rate(const std::optional<runtime::VoiceCondition> & voice) {
    if (!voice.has_value() || !voice->style.has_value() || !voice->style->speaking_rate.has_value()) {
        return 1.0f;
    }
    const float rate = *voice->style->speaking_rate;
    if (!(rate > 0.0f)) {
        throw std::runtime_error("Kokoro speaking_rate must be positive");
    }
    return rate;
}

}  // namespace

KokoroFrontendSessionState resolve_kokoro_frontend_session_state(
    const std::optional<runtime::Transcript> & text,
    const std::optional<runtime::VoiceCondition> & voice,
    const KokoroAssets & assets) {
    runtime::Transcript transcript;
    if (text.has_value()) {
        transcript = *text;
    }
    KokoroFrontendSessionState state;
    state.voice_id = resolve_voice_id(voice, assets);
    state.language_code = resolve_language_code(transcript, voice, state.voice_id);
    const auto voice_it = assets.voices.find(state.voice_id);
    if (voice_it == assets.voices.end()) {
        throw std::runtime_error("unknown Kokoro voice id: " + state.voice_id);
    }
    state.voice_pack = &voice_it->second;
    state.speaking_rate = resolve_speaking_rate(voice);
    return state;
}

void validate_kokoro_frontend_session_state(
    const runtime::Transcript & text,
    const std::optional<runtime::VoiceCondition> & voice,
    const KokoroFrontendSessionState & state,
    const KokoroAssets & assets) {
    const std::string resolved_voice_id = resolve_voice_id(voice, assets);
    if (resolved_voice_id != state.voice_id) {
        throw std::runtime_error(
            "Kokoro session voice_id changed after launch: " +
            state.voice_id + " -> " + resolved_voice_id);
    }
    const std::string resolved_language_code = resolve_language_code(text, voice, state.voice_id);
    if (resolved_language_code != state.language_code) {
        throw std::runtime_error(
            "Kokoro session language_code changed after launch: " +
            state.language_code + " -> " + resolved_language_code);
    }
    const auto voice_it = assets.voices.find(state.voice_id);
    if (voice_it == assets.voices.end() || &voice_it->second != state.voice_pack) {
        throw std::runtime_error("Kokoro session voice pack changed after launch");
    }
    const float resolved_speaking_rate = resolve_speaking_rate(voice);
    if (resolved_speaking_rate != state.speaking_rate) {
        throw std::runtime_error("Kokoro session speaking_rate changed after launch");
    }
}

KokoroSynthesisInput build_kokoro_synthesis_input(
    const runtime::Transcript & text,
    const KokoroFrontendSessionState & state,
    const KokoroAssets & assets) {
    if (state.voice_pack == nullptr) {
        throw std::runtime_error("Kokoro frontend session voice pack was not prepared");
    }
    const std::string phonemes = phonemize_text(text, state.language_code, assets);
    const EncodedInputIds encoded = encode_input_ids_and_count(phonemes, assets);
    if (encoded.phoneme_count > 510) {
        throw std::runtime_error(
            "Kokoro phoneme string exceeds 510 symbols; segmenting is not implemented in the framework path yet");
    }
    KokoroSynthesisInput input;
    input.voice_id = state.voice_id;
    input.language_code = state.language_code;
    input.phonemes = phonemes;
    input.input_ids = encoded.ids;
    input.style = style_for_phoneme_count(*state.voice_pack, encoded.phoneme_count);
    input.speaking_rate = state.speaking_rate;
    if (static_cast<int64_t>(input.input_ids.size()) > assets.context_length) {
        throw std::runtime_error("Kokoro tokenized input exceeds model context length");
    }
    return input;
}

int64_t estimate_kokoro_request_tokens(
    const runtime::SessionPreparationRequest & request,
    const KokoroFrontendSessionState & state,
    const KokoroAssets & assets) {
    if (!request.text.has_value()) {
        return 0;
    }
    const auto input = build_kokoro_synthesis_input(*request.text, state, assets);
    return static_cast<int64_t>(input.input_ids.size());
}

}  // namespace engine::models::kokoro_tts
