#include "engine/models/kokoro_tts/frontend.h"

#include "engine/models/kokoro_tts/g2p_multilingual.h"

#include "engine/framework/debug/trace.h"

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
    for (const auto & pair : std::vector<std::pair<std::string, std::string>>{
        {"e", "e"}, {"es", "e"}, {"es-es", "e"}, {"f", "f"}, {"fr", "f"}, {"fr-fr", "f"},
        {"h", "h"}, {"hi", "h"}, {"hi-in", "h"}, {"i", "i"}, {"it", "i"}, {"it-it", "i"},
        {"j", "j"}, {"ja", "j"}, {"ja-jp", "j"}, {"p", "p"}, {"pt", "p"}, {"pt-br", "p"},
        {"z", "z"}, {"zh", "z"}, {"zh-cn", "z"}, {"cmn", "z"}})
        if (normalized == pair.first) return pair.second;
    throw std::runtime_error("unsupported Kokoro language: " + value);
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
    (void) resolve_language_code_alias(language_code);
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
    if (!assets.multilingual_g2p) throw std::runtime_error("Kokoro multilingual resources were not prepared");
    return assets.multilingual_g2p->phonemize(text.text, language_code);
}

struct EncodedInputIds {
    std::vector<int32_t> ids;
    size_t phoneme_count = 0;
};

/// `reject_unknown` decides what an out-of-vocabulary symbol means, and the answer depends
/// entirely on who produced the stream.
///
/// ⚠ FALSE for our own G2P, TRUE for a caller's. The reference implementation (hexgrad/Kokoro's
/// KModel, `filter(None, map(vocab.get, phonemes))`) drops what it cannot tokenize, and for our
/// own output that is the only sane answer: eSpeak emits a syllabic mark for "button" that this
/// vocabulary has no id for, and nobody downstream can do anything about it.
///
/// A CALLER'S stream is the opposite case. They can fix it, so telling them is strictly more
/// useful than guessing -- and guessing is not harmless here. Canonical IPA writes a diphthong
/// as two symbols, and Kokoro writes it as one, so dropping the off-glide silently turns
/// `lˈaᶦk` into `lˈak`: "like" becomes "lack", with no error and audio that sounds deliberate.
EncodedInputIds encode_input_ids_and_count(
    const std::string & phonemes,
    const KokoroAssets & assets,
    bool reject_unknown) {
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
        const std::string symbol = phonemes.substr(i, width);
        const auto it = assets.vocab.find(symbol);
        if (it == assets.vocab.end()) {
            if (reject_unknown) {
                throw std::runtime_error(
                    "Kokoro vocab is missing phoneme symbol: " + symbol +
                    "; supplied phonemes must be in Kokoro's own " + std::to_string(assets.vocab.size()) +
                    "-symbol vocabulary, which is not canonical IPA -- a diphthong is one symbol there"
                    " and two in IPA, so an off-glide is a common cause");
            }
            // Skipped, not fatal, matching the reference implementation: hexgrad/Kokoro's KModel
            // tokenizes with `filter(None, map(vocab.get, phonemes))`, which drops any phoneme the
            // 114-entry vocab has no id for.
            //
            // This matters because our OWN G2P produces such symbols for ordinary words: eSpeak-ng
            // glottalises /t/ before a syllabic nasal, so "button" is `b'V?n` with a U+0329
            // syllabic mark the vocab does not carry. Throwing there loses the whole request;
            // dropping the mark gives a correct reading of the word.
            //
            // Malformed UTF-8 above still throws — that is a real error. An unknown but
            // well-formed phoneme is not.
            engine::debug::trace_log_scalar("kokoro.skipped_phoneme", std::string_view(symbol));
            i += width;
            continue;
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

KokoroSynthesisInput build_kokoro_synthesis_input(
    const runtime::Transcript & text,
    const KokoroFrontendSessionState & state,
    const KokoroAssets & assets,
    std::optional<std::string_view> phoneme_override) {
    if (state.voice_pack == nullptr) {
        throw std::runtime_error("Kokoro frontend session voice pack was not prepared");
    }
    const bool supplied = phoneme_override.has_value();
    if (supplied && phoneme_override->empty()) {
        // Falling back to the built-in G2P here would speak `text` for a chunk the caller
        // asked to be spoken from phonemes, which reads as the engine ignoring the option.
        throw std::runtime_error("Kokoro supplied phonemes are empty; an override must carry symbols");
    }
    const std::string phonemes =
        supplied ? std::string(*phoneme_override) : phonemize_text(text, state.language_code, assets);
    // Strict for a caller's stream, lenient for our own -- see encode_input_ids_and_count.
    const EncodedInputIds encoded = encode_input_ids_and_count(phonemes, assets, supplied);
    if (encoded.phoneme_count > 510) {
        // Two different failures wearing one message helps nobody: the caller who supplied the
        // phonemes can fix this by sending less, and is told so; the caller who supplied text
        // is hitting an engine limitation and is told that instead.
        throw std::runtime_error(
            supplied
                ? "Kokoro phoneme chunk exceeds 510 symbols; split the supplied phonemes across "
                  "more list entries (each entry is rendered separately and the audio is merged)"
                : "Kokoro phoneme string exceeds 510 symbols; segmenting is not implemented in the framework path yet");
    }
    if (supplied) {
        engine::debug::trace_log_scalar("kokoro.supplied_phoneme_count", static_cast<int64_t>(encoded.phoneme_count));
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
    const KokoroAssets & assets,
    std::optional<std::string_view> phoneme_override) {
    if (!request.text.has_value()) {
        return 0;
    }
    const auto input = build_kokoro_synthesis_input(*request.text, state, assets, phoneme_override);
    return static_cast<int64_t>(input.input_ids.size());
}

}  // namespace engine::models::kokoro_tts
