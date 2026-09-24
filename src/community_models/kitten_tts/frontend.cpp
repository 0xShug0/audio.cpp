#include "engine/community_models/kitten_tts/frontend.h"

#include "engine/framework/debug/trace.h"
#include "engine/framework/io/text.h"
#include <algorithm>
#include <cctype>
#include <cstring>
#include <stdexcept>

namespace engine::models::kitten_tts {

namespace {

std::string lower_ascii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return value;
}

std::string default_voice_id() { return "expr-voice-5-m"; }

std::string resolve_language_code_alias(const std::string &value) {
    const std::string normalized = lower_ascii(engine::io::trim_ascii_whitespace(value));
    if (normalized.empty()) {
        return {};
    }
    if (normalized == "en" || normalized == "en-us" || normalized == "us" || normalized == "american" ||
        normalized == "american english") {
        return "en-us";
    }
    throw std::runtime_error("unsupported Kitten language: " + value + " (English only: en-us)");
}

std::string resolve_voice_id(const std::optional<runtime::VoiceCondition> &voice, const KittenAssets &assets) {
    std::string voice_id = default_voice_id();
    if (voice.has_value() && voice->speaker.has_value() && voice->speaker->cached_voice_id.has_value()) {
        voice_id = *voice->speaker->cached_voice_id;
    }
    const auto alias = assets.voice_aliases.find(voice_id);
    if (alias != assets.voice_aliases.end()) {
        voice_id = alias->second;
    }
    if (assets.voices.find(voice_id) == assets.voices.end()) {
        throw std::runtime_error("unknown Kitten voice id: " + voice_id);
    }
    return voice_id;
}

std::string resolve_language_code(const runtime::Transcript &text, const std::optional<runtime::VoiceCondition> &voice,
                                  const std::string &voice_id) {
    std::string language_code;
    if (voice.has_value() && voice->style.has_value() && voice->style->language.has_value()) {
        language_code = resolve_language_code_alias(*voice->style->language);
    } else if (!text.language.empty()) {
        language_code = resolve_language_code_alias(text.language);
    } else {
        language_code = "en-us";
    }
    (void)voice_id;
    return language_code;
}

std::string ensure_kitten_punctuation(std::string text) {
    text = engine::io::trim_ascii_whitespace(std::move(text));
    if (text.empty()) {
        return text;
    }
    const char last = text.back();
    if (std::string(".!?,;:").find(last) == std::string::npos) {
        text.push_back(',');
    }
    return text;
}

std::string collapse_espeak_separators(std::string line) {
    line = engine::io::trim_ascii_whitespace(std::move(line));
    std::replace(line.begin(), line.end(), '\n', ' ');
    std::string out;
    out.reserve(line.size());
    bool previous_space = false;
    bool previous_underscore = false;
    for (const char ch : line) {
        if (ch == ' ') {
            if (!previous_space) {
                out.push_back(ch);
            }
            previous_space = true;
            previous_underscore = false;
            continue;
        }
        previous_space = false;
        if (ch == '_') {
            if (!previous_underscore) {
                out.push_back(ch);
            }
            previous_underscore = true;
            continue;
        }
        previous_underscore = false;
        out.push_back(ch);
    }
    line.clear();
    line.reserve(out.size());
    for (size_t index = 0; index < out.size(); ++index) {
        if (out[index] == '_' && index + 1 < out.size() && out[index + 1] == ' ') {
            continue;
        }
        line.push_back(out[index]);
    }
    return line;
}

std::string postprocess_espeak_line(const std::string &raw) {
    const std::string line = collapse_espeak_separators(raw);
    if (line.empty()) {
        return {};
    }
    std::string out;
    size_t start = 0;
    while (start <= line.size()) {
        const size_t end = line.find(' ', start);
        std::string word = line.substr(start, end == std::string::npos ? std::string::npos : end - start);
        word = engine::io::trim_ascii_whitespace(std::move(word));
        if (!word.empty()) {
            word.push_back('_');
            word.erase(std::remove(word.begin(), word.end(), '_'), word.end());
            out += word;
            out.push_back(' ');
        }
        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }
    return out;
}

bool is_ascii_mark(unsigned char ch) {
    switch (ch) {
    case ';':
    case ':':
    case ',':
    case '.':
    case '!':
    case '?':
    case '(':
    case ')':
    case '{':
    case '}':
    case '[':
    case ']':
        return true;
    default:
        return false;
    }
}

std::string phonemize_preserve_punctuation(const std::string &text, const engine::audio::EspeakPhonemizer &phonemizer) {
    struct Mark {
        size_t index = 0;
        std::string mark;
        char position = 'I';
    };
    std::vector<std::string> chunks;
    std::vector<Mark> marks;
    std::string line = text;
    constexpr int kIpaWithUnderscoreSeparator = ('_' << 8) | 0x02;
    size_t pos = 0;
    while (pos < line.size()) {
        size_t match_begin = pos;
        while (match_begin < line.size() && std::isspace(static_cast<unsigned char>(line[match_begin])) != 0) {
            ++match_begin;
        }
        if (match_begin >= line.size() || !is_ascii_mark(static_cast<unsigned char>(line[match_begin]))) {
            ++pos;
            continue;
        }
        size_t match_end = match_begin;
        while (match_end < line.size() && (std::isspace(static_cast<unsigned char>(line[match_end])) != 0 ||
                                           is_ascii_mark(static_cast<unsigned char>(line[match_end])))) {
            ++match_end;
        }
        const size_t begin = pos < match_begin ? pos : match_begin;
        marks.push_back(Mark{0, line.substr(begin, match_end - begin), 'I'});
        pos = match_end;
    }
    if (marks.empty()) {
        return postprocess_espeak_line(phonemizer.phonemize(line, kIpaWithUnderscoreSeparator));
    }
    if (marks.size() == 1 && marks.front().mark == line) {
        return marks.front().mark + " ";
    }
    if (!marks.empty() && line.rfind(marks.front().mark, 0) == 0) {
        marks.front().position = 'B';
    }
    if (!marks.empty() && line.size() >= marks.back().mark.size() &&
        line.compare(line.size() - marks.back().mark.size(), marks.back().mark.size(), marks.back().mark) == 0) {
        marks.back().position = 'E';
    }
    std::string rest = line;
    for (const auto &mark : marks) {
        const size_t found = rest.find(mark.mark);
        if (found == std::string::npos) {
            throw std::runtime_error("Kitten punctuation restoration failed");
        }
        chunks.push_back(rest.substr(0, found));
        rest = rest.substr(found + mark.mark.size());
    }
    chunks.push_back(rest);
    std::vector<std::string> phonemized;
    for (const auto &chunk : chunks) {
        if (!chunk.empty()) {
            phonemized.push_back(postprocess_espeak_line(phonemizer.phonemize(chunk, kIpaWithUnderscoreSeparator)));
        }
    }
    std::vector<std::string> punctuated;
    size_t text_pos = 0;
    size_t mark_pos = 0;
    size_t logical_pos = 0;
    while (text_pos < phonemized.size() || mark_pos < marks.size()) {
        if (mark_pos >= marks.size()) {
            for (; text_pos < phonemized.size(); ++text_pos) {
                if (!phonemized[text_pos].empty() && phonemized[text_pos].back() != ' ') {
                    phonemized[text_pos].push_back(' ');
                }
                punctuated.push_back(phonemized[text_pos]);
            }
            break;
        }
        if (text_pos >= phonemized.size()) {
            punctuated.push_back(marks[mark_pos].mark + " ");
            ++mark_pos;
            break;
        }
        const auto &mark = marks[mark_pos];
        if (mark.index == logical_pos) {
            std::string mark_text = mark.mark;
            std::replace(mark_text.begin(), mark_text.end(), ' ', ' ');
            if (!phonemized[text_pos].empty() && phonemized[text_pos].back() == ' ') {
                phonemized[text_pos].pop_back();
            }
            if (mark.position == 'B') {
                phonemized[text_pos] = mark_text + phonemized[text_pos];
            } else if (mark.position == 'E') {
                punctuated.push_back(phonemized[text_pos] + mark_text + (mark_text.back() == ' ' ? "" : " "));
                ++text_pos;
                ++logical_pos;
            } else if (mark.position == 'A') {
                punctuated.push_back(mark_text + (mark_text.back() == ' ' ? "" : " "));
                ++logical_pos;
            } else {
                if (text_pos + 1 >= phonemized.size()) {
                    phonemized[text_pos] += mark_text;
                } else {
                    const std::string first = phonemized[text_pos];
                    ++text_pos;
                    phonemized[text_pos] = first + mark_text + phonemized[text_pos];
                }
            }
            ++mark_pos;
        } else {
            punctuated.push_back(phonemized[text_pos]);
            ++text_pos;
            ++logical_pos;
        }
    }
    std::string out;
    for (const auto &item : punctuated) {
        out += item;
    }
    return out;
}

std::string basic_english_tokenize_join(const std::string &text) {
    std::vector<std::string> tokens;
    std::string current;
    auto flush = [&]() {
        if (!current.empty()) {
            tokens.push_back(current);
            current.clear();
        }
    };
    for (size_t i = 0; i < text.size();) {
        const unsigned char ch = static_cast<unsigned char>(text[i]);
        if (std::isspace(ch) != 0) {
            flush();
            ++i;
        } else if (ch < 128 && !std::isalnum(ch) && ch != '_') {
            flush();
            tokens.emplace_back(1, static_cast<char>(ch));
            ++i;
        } else {
            size_t width = 1;
            if ((ch & 0xE0u) == 0xC0u)
                width = 2;
            else if ((ch & 0xF0u) == 0xE0u)
                width = 3;
            else if ((ch & 0xF8u) == 0xF0u)
                width = 4;
            current.append(text, i, width);
            i += width;
        }
    }
    flush();
    std::string out;
    for (const auto &token : tokens) {
        if (!out.empty()) {
            out.push_back(' ');
        }
        out += token;
    }
    return out;
}

struct PreparedFrontendText {
    std::string normalized;
    std::string phonemes;
};

PreparedFrontendText prepare_frontend_text(const runtime::Transcript &text, const std::string &language_code,
                                           const KittenAssets &assets,
                                           const engine::audio::EspeakPhonemizer &phonemizer) {
    if (text.text.empty()) {
        throw std::runtime_error("KittenTTS requires non-empty text");
    }
    if (language_code != "en-us") {
        throw std::runtime_error("unsupported Kitten language code: " + language_code + " (English only: en-us)");
    }
    (void)assets;
    const std::string normalized = ensure_kitten_punctuation(text.text);
    return PreparedFrontendText{
        normalized,
        basic_english_tokenize_join(phonemize_preserve_punctuation(normalized, phonemizer)),
    };
}

struct EncodedInputIds {
    std::vector<int32_t> ids;
    size_t phoneme_count = 0;
};

EncodedInputIds encode_input_ids_and_count(const std::string &phonemes, const KittenAssets &assets) {
    EncodedInputIds encoded;
    encoded.ids.reserve(phonemes.size() + 3);
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
            throw std::runtime_error("invalid UTF-8 lead byte in Kitten phoneme string");
        }
        if (i + width > phonemes.size()) {
            throw std::runtime_error("truncated UTF-8 codepoint in Kitten phoneme string");
        }
        for (size_t j = 1; j < width; ++j) {
            const unsigned char byte = static_cast<unsigned char>(phonemes[i + j]);
            if ((byte & 0xC0u) != 0x80u) {
                throw std::runtime_error("invalid UTF-8 continuation byte in Kitten phoneme string");
            }
        }
        const std::string symbol = phonemes.substr(i, width);
        const auto it = assets.vocab.find(symbol);
        if (it == assets.vocab.end()) {
            engine::debug::trace_log_scalar("kitten.skipped_phoneme", std::string_view(symbol));
            i += width;
            continue;
        }
        encoded.ids.push_back(it->second);
        ++encoded.phoneme_count;
        i += width;
    }
    encoded.ids.push_back(10);
    encoded.ids.push_back(0);
    return encoded;
}

std::vector<float> style_for_phoneme_count(const KittenVoicePack &pack, size_t text_length) {
    if (pack.cols != 256) {
        throw std::runtime_error("Kitten voice pack must have 256 columns: " + pack.id);
    }
    if (text_length == 0) {
        throw std::runtime_error("Kitten text must not be empty");
    }
    const size_t row = std::min(text_length, static_cast<size_t>(pack.rows - 1));
    const size_t offset = row * static_cast<size_t>(pack.cols);
    std::vector<float> style(static_cast<size_t>(pack.cols));
    std::memcpy(style.data(), pack.values.data() + offset, static_cast<size_t>(pack.cols) * sizeof(float));
    return style;
}

float resolve_speaking_rate(const std::optional<runtime::VoiceCondition> &voice) {
    if (!voice.has_value() || !voice->style.has_value() || !voice->style->speaking_rate.has_value()) {
        return 1.0f;
    }
    const float rate = *voice->style->speaking_rate;
    if (!(rate > 0.0f)) {
        throw std::runtime_error("Kitten speaking_rate must be positive");
    }
    return rate;
}

float apply_speed_prior(float speaking_rate, const std::string &voice_id, const KittenAssets &assets) {
    const auto it = assets.speed_priors.find(voice_id);
    if (it == assets.speed_priors.end()) {
        return speaking_rate;
    }
    return speaking_rate * it->second;
}

} // namespace

KittenFrontendSessionState resolve_kitten_frontend_session_state(const std::optional<runtime::Transcript> &text,
                                                                 const std::optional<runtime::VoiceCondition> &voice,
                                                                 const KittenAssets &assets) {
    runtime::Transcript transcript;
    if (text.has_value()) {
        transcript = *text;
    }
    KittenFrontendSessionState state;
    state.voice_id = resolve_voice_id(voice, assets);
    state.language_code = resolve_language_code(transcript, voice, state.voice_id);
    const auto voice_it = assets.voices.find(state.voice_id);
    if (voice_it == assets.voices.end()) {
        throw std::runtime_error("unknown Kitten voice id: " + state.voice_id);
    }
    state.voice_pack = &voice_it->second;
    state.speaking_rate = apply_speed_prior(resolve_speaking_rate(voice), state.voice_id, assets);
    return state;
}

void validate_kitten_frontend_session_state(const runtime::Transcript &text,
                                            const std::optional<runtime::VoiceCondition> &voice,
                                            const KittenFrontendSessionState &state, const KittenAssets &assets) {
    const std::string resolved_voice_id = resolve_voice_id(voice, assets);
    if (resolved_voice_id != state.voice_id) {
        throw std::runtime_error("Kitten session voice_id changed after launch: " + state.voice_id + " -> " +
                                 resolved_voice_id);
    }
    const std::string resolved_language_code = resolve_language_code(text, voice, state.voice_id);
    if (resolved_language_code != state.language_code) {
        throw std::runtime_error("Kitten session language_code changed after launch: " + state.language_code + " -> " +
                                 resolved_language_code);
    }
    const auto voice_it = assets.voices.find(state.voice_id);
    if (voice_it == assets.voices.end() || &voice_it->second != state.voice_pack) {
        throw std::runtime_error("Kitten session voice pack changed after launch");
    }
    const float resolved_speaking_rate = apply_speed_prior(resolve_speaking_rate(voice), state.voice_id, assets);
    if (resolved_speaking_rate != state.speaking_rate) {
        throw std::runtime_error("Kitten session speaking_rate changed after launch");
    }
}

KittenSynthesisInput build_kitten_synthesis_input(const runtime::Transcript &text,
                                                  const KittenFrontendSessionState &state, const KittenAssets &assets,
                                                  const engine::audio::EspeakPhonemizer &phonemizer) {
    if (state.voice_pack == nullptr) {
        throw std::runtime_error("Kitten frontend session voice pack was not prepared");
    }
    const PreparedFrontendText prepared = prepare_frontend_text(text, state.language_code, assets, phonemizer);
    const EncodedInputIds encoded = encode_input_ids_and_count(prepared.phonemes, assets);
    if (encoded.phoneme_count > 510) {
        throw std::runtime_error("Kitten phoneme string exceeds 510 symbols; segmenting is not "
                                 "implemented in the framework path yet");
    }
    KittenSynthesisInput input;
    input.voice_id = state.voice_id;
    input.language_code = state.language_code;
    input.phonemes = prepared.phonemes;
    input.normalized_text_length = prepared.normalized.size();
    input.input_ids = encoded.ids;
    input.style = style_for_phoneme_count(*state.voice_pack, prepared.normalized.size());
    input.speaking_rate = state.speaking_rate;
    if (static_cast<int64_t>(input.input_ids.size()) > assets.context_length) {
        throw std::runtime_error("Kitten tokenized input exceeds model context length");
    }
    return input;
}

int64_t estimate_kitten_request_tokens(const runtime::SessionPreparationRequest &request,
                                       const KittenFrontendSessionState &state, const KittenAssets &assets,
                                       const engine::audio::EspeakPhonemizer &phonemizer) {
    if (!request.text.has_value()) {
        return 0;
    }
    const auto input = build_kitten_synthesis_input(*request.text, state, assets, phonemizer);
    return static_cast<int64_t>(input.input_ids.size());
}

} // namespace engine::models::kitten_tts
