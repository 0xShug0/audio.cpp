// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Adapted for audio.cpp from NVIDIA/NeMo-Speech.cpp, revision 66d50d91b75a406becd95fbaf39d2c9c13e95d7c.
// Build this file with floating-point contraction disabled: every float operation rounds as in
// NumPy and PyTorch, and the fused ones are explicit std::fma calls.
// numpy_log, pairwise_sum and torch_linspace reproduce NumPy and PyTorch routines, and
// linear_sum_assignment ports SciPy's (BSD 3-Clause; THIRD_PARTY_NOTICES.md).
#include "engine/community_models/nemotron_transcribe/alignment.h"

#include <sentencepiece_processor.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>

namespace engine::models::nemotron_transcribe::timestamps {
namespace {

size_t checked_elements(int rows, int columns) {
    if (rows < 0 || columns < 0 ||
        (columns && size_t(rows) > std::vector<float>().max_size() / size_t(columns)))
        throw std::invalid_argument("alignment: invalid matrix shape");
    return size_t(rows) * size_t(columns);
}

void require_finite(const float* data, size_t count) {
    if (std::any_of(data, data + count, [](float value) { return !std::isfinite(value); }))
        throw std::invalid_argument("alignment: non-finite speaker probabilities");
}

void require_log_values(const float* data, size_t count) {
    // Negative infinity is a valid impossible emission/prior. NaN and positive
    // infinity must not enter argmax, dynamic programming, or time conversion.
    if (std::any_of(data, data + count, [](float value) {
            return std::isnan(value) || value == std::numeric_limits<float>::infinity();
        }))
        throw std::invalid_argument("alignment: invalid log probabilities");
}

// Decodes the UTF-8 code point at text[i] and its byte length; an invalid byte is itself (length
// 1).
uint32_t
code_point_at(const std::string& text, size_t i, size_t& length) {
    const auto byte = [&](size_t k) { return static_cast<unsigned char>(text[k]); };
    const unsigned char b0 = byte(i);
    length = 1;
    if (b0 < 0x80)
        return b0;
    size_t n = 0;
    uint32_t cp = 0;
    if ((b0 & 0xe0) == 0xc0 && b0 >= 0xc2)
        n = 2, cp = b0 & 0x1f;
    else if ((b0 & 0xf0) == 0xe0)
        n = 3, cp = b0 & 0x0f;
    else if ((b0 & 0xf8) == 0xf0 && b0 <= 0xf4)
        n = 4, cp = b0 & 0x07;
    else
        return 0xfffd;
    if (i + n > text.size())
        return 0xfffd;
    for (size_t k = 1; k < n; ++k) {
        if ((byte(i + k) & 0xc0) != 0x80)
            return 0xfffd;
        cp = (cp << 6) | (byte(i + k) & 0x3f);
    }
    if ((n == 3 && (cp < 0x800 || (cp >= 0xd800 && cp <= 0xdfff))) ||
        (n == 4 && (cp < 0x10000 || cp > 0x10ffff)))
        return 0xfffd;
    length = n;
    return cp;
}

// Python str.isspace().
bool
python_space(uint32_t cp) {
    return (cp >= 0x09 && cp <= 0x0d) || (cp >= 0x1c && cp <= 0x20) || cp == 0x85 || cp == 0xa0 ||
           cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200a) || cp == 0x2028 || cp == 0x2029 ||
           cp == 0x202f || cp == 0x205f || cp == 0x3000;
}

// The digit zeros of Unicode 15.0 general category Nd (Python's \d and int()); each starts a run of
// ten digits.
constexpr uint32_t kDigitZeros[] = {
    0x30,    0x660,   0x6f0,   0x7c0,   0x966,   0x9e6,   0xa66,   0xae6,   0xb66,   0xbe6,
    0xc66,   0xce6,   0xd66,   0xde6,   0xe50,   0xed0,   0xf20,   0x1040,  0x1090,  0x17e0,
    0x1810,  0x1946,  0x19d0,  0x1a80,  0x1a90,  0x1b50,  0x1bb0,  0x1c40,  0x1c50,  0xa620,
    0xa8d0,  0xa900,  0xa9d0,  0xa9f0,  0xaa50,  0xabf0,  0xff10,  0x104a0, 0x10d30, 0x11066,
    0x110f0, 0x11136, 0x111d0, 0x112f0, 0x11450, 0x114d0, 0x11650, 0x116c0, 0x11730, 0x118e0,
    0x11950, 0x11c50, 0x11d50, 0x11da0, 0x11f50, 0x16a60, 0x16ac0, 0x16b50, 0x1d7ce, 0x1d7d8,
    0x1d7e2, 0x1d7ec, 0x1d7f6, 0x1e140, 0x1e2f0, 0x1e4f0, 0x1e950, 0x1fbf0};

// The decimal value of a Unicode digit, -1 for any other code point.
int
digit_value(uint32_t cp) {
    const auto* it = std::upper_bound(std::begin(kDigitZeros), std::end(kDigitZeros), cp);
    if (it == std::begin(kDigitZeros))
        return -1;
    const uint32_t zero = *(it - 1);
    return cp - zero < 10 ? static_cast<int>(cp - zero) : -1;
}

// NeMo's _SPEAKER_TAG_RE, `<spk:(\d+)>` with re.IGNORECASE, at text[i]: Python folds "s" with
// U+017F and "k" with U+212A (KELVIN SIGN). Sets the end byte and the tag number (saturated).
bool
speaker_tag_at(const std::string& text, size_t i, size_t& end, int& tag) {
    if (text[i] != '<')
        return false;
    size_t j = i + 1, length = 1;
    const auto next = [&]() {
        if (j >= text.size())
            return uint32_t(0);
        const uint32_t cp = code_point_at(text, j, length);
        j += length;
        return cp;
    };
    uint32_t cp = next();
    if (cp != 's' && cp != 'S' && cp != 0x17f)
        return false;
    cp = next();
    if (cp != 'p' && cp != 'P')
        return false;
    cp = next();
    if (cp != 'k' && cp != 'K' && cp != 0x212a)
        return false;
    if (next() != ':')
        return false;
    long long value = 0;
    int digits = 0;
    for (;;) {
        const size_t at = j;
        cp = next();
        const int d = digit_value(cp);
        if (d < 0) {
            j = at;
            break;
        }
        value = std::min<long long>(value * 10 + d, INT_MAX);
        ++digits;
    }
    if (!digits || j >= text.size() || text[j] != '>')
        return false;
    end = j + 1;
    tag = static_cast<int>(value);
    return true;
}

bool
speaker_tag_token(const std::string& token) {
    size_t end = 0;
    int tag = 0;
    return !token.empty() && speaker_tag_at(token, 0, end, tag) && end == token.size();
}

// `<(?:ev|sty):[^<>]*>` over the whole token.
bool
event_or_style_token(const std::string& token) {
    const size_t prefix = token.rfind("<ev:", 0) == 0 ? 4 : token.rfind("<sty:", 0) == 0 ? 5 : 0;
    return prefix && token.size() > prefix && token.back() == '>' &&
           token.find_first_of("<>", prefix) == token.size() - 1;
}

std::string
join_words(const std::vector<std::string>& words) {
    std::string out;
    for (const auto& word : words) {
        if (!out.empty())
            out += ' ';
        out += word;
    }
    return out;
}

}  // namespace

std::vector<std::string>
split_whitespace(const std::string& text) {
    std::vector<std::string> words;
    std::string current;
    for (size_t i = 0; i < text.size();) {
        size_t length = 1;
        const uint32_t cp = code_point_at(text, i, length);
        if (python_space(cp)) {
            if (!current.empty())
                words.push_back(std::move(current)), current.clear();
        } else {
            current.append(text, i, length);
        }
        i += length;
    }
    if (!current.empty())
        words.push_back(std::move(current));
    return words;
}

std::string
clean_sot_transcript(
    const std::string& transcript, const std::vector<std::string>& special_tokens) {
    std::string text = transcript;
    for (const auto& token : special_tokens) {
        if (token.empty() || speaker_tag_token(token) || event_or_style_token(token))
            continue;
        std::string replaced;
        size_t cursor = 0;
        for (size_t at; (at = text.find(token, cursor)) != std::string::npos;
             cursor = at + token.size())
            replaced.append(text, cursor, at - cursor).append(" ");
        replaced.append(text, cursor, std::string::npos);
        text = std::move(replaced);
    }
    return join_words(split_whitespace(text));
}

std::vector<TranscriptWord>
parse_sot_words(const std::string& transcript) {
    // Em dashes separate words.
    std::string text;
    for (size_t cursor = 0;;) {
        const size_t at = transcript.find("\xE2\x80\x94", cursor);
        text.append(transcript, cursor, at == std::string::npos ? std::string::npos : at - cursor);
        if (at == std::string::npos)
            break;
        text += ' ';
        cursor = at + 3;
    }
    // re.split: the parts between tags and the tags.
    std::vector<std::pair<std::string, int>> parts;  // text, tag in force after it (-2 = none)
    size_t cursor = 0;
    for (size_t i = 0; i < text.size(); ++i) {
        size_t end = 0;
        int tag = 0;
        if (speaker_tag_at(text, i, end, tag)) {
            parts.emplace_back(text.substr(cursor, i - cursor), tag);
            cursor = end;
            i = end - 1;
        }
    }
    parts.emplace_back(text.substr(cursor), -2);
    int tag = parts.size() == 1 ? 0 : -1;
    std::vector<TranscriptWord> words;
    for (const auto& [part, next] : parts) {
        for (auto& text_word : split_whitespace(part)) {
            TranscriptWord word;
            word.word = std::move(text_word);
            word.tag = tag;
            word.index = static_cast<int>(words.size());
            words.push_back(std::move(word));
        }
        tag = next;
    }
    return words;
}

std::string
speaker_label(int tag) {
    return tag < 0 ? std::string("unknown") : "spk:" + std::to_string(tag);
}

namespace {

struct CodePointRange {
    uint32_t first, last;
};
#include "timestamp_unit_tables.inc"

template <size_t N>
bool
in_ranges(const CodePointRange (&table)[N], uint32_t cp) {
    const auto* it = std::upper_bound(
        table, table + N, cp, [](uint32_t c, const CodePointRange& r) { return c < r.first; });
    return it != table && cp <= (it - 1)->last;
}

std::vector<uint32_t>
code_points(const std::string& text) {
    std::vector<uint32_t> out;
    for (size_t i = 0; i < text.size();) {
        size_t length = 1;
        out.push_back(code_point_at(text, i, length));
        i += length;
    }
    return out;
}

void
append_utf8(std::string& out, uint32_t cp) {
    if (cp < 0x80) {
        out += static_cast<char>(cp);
    } else if (cp < 0x800) {
        out += static_cast<char>(0xc0 | (cp >> 6));
        out += static_cast<char>(0x80 | (cp & 0x3f));
    } else if (cp < 0x10000) {
        out += static_cast<char>(0xe0 | (cp >> 12));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3f));
        out += static_cast<char>(0x80 | (cp & 0x3f));
    } else {
        out += static_cast<char>(0xf0 | (cp >> 18));
        out += static_cast<char>(0x80 | ((cp >> 12) & 0x3f));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3f));
        out += static_cast<char>(0x80 | (cp & 0x3f));
    }
}

// NeMo's event tag: `<[^<>]+>` over the whole word.
bool
event_tag(const std::string& word) {
    return word.size() >= 3 && word.front() == '<' && word.back() == '>' &&
           word.find_first_of("<>", 1) == word.size() - 1;
}

std::string
joined(const std::vector<std::string>& parts) {
    std::string out;
    for (const auto& part : parts) out += part;
    return out;
}

}  // namespace

std::vector<std::string>
lexical_characters(const std::string& text) {
    std::vector<std::string> out;
    for (uint32_t cp : code_points(text))
        if (in_ranges(kLexical, cp)) {
            out.emplace_back();
            append_utf8(out.back(), cp);
        }
    return out;
}

std::string
timestamp_script(const std::vector<TranscriptWord>& words) {
    bool ko = false, ja = false, zh = false, th = false;
    for (const auto& w : words) {
        if (event_tag(w.word))
            continue;
        for (uint32_t cp : code_points(w.word)) {
            ko = ko || in_ranges(kHangul, cp);
            ja = ja || in_ranges(kKana, cp);
            zh = zh || in_ranges(kHan, cp);
            th = th || in_ranges(kThai, cp);
        }
    }
    return ko ? "ko" : ja ? "ja" : zh ? "zh" : th ? "th" : "";
}

std::vector<TranscriptWord>
timestamp_units(const std::vector<TranscriptWord>& words) {
    const std::string script = timestamp_script(words);
    if (script != "zh" && script != "ja" && script != "th")
        return words;
    const UnitKind kind = script == "th" ? UnitKind::cluster : UnitKind::character;
    std::vector<TranscriptWord> out;
    std::string pending;  // running text of words without units, before the first unit
    const auto start_unit = [&](TranscriptWord unit) {
        if (!pending.empty())
            unit.display = pending + ' ' + unit.display;
        pending.clear();
        unit.index = static_cast<int>(out.size());
        out.push_back(std::move(unit));
    };
    for (const auto& word : words) {
        if (event_tag(word.word)) {
            TranscriptWord tag = word;
            tag.kind = UnitKind::word;
            tag.display = word.word;
            start_unit(std::move(tag));
            continue;
        }
        const size_t first = out.size();
        std::string prefix;  // punctuation before the word's first unit
        for (uint32_t cp : code_points(word.word)) {
            std::string character;
            append_utf8(character, cp);
            if (kind == UnitKind::cluster && out.size() > first && in_ranges(kMark, cp)) {
                out.back().word += character;
                out.back().display += character;
            } else if (in_ranges(kLexical, cp)) {
                TranscriptWord unit{character, word.tag,           0,
                                    kind,      prefix + character, out.size() > first};
                prefix.clear();
                start_unit(std::move(unit));
            } else if (out.size() > first) {
                out.back().display += character;
            } else {
                prefix += character;
            }
        }
        if (out.size() == first) {  // no lexical character: the text follows the previous unit
            if (out.empty())
                pending += (pending.empty() ? "" : " ") + prefix;
            else
                out.back().display += ' ' + prefix;
        }
    }
    for (auto& unit : out)
        if (unit.display == unit.word)
            unit.display.clear();
    return out;
}

std::vector<int>
character_token_ids(
    const sentencepiece::SentencePieceProcessor& tokenizer, const std::string& unit, int blank,
    bool normalized) {
    std::vector<int> ids;
    const int own = tokenizer.PieceToId(unit);
    if (own != tokenizer.unk_id()) {
        ids.push_back(own);
    } else {
        std::vector<int> encoded;
        if (!tokenizer.Encode(unit, &encoded).ok())
            throw std::runtime_error("SentencePiece could not encode a character");
        for (int id : encoded)
            if (tokenizer.IdToPiece(id) != "\xE2\x96\x81")
                ids.push_back(id);
    }
    bool ok = !ids.empty() && std::find(ids.begin(), ids.end(), tokenizer.unk_id()) == ids.end() &&
              *std::min_element(ids.begin(), ids.end()) >= 0 &&
              *std::max_element(ids.begin(), ids.end()) < blank;
    std::string decoded, expected = unit;
    if (ok)
        ok = tokenizer.Decode(ids, &decoded).ok();
    if (ok && normalized) {
        std::string normal;
        ok = tokenizer.Normalize(unit, &normal).ok();
        expected = joined(lexical_characters(normal));
    }
    if (!ok || joined(lexical_characters(decoded)) != expected)
        throw std::runtime_error("Unrepresentable character: '" + unit + "'");  // NeMo's repr
    return ids;
}

std::shared_ptr<const sentencepiece::SentencePieceProcessor>
load_tokenizer(const std::string& path) {
    auto tokenizer = std::make_shared<sentencepiece::SentencePieceProcessor>();
    if (!tokenizer->Load(path).ok())
        throw std::runtime_error("cannot load SentencePiece model " + path);
    return tokenizer;
}

namespace {

std::vector<int>
encode(const sentencepiece::SentencePieceProcessor& tokenizer, const std::string& text) {
    std::vector<int> ids;
    if (!tokenizer.Encode(text, &ids).ok())
        throw std::runtime_error("SentencePiece could not encode the transcript");
    return ids;
}

// NeMo's _tokenize_word_stream over the units `members` of one speaker stream.
std::vector<std::vector<int>>
tokenize_stream(
    const sentencepiece::SentencePieceProcessor& tokenizer,
    const std::vector<TranscriptWord>& units, const std::vector<int>& members, int blank) {
    std::vector<std::vector<int>> ids_by_word;
    if (std::any_of(members.begin(), members.end(), [&](int w) {
            return units[w].kind != UnitKind::word;
        })) {
        for (int w : members)
            ids_by_word.push_back(
                units[w].kind != UnitKind::word
                    ? character_token_ids(
                          tokenizer, units[w].word, blank, units[w].kind == UnitKind::cluster)
                    : tokenize_stream(tokenizer, units, {w}, blank).front());
        return ids_by_word;
    }
    std::vector<std::string> words;
    for (int w : members) words.push_back(units[w].word);
    const auto stream = encode(tokenizer, join_words(words));
    if (stream.empty() || *std::min_element(stream.begin(), stream.end()) < 0 ||
        *std::max_element(stream.begin(), stream.end()) >= blank)
        throw std::runtime_error("Tokenizer produced IDs outside the non-blank CTC vocabulary.");
    std::vector<int> flat;
    for (const auto& word : words) {
        ids_by_word.push_back(encode(tokenizer, word));
        if (ids_by_word.back().empty())
            throw std::runtime_error("SentencePiece produced an empty word tokenization.");
        flat.insert(flat.end(), ids_by_word.back().begin(), ids_by_word.back().end());
    }
    if (flat != stream)
        throw std::runtime_error("Per-word and full-stream SentencePiece tokenization disagree.");
    return ids_by_word;
}

}  // namespace

AlignmentTargets
alignment_targets(
    const sentencepiece::SentencePieceProcessor& tokenizer, const std::string& transcript,
    int blank, int ctc_frames) {
    AlignmentTargets out;
    out.words = timestamp_units(parse_sot_words(transcript));
    for (int w = 0; w < static_cast<int>(out.words.size()); ++w) {
        const int tag = out.words[w].tag;
        auto it = std::find_if(out.streams.begin(), out.streams.end(), [&](const SpeakerStream& s) {
            return s.tag == tag;
        });
        if (it == out.streams.end()) {
            out.streams.emplace_back();
            out.streams.back().tag = tag;
            it = out.streams.end() - 1;
        }
        it->words.push_back(w);
    }
    // Every stream is tokenized before any target is built (NeMo's _tokenize_words).
    std::vector<std::vector<std::vector<int>>> ids;
    for (const auto& stream : out.streams)
        ids.push_back(tokenize_stream(tokenizer, out.words, stream.words, blank));
    std::vector<int> tokens;
    for (size_t s = 0; s < out.streams.size(); ++s) {
        auto& stream = out.streams[s];
        stream.labels.assign(1, blank);
        stream.state_to_word.assign(1, -1);
        for (size_t w = 0; w < ids[s].size(); ++w)
            for (int id : ids[s][w]) {
                stream.labels.insert(stream.labels.end(), {id, blank});
                stream.state_to_word.insert(stream.state_to_word.end(), {static_cast<int>(w), -1});
                tokens.push_back(id);
            }
        if (minimum_frames(stream.labels) > ctc_frames)
            throw std::runtime_error(
                "Speaker " + (stream.tag < 0 ? std::string("None") : std::to_string(stream.tag)) +
                " transcript is too long for the CTC timeline.");
    }
    std::sort(tokens.begin(), tokens.end());
    tokens.erase(std::unique(tokens.begin(), tokens.end()), tokens.end());
    tokens.erase(std::remove(tokens.begin(), tokens.end(), blank), tokens.end());
    out.columns.assign(1, blank);
    out.columns.insert(out.columns.end(), tokens.begin(), tokens.end());
    for (auto& stream : out.streams)
        for (int& label : stream.labels)
            label = label == blank ? 0
                                   : 1 + static_cast<int>(
                                             std::lower_bound(tokens.begin(), tokens.end(), label) -
                                             tokens.begin());
    return out;
}

int
minimum_frames(const std::vector<int>& labels) {
    int frames = 0;
    for (size_t s = 1; s < labels.size(); s += 2) {
        ++frames;
        if (s >= 3 && labels[s] == labels[s - 2])
            ++frames;
    }
    return frames;
}

std::vector<HeadWindow>
head_windows(int frames, int core, int left, int right) {
    std::vector<HeadWindow> windows;
    if (left < 0 || right < 0)
        throw std::invalid_argument("head_windows: negative context");
    if (frames <= 0)
        return windows;
    if (core <= 0 || frames <= core)
        return {{0, frames, 0, frames}};
    for (int start = 0; start < frames;) {
        const int end = static_cast<int>(std::min<int64_t>(int64_t(start) + core, frames));
        windows.push_back({start, end, std::max(start - left, 0),
            static_cast<int>(std::min<int64_t>(int64_t(end) + right, frames))});
        start = end;
    }
    return windows;
}

// ---- numerics ----

float
numpy_log(float x) {
    // NumPy's simd_log_FLOAT: x = m * 2^k with m in (sqrt(1/2), sqrt(2)], log(m) = P(m-1)/Q(m-1)
    // (degree 5 each, Horner with FMA), plus k * ln 2.
    constexpr float P[] = {
        0.0f,
        9.999999999999998702521e-01f,
        2.112677543073053063722e+00f,
        1.480000633576506585156e+00f,
        3.808837741388407920751e-01f,
        2.589979117907922693523e-02f};
    constexpr float Q[] = {
        1.0f,
        2.612677543073109236779e+00f,
        2.453006071784736363091e+00f,
        9.864942958519418960339e-01f,
        1.546476374983906719538e-01f,
        5.875095259e-03f};
    constexpr float kLn2 = 0.693147180559945309417232121458176568f;
    constexpr float kSqrtHalf = 0.707106781186547524400844362104849039f;
    if (std::isnan(x))
        return x;
    if (x < 0.0f)
        return -std::numeric_limits<float>::quiet_NaN();
    if (x == 0.0f)
        return -std::numeric_limits<float>::infinity();
    if (std::isinf(x))
        return x;
    const auto bits = [](float f) {
        uint32_t b;
        std::memcpy(&b, &f, sizeof b);
        return b;
    };
    const auto from_bits = [](uint32_t b) {
        float f;
        std::memcpy(&f, &b, sizeof f);
        return f;
    };
    const bool denormal = x < std::numeric_limits<float>::min();
    if (denormal)
        x = x * from_bits(0x71800000u);  // 2^100
    float exponent = static_cast<float>(static_cast<int32_t>(bits(x) >> 23) - 0x7e);
    if (denormal)
        exponent = exponent - 100.0f;
    float m = from_bits((bits(x) & 0x7fffffu) | (126u << 23));  // [0.5, 1)
    if (m <= kSqrtHalf) {
        m = m + m;
        exponent = exponent - 1.0f;
    }
    m = m - 1.0f;
    float num = std::fma(P[5], m, P[4]), den = std::fma(Q[5], m, Q[4]);
    for (int k = 3; k >= 0; --k) {
        num = std::fma(num, m, P[k]);
        den = std::fma(den, m, Q[k]);
    }
    return std::fma(exponent, kLn2, num / den);
}

namespace {

// torch.linspace(start, end, steps)[idx] in float32 as PyTorch's CPU kernel computes it: the lower
// half from the start, the upper half fused from the end. Checked with start == 0.
float
torch_linspace(float start, float end, int64_t steps, int64_t idx) {
    if (steps == 1)
        return start;
    const float step = (end - start) / static_cast<float>(steps - 1);
    return idx < steps / 2 ? start + step * static_cast<float>(idx)
                           : std::fma(-step, static_cast<float>(steps - idx - 1), end);
}

// NumPy's pairwise summation (add.reduce over a contiguous axis).
double
pairwise_sum(const double* a, size_t n) {
    if (n < 8) {
        double res = 0.0;
        for (size_t i = 0; i < n; ++i) res += a[i];
        return res;
    }
    if (n <= 128) {
        double r[8];
        for (int k = 0; k < 8; ++k) r[k] = a[k];
        size_t i = 8;
        for (; i < n - n % 8; i += 8)
            for (int k = 0; k < 8; ++k) r[k] += a[i + k];
        double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) res += a[i];
        return res;
    }
    size_t n2 = n / 2;
    n2 -= n2 % 8;
    return pairwise_sum(a, n2) + pairwise_sum(a + n2, n - n2);
}

// A dot product as OpenBLAS's Haswell dgemm accumulates it: one FMA chain per K block of 256,
// a remainder under 512 split in halves rounded up to 4, the blocks added in order.
double
blas_dot(const double* a, const double* b, size_t n) {
    constexpr size_t kQ = 256, kUnroll = 4;
    double c = 0.0;
    for (size_t ls = 0; ls < n;) {
        size_t block = n - ls;
        if (block >= 2 * kQ)
            block = kQ;
        else if (block > kQ)
            block = (block / 2 + kUnroll - 1) / kUnroll * kUnroll;
        double acc = 0.0;
        for (size_t k = ls; k < ls + block; ++k) acc = std::fma(a[k], b[k], acc);
        c = ls ? c + acc : acc;
        ls += block;
    }
    return c;
}

}  // namespace

std::vector<float>
resample_speaker_probs(const float* probs, int rows, int speakers, int frames) {
    if (!probs || rows < 1 || speakers < 0 || frames < 0)
        throw std::invalid_argument("resample_speaker_probs: invalid shape");
    const size_t count = checked_elements(rows, speakers);
    checked_elements(frames, speakers);
    require_finite(probs, count);
    std::vector<float> clamped(probs, probs + count);
    for (float& v : clamped) v = std::min(std::max(v, 0.0f), 1.0f);  // torch.clamp keeps NaN
    if (rows == frames)
        return clamped;
    std::vector<float> out(static_cast<size_t>(frames) * speakers);
    for (int t = 0; t < frames; ++t) {
        const float position = torch_linspace(0.0f, static_cast<float>(rows - 1), frames, t);
        const float lower_f = std::floor(position), upper_f = std::ceil(position);
        const float fraction = position - lower_f;
        const size_t lower = std::min(static_cast<size_t>(lower_f), size_t(rows - 1)),
                     upper = std::min(static_cast<size_t>(upper_f), size_t(rows - 1));
        for (int c = 0; c < speakers; ++c) {
            const float a = clamped[lower * speakers + c] * (1.0f - fraction);
            const float b = clamped[upper * speakers + c] * fraction;
            out[static_cast<size_t>(t) * speakers + c] = rows == 1 ? clamped[c] : a + b;
        }
    }
    return out;
}

std::vector<std::pair<int, int>>
linear_sum_assignment(const std::vector<double>& cost_in, int nr, int nc) {
    // Port of SciPy's scipy/optimize/rectangular_lsap/rectangular_lsap.cpp (Crouse's shortest
    // augmenting path; BSD 3-Clause, THIRD_PARTY_NOTICES.md).
    std::vector<std::pair<int, int>> out;
    if (nr <= 0 || nc <= 0)
        return out;
    if (cost_in.size() != static_cast<size_t>(nr) * nc)
        throw std::invalid_argument("linear_sum_assignment: cost size");
    const bool transpose = nc < nr;
    std::vector<double> cost(cost_in);
    if (transpose) {
        for (int i = 0; i < nr; ++i)
            for (int j = 0; j < nc; ++j)
                cost[static_cast<size_t>(j) * nr + i] = cost_in[static_cast<size_t>(i) * nc + j];
        std::swap(nr, nc);
    }
    for (double v : cost)
        if (std::isnan(v) || v == -std::numeric_limits<double>::infinity())
            throw std::invalid_argument("matrix contains invalid numeric entries");
    const double inf = std::numeric_limits<double>::infinity();
    std::vector<double> u(nr, 0.0), v(nc, 0.0), shortest(nc);
    std::vector<int> path(nc, -1), col4row(nr, -1), row4col(nc, -1), remaining(nc);
    std::vector<char> SR(nr), SC(nc);
    for (int cur = 0; cur < nr; ++cur) {
        double min_val = 0.0;
        int i = cur, sink = -1, num_remaining = nc;
        for (int it = 0; it < nc; ++it) remaining[it] = nc - it - 1;
        std::fill(SR.begin(), SR.end(), 0);
        std::fill(SC.begin(), SC.end(), 0);
        std::fill(shortest.begin(), shortest.end(), inf);
        while (sink == -1) {
            int index = -1;
            double lowest = inf;
            SR[i] = 1;
            for (int it = 0; it < num_remaining; ++it) {
                const int j = remaining[it];
                const double r = min_val + cost[static_cast<size_t>(i) * nc + j] - u[i] - v[j];
                if (r < shortest[j]) {
                    path[j] = i;
                    shortest[j] = r;
                }
                if (shortest[j] < lowest || (shortest[j] == lowest && row4col[j] == -1)) {
                    lowest = shortest[j];
                    index = it;
                }
            }
            min_val = lowest;
            if (min_val == inf)
                throw std::invalid_argument("cost matrix is infeasible");
            const int j = remaining[index];
            if (row4col[j] == -1)
                sink = j;
            else
                i = row4col[j];
            SC[j] = 1;
            remaining[index] = remaining[--num_remaining];
        }
        u[cur] += min_val;
        for (int r = 0; r < nr; ++r)
            if (SR[r] && r != cur)
                u[r] += min_val - shortest[col4row[r]];
        for (int c = 0; c < nc; ++c)
            if (SC[c])
                v[c] -= min_val - shortest[c];
        for (int j = sink;;) {
            const int r = path[j];
            row4col[j] = r;
            std::swap(col4row[r], j);
            if (r == cur)
                break;
        }
    }
    if (transpose) {
        // argsort of col4row: original rows ascending.
        std::vector<int> order(nr);
        for (int k = 0; k < nr; ++k) order[k] = k;
        std::sort(
            order.begin(), order.end(), [&](int a, int b) { return col4row[a] < col4row[b]; });
        for (int k : order) out.emplace_back(col4row[k], k);
    } else {
        for (int r = 0; r < nr; ++r) out.emplace_back(r, col4row[r]);
    }
    return out;
}

// ---- alignment ----

std::vector<int>
speaker_mapping(
    const std::vector<int>& greedy, int columns, const std::vector<SpeakerStream>& streams,
    const float* speaker, int frames, int speakers, float threshold) {
    if (columns < 1 || frames < 0 || speakers < 0 || greedy.size() != size_t(frames) ||
        !std::isfinite(threshold) || threshold < 0 || threshold > 1 ||
        (speakers > 0 && frames > 0 && !speaker))
        throw std::invalid_argument("speaker_mapping: invalid inputs");
    checked_elements(frames, speakers);
    for (int label : greedy)
        if (label < 0 || label >= columns)
            throw std::invalid_argument("speaker_mapping: invalid greedy class ID");
    for (const auto& stream : streams)
        for (int label : stream.labels)
            if (label < 0 || label >= columns)
                throw std::invalid_argument("speaker_mapping: invalid target class ID");
    std::vector<int> mapping(streams.size(), -1);
    if (streams.empty() || speakers == 0)
        return mapping;
    if (frames) require_finite(speaker, checked_elements(frames, speakers));
    const size_t C = static_cast<size_t>(columns), S = static_cast<size_t>(speakers),
                 N = streams.size();
    // Greedy non-blank onsets per active column, and target token counts per stream.
    std::vector<int64_t> observed(S * C, 0), expected(N * C, 0);
    for (int t = 0; t < frames; ++t) {
        if (greedy[t] == 0 || (t > 0 && greedy[t] == greedy[t - 1]))
            continue;
        for (size_t c = 0; c < S; ++c)
            if (speaker[static_cast<size_t>(t) * S + c] >= threshold)
                ++observed[c * C + greedy[t]];
    }
    for (size_t s = 0; s < N; ++s)
        for (size_t k = 1; k < streams[s].labels.size(); k += 2)
            ++expected[s * C + streams[s].labels[k]];
    std::vector<double> idf(C);
    for (size_t k = 0; k < C; ++k) {
        int64_t present = 0;
        for (size_t c = 0; c < S; ++c) present += observed[c * C + k] > 0;
        idf[k] = std::log1p(static_cast<double>(S) / static_cast<double>(1 + present));
    }
    std::vector<double> a(N * C), b(S * C), squares(C);
    for (size_t s = 0; s < N; ++s)
        for (size_t k = 0; k < C; ++k)
            a[s * C + k] = std::sqrt(static_cast<double>(expected[s * C + k]) * idf[k]);
    for (size_t c = 0; c < S; ++c)
        for (size_t k = 0; k < C; ++k)
            b[c * C + k] = std::sqrt(static_cast<double>(observed[c * C + k]) * idf[k]);
    const auto norm = [&](const double* row) {
        for (size_t k = 0; k < C; ++k) squares[k] = row[k] * row[k];
        return std::sqrt(pairwise_sum(squares.data(), C));
    };
    std::vector<double> norm_b(S);
    std::vector<int64_t> total_b(S, 0);
    for (size_t c = 0; c < S; ++c) {
        norm_b[c] = norm(&b[c * C]);
        for (size_t k = 0; k < C; ++k) total_b[c] += observed[c * C + k];
    }
    std::vector<double> cost(N * S);
    for (size_t s = 0; s < N; ++s) {
        const double norm_a = norm(&a[s * C]);
        int64_t total_a = 0;
        for (size_t k = 0; k < C; ++k) total_a += expected[s * C + k];
        for (size_t c = 0; c < S; ++c) {
            double similarity =
                blas_dot(&a[s * C], &b[c * C], C) / std::max(1e-8, norm_a * norm_b[c]);
            similarity -=
                0.1 * std::abs(std::log(
                          static_cast<double>(total_a + 1) / static_cast<double>(total_b[c] + 1)));
            cost[s * S + c] = -similarity;
        }
    }
    for (const auto& [row, column] :
         linear_sum_assignment(cost, static_cast<int>(N), static_cast<int>(S)))
        mapping[row] = column;
    return mapping;
}

namespace {

// Start positions of every token trigram: first position and count per trigram.
std::map<std::array<int, 3>, std::pair<int, int>>
trigram_positions(const std::vector<int>& tokens) {
    std::map<std::array<int, 3>, std::pair<int, int>> table;
    for (size_t i = 0; i + 2 < tokens.size(); ++i) {
        auto [it, inserted] =
            table.try_emplace({tokens[i], tokens[i + 1], tokens[i + 2]}, static_cast<int>(i), 0);
        ++it->second.second;
    }
    return table;
}

// NeMo's _ctc_anchor_band_starts.
std::vector<int64_t>
anchor_band_starts(
    const std::vector<int>& greedy, const std::vector<int>& target, int64_t states, int64_t half,
    int64_t width) {
    const int64_t frames = static_cast<int64_t>(greedy.size());
    std::vector<int> observed, times;
    for (int64_t i = 0; i < frames; ++i)
        if (greedy[i] != 0 && (i == 0 || greedy[i] != greedy[i - 1])) {
            observed.push_back(greedy[i]);
            times.push_back(static_cast<int>(i));
        }
    const auto left = trigram_positions(observed), right = trigram_positions(target);
    std::vector<std::pair<int64_t, int64_t>> pairs;
    for (const auto& [key, position] : left) {
        if (position.second != 1)
            continue;
        const auto it = right.find(key);
        if (it != right.end() && it->second.second == 1)
            pairs.emplace_back(times[position.first + 1], 2 * (it->second.first + 1) + 1);
    }
    std::sort(pairs.begin(), pairs.end());
    // Longest chain increasing in state (patience sort with parents).
    std::vector<int64_t> tails;
    std::vector<int> tail_ids, parent;
    for (size_t i = 0; i < pairs.size(); ++i) {
        const size_t j =
            std::lower_bound(tails.begin(), tails.end(), pairs[i].second) - tails.begin();
        parent.push_back(j ? tail_ids[j - 1] : -1);
        if (j == tails.size()) {
            tails.push_back(pairs[i].second);
            tail_ids.push_back(static_cast<int>(i));
        } else {
            tails[j] = pairs[i].second;
            tail_ids[j] = static_cast<int>(i);
        }
    }
    std::vector<std::pair<int64_t, int64_t>> chosen;
    for (int k = tail_ids.empty() ? -1 : tail_ids.back(); k >= 0; k = parent[k])
        chosen.push_back(pairs[k]);
    std::reverse(chosen.begin(), chosen.end());
    std::vector<double> xs{0.0}, ys{0.0};
    for (const auto& [t, s] : chosen)
        if (0 < t && t < frames - 1) {
            xs.push_back(static_cast<double>(t));
            ys.push_back(static_cast<double>(s));
        }
    xs.push_back(static_cast<double>(frames - 1));
    ys.push_back(static_cast<double>(states - 1));
    // np.interp at 0..frames-1, then rint, clip and the reachability limits.
    std::vector<int64_t> starts(frames);
    const size_t n = xs.size();
    for (int64_t t = 0; t < frames; ++t) {
        const double x = static_cast<double>(t);
        const size_t j = std::upper_bound(xs.begin(), xs.end(), x) - xs.begin() - 1;
        double center;
        if (j == n - 1 || xs[j] == x)
            center = ys[j];
        else
            center = (ys[j + 1] - ys[j]) / (xs[j + 1] - xs[j]) * (x - xs[j]) + ys[j];
        const int64_t start = static_cast<int64_t>(std::nearbyint(center)) - half;
        starts[t] = std::min(std::max<int64_t>(start, 0), states - width);
    }
    int64_t running = std::numeric_limits<int64_t>::max();
    for (int64_t t = 0; t < frames; ++t) {
        running = std::min(running, starts[t] - 2 * t);
        starts[t] =
            std::max(running + 2 * t, std::max<int64_t>(0, states - width - 2 * (frames - 1 - t)));
    }
    return starts;
}

}  // namespace

BandedPath
banded_ctc_search(
    const float* log_probs, int columns, const std::vector<int>& labels,
    const std::vector<int>& positions, const float* prior, const std::vector<int>& greedy) {
    if (columns < 1 || (!positions.empty() && !log_probs) || labels.empty() ||
        greedy.size() > size_t(INT_MAX) || labels.size() > size_t(INT_MAX))
        throw std::invalid_argument("banded_ctc_search: invalid input shape");
    for (int label : labels)
        if (label < 0 || label >= columns)
            throw std::invalid_argument("banded_ctc_search: class ID outside columns");
    for (int label : greedy)
        if (label < 0 || label >= columns)
            throw std::invalid_argument("banded_ctc_search: invalid greedy class ID");
    int previous_position = -1;
    for (int position : positions) {
        if (position <= previous_position || size_t(position) >= greedy.size())
            throw std::invalid_argument("banded_ctc_search: invalid searched position");
        previous_position = position;
        require_log_values(log_probs + size_t(position) * columns, size_t(columns));
        if (prior) require_log_values(prior + position, 1);
    }
    const float neg_inf = -std::numeric_limits<float>::infinity();
    const int64_t frames = static_cast<int64_t>(positions.size());
    const int64_t states = static_cast<int64_t>(labels.size());
    BandedPath out{std::vector<int>(frames, -1), neg_inf};
    const int64_t half =
        frames < 1000
            ? states
            : std::min<int64_t>(states, static_cast<int64_t>(std::ceil(0.10 * states + 100)));
    const int64_t width = std::min<int64_t>(states, 2 * half + 1);
    std::vector<int> target;
    for (int64_t s = 1; s < states; s += 2) target.push_back(labels[s]);
    int64_t needed = static_cast<int64_t>(target.size());
    for (size_t k = 1; k < target.size(); ++k) needed += target[k] == target[k - 1];
    if (frames < needed || frames == 0)
        return out;
    std::vector<int> searched(frames);
    for (int64_t t = 0; t < frames; ++t) searched[t] = greedy[positions[t]];
    const auto starts = anchor_band_starts(searched, target, states, half, width);
    const auto row = [&](int64_t t) {
        return log_probs + static_cast<size_t>(positions[t]) * columns;
    };
    // Forward: float32 scores, 2-bit backpointers (stay 0, step 1, skip 2).
    std::vector<float> previous(width, neg_inf), padded(width + 4, neg_inf);
    const size_t row_bytes = static_cast<size_t>(width + 3) / 4;
    std::vector<uint8_t> pointers(static_cast<size_t>(frames) * row_bytes, 0);
    for (int64_t i = 0; i < std::min<int64_t>(2, width); ++i) {
        previous[i] = row(0)[labels[i]];
        if (prior)
            previous[i] += labels[i] != 0 ? prior[positions[0]] : 0.0f;
    }
    std::vector<uint8_t> can_skip(states, 0);
    for (int64_t s = 2; s < states; ++s) can_skip[s] = labels[s] != 0 && labels[s] != labels[s - 2];
    for (int64_t t = 1; t < frames; ++t) {
        const int64_t start = starts[t], delta = start - starts[t - 1];
        std::copy(previous.begin(), previous.end(), padded.begin() + 2);
        const float* emissions = row(t);
        const float p = prior ? prior[positions[t]] : 0.0f;
        uint8_t* back = pointers.data() + static_cast<size_t>(t) * row_bytes;
        for (int64_t i = 0; i < width; ++i) {
            float best = padded[2 + delta + i];
            uint8_t move = 0;
            if (padded[1 + delta + i] > best)
                best = padded[1 + delta + i], move = 1;
            if (padded[delta + i] > best && can_skip[start + i])
                best = padded[delta + i], move = 2;
            const int label = labels[start + i];
            float score = best + emissions[label];
            if (prior)
                score += label != 0 ? p : 0.0f;
            previous[i] = score;
            back[i >> 2] |= static_cast<uint8_t>(move << (2 * (i & 3)));
        }
    }
    // Backtrace from the better terminal state (the last blank on a tie).
    int64_t end_state = -1;
    for (const int64_t candidate : {states - 1, states - 2}) {
        const int64_t local = candidate - starts[frames - 1];
        if (local < 0 || local >= width)
            continue;
        if (end_state < 0 || previous[local] > out.score) {
            end_state = candidate;
            out.score = previous[local];
        }
    }
    if (end_state < 0 || !std::isfinite(out.score)) {
        out.score = neg_inf;
        return out;
    }
    for (int64_t t = frames - 1, state = end_state; t >= 0; --t) {
        out.path[t] = static_cast<int>(state);
        if (t) {
            const int64_t i = state - starts[t];
            state -= (pointers[static_cast<size_t>(t) * row_bytes + (i >> 2)] >> (2 * (i & 3))) & 3;
        }
    }
    return out;
}

std::vector<Segment>
diarization_segments(
    const float* activity10, int frames, int speakers, double frame_seconds, double duration,
    double time_offset) {
    std::vector<Segment> segments;
    if (frames < 0 || speakers < 0)
        throw std::invalid_argument("diarization_segments: negative shape");
    if (!activity10 || frames <= 0 || speakers <= 0)
        return segments;
    if (!std::isfinite(frame_seconds) || frame_seconds <= 0 || !std::isfinite(duration) ||
        duration <= 0 || !std::isfinite(time_offset) || !std::isfinite(time_offset + duration))
        throw std::invalid_argument("diarization_segments: invalid clock");
    require_finite(activity10, checked_elements(frames, speakers));
    for (int c = 0; c < speakers; ++c) {
        int begin = -1;
        for (int64_t t = 0; t <= frames; ++t) {
            const bool active =
                t < frames && activity10[static_cast<size_t>(t) * speakers + c] > 0.5f;
            if (active && begin < 0)
                begin = static_cast<int>(t);
            if (!active && begin >= 0) {
                const double start = time_offset + begin * frame_seconds;
                const double end =
                    std::min(time_offset + t * frame_seconds, time_offset + duration);
                if (end > start)
                    segments.push_back({c, start, end});
                begin = -1;
            }
        }
    }
    std::sort(segments.begin(), segments.end(), [](const Segment& a, const Segment& b) {
        return std::tie(a.start, a.end, a.speaker) < std::tie(b.start, b.end, b.speaker);
    });
    return segments;
}

RefinementStats
refine_word_timestamps(
    std::vector<WordRow>& words, const std::vector<std::pair<int, int>>& mapping,
    const std::vector<Segment>& diarization, double ctc_step, int max_frames) {
    if (max_frames < 1)
        throw std::invalid_argument("max_frames must be a positive integer");
    if (!std::isfinite(ctc_step) || ctc_step <= 0)
        throw std::invalid_argument("ctc_frame_seconds must be positive and finite");
    RefinementStats stats;
    const double limit = max_frames * std::min(ctc_step, 0.08);
    stats.max_seconds = limit;
    std::map<int, std::vector<std::pair<double, double>>> segments;
    for (const auto& segment : diarization)
        if (segment.end > segment.start)
            segments[segment.speaker].emplace_back(segment.start, segment.end);
    for (auto& [speaker, intervals] : segments) std::sort(intervals.begin(), intervals.end());
    const std::vector<std::pair<double, double>> none;
    for (size_t first = 0; first < words.size();) {
        size_t last = first;
        while (last < words.size() && words[last].tag == words[first].tag) ++last;
        auto begin = words.begin() + first, end = words.begin() + last;
        std::stable_sort(
            begin, end, [](const WordRow& a, const WordRow& b) { return a.start < b.start; });
        int column = -1;
        for (const auto& [tag, mapped] : mapping)
            if (tag == words[first].tag)
                column = mapped;
        const auto found = segments.find(column);
        const auto& intervals = column >= 0 && found != segments.end() ? found->second : none;
        const size_t n = last - first;
        std::vector<int> member(n, -1);
        for (size_t k = 0; k < n; ++k) {
            const double start = words[first + k].start;
            const int index = static_cast<int>(
                                  std::upper_bound(
                                      intervals.begin(), intervals.end(), start,
                                      [](double v, const auto& iv) { return v < iv.first; }) -
                                  intervals.begin()) -
                              1;
            member[k] = index >= 0 && start < intervals[index].second ? index : -1;
        }
        std::vector<char> seen(intervals.size(), 0);
        for (size_t k = 0; k < n; ++k) {
            auto& word = words[first + k];
            if (member[k] < 0) {
                ++stats.unmatched_words;
            } else if (!seen[member[k]]) {
                const double onset = intervals[member[k]].first;
                if (0 < word.start - onset && word.start - onset <= limit) {
                    word.start = onset;
                    ++stats.onsets_moved;
                }
                seen[member[k]] = 1;
            }
        }
        for (size_t k = 0; k < n; ++k) {
            auto& word = words[first + k];
            double stop;
            if (member[k] < 0)
                stop = k + 1 < n ? words[first + k + 1].start : word.end;
            else if (k + 1 < n && member[k + 1] == member[k])
                stop = words[first + k + 1].start;
            else
                stop = intervals[member[k]].second;
            word.end = std::max(word.start, std::min(stop, word.start + limit));
        }
        stats.words += static_cast<int>(n);
        first = last;
    }
    return stats;
}

AlignmentResult
align_record(
    const AlignmentTargets& targets, const AlignmentInputs& inputs, const AlignmentConfig& config) {
    if (targets.columns.size() > size_t(INT_MAX) || targets.words.size() > size_t(INT_MAX))
        throw std::invalid_argument("align_record: target is too large");
    const int frames = inputs.frames, C = static_cast<int>(targets.columns.size());
    if (frames < 1 || C < 1 || !inputs.log_probs || !std::isfinite(inputs.duration) ||
        inputs.duration <= 0 || !std::isfinite(inputs.time_offset) ||
        !std::isfinite(inputs.time_offset + inputs.duration) || inputs.speaker_frames < 0 ||
        inputs.speakers < 0 || inputs.label_frames < 0 || inputs.label_speakers < 0 ||
        (inputs.speaker_probs && (inputs.speaker_frames < 1 || inputs.speakers < 1)) ||
        (inputs.activity10 && (inputs.label_frames < 1 || inputs.label_speakers < 1)) ||
        !std::isfinite(inputs.label_seconds) || inputs.label_seconds <= 0 ||
        !std::isfinite(config.speaker_activity_threshold) || config.speaker_activity_threshold < 0 ||
        config.speaker_activity_threshold > 1 || !std::isfinite(config.speaker_logprob_weight) ||
        !std::isfinite(config.maximum_token_len) || config.maximum_token_len <= 0 ||
        config.max_word_frames < 1 || !std::isfinite(config.epsilon) || config.epsilon <= 0)
        throw std::invalid_argument("align_record: invalid CTC inputs");
    require_log_values(inputs.log_probs, checked_elements(frames, C));
    if (inputs.speaker_probs)
        require_finite(inputs.speaker_probs, checked_elements(inputs.speaker_frames, inputs.speakers));
    if (inputs.activity10)
        require_finite(inputs.activity10, checked_elements(inputs.label_frames, inputs.label_speakers));
    std::vector<char> assigned(targets.words.size(), 0);
    std::vector<int> tags;
    for (const auto& stream : targets.streams) {
        if (std::find(tags.begin(), tags.end(), stream.tag) != tags.end() || stream.words.empty() ||
            stream.labels.empty() || stream.labels.size() % 2 != 1 ||
            stream.state_to_word.size() != stream.labels.size())
            throw std::invalid_argument("align_record: invalid speaker stream");
        tags.push_back(stream.tag);
        for (int word : stream.words) {
            if (word < 0 || size_t(word) >= targets.words.size() || assigned[word] ||
                targets.words[word].tag != stream.tag)
                throw std::invalid_argument("align_record: invalid stream word");
            assigned[word] = 1;
        }
        for (size_t state = 0; state < stream.labels.size(); ++state) {
            const int label = stream.labels[state], word = stream.state_to_word[state];
            if (label < 0 || label >= C || (state % 2 == 0 ? label != 0 || word != -1
                    : label == 0 || word < 0 || size_t(word) >= stream.words.size()))
                throw std::invalid_argument("align_record: invalid target state");
        }
    }
    if (std::find(assigned.begin(), assigned.end(), 0) != assigned.end())
        throw std::invalid_argument("align_record: word has no speaker stream");
    AlignmentResult result;
    result.ctc_step = inputs.duration / frames;
    // Greedy column per frame (first maximum, as numpy.argmax).
    std::vector<int> greedy(frames, 0);
    for (int t = 0; t < frames; ++t) {
        const float* row = inputs.log_probs + static_cast<size_t>(t) * C;
        int best = 0;
        for (int c = 0; c < C && !std::isnan(row[best]); ++c)
            if (std::isnan(row[c]) || row[c] > row[best])
                best = c;
        greedy[t] = best;
    }
    std::vector<float> speaker;
    std::vector<int> mapping(targets.streams.size(), -1);
    const int S = inputs.speaker_probs ? inputs.speakers : 0;
    if (inputs.speaker_probs) {
        speaker = resample_speaker_probs(inputs.speaker_probs, inputs.speaker_frames, S, frames);
        mapping = speaker_mapping(
            greedy, C, targets.streams, speaker.data(), frames, S,
            config.speaker_activity_threshold);
    }
    std::vector<int> positions;
    std::vector<float> prior(frames);
    for (size_t s = 0; s < targets.streams.size(); ++s) {
        const auto& stream = targets.streams[s];
        const int column = mapping[s];
        result.mapping.emplace_back(stream.tag, column);
        positions.clear();
        for (int t = 0; t < frames; ++t) {
            const float activity = !inputs.speaker_probs ? 1.0f
                                   : column < 0          ? 0.0f
                                                : speaker[static_cast<size_t>(t) * S + column];
            if (activity >= 0.01f)
                positions.push_back(t);
            prior[t] =
                config.speaker_logprob_weight * numpy_log(std::max(activity, config.epsilon));
        }
        const auto searched = banded_ctc_search(
            inputs.log_probs, C, stream.labels, positions,
            inputs.speaker_probs ? prior.data() : nullptr, greedy);
        result.scores.push_back(searched.score);
        if (!std::isfinite(searched.score)) {
            result.failed_streams.push_back(
                {stream.tag, "no_legal_path_in_anchor_band", stream.words});
            continue;
        }
        std::vector<int> path(frames, -1);
        for (size_t k = 0; k < positions.size(); ++k) path[positions[k]] = searched.path[k];
        // NeMo's _word_rows_from_path.
        std::vector<std::vector<int>> word_frames(stream.words.size());
        for (int t = 0; t < frames; ++t)
            if (path[t] >= 0 && stream.state_to_word[path[t]] >= 0)
                word_frames[stream.state_to_word[path[t]]].push_back(t);
        for (size_t w = 0; w < stream.words.size(); ++w) {
            const auto& f = word_frames[w];
            if (f.empty())
                throw std::runtime_error(
                    "CTC path did not visit word '" + targets.words[stream.words[w]].word + "'.");
            WordRow row;
            row.word = stream.words[w];
            row.tag = stream.tag;
            row.start = inputs.time_offset + f.front() * result.ctc_step;
            row.end = std::min(
                inputs.time_offset + (f.back() + 1) * result.ctc_step,
                row.start + config.maximum_token_len);
            row.start_frame = f.front();
            row.end_frame = std::min(
                f.back(),
                std::max(
                    f.front(),
                    static_cast<int>(std::ceil((row.end - inputs.time_offset) / result.ctc_step)) -
                        1));
            row.column = column;
            if (inputs.speaker_probs && column >= 0) {
                double sum = 0.0;
                for (int t : f) sum += speaker[static_cast<size_t>(t) * S + column];
                row.speaker_confidence = static_cast<float>(sum / static_cast<double>(f.size()));
            }
            result.words.push_back(row);
        }
    }
    result.diarization = diarization_segments(
        inputs.activity10, inputs.label_frames, inputs.label_speakers, inputs.label_seconds,
        inputs.duration, inputs.time_offset);
    result.refinement = refine_word_timestamps(
        result.words, result.mapping, result.diarization, result.ctc_step, config.max_word_frames);
    return result;
}

std::vector<WordRow>
public_words(const std::vector<WordRow>& words) {
    std::vector<WordRow> out(words);
    for (auto& word : out) {
        word.start = round_decimals(word.start, 3);
        word.end = round_decimals(word.end, 3);
    }
    std::stable_sort(out.begin(), out.end(), [](const WordRow& a, const WordRow& b) {
        return std::tie(a.start, a.end) < std::tie(b.start, b.end);
    });
    return out;
}

double
round_decimals(double x, int decimals) {
    if (!std::isfinite(x))
        return x;
    char buffer[512];
    std::snprintf(buffer, sizeof(buffer), "%.*f", decimals, x);
    return std::strtod(buffer, nullptr);
}

}  // namespace engine::models::nemotron_transcribe::timestamps
