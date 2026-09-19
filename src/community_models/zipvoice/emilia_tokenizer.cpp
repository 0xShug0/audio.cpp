#include "engine/community_models/zipvoice/emilia_tokenizer.h"

#include "jieba_segmenter.h"

#include "engine/framework/audio/espeak_phonemizer.h"
#include "engine/framework/text/chinese_normalization.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <mutex>
#include <sstream>
#include <stdexcept>

namespace engine::models::zipvoice {
namespace {

// ---------------------------------------------------------------------------
// UTF-8 helpers

std::vector<std::string> split_utf8(const std::string & text) {
    std::vector<std::string> out;
    for (size_t i = 0; i < text.size();) {
        const auto lead = static_cast<unsigned char>(text[i]);
        size_t width = 1;
        if (lead >= 0xF0) width = 4;
        else if (lead >= 0xE0) width = 3;
        else if (lead >= 0xC0) width = 2;
        out.emplace_back(text, i, std::min(width, text.size() - i));
        i += out.back().size();
    }
    return out;
}

uint32_t utf8_to_codepoint(const std::string & ch) {
    const auto lead = static_cast<unsigned char>(ch[0]);
    if (lead < 0x80) return lead;
    if ((lead & 0xE0) == 0xC0) return (uint32_t(lead & 0x1F) << 6) |
        (uint32_t(static_cast<unsigned char>(ch[1])) & 0x3F);
    if ((lead & 0xF0) == 0xE0) return (uint32_t(lead & 0x0F) << 12) |
        (uint32_t(static_cast<unsigned char>(ch[1])) & 0x3F) << 6 |
        (uint32_t(static_cast<unsigned char>(ch[2])) & 0x3F);
    return (uint32_t(lead & 0x07) << 18) |
        (uint32_t(static_cast<unsigned char>(ch[1])) & 0x3F) << 12 |
        (uint32_t(static_cast<unsigned char>(ch[2])) & 0x3F) << 6 |
        (uint32_t(static_cast<unsigned char>(ch[3])) & 0x3F);
}

bool is_han_char(uint32_t cp) {
    // Reference EmiliaTokenizer::is_chinese checks U+4E00..U+9FA5 only.
    return cp >= 0x4E00 && cp <= 0x9FA5;
}

bool is_alpha_char(uint32_t cp) {
    return (cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z');
}

// ---------------------------------------------------------------------------
// Table loading

std::unordered_map<std::string, std::vector<std::string>> load_token_table(
    const std::filesystem::path & path) {
    std::unordered_map<std::string, std::vector<std::string>> out;
    std::ifstream in(path);
    if (!in) throw std::runtime_error("zipvoice: missing Chinese table " + path.string());
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        const auto tab = line.find('\t');
        if (tab == std::string::npos) continue;
        std::vector<std::string> tokens;
        std::istringstream rest(line.substr(tab + 1));
        std::string token;
        while (rest >> token) tokens.push_back(token);
        out.emplace(line.substr(0, tab), std::move(tokens));
    }
    return out;
}

// ---------------------------------------------------------------------------
// Text shaping (mirrors EmiliaTokenizer::map_punctuations)

std::string map_punctuations(std::string text) {
    static const std::pair<std::string_view, std::string_view> kMap[] = {
        {"，", ","}, {"。", "."}, {"！", "!"}, {"？", "?"}, {"；", ";"},
        {"：", ":"}, {"、", ","}, {"‘", "'"}, {"“", "\""}, {"”", "\""},
        {"’", "'"}, {"⋯", "…"}, {"···", "…"}, {"・・・", "…"}, {"...", "…"},
    };
    for (const auto & [from, to] : kMap) {
        for (size_t pos = 0; (pos = text.find(from, pos)) != std::string::npos;) {
            text.replace(pos, from.size(), to);
            pos += to.size();
        }
    }
    return text;
}

// ---------------------------------------------------------------------------
// Tone sandhi (pypinyin contrib/tone_sandhi.py semantics)
//
// Scope note: the reference only reaches this with a word the pypinyin
// phrases_dict matched (whole-word scope) or a single unmatched character;
// the caller mirrors that by applying it per phrase-match / per character.

void apply_tone_sandhi(std::vector<std::string> & tokens, const std::string & han) {
    // Syllables are the tokens whose last char is a tone digit.
    std::vector<size_t> finals;
    for (size_t i = 0; i < tokens.size(); ++i) {
        const auto & t = tokens[i];
        if (!t.empty() && t.back() >= '1' && t.back() <= '5') finals.push_back(i);
    }
    const size_t n = finals.size();
    if (n == 0) return;
    auto tone = [&](size_t s) { return tokens[finals[s]].back(); };
    auto set_tone = [&](size_t s, char d) { tokens[finals[s]].back() = d; };

    // Third tone: pypinyin counts the TRAILING consecutive run of '3'
    // syllables. A run of exactly 2 turns the first '3' in the word into
    // '2'; a longer run turns every '3' but the last into '2'.
    size_t run = 0;
    for (size_t s = n; s > 0; --s) {
        if (tone(s - 1) != '3') break;
        ++run;
    }
    if (run == 2) {
        for (size_t s = 0; s < n; ++s) {
            if (tone(s) == '3') { set_tone(s, '2'); break; }
        }
    } else if (run > 2) {
        size_t seen = 0;
        for (size_t s = 0; s < n; ++s) {
            if (tone(s) != '3') continue;
            if (++seen == run) break;
            set_tone(s, '2');
        }
    }

    // 不 / 一 (pypinyin's replace('4','2') only fires when the current tone
    // is '4'; the non-4th branch FORCES '4'; a final 一 forces '1').
    const auto han_chars = split_utf8(han);
    for (size_t s = 0; s < n && s < han_chars.size(); ++s) {
        const bool is_bu = han_chars[s] == "不";
        const bool is_yi = han_chars[s] == "一";
        if (!is_bu && !is_yi) continue;
        if (s + 1 < n) {
            if (tone(s + 1) == '4') {
                if (tone(s) == '4') set_tone(s, '2');
            } else {
                set_tone(s, '4');
            }
        } else {
            set_tone(s, is_yi ? '1' : '4');
        }
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// EmiliaTokenizer

EmiliaTokenizer::EmiliaTokenizer(
    const TablePaths & tables,
    const std::unordered_map<std::string, int32_t> & vocab,
    const EspeakConfig & espeak)
    : vocab_(vocab), tables_(tables), espeak_(espeak) {
    chars_ = load_token_table(tables_.chars);
    phrases_ = load_token_table(tables_.phrases);
    syllables_ = load_token_table(tables_.syllables);
}

EmiliaTokenizer::~EmiliaTokenizer() = default;

std::vector<int32_t> EmiliaTokenizer::encode(const std::string & raw_text) const {
    // phone string -> id
    auto emit = [&](const std::vector<std::string> & phones, std::vector<int32_t> & ids) {
        for (const auto & phone : phones) {
            if (const auto it = vocab_.find(phone); it != vocab_.end()) {
                ids.push_back(it->second);
            }
        }
    };

    // Segment into (text, lang) runs: zh (Han), en (ASCII letters), other.
    // "other" attaches to the current language; a leading "other" run takes
    // the language of the following text.
    struct Segment { std::string text; char lang; };
    std::vector<Segment> segments;
    {
        const auto chars = split_utf8(raw_text);
        char lang = 'o';
        std::string current;
        for (const auto & ch : chars) {
            const auto cp = utf8_to_codepoint(ch);
            const char type = is_han_char(cp) ? 'z' : (is_alpha_char(cp) ? 'e' : 'o');
            if (current.empty()) {
                current = ch;
                lang = type;
            } else if (lang == 'o') {
                current += ch;
                lang = type;
            } else if (type == lang || type == 'o') {
                current += ch;
            } else {
                segments.push_back({current, lang});
                current = ch;
                lang = type;
            }
        }
        if (!current.empty()) segments.push_back({current, lang});
    }

    // Split <pinyin> / [tag] parts out of every segment.
    struct Part { std::string text; char lang; bool bracket; };
    std::vector<Part> parts;
    for (auto & seg : segments) {
        std::string pending;
        for (size_t i = 0; i < seg.text.size();) {
            const char c = seg.text[i];
            if (c == '<' || c == '[') {
                const char close = c == '<' ? '>' : ']';
                const auto end = seg.text.find(close, i + 1);
                if (end != std::string::npos) {
                    if (!pending.empty()) {
                        parts.push_back({pending, seg.lang, false});
                        pending.clear();
                    }
                    parts.push_back({seg.text.substr(i, end - i + 1), seg.lang, true});
                    i = end + 1;
                    continue;
                }
            }
            pending += c;
            ++i;
        }
        if (!pending.empty()) parts.push_back({pending, seg.lang, false});
    }

    std::vector<int32_t> ids;
    for (const auto & part : parts) {
        if (part.bracket) {
            if (part.text.front() == '<' && part.text.back() == '>') {
                // Inline pinyin override, e.g. <wo3> -> "w0 o3".
                const std::string syllable = part.text.substr(1, part.text.size() - 2);
                if (!syllable.empty() && std::isalpha(static_cast<unsigned char>(syllable.front())) &&
                    syllable.back() >= '1' && syllable.back() <= '5') {
                    if (const auto it = syllables_.find(syllable); it != syllables_.end()) {
                        emit(it->second, ids);
                    }
                }
            } else {
                // [tag]: the whole bracketed string is one phone (usually OOV).
                emit({part.text}, ids);
            }
            continue;
        }
        if (part.lang == 'e') {
            // English segment: eSpeak-ng phonemization, one phone per
            // codepoint (matches the audio.cpp espeak frontend convention).
            // eSpeak treats sentence punctuation as clause markers and drops
            // it from the phoneme stream, but the reference keeps it as
            // tokens, so re-append trailing punctuation here.
            static const std::string kTrailingPunct = ".,!?;:";
            std::string text = part.text;
            std::string trailing;
            while (!text.empty() && kTrailingPunct.find(text.back()) != std::string::npos) {
                trailing.insert(trailing.begin(), text.back());
                text.pop_back();
            }
            if (!phonemizer_) {
                phonemizer_ = std::make_unique<audio::EspeakPhonemizer>(
                    std::filesystem::path{espeak_.library_path},
                    std::filesystem::path{espeak_.data_path},
                    std::vector<std::string>{espeak_.lang.empty() ? "en-us" : espeak_.lang});
            }
            if (!text.empty()) {
                bool all_space = true;
                for (const char c : text) {
                    if (c != ' ') { all_space = false; break; }
                }
                if (all_space) {
                    for (const auto & ch : split_utf8(text)) emit({ch}, ids);
                } else {
                    emit(split_utf8(phonemizer_->phonemize(text, 2)), ids);
                }
            }
            for (const auto & ch : split_utf8(trailing)) emit({ch}, ids);
            continue;
        }
        if (part.lang != 'z') continue;  // "other"-only segments are dropped

        // Chinese segment: digit normalization, then jieba word segmentation
        // (the model-local MixSegment port, identical to python jieba.cut).
        // Each word is one reading + sandhi unit, exactly like the reference
        // pipeline lazy_pinyin(jieba.cut(text), tone_sandhi=True).
        const std::string normalized = text::normalize_chinese_text(part.text);
        std::vector<std::string> words;
        {
            std::lock_guard<std::mutex> lock(segmenter_mutex_);
            if (!segmenter_) {
                if (!std::filesystem::is_regular_file(tables_.jieba_dict) ||
                    !std::filesystem::is_regular_file(tables_.hmm_model)) {
                    throw std::runtime_error(
                        "zipvoice: jieba dictionaries missing (expected zh_jieba_dict.txt "
                        "and zh_hmm_model.txt in the package or next to it); run "
                        "tools/community_models/export_zipvoice_zh_dict.py");
                }
                segmenter_ = std::make_unique<JiebaSegmenter>(
                    tables_.jieba_dict, tables_.hmm_model);
            }
            words = segmenter_->cut(normalized);
        }
        for (const auto & word : words) {
            const auto word_chars = split_utf8(word);
            bool has_han = false;
            for (const auto & ch : word_chars) {
                if (is_han_char(utf8_to_codepoint(ch))) has_han = true;
            }
            if (!has_han) {
                for (const auto & ch : word_chars) emit({ch}, ids);
                continue;
            }
            std::vector<std::string> tokens;
            if (const auto it = phrases_.find(word); it != phrases_.end()) {
                tokens = it->second;
            } else {
                // Unknown word: per-character default readings; characters
                // without a pinyin entry are skipped, like the reference.
                for (const auto & ch : word_chars) {
                    if (const auto cit = chars_.find(ch); cit != chars_.end()) {
                        tokens.insert(tokens.end(), cit->second.begin(), cit->second.end());
                    }
                }
            }
            apply_tone_sandhi(tokens, word);
            emit(tokens, ids);
        }
    }
    return ids;
}

}  // namespace engine::models::zipvoice
