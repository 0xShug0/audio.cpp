// SPDX-License-Identifier: MIT
//
// Focused port of cppjieba's MixSegment for ZipVoice. Derived from:
//   cppjieba, Copyright (c) 2013 Yanyi Wu (MIT)
//   jieba,    Copyright (c) 2012 Sun Junyi  (MIT)
//
// The MIT License (MIT)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// Only the pieces the ZipVoice Chinese frontend needs are kept: the PreFilter
// separator pass, the maximum-probability DAG segmentation over the jieba
// dictionary, and the four-state BMES HMM used for runs of dictionary-missed
// single characters. The dictionary files are model resources embedded in the
// GGUF, not a vendored library.

#include "jieba_segmenter.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace engine::models::zipvoice {
namespace {

// cppjieba Utils.hpp MIN_DOUBLE, used as the "impossible" log-probability.
constexpr double kMinDouble = -3.14e+100;

// cppjieba HMMModel::STATUS_SUM order: 0=B, 1=E, 2=M, 3=S.
constexpr size_t kStateE = 1;
constexpr size_t kStateS = 3;

// cppjieba Utils.hpp Trim.
void trim(std::string & s) {
    auto not_space = [](unsigned char c) { return !std::isspace(c); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
}

size_t count_codepoints(const std::string & s) {
    size_t n = 0;
    for (unsigned char c : s) {
        if ((c & 0xC0) != 0x80) ++n;
    }
    return n;
}

}  // namespace

JiebaSegmenter::JiebaSegmenter(const std::filesystem::path & dict_path,
                               const std::filesystem::path & hmm_path) {
    // --- dictionary: <word> <freq> <tag>, weight = log(freq / freq_sum) ---
    std::ifstream dict(dict_path);
    if (!dict) {
        throw std::runtime_error("zipvoice: cannot open jieba dictionary " + dict_path.string());
    }
    std::vector<std::pair<std::string, double>> entries;
    entries.reserve(400000);
    double freq_sum = 0.0;
    size_t max_len = 1;
    std::string line;
    while (std::getline(dict, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const size_t first = line.find(' ');
        if (first == std::string::npos) continue;
        const size_t second = line.find(' ', first + 1);
        if (second == std::string::npos) continue;
        std::string word = line.substr(0, first);
        const double freq = std::strtod(line.c_str() + first + 1, nullptr);
        freq_sum += freq;
        max_len = std::max(max_len, count_codepoints(word));
        entries.emplace_back(std::move(word), freq);
    }
    if (entries.empty() || !(freq_sum > 0.0)) {
        throw std::runtime_error("zipvoice: empty jieba dictionary " + dict_path.string());
    }
    weights_.reserve(entries.size() * 2);
    double min_weight = std::numeric_limits<double>::infinity();
    for (const auto & [word, freq] : entries) {
        const double weight = std::log(freq / freq_sum);
        weights_[word] = weight;
        min_weight = std::min(min_weight, weight);
    }
    min_weight_ = min_weight;
    max_word_len_ = max_len;

    // --- HMM: start/trans/emit tables (cppjieba hmm_model.utf8) ---
    std::ifstream hmm(hmm_path);
    if (!hmm) {
        throw std::runtime_error("zipvoice: cannot open jieba HMM model " + hmm_path.string());
    }
    const auto next_content_line = [&](std::string & out) {
        while (std::getline(hmm, out)) {
            trim(out);
            if (!out.empty() && out[0] != '#') return true;
        }
        return false;
    };
    const auto parse_doubles = [](const std::string & text, double * out, size_t count) {
        std::istringstream stream(text);
        for (size_t i = 0; i < count; ++i) {
            if (!(stream >> out[i])) {
                throw std::runtime_error("zipvoice: malformed jieba HMM model");
            }
        }
    };
    if (!next_content_line(line)) throw std::runtime_error("zipvoice: truncated jieba HMM model");
    parse_doubles(line, start_prob_, 4);
    for (size_t i = 0; i < 4; ++i) {
        if (!next_content_line(line)) throw std::runtime_error("zipvoice: truncated jieba HMM model");
        parse_doubles(line, trans_prob_[i], 4);
    }
    for (size_t state = 0; state < 4; ++state) {
        if (!next_content_line(line)) throw std::runtime_error("zipvoice: truncated jieba HMM model");
        size_t pos = 0;
        while (pos < line.size()) {
            const size_t comma = line.find(',', pos);
            const std::string entry = line.substr(
                pos, comma == std::string::npos ? std::string::npos : comma - pos);
            const size_t colon = entry.find(':');
            if (colon != std::string::npos && colon > 0) {
                const auto runes = decode(entry.substr(0, colon));
                if (!runes.empty()) {
                    emit_prob_[state][runes.front().cp] =
                        std::strtod(entry.c_str() + colon + 1, nullptr);
                }
            }
            if (comma == std::string::npos) break;
            pos = comma + 1;
        }
    }
}

std::vector<JiebaSegmenter::Rune> JiebaSegmenter::decode(const std::string & text) {
    std::vector<Rune> runes;
    runes.reserve(text.size() / 2 + 1);
    for (size_t i = 0; i < text.size();) {
        const auto lead = static_cast<unsigned char>(text[i]);
        uint32_t cp = lead;
        uint32_t len = 1;
        if (lead < 0x80) {
            cp = lead;
        } else if ((lead & 0xE0) == 0xC0 && i + 1 < text.size()) {
            cp = (uint32_t(lead & 0x1F) << 6) |
                (uint32_t(static_cast<unsigned char>(text[i + 1])) & 0x3F);
            len = 2;
        } else if ((lead & 0xF0) == 0xE0 && i + 2 < text.size()) {
            cp = (uint32_t(lead & 0x0F) << 12) |
                (uint32_t(static_cast<unsigned char>(text[i + 1])) & 0x3F) << 6 |
                (uint32_t(static_cast<unsigned char>(text[i + 2])) & 0x3F);
            len = 3;
        } else if ((lead & 0xF8) == 0xF0 && i + 3 < text.size()) {
            cp = (uint32_t(lead & 0x07) << 18) |
                (uint32_t(static_cast<unsigned char>(text[i + 1])) & 0x3F) << 12 |
                (uint32_t(static_cast<unsigned char>(text[i + 2])) & 0x3F) << 6 |
                (uint32_t(static_cast<unsigned char>(text[i + 3])) & 0x3F);
            len = 4;
        }
        runes.push_back({cp, static_cast<uint32_t>(i), len});
        i += len;
    }
    return runes;
}

std::string JiebaSegmenter::slice(const std::string & text, const std::vector<Rune> & runes,
                                  size_t begin, size_t end_inclusive) {
    const size_t start = runes[begin].offset;
    const size_t end = end_inclusive + 1 < runes.size()
        ? runes[end_inclusive + 1].offset
        : text.size();
    return text.substr(start, end - start);
}

bool JiebaSegmenter::is_separator(uint32_t cp) {
    // cppjieba SegmentBase SPECIAL_SEPARATORS: " \t\n，。".
    return cp == 0x20 || cp == 0x09 || cp == 0x0A || cp == 0xFF0C || cp == 0x3002;
}

size_t JiebaSegmenter::sequential_letter_rule(const std::vector<Rune> & runes,
                                              size_t begin, size_t end) {
    uint32_t x = runes[begin].cp;
    if (!(('a' <= x && x <= 'z') || ('A' <= x && x <= 'Z'))) return begin;
    ++begin;
    while (begin != end) {
        x = runes[begin].cp;
        if (('a' <= x && x <= 'z') || ('A' <= x && x <= 'Z') || ('0' <= x && x <= '9')) ++begin;
        else break;
    }
    if (begin != end && runes[begin].cp == '.') {
        if (begin + 1 != end && runes[begin + 1].cp >= '0' && runes[begin + 1].cp <= '9') {
            ++begin;
            while (begin != end && runes[begin].cp >= '0' && runes[begin].cp <= '9') ++begin;
        }
    }
    return begin;
}

size_t JiebaSegmenter::numbers_rule(const std::vector<Rune> & runes, size_t begin, size_t end) {
    uint32_t x = runes[begin].cp;
    if (!('0' <= x && x <= '9')) return begin;
    ++begin;
    while (begin != end) {
        x = runes[begin].cp;
        if (('0' <= x && x <= '9') || ('a' <= x && x <= 'z') || ('A' <= x && x <= 'Z')) ++begin;
        else break;
    }
    if (begin != end && runes[begin].cp == '.') {
        if (begin + 1 != end && runes[begin + 1].cp >= '0' && runes[begin + 1].cp <= '9') {
            ++begin;
            while (begin != end && runes[begin].cp >= '0' && runes[begin].cp <= '9') ++begin;
        }
    }
    return begin;
}

double JiebaSegmenter::emit_probability(size_t state, uint32_t cp) const {
    const auto & table = emit_prob_[state];
    const auto it = table.find(cp);
    return it == table.end() ? kMinDouble : it->second;
}

void JiebaSegmenter::viterbi_cut(const std::string & text, const std::vector<Rune> & runes,
                                 size_t begin, size_t end, std::vector<std::string> & words) const {
    const size_t X = end - begin;
    if (X == 0) return;
    std::vector<double> weight(4 * X);
    std::vector<int> path(4 * X);
    for (size_t y = 0; y < 4; ++y) {
        weight[y * X] = start_prob_[y] + emit_probability(y, runes[begin].cp);
        path[y * X] = -1;
    }
    for (size_t x = 1; x < X; ++x) {
        for (size_t y = 0; y < 4; ++y) {
            const size_t now = x + y * X;
            weight[now] = kMinDouble;
            path[now] = static_cast<int>(kStateE);
            const double emit = emit_probability(y, runes[begin + x].cp);
            for (size_t prev = 0; prev < 4; ++prev) {
                const double candidate = weight[x - 1 + prev * X] + trans_prob_[prev][y] + emit;
                if (candidate > weight[now]) {
                    weight[now] = candidate;
                    path[now] = static_cast<int>(prev);
                }
            }
        }
    }
    const double end_e = weight[(X - 1) + kStateE * X];
    const double end_s = weight[(X - 1) + kStateS * X];
    int state = end_e >= end_s ? static_cast<int>(kStateE) : static_cast<int>(kStateS);
    std::vector<int> status(X);
    for (size_t x = X; x-- > 0;) {
        status[x] = state;
        state = path[x + static_cast<size_t>(state) * X];
    }
    size_t left = begin;
    for (size_t i = 0; i < X; ++i) {
        if (status[i] % 2 != 0) {  // E or S ends a word
            words.push_back(slice(text, runes, left, begin + i));
            left = begin + i + 1;
        }
    }
}

void JiebaSegmenter::hmm_cut(const std::string & text, const std::vector<Rune> & runes,
                             size_t begin, size_t end, std::vector<std::string> & words) const {
    size_t left = begin;
    size_t right = begin;
    while (right < end) {
        if (runes[right].cp < 0x80) {
            if (left != right) viterbi_cut(text, runes, left, right, words);
            left = right;
            size_t next = sequential_letter_rule(runes, left, end);
            if (next == left) next = numbers_rule(runes, left, end);
            if (next == left) next = left + 1;
            right = next;
            words.push_back(slice(text, runes, left, right - 1));
            left = right;
        } else {
            ++right;
        }
    }
    if (left != right) viterbi_cut(text, runes, left, right, words);
}

void JiebaSegmenter::segment_range(const std::string & text, const std::vector<Rune> & runes,
                                   size_t begin, size_t end, std::vector<std::string> & words) const {
    const size_t n = end - begin;
    if (n == 0) return;
    struct Candidate {
        size_t end_index;
        double weight;
    };
    // MP DAG candidates: the single character is always a candidate (unknown
    // characters fall back to the dictionary's minimum weight, like cppjieba);
    // longer candidates exist only when the whole span is a dictionary word.
    std::vector<std::vector<Candidate>> nexts(n);
    for (size_t k = 0; k < n; ++k) {
        const size_t i = begin + k;
        const std::string single = slice(text, runes, i, i);
        const auto single_it = weights_.find(single);
        nexts[k].push_back({k, single_it != weights_.end() ? single_it->second : min_weight_});
        for (size_t len = 2; len <= max_word_len_ && k + len <= n; ++len) {
            const auto it = weights_.find(slice(text, runes, i, i + len - 1));
            if (it != weights_.end()) nexts[k].push_back({k + len - 1, it->second});
        }
    }
    // Maximum-probability route, right to left (cppjieba CalcDP).
    std::vector<double> best_weight(n + 1, 0.0);
    std::vector<size_t> best_len(n, 1);
    for (size_t k = n; k-- > 0;) {
        double best = kMinDouble;
        size_t length = 1;
        for (const auto & candidate : nexts[k]) {
            const double value = candidate.weight +
                (candidate.end_index + 1 < n ? best_weight[candidate.end_index + 1] : 0.0);
            if (value > best) {
                best = value;
                length = candidate.end_index - k + 1;
            }
        }
        best_weight[k] = best;
        best_len[k] = length;
    }
    // MP words as rune ranges.
    std::vector<std::pair<size_t, size_t>> mp_words;
    for (size_t k = 0; k < n;) {
        const size_t length = best_len[k];
        mp_words.push_back({begin + k, begin + k + length - 1});
        k += length;
    }
    // MixSegment: keep multi-character words; re-cut each run of consecutive
    // single-character words with the HMM.
    for (size_t r = 0; r < mp_words.size();) {
        if (mp_words[r].first != mp_words[r].second) {
            words.push_back(slice(text, runes, mp_words[r].first, mp_words[r].second));
            ++r;
            continue;
        }
        size_t run_end = r;
        while (run_end < mp_words.size() && mp_words[run_end].first == mp_words[run_end].second) {
            ++run_end;
        }
        hmm_cut(text, runes, mp_words[r].first, mp_words[run_end - 1].first + 1, words);
        r = run_end;
    }
}

std::vector<std::string> JiebaSegmenter::cut(const std::string & text) const {
    std::vector<std::string> words;
    const auto runes = decode(text);
    if (runes.empty()) return words;
    // PreFilter: separators are single-character words; the runs between them
    // are segmented independently.
    size_t i = 0;
    while (i < runes.size()) {
        if (is_separator(runes[i].cp)) {
            words.push_back(slice(text, runes, i, i));
            ++i;
            continue;
        }
        size_t j = i;
        while (j < runes.size() && !is_separator(runes[j].cp)) ++j;
        segment_range(text, runes, i, j, words);
        i = j;
    }
    return words;
}

}  // namespace engine::models::zipvoice
