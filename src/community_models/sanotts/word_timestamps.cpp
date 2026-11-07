#include "word_timestamps.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <utility>

namespace engine::models::sanotts::word_timestamps {
namespace {

// Marks that end a spoken word. Apostrophes and hyphens are not in this list:
// they stay inside a word ("don't", "well-known").
const std::string kWordEndMarks[] = {
    "!", "\"", "(", ")", ",", ".", ":", ";", "?", "—", "“", "”"};

bool is_word_end_mark(const std::string & symbol) {
    return std::find(std::begin(kWordEndMarks), std::end(kWordEndMarks), symbol) !=
        std::end(kWordEndMarks);
}

// Splits UTF-8 text into code points, so edit distances count phonemes, not bytes.
std::vector<std::string> code_points(const std::string & text) {
    std::vector<std::string> out;
    for (size_t i = 0; i < text.size();) {
        const auto lead = static_cast<unsigned char>(text[i]);
        size_t len = 1;
        if (lead >= 0xF0) {
            len = 4;
        } else if (lead >= 0xE0) {
            len = 3;
        } else if (lead >= 0xC0) {
            len = 2;
        }
        out.push_back(text.substr(i, len));
        i += len;
    }
    return out;
}

struct WordChar {
    std::string ch;
    size_t word;
};

struct StreamChar {
    std::string ch;
    size_t group;
    size_t index;   // position within the group's code points
    size_t count;   // code points in the group
};

// Frame range of one code point of a group, spread evenly over the group.
FrameRange char_frames(const PhonemeGroup & group, size_t index, size_t count) {
    const int64_t length = group.frames.end - group.frames.start;
    const auto n = static_cast<int64_t>(count);
    const auto i = static_cast<int64_t>(index);
    return {group.frames.start + length * i / n, group.frames.start + length * (i + 1) / n};
}

// Global alignment of the words' phonemes against the stream's phonemes.
// Each stream code point that matches (or substitutes for) a word code point
// belongs to that word, so a group shared by two words is split in time, and
// a word split across groups spans them. Returns, per word, the first and
// last stream index it covers (-1 when none), or empty if the match is poor.
std::vector<std::pair<int64_t, int64_t>> align_words_to_stream(
    const std::vector<PhonemeGroup> & groups,
    const std::vector<std::string> & word_phonemes,
    std::vector<StreamChar> & stream_out) {
    std::vector<WordChar> wchars;
    for (size_t w = 0; w < word_phonemes.size(); ++w) {
        for (auto & ch : code_points(word_phonemes[w])) {
            wchars.push_back({std::move(ch), w});
        }
    }
    for (size_t g = 0; g < groups.size(); ++g) {
        const auto chars = code_points(groups[g].phonemes);
        for (size_t c = 0; c < chars.size(); ++c) {
            stream_out.push_back({chars[c], g, c, chars.size()});
        }
    }
    const size_t n = wchars.size();
    const size_t m = stream_out.size();
    // Unit-cost edit distance; dp[i][j] aligns the first i word chars with the first j stream chars.
    std::vector<std::vector<size_t>> dp(n + 1, std::vector<size_t>(m + 1, 0));
    for (size_t i = 0; i <= n; ++i) {
        dp[i][0] = i;
    }
    for (size_t j = 0; j <= m; ++j) {
        dp[0][j] = j;
    }
    for (size_t i = 1; i <= n; ++i) {
        for (size_t j = 1; j <= m; ++j) {
            const size_t sub = dp[i - 1][j - 1] +
                (wchars[i - 1].ch == stream_out[j - 1].ch ? 0 : 1);
            dp[i][j] = std::min({dp[i - 1][j] + 1, dp[i][j - 1] + 1, sub});
        }
    }
    // Accept only if at most half the word phonemes disagree; otherwise the
    // caller falls back to one span for the chunk.
    if (n == 0 || dp[n][m] * 2 > n) {
        return {};
    }
    std::vector<int64_t> stream_word(m, -1);
    size_t i = n;
    size_t j = m;
    while (i > 0 && j > 0) {
        if (dp[i][j] == dp[i - 1][j - 1] + (wchars[i - 1].ch == stream_out[j - 1].ch ? 0 : 1)) {
            stream_word[j - 1] = static_cast<int64_t>(wchars[i - 1].word);
            --i;
            --j;
        } else if (dp[i][j] == dp[i - 1][j] + 1) {
            --i;
        } else {
            --j;
        }
    }
    std::vector<std::pair<int64_t, int64_t>> extent(word_phonemes.size(), {-1, -1});
    for (size_t k = 0; k < m; ++k) {
        if (stream_word[k] < 0) {
            continue;
        }
        auto & range = extent[static_cast<size_t>(stream_word[k])];
        if (range.first < 0) {
            range.first = static_cast<int64_t>(k);
        }
        range.second = static_cast<int64_t>(k);
    }
    return extent;
}

}  // namespace

bool is_phoneme_symbol(const std::string & symbol) {
    return !symbol.empty() && symbol != " " && !is_word_end_mark(symbol) &&
        symbol != "'" && symbol != "-";
}

std::vector<std::string> text_words(const std::string & text) {
    std::vector<std::string> words;
    std::string current;
    const auto flush = [&] {
        if (!current.empty()) {
            words.push_back(current);
            current.clear();
        }
    };
    size_t i = 0;
    while (i < text.size()) {
        if (std::isspace(static_cast<unsigned char>(text[i])) != 0) {
            flush();
            ++i;
            continue;
        }
        const std::string * mark = nullptr;
        for (const auto & candidate : kWordEndMarks) {
            if (text.compare(i, candidate.size(), candidate) == 0) {
                mark = &candidate;
                break;
            }
        }
        if (mark != nullptr) {
            flush();
            i += mark->size();
            continue;
        }
        current += text[i];
        ++i;
    }
    flush();
    return words;
}

std::vector<PhonemeGroup> phoneme_word_ranges(
    const std::vector<int32_t> & token_ids,
    const std::vector<int64_t> & token_frames,
    const SymbolLookup & symbol_for) {
    std::vector<PhonemeGroup> groups;
    PhonemeGroup current;
    bool open = false;
    int64_t frame = 0;
    const size_t count = std::min(token_ids.size(), token_frames.size());
    for (size_t index = 0; index < count; ++index) {
        const int64_t start = frame;
        frame += token_frames[index];
        const std::string symbol = symbol_for(token_ids[index]);
        if (!is_phoneme_symbol(symbol)) {
            // Spaces, word-end marks and specials end the group. Apostrophes
            // and hyphens are skipped and stay inside it.
            const bool breaks = symbol.empty() || symbol == " " || is_word_end_mark(symbol);
            if (breaks && open) {
                groups.push_back(current);
                current = PhonemeGroup{};
                open = false;
            }
            continue;
        }
        if (!open) {
            current.frames.start = start;
            open = true;
        }
        current.frames.end = frame;
        current.phonemes += symbol;
    }
    if (open) {
        groups.push_back(current);
    }
    return groups;
}

std::vector<LabeledRange> label_ranges(
    const std::vector<PhonemeGroup> & groups,
    const std::vector<std::string> & words,
    const WordPhonemizer & phonemize_word) {
    std::vector<LabeledRange> out;
    if (groups.empty()) {
        return out;
    }
    if (groups.size() == words.size()) {
        out.reserve(groups.size());
        for (size_t i = 0; i < groups.size(); ++i) {
            out.push_back({groups[i].frames, words[i]});
        }
        return out;
    }
    std::vector<std::string> word_phonemes;
    word_phonemes.reserve(words.size());
    for (const auto & word : words) {
        word_phonemes.push_back(phonemize_word ? phonemize_word(word) : std::string{});
    }
    std::vector<StreamChar> stream;
    const auto extent = align_words_to_stream(groups, word_phonemes, stream);
    if (!extent.empty()) {
        for (size_t w = 0; w < words.size(); ++w) {
            if (extent[w].first < 0) {
                continue;   // no audible phonemes matched this word
            }
            const auto & first = stream[static_cast<size_t>(extent[w].first)];
            const auto & last = stream[static_cast<size_t>(extent[w].second)];
            const FrameRange start_frames =
                char_frames(groups[first.group], first.index, first.count);
            const FrameRange end_frames =
                char_frames(groups[last.group], last.index, last.count);
            out.push_back({{start_frames.start, end_frames.end}, words[w]});
        }
        return out;
    }
    std::string joined;
    for (const auto & word : words) {
        if (!joined.empty()) {
            joined += ' ';
        }
        joined += word;
    }
    out.push_back({{groups.front().frames.start, groups.back().frames.end}, joined});
    return out;
}

}  // namespace engine::models::sanotts::word_timestamps
