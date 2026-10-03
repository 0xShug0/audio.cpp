#include "engine/models/crisperwhisper/model.h"

#include "unicode.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <tuple>

namespace engine::models::crisperwhisper {

std::vector<CrisperWhisperWord> align_transcript(const std::string & text,
    const std::vector<CrisperWhisperWord> & hypothesis, double duration) {
    std::vector<CrisperWhisperWord> words;
    std::istringstream input(text);
    for (std::string word; input >> word;) {
        words.push_back({std::move(word)});
    }
    const auto normalize = [](const std::string & word) {
        std::vector<uint32_t> out;
        for (const uint32_t codepoint : unicode_cpts_from_utf8(word)) {
            const uint32_t lower = unicode_tolower(codepoint);
            const auto flags = unicode_cpt_flags_from_cpt(lower);
            if (flags.is_letter || flags.is_number) {
                out.push_back(lower);
            }
        }
        return out;
    };
    std::vector<std::vector<uint32_t>> reference, observed;
    for (const auto & word : words) {
        reference.push_back(normalize(word.text));
    }
    for (const auto & word : hypothesis) {
        observed.push_back(normalize(word.text));
    }
    const auto distribute = [&](size_t lo, size_t hi, double start, double end) {
        size_t total = 0;
        for (size_t i = lo; i < hi; ++i) {
            total += reference[i].size() + 1;
        }
        const double span = std::max(0.0, end - start);
        for (size_t i = lo; i < hi; ++i) {
            words[i].start = start;
            start += span * static_cast<double>(reference[i].size() + 1) / total;
            words[i].end = start;
        }
    };

    // SequenceMatcher without junk: recursively select the longest contiguous
    // match, breaking ties by earliest reference and then hypothesis position.
    using Range = std::tuple<size_t, size_t, size_t, size_t>;
    using Match = std::tuple<size_t, size_t, size_t>;
    std::vector<Range> pending{{0, reference.size(), 0, observed.size()}};
    std::vector<Match> matches;
    while (!pending.empty()) {
        const auto [a0, a1, b0, b1] = pending.back();
        pending.pop_back();
        size_t best_a = a0, best_b = b0, best_n = 0;
        std::vector<size_t> previous(b1 - b0 + 1, 0), current(previous.size());
        for (size_t a = a0; a < a1; ++a) {
            std::fill(current.begin(), current.end(), 0);
            for (size_t b = b0; b < b1; ++b) {
                if (reference[a] == observed[b]) {
                    const auto n = previous[b - b0] + 1;
                    current[b - b0 + 1] = n;
                    if (n > best_n) {
                        best_a = a + 1 - n;
                        best_b = b + 1 - n;
                        best_n = n;
                    }
                }
            }
            previous.swap(current);
        }
        if (best_n == 0) {
            continue;
        }
        matches.emplace_back(best_a, best_b, best_n);
        if (a0 < best_a && b0 < best_b) {
            pending.emplace_back(a0, best_a, b0, best_b);
        }
        if (best_a + best_n < a1 && best_b + best_n < b1) {
            pending.emplace_back(best_a + best_n, a1, best_b + best_n, b1);
        }
    }
    std::sort(matches.begin(), matches.end());
    matches.emplace_back(reference.size(), observed.size(), 0);
    size_t a = 0, b = 0;
    for (const auto & [ma, mb, count] : matches) {
        if (a < ma && b < mb) {
            if (ma - a == mb - b) {
                for (size_t i = 0; i < ma - a; ++i) {
                    words[a + i].start = hypothesis[b + i].start;
                    words[a + i].end = hypothesis[b + i].end;
                }
            } else {
                distribute(a, ma, hypothesis[b].start, hypothesis[mb - 1].end);
            }
        }
        for (size_t i = 0; i < count; ++i) {
            words[ma + i].start = hypothesis[mb + i].start;
            words[ma + i].end = hypothesis[mb + i].end;
        }
        a = ma + count;
        b = mb + count;
    }
    for (size_t i = 0; i < words.size();) {
        if (words[i].start >= 0) {
            ++i;
            continue;
        }
        const size_t first = i;
        while (i < words.size() && words[i].start < 0) {
            ++i;
        }
        const double start = first == 0 ? 0.0 : words[first - 1].end;
        distribute(first, i, start, i < words.size() ? words[i].start : std::max(start, duration));
    }
    double previous_end = 0.0;
    for (auto & word : words) {
        word.start = std::max(previous_end, std::round(word.start * 1000.0) / 1000.0);
        word.end = std::max(word.start, std::round(word.end * 1000.0) / 1000.0);
        previous_end = word.end;
    }
    return words;
}

}  // namespace engine::models::crisperwhisper
