#include "engine/community_models/vieneu_v3_turbo/orchestration.h"

#include "engine/community_models/vieneu_v3_turbo/frame_cap.h"

#include <algorithm>
#include <cmath>
#include <string_view>

namespace engine::models::vieneu_v3_turbo {
namespace {

// Envelope settings for the burst counter, measured with the guard.
constexpr int64_t kHopMs = 10;
constexpr double kBurstThresholdDb = -18.0;  // relative to the loudest hop
constexpr int64_t kBurstMinGapMs = 60;       // closer than this = one burst
constexpr int64_t kBurstMinMs = 30;
constexpr double kEdgeThresholdDb = -45.0;   // "there is speech here"

bool ends_sentence(std::string_view text) {
    while (!text.empty() && (text.back() == ' ' || text.back() == '\t' || text.back() == '\n' || text.back() == '\r')) {
        text.remove_suffix(1);
    }
    if (text.empty()) {
        return false;
    }
    if (text.back() == '.' || text.back() == '!' || text.back() == '?') {
        return true;
    }
    // '…' is U+2026, three bytes in UTF-8.
    return text.size() >= 3 && text.compare(text.size() - 3, 3, "\xE2\x80\xA6") == 0;
}

std::string trim(std::string_view text) {
    const auto is_space = [](char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; };
    while (!text.empty() && is_space(text.front())) {
        text.remove_prefix(1);
    }
    while (!text.empty() && is_space(text.back())) {
        text.remove_suffix(1);
    }
    return std::string(text);
}

// Characters that are part of a word, not of `<|emotion_k|>` markup: the budget
// counts what will be spoken.
int64_t effective_length(const std::string & chunk) {
    std::string stripped = chunk;
    for (size_t at = stripped.find("<|emotion_"); at != std::string::npos; at = stripped.find("<|emotion_")) {
        const size_t close = stripped.find("|>", at);
        if (close == std::string::npos) {
            break;
        }
        stripped.erase(at, close + 2 - at);
    }
    return static_cast<int64_t>(trim(stripped).size());
}

// Pieces of `text` cut AFTER each character in `marks` that is followed by
// whitespace, so "3,5" and "8:30" stay whole.
std::vector<std::string> split_after(const std::string & text, std::string_view marks) {
    std::vector<std::string> out;
    size_t start = 0;
    for (size_t i = 0; i + 1 < text.size(); ++i) {
        const bool is_mark = marks.find(text[i]) != std::string_view::npos;
        const bool space_follows = text[i + 1] == ' ' || text[i + 1] == '\t' || text[i + 1] == '\n';
        if (!is_mark || !space_follows) {
            continue;
        }
        out.push_back(trim(std::string_view(text).substr(start, i + 1 - start)));
        size_t j = i + 1;
        while (j < text.size() && (text[j] == ' ' || text[j] == '\t' || text[j] == '\n')) {
            ++j;
        }
        start = j;
        i = j > 0 ? j - 1 : j;
    }
    if (start < text.size()) {
        out.push_back(trim(std::string_view(text).substr(start)));
    }
    out.erase(std::remove_if(out.begin(), out.end(), [](const std::string & s) { return s.empty(); }), out.end());
    return out;
}

// Last resort: cut a piece with no usable punctuation at whitespace, as close to
// `max_chars` as a word boundary allows.
std::vector<std::string> split_at_spaces(const std::string & text, int64_t max_chars) {
    std::vector<std::string> out;
    size_t start = 0;
    while (static_cast<int64_t>(text.size() - start) > max_chars) {
        size_t cut = text.rfind(' ', start + static_cast<size_t>(max_chars));
        if (cut == std::string::npos || cut <= start) {
            cut = text.find(' ', start + static_cast<size_t>(max_chars));
        }
        if (cut == std::string::npos || cut <= start) {
            break;
        }
        out.push_back(trim(std::string_view(text).substr(start, cut - start)));
        start = cut + 1;
    }
    if (start < text.size()) {
        out.push_back(trim(std::string_view(text).substr(start)));
    }
    out.erase(std::remove_if(out.begin(), out.end(), [](const std::string & s) { return s.empty(); }), out.end());
    return out;
}

std::vector<float> envelope(const std::vector<float> & samples, int64_t hop) {
    const int64_t hops = static_cast<int64_t>(samples.size()) / hop;
    std::vector<float> out(static_cast<size_t>(std::max<int64_t>(hops, 0)), 0.0F);
    for (int64_t i = 0; i < hops; ++i) {
        double sum = 0.0;
        for (int64_t k = 0; k < hop; ++k) {
            const double value = samples[static_cast<size_t>(i * hop + k)];
            sum += value * value;
        }
        out[static_cast<size_t>(i)] = static_cast<float>(std::sqrt(sum / static_cast<double>(hop)));
    }
    return out;
}

}  // namespace

double gap_pause_sec(Gap gap) {
    switch (gap) {
        case Gap::Para:
            return 0.70;
        case Gap::Sentence:
            return 0.50;
        case Gap::Minor:
        default:
            return 0.30;
    }
}

Gap classify_gap(const std::string & chunk) {
    return ends_sentence(chunk) ? Gap::Sentence : Gap::Minor;
}

std::vector<PhonemeChunk> split_phoneme_chunks(
    const std::string & phonemes,
    int64_t max_chars,
    int64_t min_chars) {
    std::vector<PhonemeChunk> out;
    if (max_chars <= 0) {
        out.push_back({trim(phonemes), Gap::Minor});
        return out;
    }

    // Paragraphs first: a newline in the source is the longest pause there is.
    std::vector<std::string> paragraphs;
    size_t line_start = 0;
    for (size_t i = 0; i <= phonemes.size(); ++i) {
        if (i == phonemes.size() || phonemes[i] == '\n') {
            const std::string line = trim(std::string_view(phonemes).substr(line_start, i - line_start));
            if (!line.empty()) {
                paragraphs.push_back(line);
            }
            line_start = i + 1;
        }
    }
    if (paragraphs.empty()) {
        return out;
    }

    std::vector<std::string> chunks;
    std::vector<Gap> gaps;  // gaps[i] is the seam between chunks[i] and chunks[i + 1]
    for (size_t p = 0; p < paragraphs.size(); ++p) {
        std::vector<std::string> pieces;
        // Sentences, then anything still over budget at minor punctuation, then
        // at whitespace. Each level only runs on what the previous one left too long.
        for (const auto & sentence : split_after(paragraphs[p], ".!?")) {
            if (static_cast<int64_t>(sentence.size()) <= max_chars) {
                pieces.push_back(sentence);
                continue;
            }
            for (const auto & part : split_after(sentence, ",;:")) {
                if (static_cast<int64_t>(part.size()) <= max_chars) {
                    pieces.push_back(part);
                    continue;
                }
                for (const auto & piece : split_at_spaces(part, max_chars)) {
                    pieces.push_back(piece);
                }
            }
        }
        // Pack greedily: a whole sentence per chunk when it fits, several when
        // they are short, because a chunk boundary always costs a seam.
        std::string buffer;
        for (const auto & piece : pieces) {
            if (buffer.empty()) {
                buffer = piece;
            } else if (static_cast<int64_t>(buffer.size() + 1 + piece.size()) <= max_chars) {
                buffer += ' ';
                buffer += piece;
            } else {
                if (!chunks.empty()) {
                    gaps.push_back(classify_gap(chunks.back()));
                }
                chunks.push_back(buffer);
                buffer = piece;
            }
        }
        if (!buffer.empty()) {
            if (!chunks.empty()) {
                gaps.push_back(p > 0 && chunks.size() == gaps.size() ? Gap::Para : classify_gap(chunks.back()));
            }
            chunks.push_back(buffer);
        }
        // The seam that crosses a paragraph boundary is a paragraph pause.
        if (p + 1 < paragraphs.size() && !chunks.empty()) {
            gaps.push_back(Gap::Para);
        }
    }
    gaps.resize(chunks.empty() ? 0 : chunks.size() - 1, Gap::Minor);

    // A chunk too short to stand on its own joins a neighbour, preferring a seam
    // that is not a paragraph break and then the shorter neighbour.
    while (chunks.size() > 1) {
        size_t shortest = chunks.size();
        for (size_t i = 0; i < chunks.size(); ++i) {
            if (effective_length(chunks[i]) >= min_chars) {
                continue;
            }
            if (shortest == chunks.size() || effective_length(chunks[i]) < effective_length(chunks[shortest])) {
                shortest = i;
            }
        }
        if (shortest == chunks.size()) {
            break;
        }
        const bool can_right = shortest + 1 < chunks.size();
        const bool can_left = shortest > 0;
        bool take_right = can_right;
        if (can_right && can_left) {
            const bool right_is_para = gaps[shortest] == Gap::Para;
            const bool left_is_para = gaps[shortest - 1] == Gap::Para;
            if (right_is_para != left_is_para) {
                take_right = !right_is_para;
            } else {
                take_right = chunks[shortest + 1].size() <= chunks[shortest - 1].size();
            }
        }
        if (take_right) {
            chunks[shortest] += ' ';
            chunks[shortest] += chunks[shortest + 1];
            chunks.erase(chunks.begin() + static_cast<std::ptrdiff_t>(shortest) + 1);
            gaps.erase(gaps.begin() + static_cast<std::ptrdiff_t>(shortest));
        } else {
            chunks[shortest - 1] += ' ';
            chunks[shortest - 1] += chunks[shortest];
            chunks.erase(chunks.begin() + static_cast<std::ptrdiff_t>(shortest));
            gaps.erase(gaps.begin() + static_cast<std::ptrdiff_t>(shortest) - 1);
        }
    }

    out.reserve(chunks.size());
    for (size_t i = 0; i < chunks.size(); ++i) {
        PhonemeChunk chunk;
        chunk.phonemes = chunks[i];
        chunk.gap_before = i == 0 ? Gap::Minor : gaps[i - 1];
        out.push_back(std::move(chunk));
    }
    return out;
}

void edge_silence(const std::vector<float> & samples, int64_t sample_rate, int64_t & lead, int64_t & tail) {
    const int64_t window = std::max<int64_t>(static_cast<int64_t>(0.01 * static_cast<double>(sample_rate)), 1);
    const int64_t windows = static_cast<int64_t>(samples.size()) / window;
    if (windows == 0) {
        lead = static_cast<int64_t>(samples.size());
        tail = 0;
        return;
    }
    const double threshold = std::pow(10.0, kEdgeThresholdDb / 20.0);
    const auto loud = [&](int64_t i) {
        double sum = 0.0;
        for (int64_t k = 0; k < window; ++k) {
            sum += std::abs(static_cast<double>(samples[static_cast<size_t>(i * window + k)]));
        }
        return sum / static_cast<double>(window) > threshold;
    };
    int64_t first = -1;
    int64_t last = -1;
    for (int64_t i = 0; i < windows; ++i) {
        if (loud(i)) {
            if (first < 0) {
                first = i;
            }
            last = i;
        }
    }
    if (first < 0) {
        lead = static_cast<int64_t>(samples.size());
        tail = 0;
        return;
    }
    lead = first * window;
    tail = static_cast<int64_t>(samples.size()) - (last + 1) * window;
}

int64_t pause_pad_samples(
    const std::vector<float> & previous,
    const std::vector<float> & next,
    int64_t sample_rate,
    double pause_seconds) {
    int64_t lead_previous = 0;
    int64_t tail = 0;
    edge_silence(previous, sample_rate, lead_previous, tail);
    if (lead_previous == static_cast<int64_t>(previous.size())) {
        tail = static_cast<int64_t>(previous.size());  // an all-silent piece is all tail
    }
    int64_t lead = 0;
    int64_t unused_tail = 0;
    edge_silence(next, sample_rate, lead, unused_tail);
    const int64_t wanted = static_cast<int64_t>(pause_seconds * static_cast<double>(sample_rate));
    return std::max<int64_t>(0, wanted - (tail + lead));
}

int64_t count_speech_bursts(const std::vector<float> & samples, int64_t sample_rate) {
    const int64_t hop = std::max<int64_t>(sample_rate * kHopMs / 1000, 1);
    const auto env = envelope(samples, hop);
    if (env.empty()) {
        return 0;
    }
    const float peak = *std::max_element(env.begin(), env.end());
    if (peak <= 1e-6F) {
        return 0;
    }
    const auto threshold = static_cast<float>(peak * std::pow(10.0, kBurstThresholdDb / 20.0));
    const int64_t min_gap = std::max<int64_t>(kBurstMinGapMs / kHopMs, 1);
    const int64_t min_length = std::max<int64_t>(kBurstMinMs / kHopMs, 1);
    std::vector<std::pair<int64_t, int64_t>> bursts;
    int64_t start = -1;
    int64_t last_on = 0;
    for (int64_t i = 0; i < static_cast<int64_t>(env.size()); ++i) {
        if (env[static_cast<size_t>(i)] > threshold) {
            if (start < 0) {
                start = i;
            } else if (i - last_on > min_gap) {
                bursts.emplace_back(start, last_on);
                start = i;
            }
            last_on = i;
        }
    }
    if (start >= 0) {
        bursts.emplace_back(start, last_on);
    }
    int64_t count = 0;
    for (const auto & burst : bursts) {
        if (burst.second - burst.first + 1 >= min_length) {
            ++count;
        }
    }
    return count;
}

BabbleVerdict babble_suspect(
    const std::vector<float> & samples,
    int64_t sample_rate,
    const std::string & phonemes,
    int64_t cap_frames,
    int64_t frames) {
    BabbleVerdict out;
    out.frames = frames;
    const bool has_cue = phonemes.find("<|emotion_") != std::string::npos;
    const int64_t syllables = syllable_count(phonemes);
    if (has_cue && syllables == 0) {
        // A laugh is many bursts by nature, so only the ceiling can tell.
        out.suspect = frames + 1 >= cap_frames;
        return out;
    }
    out.syllables = syllables;
    // Longer chunks have merged bursts, and a chunk that mixes words with a cue
    // cannot be counted either.
    if (syllables == 0 || syllables > 3 || has_cue) {
        return out;
    }
    out.bursts = count_speech_bursts(samples, sample_rate);
    const bool hit_cap = syllables <= 2 && frames + 1 >= cap_frames;
    out.suspect = out.bursts > syllables || hit_cap;
    return out;
}

bool babble_prefer(const BabbleVerdict & candidate, const BabbleVerdict & current) {
    if (candidate.suspect != current.suspect) {
        return !candidate.suspect;
    }
    if (candidate.bursts != current.bursts) {
        return candidate.bursts < current.bursts;
    }
    return candidate.frames < current.frames;
}

}  // namespace engine::models::vieneu_v3_turbo
