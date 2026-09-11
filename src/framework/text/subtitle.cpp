#include "engine/framework/text/subtitle.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace engine::text {
namespace {

void validate_options(const SubtitleFormatOptions & options) {
    if (options.sample_rate <= 0) {
        throw std::runtime_error("subtitle formatter requires a positive sample_rate");
    }
    if (options.max_chars_per_line <= 0) {
        throw std::runtime_error("subtitle formatter requires max_chars_per_line > 0");
    }
    if (options.max_lines <= 0) {
        throw std::runtime_error("subtitle formatter requires max_lines > 0");
    }
    if (!(options.max_block_seconds > 0.0)) {
        throw std::runtime_error("subtitle formatter requires max_block_seconds > 0");
    }
    if (options.max_gap_seconds < 0.0) {
        throw std::runtime_error("subtitle formatter requires max_gap_seconds >= 0");
    }
}

void validate_word(const runtime::WordTimestamp & word, int64_t previous_start) {
    if (word.word.empty()) {
        throw std::runtime_error("subtitle formatter received an empty word");
    }
    if (word.word.find('\n') != std::string::npos || word.word.find('\r') != std::string::npos) {
        throw std::runtime_error("subtitle formatter received a word containing a newline");
    }
    if (word.span.start_sample < 0 || word.span.end_sample < 0) {
        throw std::runtime_error("subtitle formatter received a negative timestamp");
    }
    if (word.span.end_sample < word.span.start_sample) {
        throw std::runtime_error("subtitle formatter received an invalid timestamp span");
    }
    if (word.span.start_sample < previous_start) {
        throw std::runtime_error("subtitle formatter requires sorted word timestamps");
    }
}

int64_t seconds_to_samples(double seconds, int sample_rate) {
    return static_cast<int64_t>(std::llround(seconds * static_cast<double>(sample_rate)));
}

std::vector<std::string> wrap_words(
    const std::vector<runtime::WordTimestamp> & words,
    size_t begin,
    size_t end,
    int max_chars_per_line) {
    std::vector<std::string> lines;
    std::string line;
    for (size_t i = begin; i < end; ++i) {
        const std::string & word = words[i].word;
        if (line.empty()) {
            line = word;
        } else if (line.size() + 1 + word.size() <= static_cast<size_t>(max_chars_per_line)) {
            line.push_back(' ');
            line += word;
        } else {
            lines.push_back(line);
            line = word;
        }
    }
    if (!line.empty()) {
        lines.push_back(line);
    }
    return lines;
}

bool would_exceed_text_limit(
    const std::vector<runtime::WordTimestamp> & words,
    size_t begin,
    size_t candidate_end,
    const SubtitleFormatOptions & options) {
    const auto lines = wrap_words(words, begin, candidate_end, options.max_chars_per_line);
    return static_cast<int>(lines.size()) > options.max_lines;
}

std::string format_time(int64_t sample, int sample_rate) {
    const int64_t ms = (sample * 1000 + sample_rate / 2) / sample_rate;
    const int64_t hours = ms / 3600000;
    const int64_t minutes = (ms / 60000) % 60;
    const int64_t seconds = (ms / 1000) % 60;
    const int64_t millis = ms % 1000;

    std::ostringstream out;
    out << std::setfill('0')
        << std::setw(2) << hours << ':'
        << std::setw(2) << minutes << ':'
        << std::setw(2) << seconds << ','
        << std::setw(3) << millis;
    return out.str();
}

}  // namespace

std::vector<SubtitleBlock> build_subtitle_blocks(
    const std::vector<runtime::WordTimestamp> & words,
    const SubtitleFormatOptions & options) {
    validate_options(options);
    if (words.empty()) {
        return {};
    }

    const int64_t max_block_samples = seconds_to_samples(options.max_block_seconds, options.sample_rate);
    const int64_t max_gap_samples = seconds_to_samples(options.max_gap_seconds, options.sample_rate);
    std::vector<SubtitleBlock> blocks;
    size_t block_begin = 0;
    int64_t previous_start = 0;

    for (size_t i = 0; i < words.size(); ++i) {
        validate_word(words[i], previous_start);
        previous_start = words[i].span.start_sample;

        if (i == block_begin) {
            continue;
        }

        const bool gap_break = words[i].span.start_sample - words[i - 1].span.end_sample > max_gap_samples;
        const bool duration_break = words[i].span.end_sample - words[block_begin].span.start_sample > max_block_samples;
        const bool text_break = would_exceed_text_limit(words, block_begin, i + 1, options);
        if (!gap_break && !duration_break && !text_break) {
            continue;
        }

        SubtitleBlock block;
        block.index = static_cast<int>(blocks.size()) + 1;
        block.span.start_sample = words[block_begin].span.start_sample;
        block.span.end_sample = words[i - 1].span.end_sample;
        block.lines = wrap_words(words, block_begin, i, options.max_chars_per_line);
        blocks.push_back(std::move(block));
        block_begin = i;
    }

    SubtitleBlock block;
    block.index = static_cast<int>(blocks.size()) + 1;
    block.span.start_sample = words[block_begin].span.start_sample;
    block.span.end_sample = words.back().span.end_sample;
    block.lines = wrap_words(words, block_begin, words.size(), options.max_chars_per_line);
    blocks.push_back(std::move(block));
    return blocks;
}

std::string format_srt(
    const std::vector<runtime::WordTimestamp> & words,
    const SubtitleFormatOptions & options) {
    const auto blocks = build_subtitle_blocks(words, options);
    std::ostringstream out;
    for (const auto & block : blocks) {
        out << block.index << "\n"
            << format_time(block.span.start_sample, options.sample_rate)
            << " --> "
            << format_time(block.span.end_sample, options.sample_rate)
            << "\n";
        for (const auto & line : block.lines) {
            out << line << "\n";
        }
        out << "\n";
    }
    return out.str();
}

}  // namespace engine::text
