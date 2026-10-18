#pragma once

#include "engine/framework/io/text.h"
#include "engine/framework/runtime/session.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace engine::community_models::parakeet_tdt {

struct TimestampPiece {
    std::string text;
    int64_t start_sample;
    int64_t duration_samples;
    bool punctuation;
};

// Phonon's word policy: preserve a bare word marker's timing, attach standalone
// punctuation, and use the last token's actual duration (which may be zero).
// Unicode punctuation classification is supplied by the converted tokenizer
// metadata, rather than applying a platform-dependent C locale to UTF-8 bytes.
inline std::vector<runtime::WordTimestamp> token_duration_word_timestamps(
    const std::vector<TimestampPiece> & pieces, int64_t audio_end_sample) {
    std::vector<runtime::WordTimestamp> words;
    std::string current;
    bool has_piece = false;
    int64_t start = 0;
    int64_t end = 0;
    auto flush = [&]() {
        auto text = engine::io::trim_ascii_whitespace(std::move(current));
        if (!text.empty()) {
            runtime::WordTimestamp word;
            word.word = std::move(text);
            word.span.start_sample = start;
            word.span.end_sample = std::max(start, std::min(end, audio_end_sample));
            word.confidence = 0.f;
            words.push_back(std::move(word));
        }
        current.clear();
        has_piece = false;
    };
    for (const auto & piece : pieces) {
        const bool starts_word = !piece.text.empty() && piece.text.front() == ' ';
        if (starts_word && has_piece &&
            (engine::io::trim_ascii_whitespace(piece.text).empty() || !piece.punctuation)) {
            flush();
        }
        if (!has_piece) {
            start = piece.start_sample;
            has_piece = true;
        }
        current += piece.text;
        end = piece.start_sample + piece.duration_samples;
    }
    flush();
    return words;
}

}  // namespace engine::community_models::parakeet_tdt
