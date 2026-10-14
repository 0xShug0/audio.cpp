#pragma once

#include "engine/community_models/granite5asr/encoder.h"
#include "engine/framework/io/text.h"
#include "engine/framework/runtime/session.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace engine::community_models::granite5asr {

struct CtcPiece {
    int32_t id = 0;
    int64_t start_frame = 0;
    int64_t end_frame = 0;
    double log_prob_sum = 0.0;
    int64_t frames = 0;
};

// A collapsed piece whose single-id text is already decoded.
struct DecodedCtcPiece {
    std::string text;
    int64_t start_frame = 0;
    int64_t end_frame = 0;
    double log_prob_sum = 0.0;
    int64_t frames = 0;
};

inline void accumulate_piece_confidence(CtcPiece & piece, const std::vector<CtcArgmaxFrame> & frames) {
    piece.frames = 0;
    piece.log_prob_sum = 0.0;
    if (piece.end_frame <= piece.start_frame) {
        return;
    }
    for (int64_t frame = piece.start_frame; frame < piece.end_frame; ++frame) {
        float probability = frames[static_cast<size_t>(frame)].probability;
        if (!(probability > 0.f)) {
            piece.frames = 0;
            piece.log_prob_sum = -std::numeric_limits<double>::infinity();
            return;
        }
        if (probability > 1.f) {
            probability = 1.f;
        }
        piece.log_prob_sum += std::log(static_cast<double>(probability));
        ++piece.frames;
    }
}

// Greedy CTC collapse that keeps the frame span of each emitted token.
// Blanks and repeated labels are removed, and a token's end frame is the first
// frame of the next distinct label, so word bounds come from the encoder clock.
// Confidence is the geometric mean of the greedy-class probability on the
// non-blank frames of that token. Trailing blanks are not part of the score.
inline std::vector<CtcPiece> ctc_greedy_pieces(
    const std::vector<CtcArgmaxFrame> & frames,
    int64_t blank_id) {
    std::vector<CtcPiece> pieces;
    int32_t prev = -1;
    bool open = false;
    for (int64_t frame = 0; frame < static_cast<int64_t>(frames.size()); ++frame) {
        const int32_t id = frames[static_cast<size_t>(frame)].id;
        if (id == prev) {
            continue;
        }
        if (open) {
            pieces.back().end_frame = frame;
            accumulate_piece_confidence(pieces.back(), frames);
            open = false;
        }
        if (id != static_cast<int32_t>(blank_id)) {
            pieces.push_back(CtcPiece{id, frame, frame + 1, 0.0, 0});
            open = true;
        }
        prev = id;
    }
    if (open && !pieces.empty()) {
        pieces.back().end_frame = static_cast<int64_t>(frames.size());
        accumulate_piece_confidence(pieces.back(), frames);
    }
    return pieces;
}

inline int64_t samples_per_ctc_frame(const Granite5ASRConfig & config) {
    int64_t factor = std::max<int64_t>(1, config.frontend.hop_length);
    factor *= std::max<int64_t>(1, config.frontend.stack_factor);
    int64_t subsample = 1;
    for (size_t i = 0; i < config.encoder.subsample_layers.size(); ++i) {
        subsample *= 2;
    }
    return std::max<int64_t>(1, factor * subsample);
}

inline std::vector<runtime::WordTimestamp> words_from_decoded_ctc_pieces(
    const std::vector<DecodedCtcPiece> & pieces,
    int64_t samples_per_frame,
    int64_t sample_offset,
    int64_t audio_samples) {
    struct Acc {
        std::string text;
        int64_t start = 0;
        int64_t end = 0;
        double log_prob_sum = 0.0;
        int64_t frames = 0;
        bool open = false;
    };
    Acc current;
    std::vector<runtime::WordTimestamp> words;
    auto flush = [&]() {
        const auto text = engine::io::trim_ascii_whitespace(current.text);
        if (!current.open || text.empty()) {
            current.open = false;
            return;
        }
        runtime::WordTimestamp word;
        word.word = text;
        word.span.start_sample = current.start;
        word.span.end_sample = std::max(current.end, current.start + 1);
        if (audio_samples > 0 && word.span.end_sample > audio_samples) {
            word.span.end_sample = audio_samples;
        }
        if (word.span.end_sample <= word.span.start_sample) {
            word.span.end_sample = word.span.start_sample + 1;
        }
        float confidence = 0.f;
        if (current.frames > 0 && std::isfinite(current.log_prob_sum)) {
            double mean = std::exp(current.log_prob_sum / static_cast<double>(current.frames));
            if (mean < 0.0) {
                mean = 0.0;
            }
            if (mean > 1.0) {
                mean = 1.0;
            }
            confidence = static_cast<float>(mean);
        }
        word.confidence = confidence;
        words.push_back(std::move(word));
        current.open = false;
    };
    for (const auto & piece : pieces) {
        std::string decoded = piece.text;
        const bool boundary = !decoded.empty() && decoded.front() == ' ';
        if (boundary) {
            flush();
            size_t i = 0;
            while (i < decoded.size() && decoded[i] == ' ') {
                ++i;
            }
            decoded.erase(0, i);
        }
        if (decoded.empty()) {
            continue;
        }
        const int64_t start = sample_offset + piece.start_frame * samples_per_frame;
        const int64_t end = sample_offset + piece.end_frame * samples_per_frame;
        if (!current.open) {
            current = Acc{decoded, start, end, piece.log_prob_sum, piece.frames, true};
        } else {
            current.text += decoded;
            current.end = end;
            current.log_prob_sum += piece.log_prob_sum;
            current.frames += piece.frames;
        }
    }
    flush();
    return words;
}

}  // namespace engine::community_models::granite5asr
