#pragma once

#include "engine/community_models/granite5asr/encoder.h"
#include "engine/framework/io/text.h"
#include "engine/framework/runtime/session.h"

#include <algorithm>
#include <cctype>
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

// The frontend resamples to the model rate, so frame indexes count model-rate
// samples. Word spans are reported in samples of the input buffer.
// Streaming audio is resampled to 16 kHz before the encoder. Word indexes from
// that buffer are converted onto the sample rate of the audio that was streamed.
inline int64_t scale_sample_to_rate(int64_t sample, int64_t from_rate, int64_t to_rate) {
    if (sample <= 0 || from_rate <= 0 || to_rate <= 0 || from_rate == to_rate) {
        return std::max<int64_t>(sample, 0);
    }
    return static_cast<int64_t>(std::llround(
        static_cast<double>(sample) * static_cast<double>(to_rate) /
        static_cast<double>(from_rate)));
}

// The multilingual checkpoint's tokenizer carries Parakeet language tags.
// Only the languages the model was trained to transcribe count. The English-only
// checkpoint has none of these tokens, so this returns empty for it.
inline std::string language_code_from_token(const std::string & token) {
    if (token.size() < 5 || token.front() != '<' || token.back() != '>') {
        return {};
    }
    const std::string code = token.substr(2, token.size() - 4);
    if (code == "en" || code == "fr" || code == "de" || code == "es" ||
        code == "pt" || code == "it" || code == "ja") {
        return code;
    }
    return {};
}

inline int64_t ctc_frame_to_input_sample(
    int64_t frame,
    int64_t samples_per_frame,
    int64_t model_sample_rate,
    int64_t input_sample_rate) {
    const int64_t model_sample = frame * samples_per_frame;
    if (model_sample_rate <= 0 || input_sample_rate <= 0 || model_sample_rate == input_sample_rate) {
        return model_sample;
    }
    return static_cast<int64_t>(std::llround(
        static_cast<double>(model_sample) * static_cast<double>(input_sample_rate) /
        static_cast<double>(model_sample_rate)));
}

// A piece that is only punctuation is written onto the current word. Its frames
// often run through the following pause, so they do not move the word's end or
// enter its confidence.
inline bool punctuation_only_piece(const std::string & text) {
    bool saw_mark = false;
    size_t index = 0;
    while (index < text.size()) {
        const auto lead = static_cast<unsigned char>(text[index]);
        uint32_t code = 0;
        size_t width = 1;
        if (lead < 0x80) {
            code = lead;
        } else if ((lead & 0xE0) == 0xC0 && index + 1 < text.size()) {
            code = (lead & 0x1F) << 6 | (static_cast<unsigned char>(text[index + 1]) & 0x3F);
            width = 2;
        } else if ((lead & 0xF0) == 0xE0 && index + 2 < text.size()) {
            code = (lead & 0x0F) << 12 |
                (static_cast<unsigned char>(text[index + 1]) & 0x3F) << 6 |
                (static_cast<unsigned char>(text[index + 2]) & 0x3F);
            width = 3;
        } else if ((lead & 0xF8) == 0xF0 && index + 3 < text.size()) {
            code = (lead & 0x07) << 18 |
                (static_cast<unsigned char>(text[index + 1]) & 0x3F) << 12 |
                (static_cast<unsigned char>(text[index + 2]) & 0x3F) << 6 |
                (static_cast<unsigned char>(text[index + 3]) & 0x3F);
            width = 4;
        } else {
            return false;
        }
        index += width;
        if (code == ' ' || code == '\t') {
            continue;
        }
        const bool ascii_word = code < 128 &&
            (std::isalnum(static_cast<unsigned char>(code)) || code == '\'');
        const bool letter =
            (code >= 0x00C0 && code <= 0x024F) ||
            (code >= 0x3040 && code <= 0x30FF) ||
            (code >= 0x3400 && code <= 0x9FFF) ||
            (code >= 0xAC00 && code <= 0xD7AF) ||
            (code >= 0xFF10 && code <= 0xFF19) ||
            (code >= 0xFF21 && code <= 0xFF3A) ||
            (code >= 0xFF41 && code <= 0xFF5A);
        if (ascii_word || letter) {
            return false;
        }
        saw_mark = true;
    }
    return saw_mark;
}

inline std::vector<runtime::WordTimestamp> words_from_decoded_ctc_pieces(
    const std::vector<DecodedCtcPiece> & pieces,
    int64_t samples_per_frame,
    int64_t sample_offset,
    int64_t max_end_sample,
    int64_t model_sample_rate,
    int64_t input_sample_rate) {
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
        if (max_end_sample > 0 && word.span.end_sample > max_end_sample) {
            word.span.end_sample = max_end_sample;
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
        if (punctuation_only_piece(decoded)) {
            if (current.open) {
                current.text += decoded;
            } else if (!words.empty()) {
                words.back().word += decoded;
            }
            continue;
        }
        const int64_t start = sample_offset + ctc_frame_to_input_sample(
            piece.start_frame, samples_per_frame, model_sample_rate, input_sample_rate);
        const int64_t end = sample_offset + ctc_frame_to_input_sample(
            piece.end_frame, samples_per_frame, model_sample_rate, input_sample_rate);
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
