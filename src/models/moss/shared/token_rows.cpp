#include "engine/models/moss/shared/token_rows.h"

#include <stdexcept>

namespace engine::models::moss {

TokenRowBuilder::TokenRowBuilder(int64_t num_codebooks, int32_t audio_pad_token_id)
    : num_codebooks_(num_codebooks),
      audio_pad_token_id_(audio_pad_token_id) {
    if (num_codebooks_ <= 0) {
        throw std::runtime_error("MOSS token row builder requires a positive codebook count");
    }
}

void TokenRowBuilder::push_text_token(int32_t token_id) {
    rows_.text_tokens.push_back(token_id);
    rows_.audio_codes.insert(rows_.audio_codes.end(), static_cast<size_t>(num_codebooks_), audio_pad_token_id_);
}

void TokenRowBuilder::push_text_tokens(const std::vector<int32_t> & token_ids) {
    for (const int32_t token_id : token_ids) {
        push_text_token(token_id);
    }
}

void TokenRowBuilder::push_audio_row(
    int32_t text_slot_token_id,
    const std::vector<std::vector<int32_t>> & codes,
    int64_t frame) {
    if (static_cast<int64_t>(codes.size()) != num_codebooks_) {
        throw std::runtime_error("MOSS audio row codebook count mismatch");
    }
    rows_.text_tokens.push_back(text_slot_token_id);
    for (int64_t codebook = 0; codebook < num_codebooks_; ++codebook) {
        const auto & channel = codes[static_cast<size_t>(codebook)];
        if (frame < 0 || static_cast<size_t>(frame) >= channel.size()) {
            throw std::runtime_error("MOSS audio row frame index is out of range");
        }
        rows_.audio_codes.push_back(channel[static_cast<size_t>(frame)]);
    }
}

TokenRows TokenRowBuilder::finish() {
    if (rows_.text_tokens.empty()) {
        throw std::runtime_error("MOSS token rows must not be empty");
    }
    if (static_cast<int64_t>(rows_.audio_codes.size()) !=
        static_cast<int64_t>(rows_.text_tokens.size()) * num_codebooks_) {
        throw std::runtime_error("MOSS token rows audio code shape mismatch");
    }
    return std::move(rows_);
}

}  // namespace engine::models::moss
