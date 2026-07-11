#include "engine/models/moss/shared/token_rows.h"

#include <algorithm>
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

AudioCodebookEmbeddings::AudioCodebookEmbeddings(const assets::TensorSource & source, AudioCodebookSpec spec)
    : hidden_size_(spec.hidden_size),
      num_codebooks_(spec.num_codebooks),
      audio_pad_token_id_(static_cast<int32_t>(spec.audio_pad_token_id)) {
    if (hidden_size_ <= 0 || num_codebooks_ <= 0) {
        throw std::runtime_error("MOSS audio codebook embeddings require positive dimensions");
    }
    embeddings_.reserve(static_cast<size_t>(num_codebooks_));
    for (int64_t codebook = 0; codebook < num_codebooks_; ++codebook) {
        const int64_t size = spec.audio_codebook_sizes.empty()
            ? spec.audio_vocab_size
            : spec.audio_codebook_sizes[static_cast<size_t>(codebook)];
        if (size <= 0) {
            throw std::runtime_error("MOSS audio codebook has an invalid size");
        }
        embeddings_.push_back(source.require_f32(
            spec.tensor_prefix + "." + std::to_string(codebook) + ".weight", {size, hidden_size_}));
    }
}

int64_t AudioCodebookEmbeddings::codebook_size(int64_t codebook) const {
    if (codebook < 0 || codebook >= num_codebooks_) {
        throw std::runtime_error("MOSS audio codebook index is out of range");
    }
    return static_cast<int64_t>(embeddings_[static_cast<size_t>(codebook)].size()) / hidden_size_;
}

const float * AudioCodebookEmbeddings::embedding(int64_t codebook, int32_t code) const {
    const int64_t size = codebook_size(codebook);
    if (code < 0 || code >= size) {
        throw std::runtime_error("MOSS audio code is out of range");
    }
    return embeddings_[static_cast<size_t>(codebook)].data() + static_cast<size_t>(code) * hidden_size_;
}

void AudioCodebookEmbeddings::add_bias(const int32_t * codes, float * bias) const {
    for (int64_t codebook = 0; codebook < num_codebooks_; ++codebook) {
        const int32_t code = codes[codebook];
        if (code == audio_pad_token_id_) {
            continue;
        }
        const float * row = embedding(codebook, code);
        for (int64_t index = 0; index < hidden_size_; ++index) {
            bias[static_cast<size_t>(index)] += row[index];
        }
    }
}

std::vector<float> AudioCodebookEmbeddings::bias_for(const int32_t * codes) const {
    std::vector<float> bias(static_cast<size_t>(hidden_size_), 0.0F);
    add_bias(codes, bias.data());
    return bias;
}

}  // namespace engine::models::moss
