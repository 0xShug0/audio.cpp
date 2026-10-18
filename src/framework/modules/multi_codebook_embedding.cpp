#include "engine/framework/modules/multi_codebook_embedding.h"

#include <stdexcept>

namespace engine::modules {

MultiCodebookEmbedding::MultiCodebookEmbedding(const assets::TensorSource & source, MultiCodebookEmbeddingSpec spec)
    : hidden_size_(spec.hidden_size),
      num_codebooks_(spec.num_codebooks),
      pad_token_id_(static_cast<int32_t>(spec.pad_token_id)) {
    if (hidden_size_ <= 0 || num_codebooks_ <= 0) {
        throw std::runtime_error("multi-codebook embedding requires positive dimensions");
    }
    if (!spec.codebook_sizes.empty() && static_cast<int64_t>(spec.codebook_sizes.size()) != num_codebooks_) {
        throw std::runtime_error("multi-codebook embedding codebook size count mismatch");
    }
    embeddings_.reserve(static_cast<size_t>(num_codebooks_));
    for (int64_t codebook = 0; codebook < num_codebooks_; ++codebook) {
        const int64_t size = spec.codebook_sizes.empty()
            ? spec.vocab_size
            : spec.codebook_sizes[static_cast<size_t>(codebook)];
        if (size <= 0) {
            throw std::runtime_error("multi-codebook embedding has an invalid codebook size");
        }
        embeddings_.push_back(source.require_f32(
            spec.tensor_prefix + "." + std::to_string(codebook) + ".weight", {size, hidden_size_}));
    }
}

int64_t MultiCodebookEmbedding::codebook_size(int64_t codebook) const {
    if (codebook < 0 || codebook >= num_codebooks_) {
        throw std::runtime_error("multi-codebook embedding index is out of range");
    }
    return static_cast<int64_t>(embeddings_[static_cast<size_t>(codebook)].size()) / hidden_size_;
}

const float * MultiCodebookEmbedding::embedding(int64_t codebook, int32_t code) const {
    const int64_t size = codebook_size(codebook);
    if (code < 0 || code >= size) {
        throw std::runtime_error("multi-codebook embedding code is out of range");
    }
    return embeddings_[static_cast<size_t>(codebook)].data() + static_cast<size_t>(code) * hidden_size_;
}

void MultiCodebookEmbedding::add_bias(const int32_t * codes, float * bias) const {
    for (int64_t codebook = 0; codebook < num_codebooks_; ++codebook) {
        const int32_t code = codes[codebook];
        if (code == pad_token_id_) {
            continue;
        }
        const float * row = embedding(codebook, code);
        for (int64_t index = 0; index < hidden_size_; ++index) {
            bias[static_cast<size_t>(index)] += row[index];
        }
    }
}

std::vector<float> MultiCodebookEmbedding::bias_for(const int32_t * codes) const {
    std::vector<float> bias(static_cast<size_t>(hidden_size_), 0.0F);
    add_bias(codes, bias.data());
    return bias;
}

}  // namespace engine::modules
