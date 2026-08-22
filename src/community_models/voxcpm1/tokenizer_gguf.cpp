#include "engine/community_models/voxcpm1/tokenizer_gguf.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/community_models/voxcpm1/gguf_metadata.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace engine::community_models::voxcpm1 {

VoxCPM1GgufTokenizer::VoxCPM1GgufTokenizer(std::shared_ptr<const engine::assets::TensorSource> gguf_source) {
    if (!gguf_source) {
        throw std::runtime_error("VoxCPM1 GGUF tokenizer requires a valid GGUF tensor source");
    }
    const GgufMetadataReader metadata(*gguf_source);
    if (!metadata.valid()) {
        throw std::runtime_error("VoxCPM1 GGUF tokenizer requires a GGUF tensor source");
    }

    // Read tokenizer metadata from GGUF directly in constructor
    const std::string tokenizer_model = metadata.require_string("tokenizer.ggml.model");
    const std::string tokenizer_pre = metadata.require_string("tokenizer.ggml.pre");
    const std::vector<std::string> tokens = metadata.require_string_array("tokenizer.ggml.tokens");
    const std::vector<int32_t> token_types = metadata.require_i32_array("tokenizer.ggml.token_type");
    const std::vector<std::string> merges = metadata.require_string_array("tokenizer.ggml.merges");
    const uint32_t bos_id = metadata.require_u32("tokenizer.ggml.bos_token_id");
    const uint32_t eos_id = metadata.require_u32("tokenizer.ggml.eos_token_id");
    const uint32_t unk_id = metadata.require_u32("tokenizer.ggml.unknown_token_id");

    if (tokenizer_model != "gpt2" || tokens.empty() || merges.empty() || token_types.size() != tokens.size()) {
        throw std::runtime_error("Invalid VoxCPM1 GGUF tokenizer metadata");
    }

    constexpr int32_t kTokenTypeNormal = 1;
    constexpr int32_t kTokenTypeByte = 6;

    std::unordered_map<std::string, int32_t> vocab;
    std::unordered_map<std::string, int32_t> special_tokens;
    std::unordered_map<std::string, int32_t> merge_ranks;

    for (size_t i = 0; i < tokens.size(); ++i) {
        const int32_t id = static_cast<int32_t>(i);
        vocab.emplace(tokens[i], id);
        if (token_types[i] != kTokenTypeNormal && token_types[i] != kTokenTypeByte) {
            special_tokens.emplace(tokens[i], id);
        }
    }

    // Build merge ranks
    int32_t rank = 0;
    for (const std::string & merge_text : merges) {
        const size_t split = merge_text.find(' ');
        if (split == std::string::npos) {
            ++rank;
            continue;
        }
        const std::string left = merge_text.substr(0, split);
        const std::string right = merge_text.substr(split + 1);
        const auto left_it = vocab.find(left);
        const auto right_it = vocab.find(right);
        const auto merged_it = vocab.find(left + right);
        if (left_it != vocab.end() && right_it != vocab.end() && merged_it != vocab.end()) {
            merge_ranks.emplace(left + '\0' + right, rank);
        }
        ++rank;
    }

    if (merge_ranks.empty()) {
        throw std::runtime_error("VoxCPM1 GGUF tokenizer has no valid merge rules");
    }

    const int32_t audio_start_token_id = 101;
    const int32_t audio_end_token_id = 102;
    const int32_t reference_audio_start_token_id = 103;
    const int32_t reference_audio_end_token_id = 104;
    const int32_t bos_token_id = static_cast<int32_t>(bos_id);
    const int32_t eos_token_id = static_cast<int32_t>(eos_id);
    const int32_t unk_token_id = static_cast<int32_t>(unk_id);

    common_ = std::make_shared<const VoxCPM1TokenizerCommon>(
        std::move(vocab),
        std::move(merge_ranks),
        std::move(special_tokens),
        bos_token_id,
        eos_token_id,
        unk_token_id,
        audio_start_token_id,
        audio_end_token_id,
        reference_audio_start_token_id,
        reference_audio_end_token_id);
}

std::vector<int32_t> VoxCPM1GgufTokenizer::encode(const std::string & text) const {
    return common_->encode(text);
}

VoxCPM1TextPrompt VoxCPM1GgufTokenizer::build_prompt(const std::string & text) const {
    return common_->build_prompt(text);
}

int32_t VoxCPM1GgufTokenizer::audio_start_token_id() const noexcept {
    return common_->audio_start_token_id();
}

int32_t VoxCPM1GgufTokenizer::audio_end_token_id() const noexcept {
    return common_->audio_end_token_id();
}

int32_t VoxCPM1GgufTokenizer::reference_audio_start_token_id() const noexcept {
    return common_->reference_audio_start_token_id();
}

int32_t VoxCPM1GgufTokenizer::reference_audio_end_token_id() const noexcept {
    return common_->reference_audio_end_token_id();
}

int32_t VoxCPM1GgufTokenizer::bos_token_id() const noexcept {
    return common_->bos_token_id();
}

int32_t VoxCPM1GgufTokenizer::eos_token_id() const noexcept {
    return common_->eos_token_id();
}

int32_t VoxCPM1GgufTokenizer::unk_token_id() const noexcept {
    return common_->unk_token_id();
}

bool VoxCPM1GgufTokenizer::has_tokenizer_metadata(const engine::assets::TensorSource & source) {
    const GgufMetadataReader metadata(source);
    return metadata.optional_string("tokenizer.ggml.model").has_value() &&
           metadata.optional_string_array("tokenizer.ggml.tokens").has_value() &&
           metadata.optional_string_array("tokenizer.ggml.merges").has_value();
}

}  // namespace engine::community_models::voxcpm1