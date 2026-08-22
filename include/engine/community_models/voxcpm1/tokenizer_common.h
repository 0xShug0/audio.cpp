#pragma once

#include "engine/community_models/voxcpm1/types.h"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace engine::community_models::voxcpm1 {

// Common tokenizer implementation shared by both JSON-based and GGUF-based tokenizers
class VoxCPM1TokenizerCommon {
public:
    struct Impl;

    // Constructor takes pre-loaded tokenizer data
    VoxCPM1TokenizerCommon(
        std::unordered_map<std::string, int32_t> vocab,
        std::unordered_map<std::string, int32_t> merge_ranks,
        std::unordered_map<std::string, int32_t> special_tokens,
        int32_t bos_token_id,
        int32_t eos_token_id,
        int32_t unk_token_id,
        int32_t audio_start_token_id,
        int32_t audio_end_token_id,
        int32_t reference_audio_start_token_id,
        int32_t reference_audio_end_token_id);

    std::vector<int32_t> encode(const std::string & text) const;
    VoxCPM1TextPrompt build_prompt(const std::string & text) const;
    int32_t audio_start_token_id() const noexcept;
    int32_t audio_end_token_id() const noexcept;
    int32_t reference_audio_start_token_id() const noexcept;
    int32_t reference_audio_end_token_id() const noexcept;
    int32_t bos_token_id() const noexcept;
    int32_t eos_token_id() const noexcept;
    int32_t unk_token_id() const noexcept;

private:
    std::shared_ptr<const Impl> impl_;
};

}  // namespace engine::community_models::voxcpm1