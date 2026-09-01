#pragma once

#include "engine/community_models/voxcpm1/types.h"
#include "engine/framework/assets/tensor_source.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine::community_models::voxcpm1 {

// GGUF-native tokenizer that reads tokenizer metadata directly from GGUF.
// VoxCPM1 is now GGUF-only — no JSON fallback.
class VoxCPM1GgufTokenizer {
public:
    struct Impl;

    explicit VoxCPM1GgufTokenizer(std::shared_ptr<const engine::assets::TensorSource> gguf_source);

    std::vector<int32_t> encode(const std::string & text) const;
    VoxCPM1TextPrompt build_prompt(const std::string & text) const;
    int32_t audio_start_token_id() const noexcept;
    int32_t audio_end_token_id() const noexcept;
    int32_t reference_audio_start_token_id() const noexcept;
    int32_t reference_audio_end_token_id() const noexcept;
    int32_t bos_token_id() const noexcept;
    int32_t eos_token_id() const noexcept;
    int32_t unk_token_id() const noexcept;

    // Check if the GGUF source has tokenizer metadata
    static bool has_tokenizer_metadata(const engine::assets::TensorSource & source);

private:
    std::shared_ptr<const Impl> impl_;
};

}  // namespace engine::community_models::voxcpm1
