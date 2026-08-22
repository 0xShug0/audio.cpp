#pragma once

#include "engine/community_models/voxcpm1/tokenizer_common.h"
#include "engine/community_models/voxcpm1/types.h"

#include <memory>

namespace engine::community_models::voxcpm1 {

// Forward declaration
struct VoxCPM1Assets;

class VoxCPM1TextTokenizer {
public:
    explicit VoxCPM1TextTokenizer(std::shared_ptr<const VoxCPM1Assets> assets);

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
    std::shared_ptr<const VoxCPM1TokenizerCommon> common_;
};

}  // namespace engine::community_models::voxcpm1