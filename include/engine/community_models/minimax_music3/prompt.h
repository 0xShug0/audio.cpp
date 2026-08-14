#pragma once

#include "engine/community_models/minimax_music3/assets.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::minimax_music3 {

struct MiniMaxMusic3Prompt {
    std::vector<int32_t> conditional_ids;
    std::vector<int32_t> unconditional_ids;
    int32_t audio_end_token_id = 151670;
    int32_t audio_cfg_token_id = 151654;
    int32_t audio_code_offset = 151675;
    int32_t semantic_vocab_size = 16384;
};

class MiniMaxMusic3PromptBuilder {
public:
    explicit MiniMaxMusic3PromptBuilder(std::shared_ptr<const MiniMaxMusic3Assets> assets);
    MiniMaxMusic3Prompt build(const std::string & prompt, const std::string & lyrics) const;

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

}  // namespace engine::models::minimax_music3

