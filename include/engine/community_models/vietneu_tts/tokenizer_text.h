#pragma once

#include "engine/community_models/vietneu_tts/assets.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::vietneu_tts {

class Qwen3TextTokenizer {
public:
    struct Impl;

    explicit Qwen3TextTokenizer(std::shared_ptr<const VietneuTTSAssets> assets);

    std::string build_assistant_prompt(const std::string & text) const;
    std::string build_reference_prompt(const std::string & text) const;
    std::string build_instruct_prompt(const std::string & text) const;
    std::vector<int32_t> encode(const std::string & text) const;

private:
    std::shared_ptr<const Impl> impl_;
};

}  // namespace engine::models::vietneu_tts
