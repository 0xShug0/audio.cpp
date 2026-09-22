#pragma once

#include "engine/community_models/vieneu_v3_turbo/assets.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::vieneu_v3_turbo {

class Qwen3TextTokenizer {
public:
    struct Impl;

    explicit Qwen3TextTokenizer(std::shared_ptr<const VieNeuTTSAssets> assets);

    std::string build_assistant_prompt(const std::string & text) const;
    std::string build_reference_prompt(const std::string & text) const;
    std::string build_instruct_prompt(const std::string & text) const;
    std::vector<int32_t> encode(const std::string & text) const;

private:
    std::shared_ptr<const Impl> impl_;
};

}  // namespace engine::models::vieneu_v3_turbo
