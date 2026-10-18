#pragma once

#include "engine/community_models/mira_tts/assets.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine::community_models::mira_tts {

class MiraPromptBuilder final {
public:
    explicit MiraPromptBuilder(std::shared_ptr<const MiraTTSAssets> assets);
    ~MiraPromptBuilder();

    std::vector<int32_t> build(
        const std::string & text,
        const std::vector<int32_t> & context_codes) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::community_models::mira_tts
