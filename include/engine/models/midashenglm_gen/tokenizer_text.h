#pragma once

#include "engine/models/midashenglm_gen/assets.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::midashenglm_gen {

struct MiDashengLmGenPromptBatch {
    std::vector<std::string> prompts;
    std::vector<int32_t> token_ids;
    std::vector<int32_t> attention_mask;
    int64_t batch = 0;
    int64_t tokens = 0;
};

class MiDashengLmGenTextTokenizer {
public:
    explicit MiDashengLmGenTextTokenizer(std::shared_ptr<const MiDashengLmGenAssets> assets);
    ~MiDashengLmGenTextTokenizer();

    MiDashengLmGenTextTokenizer(const MiDashengLmGenTextTokenizer &) = delete;
    MiDashengLmGenTextTokenizer & operator=(const MiDashengLmGenTextTokenizer &) = delete;

    std::string build_prompt(std::string_view text) const;
    MiDashengLmGenPromptBatch encode_batch(const std::vector<std::string> & texts) const;

private:
    class Impl;
    std::shared_ptr<const Impl> impl_;
};

}  // namespace engine::models::midashenglm_gen
