#pragma once

#include "engine/models/miotts/assets.h"
#include "engine/models/miotts/types.h"

#include <memory>
#include <string>
#include <vector>

namespace engine::tokenizers {
class LlamaBpeTokenizer;
}

namespace engine::models::miotts {

class MioTTSTokenizer {
public:
    explicit MioTTSTokenizer(std::shared_ptr<const MioTTSAssets> assets);

    MioTTSPrompt build_prompt(const std::string & text) const;
    std::string decode(const std::vector<int32_t> & token_ids, bool skip_special_tokens = false) const;

private:
    std::shared_ptr<const MioTTSAssets> assets_;
    std::shared_ptr<engine::tokenizers::LlamaBpeTokenizer> tokenizer_;
    int32_t im_start_token_id_ = 0;
    int32_t im_end_token_id_ = 0;
};

}  // namespace engine::models::miotts
