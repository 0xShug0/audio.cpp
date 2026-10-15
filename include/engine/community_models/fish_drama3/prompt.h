#pragma once

#include "engine/community_models/fish_drama3/assets.h"
#include "engine/community_models/fish_drama3/tokenizer_text.h"

namespace engine::models::fish_drama3 {

struct Drama3Take {
    std::string text;
    codecs::FishDacCodes codes;
};

void validate_drama3_text(const std::string & text, const char * location);
void validate_drama3_codes(const codecs::FishDacCodes & codes, int64_t codebook_size);
std::vector<Drama3Frame> build_drama3_prompt(const Drama3Assets & assets,
    const Drama3TextTokenizer & tokenizer, const std::string & text,
    const std::vector<Drama3Take> & references, const std::vector<Drama3Take> & history);

}  // namespace engine::models::fish_drama3
