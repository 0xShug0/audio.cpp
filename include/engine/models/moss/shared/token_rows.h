#pragma once

#include <cstdint>
#include <vector>

namespace engine::models::moss {

struct TokenRows {
    std::vector<int32_t> text_tokens;
    std::vector<int32_t> audio_codes;
};

class TokenRowBuilder {
public:
    TokenRowBuilder(int64_t num_codebooks, int32_t audio_pad_token_id);

    void push_text_token(int32_t token_id);
    void push_text_tokens(const std::vector<int32_t> & token_ids);
    void push_audio_row(int32_t text_slot_token_id, const std::vector<std::vector<int32_t>> & codes, int64_t frame);
    TokenRows finish();

private:
    int64_t num_codebooks_ = 0;
    int32_t audio_pad_token_id_ = 0;
    TokenRows rows_;
};

}  // namespace engine::models::moss
