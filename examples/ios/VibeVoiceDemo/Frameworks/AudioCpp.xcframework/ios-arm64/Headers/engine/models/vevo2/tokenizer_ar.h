#pragma once

#include "engine/models/vevo2/assets.h"
#include "engine/models/vevo2/types.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace engine::tokenizers {
class LlamaBpeTokenizer;
}

namespace engine::models::vevo2 {

class Vevo2ARTokenizer final {
public:
    explicit Vevo2ARTokenizer(std::shared_ptr<const Vevo2Assets> assets);

    Vevo2TokenizedPrompt tokenize_prompt(const std::string & prompt) const;
    std::string decode(const std::vector<int32_t> & token_ids, bool skip_special_tokens = false) const;
    std::optional<int32_t> find_token_id(const std::string & token) const;
    int32_t eos_token_id() const noexcept;
    int32_t pad_token_id() const noexcept;

private:
    std::shared_ptr<const Vevo2Assets> assets_;
    std::shared_ptr<engine::tokenizers::LlamaBpeTokenizer> tokenizer_;
    int32_t eos_token_id_ = 151645;
    int32_t pad_token_id_ = 151643;
};

Vevo2TokenSequence parse_content_style_tokens(const std::string & generated_text);

}  // namespace engine::models::vevo2
