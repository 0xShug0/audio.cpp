#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/tokenizers/llama_bpe.h"

namespace engine::models::plapre {

class PlapreTextFrontend {
public:
    explicit PlapreTextFrontend(const assets::ResourceBundle & resources);
    std::string normalize(const std::string & text, bool preserve_line_breaks = false) const;
    std::vector<int32_t> encode(const std::string & text, bool target) const;
    const tokenizers::LlamaBpeTokenizer & tokenizer() const noexcept;

private:
    tokenizers::LlamaBpeTokenizer tokenizer_;
};

}  // namespace engine::models::plapre
