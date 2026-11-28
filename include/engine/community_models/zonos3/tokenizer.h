#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine::tokenizers {
class LlamaBpeTokenizer;
}

namespace engine::models::zonos3 {

// ZONOS3's SuperBPE text tokenizer: the Split pre-tokenizer
// \p{N}{1,3}| ?[^\s\p{L}\p{N}]{2,}[\r\n/]*| +(?!\S) (Isolated), GPT-2 byte-level symbols and rank-ordered merges.
// Model ids are tokenizer ids + 4, framed by BOS 2 and EOS 3.
class Zonos3Tokenizer {
public:
    explicit Zonos3Tokenizer(const std::string & tokenizer_json);
    ~Zonos3Tokenizer();

    // [2] + (bpe ids + 4) + [3]. Rejects the U+E000 / U+E001 pronunciation groups (no phoneme rows).
    std::vector<int32_t> text_ids(const std::string & utf8) const;
    // The pre-tokenizer's pieces of one fragment, as raw UTF-8.
    static std::vector<std::string> split(const std::string & utf8);

private:
    std::shared_ptr<tokenizers::LlamaBpeTokenizer> bpe_;
};

}  // namespace engine::models::zonos3
