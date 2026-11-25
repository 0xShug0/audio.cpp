#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace engine::models::kale_tts {

// The byte-level BPE of Ettin-encoder-17m (its tokenizer.json): NFC, the GPT-2 pre-tokenizer, its added tokens matched
// first, [CLS] ... [SEP], and the code point offsets of every token as Hugging Face tokenizers reports them.
class EttinTokenizer {
public:
    explicit EttinTokenizer(const std::string & tokenizer_json);
    struct Encoded {
        std::vector<int32_t> ids;
        std::vector<std::pair<int64_t, int64_t>> offsets;   // [start, end) in code points; [0, 0] for [CLS] and [SEP]
    };
    Encoded encode(const std::string & text) const;

private:
    std::vector<std::string> bpe(const std::string & word) const;
    std::unordered_map<std::string, int32_t> vocab_;
    std::unordered_map<std::string, int32_t> ranks_;
    std::vector<std::pair<std::u32string, int32_t>> added_;  // longest first
    int32_t cls_ = 0;
    int32_t sep_ = 0;
};

// One sentence's inputs for the Ettin text side: sub-tokens of "previous sentence + sentence", and for every word of the
// sentence the average over its sub-tokens (wavg, words x tokens, row-major).
struct EttinInputs {
    std::vector<int32_t> ids;
    std::vector<float> wavg;
    int64_t words = 0;
    int64_t tokens = 0;
};
EttinInputs ettin_inputs(const EttinTokenizer & tokenizer, const std::string & text, const std::string & previous);

// The word of every phoneme id (no pads): exact when the per-word ids joined with spaces are the sentence's ids, else
// spread by the spaces in the sentence.
std::vector<int32_t> word_index(const std::vector<int32_t> & sentence_ids, const std::vector<std::vector<int32_t>> & word_ids,
                                int64_t words, int32_t space_id);

}  // namespace engine::models::kale_tts
