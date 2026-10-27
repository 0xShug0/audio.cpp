#include "engine/community_models/lfm3_audio/tokenizer.h"

#include <stdexcept>

namespace engine::community_models::lfm3_audio {

namespace {

int32_t require_token(const tokenizers::LlamaBpeTokenizer & bpe, const std::string & token) {
    const auto id = bpe.find_token_id(token);
    if (!id.has_value()) {
        throw std::runtime_error("LFM3-Audio tokenizer has no token " + token);
    }

    return *id;
}

}  // namespace

// tokenizer.json sets ignore_merges, and some 2459 vocab strings tokenize differently without it.
Lfm3Tokenizer::Lfm3Tokenizer(const std::filesystem::path & tokenizer_json)
    : Lfm3Tokenizer(tokenizers::LlamaBpeTokenizerSpec{{}, {}, {}, tokenizer_json, tokenizers::LlamaBpePreTokenizer::Llama3, {}, {}, true}) {}

// The GGUF does not store ignore_merges, so it is set as the checkpoint's tokenizer.json sets it. llama.cpp's
// converter also marks <|tool_call_start|> and <|tool_call_end|> control, where tokenizer.json has them not
// special, so a skip-special decode keeps them; they go back to user-defined (gguf-py TokenType 4).
Lfm3Tokenizer::Lfm3Tokenizer(const tokenizers::LlamaBpeGgufVocabulary & vocabulary)
    : Lfm3Tokenizer([&] {
          tokenizers::LlamaBpeTokenizerSpec spec{{}, {}, {}, std::nullopt, tokenizers::LlamaBpePreTokenizer::Llama3};
          spec.gguf_vocabulary = vocabulary;
          spec.gguf_vocabulary->ignore_merges = true;
          auto & gguf = *spec.gguf_vocabulary;
          for (size_t id = 0; id < gguf.tokens.size() && id < gguf.token_types.size(); ++id) {
              if (gguf.tokens[id] == "<|tool_call_start|>" || gguf.tokens[id] == "<|tool_call_end|>") {
                  gguf.token_types[id] = 4;
              }
          }
          return spec;
      }()) {}

Lfm3Tokenizer::Lfm3Tokenizer(tokenizers::LlamaBpeTokenizerSpec spec) : ids_{} {
    bpe_ = tokenizers::load_llama_bpe_tokenizer(spec);
    const auto & bpe = *bpe_;
    const auto newline = bpe.encode("\n", false);
    if (newline.size() != 1) {
        throw std::runtime_error("LFM3-Audio tokenizer has no single token for a newline");
    }

    ids_.bos = require_token(bpe, "<|startoftext|>");
    ids_.im_start = require_token(bpe, "<|im_start|>");
    ids_.im_end = require_token(bpe, "<|im_end|>");
    ids_.nl = newline.front();
    ids_.text_end = require_token(bpe, "<|text_end|>");
    ids_.audio_start = require_token(bpe, "<|audio_start|>");
    ids_.mixed_start = require_token(bpe, "<|mixed_start|>");
    ids_.audio = require_token(bpe, "<|audio|>");
    ids_.audio_out = require_token(bpe, "<|audio_out|>");
    ids_.think = require_token(bpe, "<think>");
    ids_.think_end = require_token(bpe, "</think>");
}

std::vector<int32_t> Lfm3Tokenizer::encode(const std::string & text) const {
    return bpe_->encode(text, true);
}

std::string Lfm3Tokenizer::decode(const std::vector<int32_t> & ids, bool skip_special) const {
    if (!skip_special) {
        return bpe_->decode(ids, false);
    }

    std::vector<int32_t> kept;
    kept.reserve(ids.size());
    for (const int32_t id : ids) {
        if (id != ids_.audio_out && !bpe_->is_control_token_id(id)) {
            kept.push_back(id);
        }
    }

    return bpe_->decode(kept, false);
}

const Lfm3TokenIds & Lfm3Tokenizer::ids() const noexcept {
    return ids_;
}

}  // namespace engine::community_models::lfm3_audio
