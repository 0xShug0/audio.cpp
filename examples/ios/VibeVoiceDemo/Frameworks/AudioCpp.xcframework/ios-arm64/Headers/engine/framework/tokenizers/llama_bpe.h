#pragma once

#include "engine/framework/tokenizers/tokenizer.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace engine::tokenizers {

enum class LlamaBpePreTokenizer {
    Llama3,
    Jais2,
    Dbrx,
    Smaug,
    DeepseekLlm,
    Deepseek3Llm,
    HunyuanDense,
    JoyaiLlm,
    Youtu,
    DeepseekCoder,
    Falcon,
    Starcoder,
    Refact,
    CommandR,
    Smollm,
    Codeshell,
    Exaone,
    Minerva,
    Gpt2,
    Mpt,
    Olmo,
    Jais,
    Trillion,
    GraniteDocling,
    StableLm2,
    Qwen2,
    Hunyuan,
    SolarOpen,
    Qwen35,
    Poro,
    Bloom,
    Gpt3Finnish,
    Chatglm4,
    Viking,
    Tekken,
    Chameleon,
    Gpt4o,
    MinimaxM2,
    TinyAya,
    KimiK2,
    SuperBpe,
    BailingMoe,
    SeedCoder,
    Grok2,
    Afmoe,
    ExaoneMoe,
    Gemma4,
    SarvamMoe,
};

struct LlamaBpeTokenizerSpec {
    std::filesystem::path vocab_path;
    std::filesystem::path merges_path;
    std::filesystem::path tokenizer_config_path;
    std::optional<std::filesystem::path> tokenizer_json_path;
    LlamaBpePreTokenizer pre_type = LlamaBpePreTokenizer::Gpt2;
};

class LlamaBpeTokenizer final : public ITokenizer {
public:
    explicit LlamaBpeTokenizer(LlamaBpeTokenizerSpec spec);

    std::string family() const override;
    TokenizedText tokenize(const std::string & text) const override;

    std::vector<int32_t> encode(const std::string & text) const;
    std::vector<int32_t> encode(const std::string & text, bool parse_special) const;
    std::string decode(const std::vector<int32_t> & token_ids, bool skip_special_tokens = false) const;

    std::optional<int32_t> find_token_id(const std::string & token) const;
    bool is_special_token_id(int32_t token_id) const;
    bool is_control_token_id(int32_t token_id) const;
    std::optional<int32_t> configured_pad_token_id() const noexcept;

private:
    struct Impl;
    std::shared_ptr<const Impl> impl_;
};

std::shared_ptr<LlamaBpeTokenizer> load_llama_bpe_tokenizer(const LlamaBpeTokenizerSpec & spec);

}  // namespace engine::tokenizers
