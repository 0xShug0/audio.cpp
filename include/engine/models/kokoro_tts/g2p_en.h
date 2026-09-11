#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace kokoro_ggml::g2p_en {

struct TokenContext {
    std::optional<bool> future_vowel;
    bool future_to = false;
};

struct LexiconResult {
    std::string phonemes;
    int rating = 0;
};

struct LexiconEntry {
    bool has_default = false;
    std::string default_phonemes;
    std::unordered_map<std::string, std::optional<std::string>> by_tag;
};

struct TokenMeta {
    bool is_head = true;
    bool prespace = false;
    std::optional<float> stress;
    std::optional<std::string> alias;
    std::optional<std::string> currency;
    std::string num_flags;
    std::optional<int> rating;
};

struct MToken {
    std::string text;
    std::string tag;
    std::string whitespace;
    std::optional<std::string> phonemes;
    TokenMeta meta;
};

using FeatureValue = std::variant<int64_t, float, std::string>;

struct PreprocessResult {
    std::string text;
    std::unordered_map<size_t, FeatureValue> features;
};

class Lexicon {
public:
    explicit Lexicon(bool british = false);
    Lexicon(std::filesystem::path gold_tsv, std::filesystem::path silver_tsv, bool british = false);

    std::optional<LexiconResult> lookup(
        const std::string & word,
        const std::optional<std::string> & tag,
        const std::optional<float> & stress,
        const TokenContext & ctx) const;

    std::optional<LexiconResult> operator()(const MToken & token, const TokenContext & ctx) const;
    static std::string apply_stress(const std::string & phonemes, const std::optional<float> & stress);
    static bool is_number_token(const std::string & word, bool is_head);

private:
    std::optional<LexiconResult> get_special_case(
        const std::string & word,
        const std::optional<std::string> & tag,
        const std::optional<float> & stress,
        const TokenContext & ctx) const;

    std::optional<LexiconResult> get_nnp(const std::string & word) const;
    std::optional<LexiconResult> lookup_raw(
        const std::string & word,
        const std::optional<std::string> & tag,
        const std::optional<float> & stress) const;

    std::optional<LexiconResult> stem_s(
        const std::string & word,
        const std::optional<std::string> & tag,
        const std::optional<float> & stress,
        const TokenContext & ctx) const;
    std::optional<LexiconResult> stem_ed(
        const std::string & word,
        const std::optional<std::string> & tag,
        const std::optional<float> & stress,
        const TokenContext & ctx) const;
    std::optional<LexiconResult> stem_ing(
        const std::string & word,
        const std::optional<std::string> & tag,
        const std::optional<float> & stress,
        const TokenContext & ctx) const;

    std::optional<LexiconResult> get_word(
        const std::string & word,
        const std::optional<std::string> & tag,
        const std::optional<float> & stress,
        const TokenContext & ctx) const;

    std::optional<LexiconResult> get_number(
        const std::string & word,
        const std::optional<std::string> & currency,
        bool is_head) const;

    bool is_known(const std::string & word) const;

    static std::optional<std::string> parent_tag(const std::optional<std::string> & tag);
    static bool is_currency_shape(const std::string & word);

    std::optional<std::string> s_suffix(const std::string & stem) const;
    std::optional<std::string> ed_suffix(const std::string & stem) const;
    std::optional<std::string> ing_suffix(const std::string & stem) const;

    void load_tsv(
        const std::filesystem::path & path,
        std::unordered_map<std::string, LexiconEntry> & target);

    bool british_ = false;
    std::unordered_map<std::string, LexiconEntry> golds_;
    std::unordered_map<std::string, LexiconEntry> silvers_;
};

class EnglishG2P {
public:
    explicit EnglishG2P(bool british = false);
    EnglishG2P(std::filesystem::path lexicon_dir, bool british = false);

    static PreprocessResult preprocess(const std::string & text);
    std::vector<MToken> tokenize(
        const std::string & text,
        const std::unordered_map<size_t, FeatureValue> & features) const;

    std::pair<std::string, std::vector<MToken>> operator()(const std::string & text, bool enable_preprocess = true) const;

private:
    static std::vector<MToken> fold_left(const std::vector<MToken> & tokens);
    static std::vector<std::variant<MToken, std::vector<MToken>>> retokenize(const std::vector<MToken> & tokens);
    static void resolve_tokens(std::vector<MToken> & tokens);
    static TokenContext token_context(const TokenContext & ctx, const std::optional<std::string> & phonemes, const MToken & token);
    static std::string tokens_to_phonemes(const std::vector<MToken> & tokens);
    static MToken merge_token_pair(
        const MToken & left,
        const MToken & right,
        const std::optional<std::string> & unk);
    static MToken merge_tokens(
        const std::vector<MToken> & tokens,
        size_t begin,
        size_t end,
        const std::optional<std::string> & unk);

    Lexicon lexicon_;
};

}  // namespace kokoro_ggml::g2p_en
