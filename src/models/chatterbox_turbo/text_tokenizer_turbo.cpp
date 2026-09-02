#include "engine/models/chatterbox_turbo/text_tokenizer_turbo.h"

#include <ggml.h>
#include <gguf.h>

#include <cctype>
#include <fstream>
#include <stdexcept>
#include <unordered_set>

namespace engine::models::chatterbox_turbo {

namespace {

struct GgufContextDeleter {
    void operator()(gguf_context * ctx) const noexcept {
        if (ctx != nullptr) {
            gguf_free(ctx);
        }
    }
};

std::vector<std::string> read_gguf_string_array(gguf_context * ctx, const char * key) {
    const int64_t key_id = gguf_find_key(ctx, key);
    if (key_id < 0) {
        throw std::runtime_error(std::string("Chatterbox Turbo GGUF is missing metadata key: ") + key);
    }
    if (gguf_get_arr_type(ctx, key_id) != GGUF_TYPE_STRING) {
        throw std::runtime_error(std::string("Chatterbox Turbo GGUF metadata key is not a string array: ") + key);
    }
    const size_t count = gguf_get_arr_n(ctx, key_id);
    std::vector<std::string> out;
    out.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        out.emplace_back(gguf_get_arr_str(ctx, key_id, i));
    }
    return out;
}

std::string json_escape(const std::string & value) {
    std::string out;
    out.reserve(value.size() + 8);
    for (unsigned char c : value) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += static_cast<char>(c);
                }
        }
    }
    return out;
}

bool looks_like_bracket_tag(const std::string & token) {
    return token.size() >= 3 && token.front() == '[' && token.back() == ']';
}

}  // namespace

std::shared_ptr<engine::tokenizers::LlamaBpeTokenizer> load_chatterbox_turbo_tokenizer(
    const std::filesystem::path & t3_gguf_path,
    const std::filesystem::path & scratch_dir) {
    gguf_init_params params{};
    params.no_alloc = true;
    params.ctx = nullptr;
    std::unique_ptr<gguf_context, GgufContextDeleter> ctx(gguf_init_from_file(t3_gguf_path.string().c_str(), params));
    if (!ctx) {
        throw std::runtime_error("failed to open Chatterbox Turbo T3 GGUF for tokenizer metadata: " + t3_gguf_path.string());
    }

    const auto tokens = read_gguf_string_array(ctx.get(), "tokenizer.ggml.tokens");
    const auto merges = read_gguf_string_array(ctx.get(), "tokenizer.ggml.merges");
    if (tokens.empty()) {
        throw std::runtime_error("Chatterbox Turbo T3 GGUF has an empty tokenizer.ggml.tokens array");
    }

    std::filesystem::create_directories(scratch_dir);
    const auto vocab_path = scratch_dir / "chatterbox_turbo_vocab.json";
    const auto merges_path = scratch_dir / "chatterbox_turbo_merges.txt";

    // The 19 emotion/style control tags ([laugh], [sigh], ...) sit at the tail of the vocab and
    // cannot be reached through byte-pair merges (they were never trained into the merge table),
    // so they must be registered as atomic special tokens the pre-tokenizer matches directly
    // (LlamaBpeTokenizerSpec::additional_special_tokens) rather than as plain vocab entries.
    // Crucially they must NOT also appear in vocab.json below: add_runtime_special_tokens only
    // sets a token's "special" attribute when it is inserting a genuinely new id -- if the id
    // already exists (which it would, from vocab.json) it just verifies the id and continues,
    // silently leaving the token without its special/atomic-match flag, so it gets shredded into
    // ordinary byte-level BPE pieces instead of matched as one unit.
    std::vector<engine::tokenizers::LlamaBpeAddedToken> special_tokens;
    int64_t special_tokens_start = static_cast<int64_t>(tokens.size());
    for (int64_t id = static_cast<int64_t>(tokens.size()) - 1; id >= 0; --id) {
        if (!looks_like_bracket_tag(tokens[static_cast<size_t>(id)])) {
            break;
        }
        special_tokens.emplace_back(tokens[static_cast<size_t>(id)], static_cast<int32_t>(id));
        special_tokens_start = id;
    }

    {
        std::ofstream out(vocab_path, std::ios::trunc);
        if (!out) {
            throw std::runtime_error("failed to write Chatterbox Turbo tokenizer vocab: " + vocab_path.string());
        }
        out << "{";
        bool first = true;
        for (size_t i = 0; i < tokens.size(); ++i) {
            if (static_cast<int64_t>(i) >= special_tokens_start) {
                continue;
            }
            if (!first) {
                out << ",";
            }
            first = false;
            out << "\"" << json_escape(tokens[i]) << "\":" << i;
        }
        out << "}";
    }
    {
        std::ofstream out(merges_path, std::ios::trunc);
        if (!out) {
            throw std::runtime_error("failed to write Chatterbox Turbo tokenizer merges: " + merges_path.string());
        }
        for (const auto & merge : merges) {
            out << merge << "\n";
        }
    }

    engine::tokenizers::LlamaBpeTokenizerSpec spec(
        vocab_path,
        merges_path,
        /*tokenizer_config_path=*/{},
        /*tokenizer_json_path=*/std::nullopt,
        engine::tokenizers::LlamaBpePreTokenizer::Gpt2,
        std::move(special_tokens));
    return engine::tokenizers::load_llama_bpe_tokenizer(spec);
}

std::string chatterbox_turbo_punc_norm(const std::string & text) {
    if (text.empty()) {
        return "You need to add some text for me to talk.";
    }
    std::string normalized = text;
    if (std::islower(static_cast<unsigned char>(normalized.front()))) {
        normalized.front() = static_cast<char>(std::toupper(static_cast<unsigned char>(normalized.front())));
    }

    // Collapse runs of whitespace to single spaces (Python's " ".join(text.split())).
    {
        std::string collapsed;
        collapsed.reserve(normalized.size());
        bool in_space = false;
        for (char c : normalized) {
            const bool is_space = std::isspace(static_cast<unsigned char>(c)) != 0;
            if (is_space) {
                in_space = true;
                continue;
            }
            if (in_space && !collapsed.empty()) {
                collapsed += ' ';
            }
            in_space = false;
            collapsed += c;
        }
        normalized = std::move(collapsed);
    }

    static const std::vector<std::pair<std::string, std::string>> kReplacements = {
        {"\xE2\x80\xA6", ", "},  // U+2026 HORIZONTAL ELLIPSIS
        {":", ","},
        {"\xE2\x80\x94", "-"},  // U+2014 EM DASH
        {"\xE2\x80\x93", "-"},  // U+2013 EN DASH
        {" ,", ","},
        {"\xE2\x80\x9C", "\""},  // U+201C LEFT DOUBLE QUOTATION MARK
        {"\xE2\x80\x9D", "\""},  // U+201D RIGHT DOUBLE QUOTATION MARK
        {"\xE2\x80\x98", "'"},   // U+2018 LEFT SINGLE QUOTATION MARK
        {"\xE2\x80\x99", "'"},   // U+2019 RIGHT SINGLE QUOTATION MARK
    };
    for (const auto & [from, to] : kReplacements) {
        size_t pos = 0;
        while ((pos = normalized.find(from, pos)) != std::string::npos) {
            normalized.replace(pos, from.size(), to);
            pos += to.size();
        }
    }

    while (!normalized.empty() && normalized.back() == ' ') {
        normalized.pop_back();
    }
    static const std::unordered_set<char> kSentenceEnders = {'.', '!', '?', '-', ','};
    if (normalized.empty() || kSentenceEnders.find(normalized.back()) == kSentenceEnders.end()) {
        normalized += '.';
    }
    return normalized;
}

std::vector<int32_t> encode_chatterbox_turbo_text(
    const engine::tokenizers::LlamaBpeTokenizer & tokenizer,
    const std::string & text) {
    return tokenizer.encode(chatterbox_turbo_punc_norm(text), /*parse_special=*/true);
}

}  // namespace engine::models::chatterbox_turbo
