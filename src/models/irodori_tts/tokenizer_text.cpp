#include "engine/models/irodori_tts/tokenizer_text.h"

#include "engine/framework/io/json.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::irodori_tts {
namespace {

constexpr const char * kSentencePieceSpace = "\xE2\x96\x81";

std::shared_ptr<const IrodoriTTSAssets> require_assets(std::shared_ptr<const IrodoriTTSAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("Irodori-TTS text tokenizer requires assets");
    }
    return assets;
}

std::string normalize_text_for_llm_jp(std::string text) {
    text.insert(0, kSentencePieceSpace);
    size_t pos = 0;
    while ((pos = text.find(' ', pos)) != std::string::npos) {
        text.replace(pos, 1, kSentencePieceSpace);
        pos += 3;
    }
    return text;
}

std::vector<std::string> utf8_chars(const std::string & text) {
    std::vector<std::string> chars;
    for (size_t i = 0; i < text.size();) {
        const unsigned char ch = static_cast<unsigned char>(text[i]);
        size_t width = 1;
        if ((ch & 0x80U) == 0) {
            width = 1;
        } else if ((ch & 0xE0U) == 0xC0U) {
            width = 2;
        } else if ((ch & 0xF0U) == 0xE0U) {
            width = 3;
        } else if ((ch & 0xF8U) == 0xF0U) {
            width = 4;
        } else {
            throw std::runtime_error("Irodori-TTS tokenizer received invalid UTF-8");
        }
        if (i + width > text.size()) {
            throw std::runtime_error("Irodori-TTS tokenizer received truncated UTF-8");
        }
        chars.push_back(text.substr(i, width));
        i += width;
    }
    return chars;
}

std::string byte_token(unsigned char value) {
    constexpr char kHex[] = "0123456789ABCDEF";
    std::string out = "<0x";
    out.push_back(kHex[(value >> 4U) & 0x0FU]);
    out.push_back(kHex[value & 0x0FU]);
    out.push_back('>');
    return out;
}

}  // namespace

struct IrodoriTextTokenizer::Impl {
    explicit Impl(std::shared_ptr<const IrodoriTTSAssets> input_assets)
        : assets(std::move(input_assets)) {
        const auto root = assets->resources.parse_json("tokenizer_json");
        const auto & model = root.require("model");
        if (engine::io::json::require_string(model, "type") != "Unigram") {
            throw std::runtime_error("Irodori-TTS tokenizer expects HF Unigram tokenizer.json");
        }
        unk = static_cast<int32_t>(engine::io::json::optional_i64(model, "unk_id", 0));
        const auto & vocab_json = model.require("vocab").as_array();
        pieces.reserve(vocab_json.size());
        scores.reserve(vocab_json.size());
        int32_t id = 0;
        for (const auto & item : vocab_json) {
            const auto & pair = item.as_array();
            if (pair.size() != 2 || !pair[0].is_string() || !pair[1].is_number()) {
                throw std::runtime_error("Irodori-TTS tokenizer vocab entry must be [piece, score]");
            }
            pieces.push_back(pair[0].as_string());
            scores.push_back(pair[1].as_f32());
            token_to_id.emplace(pieces.back(), id);
            ++id;
        }
        auto bos_it = token_to_id.find("<s>");
        auto pad_it = token_to_id.find("<PAD|LLM-jp>");
        if (bos_it == token_to_id.end() || pad_it == token_to_id.end()) {
            throw std::runtime_error("Irodori-TTS tokenizer missing BOS or PAD token");
        }
        bos = bos_it->second;
        pad = pad_it->second;
        if (assets->config.text_vocab_size != static_cast<int64_t>(pieces.size())) {
            throw std::runtime_error("Irodori-TTS text_vocab_size does not match tokenizer vocab size");
        }
    }

    std::vector<int32_t> encode_piece_sequence(const std::string & normalized) const {
        const auto chars = utf8_chars(normalized);
        const size_t n = chars.size();
        std::vector<std::string> suffix(n + 1);
        for (size_t i = n; i-- > 0;) {
            suffix[i] = chars[i] + suffix[i + 1];
        }

        constexpr float kNegInf = -std::numeric_limits<float>::infinity();
        std::vector<float> best(n + 1, kNegInf);
        std::vector<size_t> next(n + 1, 0);
        std::vector<int32_t> best_id(n + 1, unk);
        best[n] = 0.0F;
        for (size_t i = n; i-- > 0;) {
            auto byte_fallback = [&]() {
                float score = 0.0F;
                std::vector<int32_t> ids;
                for (const unsigned char byte : chars[i]) {
                    const auto it = token_to_id.find(byte_token(byte));
                    if (it == token_to_id.end()) {
                        ids.push_back(unk);
                    } else {
                        ids.push_back(it->second);
                    }
                    score += scores[static_cast<size_t>(ids.back())];
                }
                return std::pair<float, std::vector<int32_t>>(score, std::move(ids));
            };

            for (size_t j = i + 1; j <= n; ++j) {
                const size_t byte_len = suffix[i].size() - suffix[j].size();
                const std::string piece = suffix[i].substr(0, byte_len);
                const auto it = token_to_id.find(piece);
                if (it == token_to_id.end()) {
                    continue;
                }
                const float candidate = scores[static_cast<size_t>(it->second)] + best[j];
                if (candidate > best[i]) {
                    best[i] = candidate;
                    next[i] = j;
                    best_id[i] = it->second;
                }
            }
            if (best[i] == kNegInf) {
                const auto fallback = byte_fallback();
                best[i] = fallback.first + best[i + 1];
                next[i] = i + 1;
                best_id[i] = fallback.second.empty() ? unk : fallback.second.front();
            }
        }

        std::vector<int32_t> ids;
        for (size_t i = 0; i < n;) {
            if (next[i] == i + 1 && token_to_id.find(chars[i]) == token_to_id.end()) {
                for (const unsigned char byte : chars[i]) {
                    const auto it = token_to_id.find(byte_token(byte));
                    ids.push_back(it == token_to_id.end() ? unk : it->second);
                }
                ++i;
                continue;
            }
            ids.push_back(best_id[i]);
            i = next[i];
        }
        return ids;
    }

    std::shared_ptr<const IrodoriTTSAssets> assets;
    std::vector<std::string> pieces;
    std::vector<float> scores;
    std::unordered_map<std::string, int32_t> token_to_id;
    int32_t bos = 1;
    int32_t pad = 4;
    int32_t unk = 0;
};

IrodoriTextTokenizer::IrodoriTextTokenizer(std::shared_ptr<const IrodoriTTSAssets> assets)
    : impl_(std::make_shared<Impl>(require_assets(std::move(assets)))) {}

std::vector<int32_t> IrodoriTextTokenizer::encode(const std::string & text) const {
    auto ids = impl_->encode_piece_sequence(normalize_text_for_llm_jp(text));
    if (impl_->assets->config.text_add_bos) {
        ids.insert(ids.begin(), impl_->bos);
    }
    return ids;
}

IrodoriTokenizedText IrodoriTextTokenizer::encode_padded(const std::string & text, int64_t max_length) const {
    if (max_length <= 0) {
        throw std::runtime_error("Irodori-TTS tokenizer max_length must be positive");
    }
    auto ids = encode(text);
    if (ids.size() > static_cast<size_t>(max_length)) {
        ids.resize(static_cast<size_t>(max_length));
    }
    IrodoriTokenizedText out;
    out.token_ids = ids;
    out.mask.assign(out.token_ids.size(), 1);
    while (out.token_ids.size() < static_cast<size_t>(max_length)) {
        out.token_ids.push_back(impl_->pad);
        out.mask.push_back(0);
    }
    return out;
}

int32_t IrodoriTextTokenizer::bos_id() const noexcept {
    return impl_->bos;
}

int32_t IrodoriTextTokenizer::pad_id() const noexcept {
    return impl_->pad;
}

int32_t IrodoriTextTokenizer::unk_id() const noexcept {
    return impl_->unk;
}

int64_t IrodoriTextTokenizer::vocab_size() const noexcept {
    return static_cast<int64_t>(impl_->pieces.size());
}

}  // namespace engine::models::irodori_tts
