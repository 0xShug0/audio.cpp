#include "engine/models/voxtral_realtime/tokenizer_text.h"

#include "engine/framework/io/json.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace engine::models::voxtral_realtime {
namespace {

namespace json = engine::io::json;
constexpr int64_t kOfflineStreamingBufferTokens = 10;
constexpr int32_t kBosToken = 1;
constexpr int32_t kStreamingPadToken = 32;

std::string decode_base64(const std::string & input) {
    static const std::array<int8_t, 256> table = [] {
        std::array<int8_t, 256> values{};
        values.fill(-1);
        for (int i = 0; i < 26; ++i) {
            values[static_cast<size_t>('A' + i)] = static_cast<int8_t>(i);
            values[static_cast<size_t>('a' + i)] = static_cast<int8_t>(26 + i);
        }
        for (int i = 0; i < 10; ++i) {
            values[static_cast<size_t>('0' + i)] = static_cast<int8_t>(52 + i);
        }
        values[static_cast<size_t>('+')] = 62;
        values[static_cast<size_t>('/')] = 63;
        return values;
    }();

    std::string out;
    int bits = 0;
    int value = 0;
    for (const unsigned char ch : input) {
        if (ch == '=') {
            break;
        }
        const int8_t digit = table[ch];
        if (digit < 0) {
            throw std::runtime_error("VoxTral Tekken tokenizer contains invalid base64 token bytes");
        }
        value = (value << 6) | digit;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(static_cast<char>((value >> bits) & 0xff));
        }
    }
    return out;
}

int64_t ceil_div(int64_t value, int64_t divisor) {
    if (divisor <= 0) {
        throw std::runtime_error("VoxTral tokenizer divisor must be positive");
    }
    return (value + divisor - 1) / divisor;
}

int64_t audio_tokens_for_samples(const VoxtralRealtimeConfig & config, int64_t samples) {
    int64_t frames = 0;
    if (samples % config.frontend.hop_length != 0) {
        frames = static_cast<int64_t>(std::ceil(static_cast<double>(samples) /
                                                static_cast<double>(config.frontend.hop_length) -
                                                1.0));
    } else {
        frames = samples / config.frontend.hop_length;
    }
    return ceil_div(frames, config.audio_length_per_tok);
}

}  // namespace

struct VoxtralRealtimeTokenizer::Impl {
    explicit Impl(std::shared_ptr<const VoxtralRealtimeAssets> assets)
        : assets(std::move(assets)) {
        if (this->assets == nullptr) {
            throw std::runtime_error("VoxTral tokenizer requires assets");
        }
        const auto root = this->assets->resources.parse_json("tekken");
        const auto & config_json = root.require("config");
        const int64_t default_vocab_size = json::require_i64(config_json, "default_vocab_size");
        const int64_t num_special_tokens = json::require_i64(config_json, "default_num_special_tokens");
        if (default_vocab_size != this->assets->config.text.vocab_size) {
            throw std::runtime_error("VoxTral Tekken vocab size does not match text config");
        }
        if (num_special_tokens <= 0) {
            throw std::runtime_error("VoxTral Tekken tokenizer requires special tokens");
        }
        const int64_t mergeable_limit = default_vocab_size - num_special_tokens;
        const auto & vocab_json = root.require("vocab").as_array();
        if (static_cast<int64_t>(vocab_json.size()) < mergeable_limit) {
            throw std::runtime_error("VoxTral Tekken tokenizer vocab is shorter than configured mergeable size");
        }
        for (int64_t rank = 0; rank < mergeable_limit; ++rank) {
            const auto & item = vocab_json[static_cast<size_t>(rank)];
            const int64_t actual_rank = json::require_i64(item, "rank");
            if (actual_rank != rank) {
                throw std::runtime_error("VoxTral Tekken tokenizer vocab ranks must be contiguous");
            }
            const std::string token = decode_base64(json::require_string(item, "token_bytes"));
            const int32_t token_id = static_cast<int32_t>(rank + num_special_tokens);
            id_to_bytes.emplace(token_id, token);
        }
        const auto & specials = root.require("special_tokens").as_array();
        for (const auto & item : specials) {
            const int32_t rank = static_cast<int32_t>(json::require_i64(item, "rank"));
            const std::string text = json::require_string(item, "token_str");
            special_ids.emplace(rank);
            special_token_to_id[text] = rank;
        }
        for (int64_t id = static_cast<int64_t>(specials.size()); id < num_special_tokens; ++id) {
            const std::string text = "<SPECIAL_" + std::to_string(id) + ">";
            const int32_t token_id = static_cast<int32_t>(id);
            special_ids.emplace(token_id);
            special_token_to_id[text] = token_id;
        }
        if (special_token_to_id.at("<s>") != kBosToken || special_token_to_id.at("[STREAMING_PAD]") != kStreamingPadToken) {
            throw std::runtime_error("VoxTral Tekken special token ids do not match the realtime prompt contract");
        }
    }

    std::shared_ptr<const VoxtralRealtimeAssets> assets;
    std::unordered_map<int32_t, std::string> id_to_bytes;
    std::unordered_map<std::string, int32_t> special_token_to_id;
    std::unordered_set<int32_t> special_ids;
};

VoxtralRealtimeTokenizer::VoxtralRealtimeTokenizer(std::shared_ptr<const VoxtralRealtimeAssets> assets)
    : impl_(std::make_shared<Impl>(std::move(assets))) {}

VoxtralRealtimeTokenizer::~VoxtralRealtimeTokenizer() = default;

VoxtralRealtimePrompt VoxtralRealtimeTokenizer::build_transcription_prompt(int64_t audio_samples, bool streaming) const {
    (void) streaming;
    const auto & config = impl_->assets->config;
    const int64_t raw_audio_length_per_tok =
        static_cast<int64_t>(static_cast<double>(config.frontend.sample_rate) / 12.5);
    const int64_t delay_samples = static_cast<int64_t>(0.480 * static_cast<double>(config.frontend.sample_rate));
    const int64_t delay_tokens = audio_tokens_for_samples(config, delay_samples);
    const int64_t left_pad_tokens = 32;
    const int64_t prompt_pad_tokens = left_pad_tokens + delay_tokens;
    if (raw_audio_length_per_tok != config.audio_length_per_tok * config.frontend.hop_length) {
        throw std::runtime_error("VoxTral streaming audio/token ratio is inconsistent");
    }
    VoxtralRealtimePrompt prompt;
    prompt.input_ids.reserve(static_cast<size_t>(prompt_pad_tokens + 1));
    prompt.input_ids.push_back(kBosToken);
    for (int64_t i = 0; i < prompt_pad_tokens; ++i) {
        prompt.input_ids.push_back(kStreamingPadToken);
    }
    const int64_t right_base = (raw_audio_length_per_tok - (audio_samples % raw_audio_length_per_tok)) %
        raw_audio_length_per_tok;
    const int64_t right_extra =
        raw_audio_length_per_tok * (delay_tokens + 1 + kOfflineStreamingBufferTokens);
    const int64_t padded_samples = raw_audio_length_per_tok * left_pad_tokens + audio_samples +
        right_base + right_extra;
    prompt.audio_tokens = audio_tokens_for_samples(config, padded_samples) / config.downsample_factor;
    prompt.num_delay_tokens = delay_tokens;
    return prompt;
}

std::string VoxtralRealtimeTokenizer::decode(const std::vector<int32_t> & token_ids) const {
    std::string out;
    for (const int32_t token : token_ids) {
        if (impl_->special_ids.find(token) != impl_->special_ids.end()) {
            continue;
        }
        const auto it = impl_->id_to_bytes.find(token);
        if (it == impl_->id_to_bytes.end()) {
            throw std::runtime_error("VoxTral Tekken decode saw an unknown token id");
        }
        out += it->second;
    }
    return out;
}

bool VoxtralRealtimeTokenizer::is_stream_text_token(int32_t token_id) const {
    if (impl_->special_ids.find(token_id) != impl_->special_ids.end()) {
        return false;
    }
    const auto it = impl_->id_to_bytes.find(token_id);
    return it != impl_->id_to_bytes.end() && !it->second.empty();
}

}  // namespace engine::models::voxtral_realtime
