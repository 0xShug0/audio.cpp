#include "engine/community_models/voxcpm1/tokenizer_text.h"
#include "engine/community_models/voxcpm1/assets.h"

#include "engine/framework/io/json.h"

#include <algorithm>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace engine::community_models::voxcpm1 {
namespace json = engine::io::json;
namespace {

int32_t require_token_id(
    const std::unordered_map<std::string, int32_t> & vocab,
    const std::string & token) {
    const auto it = vocab.find(token);
    if (it == vocab.end()) {
        throw std::runtime_error("VoxCPM1 tokenizer missing token: " + token);
    }
    return it->second;
}

void load_tokenizer_json(
    const std::filesystem::path & path,
    std::unordered_map<std::string, int32_t> & vocab,
    std::unordered_map<std::string, int32_t> & merge_ranks) {
    const auto root = json::parse_file(path);
    const auto & model = root.require("model");
    if (model.require("type").as_string() != "BPE") {
        throw std::runtime_error("VoxCPM1 tokenizer expects BPE tokenizer.json");
    }
    const auto & vocab_json = model.require("vocab").as_object();
    for (const auto & [token, id_value] : vocab_json) {
        const int32_t id = static_cast<int32_t>(id_value.as_i64());
        vocab.emplace(token, id);
    }
    const auto & merges = model.require("merges").as_array();
    int32_t rank = 0;
    for (const auto & merge_value : merges) {
        const std::string merge = merge_value.as_string();
        const size_t split = merge.find(' ');
        if (split == std::string::npos) {
            throw std::runtime_error("invalid VoxCPM1 tokenizer merge: " + merge);
        }
        merge_ranks.emplace(
            merge.substr(0, split) + '\0' + merge.substr(split + 1), rank);
        ++rank;
    }
}

void load_special_tokens(
    const std::filesystem::path & path,
    std::unordered_map<std::string, int32_t> & vocab,
    std::unordered_map<std::string, int32_t> & special_tokens,
    int32_t & audio_start_token_id,
    int32_t & audio_end_token_id,
    int32_t & reference_audio_start_token_id,
    int32_t & reference_audio_end_token_id,
    int32_t & bos_token_id,
    int32_t & eos_token_id,
    int32_t & unk_token_id) {
    const auto root = json::parse_file(path);
    const auto * added = root.find("added_tokens_decoder");
    if (added == nullptr) {
        throw std::runtime_error("VoxCPM1 tokenizer_config missing added_tokens_decoder");
    }
    for (const auto & [id_text, token_config] : added->as_object()) {
        const auto * content = token_config.find("content");
        if (content == nullptr || !content->is_string()) {
            continue;
        }
        const int32_t id = static_cast<int32_t>(std::stoll(id_text));
        const std::string token = content->as_string();
        vocab[token] = id;
        special_tokens[token] = id;
    }
    audio_start_token_id = require_token_id(vocab, "<|audio_start|>");
    audio_end_token_id = require_token_id(vocab, "<|audio_end|>");
    reference_audio_start_token_id = require_token_id(vocab, "<|audio_prompt_start|>");
    reference_audio_end_token_id = require_token_id(vocab, "<|audio_prompt_end|>");
    bos_token_id = require_token_id(vocab, "<s>");
    eos_token_id = require_token_id(vocab, "</s>");
    unk_token_id = require_token_id(vocab, "<unk>");
}

}  // namespace

VoxCPM1TextTokenizer::VoxCPM1TextTokenizer(std::shared_ptr<const VoxCPM1Assets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("VoxCPM1 text tokenizer requires assets");
    }

    std::unordered_map<std::string, int32_t> vocab;
    std::unordered_map<std::string, int32_t> merge_ranks;
    std::unordered_map<std::string, int32_t> special_tokens;
    int32_t audio_start_token_id = 101;
    int32_t audio_end_token_id = 102;
    int32_t reference_audio_start_token_id = 103;
    int32_t reference_audio_end_token_id = 104;
    int32_t bos_token_id = 1;
    int32_t eos_token_id = 2;
    int32_t unk_token_id = 3;

    load_tokenizer_json(
        assets->resources.require_file("tokenizer_json"),
        vocab,
        merge_ranks);

    load_special_tokens(
        assets->resources.require_file("tokenizer_config"),
        vocab,
        special_tokens,
        audio_start_token_id,
        audio_end_token_id,
        reference_audio_start_token_id,
        reference_audio_end_token_id,
        bos_token_id,
        eos_token_id,
        unk_token_id);

    common_ = std::make_shared<const VoxCPM1TokenizerCommon>(
        std::move(vocab),
        std::move(merge_ranks),
        std::move(special_tokens),
        bos_token_id,
        eos_token_id,
        unk_token_id,
        audio_start_token_id,
        audio_end_token_id,
        reference_audio_start_token_id,
        reference_audio_end_token_id);
}

std::vector<int32_t> VoxCPM1TextTokenizer::encode(const std::string & text) const {
    return common_->encode(text);
}

VoxCPM1TextPrompt VoxCPM1TextTokenizer::build_prompt(const std::string & text) const {
    return common_->build_prompt(text);
}

int32_t VoxCPM1TextTokenizer::audio_start_token_id() const noexcept {
    return common_->audio_start_token_id();
}

int32_t VoxCPM1TextTokenizer::audio_end_token_id() const noexcept {
    return common_->audio_end_token_id();
}

int32_t VoxCPM1TextTokenizer::reference_audio_start_token_id() const noexcept {
    return common_->reference_audio_start_token_id();
}

int32_t VoxCPM1TextTokenizer::reference_audio_end_token_id() const noexcept {
    return common_->reference_audio_end_token_id();
}

int32_t VoxCPM1TextTokenizer::bos_token_id() const noexcept {
    return common_->bos_token_id();
}

int32_t VoxCPM1TextTokenizer::eos_token_id() const noexcept {
    return common_->eos_token_id();
}

int32_t VoxCPM1TextTokenizer::unk_token_id() const noexcept {
    return common_->unk_token_id();
}

}  // namespace engine::community_models::voxcpm1