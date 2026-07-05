#pragma once

#include "engine/models/irodori_tts/assets.h"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace engine::models::irodori_tts {

struct IrodoriTokenizedText {
    std::vector<int32_t> token_ids;
    std::vector<uint8_t> mask;
};

class IrodoriTextTokenizer {
public:
    explicit IrodoriTextTokenizer(std::shared_ptr<const IrodoriAssets> assets);

    IrodoriTokenizedText encode_padded(const std::string & text, int64_t max_length) const;
    std::vector<int32_t> encode(const std::string & text) const;
    int32_t bos_id() const noexcept;
    int32_t pad_id() const noexcept;
    int32_t unk_id() const noexcept;
    int64_t vocab_size() const noexcept;

private:
    struct Impl;
    std::shared_ptr<const Impl> impl_;
};

}  // namespace engine::models::irodori_tts
