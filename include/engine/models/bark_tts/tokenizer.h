#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace engine::models::bark_tts {

class BarkTokenizer {
public:
    explicit BarkTokenizer(const std::string & tokenizer_json);
    std::vector<int32_t> encode(const std::string & text) const;

private:
    std::unordered_map<std::string, int32_t> vocab_;
    int32_t unknown_ = 100;
};

}  // namespace engine::models::bark_tts
