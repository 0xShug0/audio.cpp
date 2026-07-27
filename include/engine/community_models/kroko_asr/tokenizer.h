#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace engine::models::kroko_asr {

class KrokoTokenizer {
public:
    KrokoTokenizer(std::vector<std::string> pieces, int32_t blank_id, int32_t unk_id);

    std::string decode(const std::vector<int32_t> & ids) const;
    std::vector<int32_t> encode_hotword(const std::string & text) const;
    const std::string & piece(int32_t id) const;
    int32_t blank_id() const noexcept;
    int32_t vocab_size() const noexcept;

private:
    std::vector<std::string> pieces_;
    int32_t blank_id_ = 0;
    int32_t unk_id_ = 2;
};

}  // namespace engine::models::kroko_asr
