#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::xtts_v2 {

class XttsV2Tokenizer {
public:
    explicit XttsV2Tokenizer(const std::filesystem::path & path);
    ~XttsV2Tokenizer();
    XttsV2Tokenizer(XttsV2Tokenizer &&) noexcept;
    XttsV2Tokenizer & operator=(XttsV2Tokenizer &&) noexcept;

    std::vector<int32_t> encode(const std::string & text, const std::string & language) const;
    int32_t start_token() const noexcept;
    int32_t stop_token() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::xtts_v2
