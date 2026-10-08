#include "engine/community_models/whistle_asr/assets.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using engine::community_models::whistle_asr::WhistleToken;

void expect_decode(
    const char * label, const std::vector<WhistleToken> & pieces,
    const std::vector<int32_t> & ids, const std::string & expected) {
    const auto actual = engine::community_models::whistle_asr::decode_whistle_tokens(pieces, ids);
    if (actual != expected) {
        throw std::runtime_error(std::string("Whistle tokenizer failed: ") + label);
    }
}

void expect_throw(const char * label, const std::vector<WhistleToken> & pieces,
                  const std::vector<int32_t> & ids) {
    try {
        (void)engine::community_models::whistle_asr::decode_whistle_tokens(pieces, ids);
    } catch (const std::exception &) {
        return;
    }
    throw std::runtime_error(std::string("Whistle tokenizer accepted: ") + label);
}

}  // namespace

int main() {
    constexpr const char * kReplacement = "\xEF\xBF\xBD";
    expect_decode("valid UTF-8", {
        {"A", 0.0f, 0}, {"\xC2\xA2", 0.0f, 0}, {"\xE2\x82\xAC", 0.0f, 0},
        {"\xF0\x90\x8D\x88", 0.0f, 0},
    }, {0, 1, 2, 3}, "A\xC2\xA2\xE2\x82\xAC\xF0\x90\x8D\x88");
    expect_decode("byte fallback joins before UTF-8 decoding", {
        {"A\xE2\x96\x81" "B", 0.0f, 0}, {"<0xE2>", 0.0f, 4},
        {"<0x96>", 0.0f, 4}, {"<0x81>", 0.0f, 4}, {"C", 0.0f, 0},
    }, {0, 1, 2, 3, 4}, "A B C");
    expect_decode("valid byte fallback UTF-8", {
        {"<0xC2>", 0.0f, 4}, {"<0xA2>", 0.0f, 4}, {"<0xF0>", 0.0f, 4},
        {"<0x90>", 0.0f, 4}, {"<0x8D>", 0.0f, 4}, {"<0x88>", 0.0f, 4},
    }, {0, 1, 2, 3, 4, 5}, "\xC2\xA2\xF0\x90\x8D\x88");
    expect_decode("unknown and control tokens", {
        {"<unk>", 0.0f, 1}, {"<s>", 0.0f, 2}, {"<user>", 0.0f, 3}, {"word", 0.0f, 0},
    }, {0, 1, 2, 3}, "<user>word");
    expect_decode("lone and incomplete bytes", {
        {"<0xFF>", 0.0f, 4}, {"<0xE2>", 0.0f, 4}, {"<0x82>", 0.0f, 4},
    }, {0, 1, 2}, std::string(kReplacement) + kReplacement);
    expect_decode("misplaced continuation", {
        {"<0xE2>", 0.0f, 4}, {"<0x28>", 0.0f, 4}, {"<0xA1>", 0.0f, 4},
    }, {0, 1, 2}, std::string(kReplacement) + "(" + kReplacement);
    expect_decode("overlong surrogate and out-of-range sequences", {
        {"\xC0\x80", 0.0f, 0}, {"\xED\xA0\x80", 0.0f, 0}, {"\xF4\x90\x80\x80", 0.0f, 0},
    }, {0, 1, 2}, std::string(kReplacement) + kReplacement + kReplacement + kReplacement +
        kReplacement + kReplacement + kReplacement + kReplacement + kReplacement);
    expect_decode("truncated four-byte sequence", {
        {"\xF0\x90\x80" "A", 0.0f, 0},
    }, {0}, std::string(kReplacement) + "A");
    expect_throw("invalid fallback hex", {{"<0xG0>", 0.0f, 4}}, {0});
    expect_throw("out-of-range ID", {{"word", 0.0f, 0}}, {1});
    std::cout << "Whistle tokenizer byte-fallback regressions pass\n";
}
