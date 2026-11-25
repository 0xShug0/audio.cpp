#pragma once

// Shared by the lfm3_audio tokenizer, prompt, chat and session tests: a tiny
// byte-level BPE tokenizer.json with the markup tokens the prompts use, and
// the assertions on token ids and on errors.

#include "../lfm2_audio/lfm2_audio_test_package.h"
#include "engine/framework/io/json.h"
#include "engine/framework/runtime/errors.h"
#include "test_assert.h"

#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>

namespace lfm3_audio_test {

using lfm2_audio_test::fresh_directory;
using lfm2_audio_test::require_throws_with;

// Ids of the synthetic tokenizer: byte b is token b, then three multi-byte
// tokens, then the added tokens.
inline constexpr int32_t kBc = 256;   // "bc", from the merge "b c"
inline constexpr int32_t kAbc = 257;  // "abc", a vocab entry no merge reaches
inline constexpr int32_t kSy = 258;   // "sy", from the merge "s y"

struct AddedToken {
    int32_t id;
    const char * content;
    bool special;
};

inline const std::vector<AddedToken> & added_tokens() {
    static const std::vector<AddedToken> tokens = {
        {300, "<|startoftext|>", true}, {301, "<|im_start|>", true},  {302, "<|im_end|>", true},     {303, "<think>", false},
        {304, "</think>", false},       {305, "<|text_end|>", true},  {306, "<|audio_start|>", true}, {307, "<|mixed_start|>", true},
        {308, "<|audio|>", true},       {309, "<|audio_out|>", true},
    };
    return tokens;
}

inline int32_t added_id(const std::string & content) {
    for (const auto & token : added_tokens()) {
        if (content == token.content) {
            return token.id;
        }
    }

    throw std::runtime_error("no added token " + content);
}

// `skip` names an added token to leave out.
inline std::filesystem::path write_tokenizer(const std::filesystem::path & dir, bool ignore_merges, const std::string & skip = {}) {
    namespace json = engine::io::json;
    std::filesystem::create_directories(dir);
    const auto byte_tokens = lfm2_audio_test::byte_tokens();
    std::string vocab;
    for (int byte = 0; byte < 256; ++byte) {
        vocab += json::stringify_string(byte_tokens[static_cast<size_t>(byte)]) + ":" + std::to_string(byte) + ",";
    }
    vocab += "\"bc\":256,\"abc\":257,\"sy\":258";

    std::string added;
    for (const auto & token : added_tokens()) {
        if (skip == token.content) {
            continue;
        }

        added += std::string(added.empty() ? "" : ",") + "{\"id\":" + std::to_string(token.id) + ",\"content\":" +
                 json::stringify_string(token.content) +
                 ",\"single_word\":false,\"lstrip\":false,\"rstrip\":false,\"normalized\":false,\"special\":" +
                 (token.special ? "true" : "false") + "}";
    }

    const auto path = dir / "tokenizer.json";
    std::ofstream out(path, std::ios::binary);
    out << "{\"version\":\"1.0\",\"added_tokens\":[" << added << "],\"normalizer\":null,"
        << "\"model\":{\"type\":\"BPE\",\"ignore_merges\":" << (ignore_merges ? "true" : "false") << ",\"vocab\":{" << vocab
        << "},\"merges\":[\"b c\",\"s y\"]}}";
    if (!out) {
        throw std::runtime_error("cannot write " + path.string());
    }

    return path;
}

inline std::string show(const std::vector<int32_t> & ids) {
    std::string out = "[";
    for (const int32_t id : ids) {
        out += (out.size() > 1 ? "," : "") + std::to_string(id);
    }

    return out + "]";
}

inline void require_ids(const std::vector<int32_t> & actual, const std::vector<int32_t> & expected, const std::string & label) {
    if (actual == expected) {
        return;
    }

    size_t at = 0;
    while (at < actual.size() && at < expected.size() && actual[at] == expected[at]) {
        ++at;
    }

    engine::test::require(false, label + ": first difference at " + std::to_string(at) + " of " + std::to_string(expected.size()) +
                                     " (got " + std::to_string(actual.size()) + " ids)" +
                                     (expected.size() < 200 ? ": expected " + show(expected) + ", got " + show(actual) : ""));
}

inline std::vector<int32_t> join(std::initializer_list<std::vector<int32_t>> parts) {
    std::vector<int32_t> out;
    for (const auto & part : parts) {
        out.insert(out.end(), part.begin(), part.end());
    }

    return out;
}

// `fn` throws Error, and no other exception, with `needle` in its message.
template <typename Error, typename Fn>
void require_error(Fn && fn, const std::string & needle, const std::string & label) {
    try {
        fn();
    } catch (const Error & error) {
        const std::string message = error.what();
        engine::test::require(message.find(needle) != std::string::npos, label + " threw \"" + message + "\", expected \"" + needle + "\"");
        return;
    } catch (const std::exception & error) {
        engine::test::require(false, label + " threw another error: " + error.what());
    }

    engine::test::require(false, label + " must throw");
}

template <typename Fn>
void require_request_error(Fn && fn, const std::string & needle, const std::string & label) {
    require_error<engine::runtime::InvalidRequestError>(fn, needle, label);
}

}  // namespace lfm3_audio_test
