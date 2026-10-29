// The LFM3-Audio tokenizer: markup ids resolved by content, HF special-token
// parsing, the skip-special decode filter and tokenizer.json ignore_merges,
// on a synthetic tokenizer.json. With --model and --fixtures it also checks
// the real tokenizer against HF: probes/tokenizer_cases.json (ids,
// keep-special and skip-special decode) and probes/ignore_merges_cases.json
// (the vocab strings ignore_merges tokenizes as one token). With --gguf, a
// published backbone GGUF's tokenizer must match tokenizer.json.
#include "engine/community_models/lfm3_audio/tokenizer.h"
#include "engine/community_models/lfm3_audio/published_gguf.h"
#include "engine/framework/io/json.h"
#include "engine/framework/tokenizers/llama_bpe.h"
#include "lfm3_audio_test_tokenizer.h"
#include "test_assert.h"

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace {

namespace lfm3 = engine::community_models::lfm3_audio;
namespace json = engine::io::json;
namespace tokenizers = engine::tokenizers;
using engine::test::require;
using engine::test::require_eq;
using lfm3_audio_test::added_id;
using lfm3_audio_test::require_ids;
using lfm3_audio_test::show;

std::vector<int32_t> bytes(const std::string & text) {
    std::vector<int32_t> out;
    for (const char c : text) {
        out.push_back(static_cast<unsigned char>(c));
    }

    return out;
}

void test_token_ids() {
    const auto dir = lfm3_audio_test::fresh_directory("lfm3_audio_tokenizer_test_ids");
    const lfm3::Lfm3Tokenizer tokenizer(lfm3_audio_test::write_tokenizer(dir, true));
    const auto & ids = tokenizer.ids();
    require_eq(ids.bos, added_id("<|startoftext|>"), "bos");
    require_eq(ids.im_start, added_id("<|im_start|>"), "im_start");
    require_eq(ids.im_end, added_id("<|im_end|>"), "im_end");
    require_eq(ids.nl, 10, "the newline byte token");
    require_eq(ids.text_end, added_id("<|text_end|>"), "text_end");
    require_eq(ids.audio_start, added_id("<|audio_start|>"), "audio_start");
    require_eq(ids.mixed_start, added_id("<|mixed_start|>"), "mixed_start");
    require_eq(ids.audio, added_id("<|audio|>"), "audio");
    require_eq(ids.audio_out, added_id("<|audio_out|>"), "audio_out");
    require_eq(ids.think, added_id("<think>"), "think");
    require_eq(ids.think_end, added_id("</think>"), "think_end");

    for (const auto & token : lfm3_audio_test::added_tokens()) {
        const auto missing = lfm3_audio_test::fresh_directory("lfm3_audio_tokenizer_test_missing");
        lfm3_audio_test::require_throws_with([&] { (void)lfm3::Lfm3Tokenizer(lfm3_audio_test::write_tokenizer(missing, true, token.content)); },
            std::string("LFM3-Audio tokenizer has no token ") + token.content, std::string("a tokenizer without ") + token.content);
    }
}

// HF matches every added token in raw text before pre-tokenization, so the
// space before one is a token of its own.
void test_encode() {
    const auto dir = lfm3_audio_test::fresh_directory("lfm3_audio_tokenizer_test_encode");
    const lfm3::Lfm3Tokenizer tokenizer(lfm3_audio_test::write_tokenizer(dir, true));
    const int32_t space = 32;
    require_ids(tokenizer.encode("pre <|im_end|> post"),
        {'p', 'r', 'e', space, added_id("<|im_end|>"), space, 'p', 'o', 's', 't'}, "an added token inside text");
    require_ids(tokenizer.encode("<think>sy</think>"), {added_id("<think>"), lfm3_audio_test::kSy, added_id("</think>")},
        "non-special added tokens and a merge");
    require_ids(tokenizer.encode("<|IM_END|>"), bytes("<|IM_END|>"), "no case folding of added tokens");
    require(tokenizer.encode("").empty(), "empty text");
}

// decode(skip_special_tokens=True) keeps <think>/</think>, drops the
// special=true tokens and the frame placeholders.
void test_decode() {
    const auto dir = lfm3_audio_test::fresh_directory("lfm3_audio_tokenizer_test_decode");
    const lfm3::Lfm3Tokenizer tokenizer(lfm3_audio_test::write_tokenizer(dir, true));
    const std::vector<int32_t> ids = {added_id("<think>"), '!', added_id("</think>"), added_id("<|text_end|>"), added_id("<|audio_out|>"),
                                      added_id("<|audio_out|>"), added_id("<|im_end|>")};
    require_eq(tokenizer.decode(ids), std::string("<think>!</think>"), "skip-special decode");
    require_eq(tokenizer.decode(ids, false), std::string("<think>!</think><|text_end|><|audio_out|><|audio_out|><|im_end|>"),
        "keep-special decode");
    require_eq(tokenizer.decode(tokenizer.encode("pre <|im_end|> post")), std::string("pre  post"), "the space around a skipped token stays");
    require_eq(tokenizer.decode({lfm3_audio_test::kAbc, ' ', lfm3_audio_test::kBc, 10}), std::string("abc bc\n"), "byte-level decode");
}

// tokenizer.json ignore_merges: "abc" is a vocab entry that no merge reaches.
// Lfm3Tokenizer honours the flag; the framework default keeps merges only.
void test_ignore_merges() {
    const auto on = lfm3_audio_test::write_tokenizer(lfm3_audio_test::fresh_directory("lfm3_audio_tokenizer_test_im_on"), true);
    const auto off = lfm3_audio_test::write_tokenizer(lfm3_audio_test::fresh_directory("lfm3_audio_tokenizer_test_im_off"), false);
    const std::vector<int32_t> merged = {'a', lfm3_audio_test::kBc};
    require_ids(lfm3::Lfm3Tokenizer(on).encode("abc"), {lfm3_audio_test::kAbc}, "a whole vocab word with ignore_merges");
    require_ids(lfm3::Lfm3Tokenizer(on).encode("abcd"), {'a', lfm3_audio_test::kBc, 'd'}, "a longer word still merges");
    require_ids(lfm3::Lfm3Tokenizer(on).encode("abc abc"), {lfm3_audio_test::kAbc, ' ', 'a', lfm3_audio_test::kBc},
        "only whole pre-tokens are looked up");
    require_ids(lfm3::Lfm3Tokenizer(off).encode("abc"), merged, "ignore_merges false in tokenizer.json");
    const tokenizers::LlamaBpeTokenizer framework_default(
        tokenizers::LlamaBpeTokenizerSpec{{}, {}, {}, on, tokenizers::LlamaBpePreTokenizer::Llama3});
    require_ids(framework_default.encode("abc", true), merged, "the framework default is unchanged");
}

// The real-data paths: --model <upstream dir with tokenizer.json> and
// --fixtures <fixture dir>. Real-data checks run only when both are given,
// and the test exits 77 (skipped) when a given path is missing.
struct RealData {
    std::filesystem::path model, fixtures;
    std::filesystem::path gguf;  // --gguf: a published backbone GGUF to check against tokenizer.json
    [[nodiscard]] bool wanted() const { return !model.empty() || !fixtures.empty(); }
    [[nodiscard]] bool present() const {
        return !model.empty() && !fixtures.empty() && std::filesystem::exists(model / "tokenizer.json") && std::filesystem::is_directory(fixtures);
    }
    [[nodiscard]] std::filesystem::path tokenizer_json() const { return model / "tokenizer.json"; }
};

RealData parse_args(int argc, char ** argv) {
    RealData out;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if ((arg == "--model" || arg == "--fixtures" || arg == "--gguf") && i + 1 < argc) {
            (arg == "--model" ? out.model : arg == "--fixtures" ? out.fixtures : out.gguf) = argv[++i];
        } else {
            throw std::runtime_error("unknown argument " + arg + " (takes --model <dir> --fixtures <dir> [--gguf <file>])");
        }
    }

    return out;
}

// The bytes a byte-level token string stands for (added tokens are printable
// ASCII, which stands for itself).
std::string bytes_of_tokens(const std::vector<std::string> & tokens) {
    std::string chars;
    for (const auto & token : tokens) {
        chars += token;
    }

    static const auto byte_of = [] {
        std::unordered_map<std::string, char> out;
        const auto byte_tokens = lfm2_audio_test::byte_tokens();
        for (int byte = 0; byte < 256; ++byte) {
            out.emplace(byte_tokens[static_cast<size_t>(byte)], static_cast<char>(byte));
        }
        return out;
    }();
    std::string out;
    for (size_t at = 0; at < chars.size();) {
        const size_t n = (static_cast<unsigned char>(chars[at]) & 0x80) == 0 ? 1 : 2;
        const auto it = byte_of.find(chars.substr(at, n));
        if (it == byte_of.end()) {
            throw std::runtime_error("not a byte-level token string: " + chars);
        }
        out.push_back(it->second);
        at += n;
    }

    return out;
}

std::vector<int32_t> int_array(const json::Value & value) {
    std::vector<int32_t> out;
    for (const auto & item : value.as_array()) {
        out.push_back(static_cast<int32_t>(item.as_i64()));
    }

    return out;
}

std::vector<std::string> string_array(const json::Value & value) {
    std::vector<std::string> out;
    for (const auto & item : value.as_array()) {
        out.push_back(item.as_string());
    }

    return out;
}

// cJSON strings end at a NUL, so a case's text is rebuilt from its tokens,
// and expected strings are compared up to the first NUL.
bool same_up_to_nul(const std::string & actual, const std::string & expected) {
    return actual.substr(0, actual.find('\0')) == expected;
}

struct Counts {
    size_t cases = 0, encode = 0, keep = 0, skip = 0;
};

Counts check_cases(const lfm3::Lfm3Tokenizer & tokenizer, const std::filesystem::path & path, bool decodes) {
    Counts counts;
    std::string first_failure;
    const auto doc = json::parse_file(path);
    for (const auto & item : doc.require("cases").as_array()) {
        ++counts.cases;
        const auto text = bytes_of_tokens(string_array(item.require("tokens")));
        require(same_up_to_nul(text, item.require("text").as_string()), "case " + std::to_string(counts.cases) + " tokens spell its text");
        const auto ids = int_array(item.require("ids"));
        const auto encoded = tokenizer.encode(text);
        counts.encode += encoded == ids;
        if (encoded != ids && first_failure.empty()) {
            first_failure = "encode " + json::stringify_string(text) + ": expected " + show(ids) + ", got " + show(encoded);
        }

        if (decodes) {
            counts.keep += same_up_to_nul(tokenizer.decode(ids, false), item.require("decode_keep_special").as_string());
            counts.skip += same_up_to_nul(tokenizer.decode(ids), item.require("decode_skip_special").as_string());
        }
    }

    if (!first_failure.empty()) {
        std::cerr << "first failure: " << first_failure << '\n';
    }

    return counts;
}

// The tokenizer built from a published GGUF's tokenizer.ggml.* against the one from tokenizer.json.
void test_gguf(const RealData & real) {
    const lfm3::Lfm3Tokenizer from_json(real.tokenizer_json());
    const auto vocabulary = lfm3::open_lfm3_audio_published_gguf(real.gguf).vocabulary;
    const lfm3::Lfm3Tokenizer from_gguf(vocabulary);
    const auto & a = from_json.ids();
    const auto & b = from_gguf.ids();
    require(std::tie(a.bos, a.im_start, a.im_end, a.nl, a.text_end, a.audio_start, a.mixed_start, a.audio, a.audio_out, a.think, a.think_end) ==
                std::tie(b.bos, b.im_start, b.im_end, b.nl, b.text_end, b.audio_start, b.mixed_start, b.audio, b.audio_out, b.think, b.think_end),
            "the GGUF tokenizer has the same markup ids");

    size_t encode = 0, keep = 0, skip = 0;
    std::string first_failure;
    for (size_t id = 0; id < vocabulary.tokens.size(); ++id) {
        const auto text = bytes_of_tokens({vocabulary.tokens[id]});
        const auto expected = from_json.encode(text);
        const auto actual = from_gguf.encode(text);
        encode += actual == expected;
        if (actual != expected && first_failure.empty()) {
            first_failure = "encode " + json::stringify_string(text) + ": expected " + show(expected) + ", got " + show(actual);
        }

        // Ids past tokenizer.json (llama.cpp's [PADn] filler) decode in neither.
        const std::vector<int32_t> one{static_cast<int32_t>(id)};
        const auto decoded = [&](const lfm3::Lfm3Tokenizer & tokenizer, bool skip_special) -> std::optional<std::string> {
            try {
                return tokenizer.decode(one, skip_special);
            } catch (const std::exception &) {
                return std::nullopt;
            }
        };
        keep += decoded(from_gguf, false) == decoded(from_json, false);
        skip += decoded(from_gguf, true) == decoded(from_json, true);
    }

    if (!first_failure.empty()) {
        std::cerr << "first failure: " << first_failure << '\n';
    }

    const auto n = vocabulary.tokens.size();
    std::cout << "GGUF tokenizer: " << n << " vocab entries, encode " << encode << ", keep-special decode " << keep
              << ", skip-special decode " << skip << " as tokenizer.json\n";
    require(n > 0 && encode == n && keep == n && skip == n, "the GGUF tokenizer matches tokenizer.json");

    const auto cases = real.fixtures / "probes" / "tokenizer_cases.json";
    if (std::filesystem::exists(cases)) {
        const auto c = check_cases(from_gguf, cases, true);
        std::cout << "tokenizer_cases.json with the GGUF tokenizer: " << c.cases << " cases, " << c.encode << " / " << c.keep << " / " << c.skip
                  << " exact\n";
        require(c.cases > 0 && c.encode == c.cases && c.keep == c.cases && c.skip == c.cases, "every tokenizer case matches HF");
    }
}

void test_real(const RealData & real) {
    const lfm3::Lfm3Tokenizer tokenizer(real.tokenizer_json());
    const auto cases = real.fixtures / "probes" / "tokenizer_cases.json";
    if (!std::filesystem::exists(cases)) {
        std::cout << "no " << cases.string() << "; tokenizer cases skipped\n";
    } else {
        const auto c = check_cases(tokenizer, cases, true);
        std::cout << "tokenizer_cases.json: " << c.cases << " cases, encode " << c.encode << ", keep-special decode " << c.keep
                  << ", skip-special decode " << c.skip << " exact\n";
        require(c.cases > 0 && c.encode == c.cases && c.keep == c.cases && c.skip == c.cases, "every tokenizer case matches HF");
    }

    const auto merges = real.fixtures / "probes" / "ignore_merges_cases.json";
    if (!std::filesystem::exists(merges)) {
        std::cout << "no " << merges.string() << "; ignore_merges cases skipped\n";
        return;
    }

    const auto m = check_cases(tokenizer, merges, false);
    std::cout << "ignore_merges_cases.json: " << m.cases << " cases, encode " << m.encode << " exact\n";
    require(m.cases > 0 && m.encode == m.cases, "every ignore_merges case matches HF");

    const tokenizers::LlamaBpeTokenizer merges_only(
        tokenizers::LlamaBpeTokenizerSpec{{}, {}, {}, real.tokenizer_json(), tokenizers::LlamaBpePreTokenizer::Llama3});
    size_t differ = 0;
    const auto doc = json::parse_file(merges);
    for (const auto & item : doc.require("cases").as_array()) {
        const auto text = bytes_of_tokens(string_array(item.require("tokens")));
        differ += merges_only.encode(text, true) != int_array(item.require("ids"));
    }

    std::cout << "without honor_ignore_merges: " << differ << " of " << m.cases << " differ from HF\n";
    require(differ == m.cases, "every ignore_merges case needs the flag");
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const auto real = parse_args(argc, argv);
        test_token_ids();
        test_encode();
        test_decode();
        test_ignore_merges();
        if (real.wanted()) {
            if (!real.present()) {
                std::cout << "lfm3_audio_tokenizer_test: SKIP (no tokenizer.json under --model or no --fixtures)\n";
                return 77;
            }

            test_real(real);
            if (!real.gguf.empty()) {
                test_gguf(real);
            }
        }

        std::cout << "lfm3_audio_tokenizer_test: PASS\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "lfm3_audio_tokenizer_test: " << error.what() << '\n';
        return 1;
    }
}
