#include "engine/community_models/lfm2_audio/tokenizer.h"
#include "engine/framework/runtime/partial_text.h"
#include "test_assert.h"

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using engine::community_models::lfm2_audio::Lfm2StreamedText;
using engine::community_models::lfm2_audio::Lfm2TextTokenizer;
using engine::community_models::lfm2_audio::Lfm2TextVocabulary;
using engine::community_models::lfm2_audio::lfm2_take_text;
using engine::test::require;
using engine::test::require_eq;

// tokenizer.ggml.token_type values.
constexpr int32_t kNormal = 1;
constexpr int32_t kControl = 3;
constexpr int32_t kUserDefined = 4;
constexpr int32_t kUnused = 5;

// A byte-level vocabulary in the GGUF layout: "Ġ" is the space byte and "Ċ"
// the newline byte.
Lfm2TextVocabulary vocabulary() {
    Lfm2TextVocabulary vocab;
    vocab.pre_tokenizer = "lfm2";
    const std::pair<const char *, int32_t> tokens[] = {
        {"<|startoftext|>", kControl},  // 0
        {"<|im_start|>", kControl},     // 1
        {"<|im_end|>", kControl},       // 2
        {"python", kUserDefined},       // 3
        {"h", kNormal},                 // 4
        {"e", kNormal},                 // 5
        {"l", kNormal},                 // 6
        {"o", kNormal},                 // 7
        {"Ġ", kNormal},                 // 8
        {"w", kNormal},                 // 9
        {"r", kNormal},                 // 10
        {"d", kNormal},                 // 11
        {"u", kNormal},                 // 12
        {"s", kNormal},                 // 13
        {"Ċ", kNormal},                 // 14
        {"he", kNormal},                // 15
        {"ll", kNormal},                // 16
        {"hell", kNormal},              // 17
        {"hello", kNormal},             // 18
        {"Ġw", kNormal},                // 19
        {"or", kNormal},                // 20
        {"Ġwor", kNormal},              // 21
        {"ld", kNormal},                // 22
        {"Ġworld", kNormal},            // 23
        {"1", kNormal},                 // 24
        {"2", kNormal},                 // 25
        {"3", kNormal},                 // 26
        {"4", kNormal},                 // 27
        {"5", kNormal},                 // 28
        {"12", kNormal},                // 29
        {"34", kNormal},                // 30
        {"123", kNormal},               // 31
        {"45", kNormal},                // 32
        {"I", kNormal},                 // 33
        {"'", kNormal},                 // 34
        {"T", kNormal},                 // 35
        {"'T", kNormal},                // 36
        {"[PAD37]", kUnused},           // 37, converter filler past the trained vocabulary
    };
    for (const auto & [text, type] : tokens) {
        vocab.tokens.emplace_back(text);
        vocab.token_types.push_back(type);
    }

    vocab.merges = {
        "h e", "l l", "he ll", "hell o", "Ġ w", "o r", "Ġw or", "l d", "Ġwor ld",
        // "3 4" outranks "12 3", so only a split into "123" and "45" can
        // produce "123".
        "1 2", "3 4", "4 5", "12 3",
        "' T",
    };
    return vocab;
}

std::string ids(const std::vector<int32_t> & values) {
    std::ostringstream out;
    for (size_t i = 0; i < values.size(); ++i) {
        out << (i == 0 ? "" : ",") << values[i];
    }

    return out.str();
}

void require_ids(const std::vector<int32_t> & actual, const std::vector<int32_t> & expected, const std::string & label) {
    require_eq(ids(actual), ids(expected), label);
}

void test_encode() {
    const Lfm2TextTokenizer tokenizer(vocabulary());
    require_ids(tokenizer.encode("hello world"), {18, 23}, "hello world");
    require_ids(tokenizer.encode("<|im_start|>user\n"), {1, 12, 13, 5, 10, 14}, "control token prefix");
    require_ids(tokenizer.encode("hello<|im_end|>"), {18, 2}, "control token suffix");
    // User-defined tokens are matched whole too, as in the HF tokenizer.
    require_ids(tokenizer.encode("hello python"), {18, 8, 3}, "user-defined token");
    require(tokenizer.encode("").empty(), "empty text must encode to no tokens");
}

// llama.cpp tokenizes pre = "lfm2" with its Llama 3 split: digits go in groups
// of up to three, and contractions match in any case. GPT-2's split would give
// 12,34,5 and I,',T here.
void test_llama3_split() {
    const Lfm2TextTokenizer tokenizer(vocabulary());
    require_ids(tokenizer.encode("12345"), {31, 32}, "digit groups");
    require_ids(tokenizer.encode("I'T"), {33, 36}, "uppercase contraction");
}

void test_decode() {
    const Lfm2TextTokenizer tokenizer(vocabulary());
    require_eq(tokenizer.decode({18, 23}), std::string("hello world"), "byte-level decode");
    require_eq(tokenizer.decode({0, 1, 18, 2}), std::string("hello"), "control tokens are dropped");
    require_eq(tokenizer.decode({18, 8, 3}), std::string("hello python"), "user-defined tokens are kept");
    require_eq(tokenizer.decode({12, 13, 5, 10, 14}), std::string("user\n"), "newline byte");
    require_eq(tokenizer.decode({18, 37}), std::string("hello"), "unused tokens are dropped");
}

// Bytes from hex such as "E0 B8", and back, so that a failure shows bytes.
std::string from_hex(const std::string & hex) {
    std::istringstream in(hex);
    std::string out;
    std::string byte;
    while (in >> byte) {
        out.push_back(static_cast<char>(std::stoi(byte, nullptr, 16)));
    }

    return out;
}

std::string to_hex(const std::string & bytes) {
    std::ostringstream out;
    for (size_t i = 0; i < bytes.size(); ++i) {
        out << (i == 0 ? "" : " ") << std::uppercase << std::hex << std::setw(2) << std::setfill('0')
            << static_cast<int>(static_cast<unsigned char>(bytes[i]));
    }

    return out.str();
}

// A tail held for more bytes: the start of one character, which gives no
// text by itself.
void require_open_character(const std::string & tail, const std::string & label) {
    auto held = tail;
    require(tail.size() < 4 && lfm2_take_text(held).empty() && held == tail, label + " is the start of one character");
}

// `bytes` give `text` whole, and in two parts split anywhere or byte by byte,
// as tokens bring them: what the first part gave is never taken back, and a
// tail held for more bytes is one open character, left out of the text. The
// text is well-formed, so taking it again changes nothing. The S2S stream's
// events, split the same ways, carry the same text.
void require_text(const std::string & bytes_hex, const std::string & text_hex, const std::string & label) {
    const auto bytes = from_hex(bytes_hex);
    auto tail = bytes;
    const auto text = lfm2_take_text(tail);
    require_eq(to_hex(text), text_hex, label);
    if (!tail.empty()) {
        require_open_character(tail, label + ": the tail held at the end");
    }

    auto again = text;
    require_eq(to_hex(lfm2_take_text(again)), text_hex, label + " taken again");
    require(again.empty(), label + ": the text taken again must all be taken");

    for (size_t split = 0; split <= bytes.size(); ++split) {
        auto rest = bytes.substr(0, split);
        auto parts = lfm2_take_text(rest);
        if (!rest.empty()) {
            require_open_character(rest, label + ": the tail held at " + std::to_string(split));
        }

        rest += bytes.substr(split);
        parts += lfm2_take_text(rest);
        require_eq(to_hex(parts), text_hex, label + " split at " + std::to_string(split));
        require_eq(to_hex(rest), to_hex(tail), label + " split at " + std::to_string(split) + ", the tail held at the end");
    }

    std::string pending;
    std::string bytewise;
    for (const char byte : bytes) {
        pending.push_back(byte);
        bytewise += lfm2_take_text(pending);
    }

    require_eq(to_hex(bytewise), text_hex, label + " byte by byte");
    require_eq(to_hex(pending), to_hex(tail), label + " byte by byte, the tail held at the end");

    for (size_t split = 0; split <= bytes.size(); ++split) {
        Lfm2StreamedText stream;
        auto events = stream.add(bytes.substr(0, split));
        events += stream.add(bytes.substr(split));
        require_eq(to_hex(events), text_hex, label + " streamed, split at " + std::to_string(split));
        require_eq(to_hex(stream.text()), text_hex, label + " streamed text, split at " + std::to_string(split));
    }

    Lfm2StreamedText stream;
    std::string events;
    for (const char byte : bytes) {
        events += stream.add(std::string(1, byte));
    }

    require_eq(to_hex(events), text_hex, label + " streamed byte by byte");
    require_eq(to_hex(stream.text()), text_hex, label + " streamed text, byte by byte");
}

// The text liquid-audio's tokenizer decodes from the same bytes: one U+FFFD
// (EF BF BD) per maximal subpart of an ill-formed sequence.
void test_text_from_bytes() {
    // Unicode 3.9, U+FFFD Substitution of Maximal Subparts, Tables 3-8 to 3-12.
    require_text("61 F1 80 80 E1 80 C2 62 80 63 80 BF 64",
        "61 EF BF BD EF BF BD EF BF BD 62 EF BF BD 63 EF BF BD EF BF BD 64", "Table 3-8");
    require_text("C0 AF E0 80 BF F0 81 82 41",
        "EF BF BD EF BF BD EF BF BD EF BF BD EF BF BD EF BF BD EF BF BD EF BF BD 41", "Table 3-9, non-shortest forms");
    require_text("ED A0 80 ED BF BF ED AF 41",
        "EF BF BD EF BF BD EF BF BD EF BF BD EF BF BD EF BF BD EF BF BD EF BF BD 41", "Table 3-10, surrogates");
    require_text("F4 91 92 93 FF 41 80 BF 42",
        "EF BF BD EF BF BD EF BF BD EF BF BD EF BF BD 41 EF BF BD EF BF BD 42", "Table 3-11, other ill-formed sequences");
    require_text("E1 80 E2 F0 91 92 F1 BF 41", "EF BF BD EF BF BD EF BF BD EF BF BD 41", "Table 3-12, truncated sequences");

    // Well-formed text is kept as it is: "hé 日本 🙂".
    require_text("68 C3 A9 20 E6 97 A5 E6 9C AC 20 F0 9F 99 82", "68 C3 A9 20 E6 97 A5 E6 9C AC 20 F0 9F 99 82", "well-formed text");
    require_text("", "", "no bytes");

    // A Thai character left open before " ay" and a reply that ends inside
    // one, as English replies in Thai script have them; an emoji cut off by
    // the next character, and two open characters at the end. Mid-text the
    // open character is U+FFFD; at the end it is dropped.
    require_text("E0 B8 AB E0 B8 20 61 79", "E0 B8 AB EF BF BD 20 61 79", "a character left open mid-text");
    require_text("E0 B8 B5 E0 B8", "E0 B8 B5", "a reply that ends inside a character");
    require_text("F0 9F E3 83 AA", "EF BF BD E3 83 AA", "an emoji cut off by the next character");
    require_text("E3 82 F0 9F", "EF BF BD", "two open characters at the end");

    // Boundary bytes, and whole and cut characters from the edges of Table
    // 3-7, at random; the text is what Python's incremental decoder gives
    // before the end, codecs.getincrementaldecoder("utf-8")("replace")
    // .decode(bytes, final=False), which holds an open character at the end.
    // CPython also holds a surrogate's first two bytes there (ED A0 to ED
    // BF), which no byte can finish and which are two U+FFFD here, as in
    // Rust's String::from_utf8_lossy and Python's final decode; no case
    // below ends with them.
    const std::pair<const char *, const char *> python[] = {
        {"EF BF E0 F0 F4 80 80 80 ED", "EF BF BD EF BF BD EF BF BD F4 80 80 80"},
        {"F4 8F BF BF F1 80", "F4 8F BF BF"},
        {"F3 BF BF F4 90 F1 80 80 80 90", "EF BF BD EF BF BD EF BF BD F1 80 80 80 EF BF BD"},
        {"EF BF 00 ED F3 BF", "EF BF BD 00 EF BF BD"},
        {"EF F3 BF BF BF F3 EF", "EF BF BD F3 BF BF BF EF BF BD"},
        {"E0 F4 8F BF BF ED 9F F4 80 80 80", "EF BF BD F4 8F BF BF EF BF BD F4 80 80 80"},
        {"00 90 F4 EC F0", "00 EF BF BD EF BF BD EF BF BD"},
        {"41 BF ED 9F BF ED C0", "41 EF BF BD ED 9F BF EF BF BD EF BF BD"},
        {"EF C2 F1 EF BF F4", "EF BF BD EF BF BD EF BF BD EF BF BD"},
        {"8F BF ED F4 80 F1 80 80 80 EE", "EF BF BD EF BF BD EF BF BD EF BF BD F1 80 80 80"},
        {"EC F5 E0 A0 80 41", "EF BF BD EF BF BD E0 A0 80 41"},
        {"EE E0 A0 80", "EF BF BD E0 A0 80"},
        {"E0 A0 80 F0 90 E1 80 80 EF BF BD ED 9F BF FF", "E0 A0 80 EF BF BD E1 80 80 EF BF BD ED 9F BF EF BF BD"},
        {"C1 EF 00 F0 90 80 80", "EF BF BD EF BF BD 00 F0 90 80 80"},
        {"EE EC BF 90", "EF BF BD EC BF 90"},
        {"EF E0 F5 ED 80 80", "EF BF BD EF BF BD EF BF BD ED 80 80"},
        {"A0 E1", "EF BF BD"},
        {"F3 E1 80 80", "EF BF BD E1 80 80"},
        {"EC ED", "EF BF BD"},
        {"EF BF BD F3", "EF BF BD"},
        {"F0 90 80 80 C2 80 80 F1", "F0 90 80 80 C2 80 EF BF BD"},
        {"EC BF BF ED", "EC BF BF"},
        {"F1 41", "EF BF BD 41"},
        {"EC F3 BF F3", "EF BF BD EF BF BD"},
    };
    for (size_t i = 0; i < std::size(python); ++i) {
        require_text(python[i].first, python[i].second, "random sequence " + std::to_string(i));
    }

    // While more bytes may come, an open character waits for them.
    auto bytes = from_hex("61 E0 B8");
    require_eq(to_hex(lfm2_take_text(bytes)), std::string("61"), "text before an open character");
    require_eq(to_hex(bytes), std::string("E0 B8"), "the open character's bytes");
    bytes += from_hex("AB");
    require_eq(to_hex(lfm2_take_text(bytes)), std::string("E0 B8 AB"), "the character once finished");
    require(bytes.empty(), "a finished character must be taken");
}

// Streams `events` (hex, one per event) and checks each event's text and then
// the stream's text, which is the events' text joined.
void require_streamed(const std::vector<std::string> & events, const std::vector<std::string> & deltas, const std::string & text_hex,
    const std::string & label) {
    Lfm2StreamedText stream;
    for (size_t i = 0; i < events.size(); ++i) {
        const auto delta = stream.add(from_hex(events[i]));
        require_eq(to_hex(delta), deltas[i], label + ": event " + std::to_string(i));
    }

    require_eq(to_hex(stream.text()), text_hex, label + ": the stream's text");
}

// The S2S stream's text, event by event. Each event's text is whole
// characters; bytes that make none are U+FFFD as soon as a later byte shows
// it, and a character the reply leaves open is left out, as offline, so no
// event carries any of it. A stream finished before its reply is over has the
// text its events carried.
void test_streamed_text() {
    require_streamed({"61 E6 97", "A5 62"}, {"61", "E6 97 A5 62"}, "61 E6 97 A5 62", "a character split across events");
    require_streamed({"61 80 62", "E0 B8 AB E0 B8", "20 61 79"}, {"61 EF BF BD 62", "E0 B8 AB", "EF BF BD 20 61 79"},
        "61 EF BF BD 62 E0 B8 AB EF BF BD 20 61 79", "ill-formed sequences mid-text");
    require_streamed({"E0 B8 B5 E0 B8", ""}, {"E0 B8 B5", ""}, "E0 B8 B5", "a reply over with a character open");
    require_streamed({"E0 B8 B5", "E0 B8"}, {"E0 B8 B5", ""}, "E0 B8 B5", "a reply over in the event that opens a character");
    require_streamed({"80 61 E0 B8", "AB 62 E3 81"}, {"EF BF BD 61", "E0 B8 AB 62"}, "EF BF BD 61 E0 B8 AB 62",
        "ill-formed bytes mid-text and a character open at the end");
    require_streamed({"61 E6 97"}, {"61"}, "61", "a stream finished early inside a character");
    require_streamed({"E3 82 AA F0 9F", "98"}, {"E3 82 AA", ""}, "E3 82 AA", "a stream finished early inside an emoji");
}

// Text that is well-formed but for a character left open at its end, which
// every transcript seen so far is, comes out byte for byte as
// runtime::transcript_publishable_end ends it, as the ASR did before
// lfm2_take_text: every pair of characters from the edges of Table 3-7 with
// the second cut after each of its bytes, and longer texts at random.
void test_whole_text_as_before() {
    const std::vector<std::string> characters = {"00", "41", "7F", "C2 80", "C3 A9", "DF BF", "E0 A0 80", "E0 BF BF",
        "E1 80 80", "E6 97 A5", "EC BF BF", "ED 80 80", "ED 9F BF", "EE 80 80", "EF BF BF", "F0 90 80 80", "F0 9F 99 82",
        "F0 BF BF BF", "F1 80 80 80", "F3 BF BF BF", "F4 80 80 80", "F4 8F BF BF"};
    const auto require_as_before = [](const std::string & text, const std::string & label) {
        auto bytes = text;
        const auto before = text.substr(0, engine::runtime::transcript_publishable_end(text));
        require_eq(to_hex(lfm2_take_text(bytes)), to_hex(before), label + " (" + to_hex(text) + ")");
    };

    for (const auto & first : characters) {
        for (const auto & second : characters) {
            const auto whole = from_hex(first) + from_hex(second);
            for (size_t cut = 0; cut <= whole.size(); ++cut) {
                require_as_before(whole.substr(0, cut), "a pair of characters cut after byte " + std::to_string(cut));
            }
        }
    }

    uint32_t state = 20261004;  // a fixed linear congruential sequence
    const auto next = [&state](uint32_t bound) {
        state = state * 1664525u + 1013904223u;
        return (state >> 8) % bound;
    };
    for (int round = 0; round < 2000; ++round) {
        std::string text;
        const auto count = next(12);
        for (uint32_t i = 0; i < count; ++i) {
            text += from_hex(characters[next(static_cast<uint32_t>(characters.size()))]);
        }

        const auto last = from_hex(characters[next(static_cast<uint32_t>(characters.size()))]);
        text += last.substr(0, next(static_cast<uint32_t>(last.size()) + 1));
        require_as_before(text, "random text " + std::to_string(round));
    }
}

void test_token_lookup() {
    const Lfm2TextTokenizer tokenizer(vocabulary());
    require_eq(tokenizer.require_token_id("<|im_end|>"), 2, "<|im_end|> id");
    require(tokenizer.is_control_token(2), "<|im_end|> is a control token");
    require(!tokenizer.is_control_token(3), "python is not a control token");
    require(!tokenizer.is_control_token(18), "hello is not a control token");
    require(!tokenizer.is_control_token(1000), "an unknown id is not a control token");
}

template <typename Fn>
void require_throws(Fn && fn, const std::string & label) {
    bool threw = false;
    try {
        fn();
    } catch (const std::runtime_error &) {
        threw = true;
    }

    require(threw, label + " must throw");
}

void test_rejects_bad_vocabulary() {
    const Lfm2TextTokenizer tokenizer(vocabulary());
    require_throws([&] { (void)tokenizer.require_token_id("<|audio_start|>"); }, "a missing token");
    require_throws([&] { (void)tokenizer.decode({1000}); }, "decoding an unknown id");

    auto other_pre = vocabulary();
    other_pre.pre_tokenizer = "llama-bpe";
    require_throws([&] { (void)Lfm2TextTokenizer{other_pre}; }, "another pre-tokenizer");

    auto no_pre = vocabulary();
    no_pre.pre_tokenizer.clear();
    require_throws([&] { (void)Lfm2TextTokenizer{no_pre}; }, "a missing pre-tokenizer");

    auto bad_merge = vocabulary();
    bad_merge.merges.emplace_back("hello");
    require_throws([&] { (void)Lfm2TextTokenizer{bad_merge}; }, "a merge without a separator");
}

}  // namespace

int main() {
    try {
        test_encode();
        test_llama3_split();
        test_decode();
        test_text_from_bytes();
        test_streamed_text();
        test_whole_text_as_before();
        test_token_lookup();
        test_rejects_bad_vocabulary();
        std::cout << "lfm2_audio_tokenizer_test: PASS\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "lfm2_audio_tokenizer_test: " << error.what() << '\n';
        return 1;
    }
}
