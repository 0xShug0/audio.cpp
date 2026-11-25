#include "engine/models/chatterbox/text_tokenizer.h"

#include "test_assert.h"

#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace chatterbox = engine::models::chatterbox;
namespace fs = std::filesystem;

// One token per character and no merges, so every id stands for one codepoint.
std::vector<std::string> tokenizer_pieces() {
    std::vector<std::string> pieces = {"[START]", "[STOP]", "[UNK]", "[SPACE]", "[tr]"};
    for (char ch = 'a'; ch <= 'z'; ++ch) {
        pieces.emplace_back(1, ch);
    }
    pieces.emplace_back("\u0131");  // dotless i
    pieces.emplace_back("\u0307");  // combining dot above
    pieces.emplace_back("\u0327");  // combining cedilla
    return pieces;
}

constexpr size_t kSpecialTokenCount = 5;

fs::path write_tokenizer(const std::vector<std::string> & pieces) {
    const auto path = fs::temp_directory_path() /
        ("chatterbox-tokenizer-test-" + std::to_string(std::random_device{}()) + ".json");
    std::ofstream out(path, std::ios::binary);
    out << "{\"model\":{\"type\":\"BPE\",\"vocab\":{";
    for (size_t i = 0; i < pieces.size(); ++i) {
        out << (i == 0 ? "" : ",") << '"' << pieces[i] << "\":" << i;
    }
    out << "},\"merges\":[]},\"added_tokens\":[";
    for (size_t i = 0; i < kSpecialTokenCount; ++i) {
        out << (i == 0 ? "" : ",") << "{\"content\":\"" << pieces[i] << "\"}";
    }
    out << "]}";
    return path;
}

int32_t id_of(const std::vector<std::string> & pieces, const std::string & piece) {
    for (size_t i = 0; i < pieces.size(); ++i) {
        if (pieces[i] == piece) {
            return static_cast<int32_t>(i);
        }
    }
    throw std::runtime_error("piece missing from the test vocab: " + piece);
}

// The ids for already prepared text: the language token, then one id per codepoint,
// with [SPACE] for a space.
std::vector<int32_t> ids_for_prepared(const std::vector<std::string> & pieces, const std::string & prepared) {
    std::vector<int32_t> ids = {id_of(pieces, "[tr]")};
    for (size_t pos = 0; pos < prepared.size();) {
        size_t end = pos + 1;
        while (end < prepared.size() && (static_cast<unsigned char>(prepared[end]) & 0xC0U) == 0x80U) {
            ++end;
        }
        const std::string piece = prepared.substr(pos, end - pos);
        ids.push_back(id_of(pieces, piece == " " ? "[SPACE]" : piece));
        pos = end;
    }
    return ids;
}

void test_multilingual_lowercase_matches_python_str_lower() {
    const auto pieces = tokenizer_pieces();
    const auto path = write_tokenizer(pieces);
    const auto tokenizer = chatterbox::load_chatterbox_english_tokenizer(path);
    fs::remove(path);

    // The expected text is unicodedata.normalize("NFKD", text.lower()), what the
    // reference MTLTokenizer.preprocess_text returns.
    struct Case {
        const char * text;
        const char * prepared;
    };
    const Case cases[] = {
        {"İstanbul", "i\u0307stanbul"},
        {"İZMİR", "i\u0307zmi\u0307r"},
        {"İyi akşamlar", "i\u0307yi aks\u0327amlar"},
        // str.lower() does not depend on the locale: I gives i, not the dotless ı.
        {"Işık", "is\u0327\u0131k"},
    };
    for (const auto & c : cases) {
        const auto actual = chatterbox::encode_chatterbox_multilingual_text(*tokenizer, c.text, "tr");
        engine::test::require(
            actual == ids_for_prepared(pieces, c.prepared),
            std::string("multilingual ids for ") + c.text);
    }
}

}  // namespace

int main() {
    try {
        test_multilingual_lowercase_matches_python_str_lower();
        std::cout << "chatterbox_text_tokenizer_test passed\n";
        return 0;
    } catch (const std::exception & ex) {
        std::cerr << "chatterbox_text_tokenizer_test failed: " << ex.what() << "\n";
        return 1;
    }
}
