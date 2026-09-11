// Token ids in tokenizer.json size the id_to_token table directly:
//
//     max_id = max(max_id, static_cast<size_t>(value.as_i64()));
//     std::vector<std::string> id_to_token(max_id + 1);
//
// A negative id becomes ~SIZE_MAX on that cast, and a large positive one asks
// for terabytes -- an out-of-memory condition driven by a model file, not a
// parse error. Both are now rejected with a range check.

#include "engine/framework/tokenizers/hf_tokenizer_json.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::filesystem::path write_tokenizer(const std::string & vocab_json) {
    static int counter = 0;
    const auto path = std::filesystem::temp_directory_path() /
                      ("hf_vocab_bounds_" + std::to_string(counter++) + ".json");
    std::ofstream out(path);
    require(out.good(), "failed to open temp tokenizer file");
    out << R"({"model":{"vocab":)" << vocab_json << "}}";
    require(out.good(), "failed to write temp tokenizer file");
    return path;
}

bool rejects(const std::string & vocab_json) {
    const auto path = write_tokenizer(vocab_json);
    bool threw = false;
    try {
        (void) engine::tokenizers::load_huggingface_tokenizer_json(path);
    } catch (const std::runtime_error &) {
        threw = true;
    }
    std::filesystem::remove(path);
    return threw;
}

void test_absurd_token_id_is_rejected() {
    // Would size the table at ~10^12 entries of std::string.
    require(rejects(R"({"a":0,"b":1000000000000})"),
            "an out-of-range token id must be rejected");
}

void test_negative_token_id_is_rejected() {
    require(rejects(R"({"a":0,"b":-1})"),
            "a negative token id must be rejected");
}

void test_ordinary_vocab_still_loads() {
    const auto path = write_tokenizer(R"({"hello":0,"world":1,"!":2})");
    const auto tokenizer = engine::tokenizers::load_huggingface_tokenizer_json(path);
    std::filesystem::remove(path);
    require(tokenizer != nullptr, "a well-formed vocab should load");
}

}  // namespace

int main() {
    try {
        test_absurd_token_id_is_rejected();
        test_negative_token_id_is_rejected();
        test_ordinary_vocab_still_loads();
    } catch (const std::exception & error) {
        std::cerr << "hf tokenizer vocab bounds test failed: " << error.what() << "\n";
        return 1;
    }
    std::cout << "hf tokenizer vocab bounds test passed\n";
    return 0;
}
