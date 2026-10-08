#include "engine/community_models/fish_drama3/tokenizer_text.h"
#include "engine/framework/io/json.h"
#include "engine/framework/runtime/errors.h"

#include <iostream>
#include <stdexcept>

namespace drama = engine::models::fish_drama3;
namespace json = engine::io::json;

int main(int argc, char ** argv) {
    try {
        if (argc != 3) throw std::runtime_error("usage: tokenizer_test SIDECAR_DIRECTORY VECTORS_JSON");
        engine::assets::ResourceBundle resources(argv[1]);
        resources.add_model_files({{"tokenizer_json", "tokenizer.json"},
                                   {"tokenizer_config", "tokenizer_config.json"}});
        drama::Drama3TextTokenizer tokenizer(resources);
        const auto vectors = json::parse_file(argv[2]);
        int passed = 0;
        for (const auto & item : vectors.require("cases").as_array()) {
            auto text = item.require("text").as_string();
            if (const auto * bytes = item.find("utf8")) {
                text.clear();
                for (const auto & byte : bytes->as_array()) text.push_back(static_cast<char>(byte.as_i64()));
            }
            const auto tokens = tokenizer.encode(text);
            const auto & expected = item.require("tokens").as_array();
            bool equal = tokens.size() == expected.size();
            for (size_t i = 0; equal && i < tokens.size(); ++i) equal = tokens[i] == expected[i].as_i64();
            if (!equal) throw std::runtime_error("token mismatch: " + item.require("id").as_string());
            ++passed;
        }
        const std::vector<std::string> invalid = {
            std::string("\xc0\xaf", 2), std::string("\xed\xa0\x80", 3),
            std::string("\xf4\x90\x80\x80", 4), std::string("\xe2\x82", 2),
            std::string("\x80", 1),
        };
        for (const auto & text : invalid) {
            bool rejected = false;
            try { tokenizer.encode(text); }
            catch (const engine::runtime::InvalidRequestError &) { rejected = true; }
            if (!rejected) throw std::runtime_error("malformed UTF-8 accepted");
        }
        // A rejected input must not poison the persistent tokenizer.
        const auto & first = vectors.require("cases").as_array().front();
        const auto recovered = tokenizer.encode(first.require("text").as_string());
        const auto & expected = first.require("tokens").as_array();
        bool recovered_exact = recovered.size() == expected.size();
        for (size_t i = 0; recovered_exact && i < recovered.size(); ++i) {
            recovered_exact = recovered[i] == expected[i].as_i64();
        }
        if (!recovered_exact) {
            throw std::runtime_error("tokenizer recovery failed");
        }
        std::cout << "reference_vectors_exact=" << passed
                  << " malformed_utf8_rejected=" << invalid.size() << " recovery=true\n";
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
