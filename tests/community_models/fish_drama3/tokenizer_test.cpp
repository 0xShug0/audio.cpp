#include "engine/community_models/fish_drama3/tokenizer_text.h"
#include "engine/community_models/fish_drama3/prompt.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/io/json.h"
#include "engine/framework/runtime/errors.h"

#include <iostream>
#include <stdexcept>

namespace drama = engine::models::fish_drama3;
namespace json = engine::io::json;

int main(int argc, char ** argv) {
    try {
        if (argc != 3 && !(argc == 4 && std::string(argv[3]) == "--log"))
            throw std::runtime_error("usage: tokenizer_test SIDECAR_DIRECTORY VECTORS_JSON [--log]");
        if (argc == 4) engine::debug::configure_logging({true, std::nullopt});
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
        // Compare actual prompt rows, including fixed references and explicit takes.
        drama::Drama3Assets assets;
        const drama::Drama3Take reference{"A reference.", {std::vector<int32_t>(20, 1), 10, 2}};
        const drama::Drama3Take explicit_take{"Caller history.", {std::vector<int32_t>(30, 2), 10, 3}};
        const drama::Drama3Take first_take{"First chunk.", {std::vector<int32_t>(40, 3), 10, 4}};
        const drama::Drama3Take middle_take{"Middle chunk.", {std::vector<int32_t>(50, 4), 10, 5}};
        const drama::Drama3Take latest_take{"Latest chunk.", {std::vector<int32_t>(60, 5), 10, 6}};
        const std::string next_text = "The next chunk.";
        const auto mandatory = drama::build_drama3_prompt(assets, tokenizer, next_text, {reference}, {explicit_take});
        const auto anchor = drama::build_drama3_prompt(assets, tokenizer, next_text, {reference}, {explicit_take, first_take});
        const auto full = drama::build_drama3_prompt(assets, tokenizer, next_text, {reference}, {explicit_take, first_take, latest_take});
        const std::vector<drama::Drama3Take> generated{first_take, middle_take, latest_take};
        if (drama::build_drama3_prompt(assets, tokenizer, next_text, {reference}, {explicit_take}, generated) != full)
            throw std::runtime_error("automatic history must retain first and latest, excluding middle");
        if (drama::build_drama3_prompt(assets, tokenizer, next_text, {reference}, {explicit_take}, {first_take}) != anchor)
            throw std::runtime_error("the first generated turn must not be duplicated");
        if (drama::build_drama3_prompt(assets, tokenizer, next_text, {reference}, {explicit_take}, generated, full.size()) != full)
            throw std::runtime_error("an exactly fitting full history must remain unchanged");
        if (drama::build_drama3_prompt(assets, tokenizer, next_text, {reference}, {explicit_take}, generated, anchor.size()) != anchor)
            throw std::runtime_error("drop the latest automatic turn before the anchor");
        if (drama::build_drama3_prompt(assets, tokenizer, next_text, {reference}, {explicit_take}, generated, mandatory.size()) != mandatory)
            throw std::runtime_error("drop only automatic history when necessary");
        bool overflow_rejected = false;
        try { drama::build_drama3_prompt(assets, tokenizer, next_text, {reference}, {explicit_take}, generated, mandatory.size() - 1); }
        catch (const engine::runtime::CapacityError &) { overflow_rejected = true; }
        if (!overflow_rejected) throw std::runtime_error("required references, caller history and input must not be dropped");
        bool invalid_rejected = false;
        try { drama::build_drama3_prompt(assets, tokenizer, "<|im_start|>", {reference}, {explicit_take}, generated, 1); }
        catch (const engine::runtime::InvalidRequestError &) { invalid_rejected = true; }
        if (!invalid_rejected) throw std::runtime_error("prompt fitting must not hide malformed input");
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
