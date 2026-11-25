// zonos3_tokenizer_test <backbone-ptq.gguf> <tokenizer_vectors.json>
// The SuperBPE ids must equal ZONOS3's tests/fixtures/tokenizer-expected.json for every case.
#include "engine/community_models/zonos3/tokenizer.h"
#include "engine/framework/assets/gguf_metadata.h"
#include "engine/framework/io/json.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace zonos3 = engine::models::zonos3;
namespace json = engine::io::json;

int main(int argc, char ** argv) {
    try {
        if (argc != 3) throw std::runtime_error("usage: zonos3_tokenizer_test BACKBONE_GGUF VECTORS_JSON");
        const engine::assets::GgufMetadata meta(argv[1]);
        const zonos3::Zonos3Tokenizer tokenizer(meta.require_string("zonos3.tokenizer.json"));
        const auto vectors = json::parse_file(argv[2]);
        int passed = 0;
        for (const auto & item : vectors.require("cases").as_array()) {
            const auto & text = item.require("text").as_string();
            const auto ids = tokenizer.text_ids(text);
            const auto & expected = item.require("ids").as_array();
            bool equal = ids.size() == expected.size();
            for (size_t i = 0; equal && i < ids.size(); ++i) equal = ids[i] == expected[i].as_i64();
            if (!equal) {
                std::string got;
                for (const auto id : ids) got += std::to_string(id) + " ";
                throw std::runtime_error("token mismatch for \"" + text + "\": " + got);
            }
            ++passed;
        }
        std::cout << "zonos3 tokenizer vectors passed: " << passed << "\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "zonos3_tokenizer_test: " << error.what() << "\n";
        return 1;
    }
}
