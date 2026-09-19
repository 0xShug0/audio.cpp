#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/tokenizers/qwen_bpe_bundle.h"
#include "engine/models/auk/message.h"
#include <fstream>
#include <iostream>
#include <vector>
int main(int argc, char ** argv) {
    engine::assets::ResourceBundle bundle(argv[1]);
    bundle.add_model_file("tokenizer_config", "tokenizer_config.json");
    bundle.add_model_file("tokenizer_json", "tokenizer.json");
    bundle.add_optional_model_file("vocab", "vocab.json");
    bundle.add_optional_model_file("merges", "merges.txt");
    auto tok = engine::tokenizers::load_qwen_bpe_tokenizer(bundle);
    auto ids = tok->encode(engine::models::auk::build_user_message(
        "Read this line in a calm voice: the harbour was quiet.", 0), true);
    std::ifstream ref(argv[2], std::ios::binary | std::ios::ate);
    const auto bytes = static_cast<std::streamoff>(ref.tellg());
    std::vector<int32_t> expected(static_cast<size_t>(bytes) / sizeof(int32_t));
    ref.seekg(0);
    ref.read(reinterpret_cast<char *>(expected.data()), bytes);
    std::cout << "ours=" << ids.size() << " reference=" << expected.size() << "\n";
    size_t mismatches = 0;
    for (size_t i = 0; i < std::min(ids.size(), expected.size()); ++i) {
        if (ids[i] != expected[i]) {
            if (mismatches < 5) std::cout << "  [" << i << "] " << ids[i] << " != " << expected[i] << "\n";
            ++mismatches;
        }
    }
    std::cout << (mismatches == 0 && ids.size() == expected.size() ? "TOKENIZATION IDENTICAL\n"
                                                                  : "mismatches: " + std::to_string(mismatches) + "\n");
    return 0;
}
