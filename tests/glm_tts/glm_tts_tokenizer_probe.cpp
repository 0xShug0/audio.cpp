#include "engine/community_models/glm_tts/tokenizer_text.h"

#include <exception>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>

int main(int argc, char ** argv) try {
    if (argc != 5) {
        std::cerr
            << "usage: glm_tts_tokenizer_probe <vocab.json> <merges.txt> "
               "<tokenizer_config.json> <text-or-@utf8-file>\n";
        return 2;
    }

    std::string text = argv[4];
    if (!text.empty() && text.front() == '@') {
        std::ifstream stream(text.substr(1), std::ios::binary);
        if (!stream) {
            throw std::runtime_error(
                "could not open UTF-8 input file: " + text.substr(1));
        }
        text.assign(
            std::istreambuf_iterator<char>(stream),
            std::istreambuf_iterator<char>());
        while (!text.empty() &&
               (text.back() == '\n' || text.back() == '\r')) {
            text.pop_back();
        }
    }

    const engine::models::glm_tts::GlmTTSTextTokenizer tokenizer(
        argv[1], argv[2], argv[3]);
    const auto ids = tokenizer.encode(text);
    std::cout << "{\"ids\":[";
    for (size_t index = 0; index < ids.size(); ++index) {
        if (index != 0) {
            std::cout << ',';
        }
        std::cout << ids[index];
    }
    std::cout << "],\"special\":{"
              << "\"audio_0\":"
              << tokenizer.require_token_id("<|audio_0|>") << ','
              << "\"audio_32767\":"
              << tokenizer.require_token_id("<|audio_32767|>") << ','
              << "\"begin_of_audio\":"
              << tokenizer.require_token_id("<|begin_of_audio|>") << ','
              << "\"end_of_audio\":"
              << tokenizer.require_token_id("<|user|>") << ','
              << "\"pad\":"
              << tokenizer.require_token_id("<|endoftext|>") << "}}\n";
    return 0;
} catch (const std::exception & error) {
    std::cerr << "glm_tts_tokenizer_probe failed: " << error.what() << '\n';
    return 1;
}
