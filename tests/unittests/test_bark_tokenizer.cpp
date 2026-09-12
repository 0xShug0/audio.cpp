#include "engine/models/bark_tts/tokenizer.h"

#include <iostream>
#include <vector>

int main() {
    const std::string json = R"({"model":{"vocab":{"[UNK]":100,"[CLS]":101,"[SEP]":102,"Hello":31178,",":117,"Bar":20698,"##k":10174,"!":106,"你":2262,"好":3240}}})";
    engine::models::bark_tts::BarkTokenizer tokenizer(json);
    const std::vector<int32_t> expected{101, 31178, 117, 20698, 10174, 106, 102};
    if (tokenizer.encode("Hello, Bark!") != expected) {
        std::cerr << "Bark WordPiece tokenization mismatch\n";
        return 1;
    }
    const std::vector<int32_t> chinese{101, 2262, 3240, 102};
    if (tokenizer.encode("你好") != chinese) {
        std::cerr << "Bark Chinese-character splitting mismatch\n";
        return 1;
    }
    return 0;
}
