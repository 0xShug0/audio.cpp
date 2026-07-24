#include "engine/community_models/glm_tts/prompt.h"

#include <exception>
#include <iostream>

int main(int argc, char ** argv) try {
    if (argc != 5) {
        std::cerr
            << "usage: glm_tts_prompt_probe <vocab.json> <merges.txt> "
               "<tokenizer_config.json> <text>\n";
        return 2;
    }
    engine::models::glm_tts::GlmTTSConfig config;
    config.llama.max_position_embeddings = 8192;
    config.audio_token_start = 61498;
    config.audio_token_end = 94265;
    config.begin_audio_token = 59260;
    const engine::models::glm_tts::GlmTTSTextTokenizer tokenizer(
        argv[1], argv[2], argv[3]);
    const auto prompt = engine::models::glm_tts::build_glm_tts_prompt(
        config,
        tokenizer,
        "Hello.",
        argv[4],
        {0, 42, 32767});
    std::cout << "{\"ids\":[";
    for (size_t index = 0; index < prompt.input_ids.size(); ++index) {
        if (index != 0) {
            std::cout << ',';
        }
        std::cout << prompt.input_ids[index];
    }
    std::cout
        << "],\"text_tokens\":" << prompt.text_token_count
        << ",\"minimum_audio_tokens\":" << prompt.minimum_audio_tokens
        << ",\"maximum_audio_tokens\":" << prompt.maximum_audio_tokens
        << "}\n";
    return 0;
} catch (const std::exception & error) {
    std::cerr << "glm_tts_prompt_probe failed: " << error.what() << '\n';
    return 1;
}
