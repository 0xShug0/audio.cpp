#include "engine/community_models/glm_tts/assets.h"

#include <exception>
#include <iostream>

int main(int argc, char ** argv) try {
    if (argc != 2) {
        std::cerr << "usage: glm_tts_assets_probe <model-path>\n";
        return 2;
    }
    const auto assets =
        engine::models::glm_tts::load_glm_tts_assets(argv[1]);
    const auto & config = assets->config;
    std::cout
        << "{\"llama_layers\":" << config.llama.num_hidden_layers
        << ",\"llama_hidden\":" << config.llama.hidden_size
        << ",\"speech_layers\":" << config.speech_tokenizer.pooling_position
        << ",\"speech_vocab\":" << config.speech_tokenizer.quantize_vocab_size
        << ",\"flow_depth\":" << config.flow.depth
        << ",\"sample_rate\":" << config.sample_rate
        << ",\"audio_token_start\":" << config.audio_token_start
        << ",\"audio_token_end\":" << config.audio_token_end
        << "}\n";
    return 0;
} catch (const std::exception & error) {
    std::cerr << "glm_tts_assets_probe failed: " << error.what() << '\n';
    return 1;
}
