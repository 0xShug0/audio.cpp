#include "engine/community_models/glm_tts/assets.h"
#include "engine/community_models/glm_tts/speech_tokenizer.h"
#include "engine/framework/audio/wav_reader.h"

#include <exception>
#include <iostream>

int main(int argc, char ** argv) try {
    if (argc != 3) {
        std::cerr
            << "usage: glm_tts_speech_tokenizer_probe <model-path> "
               "<reference.wav>\n";
        return 2;
    }
    auto assets = engine::models::glm_tts::load_glm_tts_assets(argv[1]);
    const auto wav =
        engine::audio::read_wav_f32(std::filesystem::path(argv[2]));
    engine::runtime::AudioBuffer audio;
    audio.sample_rate = wav.sample_rate;
    audio.channels = wav.channels;
    audio.samples = wav.samples;
    const engine::models::glm_tts::GlmTTSSpeechTokenizer tokenizer(
        std::move(assets),
        {engine::core::BackendType::Cpu, 0, 8});
    const auto ids = tokenizer.encode(audio);
    std::cout << "{\"count\":" << ids.size() << ",\"ids\":[";
    for (size_t index = 0; index < ids.size(); ++index) {
        if (index != 0) {
            std::cout << ',';
        }
        std::cout << ids[index];
    }
    std::cout << "]}\n";
    return 0;
} catch (const std::exception & error) {
    std::cerr
        << "glm_tts_speech_tokenizer_probe failed: "
        << error.what() << '\n';
    return 1;
}
