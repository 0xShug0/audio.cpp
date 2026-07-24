#include "engine/community_models/glm_tts/assets.h"
#include "engine/community_models/glm_tts/llama.h"
#include "engine/community_models/glm_tts/prompt.h"
#include "engine/community_models/glm_tts/speech_tokenizer.h"
#include "engine/community_models/glm_tts/tokenizer_text.h"
#include "engine/framework/audio/wav_reader.h"

#include <exception>
#include <iostream>

int main(int argc, char ** argv) try {
    if (argc < 5) {
        std::cerr
            << "usage: glm_tts_llama_probe <model-path> <reference.wav> "
               "<reference-text> <text> [--greedy] [--seed n] "
               "[--full-head]\n";
        return 2;
    }

    auto assets = engine::models::glm_tts::load_glm_tts_assets(argv[1]);
    const auto wav =
        engine::audio::read_wav_f32(std::filesystem::path(argv[2]));
    engine::runtime::AudioBuffer audio;
    audio.sample_rate = wav.sample_rate;
    audio.channels = wav.channels;
    audio.samples = wav.samples;

    const engine::models::glm_tts::GlmTTSSpeechTokenizer speech_tokenizer(
        assets, {engine::core::BackendType::Cuda, 0, 8});
    const auto reference_tokens = speech_tokenizer.encode(audio);
    const engine::models::glm_tts::GlmTTSTextTokenizer text_tokenizer(
        assets->resources.require_file("tokenizer_vocab"),
        assets->resources.require_file("tokenizer_merges"),
        assets->resources.require_file("tokenizer_config"));
    const auto prompt = engine::models::glm_tts::build_glm_tts_prompt(
        assets->config,
        text_tokenizer,
        argv[3],
        argv[4],
        reference_tokens);

    const engine::models::glm_tts::GlmTTSLlamaRuntime llama(
        assets,
        engine::core::BackendType::Cuda,
        0,
        8);
    engine::models::glm_tts::GlmTTSGenerateOptions options;
    options.seed = 0;
    for (int index = 5; index < argc; ++index) {
        const std::string option = argv[index];
        if (option == "--greedy") {
            options.temperature = 0.0F;
        } else if (option == "--full-head") {
            options.restrict_output_head = false;
        } else if (option == "--seed" && index + 1 < argc) {
            options.seed = static_cast<uint32_t>(
                std::stoul(argv[++index]));
        } else {
            throw std::runtime_error(
                "unknown or incomplete probe option: " + option);
        }
    }
    const auto generated = llama.generate(prompt, options);

    std::cout << "{\"prompt_ids\":[";
    for (size_t index = 0; index < prompt.input_ids.size(); ++index) {
        if (index != 0) std::cout << ',';
        std::cout << prompt.input_ids[index];
    }
    std::cout << "],\"speech_tokens\":[";
    for (size_t index = 0; index < generated.speech_tokens.size(); ++index) {
        if (index != 0) std::cout << ',';
        std::cout << generated.speech_tokens[index];
    }
    std::cout
        << "],\"stopped\":"
        << (generated.stopped_on_end_of_audio ? "true" : "false")
        << "}\n";
    return 0;
} catch (const std::exception & error) {
    std::cerr
        << "glm_tts_llama_probe failed: "
        << error.what() << '\n';
    return 1;
}
