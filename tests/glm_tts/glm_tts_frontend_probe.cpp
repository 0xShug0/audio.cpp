#include "engine/community_models/glm_tts/frontend.h"
#include "engine/framework/audio/wav_reader.h"

#include <algorithm>
#include <exception>
#include <iomanip>
#include <iostream>
#include <numeric>

namespace {

void print_values(const std::vector<float> & values) {
    const size_t count = std::min<size_t>(values.size(), 32);
    std::cout << '[';
    for (size_t index = 0; index < count; ++index) {
        if (index != 0) {
            std::cout << ',';
        }
        std::cout << values[index];
    }
    std::cout << ']';
}

}  // namespace

int main(int argc, char ** argv) try {
    if (argc != 2) {
        std::cerr << "usage: glm_tts_frontend_probe <reference.wav>\n";
        return 2;
    }

    const auto wav =
        engine::audio::read_wav_f32(std::filesystem::path(argv[1]));
    engine::runtime::AudioBuffer audio;
    audio.sample_rate = wav.sample_rate;
    audio.channels = wav.channels;
    audio.samples = wav.samples;

    const auto mel =
        engine::models::glm_tts::compute_glm_tts_prompt_mel(audio);
    const auto fbank =
        engine::models::glm_tts::compute_glm_tts_campplus_fbank(audio);

    std::cout << std::setprecision(9);
    std::cout
        << "{\"mel\":{\"frames\":" << mel.frames
        << ",\"dims\":" << mel.dims
        << ",\"sum\":"
        << std::accumulate(mel.values.begin(), mel.values.end(), 0.0)
        << ",\"first\":";
    print_values(mel.values);
    std::cout
        << "},\"fbank\":{\"frames\":" << fbank.frames
        << ",\"dims\":" << fbank.dims
        << ",\"sum\":"
        << std::accumulate(fbank.values.begin(), fbank.values.end(), 0.0)
        << ",\"first\":";
    print_values(fbank.values);
    std::cout << "}}\n";
    return 0;
} catch (const std::exception & error) {
    std::cerr
        << "glm_tts_frontend_probe failed: "
        << error.what() << '\n';
    return 1;
}
