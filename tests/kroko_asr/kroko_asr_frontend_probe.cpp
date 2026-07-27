#include "engine/community_models/kroko_asr/frontend.h"
#include "engine/framework/audio/wav_reader.h"

#include <algorithm>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>

int main(int argc, char ** argv) try {
    if (argc != 2) {
        std::cerr << "usage: kroko_asr_frontend_probe <audio.wav>\n";
        return 2;
    }
    const auto wav =
        engine::audio::read_wav_f32(std::filesystem::path(argv[1]));
    const engine::runtime::AudioBuffer audio{
        wav.sample_rate, wav.channels, wav.samples};
    const auto features =
        engine::models::kroko_asr::compute_kroko_fbank(audio);

    const size_t count = std::min<size_t>(features.values.size(), 32);
    std::cout << std::setprecision(9)
              << "{\"frames\":" << features.frames
              << ",\"dims\":" << features.feature_dim
              << ",\"sum\":"
              << std::accumulate(
                     features.values.begin(),
                     features.values.end(),
                     0.0)
              << ",\"first\":[";
    for (size_t index = 0; index < count; ++index) {
        if (index != 0) {
            std::cout << ',';
        }
        std::cout << features.values[index];
    }
    std::cout << "]}\n";
    return 0;
} catch (const std::exception & error) {
    std::cerr << "kroko_asr_frontend_probe failed: "
              << error.what() << '\n';
    return 1;
}
