#include "engine/models/sheetsage/audio_frontend.h"
#include "engine/framework/audio/wav_reader.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char ** argv) {
    try {
        engine::models::sheetsage::SheetSage2AudioFrontend frontend;
        const std::vector<float> stereo{1, 0, 0, 1, 1, -1};
        const auto mono = frontend.prepare(stereo, 24000, 2, 24000, 8);
        if (mono.size() != 3 || std::abs(mono[0] - std::sqrt(0.5F)) > 1e-7F ||
            mono[0] != mono[1] || mono[2] != 0) {
            throw std::runtime_error("stereo mixing mismatch");
        }
        for (int rate : {8000, 16000, 22050, 32000, 44100, 48000, 96000}) {
            const std::vector<float> constant(2001, 0.25F);
            const auto first = frontend.prepare(constant, rate, 1, 24000, 8);
            if (first != frontend.prepare(constant, rate, 1, 24000, 8)) {
                throw std::runtime_error("cached filter mismatch");
            }
            for (float value : first) {
                if (std::abs(value - 0.25F) > 2e-5F) {
                    throw std::runtime_error("DC gain mismatch");
                }
            }
        }
        // Optional external oracle pairs: input WAV followed by reference mono WAV.
        for (int arg = 1; arg < argc; ++arg) {
            if (std::string(argv[arg]) == "--log") continue;
            if (arg + 1 >= argc) throw std::runtime_error("expected WAV pair");
            const auto input = engine::audio::read_wav_f32(std::filesystem::path(argv[arg]));
            const auto reference = engine::audio::read_wav_f32(std::filesystem::path(argv[++arg]));
            const auto actual = frontend.prepare(input.samples, input.sample_rate, input.channels,
                                                 reference.sample_rate, 8);
            if (reference.channels != 1 || actual.size() != reference.samples.size()) {
                throw std::runtime_error("oracle sample count mismatch: " + std::to_string(actual.size()) +
                                         " vs " + std::to_string(reference.samples.size()));
            }
            double maximum = 0, squared = 0;
            for (size_t i = 0; i < actual.size(); ++i) {
                const double error = actual[i] - reference.samples[i];
                maximum = std::max(maximum, std::abs(error));
                squared += error * error;
            }
            std::cout << argv[arg - 1] << " samples=" << actual.size() << " max=" << maximum
                      << " rms=" << std::sqrt(squared / std::max<size_t>(1, actual.size())) << '\n';
            if (maximum > 2e-6) throw std::runtime_error("oracle sample mismatch");
        }
        std::cout << "SheetSage2 frontend passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
