// Speaks given phoneme ids with given noise, for the comparison with the
// Python model (reference_audio.py writes the cases and the Python audio).
//
//   tiny_voca_synth_probe MODEL CASES OUT_DIR [THREADS]
//
// CASES has one utterance per line, fields separated by tabs:
//   name  voice  speed  pitch_shift  pitch_range  ids  stress  noise_file
// For each it writes OUT_DIR/name.f32 (float32 samples), OUT_DIR/name.f0 (the
// pitch of every frame, float32) and OUT_DIR/name.dur (the frames of every token).
#include "engine/community_models/tiny_voca/assets.h"
#include "engine/community_models/tiny_voca/runtime.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace engine::models::tiny_voca;

namespace {

std::vector<std::string> fields(const std::string & line) {
    std::vector<std::string> out;
    std::istringstream stream(line);
    std::string item;
    while (std::getline(stream, item, '\t')) {
        out.push_back(item);
    }
    return out;
}

std::vector<int32_t> numbers(const std::string & text) {
    std::vector<int32_t> out;
    std::istringstream stream(text);
    int value = 0;
    while (stream >> value) {
        out.push_back(value);
    }
    return out;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 4) {
        std::cerr << "usage: tiny_voca_synth_probe MODEL CASES OUT_DIR [THREADS]\n";
        return 2;
    }
    try {
        const auto assets = load_tiny_voca_assets(argv[1]);
        engine::core::BackendConfig backend;
        backend.type = engine::core::BackendType::Cpu;
        backend.threads = argc > 4 ? std::stoi(argv[4]) : 4;
        TinyVocaRuntime runtime(assets, backend);
        std::ifstream cases(argv[2]);
        std::string line;
        while (std::getline(cases, line)) {
            const auto f = fields(line);
            if (f.size() < 8) {
                continue;
            }
            TinyVocaGenerationOptions options;
            options.voice = std::stoi(f[1]);
            options.speed = std::stof(f[2]);
            options.pitch_shift = std::stof(f[3]);
            options.pitch_range = std::stof(f[4]);
            std::ifstream noise_file(f[7], std::ios::binary | std::ios::ate);
            std::vector<float> noise(static_cast<size_t>(noise_file.tellg()) / sizeof(float));
            noise_file.seekg(0);
            noise_file.read(reinterpret_cast<char *>(noise.data()), static_cast<std::streamsize>(noise.size() * sizeof(float)));
            const auto start = std::chrono::steady_clock::now();
            const TinyVocaOutput out = runtime.synthesize(numbers(f[5]), numbers(f[6]), options, &noise);
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            const std::string base = std::string(argv[3]) + "/" + f[0];
            std::ofstream(base + ".f32", std::ios::binary)
                .write(reinterpret_cast<const char *>(out.samples.data()), static_cast<std::streamsize>(out.samples.size() * sizeof(float)));
            std::ofstream(base + ".f0", std::ios::binary)
                .write(reinterpret_cast<const char *>(out.f0.data()), static_cast<std::streamsize>(out.f0.size() * sizeof(float)));
            std::ofstream durations(base + ".dur");
            for (size_t i = 0; i < out.durations.size(); ++i) {
                durations << (i ? " " : "") << out.durations[i];
            }
            durations << "\n";
            std::cout << f[0] << "\t" << out.samples.size() << " samples\t" << seconds << " s\n";
        }
    } catch (const std::exception & error) {
        std::cerr << "tiny_voca_synth_probe: " << error.what() << "\n";
        return 1;
    }
    return 0;
}
