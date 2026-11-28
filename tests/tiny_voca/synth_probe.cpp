// Speaks given phoneme ids with given noise, for the comparison with the
// Python model (reference_audio.py writes the cases and the Python audio).
//
//   tiny_voca_synth_probe MODEL CASES OUT_DIR [THREADS] [CACHE_GRAPHS CACHE_MB] [--log FILE]
//
// CACHE_GRAPHS and CACHE_MB set the limits of the graph cache (0 graphs turns
// it off); without them the runtime's own limits hold. The use of the cache is
// written to stderr at the end.
//
// CASES has one utterance per line, fields separated by tabs:
//   name  voice  speed  pitch_shift  pitch_range  ids  stress  noise_file
// For each it writes OUT_DIR/name.f32 (float32 samples), OUT_DIR/name.f0 (the
// pitch of every frame, float32) and OUT_DIR/name.dur (the frames of every token).
#include "engine/community_models/tiny_voca/assets.h"
#include "engine/community_models/tiny_voca/runtime.h"
#include "engine/framework/debug/trace.h"

#include <chrono>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <numeric>
#include <set>
#include <stdexcept>
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
    if (!line.empty() && line.back() == '\t') out.emplace_back();
    return out;
}

std::vector<int32_t> numbers(const std::string & text) {
    std::vector<int32_t> out;
    std::istringstream stream(text);
    std::string token;
    while (stream >> token) {
        size_t consumed = 0;
        const int value = std::stoi(token, &consumed);
        if (consumed != token.size()) throw std::runtime_error("malformed integer list");
        out.push_back(value);
    }
    if (out.empty()) throw std::runtime_error("empty integer list");
    return out;
}

}  // namespace

int main(int argc, char ** argv) {
    std::string log_path;
    if (argc >= 3 && std::string(argv[argc - 2]) == "--log") {
        log_path = argv[argc - 1];
        argc -= 2;
    }
    if (argc != 4 && argc != 5 && argc != 7) {
        std::cerr << "usage: tiny_voca_synth_probe MODEL CASES OUT_DIR [THREADS] [CACHE_GRAPHS CACHE_MB] [--log FILE]\n";
        return 2;
    }
    try {
        engine::debug::LoggingConfig logging;
        logging.enabled = !log_path.empty();
        if (!log_path.empty()) logging.file_path = log_path;
        engine::debug::configure_logging(logging);
        const auto session_start = std::chrono::steady_clock::now();
        const auto assets = load_tiny_voca_assets(argv[1]);
        engine::core::BackendConfig backend;
        backend.type = engine::core::BackendType::Cpu;
        backend.threads = argc > 4 ? std::stoi(argv[4]) : 4;
        TinyVocaGraphCacheLimits limits;
        if (argc > 6) {
            limits.max_graphs = std::stoul(argv[5]);
            limits.max_bytes = static_cast<size_t>(std::stod(argv[6]) * 1024.0 * 1024.0);
        }
        TinyVocaRuntime runtime(assets, backend, limits);
        std::ifstream cases(argv[2]);
        if (!cases) throw std::runtime_error("cannot open cases file");
        std::filesystem::create_directories(argv[3]);
        std::set<std::string> names;
        std::string line;
        while (std::getline(cases, line)) {
            const auto f = fields(line);
            if (f.size() != 8) throw std::runtime_error("case must have exactly eight tab-separated fields");
            if (f[0].empty() || f[0] == "." || f[0] == ".." ||
                f[0].find_first_of("/\\") != std::string::npos || !names.insert(f[0]).second)
                throw std::runtime_error("invalid or duplicate case name");
            TinyVocaGenerationOptions options;
            size_t consumed = 0;
            options.voice = std::stoi(f[1], &consumed);
            if (consumed != f[1].size()) throw std::runtime_error("malformed voice");
            options.speed = std::stof(f[2], &consumed);
            if (consumed != f[2].size()) throw std::runtime_error("malformed speed");
            options.pitch_shift = std::stof(f[3], &consumed);
            if (consumed != f[3].size()) throw std::runtime_error("malformed pitch shift");
            options.pitch_range = std::stof(f[4], &consumed);
            if (consumed != f[4].size()) throw std::runtime_error("malformed pitch range");
            if (!std::isfinite(options.speed) || options.speed <= 0 ||
                !std::isfinite(options.pitch_shift) || !std::isfinite(options.pitch_range))
                throw std::runtime_error("invalid synthesis options");
            const auto ids = numbers(f[5]);
            const auto stress = numbers(f[6]);
            if (ids.size() != stress.size()) throw std::runtime_error("ids/stress length mismatch");
            std::ifstream noise_file(f[7], std::ios::binary | std::ios::ate);
            if (!noise_file) throw std::runtime_error("cannot open noise file: " + f[7]);
            const auto bytes = noise_file.tellg();
            if (bytes <= 0 || bytes % static_cast<std::streamoff>(sizeof(float)) != 0)
                throw std::runtime_error("empty or malformed noise file");
            std::vector<float> noise(static_cast<size_t>(bytes) / sizeof(float));
            noise_file.seekg(0);
            noise_file.read(reinterpret_cast<char *>(noise.data()), static_cast<std::streamsize>(noise.size() * sizeof(float)));
            if (!noise_file || !std::all_of(noise.begin(), noise.end(), [](float x) { return std::isfinite(x); }))
                throw std::runtime_error("truncated or nonfinite noise");
            const auto start = std::chrono::steady_clock::now();
            const TinyVocaOutput out = runtime.synthesize(ids, stress, options, &noise);
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            if (out.samples.empty() || out.f0.empty() || out.samples.size() != noise.size() ||
                assets->config.hop <= 0 || out.samples.size() / static_cast<size_t>(assets->config.hop) != out.f0.size() ||
                out.samples.size() % static_cast<size_t>(assets->config.hop) != 0 ||
                out.durations.size() != ids.size() ||
                !std::all_of(out.samples.begin(), out.samples.end(), [](float x) { return std::isfinite(x); }) ||
                !std::all_of(out.f0.begin(), out.f0.end(), [](float x) { return std::isfinite(x); }) ||
                !std::all_of(out.durations.begin(), out.durations.end(), [](auto x) { return x >= 0; }) ||
                std::accumulate(out.durations.begin(), out.durations.end(), int64_t{0}) != static_cast<int64_t>(out.f0.size()))
                throw std::runtime_error("invalid synthesis output shape or values");
            const std::string base = std::string(argv[3]) + "/" + f[0];
            std::ofstream samples(base + ".f32", std::ios::binary);
            samples.write(reinterpret_cast<const char *>(out.samples.data()), static_cast<std::streamsize>(out.samples.size() * sizeof(float)));
            std::ofstream pitch(base + ".f0", std::ios::binary);
            pitch.write(reinterpret_cast<const char *>(out.f0.data()), static_cast<std::streamsize>(out.f0.size() * sizeof(float)));
            std::ofstream durations(base + ".dur");
            for (size_t i = 0; i < out.durations.size(); ++i) {
                durations << (i ? " " : "") << out.durations[i];
            }
            durations << "\n";
            engine::debug::timing_log_scalar("request." + f[0] + ".wall_ms", seconds * 1000.0);
            samples.close();
            pitch.close();
            durations.close();
            if (!samples || !pitch || !durations) throw std::runtime_error("cannot write synthesis outputs");
            std::cout << f[0] << "\t" << out.samples.size() << " samples\t" << seconds << " s\n";
        }
        if (cases.bad() || names.empty()) throw std::runtime_error("unreadable or empty cases file");
        engine::debug::timing_log_scalar("session.wall_ms", std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - session_start).count());
        const GraphCacheStats cache = runtime.graph_cache_stats();
        std::cerr << "graph cache: " << cache.hits << " hits, " << cache.misses << " misses, " << cache.kept_graphs << " graphs kept, "
                  << static_cast<double>(cache.kept_bytes) / (1024.0 * 1024.0) << " MB kept\n";
        engine::debug::reset_logging();
    } catch (const std::exception & error) {
        engine::debug::reset_logging();
        std::cerr << "tiny_voca_synth_probe: " << error.what() << "\n";
        return 1;
    }
    return 0;
}
