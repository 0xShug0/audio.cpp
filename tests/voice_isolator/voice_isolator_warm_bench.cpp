// Voice Isolator speed: real-time factor and per-push latency of a stream,
// or the real-time factor of an offline render.
//   voice_isolator_warm_bench --model CHECKPOINT.safetensors [--offline] [--seconds S] [--push SAMPLES]
//       [--block FRAMES] [--lookahead FRAMES] [--chunk SECONDS] [--threads N] [--weights f32|bf16|...] [--cache f32|f16|q8_0|q5_0|q4_0]
//       [--backend cpu|cuda|vulkan|metal|...]
#include "app/cli/args.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/community_models/voice_isolator/runtime.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

std::vector<float> speech_like(int64_t samples) {
    std::vector<float> x(static_cast<size_t>(samples));
    uint32_t state = 12345U;
    const double pi = 3.14159265358979323846;
    for (int64_t i = 0; i < samples; ++i) {
        const double t = static_cast<double>(i) / 48000.0;
        state = state * 1664525U + 1013904223U;
        const double noise = static_cast<double>(state) / 4294967296.0 * 2.0 - 1.0;
        x[static_cast<size_t>(i)] = static_cast<float>(0.3 * std::sin(2.0 * pi * 150.0 * t) * (0.6 + 0.4 * std::sin(2.0 * pi * 3.0 * t)) + 0.08 * noise);
    }
    return x;
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        namespace cli = minitts::cli;
        const auto arg = [&](const char * name) { return cli::find_arg(argc, argv, name); };
        const bool offline = cli::has_arg(argc, argv, "--offline");
        const double seconds = std::stod(arg("--seconds").value_or("10"));
        const size_t push = std::stoul(arg("--push").value_or("480"));
        engine::models::voice_isolator::Options options;
        options.block_frames = cli::parse_int_arg(argc, argv, "--block", static_cast<int>(options.block_frames));
        options.lookahead_frames = cli::parse_int_arg(argc, argv, "--lookahead", static_cast<int>(options.lookahead_frames));
        options.chunk_seconds = std::stod(arg("--chunk").value_or("0"));
        options.weight_storage = engine::assets::parse_tensor_storage_type(arg("--weights").value_or("native"));
        if (const auto cache = arg("--cache")) {
            options.attention_cache_storage = engine::assets::parse_tensor_storage_type(*cache);
        }
        engine::core::BackendConfig backend;
        backend.type = cli::parse_backend(arg("--backend").value_or("cpu"));
        backend.threads = cli::parse_int_arg(argc, argv, "--threads",
            static_cast<int>(std::max(1U, std::thread::hardware_concurrency() / 2)));
        // denoiser.safetensors or denoiser_realtime.safetensors of the release.
        const auto model_path = arg("--model");
        if (!model_path) {
            throw std::runtime_error("--model <checkpoint.safetensors> is required");
        }
        cli::require_known_args(argc, argv);
        auto start = Clock::now();
        engine::core::ExecutionContext execution(backend);
        const auto model = engine::models::voice_isolator::Model::load_from_safetensors(*model_path, execution, options);
        std::printf("load: %.2f s, threads %d\n", seconds_since(start), backend.threads);
        const auto input = speech_like(static_cast<int64_t>(seconds * 48000.0));
        if (offline || !model.streams()) {
            model.isolate_mono_48k(std::vector<float>(input.begin(), input.begin() + std::min<size_t>(input.size(), 48000)));
            start = Clock::now();
            const auto output = model.isolate_mono_48k(input);
            const double elapsed = seconds_since(start);
            std::printf("offline: %.1f s of audio in %.2f s, RTF %.3f\n", seconds, elapsed, elapsed / seconds);
            return output.samples.size() == input.size() ? 0 : 1;
        }
        auto session = model.create_streaming_session();
        // Warm the caches and weights before timing.
        session->push(std::vector<float>(input.begin(), input.begin() + std::min<size_t>(input.size(), 48000)));
        session->reset();
        std::vector<double> latencies;
        size_t produced = 0;
        start = Clock::now();
        for (size_t first = 0; first < input.size(); first += push) {
            const auto t = Clock::now();
            produced += session->push(input.data() + first, std::min(push, input.size() - first)).size();
            latencies.push_back(seconds_since(t) * 1000.0);
        }
        produced += session->flush().size();
        const double elapsed = seconds_since(start);
        std::sort(latencies.begin(), latencies.end());
        auto at = [&](double q) { return latencies[static_cast<size_t>(q * static_cast<double>(latencies.size() - 1))]; };
        std::printf("stream: look-ahead %lld, block %lld, push %zu samples (%.1f ms), algorithmic latency %.1f ms\n",
            static_cast<long long>(model.lookahead_frames()), static_cast<long long>(options.block_frames), push,
            static_cast<double>(push) / 48.0, static_cast<double>(session->latency_samples()) / 48.0);
        std::printf("stream: %.1f s of audio in %.2f s, RTF %.3f; push ms p50 %.2f p95 %.2f max %.2f\n", seconds, elapsed,
            elapsed / seconds, at(0.5), at(0.95), latencies.back());
        return produced == input.size() ? 0 : 1;
    } catch (const std::exception & ex) {
        std::cerr << "voice_isolator_warm_bench failed: " << ex.what() << "\n";
        return 1;
    }
}
