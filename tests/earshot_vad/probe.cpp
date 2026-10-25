// Dump Earshot VAD per-frame scores and segments as JSON for parity checks.
//
// Usage: earshot_vad_probe <model dir|file> <wav> [--threads N] [--bench N] [--upstream-segments]
//                          [--log-file path]
//
// The wav is converted to 16 kHz mono. `--upstream-segments` disables
// keep_trailing_segment to reproduce upstream earshot::segments() exactly.

#include "engine/framework/audio/conversion.h"
#include "engine/framework/debug/trace.h"
#include "engine/community_models/earshot_vad/runtime.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace ev = engine::community_models::earshot_vad;

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::cerr << "usage: earshot_vad_probe <model> <wav> [--threads N] [--bench N] [--upstream-segments]\n";
        return 2;
    }
    int threads = 1;
    int bench = 0;
    bool upstream_segments = false;
    for (int i = 3; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--threads" && i + 1 < argc) {
            threads = std::atoi(argv[++i]);
        } else if (arg == "--bench" && i + 1 < argc) {
            bench = std::atoi(argv[++i]);
        } else if (arg == "--log-file" && i + 1 < argc) {
            engine::debug::configure_logging({true, std::string(argv[++i])});
        } else if (arg == "--upstream-segments") {
            upstream_segments = true;
        } else {
            std::cerr << "unknown argument: " << arg << "\n";
            return 2;
        }
    }
    try {
        const auto weights = ev::load_earshot_weights_cached(ev::resolve_earshot_assets(argv[1]).checkpoint_path);
        const auto audio = engine::audio::read_wav_f32_as_mono_linear_resampled(argv[2], ev::kSampleRate);
        const auto n = static_cast<int64_t>(audio.size());

        const auto scores = ev::score_earshot_frames(weights, audio.data(), n, threads);

        // Streaming path, frame by frame, must reproduce the batched scores.
        ev::EarshotDetector detector(weights);
        double max_stream_diff = 0.0;
        for (size_t f = 0; f < scores.size(); ++f) {
            const float s = detector.predict(audio.data() + static_cast<int64_t>(f) * ev::kFrameSize);
            max_stream_diff = std::max(max_stream_diff, static_cast<double>(std::abs(s - scores[f])));
        }

        ev::EarshotSegmenterConfig config;
        config.keep_trailing_segment = !upstream_segments;
        const auto segments = ev::segment_earshot_scores(scores, config);

        double offline_ms = 0.0;
        double streaming_ms = 0.0;
        if (bench > 0) {
            auto start = std::chrono::steady_clock::now();
            for (int r = 0; r < bench; ++r) {
                const auto again = ev::score_earshot_frames(weights, audio.data(), n, threads);
                if (again.size() != scores.size()) return 3;
            }
            offline_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() / bench;
            start = std::chrono::steady_clock::now();
            float sink = 0.0f;
            for (int r = 0; r < bench; ++r) {
                detector.reset();
                for (size_t f = 0; f < scores.size(); ++f) {
                    sink += detector.predict(audio.data() + static_cast<int64_t>(f) * ev::kFrameSize);
                }
            }
            streaming_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() / bench;
            if (sink < 0.0f) return 3;
        }

        std::printf("{\"sample_rate\":16000,\"frame_size\":%lld,\"num_samples\":%lld,\"scores\":[",
                    static_cast<long long>(ev::kFrameSize), static_cast<long long>(n));
        for (size_t i = 0; i < scores.size(); ++i) {
            std::printf(i ? ",%.9e" : "%.9e", scores[i]);
        }
        std::printf("],\"segments\":[");
        for (size_t i = 0; i < segments.size(); ++i) {
            std::printf(i ? ",[%lld,%lld]" : "[%lld,%lld]",
                        static_cast<long long>(segments[i].start_frame * ev::kFrameSize),
                        static_cast<long long>(segments[i].end_frame * ev::kFrameSize));
        }
        std::printf("],\"max_stream_diff\":%.9e,\"offline_ms\":%.6f,\"streaming_ms\":%.6f}\n",
                    max_stream_diff, offline_ms, streaming_ms);
    } catch (const std::exception & error) {
        std::cerr << "earshot_vad_probe: " << error.what() << "\n";
        return 1;
    }
    return 0;
}
