#include "app/cli/args.h"
#include "engine/community_models/realtime_reuse/frontend.h"
#include "engine/framework/audio/conversion.h"
#include "engine/framework/debug/profiler.h"

#include <algorithm>
#include <cmath>
#include <iostream>

using namespace engine;

std::vector<float> stream(models::realtime_reuse::Runtime & runtime, const std::vector<float> & input,
                          int rate, bool fragmented) {
    models::realtime_reuse::StreamingFrontend frontend(runtime, rate, 8);
    std::vector<float> output;
    bool early_output = false;
    for (size_t offset = 0, index = 0; offset < input.size(); ++index) {
        const size_t pattern[] = {1, 317, 7, 2048, 23};
        const size_t count = std::min(input.size() - offset, fragmented ? pattern[index % 5] : input.size());
        auto part = frontend.push(std::vector<float>(input.begin() + offset, input.begin() + offset + count), false);
        early_output |= !part.empty() && offset + count < input.size();
        output.insert(output.end(), part.begin(), part.end());
        offset += count;
    }
    if (fragmented && !early_output) throw std::runtime_error("Streaming buffered the complete input before emitting audio");
    auto tail = frontend.push({}, true);
    output.insert(output.end(), tail.begin(), tail.end());
    if (output.size() != input.size()) throw std::runtime_error("Streaming changed the sample count");
    for (auto value : output) if (!std::isfinite(value)) throw std::runtime_error("Nonfinite streaming output");
    return output;
}

int main(int argc, char ** argv) {
    try {
        const auto model = minitts::cli::find_arg(argc, argv, "--model").value();
        const auto audio_file = minitts::cli::find_arg(argc, argv, "--audio").value();
        const auto backend = minitts::cli::find_arg(argc, argv, "--backend").value_or("cuda");
        const auto log = minitts::cli::find_arg(argc, argv, "--log-file").value();
        const int layer = minitts::cli::parse_int_arg(argc, argv, "--exit-layer", 8);
        const int lookahead = minitts::cli::parse_int_arg(argc, argv, "--lookahead-frames", 0);
        const int rate = minitts::cli::parse_int_arg(argc, argv, "--sample-rate", 8000);
        const bool all = minitts::cli::has_arg(argc, argv, "--all-configs");
        if (!minitts::cli::has_arg(argc, argv, "--log")) throw std::runtime_error("--log is required");
        minitts::cli::require_known_args(argc, argv);
        debug::configure_logging({true, log});
        const auto wav = audio::read_wav_f32(std::filesystem::path(audio_file));
        const auto input = audio::convert_interleaved_audio_to_mono_torchaudio_sinc_hann_resampled(
            wav.samples, wav.sample_rate, wav.channels, rate);
        auto assets = models::realtime_reuse::load_assets(model);
        core::ExecutionContext execution({minitts::cli::parse_backend(backend), 0, 8});
        const auto started = std::chrono::steady_clock::now();
        for (int exit_layer = all ? 3 : layer; exit_layer <= (all ? 12 : layer); ++exit_layer) {
            for (int ahead = all ? 0 : lookahead; ahead <= (all ? 2 : lookahead); ++ahead) {
                models::realtime_reuse::Runtime runtime(assets, execution, assets::TensorStorageType::Native, {exit_layer, ahead});
                const auto whole = stream(runtime, input, rate, false);
                const auto fragmented = stream(runtime, input, rate, true);
                if (whole != fragmented) throw std::runtime_error("Output depends on input chunk boundaries");
                stream(runtime, std::vector<float>(input.begin(), input.begin() + input.size() / 3), rate, true);
                const auto repeated = stream(runtime, input, rate, true);
                if (whole != repeated) throw std::runtime_error("Long-short-long changed streaming output");
                const auto offline = models::realtime_reuse::restore(runtime, input, rate, 8);
                if (offline.size() != input.size()) throw std::runtime_error("Offline changed the sample count");
                double aa = 0, bb = 0, ab = 0;
                // Upstream's online path intentionally omits the first-frame sweep filter.
                for (size_t i = rate / 25; i < whole.size(); ++i) {
                    aa += double(whole[i]) * whole[i];
                    bb += double(offline[i]) * offline[i];
                    ab += double(whole[i]) * offline[i];
                }
                const auto cosine = ab / std::sqrt(aa * bb);
                debug::log_message("exit_layer=" + std::to_string(exit_layer) + " lookahead_frames=" + std::to_string(ahead));
                debug::trace_log_scalar("realtime_reuse.streaming.offline_cosine", cosine);
                if (!(cosine > .99999)) throw std::runtime_error("Streaming/offline waveform parity failed");
            }
        }
        debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(started));
        std::cout << "Streaming boundaries, exact repeatability and long-short-long passed\n";
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
