#include "app/cli/args.h"
#include "engine/community_models/realtime_reuse/frontend.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/io/safetensors.h"

#include <cmath>
#include <iostream>

int main(int argc, char ** argv) {
    try {
        const auto model = minitts::cli::find_arg(argc, argv, "--model").value();
        const auto reference = minitts::cli::find_arg(argc, argv, "--reference").value();
        const auto backend = minitts::cli::find_arg(argc, argv, "--backend").value_or("cuda");
        const auto log = minitts::cli::find_arg(argc, argv, "--log-file").value();
        if (!minitts::cli::has_arg(argc, argv, "--log")) throw std::runtime_error("--log is required");
        minitts::cli::require_known_args(argc, argv);
        engine::debug::configure_logging({true, log});
        auto tensor_source = engine::assets::open_tensor_source(reference);
        auto index = engine::io::load_safetensors_index(reference);
        const auto shape = index.tensors.at("features").shape;
        const auto samples = index.tensors.at("waveform").shape[0];
        auto waveform = tensor_source->require_f32("waveform", {samples});
        auto expected = tensor_source->require_f32("expected", {samples});
        auto features = tensor_source->require_f32("features", shape);
        const int sr = std::stoi(index.metadata.at("sample_rate"));
        auto spectrum = engine::models::realtime_reuse::analyze_spectrum(waveform, sr, 8);
        double max_abs = 0;
        size_t jumps = 0;
        for (size_t i = 0; i < features.size(); ++i) {
            max_abs = std::max(max_abs, std::abs(double(features[i]) - spectrum.features[i]));
            jumps += std::abs(features[i] - spectrum.features[i]) > 1;
        }
        engine::debug::trace_log_scalar("realtime_reuse.reference.feature_max_abs", max_abs);
        engine::debug::trace_log_scalar("realtime_reuse.reference.feature_jumps", jumps);
        engine::debug::trace_log_f32("realtime_reuse.reference.features", {shape[0], shape[1], shape[2]}, features);
        engine::debug::trace_log_f32("realtime_reuse.frontend.features", {shape[0], shape[1], shape[2]}, spectrum.features);
        engine::core::ExecutionContext execution({minitts::cli::parse_backend(backend), 0, 8});
        engine::models::realtime_reuse::Config config;
        config.exit_layer = std::stoi(index.metadata.at("exit_layer"));
        config.lookahead_frames = std::stoi(index.metadata.at("lookahead_frames"));
        engine::models::realtime_reuse::Runtime runtime(engine::models::realtime_reuse::load_assets(model), execution,
            engine::assets::TensorStorageType::Native, config);
        const auto started = std::chrono::steady_clock::now();
        auto heads = runtime.infer(features, shape[1], shape[2], false);
        auto output = engine::models::realtime_reuse::synthesize_spectrum(spectrum, heads, 8);
        double aa = 0, bb = 0, ab = 0;
        for (size_t i = 0; i < output.size(); ++i) {
            aa += double(output[i]) * output[i];
            bb += double(expected[i]) * expected[i];
            ab += double(output[i]) * expected[i];
        }
        const double cosine = ab / std::sqrt(aa * bb);
        engine::debug::trace_log_scalar("realtime_reuse.reference.waveform_cosine", cosine);
        engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(started));
        if (!(cosine > .99999)) throw std::runtime_error("Aligned-input waveform parity failed");
        std::cout << "Real-time RE-USE aligned-input waveform parity passed\n";
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
