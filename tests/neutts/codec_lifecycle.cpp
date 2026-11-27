#include "app/cli/args.h"
#include "engine/models/neutts/assets.h"
#include "engine/models/neutts/codec.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/core/execution_context.h"
#include "tests/codecs/lifecycle.h"

#include <memory>
#include <numeric>
#include <functional>

int main(int argc, char ** argv) {
    const auto started = std::chrono::steady_clock::now();
    try {
        using minitts::cli::find_arg;
        engine::debug::configure_logging({true, find_arg(argc, argv, "--log-file").value()});
        if (!minitts::cli::has_arg(argc, argv, "--log")) throw std::runtime_error("--log required");
        // The model adapter supplies only checkpoint bindings and dimensions.
        // Every inference call below goes directly to the framework codec.
        const auto assets = engine::models::neutts::load_neutts_assets(find_arg(argc, argv, "--model").value());
        const auto config = engine::models::neutts::make_neutts_fsq_audio_codec_config(assets->codec);
        engine::core::ExecutionContext execution(minitts::cli::parse_backend_config_from_cli(argc, argv, 8));
        const auto make = [&] {
            return std::make_unique<engine::codecs::FsqAudioCodecDecoderRuntime>(
                config, assets->codec_weights, execution, 32ull << 20, 32ull << 20,
                engine::assets::TensorStorageType::Native, engine::assets::TensorStorageType::Native);
        };
        const auto encode = [](auto & runtime, const auto & codes) { return runtime.decode_head(codes).values; };
        const auto size = static_cast<size_t>(std::accumulate(config.quantization_levels.begin(), config.quantization_levels.end(), int64_t{1}, std::multiplies<int64_t>()));
        std::vector<int32_t> a(128), b(128);
        for (size_t i = 0; i < a.size(); ++i) {
            a[i] = static_cast<int32_t>((i * 997) % size);
            b[i] = static_cast<int32_t>((i * 431 + 17) % size);
        }
        auto c = a;
        c.resize(c.size() / 2);
        check_runtime_lifecycle(make, encode, a, b, c, [](auto & runtime) { runtime.release_runtime_graphs(); });
        engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(started));
        return 0;
    } catch (const std::exception & error) {
        engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(started));
        std::cerr << error.what() << '\n';
        return 1;
    }
}
