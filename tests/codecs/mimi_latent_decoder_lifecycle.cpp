#include "app/cli/args.h"
#include "engine/framework/codecs/mimi_codec_runtime.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/debug/profiler.h"
#include "engine/models/pocket_tts_small/assets.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void check_lifecycle(const engine::core::ExecutionContext & execution,
    const engine::codecs::MimiLatentDecoderConfig & config,
    const engine::codecs::MimiLatentDecoderWeights & weights) {
    using engine::codecs::MimiLatentDecoderRuntime;
    const engine::codecs::MimiLatentDecoderExecutionConfig settings{
        32ull << 20, 96ull << 20, 64ull << 20, 4, 90, 900, "mimi.lifecycle"};
    MimiLatentDecoderRuntime reused(execution, config, weights, settings);
    for (const bool full : {false, true}) {
        for (const int64_t frames : {1, 13, 56, 13}) {
            std::vector<float> input(static_cast<size_t>(frames * config.latent_size));
            for (size_t i = 0; i < input.size(); ++i) input[i] = .3F * std::sin(static_cast<float>(i) * .173F);
            MimiLatentDecoderRuntime fresh(execution, config, weights, settings);
            const auto expected = fresh.decode(input, frames, full);
            if (expected.size() != static_cast<size_t>(frames * 1920) ||
                !std::all_of(expected.begin(), expected.end(), [](float value) { return std::isfinite(value); })) {
                throw std::runtime_error("Invalid offline waveform");
            }
            if (reused.decode(input, frames, full) != expected) throw std::runtime_error("Cached offline decode differs");
            auto changed = input;
            for (auto & value : changed) value *= -.7F;
            if (reused.decode(changed, frames, full) == expected) throw std::runtime_error("Offline decoder ignored input");
            if (reused.decode(input, frames, full) != expected) throw std::runtime_error("Offline state leaked between requests");
        }
    }
    // Cross the history-fill boundary and exercise the saturated mask as well.
    std::vector<std::vector<float>> inputs(20, std::vector<float>(static_cast<size_t>(config.latent_size)));
    for (size_t frame = 0; frame < inputs.size(); ++frame) {
        for (size_t i = 0; i < inputs[frame].size(); ++i) inputs[frame][i] = .2F * std::sin(static_cast<float>(frame * config.latent_size + i));
    }
    MimiLatentDecoderRuntime fresh(execution, config, weights, settings);
    std::vector<std::vector<float>> expected;
    for (const auto & input : inputs) expected.push_back(fresh.decode_streaming(input));
    for (int cycle = 0; cycle < 3; ++cycle) {
        if (cycle == 2) reused.clear_runtime_cache();
        reused.reset_streaming();
        for (size_t frame = 0; frame < inputs.size(); ++frame) {
            if (frame == 1) {
                // Offline execution must not overwrite the active stream's KV/history.
                reused.decode(inputs[0], 1, true);
            }
            if (reused.decode_streaming(inputs[frame]) != expected[frame]) {
                throw std::runtime_error("Streaming reset/interleaving changed output");
            }
        }
    }
    if (execution.uses_host_graph_plan()) {
        // A causal decoder's valid samples cannot depend on how the two
        // bounded tail stages partition frames. Use non-aligned partitions
        // to catch histories incorrectly advanced by intermediate zero padding.
        std::vector<float> latent(static_cast<size_t>(13 * config.latent_size));
        for (size_t i = 0; i < latent.size(); ++i) latent[i] = .3F * std::sin(static_cast<float>(i) * .173F);
        MimiLatentDecoderRuntime reference(execution, config, weights, settings);
        const auto expected = reference.decode(latent, 13, true);
        for (const auto & partition : {std::pair<int64_t, int64_t>{4, 900}, {7, 97}, {13, 11}}) {
            auto alternate = settings;
            alternate.tail_chunk_frames = partition.first;
            alternate.final_stage_chunk_frames = partition.second;
            MimiLatentDecoderRuntime decoder(execution, config, weights, alternate);
            const auto actual = decoder.decode(latent, 13, true);
            if (actual.size() != expected.size()) throw std::runtime_error("Tail partition changed sample count");
            double aa = 0, bb = 0, ab = 0;
            for (size_t i = 0; i < actual.size(); ++i) {
                if (!std::isfinite(actual[i])) throw std::runtime_error("Nonfinite partitioned decoder output");
                aa += double(actual[i]) * actual[i];
                bb += double(expected[i]) * expected[i];
                ab += double(actual[i]) * expected[i];
            }
            if (!(ab / std::sqrt(aa * bb) > .99999))
                throw std::runtime_error("Tail partition changed the waveform");
        }
    }
    auto block = inputs[0];
    block.insert(block.end(), inputs[1].begin(), inputs[1].end());
    bool rejected = false;
    try { reused.decode_streaming(block); }
    catch (const std::invalid_argument &) { rejected = true; }
    catch (const std::runtime_error &) { rejected = true; }
    if (!rejected) throw std::runtime_error("Changing stream block size without reset was accepted");
    reused.reset_streaming();
    MimiLatentDecoderRuntime block_reference(execution, config, weights, settings);
    if (reused.decode_streaming(block) != block_reference.decode_streaming(block)) {
        throw std::runtime_error("Reset did not permit a new block size");
    }
}

} // namespace

int main(int argc, char ** argv) {
    try {
        const auto model = minitts::cli::find_arg(argc, argv, "--model").value();
        const auto backend = minitts::cli::find_arg(argc, argv, "--backend").value_or("cuda");
        const auto log = minitts::cli::find_arg(argc, argv, "--log-file").value();
        if (!minitts::cli::has_arg(argc, argv, "--log")) throw std::invalid_argument("--log required");
        minitts::cli::require_known_args(argc, argv);
        engine::debug::configure_logging({true, log});
        const auto started = std::chrono::steady_clock::now();
        engine::core::ExecutionContext execution({minitts::cli::parse_backend(backend), 0, 8});
        {
            auto assets = engine::models::pocket_tts_small::load_assets(model);
            auto weights = engine::models::pocket_tts_small::load_pocket_tts_small_backend_weights(*assets,
                execution.backend(), execution.backend_type(), engine::assets::TensorStorageType::Native,
                engine::assets::TensorStorageType::Native, 64ull << 20, 64ull << 20, 64ull << 20);
            check_lifecycle(execution, assets->model_config.decoder, weights->mimi_decoder);
            std::cout << "PocketTTS Small lifecycle PASS\n";
        }
        engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(started));
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
