#include "app/cli/args.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/debug/profiler.h"
#include "engine/models/pocket_tts/assets.h"
#include "engine/models/pocket_tts/backend_weights.h"
#include "engine/models/pocket_tts/audio_decoder.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

// A stateless offline decoder must match a fresh instance after graph reuse,
// changed inputs and shape changes. This model test requires real GGUF weights.
int main(int argc, char ** argv) {
    try {
        const auto model = minitts::cli::find_arg(argc,argv,"--model").value();
        const auto backend = minitts::cli::find_arg(argc,argv,"--backend").value_or("vulkan");
        const auto log = minitts::cli::find_arg(argc,argv,"--log-file").value();
        if (!minitts::cli::has_arg(argc,argv,"--log")) throw std::invalid_argument("--log required");
        minitts::cli::require_known_args(argc,argv);
        engine::debug::configure_logging({true,log});
        const auto started = std::chrono::steady_clock::now();
        auto assets = engine::models::pocket_tts::load_pocket_tts_assets(model,"english");
        engine::core::ExecutionContext execution({minitts::cli::parse_backend(backend),0,4});
        auto weights = engine::models::pocket_tts::load_pocket_tts_backend_weights(*assets,
            execution.backend(),execution.backend_type(),engine::assets::TensorStorageType::Native,
            engine::assets::TensorStorageType::Native,64ull<<20,64ull<<20,64ull<<20);
        const auto & c = assets->model_config;
        engine::models::pocket_tts::MimiDecoderConfig config{
            c.latent_dim,c.mimi_dim,c.mimi_heads,c.mimi_intermediate_size,c.mimi_layers,c.mimi_encoder_upsample_stride};
        engine::models::pocket_tts::AudioDecoder reused(config);
        auto decode = [&](const auto & decoder, const std::vector<float> & input, int64_t frames) {
            return decoder.decode(execution.backend(),4,*assets,*weights,input,frames,
                32ull<<20,96ull<<20,512ull<<20,90,900,true);
        };
        for (const int64_t frames : {13,56,145,56}) {
            std::vector<float> input(static_cast<size_t>(frames*c.latent_dim));
            for (size_t i = 0; i < input.size(); ++i) input[i] = .3f*std::sin(static_cast<float>(i)*.173f);
            engine::models::pocket_tts::AudioDecoder fresh(config);
            const auto expected = decode(fresh,input,frames);
            if (expected.empty() || !std::all_of(expected.begin(),expected.end(),[](float v) { return std::isfinite(v); }))
                throw std::runtime_error("Fresh decoder produced invalid samples");
            if (decode(reused,input,frames) != expected) throw std::runtime_error("Fresh and reused decoder differ at first decode");
            auto changed = input;
            for (auto & value : changed) value *= -.7f;
            if (decode(reused,changed,frames) == expected) throw std::runtime_error("Decoder ignored changed latents");
            if (decode(reused,input,frames) != expected) throw std::runtime_error("Offline decoder changed after graph reuse");
            std::cout << "PASS frames=" << frames << '\n';
        }
        std::vector<std::vector<float>> expected_stream;
        engine::models::pocket_tts::AudioDecoder streaming(config);
        for (int cycle = 0; cycle < 3; ++cycle) {
            if (cycle == 2) streaming.clear_runtime_cache();
            streaming.reset_streaming_state();
            for (int frame = 0; frame < 20; ++frame) {
                std::vector<float> latent(static_cast<size_t>(c.latent_dim));
                for (size_t i = 0; i < latent.size(); ++i)
                    latent[i] = .3f * std::sin(static_cast<float>(i + frame * latent.size()) * .173f);
                const auto audio = streaming.decode_streaming_step(execution.backend(),4,*assets,*weights,
                    latent,32ull<<20,96ull<<20,512ull<<20);
                if (audio.empty() || !std::all_of(audio.begin(),audio.end(),[](float v) { return std::isfinite(v); }))
                    throw std::runtime_error("Invalid streaming decoder output");
                if (cycle == 0) expected_stream.push_back(audio);
                else if (audio != expected_stream[static_cast<size_t>(frame)])
                    throw std::runtime_error("Streaming output changed after reset or cache recreation");
            }
        }
        std::cout << "PASS streaming reset and cache recreation\n";
        engine::debug::timing_log_scalar("session.wall_ms",engine::debug::elapsed_ms(started));
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
