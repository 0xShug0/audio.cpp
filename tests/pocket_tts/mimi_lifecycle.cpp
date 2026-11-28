#include "app/cli/args.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/audio/wav_writer.h"
#include "engine/framework/io/safetensors.h"
#include "engine/models/pocket_tts/assets.h"
#include "engine/models/pocket_tts/backend_weights.h"
#include "engine/framework/codecs/mimi_codec_runtime.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
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
        const auto out = minitts::cli::find_arg(argc,argv,"--out-dir");
        minitts::cli::require_known_args(argc,argv);
        if (out) std::filesystem::create_directories(*out);
        engine::debug::configure_logging({true,log});
        const auto started = std::chrono::steady_clock::now();
        auto assets = engine::models::pocket_tts::load_pocket_tts_assets(model,"english");
        engine::core::ExecutionContext execution({minitts::cli::parse_backend(backend),0,8});
        auto weights = engine::models::pocket_tts::load_pocket_tts_backend_weights(*assets,
            execution.backend(),execution.backend_type(),engine::assets::TensorStorageType::Native,
            engine::assets::TensorStorageType::Native,64ull<<20,64ull<<20,64ull<<20);
        const auto & c = assets->model_config;
        engine::codecs::MimiLatentDecoderConfig config{
            c.latent_dim,c.mimi_dim,c.mimi_heads,c.mimi_intermediate_size,c.mimi_layers,c.mimi_encoder_upsample_stride,
            c.mimi_context,{6,5,4},engine::modules::GeluApproximation::ExactErf};
        const engine::codecs::MimiLatentDecoderExecutionConfig settings{
            32ull<<20,96ull<<20,512ull<<20,1,90,900,"pocket_tts.mimi"};
        engine::codecs::MimiLatentDecoderRuntime reused(execution,config,weights->mimi_decoder,settings);
        auto decode = [&](auto & decoder, const std::vector<float> & input, int64_t frames) {
            auto latent = input;
            for (size_t i = 0; i < latent.size(); ++i)
                latent[i] = latent[i] * weights->host.emb_std[i % c.latent_dim] + weights->host.emb_mean[i % c.latent_dim];
            return decoder.decode(latent,frames,true);
        };
        for (const int64_t frames : {13,56,145,56}) {
            std::vector<float> input(static_cast<size_t>(frames*c.latent_dim));
            for (size_t i = 0; i < input.size(); ++i) input[i] = .3f*std::sin(static_cast<float>(i)*.173f);
            engine::codecs::MimiLatentDecoderRuntime fresh(execution,config,weights->mimi_decoder,settings);
            const auto expected = decode(fresh,input,frames);
            if (expected.empty() || !std::all_of(expected.begin(),expected.end(),[](float v) { return std::isfinite(v); }))
                throw std::runtime_error("Fresh decoder produced invalid samples");
            if (decode(reused,input,frames) != expected) throw std::runtime_error("Fresh and reused decoder differ at first decode");
            auto changed = input;
            for (auto & value : changed) value *= -.7f;
            if (decode(reused,changed,frames) == expected) throw std::runtime_error("Decoder ignored changed latents");
            if (decode(reused,input,frames) != expected) throw std::runtime_error("Offline decoder changed after graph reuse");
            if (out) {
                engine::audio::WavWriteOptions options;
                options.format = engine::audio::WavSampleFormat::Float32;
                engine::audio::write_wav(std::filesystem::path(*out) / ("full-" + std::to_string(frames) + ".wav"),
                    24000, 1, expected, options);
            }
            std::cout << "PASS frames=" << frames << '\n';
        }
        std::vector<std::vector<float>> expected_stream;
        std::vector<float> streaming_latents;
        engine::codecs::MimiLatentDecoderRuntime streaming(execution,config,weights->mimi_decoder,settings);
        for (int cycle = 0; cycle < 3; ++cycle) {
            if (cycle == 2) streaming.clear_runtime_cache();
            streaming.reset_streaming();
            for (int frame = 0; frame < 20; ++frame) {
                std::vector<float> latent(static_cast<size_t>(c.latent_dim));
                for (size_t i = 0; i < latent.size(); ++i)
                    latent[i] = .3f * std::sin(static_cast<float>(i + frame * latent.size()) * .173f);
                for (size_t i = 0; i < latent.size(); ++i)
                    latent[i] = latent[i] * weights->host.emb_std[i] + weights->host.emb_mean[i];
                if (cycle == 0) streaming_latents.insert(streaming_latents.end(),latent.begin(),latent.end());
                const auto audio = streaming.decode_streaming(latent);
                if (audio.empty() || !std::all_of(audio.begin(),audio.end(),[](float v) { return std::isfinite(v); }))
                    throw std::runtime_error("Invalid streaming decoder output");
                if (cycle == 0) expected_stream.push_back(audio);
                else if (audio != expected_stream[static_cast<size_t>(frame)])
                    throw std::runtime_error("Streaming output changed after reset or cache recreation");
            }
        }
        if (execution.uses_host_graph_plan()) {
            // A causal decoder's output must not depend on how identical
            // latents are partitioned into calls, including after KV fills.
            std::vector<float> expected;
            for (const auto & audio : expected_stream) expected.insert(expected.end(),audio.begin(),audio.end());
            for (const size_t group : {size_t{2},size_t{5}}) {
                engine::codecs::MimiLatentDecoderRuntime partitioned(execution,config,weights->mimi_decoder,settings);
                std::vector<float> actual;
                const size_t block = group * static_cast<size_t>(c.latent_dim);
                for (size_t i = 0; i < streaming_latents.size(); i += block) {
                    const std::vector<float> input(streaming_latents.begin()+i,streaming_latents.begin()+i+block);
                    const auto audio = partitioned.decode_streaming(input);
                    actual.insert(actual.end(),audio.begin(),audio.end());
                }
                if (actual.size() != expected.size()) throw std::runtime_error("Streaming partition changed sample count");
                double aa = 0, bb = 0, ab = 0;
                for (size_t i = 0; i < actual.size(); ++i) {
                    aa += double(actual[i])*actual[i]; bb += double(expected[i])*expected[i];
                    ab += double(actual[i])*expected[i];
                }
                if (!(ab/std::sqrt(aa*bb) > .99999))
                    throw std::runtime_error("Streaming partition changed causal decoder output");
            }
        }
        if (out) {
            std::vector<float> samples;
            for (const auto & audio : expected_stream) samples.insert(samples.end(), audio.begin(), audio.end());
            engine::audio::WavWriteOptions options;
            options.format = engine::audio::WavSampleFormat::Float32;
            engine::audio::write_wav(std::filesystem::path(*out) / "stream.wav", 24000, 1, samples, options);
            engine::io::SafeTensorWriteEntry input{"latents","F32",{1,20,c.latent_dim},
                std::vector<unsigned char>(streaming_latents.size()*sizeof(float))};
            std::memcpy(input.data.data(),streaming_latents.data(),input.data.size());
            engine::io::write_safetensors_file(std::filesystem::path(*out) / "stream-latents.safetensors",{input});
        }
        std::cout << "PASS streaming reset and cache recreation\n";
        engine::debug::timing_log_scalar("session.wall_ms",engine::debug::elapsed_ms(started));
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
