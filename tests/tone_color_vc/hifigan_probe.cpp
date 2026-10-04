#include "engine/framework/audio/wav_writer.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/model_spec/package.h"
#include "engine/framework/modules/vocoders/hifigan_vocoder.h"

#include <fstream>
#include <iostream>
#include <random>
#include <algorithm>

int main(int argc, char ** argv) {
    try {
        if (argc < 9 || std::string(argv[7]) != "--log") {
            throw std::runtime_error("usage: tone_color_hifigan_probe MODEL FRAMES THREADS REPEATS cpu|cuda OUTPUT.wav --log LOG [LATENTS.f32] [--optimize] [--f16] [--check-reuse]");
        }
        engine::debug::configure_logging({true, argv[8]});
        const auto frames = std::stoll(argv[2]);
        const auto repeats = std::stoi(argv[4]);
        if (frames <= 0 || repeats <= 0) throw std::runtime_error("frames and repeats must be positive");
        engine::core::BackendConfig backend;
        backend.threads = std::stoi(argv[3]);
        const std::string backend_name = argv[5];
        if (backend_name == "cpu") backend.type = engine::core::BackendType::Cpu;
        else if (backend_name == "cuda") backend.type = engine::core::BackendType::Cuda;
        else throw std::runtime_error("unsupported backend");
        auto resources = engine::model_spec::load_resource_bundle_for_family(argv[1], "tone_color_vc");
        const auto json = resources.parse_json("config");
        const auto & model = json.require("model");
        engine::modules::HifiGanVocoderConfig config;
        config.sampling_rate = json.require("data").require("sampling_rate").as_i64();
        config.num_mels = model.require("inter_channels").as_i64();
        config.upsample_initial_channel = model.require("upsample_initial_channel").as_i64();
        for (const auto & v : model.require("upsample_rates").as_array()) config.upsample_rates.push_back(v.as_i64());
        for (const auto & v : model.require("upsample_kernel_sizes").as_array()) config.upsample_kernel_sizes.push_back(v.as_i64());
        for (const auto & v : model.require("resblock_kernel_sizes").as_array()) config.resblock_kernel_sizes.push_back(v.as_i64());
        for (const auto & row : model.require("resblock_dilation_sizes").as_array()) {
            std::vector<int64_t> dilations;
            for (const auto & v : row.as_array()) dilations.push_back(v.as_i64());
            config.resblock_dilation_sizes.push_back(std::move(dilations));
        }
        config.resblock_kind = model.require("resblock").as_string() == "1"
            ? engine::modules::HifiGanResBlockKind::PairedConv : engine::modules::HifiGanResBlockKind::SingleConv;
        config.conv_post_use_bias = false;
        config.post_leaky_relu_slope = 0.01f;
        config.tensor_prefix = "dec";
        config.global_conditioning.channels = model.require("gin_channels").as_i64();
        config.lower_padded_conv_transpose_as_crop = backend.type == engine::core::BackendType::Cpu;
        std::string latent_path;
        bool check_reuse = false;
        for (int i = 9; i < argc; ++i) {
            const std::string option = argv[i];
            if (option == "--optimize") config.optimize_cpu_graph = true;
            else if (option == "--f16") config.weight_storage_type = engine::assets::TensorStorageType::F16;
            else if (option == "--check-reuse") check_reuse = true;
            else if (option.rfind("--", 0) != 0 && latent_path.empty()) latent_path = option;
            else throw std::runtime_error("unknown probe argument: " + option);
        }
        auto decoder = engine::modules::HifiGanVocoderComponent::load_from_tensor_source(
            resources.open_tensor_source("weights"), backend, config);
        std::vector<float> latent(config.num_mels * frames);
        if (!latent_path.empty()) {
            std::ifstream input(latent_path, std::ios::binary);
            if (!input.read(reinterpret_cast<char *>(latent.data()), latent.size() * sizeof(float)) ||
                input.peek() != std::char_traits<char>::eof()) throw std::runtime_error("latent input size mismatch");
        } else {
            std::mt19937 generator(42);
            std::normal_distribution<float> normal;
            for (auto & v : latent) v = normal(generator);
        }
        engine::debug::trace_log_scalar("hifigan.frames", frames);
        engine::debug::trace_log_scalar("hifigan.captured_input", !latent_path.empty());
        engine::debug::trace_log_scalar("hifigan.optimize_cpu_graph", config.optimize_cpu_graph);
        std::vector<float> conditioning(config.global_conditioning.channels, 0);
        engine::modules::HifiGanVocoderRequest request;
        request.mel = &latent;
        request.frames = frames;
        request.conditioning = &conditioning;
        request.conditioning_frames = 1;
        auto output = decoder.synthesize(request);
        for (int i = 0; i < repeats; ++i) {
            const auto ms = engine::debug::measure_ms([&] { output = decoder.synthesize(request); });
            engine::debug::timing_log_scalar("session.wall_ms", ms);
            engine::debug::timing_log_scalar("session.rtf", ms * config.sampling_rate / (1000.0 * output.samples));
        }
        engine::audio::WavWriteOptions wav_options;
        wav_options.format = engine::audio::WavSampleFormat::Float32;
        engine::audio::write_wav(argv[6], output.sample_rate, 1, output.waveform, wav_options);
        if (check_reuse) {
            const auto original = latent;
            for (auto & v : latent) v *= -0.5f;
            if (decoder.synthesize(request).waveform == output.waveform) {
                throw std::runtime_error("changed latent produced unchanged waveform");
            }
            const int64_t short_frames = std::max<int64_t>(1, frames / 2);
            std::vector<float> short_latent(config.num_mels * short_frames);
            for (int64_t c = 0; c < config.num_mels; ++c) {
                std::copy_n(original.data() + c * frames, short_frames, short_latent.data() + c * short_frames);
            }
            request.frames = short_frames;
            request.mel = &short_latent;
            const auto shorter = decoder.synthesize(request);
            if (shorter.samples * frames != output.samples * short_frames) {
                throw std::runtime_error("shape change produced wrong output length");
            }
            latent = original;
            request.frames = frames;
            request.mel = &latent;
            if (decoder.synthesize(request).waveform != output.waveform) {
                throw std::runtime_error("same input changed after graph reuse/rebuild");
            }
            engine::debug::trace_log_scalar("hifigan.reuse_check", "passed");
        }
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
