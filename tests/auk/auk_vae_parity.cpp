// Stage 1 of the AuK port: does the framework's BigVGAN reproduce AuK's VAE decoder?
//
// AuK's decoder (bigvgan_flow_vae.py:inference_from_latents) is conv_pre -> per-stage
// upsample + AMP resblocks averaged over the kernel set -> activation_post -> conv_post
// -> clamp. That is what framework BigVGAN already builds, so this test is the question
// "is it the same decoder" asked as a diff rather than as a listening session.
//
// ⚠ The latent fixture is in NORMALIZED space, as the model stores it. Denormalizing is
// a step of the model, not of the fixture: latent * sqrt(global_log_std) + global_mean.
// The buffer is named log_std and used as a variance -- that is AuK's own naming, kept
// here so the two sides stay comparable.
//
//   auk_vae_parity --model auk_vae_raw.gguf --latent stage1_clean_latent.f32
//                  --reference stage1_clean_wav.f32 [--backend cpu] [--min-cosine 0.9999]

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/modules/vocoders/bigvgan_vocoder.h"
#include "engine/models/auk/vae_encoder.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string arg_value(int argc, char ** argv, const std::string & name, const std::string & fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) return argv[i + 1];
    }
    return fallback;
}

double double_arg(int argc, char ** argv, const std::string & name, double fallback) {
    return std::stod(arg_value(argc, argv, name, std::to_string(fallback)));
}

int int_arg(int argc, char ** argv, const std::string & name, int fallback) {
    return std::stoi(arg_value(argc, argv, name, std::to_string(fallback)));
}

engine::core::BackendType parse_backend(const std::string & value) {
    // Unlike dots_tts_vocoder_parity, CPU is allowed and is the default: the reference
    // dump is CPU torch, and a CPU-to-CPU diff removes one variable from the first
    // comparison that has ever been made between these two implementations.
    if (value == "cpu") return engine::core::BackendType::Cpu;
    if (value == "cuda") return engine::core::BackendType::Cuda;
    if (value == "vulkan") return engine::core::BackendType::Vulkan;
    if (value == "best") return engine::core::BackendType::BestAvailable;
    throw std::runtime_error("unknown backend: " + value);
}

std::vector<float> read_f32_file(const std::filesystem::path & path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("cannot read " + path.string());
    const auto bytes = static_cast<std::streamoff>(input.tellg());
    if (bytes % static_cast<std::streamoff>(sizeof(float)) != 0) {
        throw std::runtime_error(path.string() + " is not a whole number of floats");
    }
    std::vector<float> values(static_cast<size_t>(bytes) / sizeof(float));
    input.seekg(0);
    input.read(reinterpret_cast<char *>(values.data()), bytes);
    return values;
}

double cosine(const std::vector<float> & lhs, const std::vector<float> & rhs) {
    const size_t count = std::min(lhs.size(), rhs.size());
    double dot = 0.0, ln = 0.0, rn = 0.0;
    for (size_t i = 0; i < count; ++i) {
        dot += double(lhs[i]) * double(rhs[i]);
        ln += double(lhs[i]) * double(lhs[i]);
        rn += double(rhs[i]) * double(rhs[i]);
    }
    return (ln == 0.0 || rn == 0.0) ? 0.0 : dot / (std::sqrt(ln) * std::sqrt(rn));
}

double max_abs_diff(const std::vector<float> & lhs, const std::vector<float> & rhs) {
    const size_t count = std::min(lhs.size(), rhs.size());
    double worst = 0.0;
    for (size_t i = 0; i < count; ++i) {
        worst = std::max(worst, std::fabs(double(lhs[i]) - double(rhs[i])));
    }
    return worst;
}

// The published config.yaml, which the checkpoint loads into with no missing or
// unexpected tensors -- so these are the model's numbers, not a guess at them.
engine::modules::BigVganVocoderConfig auk_vae_config() {
    engine::modules::BigVganVocoderConfig config;
    config.sampling_rate = 24000;
    config.num_mels = 64;                 // latent_dim: the decoder's input is a latent
    config.hop_size = 480;                // prod(downsample_rates) = 2*2*2*3*4*5
    config.upsample_initial_channel = 1536;
    // ⚠ ARBITRARY, and deliberately so. validate_config() demands n_fft and win_size be
    // positive, but they appear nowhere else in bigvgan_vocoder.cpp -- the decode graph
    // never reads them. AuK's decoder consumes a latent, not a mel, so it has no STFT and
    // no honest value to put here. These are whatever satisfies the check; do not read
    // them as AuK's analysis parameters.
    config.n_fft = 1920;
    config.win_size = 1920;
    config.snake_logscale = true;
    config.upsample_rates = {5, 4, 3, 2, 2, 2};
    config.upsample_kernel_sizes = {10, 8, 6, 4, 4, 4};
    config.resblock_kernel_sizes = {3, 7, 11};
    return config;
}

}  // namespace

int main(int argc, char ** argv) try {
    const std::filesystem::path model_path = arg_value(argc, argv, "--model", "");
    const std::filesystem::path latent_path = arg_value(argc, argv, "--latent", "");
    const std::filesystem::path reference_path = arg_value(argc, argv, "--reference", "");
    const double min_cosine = double_arg(argc, argv, "--min-cosine", 0.9999);
    const int threads = int_arg(argc, argv, "--threads", 1);

    if (model_path.empty() || latent_path.empty() || reference_path.empty()) {
        std::cerr << "usage: auk_vae_parity --model <vae.gguf> --latent <f32> "
                     "--reference <f32> [--backend cpu|cuda|best] [--threads N] "
                     "[--min-cosine 0.9999]\n";
        return 2;
    }

    engine::core::BackendConfig backend;
    backend.type = parse_backend(arg_value(argc, argv, "--backend", "cpu"));
    backend.device = 0;
    backend.threads = threads;

    // --mode encode diffs the encoder's latent DISTRIBUTION (mean and log_std
    // concatenated) against a PyTorch dump. The sampled latent cannot be diffed --
    // encoding_and_normalization() draws noise on top of it -- so the distribution is
    // the deterministic thing a port has to reproduce.
    const std::string mode = arg_value(argc, argv, "--mode", "decode");
    if (mode == "encode") {
        auto encode_source = engine::assets::make_prefixed_tensor_source(
            engine::assets::open_tensor_source(model_path), "vae/");
        engine::core::ExecutionContext execution(backend);
        engine::models::auk::AukVaeEncoderConfig encoder_config;
        engine::models::auk::AukVaeEncoder encoder(encoder_config, *encode_source, execution);

        const auto waveform = read_f32_file(latent_path);   // --latent carries the waveform here
        const auto stats = encoder.encode(waveform);
        const auto expected = read_f32_file(reference_path);
        std::cout << "loaded tensors=" << encoder.loaded_tensor_count()
                  << " samples=" << waveform.size()
                  << " -> channels=" << stats.channels << " frames=" << stats.frames << "\n";
        if (stats.values.size() != expected.size()) {
            std::cerr << "FAIL: encoder produced " << stats.values.size()
                      << " values against the reference " << expected.size() << "\n";
            return 1;
        }
        const double cos = cosine(stats.values, expected);
        const double worst = max_abs_diff(stats.values, expected);
        std::cout << std::fixed << std::setprecision(8)
                  << "cosine=" << cos << " max_abs_diff=" << worst << "\n";
        if (cos < min_cosine) {
            std::cerr << "FAIL: cosine " << cos << " is below " << min_cosine << "\n";
            return 1;
        }
        std::cout << "auk vae encode parity OK\n";
        return 0;
    }

    // The GGUF namespaces every tensor as "vae/...", while the component reads bare
    // names -- conv_pre, ups.N.0, resblocks.N, activation_post, conv_post.
    auto source = engine::assets::make_prefixed_tensor_source(
        engine::assets::open_tensor_source(model_path), "vae/");

    const auto config = auk_vae_config();
    const auto latent = read_f32_file(latent_path);
    if (latent.size() % static_cast<size_t>(config.num_mels) != 0) {
        throw std::runtime_error("latent fixture is not a whole number of frames");
    }
    const int64_t frames = static_cast<int64_t>(latent.size()) / config.num_mels;

    // denormalize, then [T, D] -> [D, T]: the fixture is written frame-major by the
    // reference dump, and the decoder wants channel-major.
    const auto global_mean = source->require_f32("global_mean", {config.num_mels});
    const auto global_log_std = source->require_f32("global_log_std", {config.num_mels});
    std::vector<float> input(latent.size(), 0.0F);
    for (int64_t frame = 0; frame < frames; ++frame) {
        for (int64_t channel = 0; channel < config.num_mels; ++channel) {
            const float value = latent[static_cast<size_t>(frame * config.num_mels + channel)];
            const float denorm =
                value * std::sqrt(global_log_std[static_cast<size_t>(channel)]) +
                global_mean[static_cast<size_t>(channel)];
            input[static_cast<size_t>(channel * frames + frame)] = denorm;
        }
    }

    // AuK's VAE is causal: layers.py builds every Conv1d and ConvTranspose1d with
    // causal=True, which is a different padding rule, not a different weight set.
    engine::modules::BigVganGraphOptions options;
    // --causal 0 runs the non-causal path, which exists to be diffed against a PyTorch
    // reference built with causal=False: it separates "the framework's BigVGAN is not
    // AuK's decoder" from "the causal rule is implemented wrongly".
    options.causal = int_arg(argc, argv, "--causal", 1) != 0;
    auto vocoder = engine::modules::BigVganVocoderComponent::load_from_tensor_source(
        source, backend, config, options);
    std::cout << "loaded tensors=" << vocoder.loaded_tensor_count()
              << " parameters=" << vocoder.parameter_count() << "\n";

    const auto decoded = vocoder.synthesize(input, frames);
    const auto reference = read_f32_file(reference_path);

    std::cout << "frames=" << frames << " samples=" << decoded.samples
              << " reference=" << reference.size() << "\n";
    if (decoded.waveform.size() != reference.size()) {
        std::cerr << "FAIL: sample count " << decoded.waveform.size()
                  << " does not match the reference " << reference.size() << "\n";
        return 1;
    }

    const double cos = cosine(decoded.waveform, reference);
    const double worst = max_abs_diff(decoded.waveform, reference);
    std::cout << std::fixed << std::setprecision(8)
              << "cosine=" << cos << " max_abs_diff=" << worst << "\n";
    if (cos < min_cosine) {
        std::cerr << "FAIL: cosine " << cos << " is below " << min_cosine << "\n";
        return 1;
    }
    std::cout << "auk vae parity OK\n";
    return 0;
} catch (const std::exception & error) {
    std::cerr << "FAIL: " << error.what() << "\n";
    return 1;
}
