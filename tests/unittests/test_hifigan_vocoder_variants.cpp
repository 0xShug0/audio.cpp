#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/modules/vocoders/hifigan_vocoder.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<float> read_f32_file(const std::filesystem::path & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open input file: " + path.string());
    }
    in.seekg(0, std::ios::end);
    const auto bytes = in.tellg();
    in.seekg(0, std::ios::beg);
    if (bytes < 0 || bytes % static_cast<std::streamoff>(sizeof(float)) != 0) {
        throw std::runtime_error("invalid f32 input file size: " + path.string());
    }
    std::vector<float> values(static_cast<size_t>(bytes / static_cast<std::streamoff>(sizeof(float))));
    in.read(reinterpret_cast<char *>(values.data()), bytes);
    return values;
}

void write_f32_file(const std::filesystem::path & path, const std::vector<float> & values) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("failed to open output file: " + path.string());
    }
    out.write(reinterpret_cast<const char *>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(float)));
}

engine::modules::HifiGanVocoderConfig make_config(const std::string & variant) {
    engine::modules::HifiGanVocoderConfig config;
    config.sampling_rate = 24000;
    config.num_mels = 3;
    config.upsample_initial_channel = 8;
    config.output_channels = 1;
    config.upsample_rates = {2, 2};
    config.upsample_kernel_sizes = {4, 4};
    config.resblock_kernel_sizes = {3, 5};
    config.resblock_dilation_sizes = {{1, 2}, {1, 2}};
    config.weight_storage_type = engine::assets::TensorStorageType::F32;
    if (variant == "pc_nsf_resblock1") {
        config.resblock_kind = engine::modules::HifiGanResBlockKind::PairedConv;
        config.source.enabled = true;
        config.source.harmonic_num = 0;
        config.source.noise_std = 0.0F;
        config.source.voiced_threshold = 0.0F;
    } else if (variant == "vits_resblock2_conditioned") {
        config.resblock_kind = engine::modules::HifiGanResBlockKind::SingleConv;
        config.global_conditioning.channels = 2;
        config.conv_post_use_bias = false;
        config.post_leaky_relu_slope = 0.01F;
    } else {
        throw std::runtime_error("unknown HiFi-GAN variant test: " + variant);
    }
    return config;
}

int64_t parse_i64(const char * value, const char * name) {
    try {
        return std::stoll(value);
    } catch (...) {
        throw std::runtime_error(std::string("invalid integer for ") + name + ": " + value);
    }
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        if (argc != 7) {
            std::cerr << "usage: hifigan_vocoder_variant_parity VARIANT WEIGHTS SAFETENSORS_DIR FRAMES COND_FRAMES OUT\n";
            return 2;
        }
        const std::string variant = argv[1];
        const auto weights_path = std::filesystem::path(argv[2]);
        const auto case_dir = std::filesystem::path(argv[3]);
        const int64_t frames = parse_i64(argv[4], "frames");
        const int64_t conditioning_frames = parse_i64(argv[5], "conditioning_frames");
        const auto out_path = std::filesystem::path(argv[6]);

        const auto source = engine::assets::open_tensor_source(weights_path);
        const auto config = make_config(variant);
        auto component = engine::modules::HifiGanVocoderComponent::load_from_tensor_source(
            source,
            engine::core::BackendConfig{engine::core::BackendType::Cuda, 0, 1},
            config);

        const auto mel = read_f32_file(case_dir / "mel.f32");
        engine::modules::HifiGanVocoderRequest request;
        request.mel = &mel;
        request.frames = frames;
        std::vector<float> f0;
        std::vector<float> conditioning;
        if (config.source.enabled) {
            f0 = read_f32_file(case_dir / "f0.f32");
            request.f0 = &f0;
        }
        if (config.global_conditioning.channels > 0) {
            conditioning = read_f32_file(case_dir / "conditioning.f32");
            request.conditioning = &conditioning;
            request.conditioning_frames = conditioning_frames;
        }
        const auto output = component.synthesize(request);
        write_f32_file(out_path, output.waveform);
        std::cout << "variant=" << variant << " samples=" << output.waveform.size() << "\n";
        return 0;
    } catch (const std::exception & e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
}
