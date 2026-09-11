#include "engine/framework/audio/utility_api.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/deepfilternet2.h"
#include "engine/framework/audio/flashsr.h"
#include "engine/framework/audio/gtcrn.h"
#include "engine/framework/audio/rnnoise.h"
#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/audio/wav_writer.h"
#include "engine/framework/audio/zipenhancer.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>
#include <string_view>

namespace engine::audio {
namespace {

std::filesystem::path require_model_dir(const AudioUtilityPaths & paths, const char * name) {
    const auto dir = paths.assets_root / name;
    if (!std::filesystem::is_directory(dir)) {
        throw std::runtime_error("audio utility assets directory is missing: " + dir.string());
    }
    return dir;
}

bool is_wav_file(const std::filesystem::path & path) {
    auto ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return ext == ".wav";
}

std::vector<std::filesystem::path> sorted_wav_files(const std::filesystem::path & input_dir) {
    if (!std::filesystem::is_directory(input_dir)) {
        throw std::runtime_error("audio utility input directory is missing: " + input_dir.string());
    }
    std::vector<std::filesystem::path> files;
    for (const auto & entry : std::filesystem::directory_iterator(input_dir)) {
        if (entry.is_regular_file() && is_wav_file(entry.path())) {
            files.push_back(entry.path());
        }
    }
    std::sort(files.begin(), files.end());
    if (files.empty()) {
        throw std::runtime_error("audio utility input directory contains no WAV files: " + input_dir.string());
    }
    return files;
}

std::filesystem::path output_path_for(
    const std::filesystem::path & input_file,
    const std::filesystem::path & output_dir,
    const std::string & suffix) {
    std::filesystem::create_directories(output_dir);
    return output_dir / (input_file.stem().string() + suffix + ".wav");
}

void create_output_parent(const std::filesystem::path & output_wav) {
    const auto parent = output_wav.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
}

std::vector<float> read_mono_resampled(const std::filesystem::path & path, int sample_rate) {
    return read_wav_f32_as_mono_linear_resampled(path, sample_rate);
}

[[noreturn]] void throw_unsupported_model(std::string_view task, std::string_view model, std::string_view valid_models) {
    throw std::runtime_error(
        "unsupported audio utility model for " + std::string(task) + ": " + std::string(model) +
        " (valid: " + std::string(valid_models) + ")");
}

std::filesystem::path gtcrn_checkpoint_for(const AudioUtilityPaths & paths, std::string_view model) {
    const auto dir = require_model_dir(paths, "gtcrn");
    if (model == "gtcrn" || model == "gtcrn_streaming") {
        return dir / "gtcrn_streaming.safetensors";
    }
    if (model == "gtcrn_dns3") {
        return dir / "gtcrn_dns3.safetensors";
    }
    if (model == "gtcrn_vctk") {
        return dir / "gtcrn_vctk.safetensors";
    }
    throw_unsupported_model("denoise", model, "deepfilternet2, gtcrn, gtcrn_dns3, gtcrn_vctk, gtcrn_streaming, rnnoise, zipenhancer");
}

bool is_gtcrn_model(std::string_view model) {
    return model == "gtcrn" || model == "gtcrn_dns3" || model == "gtcrn_vctk" || model == "gtcrn_streaming";
}

}  // namespace

void denoise_file(
    const std::filesystem::path & input_wav,
    const std::filesystem::path & output_wav,
    std::string_view model,
    const AudioUtilityPaths & paths) {
    if (model == "deepfilternet2") {
        const auto denoiser = DeepFilterNet2Model::load_from_directory(require_model_dir(paths, "deepfilternet2"), paths.backend);
        const auto input = read_mono_resampled(input_wav, 48000);
        const auto output = denoiser.run_mono_48k(input);
        create_output_parent(output_wav);
        write_pcm16_wav(output_wav, output.sample_rate, 1, output.samples);
        return;
    }
    if (model == "rnnoise") {
        const auto denoiser = RnnoiseModel::load_from_safetensors(
            require_model_dir(paths, "rnnoise") / "rnnoise10Gb_15.safetensors",
            paths.backend);
        const auto input = read_mono_resampled(input_wav, 48000);
        const auto output = denoiser.process_mono_48k(input);
        create_output_parent(output_wav);
        write_pcm16_wav(output_wav, output.sample_rate, 1, output.samples);
        return;
    }
    if (model == "zipenhancer") {
        const auto denoiser = ZipEnhancerModel::load_from_directory(require_model_dir(paths, "zipenhancer"), paths.backend);
        const auto input = read_mono_resampled(input_wav, 16000);
        const auto output = denoiser.denoise_mono_16k(input);
        create_output_parent(output_wav);
        write_pcm16_wav(output_wav, output.sample_rate, 1, output.samples);
        return;
    }
    if (is_gtcrn_model(model)) {
        const auto denoiser = GTCRNModel::load_from_safetensors(gtcrn_checkpoint_for(paths, model), paths.backend);
        const auto input = read_mono_resampled(input_wav, 16000);
        const auto output = denoiser.denoise_mono_16k(input);
        create_output_parent(output_wav);
        write_pcm16_wav(output_wav, output.sample_rate, 1, output.samples);
        return;
    }
    throw_unsupported_model("denoise", model, "deepfilternet2, gtcrn, gtcrn_dns3, gtcrn_vctk, gtcrn_streaming, rnnoise, zipenhancer");
}

AudioUtilityBatchResult denoise_directory(
    const std::filesystem::path & input_dir,
    const std::filesystem::path & output_dir,
    std::string_view model,
    const AudioUtilityPaths & paths) {
    if (model != "deepfilternet2" && model != "rnnoise" && model != "zipenhancer" && !is_gtcrn_model(model)) {
        throw_unsupported_model("denoise", model, "deepfilternet2, gtcrn, gtcrn_dns3, gtcrn_vctk, gtcrn_streaming, rnnoise, zipenhancer");
    }
    AudioUtilityBatchResult result;
    if (model == "deepfilternet2") {
        const auto denoiser = DeepFilterNet2Model::load_from_directory(require_model_dir(paths, "deepfilternet2"), paths.backend);
        for (const auto & input_file : sorted_wav_files(input_dir)) {
            const auto output_file = output_path_for(input_file, output_dir, "_deepfilternet2");
            const auto input = read_mono_resampled(input_file, 48000);
            const auto output = denoiser.run_mono_48k(input);
            write_pcm16_wav(output_file, output.sample_rate, 1, output.samples);
            result.outputs.push_back(output_file);
        }
        return result;
    }
    if (model == "zipenhancer") {
        const auto denoiser = ZipEnhancerModel::load_from_directory(require_model_dir(paths, "zipenhancer"), paths.backend);
        for (const auto & input_file : sorted_wav_files(input_dir)) {
            const auto output_file = output_path_for(input_file, output_dir, "_zipenhancer");
            const auto input = read_mono_resampled(input_file, 16000);
            const auto output = denoiser.denoise_mono_16k(input);
            write_pcm16_wav(output_file, output.sample_rate, 1, output.samples);
            result.outputs.push_back(output_file);
        }
        return result;
    }
    if (is_gtcrn_model(model)) {
        const auto denoiser = GTCRNModel::load_from_safetensors(gtcrn_checkpoint_for(paths, model), paths.backend);
        for (const auto & input_file : sorted_wav_files(input_dir)) {
            const auto output_file = output_path_for(input_file, output_dir, "_" + std::string(model));
            const auto input = read_mono_resampled(input_file, 16000);
            const auto output = denoiser.denoise_mono_16k(input);
            write_pcm16_wav(output_file, output.sample_rate, 1, output.samples);
            result.outputs.push_back(output_file);
        }
        return result;
    }
    const auto denoiser = RnnoiseModel::load_from_safetensors(
        require_model_dir(paths, "rnnoise") / "rnnoise10Gb_15.safetensors",
        paths.backend);
    for (const auto & input_file : sorted_wav_files(input_dir)) {
        const auto output_file = output_path_for(input_file, output_dir, "_rnnoise");
        const auto input = read_mono_resampled(input_file, 48000);
        const auto output = denoiser.process_mono_48k(input);
        write_pcm16_wav(output_file, output.sample_rate, 1, output.samples);
        result.outputs.push_back(output_file);
    }
    return result;
}

void super_resolve_file(
    const std::filesystem::path & input_wav,
    const std::filesystem::path & output_wav,
    std::string_view model,
    const AudioUtilityPaths & paths) {
    if (model != "flashsr") {
        throw_unsupported_model("super_resolve", model, "flashsr");
    }
    const auto super_resolver = FlashSrModel::load_from_directory(require_model_dir(paths, "flashsr"), paths.backend);
    const auto input = read_mono_resampled(input_wav, 16000);
    const auto output = super_resolver.super_resolve_mono_16k(input);
    create_output_parent(output_wav);
    write_pcm16_wav(output_wav, output.sample_rate, 1, output.samples);
}

AudioUtilityBatchResult super_resolve_directory(
    const std::filesystem::path & input_dir,
    const std::filesystem::path & output_dir,
    std::string_view model,
    const AudioUtilityPaths & paths) {
    if (model != "flashsr") {
        throw_unsupported_model("super_resolve", model, "flashsr");
    }
    const auto super_resolver = FlashSrModel::load_from_directory(require_model_dir(paths, "flashsr"), paths.backend);
    AudioUtilityBatchResult result;
    for (const auto & input_file : sorted_wav_files(input_dir)) {
        const auto output_file = output_path_for(input_file, output_dir, "_flashsr");
        const auto input = read_mono_resampled(input_file, 16000);
        const auto output = super_resolver.super_resolve_mono_16k(input);
        write_pcm16_wav(output_file, output.sample_rate, 1, output.samples);
        result.outputs.push_back(output_file);
    }
    return result;
}

}  // namespace engine::audio
