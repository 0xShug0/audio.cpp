#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/core/execution_context.h"
#include "engine/models/xtts_v2/assets.h"
#include "engine/models/xtts_v2/audio_features.h"
#include "engine/models/xtts_v2/conditioning.h"
#include "engine/models/xtts_v2/speaker_encoder.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>

int main(int argc, char ** argv) try {
    if (argc != 3 && argc != 4) {
        std::cerr << "usage: xtts_v2_conditioning_probe <model-dir-or-gguf> <reference.wav> [speaker-f32]\n";
        return 2;
    }
    auto assets = engine::models::xtts_v2::load_xtts_v2_assets(argv[1]);
    const auto wav = engine::audio::read_wav_f32(std::filesystem::path(argv[2]));
    engine::runtime::AudioBuffer audio{wav.sample_rate, wav.channels, wav.samples};
    const auto reference = engine::models::xtts_v2::prepare_xtts_v2_reference(audio);
    const size_t chunk_samples = 4U * 22050U;
    const std::vector<float> chunk(
        reference.waveform_22050.begin(),
        reference.waveform_22050.begin() + static_cast<std::ptrdiff_t>(
            std::min(chunk_samples, reference.waveform_22050.size())));

    engine::core::ExecutionContext execution({engine::core::BackendType::Cpu, 0, 4});
    engine::models::xtts_v2::XttsV2ConditioningRuntime runtime(
        *assets, execution, 512U * 1024U * 1024U, 512U * 1024U * 1024U,
        engine::assets::TensorStorageType::Native,
        engine::assets::TensorStorageType::Native);
    const auto mel = engine::models::xtts_v2::compute_xtts_v2_conditioning_mel(
        chunk, runtime.mel_stats(), 4);
    const auto latent = runtime.encode(mel);
    engine::models::xtts_v2::XttsV2SpeakerEncoderRuntime speaker_runtime(
        *assets, execution, 128U * 1024U * 1024U, 512U * 1024U * 1024U,
        engine::assets::TensorStorageType::Native,
        engine::assets::TensorStorageType::Native);
    const auto speaker_mel = engine::models::xtts_v2::compute_xtts_v2_speaker_mel(
        reference.waveform_16000,
        assets->speaker_encoder->require_f32("torch_spec.1.spectrogram.window", {400}),
        assets->speaker_encoder->require_f32("torch_spec.1.mel_scale.fb", {257, 64}),
        4);
    const auto speaker = speaker_runtime.encode(speaker_mel);
    if (argc == 4) {
        std::ofstream output(argv[3], std::ios::binary);
        output.write(reinterpret_cast<const char *>(speaker.values.data()),
                     static_cast<std::streamsize>(speaker.values.size() * sizeof(float)));
        if (!output) throw std::runtime_error("failed to write speaker embedding dump");
        std::ofstream mel_output(std::string(argv[3]) + ".mel", std::ios::binary);
        mel_output.write(reinterpret_cast<const char *>(speaker_mel.values.data()),
                         static_cast<std::streamsize>(speaker_mel.values.size() * sizeof(float)));
        if (!mel_output) throw std::runtime_error("failed to write speaker mel dump");
    }
    double speaker_sq = 0.0;
    for (float value : speaker.values) speaker_sq += static_cast<double>(value) * value;
    double sum = std::accumulate(latent.values.begin(), latent.values.end(), 0.0);
    double sq = 0.0;
    for (float value : latent.values) sq += static_cast<double>(value) * value;
    std::cout << std::setprecision(10)
              << "{\"mel_frames\":" << mel.frames
              << ",\"frames\":" << latent.frames
              << ",\"dims\":" << latent.dims
              << ",\"speaker_frames\":" << speaker_mel.frames
              << ",\"speaker_norm\":" << std::sqrt(speaker_sq)
              << ",\"sum\":" << sum
              << ",\"rms\":" << std::sqrt(sq / latent.values.size())
              << ",\"first\":[";
    for (size_t i = 0; i < std::min<size_t>(16, latent.values.size()); ++i) {
        if (i) std::cout << ',';
        std::cout << latent.values[i];
    }
    std::cout << "]}\n";
    return 0;
} catch (const std::exception & error) {
    std::cerr << "xtts_v2_conditioning_probe failed: " << error.what() << '\n';
    return 1;
}
