#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/core/execution_context.h"
#include "engine/models/xtts_v2/assets.h"
#include "engine/models/xtts_v2/audio_features.h"
#include "engine/models/xtts_v2/conditioning.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>

int main(int argc, char ** argv) try {
    if (argc != 3) {
        std::cerr << "usage: xtts_v2_conditioning_probe <model-dir-or-gguf> <reference.wav>\n";
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
    double sum = std::accumulate(latent.values.begin(), latent.values.end(), 0.0);
    double sq = 0.0;
    for (float value : latent.values) sq += static_cast<double>(value) * value;
    std::cout << std::setprecision(10)
              << "{\"mel_frames\":" << mel.frames
              << ",\"frames\":" << latent.frames
              << ",\"dims\":" << latent.dims
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
