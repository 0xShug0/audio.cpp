#include "engine/community_models/glm_tts/assets.h"
#include "engine/community_models/glm_tts/frontend.h"
#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/modules/speech_encoders/campplus_encoder.h"

#include <exception>
#include <iomanip>
#include <iostream>

int main(int argc, char ** argv) try {
    if (argc != 3) {
        std::cerr
            << "usage: glm_tts_campplus_probe <model-path> "
               "<reference.wav>\n";
        return 2;
    }

    auto assets =
        engine::models::glm_tts::load_glm_tts_assets(argv[1]);
    const auto wav =
        engine::audio::read_wav_f32(std::filesystem::path(argv[2]));
    engine::runtime::AudioBuffer audio;
    audio.sample_rate = wav.sample_rate;
    audio.channels = wav.channels;
    audio.samples = wav.samples;
    const auto fbank =
        engine::models::glm_tts::compute_glm_tts_campplus_fbank(audio);

    engine::modules::CampplusEncoderConfig config;
    config.feat_dim = 80;
    config.embedding_size = 192;
    config.weight_storage_type =
        engine::assets::TensorStorageType::Native;
    config.normalize_partial_segment_by_full_length = true;
    auto encoder =
        engine::modules::CampplusEncoderComponent::
            load_from_tensor_source(
                assets->campplus_weights,
                {engine::core::BackendType::Cpu, 0, 8},
                config);
    const auto output = encoder.embed_from_features(
        fbank.values, fbank.frames, fbank.dims);

    std::cout << std::setprecision(9);
    std::cout
        << "{\"frames\":" << fbank.frames
        << ",\"embedding\":[";
    for (size_t index = 0; index < output.embedding.size(); ++index) {
        if (index != 0) {
            std::cout << ',';
        }
        std::cout << output.embedding[index];
    }
    std::cout << "]}\n";
    return 0;
} catch (const std::exception & error) {
    std::cerr
        << "glm_tts_campplus_probe failed: "
        << error.what() << '\n';
    return 1;
}
