#include "engine/community_models/glm_tts/frontend.h"
#include "engine/community_models/mira_tts/speaker_encoder.h"
#include "engine/community_models/kroko_asr/frontend.h"
#include "components/component_weights.h"
#include "engine/models/seed_vc/whisper_content.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/audio/waveform_ops.h"
#include "engine/models/confucius4_tts/audio_features.h"
#include "engine/models/chatterbox/components.h"
#include "engine/models/dramabox/audio_vae.h"
#include "engine/models/dots_tts/audio_features.h"
#include "engine/models/index_tts2/audio_features.h"
#include "engine/models/niagara_asr/frontend.h"
#include "engine/models/qwen3_tts/speaker_encoder.h"
#include "engine/models/seed_vc/audio_features.h"
#include "engine/community_models/vieneu_v3_turbo/speaker_encoder.h"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string argument(int argc, char ** argv, const std::string & name) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) return argv[i + 1];
    }
    throw std::runtime_error("missing required argument: " + name);
}

std::vector<float> waveform(int64_t samples) {
    std::vector<float> result(static_cast<size_t>(samples));
    for (int64_t i = 0; i < samples; ++i) {
        result[static_cast<size_t>(i)] =
            0.27f * std::sin(static_cast<float>(i) * 0.017f) +
            0.13f * std::cos(static_cast<float>(i) * 0.0031f);
    }
    return result;
}

void write(const std::filesystem::path & prefix, const std::vector<float> & values,
           int64_t frames, int64_t bins) {
    if (frames <= 0 || bins <= 0 || values.size() != static_cast<size_t>(frames * bins)) {
        throw std::runtime_error("invalid feature shape");
    }
    std::ofstream data(prefix.string() + ".f32", std::ios::binary | std::ios::trunc);
    if (!data.write(reinterpret_cast<const char *>(values.data()),
                    static_cast<std::streamsize>(values.size() * sizeof(float)))) {
        throw std::runtime_error("feature write failed");
    }
    std::ofstream meta(prefix.string() + ".meta", std::ios::trunc);
    meta << frames << ' ' << bins << '\n';
    if (!meta) throw std::runtime_error("metadata write failed");
    std::cout << prefix.filename() << ": " << frames << " x " << bins << '\n';
}

}  // namespace

int main(int argc, char ** argv) try {
    const std::filesystem::path out = argument(argc, argv, "--out");
    engine::debug::configure_logging({true, argument(argc, argv, "--log")});
    std::filesystem::create_directories(out);
    const auto niagara_assets = engine::models::niagara_asr::load_niagara_asr_assets(
        argument(argc, argv, "--niagara-model"));
    engine::models::niagara_asr::NiagaraFrontend niagara_frontend(niagara_assets);
    for (const int64_t samples : {2048, 22050 + 137}) {
        const auto input = waveform(samples);
        const std::string suffix = std::to_string(samples);

        engine::models::index_tts2::IndexTTS2S2MelConfig index_config;
        const auto index = engine::models::index_tts2::compute_index_tts2_mel_spectrogram(
            input, index_config, 8);
        write(out / ("index_tts2_" + suffix), index.values, index.frames, index.channels);

        for (const bool semantic_source : {true, false}) {
            const auto reference = engine::models::index_tts2::prepare_index_tts2_reference_audio(
                input, 22050, 1, index_config, 8, semantic_source);
            const std::string prefix = "index_reference_" + suffix +
                (semantic_source ? "_semantic" : "_speaker");
            write(out / (prefix + "_target"), reference.waveform_22k,
                  static_cast<int64_t>(reference.waveform_22k.size()), 1);
            write(out / (prefix + "_16k"), reference.waveform_16k,
                  static_cast<int64_t>(reference.waveform_16k.size()), 1);
            write(out / (prefix + "_mel"), reference.mel.values,
                  reference.mel.frames, reference.mel.channels);
            write(out / (prefix + "_campplus"), reference.campplus_fbank.values,
                  reference.campplus_fbank.frames, reference.campplus_fbank.dims);
            write(out / (prefix + "_semantic"), reference.semantic_features.values,
                  reference.semantic_features.frames, reference.semantic_features.dims);
            const std::vector<float> mask(reference.semantic_features.attention_mask.begin(),
                                          reference.semantic_features.attention_mask.end());
            write(out / (prefix + "_mask"), mask, static_cast<int64_t>(mask.size()), 1);
        }

        engine::models::confucius4_tts::ConfuciusAudioConfig confucius_config;
        const auto confucius = engine::models::confucius4_tts::compute_confucius_mel_spectrogram(
            input, confucius_config, 8);
        write(out / ("confucius4_tts_" + suffix), confucius.values,
              confucius.frames, confucius.channels);

        const auto confucius_reference = engine::models::confucius4_tts::prepare_confucius_reference_audio(
            input, 22050, 1, confucius_config, 8);
        const std::string confucius_prefix = "confucius_reference_" + suffix;
        write(out / (confucius_prefix + "_target"), confucius_reference.waveform_target,
              static_cast<int64_t>(confucius_reference.waveform_target.size()), 1);
        write(out / (confucius_prefix + "_16k"), confucius_reference.waveform_16k,
              static_cast<int64_t>(confucius_reference.waveform_16k.size()), 1);
        write(out / (confucius_prefix + "_mel"), confucius_reference.reference_mel.values,
              confucius_reference.reference_mel.frames, confucius_reference.reference_mel.channels);
        write(out / (confucius_prefix + "_campplus"), confucius_reference.campplus_fbank.values,
              confucius_reference.campplus_fbank.frames, confucius_reference.campplus_fbank.dims);
        write(out / (confucius_prefix + "_semantic"), confucius_reference.semantic_features.values,
              confucius_reference.semantic_features.frames, confucius_reference.semantic_features.dims);
        const std::vector<float> confucius_mask(confucius_reference.semantic_features.attention_mask.begin(),
                                                confucius_reference.semantic_features.attention_mask.end());
        write(out / (confucius_prefix + "_mask"), confucius_mask,
              static_cast<int64_t>(confucius_mask.size()), 1);

        engine::models::seed_vc::SeedVcMelConfig seed_config;
        seed_config.sample_rate = 22050;
        seed_config.n_fft = 1024;
        seed_config.win_size = 1024;
        seed_config.hop_size = 256;
        seed_config.num_mels = 80;
        seed_config.fmax = 8000.0f;
        const auto seed = engine::models::seed_vc::compute_seed_vc_mel_spectrogram(
            input, seed_config, 8);
        write(out / ("seed_vc_" + suffix), seed.mel, seed.frames, seed.channels);
        const auto seed_camp = engine::models::seed_vc::compute_seed_vc_campplus_fbank_16k(input);
        write(out / ("seed_vc_campplus_" + suffix), seed_camp.features,
              seed_camp.frames, seed_camp.dims);
        const auto dots_camp = engine::models::dots_tts::compute_dots_speaker_fbank_16k(input);
        write(out / ("dots_tts_campplus_" + suffix), dots_camp.values,
              dots_camp.frames, dots_camp.dims);

        engine::runtime::AudioBuffer audio;
        audio.sample_rate = 22050;
        audio.channels = 1;
        audio.samples = input;
        const auto glm = engine::models::glm_tts::compute_glm_tts_prompt_mel(audio);
        write(out / ("glm_tts_" + suffix), glm.values, glm.frames, glm.dims);
        const auto glm_camp = engine::models::glm_tts::compute_glm_tts_campplus_fbank(audio);
        write(out / ("glm_tts_campplus_" + suffix), glm_camp.values,
              glm_camp.frames, glm_camp.dims);
        const auto kroko = engine::models::kroko_asr::compute_kroko_fbank(audio);
        write(out / ("kroko_asr_fbank_" + suffix), kroko.values,
              kroko.frames, kroko.feature_dim);

        const auto niagara = niagara_frontend.extract(audio);
        write(out / ("niagara_asr_" + suffix), niagara.values,
              niagara.frames, niagara.feature_dim);

        const auto qwen3 = engine::models::qwen3_tts::compute_qwen3_speaker_mel(audio, 8);
        write(out / ("qwen3_tts_" + suffix), qwen3.values,
              qwen3.shape.at(2), qwen3.shape.at(1));
        const auto vieneu = engine::models::vieneu_v3_turbo::compute_vieneu_speaker_mel(audio, 8);
        write(out / ("vieneu_v3_turbo_" + suffix), vieneu.values,
              vieneu.shape.at(2), vieneu.shape.at(1));

        engine::models::dramabox::DramaBoxConfig dramabox_config;
        int64_t dramabox_frames = 0;
        const auto dramabox = engine::models::dramabox::reference_log_mel(
            audio, dramabox_config, 0.25f, 8, dramabox_frames);
        write(out / ("dramabox_" + suffix), dramabox,
              dramabox_frames, 2 * dramabox_config.audio_vae.mel_bins);

        const auto mira = engine::community_models::mira_tts::compute_mira_reference_mel(audio, 8);
        write(out / ("mira_tts_" + suffix), mira,
              static_cast<int64_t>(mira.size()) / 128, 128);

        const auto whisper = engine::models::seed_vc::compute_whisper_log_mel(
            engine::audio::copy_or_zero_pad_samples_to_count(input, 480000), 8);
        write(out / ("seed_vc_whisper_" + suffix), whisper, 3000, 80);

        const auto s3_tokenizer = engine::models::chatterbox::components::compute_s3tokenizer_log_mel(audio);
        write(out / ("chatterbox_tokenizer_" + suffix), s3_tokenizer.log_mel,
              s3_tokenizer.frames, s3_tokenizer.n_mels);
        const auto s3_prompt = engine::models::chatterbox::components::compute_s3_prompt_mel(audio);
        write(out / ("chatterbox_prompt_" + suffix), s3_prompt.mel,
              s3_prompt.frames, s3_prompt.n_mels);
        const auto s3_camp = engine::models::chatterbox::components::compute_campplus_fbank(audio);
        write(out / ("chatterbox_campplus_" + suffix), s3_camp.features,
              s3_camp.frames, s3_camp.dims);

        const engine::models::chatterbox::VoiceEncoderConfig voice_config;
        const auto voice_mel = engine::models::chatterbox::compute_voice_encoder_mel(input, voice_config);
        write(out / ("chatterbox_voice_" + suffix), voice_mel,
              static_cast<int64_t>(voice_mel.size()) / voice_config.num_mels,
              voice_config.num_mels);
    }
    return 0;
} catch (const std::exception & error) {
    std::cerr << "shared_mel_migration_probe: " << error.what() << '\n';
    return 1;
}
