#include "engine/community_models/whistle_asr/assets.h"
#include "engine/community_models/whistle_asr/runtime.h"
#include "engine/framework/audio/wav_reader.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

int main(int argc, char ** argv) {
    try {
        if (argc < 2 || argc > 5 || (argc == 5 && std::string(argv[4]) != "--full")) {
            throw std::invalid_argument(
                "Pass <GGUF> [16k-mono.wav] [expected transcript] [--full]");
        }
        const auto assets = engine::community_models::whistle_asr::load_whistle_assets(argv[1]);
        if (assets->mel_filterbank.size() != 257 * 80 ||
            assets->hadamard_permutations[0][0] != 396 ||
            assets->hadamard_permutations[1][0] != 458 ||
            engine::community_models::whistle_asr::decode_whistle_tokens(
                assets->tokenizer_pieces, {3633, 1313, 409, 8116}) != " Some call me.") {
            throw std::runtime_error("Whistle GGUF assets do not match the reference model");
        }
        std::cout << "Loaded " << assets->weights->tensors().size()
                  << " GGUF tensors and " << assets->tokenizer_pieces.size()
                  << " tokenizer pieces\n";
        if (argc >= 3) {
            auto wave = engine::audio::read_wav_f32(std::filesystem::path(argv[2]));
            if (wave.sample_rate != 16000 || wave.channels != 1) {
                throw std::invalid_argument("Whistle test requires 16 kHz mono WAV");
            }
            if (argc != 5) {
                wave.samples.resize(std::min<size_t>(wave.samples.size(), 3 * 16000));
            }
            engine::runtime::AudioBuffer audio;
            audio.sample_rate = wave.sample_rate;
            audio.channels = wave.channels;
            audio.samples = std::move(wave.samples);
            const auto start = std::chrono::steady_clock::now();
            engine::community_models::whistle_asr::WhistleRuntime runtime(assets);
            const auto transcript = runtime.transcribe(audio);
            std::cout << "Whistle: " << transcript.text << " [" << transcript.language << "] in "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::steady_clock::now() - start).count() << " ms\n";
            if (argc >= 4 && transcript.text != argv[3]) {
                throw std::runtime_error("Whistle transcript differs from the expected speech");
            }
            const auto repeated = runtime.transcribe(audio);
            if (repeated.text != transcript.text || repeated.language != transcript.language) {
                throw std::runtime_error("Whistle retained decoder state across recordings");
            }
            auto quiet = audio;
            for (float & sample : quiet.samples) {
                sample *= 0.0005f;
            }
            if (runtime.transcribe(quiet).text.rfind("Some call me", 0) != 0) {
                throw std::runtime_error("Whistle dropped quiet speech despite front-end AGC");
            }
            auto constant_rms = audio;
            constant_rms.samples.resize(std::min<size_t>(constant_rms.samples.size(), 3 * 16000));
            constexpr size_t kRmsBlockSamples = 320;
            constexpr float kTargetRms = 1000.0f / 32768.0f;
            for (size_t offset = 0; offset < constant_rms.samples.size();
                 offset += kRmsBlockSamples) {
                const size_t count = std::min(
                    kRmsBlockSamples, constant_rms.samples.size() - offset);
                double sum_squares = 0.0;
                for (size_t index = 0; index < count; ++index) {
                    const double sample = constant_rms.samples[offset + index];
                    sum_squares += sample * sample;
                }
                const float rms = static_cast<float>(std::sqrt(sum_squares / count));
                if (rms > 0.0f) {
                    const float scale = kTargetRms / rms;
                    for (size_t index = 0; index < count; ++index) {
                        constant_rms.samples[offset + index] *= scale;
                    }
                }
            }
            if (runtime.transcribe(constant_rms).text != "Some call me nature.") {
                throw std::runtime_error("Whistle dropped speech with constant block RMS");
            }
            const auto silence = runtime.transcribe(
                engine::runtime::AudioBuffer{16000, 1, std::vector<float>(16000, 0.0f)});
            if (!silence.text.empty() || !silence.language.empty()) {
                throw std::runtime_error("Whistle invented words for silence");
            }
            bool rejected = false;
            try {
                (void)runtime.transcribe(engine::runtime::AudioBuffer{
                    16000, 1, std::vector<float>(30 * 16000 + 1, 0.0f)});
            } catch (const std::invalid_argument &) {
                rejected = true;
            }
            if (!rejected) {
                throw std::runtime_error("Whistle accepted audio longer than 30 seconds");
            }
            rejected = false;
            try {
                (void)runtime.transcribe(engine::runtime::AudioBuffer{
                    16000, 1, std::vector<float>(1, std::numeric_limits<float>::quiet_NaN())});
            } catch (const std::invalid_argument &) {
                rejected = true;
            }
            if (!rejected) {
                throw std::runtime_error("Whistle accepted a non-finite short recording");
            }
        }
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
