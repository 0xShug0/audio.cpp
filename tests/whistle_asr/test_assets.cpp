#include "engine/community_models/whistle_asr/assets.h"
#include "engine/community_models/whistle_asr/encoder.h"
#include "engine/community_models/whistle_asr/frontend.h"
#include "engine/community_models/whistle_asr/runtime.h"
#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/core/execution_context.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace whistle = engine::community_models::whistle_asr;

struct Options {
    std::string gguf;
    std::string wav;
    std::string expected;
    bool full = false;
    std::string reference_dir;
    std::string dump_dir;
    // Allowed max |difference| as a fraction of the reference tensor's largest magnitude.
    double tolerance = 2.0e-3;
    engine::core::BackendConfig backend{engine::core::BackendType::Cpu, 0, 4};
};

Options parse(int argc, char ** argv) {
    Options options;
    std::vector<std::string> positional;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        const auto value = [&]() -> std::string {
            if (index + 1 >= argc) {
                throw std::invalid_argument(argument + " needs a value");
            }
            return argv[++index];
        };
        if (argument == "--full") {
            options.full = true;
        } else if (argument == "--reference") {
            options.reference_dir = value();
        } else if (argument == "--dump") {
            options.dump_dir = value();
        } else if (argument == "--backend") {
            const std::string backend = value();
            if (backend == "cpu") {
                options.backend.type = engine::core::BackendType::Cpu;
            } else if (backend == "metal") {
                options.backend.type = engine::core::BackendType::Metal;
            } else {
                throw std::invalid_argument("Unsupported test backend: " + backend);
            }
        } else if (argument == "--threads") {
            options.backend.threads = std::stoi(value());
        } else if (argument == "--tolerance") {
            options.tolerance = std::stod(value());
        } else {
            positional.push_back(argument);
        }
    }
    if (positional.empty() || positional.size() > 3) {
        throw std::invalid_argument(
            "Pass <GGUF> [16k-mono.wav] [expected transcript] [--full] "
            "[--reference <dump-dir>] [--tolerance <fraction>] [--dump <dir>] (encoder tensors and decoder logits/tokens/transcript) [--backend cpu|metal] [--threads n]");
    }
    options.gguf = positional[0];
    if (positional.size() > 1) options.wav = positional[1];
    if (positional.size() > 2) options.expected = positional[2];
    return options;
}

engine::runtime::AudioBuffer read_audio(const std::string & path, bool full) {
    auto wave = engine::audio::read_wav_f32(std::filesystem::path(path));
    if (wave.sample_rate != 16000 || wave.channels != 1) {
        throw std::invalid_argument("Whistle test requires 16 kHz mono WAV");
    }
    if (!full) {
        wave.samples.resize(std::min<size_t>(wave.samples.size(), 3 * 16000));
    }
    engine::runtime::AudioBuffer audio;
    audio.sample_rate = wave.sample_rate;
    audio.channels = wave.channels;
    audio.samples = std::move(wave.samples);
    return audio;
}

std::vector<float> read_f32(const std::string & path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    const auto bytes = static_cast<std::streamoff>(stream.tellg());
    if (!stream || bytes < 0 || bytes % sizeof(float) != 0) {
        throw std::runtime_error("Invalid float reference: " + path);
    }
    std::vector<float> values(static_cast<size_t>(bytes) / sizeof(float));
    stream.seekg(0);
    if (!stream.read(reinterpret_cast<char *>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(float)))) {
        throw std::runtime_error("Cannot read float reference: " + path);
    }
    return values;
}

// Reports the largest absolute difference against a reference dump, scaled by the
// reference's largest magnitude, and fails when that fraction exceeds the tolerance.
void compare(const std::string & name, const std::vector<float> & actual, const std::string & path, double tolerance) {
    const auto expected = read_f32(path);
    if (expected.size() != actual.size()) {
        throw std::runtime_error(name + " reference has " + std::to_string(expected.size()) +
            " values, runtime produced " + std::to_string(actual.size()));
    }
    double max_abs = 0.0;
    double scale = 0.0;
    for (size_t index = 0; index < actual.size(); ++index) {
        if (!std::isfinite(actual[index]) || !std::isfinite(expected[index])) {
            throw std::runtime_error(name + " has a non-finite value at index " + std::to_string(index));
        }
        max_abs = std::max(max_abs, std::abs(static_cast<double>(actual[index]) - expected[index]));
        scale = std::max(scale, std::abs(static_cast<double>(expected[index])));
    }
    const double fraction = scale > 0.0 ? max_abs / scale : max_abs;
    std::cout << name << ": max_abs=" << max_abs << " reference_max=" << scale
              << " fraction=" << fraction << "\n";
    if (!(fraction <= tolerance)) {
        throw std::runtime_error(name + " differs from the reference beyond the tolerance");
    }
}

void write_f32(const std::string & path, const std::vector<float> & values) {
    std::ofstream stream(path, std::ios::binary);
    if (!stream.write(reinterpret_cast<const char *>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(float)))) {
        throw std::runtime_error("Cannot write float dump: " + path);
    }
}

// Frontend and encoder tensors of the full recording, either written as raw
// little-endian f32 dumps (--dump) or compared against such dumps (--reference).
void check_encoder_tensors(const Options & options, const std::shared_ptr<const whistle::WhistleAssets> & assets,
                           engine::core::ExecutionContext & execution_context) {
    const auto audio = read_audio(options.wav, true);
    const whistle::WhistleFrontend frontend(assets->mel_filterbank);
    const auto mel = frontend.extract(audio.samples);
    whistle::WhistleEncoderRuntime encoder(assets, execution_context);
    auto start = std::chrono::steady_clock::now();
    const auto output = encoder.encode(mel);
    const double first_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    start = std::chrono::steady_clock::now();
    (void)encoder.encode(mel);
    const double warm_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    std::cout << "encoder: " << output.frames << " frames, first " << first_ms << " ms, warm " << warm_ms << " ms\n";
    const std::vector<std::pair<std::string, const std::vector<float> *>> tensors = {
        {"mel", &mel.values},
        {"memory", &output.memory},
        {"cross_k_0", &output.cross_k[0]},
        {"cross_v_0", &output.cross_v[0]},
        {"cross_k_7", &output.cross_k[7]},
        {"cross_v_7", &output.cross_v[7]},
    };
    if (!options.dump_dir.empty()) {
        for (const auto & [name, values] : tensors) {
            write_f32(options.dump_dir + "/" + name + ".f32", *values);
        }
        std::cout << "Wrote " << tensors.size() << " reference dumps to " << options.dump_dir << "\n";
    }
    if (!options.reference_dir.empty()) {
        for (const auto & [name, values] : tensors) {
            compare(name, *values, options.reference_dir + "/" + name + ".f32", options.tolerance);
        }
    }
}

std::vector<int32_t> read_i32(const std::string & path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    const auto bytes = static_cast<std::streamoff>(stream.tellg());
    if (!stream || bytes < 0 || bytes % sizeof(int32_t) != 0) {
        throw std::runtime_error("Invalid token reference: " + path);
    }
    std::vector<int32_t> values(static_cast<size_t>(bytes) / sizeof(int32_t));
    stream.seekg(0);
    if (!stream.read(reinterpret_cast<char *>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(int32_t)))) {
        throw std::runtime_error("Cannot read token reference: " + path);
    }
    return values;
}

std::string read_text(const std::string & path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("Cannot read transcript reference: " + path);
    }
    std::ostringstream contents;
    contents << stream.rdbuf();
    return contents.str();
}

// Per-step decoder logits (decoder_logits_<position>.f32), the input token sequence
// (decoder_tokens.i32, BOS and language token first) and the transcript of the full
// recording, written by --dump or required to match by --reference.
void check_decoder(const Options & options, const std::shared_ptr<const whistle::WhistleAssets> & assets,
                   engine::core::ExecutionContext & execution_context) {
    const auto audio = read_audio(options.wav, true);
    whistle::WhistleRuntime runtime(assets, execution_context);
    std::vector<whistle::WhistleDecodeStep> steps;
    const auto transcript = runtime.transcribe(audio, "", [&](const whistle::WhistleDecodeStep & step) {
        steps.push_back(step);
    });
    std::vector<int32_t> tokens;
    for (const auto & step : steps) {
        tokens.push_back(step.input_token);
    }
    const auto logits_path = [](const std::string & dir, size_t position) {
        return dir + "/decoder_logits_" + std::to_string(position) + ".f32";
    };
    if (!options.dump_dir.empty()) {
        for (const auto & step : steps) {
            write_f32(logits_path(options.dump_dir, step.position), step.logits);
        }
        std::ofstream token_stream(options.dump_dir + "/decoder_tokens.i32", std::ios::binary);
        if (!token_stream.write(reinterpret_cast<const char *>(tokens.data()),
                static_cast<std::streamsize>(tokens.size() * sizeof(int32_t)))) {
            throw std::runtime_error("Cannot write decoder tokens");
        }
        std::ofstream text_stream(options.dump_dir + "/decoder_transcript.txt", std::ios::binary);
        if (!text_stream.write(transcript.text.data(), static_cast<std::streamsize>(transcript.text.size()))) {
            throw std::runtime_error("Cannot write decoder transcript");
        }
    }
    if (!options.reference_dir.empty()) {
        if (read_i32(options.reference_dir + "/decoder_tokens.i32") != tokens) {
            throw std::runtime_error("Decoder token sequence differs from the reference");
        }
        if (read_text(options.reference_dir + "/decoder_transcript.txt") != transcript.text) {
            throw std::runtime_error("Decoder transcript differs from the reference");
        }
        for (const auto & step : steps) {
            compare("decoder_logits_" + std::to_string(step.position), step.logits,
                    logits_path(options.reference_dir, step.position), options.tolerance);
        }
    }
    std::cout << "decoder: " << steps.size() << " steps, transcript: " << transcript.text << "\n";
}

void check_transcription(const Options & options, const std::shared_ptr<const whistle::WhistleAssets> & assets,
                         engine::core::ExecutionContext & execution_context) {
    const auto audio = read_audio(options.wav, options.full);
    const auto start = std::chrono::steady_clock::now();
    whistle::WhistleRuntime runtime(assets, execution_context);
    const auto transcript = runtime.transcribe(audio);
    std::cout << "Whistle: " << transcript.text << " [" << transcript.language << "] in "
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::steady_clock::now() - start).count() << " ms\n";
    if (!options.expected.empty() && transcript.text != options.expected) {
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
    for (size_t offset = 0; offset < constant_rms.samples.size(); offset += kRmsBlockSamples) {
        const size_t count = std::min(kRmsBlockSamples, constant_rms.samples.size() - offset);
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

}  // namespace

int main(int argc, char ** argv) {
    try {
        const Options options = parse(argc, argv);
        const auto assets = whistle::load_whistle_assets(options.gguf);
        if (assets->mel_filterbank.size() != 257 * 80 ||
            assets->hadamard_permutations[0][0] != 396 ||
            assets->hadamard_permutations[1][0] != 458 ||
            whistle::decode_whistle_tokens(assets->tokenizer_pieces, {3633, 1313, 409, 8116}) != " Some call me.") {
            throw std::runtime_error("Whistle GGUF assets do not match the reference model");
        }
        std::cout << "Loaded " << assets->weights->tensors().size()
                  << " GGUF tensors and " << assets->tokenizer_pieces.size()
                  << " tokenizer pieces\n";
        if (options.wav.empty()) {
            return 0;
        }
        engine::core::ExecutionContext execution_context(options.backend);
        if (!options.reference_dir.empty() || !options.dump_dir.empty()) {
            check_encoder_tensors(options, assets, execution_context);
            check_decoder(options, assets, execution_context);
            return 0;
        }
        check_transcription(options, assets, execution_context);
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
