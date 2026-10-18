/*
 * Decodes one fixed code matrix at a range of LENGTHS and writes a wav for each.
 *
 * Reported on #663: the MOSS codec decoder agrees with the reference on short
 * inputs and drifts from it as the input grows. Every existing codec parity
 * harness decodes a fixed short matrix -- moss_voicegen_codec_parity uses 40
 * frames, 3.2 s -- so none of them can see a defect whose whole character is
 * that it grows with length.
 *
 * This takes the codes as data and the lengths as an argument, so the same
 * matrix can be decoded as its own prefix at 43, 94, 171 frames and so on, and
 * the reference can decode exactly the same prefixes. Nothing upstream of the
 * decoder differs between the two sides, which is what makes the comparison
 * about the decoder rather than about generation.
 *
 *   moss_codec_decode_length_sweep --codec <audio_tokenizer-dir> --codes <json>
 *       --lengths 43,94,171 --out-prefix /tmp/cpp
 *       [--tensor-prefix audio_tokenizer_weights]
 */

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/audio/wav_writer.h"
#include "engine/framework/codecs/moss_audio_tokenizer_codec_runtime.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/io/json.h"

#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

namespace {

namespace json = engine::io::json;

std::string arg_value(int argc, char ** argv, const std::string & name, const std::string & fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) {
            return argv[i + 1];
        }
    }
    return fallback;
}

std::vector<int64_t> parse_lengths(const std::string & value) {
    std::vector<int64_t> out;
    std::string current;
    for (const char c : value) {
        if (c == ',') {
            if (!current.empty()) out.push_back(std::stoll(current));
            current.clear();
            continue;
        }
        current += c;
    }
    if (!current.empty()) out.push_back(std::stoll(current));
    return out;
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const std::string codec_path = arg_value(argc, argv, "--codec", "");
        const std::string codes_path = arg_value(argc, argv, "--codes", "");
        const std::string lengths_arg = arg_value(argc, argv, "--lengths", "");
        const std::string out_prefix = arg_value(argc, argv, "--out-prefix", "");
        const std::string codec_version = arg_value(argc, argv, "--codec-version", "v1");
        if (codec_path.empty() || codes_path.empty() || lengths_arg.empty() || out_prefix.empty()) {
            std::cerr << "usage: moss_codec_decode_length_sweep --codec <dir> --codes <json> "
                         "--lengths 43,94,171 --out-prefix <path> [--codec-version v1|v2|nano]\n";
            return 2;
        }

        // [n_vq][frames], the codec's own layout.
        const auto codes_json = json::parse_file(codes_path);
        const auto & rows = codes_json.require("codes").as_array();
        std::vector<std::vector<int32_t>> codes;
        codes.reserve(rows.size());
        for (const auto & row : rows) {
            std::vector<int32_t> values;
            for (const auto & code : row.as_array()) {
                values.push_back(static_cast<int32_t>(code.as_number()));
            }
            codes.push_back(std::move(values));
        }
        if (codes.empty() || codes.front().empty()) {
            throw std::runtime_error("codes must be a non-empty [n_vq][frames] matrix");
        }
        const auto total_frames = static_cast<int64_t>(codes.front().size());
        std::cout << "codes: " << codes.size() << " codebooks x " << total_frames << " frames\n";

        auto config = engine::codecs::moss_audio_tokenizer_v1_config();
        if (codec_version == "v2") {
            config = engine::codecs::moss_audio_tokenizer_v2_config();
        } else if (codec_version == "nano") {
            config = engine::codecs::moss_audio_tokenizer_nano_config();
        }

        engine::core::BackendConfig backend_config;
        backend_config.type = arg_value(argc, argv, "--backend", "cpu") == "cuda"
            ? engine::core::BackendType::Cuda
            : engine::core::BackendType::Cpu;
        backend_config.device = 0;
        backend_config.threads = std::stoi(arg_value(argc, argv, "--threads", "8"));
        engine::core::ExecutionContext context(backend_config);

        // A GGUF package namespaces its tensors, and the codec inside one is
        // stored at the package's own precision -- f16 in everything we publish,
        // against f32 in the safetensors checkpoint. Being able to point this at
        // either is the whole reason the flag exists: it is the difference the
        // report on #663 could not isolate, because changing weight_type there
        // also changed the codes being decoded.
        const std::string prefix = arg_value(argc, argv, "--tensor-prefix", "");
        auto weights = prefix.empty()
            ? engine::assets::open_tensor_source(codec_path)
            : engine::assets::open_tensor_source(codec_path, prefix);
        engine::codecs::MossAudioTokenizerCodecRuntime runtime(
            weights,
            context,
            static_cast<int64_t>(codes.size()),
            engine::codecs::MossAudioTokenizerCodecRuntimeOptions{},
            config);
        runtime.prepare_decoder();

        for (const int64_t frames : parse_lengths(lengths_arg)) {
            if (frames <= 0 || frames > total_frames) {
                std::cout << "  skipping " << frames << " frames: outside the matrix\n";
                continue;
            }
            std::vector<std::vector<int32_t>> prefix;
            prefix.reserve(codes.size());
            for (const auto & row : codes) {
                prefix.emplace_back(row.begin(), row.begin() + static_cast<std::ptrdiff_t>(frames));
            }
            auto audio = runtime.decode(
                engine::codecs::MossAudioTokenizerCodes{frames, std::move(prefix)});
            if (audio.channels.empty()) {
                throw std::runtime_error("codec returned no audio");
            }
            // ⚠ ALL CHANNELS, INTERLEAVED. Nano is the 48 kHz STEREO codec, and
            // comparing only its first channel against a stereo reference
            // measures the harness rather than the decoder.
            const auto channel_count = static_cast<int>(audio.channels.size());
            const auto frame_samples = audio.channels.front().size();
            std::vector<float> samples(frame_samples * static_cast<size_t>(channel_count));
            for (size_t i = 0; i < frame_samples; ++i) {
                for (int ch = 0; ch < channel_count; ++ch) {
                    samples[i * static_cast<size_t>(channel_count) + static_cast<size_t>(ch)] =
                        audio.channels[static_cast<size_t>(ch)][i];
                }
            }
            double sum_sq = 0.0;
            for (const float sample : samples) {
                sum_sq += static_cast<double>(sample) * sample;
            }
            const double rms = samples.empty() ? 0.0 : std::sqrt(sum_sq / static_cast<double>(samples.size()));
            const std::string path = out_prefix + "_" + std::to_string(frames) + ".wav";
            // ⚠ FLOAT32, NOT THE DEFAULT PCM16. The defect being measured is a
            // noise floor around -56 dBFS in passages the model generated as
            // silence. 16-bit quantisation sits at about -96 dB so it would not
            // erase it, but dither and hard-clip shaping are exactly the kind of
            // post-processing that muddies a measurement of the decoder alone.
            engine::audio::WavWriteOptions wav_options;
            wav_options.format = engine::audio::WavSampleFormat::Float32;
            engine::audio::write_wav(
                path, static_cast<int>(config.sampling_rate), channel_count, samples, wav_options);
            std::cout << "  " << frames << " frames -> " << frame_samples << " samples x "
                      << channel_count << " ch, rms " << rms << ", " << path << "\n";
        }
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "moss_codec_decode_length_sweep failed: " << error.what() << "\n";
        return 1;
    }
}
