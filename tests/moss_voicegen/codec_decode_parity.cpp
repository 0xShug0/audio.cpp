// Codec parity for MOSS-Audio-Tokenizer v1 (codes -> 24 kHz mono waveform). Decodes the
// same fixed, deterministic code matrix as the reference dumper and compares length, peak,
// RMS and a spread of individual samples against
// tests/moss_voicegen/reference/ref_codec_v1.json.
//
// This is the check that the v1 additions to moss/shared are right: the mono tail, the hop
// taken from the config, the optional stage output projection, and the v1 tensor names.
//
//   moss_voicegen_codec_parity --codec <audio_tokenizer-dir-or-model> \
//       --reference tests/moss_voicegen/reference/ref_codec_v1.json [--out out.wav]

#include "engine/community_models/moss_voicegen/assets.h"
#include "engine/framework/audio/wav_writer.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/io/json.h"
#include "engine/models/moss/shared/audio_tokenizer_decoder.h"

#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

namespace {

namespace json = engine::io::json;

constexpr int64_t kCodebooks = 16;
constexpr int64_t kFrames = 40;

std::string arg_value(int argc, char ** argv, const std::string & name, const std::string & fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) {
            return argv[i + 1];
        }
    }
    return fallback;
}

// Same formula as tests/moss_tts_local/codec_decode_parity.cpp and the reference dumper.
std::vector<std::vector<int32_t>> code_matrix() {
    std::vector<std::vector<int32_t>> codes(static_cast<size_t>(kCodebooks));
    for (int64_t q = 0; q < kCodebooks; ++q) {
        codes[static_cast<size_t>(q)].resize(static_cast<size_t>(kFrames));
        for (int64_t t = 0; t < kFrames; ++t) {
            codes[static_cast<size_t>(q)][static_cast<size_t>(t)] = static_cast<int32_t>((q * 37 + t * 5) % 1024);
        }
    }
    return codes;
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const std::string codec_path = arg_value(argc, argv, "--codec", "");
        const std::string reference_path = arg_value(argc, argv, "--reference", "");
        const std::string wav_out = arg_value(argc, argv, "--out", "");
        const float tolerance = std::stof(arg_value(argc, argv, "--tolerance", "0.001"));
        if (codec_path.empty() || reference_path.empty()) {
            std::cerr << "usage: moss_voicegen_codec_parity --codec <dir> --reference <json>\n";
            return 2;
        }

        const auto reference = json::parse_file(reference_path);
        const auto assets = engine::models::moss_voicegen::load_moss_voicegen_assets(codec_path);

        engine::core::BackendConfig backend_config;
        backend_config.type = engine::core::BackendType::Cpu;
        backend_config.device = 0;
        backend_config.threads = std::stoi(arg_value(argc, argv, "--threads", "8"));
        engine::core::ExecutionContext execution_context(backend_config);

        const engine::models::moss::MossAudioTokenizerDecoder codec(
            *assets->audio_tokenizer_weights,
            execution_context,
            kCodebooks,
            4096ull * 1024ull * 1024ull,
            2048ull * 1024ull * 1024ull,
            engine::models::moss::moss_audio_tokenizer_v1_config());

        const auto channels = codec.decode(code_matrix());
        if (channels.size() != 1) {
            std::cerr << "FAIL: v1 is mono but the decoder returned " << channels.size() << " channels\n";
            return 1;
        }
        const auto & audio = channels.front();

        double peak = 0.0;
        double energy = 0.0;
        int64_t non_finite = 0;
        for (const float sample : audio) {
            if (!std::isfinite(sample)) {
                ++non_finite;
                continue;
            }
            peak = std::max(peak, static_cast<double>(std::fabs(sample)));
            energy += static_cast<double>(sample) * sample;
        }
        const double rms = audio.empty() ? 0.0 : std::sqrt(energy / static_cast<double>(audio.size()));

        const auto expected_samples = json::require_i64(reference, "samples");
        const auto expected_peak = static_cast<double>(json::require_f32(reference, "peak"));
        const auto expected_rms = static_cast<double>(json::require_f32(reference, "rms"));

        std::cout << "samples=" << audio.size() << " (reference " << expected_samples << ")\n";
        std::cout << "peak=" << peak << " (reference " << expected_peak << ")\n";
        std::cout << "rms=" << rms << " (reference " << expected_rms << ")\n";
        std::cout << "sample rate=" << codec.sampling_rate() << " Hz, channels=" << channels.size() << "\n";

        bool passed = non_finite == 0
            && static_cast<int64_t>(audio.size()) == expected_samples
            && std::fabs(peak - expected_peak) <= tolerance
            && std::fabs(rms - expected_rms) <= tolerance;
        if (non_finite != 0) {
            std::cerr << "FAIL: " << non_finite << " non-finite samples\n";
        }

        // Individual samples spread across the waveform: statistics alone would not catch a
        // decoder that is right on average and wrong in shape.
        const auto probe_index = json::require_i64_array(reference, "probe_index");
        const auto & probe_values = reference.require("probe_values").as_array();
        double worst = 0.0;
        int64_t worst_at = -1;
        for (size_t i = 0; i < probe_index.size() && i < probe_values.size(); ++i) {
            const auto index = static_cast<size_t>(probe_index[i]);
            if (index >= audio.size()) {
                std::cerr << "FAIL: reference probe index " << index << " is past the decoded audio\n";
                passed = false;
                break;
            }
            const double delta = std::fabs(audio[index] - probe_values[i].as_number());
            if (delta > worst) {
                worst = delta;
                worst_at = static_cast<int64_t>(index);
            }
        }
        std::cout << "worst probe deviation=" << worst << " at sample " << worst_at
                  << " over " << probe_index.size() << " probes\n";
        passed = passed && worst <= tolerance;

        if (!wav_out.empty()) {
            engine::audio::write_pcm16_wav(wav_out, static_cast<int>(codec.sampling_rate()), 1, audio);
            std::cout << "wrote " << wav_out << "\n";
        }

        std::cout << (passed ? "PASS" : "FAIL") << ": tolerance " << tolerance << "\n";
        return passed ? 0 : 1;
    } catch (const std::exception & error) {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }
}
