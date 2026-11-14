// ZONOS3 parity against the native ZONOS3 runtime's outputs (tests/community_models/zonos3/make_oracle.sh).
//   zonos3_parity backbone <backbone.gguf> <oracle_dir> [--disk-experts]
//   zonos3_parity speaker <spk-encoder.gguf> <voice.wav> <oracle_dir>
//   zonos3_parity decoder <zac-keep2.gguf> <oracle_dir>
//   zonos3_parity reference <gguf_dir> <voice.wav> <oracle_dir>
#include "engine/community_models/zonos3/assets.h"
#include "engine/community_models/zonos3/backbone.h"
#include "engine/community_models/zonos3/codec_decoder.h"
#include "engine/community_models/zonos3/npy.h"
#include "engine/community_models/zonos3/prompt.h"
#include "engine/community_models/zonos3/reference_encoder.h"
#include "engine/community_models/zonos3/speaker_encoder.h"
#include "engine/community_models/zonos3/tokenizer.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/debug/trace.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace zonos3 = engine::models::zonos3;
namespace fs = std::filesystem;

namespace {

std::string read_text(const fs::path & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path.string());
    std::stringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

engine::core::BackendConfig cpu_backend() {
    engine::core::BackendConfig config;
    config.type = engine::core::BackendType::Cpu;
    config.threads = 8;
    return config;
}

double cosine(const float * a, const float * b, size_t n) {
    double ab = 0, aa = 0, bb = 0;
    for (size_t i = 0; i < n; ++i) {
        ab += static_cast<double>(a[i]) * b[i];
        aa += static_cast<double>(a[i]) * a[i];
        bb += static_cast<double>(b[i]) * b[i];
    }
    return ab / std::sqrt(std::max(aa * bb, 1e-300));
}

size_t argmax(const float * x, size_t n) {
    return static_cast<size_t>(std::max_element(x, x + n) - x);
}

struct OracleRun {
    zonos3::Zonos3Prompt prompt;
    zonos3::Zonos3NpyF32 ids;
    zonos3::Zonos3NpyF32 reference;  // [1 + steps, 10, 4098]
    int steps = 0;
};

OracleRun load_backbone_oracle(const zonos3::Zonos3Assets & assets, const fs::path & oracle) {
    const auto & hp = assets.hparams;
    const zonos3::Zonos3Tokenizer tokenizer(assets.tokenizer_json);
    OracleRun out;
    out.prompt = zonos3::build_zonos3_prompt(hp, tokenizer, read_text(oracle / "text.txt"), {});
    out.ids = zonos3::read_f32_npy(oracle / "tts_ids.npy");
    out.reference = zonos3::read_f32_npy(oracle / "tts_logits.npy");
    const int width = hp.n_codebooks + 1;
    if (out.ids.shape.size() != 2 || out.ids.shape[1] != width || out.ids.shape[0] < out.prompt.n_rows) {
        throw std::runtime_error("tts_ids.npy must be [rows, 11] with at least the prompt rows");
    }
    for (size_t i = 0; i < out.prompt.ids.size(); ++i) {
        if (static_cast<float>(out.prompt.ids[i]) != out.ids.data[i]) {
            throw std::runtime_error("prompt ids differ from the oracle at row " + std::to_string(i / width) + " column " +
                                     std::to_string(i % width));
        }
    }
    out.steps = static_cast<int>(out.reference.shape[0]);
    if (out.reference.shape.size() != 3 || out.reference.shape[1] != hp.n_codebooks ||
        out.reference.shape[2] != hp.audio_vocab || out.steps - 1 > out.ids.shape[0] - out.prompt.n_rows) {
        throw std::runtime_error("tts_logits.npy must be [1 + steps, 10, 4098] with matching tts_ids rows");
    }
    return out;
}

// Prefill the oracle prompt, then teacher-force its generated rows; returns [1 + steps][10 * 4098] logits.
std::vector<float> teacher_forced_logits(std::shared_ptr<const zonos3::Zonos3Assets> assets, const OracleRun & run,
                                         const zonos3::Zonos3BackboneOptions & options) {
    const auto & hp = assets->hparams;
    const int width = hp.n_codebooks + 1;
    const size_t per_step = static_cast<size_t>(hp.audio_vocab) * static_cast<size_t>(hp.n_codebooks);
    engine::core::ExecutionContext execution(cpu_backend());
    zonos3::Zonos3Backbone backbone(std::move(assets), execution, options);
    backbone.begin(1, run.prompt.n_rows + run.steps);
    std::vector<float> all, logits;
    all.reserve(per_step * static_cast<size_t>(run.steps));
    backbone.prefill(0, run.prompt, logits);
    std::vector<int32_t> row(static_cast<size_t>(width));
    for (int step = 0; step < run.steps; ++step) {
        if (step > 0) {
            const size_t at = static_cast<size_t>(run.prompt.n_rows + step - 1) * static_cast<size_t>(width);
            for (int c = 0; c < width; ++c) {
                row[static_cast<size_t>(c)] = static_cast<int32_t>(run.ids.data[at + static_cast<size_t>(c)]);
            }
            backbone.step({row.data(), row.data()}, 1, logits);
        }
        all.insert(all.end(), logits.begin(), logits.end());
    }
    return all;
}

// Minimum per-(step, codebook) cosine and argmax agreement of two [steps][10][4098] logit sets.
std::pair<double, double> compare_logits(const std::vector<float> & ours, const std::vector<float> & theirs, size_t vocab) {
    double cos_min = 1.0;
    size_t agree = 0;
    const size_t rows = ours.size() / vocab;
    for (size_t r = 0; r < rows; ++r) {
        const float * a = ours.data() + r * vocab;
        const float * b = theirs.data() + r * vocab;
        cos_min = std::min(cos_min, cosine(a, b, vocab));
        agree += argmax(a, vocab) == argmax(b, vocab);
    }
    return {cos_min, static_cast<double>(agree) / static_cast<double>(rows)};
}

// Resident: the oracle logits within cos >= 0.9999 and argmax agreement >= 0.99.
// --disk-experts: the expert-cache path (8 slots, so prefill chunks stage several groups and evict) must be
// bit-identical to the resident path using the same framework expert kernels; oracle metrics are also reported.
int backbone(const fs::path & model, const fs::path & oracle, bool disk_experts) {
    const auto assets = zonos3::load_zonos3_assets(model);
    const auto run = load_backbone_oracle(*assets, oracle);
    const size_t vocab = static_cast<size_t>(assets->hparams.audio_vocab);
    zonos3::Zonos3BackboneOptions options;
    if (disk_experts) {
        options.moe.residency = engine::core::MoeExpertResidency::Staged;
        options.moe.cache_slots = 8;
    }
    const auto ours = teacher_forced_logits(assets, run, options);
    const auto [cos_min, rate] = compare_logits(ours, run.reference.data, vocab);
    if (!disk_experts) {
        const bool pass = cos_min >= 0.9999 && rate >= 0.99;
        std::printf("zonos3 backbone parity: %s cos_min=%.6f argmax=%.4f steps=%d\n", pass ? "PASS" : "FAIL", cos_min, rate,
                    run.steps);
        return pass ? 0 : 1;
    }
    zonos3::Zonos3BackboneOptions plain;
    plain.repack_experts = false;
    const auto resident = teacher_forced_logits(assets, run, plain);
    float max_diff = 0.0f;
    for (size_t i = 0; i < ours.size(); ++i) max_diff = std::max(max_diff, std::fabs(ours[i] - resident[i]));
    const bool pass = max_diff == 0.0f;
    std::printf("zonos3 backbone parity (disk experts): %s max_diff_vs_resident_plain=%.3g oracle_cos_min=%.6f "
                "oracle_argmax=%.4f steps=%d\n", pass ? "PASS" : "FAIL", max_diff, cos_min, rate, run.steps);
    return pass ? 0 : 1;
}

// The oracle (spk-encoder-cli --audio) decodes the WAV to 44.1 kHz mono floats, as read_wav_f32 does for a
// 44.1 kHz mono 16-bit WAV. Requires cosine(ours, spk.npy) >= 0.999.
int speaker(const fs::path & model, const fs::path & wav_path, const fs::path & oracle) {
    const auto wav = engine::audio::read_wav_f32(wav_path);
    if (wav.sample_rate != 44100 || wav.channels != 1) throw std::runtime_error("voice.wav must be 44.1 kHz mono");
    const auto reference = zonos3::read_f32_npy(oracle / "spk.npy");
    if (reference.data.size() != 192) throw std::runtime_error("spk.npy must hold 192 floats");
    engine::core::ExecutionContext execution(cpu_backend());
    const zonos3::Zonos3SpeakerEncoder encoder(engine::assets::open_tensor_source(model), execution);
    const auto ours = encoder.embed(wav.samples);
    const double cos = cosine(ours.data(), reference.data.data(), ours.size());
    const bool pass = cos >= 0.999;
    std::printf("zonos3 speaker parity: %s cos=%.6f\n", pass ? "PASS" : "FAIL", cos);
    return pass ? 0 : 1;
}

// ZAC decoder: the oracle's delayed codes (tts.npy, [eos, prefill] in tts.npy.eos.npy) un-sheared, decoded in one
// pass, prefill dropped and the 3 dB headroom applied, against zac-cli's tts_audio.npy (SNR >= 40 dB, equal length).
// Seam: the same frames 7 at a time on a fresh stream match the single pass within 1e-5.
int decoder(const fs::path & model, const fs::path & oracle) {
    const auto codes = zonos3::read_f32_npy(oracle / "tts.npy");
    const auto eos = zonos3::read_f32_npy(oracle / "tts.npy.eos.npy");
    const auto reference = zonos3::read_f32_npy(oracle / "tts_audio.npy");
    constexpr int kCodebooks = zonos3::kZonos3Codebooks;
    if (codes.shape.size() != 2 || codes.shape[1] != kCodebooks) throw std::runtime_error("tts.npy must be [rows, 10]");
    if (eos.data.size() != 2) throw std::runtime_error("tts.npy.eos.npy must be [eos, prefill]");
    const int rows = static_cast<int>(codes.shape[0]);
    const int eos_frame = static_cast<int>(std::lrint(eos.data[0]));
    const int prefill = static_cast<int>(std::lrint(eos.data[1]));
    const int frames = std::min(eos_frame >= 0 ? eos_frame : rows, rows - (kCodebooks - 1));
    if (frames <= 0) throw std::runtime_error("tts.npy has no complete frame");
    // Frame f, codebook c is delayed row f + c; rows past the matrix or ids >= 4096 clamp to 4095 (zac.cpp:1268-1276).
    std::vector<int32_t> undelayed(static_cast<size_t>(frames) * kCodebooks);
    for (int f = 0; f < frames; ++f) {
        for (int c = 0; c < kCodebooks; ++c) {
            const int row = f + c;
            const int code = row < rows ? static_cast<int>(std::lrint(codes.data[static_cast<size_t>(row) * kCodebooks + c])) : 4097;
            undelayed[static_cast<size_t>(f) * kCodebooks + c] = std::clamp(code, 0, zonos3::kZonos3CodebookSize - 1);
        }
    }
    engine::core::ExecutionContext execution(cpu_backend());
    zonos3::Zonos3CodecDecoder codec(engine::assets::open_tensor_source(model), execution);
    auto run = [&](int block) {
        zonos3::Zonos3DecoderStream stream(codec);
        std::vector<float> audio;
        for (int f = 0; f < frames; f += block) {
            const auto part = stream.decode(undelayed.data() + static_cast<size_t>(f) * kCodebooks, std::min(block, frames - f));
            audio.insert(audio.end(), part.begin(), part.end());
        }
        const auto tail = stream.flush();
        audio.insert(audio.end(), tail.begin(), tail.end());
        if (audio.size() != static_cast<size_t>(frames) * static_cast<size_t>(codec.samples_per_frame())) {
            throw std::runtime_error("decoder output is not frames * samples_per_frame samples");
        }
        return audio;
    };
    const auto single = run(frames);
    const auto tiled = run(7);
    double seam_max = 0.0;
    for (size_t i = 0; i < single.size(); ++i) seam_max = std::max(seam_max, static_cast<double>(std::fabs(single[i] - tiled[i])));
    const size_t drop = std::min(single.size(), static_cast<size_t>(prefill) * static_cast<size_t>(codec.samples_per_frame()));
    const float gain = std::pow(10.0f, -codec.headroom_db() / 20.0f);
    std::vector<float> ours(single.begin() + static_cast<std::ptrdiff_t>(drop), single.end());
    for (auto & v : ours) v *= gain;
    double signal = 0.0, noise = 0.0;
    const size_t n = std::min(ours.size(), reference.data.size());
    for (size_t i = 0; i < n; ++i) {
        const double r = reference.data[i];
        signal += r * r;
        noise += (ours[i] - r) * (ours[i] - r);
    }
    const double snr = 10.0 * std::log10(signal / std::max(noise, 1e-30));
    const bool same_length = ours.size() == reference.data.size();
    const bool pass = same_length && snr >= 40.0 && seam_max <= 1e-5;
    std::printf("zonos3 decoder parity: %s snr=%.2f seam_max=%.3g samples=%zu oracle_samples=%zu frames=%d\n",
                pass ? "PASS" : "FAIL", snr, seam_max, ours.size(), reference.data.size(), frames);
    return pass ? 0 : 1;
}

// Raw-reference codes of <voice.wav> against zac-encoder-cli's ref_codes.npy: same frame count, codebook-0
// agreement >= 0.95 and all-codebook agreement >= 0.90. zac-encoder-cli decodes the WAV with ffmpeg to f32 mono at
// 44.1 kHz; for a 16-bit mono 44.1 kHz WAV that equals read_wav_f32's sample / 32768 exactly.
int reference(const fs::path & gguf_dir, const fs::path & voice, const fs::path & oracle) {
    auto wav = engine::audio::read_wav_f32(voice);
    if (wav.sample_rate != 44100) throw std::runtime_error("reference parity needs a 44.1 kHz WAV");
    auto mono = wav.channels == 1 ? std::move(wav.samples)
                                  : engine::audio::mixdown_interleaved_to_mono_average(wav.samples, wav.channels);
    const auto expected = zonos3::read_f32_npy(oracle / "ref_codes.npy");
    if (expected.shape.size() != 2 || expected.shape[1] != 10) throw std::runtime_error("ref_codes.npy must be [T, 10]");
    const auto start = std::chrono::steady_clock::now();
    engine::core::ExecutionContext execution(cpu_backend());
    const zonos3::Zonos3ReferenceEncoder encoder(engine::assets::open_tensor_source(gguf_dir / "zac-encoder.gguf"),
                                                 engine::assets::open_tensor_source(gguf_dir / "ssl-teacher.gguf"), execution);
    const auto codes = encoder.encode(mono);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    const size_t frames = codes.size() / 10;
    size_t cb0 = 0;
    size_t all = 0;
    const bool same_frames = frames == static_cast<size_t>(expected.shape[0]);
    if (same_frames) {
        for (size_t i = 0; i < codes.size(); ++i) {
            const bool match = static_cast<float>(codes[i]) == expected.data[i];
            all += match;
            cb0 += match && i % 10 == 0;
        }
    }
    const double cb0_rate = same_frames ? static_cast<double>(cb0) / static_cast<double>(frames) : 0.0;
    const double all_rate = same_frames ? static_cast<double>(all) / static_cast<double>(codes.size()) : 0.0;
    const bool pass = same_frames && cb0_rate >= 0.95 && all_rate >= 0.90;
    std::printf("zonos3 reference parity: %s cb0=%.4f all=%.4f frames=%zu oracle_frames=%lld seconds=%.1f\n",
                pass ? "PASS" : "FAIL", cb0_rate, all_rate, frames, static_cast<long long>(expected.shape[0]), seconds);
    return pass ? 0 : 1;
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        std::vector<std::string> args(argv + 1, argv + argc);
        // --log prints the runtime's timing and statistics lines (disk-expert loads and cache hits).
        if (const auto log = std::find(args.begin(), args.end(), "--log"); log != args.end()) {
            engine::debug::configure_logging({true, std::nullopt});
            args.erase(log);
        }
        if (args.size() >= 3 && args[0] == "backbone") {
            const bool disk = args.size() == 4 && args[3] == "--disk-experts";
            if (args.size() > 4 || (args.size() == 4 && !disk)) throw std::runtime_error("unknown backbone option");
            return backbone(args[1], args[2], disk);
        }
        if (args.size() == 4 && args[0] == "speaker") return speaker(args[1], args[2], args[3]);
        if (args.size() == 3 && args[0] == "decoder") return decoder(args[1], args[2]);
        if (args.size() == 4 && args[0] == "reference") return reference(args[1], args[2], args[3]);
        std::cerr << "usage: zonos3_parity [--log] backbone <backbone.gguf> <oracle_dir> [--disk-experts]\n"
                     "       zonos3_parity [--log] speaker <spk-encoder.gguf> <voice.wav> <oracle_dir>\n"
                     "       zonos3_parity [--log] decoder <zac-keep2.gguf> <oracle_dir>\n"
                     "       zonos3_parity [--log] reference <gguf_dir> <voice.wav> <oracle_dir>\n";
        return 2;
    } catch (const std::exception & error) {
        std::cerr << "zonos3_parity: " << error.what() << "\n";
        return 1;
    }
}
