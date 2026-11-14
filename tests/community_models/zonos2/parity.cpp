// ZONOS2 parity against Zyphra/zonos2.cpp's outputs (tests/community_models/zonos2/make_oracle.sh).
//   zonos2_parity backbone <zonos2-*.gguf> <oracle_dir> [--disk-experts]
//   zonos2_parity backbone_speaker <zonos2-*.gguf> <oracle_dir>
//   zonos2_parity decoder <dac.gguf> <oracle_dir>
//   zonos2_parity speaker <spk-encoder.gguf> <oracle_dir>
#include "engine/community_models/zonos2/assets.h"
#include "engine/community_models/zonos2/backbone.h"
#include "engine/community_models/zonos2/codec_decoder.h"
#include "engine/community_models/zonos2/npy.h"
#include "engine/community_models/zonos2/prompt.h"
#include "engine/community_models/zonos2/speaker_encoder.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/debug/trace.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace zonos2 = engine::models::zonos2;
namespace fs = std::filesystem;

namespace {

constexpr int kOracleMaxFrames = 300;  // make_oracle.sh --max
constexpr int kMaxSteps = 100;

std::string read_text(const fs::path & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path.string());
    std::stringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

// zonos2.cpp runs its backbone with ggml's default 4 CPU threads.
engine::core::BackendConfig cpu_backend() {
    engine::core::BackendConfig config;
    config.type = engine::core::BackendType::Cpu;
    config.threads = 4;
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
    zonos2::Zonos2Prompt prompt;
    std::vector<float> speaker;      // empty without a speaker
    zonos2::Zonos2NpyF32 ids;        // [n0 + frames, 10]
    zonos2::Zonos2NpyF32 decode;     // [1 + frames, 9, 1026], zonos2.cpp's prefill + KV-cache decode
    zonos2::Zonos2NpyF32 prefill;    // [n0 + frames, 9, 1026], zonos2.cpp's single F32 prefill pass
    int steps = 0;                   // decode steps compared after the prefill
};

OracleRun load_backbone_oracle(const zonos2::Zonos2Assets & assets, const fs::path & oracle, bool speaker) {
    const auto & hp = assets.hparams;
    OracleRun out;
    zonos2::Zonos2PromptOptions options;
    options.speaker = speaker;
    out.prompt = zonos2::build_zonos2_prompt(hp, read_text(oracle / "text.txt"), options);
    out.ids = zonos2::read_f32_npy(oracle / (speaker ? "tts_spk_ids.npy" : "tts_ids.npy"));
    out.decode = zonos2::read_f32_npy(oracle / (speaker ? "decode_spk_logits.npy" : "decode_logits.npy"));
    out.prefill = zonos2::read_f32_npy(oracle / (speaker ? "val_spk" : "val") / "logits.npy");
    if (speaker) {
        out.speaker = zonos2::read_speaker_embedding_npy(oracle / "spk.npy");
    }
    const int width = hp.n_codebooks + 1;
    const int n0 = out.prompt.n_rows;
    if (out.ids.shape.size() != 2 || out.ids.shape[1] != width || out.ids.shape[0] <= n0) {
        throw std::runtime_error("oracle ids must be [rows, 10] with generated rows after the prompt");
    }
    for (size_t i = 0; i < out.prompt.ids.size(); ++i) {
        if (static_cast<float>(out.prompt.ids[i]) != out.ids.data[i]) {
            throw std::runtime_error("prompt ids differ from the oracle at row " + std::to_string(i / width) + " column " +
                                     std::to_string(i % width));
        }
    }
    const int64_t frames = out.ids.shape[0] - n0;
    if (out.decode.shape != std::vector<int64_t>{1 + frames, hp.n_codebooks, hp.audio_vocab} ||
        out.prefill.shape != std::vector<int64_t>{out.ids.shape[0], hp.n_codebooks, hp.audio_vocab}) {
        throw std::runtime_error("oracle logits must be [1 + frames, 9, 1026] (decode) and [rows, 9, 1026] (prefill)");
    }
    out.steps = std::min<int>(kMaxSteps, static_cast<int>(frames));
    return out;
}

// Prefill the oracle prompt, then teacher-force its generated rows; returns [1 + steps][9 * 1026] logits.
std::vector<float> teacher_forced_logits(std::shared_ptr<const zonos2::Zonos2Assets> assets, const OracleRun & run,
                                         const zonos2::Zonos2BackboneOptions & options) {
    const auto & hp = assets->hparams;
    const int width = hp.n_codebooks + 1;
    const size_t per_step = static_cast<size_t>(hp.audio_vocab) * static_cast<size_t>(hp.n_codebooks);
    engine::core::ExecutionContext execution(cpu_backend());
    zonos2::Zonos2Backbone backbone(std::move(assets), execution, options);
    // zonos2-cli's KV capacity: round_up_256(n0 + max + 8).
    backbone.begin(run.prompt.n_rows + kOracleMaxFrames + 8);
    std::vector<float> all, logits;
    all.reserve(per_step * static_cast<size_t>(run.steps + 1));
    backbone.prefill(run.prompt, run.speaker.empty() ? nullptr : run.speaker.data(), logits);
    all.insert(all.end(), logits.begin(), logits.end());
    std::vector<int32_t> row(static_cast<size_t>(width));
    for (int step = 0; step < run.steps; ++step) {
        const size_t at = static_cast<size_t>(run.prompt.n_rows + step) * static_cast<size_t>(width);
        for (int c = 0; c < width; ++c) row[static_cast<size_t>(c)] = static_cast<int32_t>(run.ids.data[at + static_cast<size_t>(c)]);
        backbone.step(row.data(), logits);
        all.insert(all.end(), logits.begin(), logits.end());
    }
    return all;
}

struct LogitMatch {
    double cos_min = 1.0;
    double argmax = 0.0;
    float max_diff = 0.0f;
};

// Minimum per-step cosine over all 9 x 1026 logits, per-codebook argmax agreement and the largest absolute
// difference; reference row first_row + s holds step s.
LogitMatch compare_logits(const std::vector<float> & ours, const zonos2::Zonos2NpyF32 & reference, int first_row,
                          int steps, int codebooks, int vocab) {
    const size_t per_step = static_cast<size_t>(codebooks) * static_cast<size_t>(vocab);
    LogitMatch out;
    size_t agree = 0;
    for (int s = 0; s <= steps; ++s) {
        const float * a = ours.data() + static_cast<size_t>(s) * per_step;
        const float * b = reference.data.data() + static_cast<size_t>(first_row + s) * per_step;
        out.cos_min = std::min(out.cos_min, cosine(a, b, per_step));
        for (size_t i = 0; i < per_step; ++i) out.max_diff = std::max(out.max_diff, std::fabs(a[i] - b[i]));
        for (int c = 0; c < codebooks; ++c) {
            const size_t o = static_cast<size_t>(c) * static_cast<size_t>(vocab);
            agree += argmax(a + o, static_cast<size_t>(vocab)) == argmax(b + o, static_cast<size_t>(vocab));
        }
    }
    out.argmax = static_cast<double>(agree) / static_cast<double>((steps + 1) * codebooks);
    return out;
}

// Resident: against zonos2.cpp's own prefill + KV-cache decode of the same rows (oracle_decode.cpp), cos_min >= 0.9999
// and argmax agreement >= 0.99. zonos2.cpp's single F32 prefill pass (val/logits.npy) is reported for information:
// its F16-cache flash-attention decode already departs from that pass on near-tied codes.
// --disk-experts: the expert-cache path (8 slots) must be bit-identical to resident experts; both run the same plain
// kernels.
int backbone(const fs::path & model, const fs::path & oracle, bool speaker, bool disk_experts) {
    const auto assets = zonos2::load_zonos2_assets(model);
    const auto run = load_backbone_oracle(*assets, oracle, speaker);
    const int codebooks = assets->hparams.n_codebooks;
    const int vocab = assets->hparams.audio_vocab;
    const auto resident = teacher_forced_logits(assets, run, {});
    const auto decode = compare_logits(resident, run.decode, 0, run.steps, codebooks, vocab);
    const auto prefill = compare_logits(resident, run.prefill, run.prompt.n_rows - 1, run.steps, codebooks, vocab);
    const char * label = speaker ? "zonos2 backbone speaker parity" : "zonos2 backbone parity";
    if (!disk_experts) {
        const bool pass = decode.cos_min >= 0.9999 && decode.argmax >= 0.99;
        std::printf("%s: %s cos_min=%.6f argmax=%.4f max_diff=%.3g steps=%d (f32 prefill pass: cos_min=%.6f argmax=%.4f)\n",
                    label, pass ? "PASS" : "FAIL", decode.cos_min, decode.argmax, decode.max_diff, run.steps,
                    prefill.cos_min, prefill.argmax);
        return pass ? 0 : 1;
    }
    zonos2::Zonos2BackboneOptions options;
    options.disk_experts = true;
    options.expert_cache_size = 8;
    const auto disk = teacher_forced_logits(assets, run, options);
    const auto disk_decode = compare_logits(disk, run.decode, 0, run.steps, codebooks, vocab);
    float max_diff = 0.0f;
    for (size_t i = 0; i < disk.size(); ++i) max_diff = std::max(max_diff, std::fabs(disk[i] - resident[i]));
    const bool pass = max_diff == 0.0f;
    std::printf("%s (disk experts): %s max_diff_vs_resident=%.3g oracle_cos_min=%.6f oracle_argmax=%.4f steps=%d\n", label,
                pass ? "PASS" : "FAIL", max_diff, disk_decode.cos_min, disk_decode.argmax, run.steps);
    return pass ? 0 : 1;
}

// DAC: the oracle's raw delayed codes (tts.npy, eos in tts.npy.eos.npy) un-sheared over [0, E) and decoded in one
// pass, against dac-cli's 16-bit WAV (equal length, SNR >= 40 dB after the same int16 quantization). Seam: the same
// frames in 40-frame tiles with up to 24 context frames on each side, clamped at both ends as the streaming session
// does, match the single pass within 1e-5.
int decoder(const fs::path & model, const fs::path & oracle) {
    constexpr int kCodebooks = 9;
    constexpr int kBlock = 40;
    constexpr int kContext = 24;
    const auto codes = zonos2::read_f32_npy(oracle / "tts.npy");
    const auto eos = zonos2::read_f32_npy(oracle / "tts.npy.eos.npy");
    const auto wav = engine::audio::read_wav_f32(oracle / "tts_audio.wav");
    if (codes.shape.size() != 2 || codes.shape[1] != kCodebooks) throw std::runtime_error("tts.npy must be [rows, 9]");
    if (eos.data.size() != 1) throw std::runtime_error("tts.npy.eos.npy must hold one value");
    if (wav.sample_rate != 44100 || wav.channels != 1) throw std::runtime_error("tts_audio.wav must be 44.1 kHz mono");
    const int rows = static_cast<int>(codes.shape[0]);
    const int eos_frame = static_cast<int>(std::lrint(eos.data[0]));
    const int frames = eos_frame >= 0 && eos_frame < rows ? eos_frame : rows;
    std::vector<int32_t> raw(codes.data.size());
    for (size_t i = 0; i < raw.size(); ++i) raw[i] = static_cast<int32_t>(std::lrint(codes.data[i]));
    engine::core::ExecutionContext execution(cpu_backend());
    zonos2::Zonos2CodecDecoder codec(engine::assets::open_tensor_source(model), execution);
    const auto all = zonos2::zonos2_undelay(raw.data(), rows, 0, frames);
    const auto single = codec.decode(all.data(), frames, 0, 0);
    std::vector<float> tiled;
    for (int lo = 0; lo < frames; lo += kBlock) {
        const int hi = std::min(frames, lo + kBlock);
        const int a = std::max(0, lo - kContext);
        const int b = std::min(frames, hi + kContext);
        const auto window = zonos2::zonos2_undelay(raw.data(), rows, a, b);
        const auto part = codec.decode(window.data(), b - a, lo - a, b - hi);
        tiled.insert(tiled.end(), part.begin(), part.end());
    }
    double seam_max = tiled.size() == single.size() ? 0.0 : 1.0;
    for (size_t i = 0; i < std::min(tiled.size(), single.size()); ++i) {
        seam_max = std::max(seam_max, static_cast<double>(std::fabs(single[i] - tiled[i])));
    }
    double signal = 0.0, noise = 0.0;
    const size_t n = std::min(single.size(), wav.samples.size());
    for (size_t i = 0; i < n; ++i) {
        // dac_wav_bytes: lrintf(clamp(v, -1, 1) * 32767); read_wav_f32: / 32768.
        const double ours = static_cast<double>(std::lrint(std::clamp(single[i], -1.0f, 1.0f) * 32767.0f)) / 32768.0;
        const double r = wav.samples[i];
        signal += r * r;
        noise += (ours - r) * (ours - r);
    }
    const double snr = 10.0 * std::log10(signal / std::max(noise, 1e-30));
    const bool same_length = single.size() == wav.samples.size() &&
                             single.size() == static_cast<size_t>(frames) * static_cast<size_t>(codec.samples_per_frame());
    const bool pass = same_length && snr >= 40.0 && seam_max <= 1e-5;
    std::printf("zonos2 decoder parity: %s snr=%.2f seam_max=%.3g samples=%zu oracle_samples=%zu frames=%d\n",
                pass ? "PASS" : "FAIL", snr, seam_max, single.size(), wav.samples.size(), frames);
    return pass ? 0 : 1;
}

// ECAPA: the exact 24 kHz samples spk-encoder-cli encoded (voice24k.f32, raw little-endian f32) against spk.npy;
// cosine >= 0.9999.
int speaker(const fs::path & model, const fs::path & oracle) {
    std::ifstream in(oracle / "voice24k.f32", std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("cannot open voice24k.f32");
    std::vector<float> samples(static_cast<size_t>(in.tellg()) / sizeof(float));
    in.seekg(0);
    in.read(reinterpret_cast<char *>(samples.data()), static_cast<std::streamsize>(samples.size() * sizeof(float)));
    const auto reference = zonos2::read_speaker_embedding_npy(oracle / "spk.npy");
    engine::core::ExecutionContext execution(cpu_backend());
    const zonos2::Zonos2SpeakerEncoder encoder(engine::assets::open_tensor_source(model), execution);
    const auto ours = encoder.embed(samples);
    const double cos = ours.size() == reference.size() ? cosine(ours.data(), reference.data(), ours.size()) : 0.0;
    const bool pass = cos >= 0.9999;
    std::printf("zonos2 speaker parity: %s cos=%.6f\n", pass ? "PASS" : "FAIL", cos);
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
            return backbone(args[1], args[2], false, disk);
        }
        if (args.size() == 3 && args[0] == "backbone_speaker") return backbone(args[1], args[2], true, false);
        if (args.size() == 3 && args[0] == "decoder") return decoder(args[1], args[2]);
        if (args.size() == 3 && args[0] == "speaker") return speaker(args[1], args[2]);
        std::fprintf(stderr,
                     "usage: zonos2_parity [--log] backbone <zonos2-*.gguf> <oracle_dir> [--disk-experts]\n"
                     "       zonos2_parity [--log] backbone_speaker <zonos2-*.gguf> <oracle_dir>\n"
                     "       zonos2_parity [--log] decoder <dac.gguf> <oracle_dir>\n"
                     "       zonos2_parity [--log] speaker <spk-encoder.gguf> <oracle_dir>\n");
        return 2;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "zonos2_parity: %s\n", error.what());
        return 1;
    }
}
