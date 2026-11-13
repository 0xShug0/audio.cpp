// ZONOS2 parity against Zyphra/zonos2.cpp's outputs (tests/community_models/zonos2/make_oracle.sh).
//   zonos2_parity backbone <zonos2-*.gguf> <oracle_dir> [--disk-experts]
//   zonos2_parity backbone_speaker <zonos2-*.gguf> <oracle_dir>
#include "engine/community_models/zonos2/assets.h"
#include "engine/community_models/zonos2/backbone.h"
#include "engine/community_models/zonos2/npy.h"
#include "engine/community_models/zonos2/prompt.h"
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
        std::fprintf(stderr,
                     "usage: zonos2_parity [--log] backbone <zonos2-*.gguf> <oracle_dir> [--disk-experts]\n"
                     "       zonos2_parity [--log] backbone_speaker <zonos2-*.gguf> <oracle_dir>\n");
        return 2;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "zonos2_parity: %s\n", error.what());
        return 1;
    }
}
