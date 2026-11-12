// ZONOS3 parity against the native ZONOS3 runtime's outputs (tests/community_models/zonos3/make_oracle.sh).
//   zonos3_parity backbone <backbone.gguf> <oracle_dir> [--disk-experts]
#include "engine/community_models/zonos3/assets.h"
#include "engine/community_models/zonos3/backbone.h"
#include "engine/community_models/zonos3/npy.h"
#include "engine/community_models/zonos3/prompt.h"
#include "engine/community_models/zonos3/tokenizer.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/debug/trace.h"

#include <algorithm>
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
// bit-identical to the resident path with plain expert kernels, the kernels the expert cache runs; the oracle
// numbers are reported (ZONOS3's CPU runtime uses repacked expert kernels, which round differently).
int backbone(const fs::path & model, const fs::path & oracle, bool disk_experts) {
    const auto assets = zonos3::load_zonos3_assets(model);
    const auto run = load_backbone_oracle(*assets, oracle);
    const size_t vocab = static_cast<size_t>(assets->hparams.audio_vocab);
    zonos3::Zonos3BackboneOptions options;
    if (disk_experts) {
        options.disk_experts = true;
        options.expert_cache_size = 8;
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
        std::cerr << "usage: zonos3_parity [--log] backbone <backbone.gguf> <oracle_dir> [--disk-experts]\n";
        return 2;
    } catch (const std::exception & error) {
        std::cerr << "zonos3_parity: " << error.what() << "\n";
        return 1;
    }
}
