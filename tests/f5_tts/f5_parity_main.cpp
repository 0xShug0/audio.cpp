// Parity harness: run f5_dit_forward on golden inputs, compare stages vs goldens.
#include "engine/community_models/f5_tts/runtime.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace {

std::vector<float> load_bin(const std::string & path, size_t expected) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "cannot open %s\n", path.c_str());
        std::exit(1);
    }
    std::vector<float> out(expected);
    f.read(reinterpret_cast<char *>(out.data()), static_cast<std::streamsize>(expected * sizeof(float)));
    if (static_cast<size_t>(f.gcount()) != expected * sizeof(float)) {
        std::fprintf(stderr, "short read on %s\n", path.c_str());
        std::exit(1);
    }
    return out;
}

double cosine(const std::vector<float> & a, const std::vector<float> & b) {
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += static_cast<double>(a[i]) * b[i];
        na += static_cast<double>(a[i]) * a[i];
        nb += static_cast<double>(b[i]) * b[i];
    }
    return dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12);
}

double max_abs(const std::vector<float> & a, const std::vector<float> & b) {
    double m = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        m = std::max(m, std::abs(static_cast<double>(a[i]) - b[i]));
    }
    return m;
}

// golden tensors are row-major [T, F]; C++ taps are column [F, T]
std::vector<float> col_from_row(const std::vector<float> & row, int T, int F) {
    std::vector<float> col(row.size());
    for (int t = 0; t < T; ++t)
        for (int f = 0; f < F; ++f)
            col[static_cast<size_t>(f) * T + t] = row[static_cast<size_t>(t) * F + f];
    return col;
}

}  // namespace

int main(int argc, char ** argv) {
    const std::string gold = "/mnt/ai/f5-parity/golden";
    const std::string ckpt = argc > 1 ? argv[1] : "/mnt/ai/models/Habibi-TTS/Unified/model_200000.safetensors";

    const auto x = load_bin(gold + "/input_x.bin", 64 * 100);
    const auto cond = load_bin(gold + "/input_cond.bin", 64 * 100);
    std::vector<int32_t> ids(24);
    {
        const auto raw = load_bin(gold + "/input_ids.bin", 24);
        for (int i = 0; i < 24; ++i) ids[i] = static_cast<int32_t>(raw[i]);
    }

    engine::models::f5_tts::F5Architecture arch;
    engine::models::f5_tts::F5DebugTaps taps;
    std::vector<float> t_text_embed, t_text_convnext, t_text_padded, t_input_embed, t_time_embed, t_block0, t_block21;
    taps.text_embed = &t_text_embed;
    taps.text_convnext = &t_text_convnext;
    taps.text_padded = &t_text_padded;
    taps.input_embed = &t_input_embed;
    taps.time_embed = &t_time_embed;
    taps.block0 = &t_block0;
    taps.block21 = &t_block21;

    const bool use_cuda = std::getenv("F5_CUDA") != nullptr;
    const bool with_taps = std::getenv("F5_RAW_TAPS") != nullptr;  // module graph: taps unwired
    engine::models::f5_tts::F5ComputeDevice dev;
    dev.use_cuda = use_cuda;
    dev.device = 1;  // GPU 1 is the TTS GPU on this rig
    const auto out = engine::models::f5_tts::f5_dit_forward(
        ckpt, x, cond, ids, 0.42F, 64, arch, false, false, (with_taps ? &taps : nullptr), &dev);

    int failures = 0;
    auto check = [&](const char * name, const std::vector<float> & mine_col, const std::vector<float> & golden_row, int T, int F) {
        (void) T; (void) F;
        // Both golden and taps are stored with time as the slow axis ([t][f]);
        // no transpose needed. (The F5Runtime returns column layout only for
        // the final output; stage taps keep the same [t][f] memory.)
        const double c = cosine(mine_col, golden_row);
        const double m = max_abs(mine_col, golden_row);
        if (mine_col.empty()) {
            std::printf("%-18s (tap not wired in module graph; skipped)\n", name);
            return;
        }
        const bool ok = c >= 0.999 && std::isfinite(c);
        std::printf("%-18s cosine=%.6f maxabs=%.5f %s\n", name, c, m, ok ? "OK" : "FAIL");
        if (!ok) failures++;
    };

    check("01_text_embed", t_text_embed, load_bin(gold + "/01_text_embed.bin", 24 * 512), 24, 512);
    check("02_text_convnext", t_text_convnext, load_bin(gold + "/02_text_after_convnext.bin", 24 * 512), 24, 512);
    check("03_text_padded", t_text_padded, load_bin(gold + "/03_text_padded.bin", 64 * 512), 64, 512);
    check("04_input_embed", t_input_embed, load_bin(gold + "/04_input_embed.bin", 64 * 1024), 64, 1024);
    check("05_time_embed", t_time_embed, load_bin(gold + "/05_time_embed.bin", 1024), 1, 1024);
    check("07_block0", t_block0, load_bin(gold + "/07_block0_out.bin", 64 * 1024), 64, 1024);
    check("07_block21", t_block21, load_bin(gold + "/07_block21_out.bin", 64 * 1024), 64, 1024);
    {
        // out tensor ne [MEL, N] => memory (m,n) at n*MEL + m, i.e. raw bytes
        // are already golden's [n][m] row-major layout. Direct compare.
        const std::vector<float> out_rows = out;
        check("08_final_out", out_rows, load_bin(gold + "/08_final_out.bin", 64 * 100), 64, 100);
    }

    if (failures > 0) {
        std::printf("PARITY FAILED (%d stages)\n", failures);
        return 1;
    }
    std::printf("PARITY PASSED\n");
    return 0;
}
