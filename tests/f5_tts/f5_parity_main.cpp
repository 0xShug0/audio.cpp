// Compare the production final forward against the independent upstream reference.
#include "engine/community_models/f5_tts/runtime.h"
#include "engine/framework/debug/trace.h"
#include <chrono>
#include <cerrno>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
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
        const double error = std::abs(static_cast<double>(a[i]) - b[i]);
        if (!std::isfinite(error)) return std::numeric_limits<double>::infinity();
        m = std::max(m, error);
    }
    return m;
}

}  // namespace

int main(int argc, char ** argv) {
    engine::debug::configure_logging({true, std::nullopt});
    const auto started = std::chrono::steady_clock::now();
    if (argc < 5 || argc > 7 || std::string(argv[3]) != "--max-abs-error" ||
        (argc == 7 && std::string(argv[6]) != "--log")) {
        std::fprintf(stderr, "usage: %s <checkpoint> <reference-dir> --max-abs-error <bound> [device] [--log]\n", argv[0]);
        return 2;
    }
    char * bound_end = nullptr;
    errno = 0;
    const double max_abs_error = std::strtod(argv[4], &bound_end);
    if (errno == ERANGE || bound_end == argv[4] || *bound_end != '\0' ||
        !std::isfinite(max_abs_error) || max_abs_error < 0.0) {
        std::fprintf(stderr, "--max-abs-error requires a finite non-negative acceptance bound\n");
        return 2;
    }
    int device = 0;
    if (argc > 5 && std::string(argv[5]) != "--log") {
        char * device_end = nullptr;
        errno = 0;
        const long parsed = std::strtol(argv[5], &device_end, 10);
        if (errno == ERANGE || device_end == argv[5] || *device_end != '\0' || parsed < 0 ||
            parsed > std::numeric_limits<int>::max()) {
            std::fprintf(stderr, "device must be a non-negative integer\n");
            return 2;
        }
        device = static_cast<int>(parsed);
    } else if (argc == 7) {
        std::fprintf(stderr, "unexpected argument after --log\n");
        return 2;
    }
    const std::string ckpt = argv[1];
    const std::string gold = argv[2];

    const auto x = load_bin(gold + "/input_x.bin", 64 * 100);
    const auto cond = load_bin(gold + "/input_cond.bin", 64 * 100);
    std::vector<int32_t> ids(24);
    {
        const auto raw = load_bin(gold + "/input_ids.bin", 24);
        for (int i = 0; i < 24; ++i) ids[i] = static_cast<int32_t>(raw[i]);
    }

    engine::models::f5_tts::F5Architecture arch;
    engine::models::f5_tts::F5ComputeDevice dev;
    dev.use_cuda = std::getenv("F5_CUDA") != nullptr;
    dev.device = device;
    const auto out = engine::models::f5_tts::f5_dit_forward(
        ckpt, x, cond, ids, 0.42F, 64, arch, false, false, nullptr, &dev);

    int failures = 0;
    auto check = [&](const char * name, const std::vector<float> & mine_col, const std::vector<float> & golden_row) {
        if (mine_col.empty() || mine_col.size() != golden_row.size()) {
            std::fprintf(stderr, "%s: empty output or mismatched tensor size\n", name);
            ++failures;
            return;
        }
        const double c = cosine(mine_col, golden_row);
        const double m = max_abs(mine_col, golden_row);
        const bool ok = c >= 0.999 && std::isfinite(c) && m <= max_abs_error;
        std::printf("%-18s cosine=%.6f maxabs=%.9g bound=%.9g %s\n", name, c, m, max_abs_error, ok ? "OK" : "FAIL");
        if (!ok) failures++;
    };

    // Final output bytes already match the reference's [time][mel] layout.
    check("08_final_out", out, load_bin(gold + "/08_final_out.bin", 64 * 100));

    if (failures > 0) {
        std::printf("PARITY FAILED (%d outputs)\n", failures);
        return 1;
    }
    std::printf("PARITY PASSED\n");
    engine::debug::timing_log_scalar("session.wall_ms",
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count());
    return 0;
}
