// Tokenizer parity: the C++ text pipeline (dialect wrap + ref trailing-space
// rule + UTF-8 char tokenization) must reproduce python list_str_to_idx ids
// exactly. See tests/f5_tts/README.md for the reference schema.
#include "engine/community_models/f5_tts/synthesize.h"
#include "engine/framework/debug/trace.h"
#include <chrono>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <stdexcept>
#include <vector>

namespace {

std::string read_text(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "cannot open %s\n", path.c_str());
        std::exit(1);
    }
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

std::vector<int32_t> read_ids(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "cannot open %s\n", path.c_str());
        std::exit(1);
    }
    const std::string raw((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::vector<int32_t> ids(raw.size() / 4);
    std::memcpy(ids.data(), raw.data(), ids.size() * 4);
    return ids;
}

}  // namespace

int main(int argc, char ** argv) {
    engine::debug::configure_logging({true, std::nullopt});
    const auto started = std::chrono::steady_clock::now();
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <reference-dir> <checkpoint>\n", argv[0]);
        return 2;
    }
    const std::string gold = argv[1];
    const std::string model = argv[2];

    const int count = std::atoi(read_text(gold + "/count").c_str());
    if (count <= 0) throw std::runtime_error("reference count must be positive");
    int failures = 0;
    for (int i = 0; i < count; ++i) {
        const std::string base = gold + "/case" + std::to_string(i);
        const std::string dialect = read_text(base + ".dialect");
        const std::string ref = read_text(base + ".ref");
        const std::string gen = read_text(base + ".gen");
        const auto want = read_ids(base + ".ids");

        const auto got = engine::models::f5_tts::f5_test_token_ids(model, dialect, ref, gen);
        bool ok = got.size() == want.size();
        size_t first_bad = 0;
        if (ok) {
            for (size_t k = 0; k < want.size(); ++k) {
                if (got[k] != want[k]) {
                    ok = false;
                    first_bad = k;
                    break;
                }
            }
        }
        std::printf("case%d dialect=%-3s ids=%zu %s\n", i, dialect.c_str(), want.size(),
                    ok ? "OK" : "FAIL");
        if (!ok) {
            ++failures;
            if (got.size() != want.size()) {
                std::printf("  length mismatch: got %zu want %zu\n", got.size(), want.size());
            } else {
                std::printf("  first mismatch at %zu: got %d want %d\n",
                            first_bad, got[first_bad], want[first_bad]);
            }
        }
    }
    if (failures > 0) {
        std::printf("TOKENIZER PARITY FAILED (%d/%d cases)\n", failures, count);
        return 1;
    }
    std::printf("TOKENIZER PARITY PASSED (%d cases)\n", count);
    engine::debug::timing_log_scalar("session.wall_ms",
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count());
    return 0;
}
