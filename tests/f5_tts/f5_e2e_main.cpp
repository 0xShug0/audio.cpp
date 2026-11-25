// E2E: Habibi Arabic synthesis via f5_synthesize + vocos, writes WAV.
#include "engine/community_models/f5_tts/synthesize.h"
#include "engine/framework/debug/trace.h"
#include <chrono>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

namespace {

// minimal WAV reader (16-bit PCM mono/stereo)
struct Wav {
    int sample_rate = 0;
    int channels = 1;
    std::vector<float> samples;  // mono mixdown
};

bool read_wav(const std::string & path, Wav & out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::vector<char> data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (data.size() < 44 || std::memcmp(data.data(), "RIFF", 4) != 0) return false;
    size_t pos = 12;
    int bits = 16;
    while (pos + 8 <= data.size()) {
        const std::string id(data.data() + pos, 4);
        const uint32_t sz = *reinterpret_cast<const uint32_t *>(data.data() + pos + 4);
        if (id == "fmt ") {
            const char * p = data.data() + pos + 8;  // chunk payload
            out.channels = *reinterpret_cast<const uint16_t *>(p + 2);
            out.sample_rate = *reinterpret_cast<const uint32_t *>(p + 4);
            bits = *reinterpret_cast<const uint16_t *>(p + 14);
        } else if (id == "data") {
            if (bits == 0) return false;
            const size_t n = sz / (bits / 8);
            const auto * p = reinterpret_cast<const int16_t *>(data.data() + pos + 8);
            out.samples.resize(n);
            for (size_t i = 0; i < n; ++i) {
                out.samples[i] = static_cast<float>(p[i]) / 32768.0F;
            }
            break;
        }
        pos += 8 + sz + (sz & 1);
    }
    return !out.samples.empty();
}

void write_wav(const std::string & path, const std::vector<float> & mono, int sr) {
    std::ofstream f(path, std::ios::binary);
    const uint32_t n = static_cast<uint32_t>(mono.size());
    const uint32_t data_bytes = n * 2;
    f.write("RIFF", 4);
    const uint32_t riff = 36 + data_bytes;
    f.write(reinterpret_cast<const char *>(&riff), 4);
    f.write("WAVEfmt ", 8);
    const uint32_t fmt = 16;
    f.write(reinterpret_cast<const char *>(&fmt), 4);
    const uint16_t pcm = 1, ch = 1;
    const uint16_t bps = 16;
    const uint32_t br = sr * 2;
    const uint16_t align = 2;
    f.write(reinterpret_cast<const char *>(&pcm), 2);
    f.write(reinterpret_cast<const char *>(&ch), 2);
    f.write(reinterpret_cast<const char *>(&sr), 4);
    f.write(reinterpret_cast<const char *>(&br), 4);
    f.write(reinterpret_cast<const char *>(&align), 2);
    f.write(reinterpret_cast<const char *>(&bps), 2);
    f.write("data", 4);
    f.write(reinterpret_cast<const char *>(&data_bytes), 4);
    for (const float v : mono) {
        const auto s = static_cast<int16_t>(std::clamp(v * 32767.0F, -32768.0F, 32767.0F));
        f.write(reinterpret_cast<const char *>(&s), 2);
    }
}

}  // namespace

int main(int argc, char ** argv) {
    engine::debug::configure_logging({true, std::nullopt});
    const auto started = std::chrono::steady_clock::now();
    if (argc < 6) {
        std::fprintf(stderr, "usage: %s <checkpoint> <vocos> <reference.wav> <output.wav> <reference-text> [device]\n", argv[0]);
        return 2;
    }
    const std::string model = argv[1];
    const std::string vocos = argv[2];
    const std::string ref = argv[3];
    const std::string out_path = argv[4];

    Wav ref_wav;
    if (!read_wav(ref, ref_wav)) {
        std::fprintf(stderr, "cannot read ref wav %s\n", ref.c_str());
        return 1;
    }
    std::printf("ref: %d Hz, %zu samples (%.2fs)\n", ref_wav.sample_rate, ref_wav.samples.size(),
                static_cast<double>(ref_wav.samples.size()) / ref_wav.sample_rate);

    engine::models::f5_tts::F5SynthesisRequest req;
    req.text = std::getenv("F5_TEXT") ? std::getenv("F5_TEXT")
        : "أهلا، هذه تجربة للنطق باللغة العربية، من نموذج حبيبي، داخل أوديو سي بي بي، على مجموعة جي بي يو بالبشرة.";
    req.dialect = std::getenv("F5_DIALECT") ? std::getenv("F5_DIALECT") : "UNK";
    req.ref_audio = ref_wav.samples;
    req.ref_sample_rate = ref_wav.sample_rate;
    req.ref_text = argv[5];
    if (std::getenv("F5_LONG") != nullptr) {
        // long-text test: ~4x the cap; exercises chunking + chaining
        req.text = req.text + " " + req.text + " " + req.text + " " + req.text;
    }
    req.steps = std::getenv("F5_STEPS") ? std::atoi(std::getenv("F5_STEPS")) : 16;
    req.cfg_strength = 2.0F;
    req.seed = 42;
    req.fixed_seed = std::getenv("F5_RANDSEED") == nullptr;  // F5_RANDSEED=1: random per run
    req.use_cuda = std::getenv("F5_CUDA") != nullptr;
    req.cuda_device = argc > 6 && std::string(argv[6]) != "--log" ? std::stoi(argv[6]) : 0;

    std::printf("synthesizing...\n");
    fflush(stdout);
    // first call = graph build; second = cached graphs (server steady state)
    const auto warm = engine::models::f5_tts::f5_synthesize(model, vocos, req);
    (void)warm;
    const auto result = engine::models::f5_tts::f5_synthesize(model, vocos, req);
    std::printf("generated %.2fs audio in %.2fs wall (%.2fx RTF)\n",
                static_cast<double>(result.audio.size()) / result.sample_rate,
                result.generation_seconds,
                result.generation_seconds / (static_cast<double>(result.audio.size()) / result.sample_rate));
    write_wav(out_path, result.audio, result.sample_rate);
    std::printf("written %s\n", out_path.c_str());
    engine::debug::timing_log_scalar("session.wall_ms",
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count());
    return 0;
}
