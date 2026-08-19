// E2E: Habibi Arabic synthesis via f5_synthesize + vocos, writes WAV.
#include "engine/community_models/f5_tts/synthesize.h"

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
    std::fprintf(stderr, "start\n"); fflush(stderr);
    const std::string model = argc > 1 ? argv[1] : "/mnt/ai/models/Habibi-TTS/Unified/model_200000.safetensors";
    const std::string vocos = argc > 2 ? argv[2] : "/mnt/ai/models/vocos-mel-24khz/vocos.safetensors";
    const std::string ref = argc > 3 ? argv[3] : "/tmp/f5ref/pkg/habibi/habibi_tts/assets/IRQ.wav";
    const std::string out_path = argc > 4 ? argv[4] : "/tmp/habibi_e2e.wav";

    Wav ref_wav;
    if (!read_wav(ref, ref_wav)) {
        std::fprintf(stderr, "cannot read ref wav %s\n", ref.c_str());
        return 1;
    }
    std::printf("ref: %d Hz, %zu samples (%.2fs)\n", ref_wav.sample_rate, ref_wav.samples.size(),
                static_cast<double>(ref_wav.samples.size()) / ref_wav.sample_rate);

    engine::models::f5_tts::F5SynthesisRequest req;
    req.text = "\xD8\xA3\xD9\x87\xD9\x84\xD8\xA7\xD9\x8B\xD8\x8C \xD9\x87\xD8\xB0\xD9\x87 "
               "\xD8\xAA\D8\xAC\xD8\xB1\xD8\xA8\xD8\xA9 \xD9\x84\xD9\x84\xD9\x86\xD8\xB7\xD9\x82 "
               "\xD8\xA8\xD8\xA7\xD9\x84\xD9\x84\xD8\xBA\xD8\xA9 \xD8\xA7\xD9\x84\xD8\xB9\xD8\xB1\xD8\xA8\xD9\x8A\xD8\xA9\xD8\x8C "
               "\xD9\x85\xD9\x86 \xD9\x86\xD9\x85\xD9\x88\xD8\xB0\xD8\xAC \xD9\x87\xD8\xA8\xD9\x8A\xD8\xA8\xD9\x8A\xD8\x8C "
               "\xD8\xAF\xD8\xA7\xD8\xAE\xD9\x84 \xD8\xA3\xD9\x88\xD8\xAF\xD9\x8A\xD9\x88 \xD8\xB3\xD9\x8A \xD8\xA8\xD9\x8A \xD8\xA8\xD9\x8A\xD8\x8C "
               "\xD8\xB9\xD9\x84\xD9\x89 \xD9\x85\xD8\xAC\xD9\x85\xD9\x88\xD8\xB9\xD8\xA9 \xD8\xAC\xD9\x8A \xD9\xBE\xD9\x8A \D9\x8A\xD9\x88 "
               "\xD8\xA8\xD8\xA7\xD9\x84\xD8\xA8\xD9\x88\xD8\xB4\xD8\xB1\xD8\xB9.\n";
    req.dialect = "UNK";
    req.ref_audio = ref_wav.samples;
    req.ref_sample_rate = ref_wav.sample_rate;
    req.ref_text = "\xD9\x83\xD8\xA7\xD9\x86\x20\xD8\xA7\xD9\x84\xD9\x84\xD8\xB9\xD9\x8A\xD8\xA8\x20\xD8\xAD\xD8\xA7\xD8\xB6\xD8\xB1\xD9\x8B\xD8\xA7\x2E";
    if (std::getenv("F5_LONG") != nullptr) {
        // long-text test: ~4x the cap; exercises chunking + chaining
        req.text = req.text + " " + req.text + " " + req.text + " " + req.text;
    }
    req.steps = 16;
    req.cfg_strength = 2.0F;
    req.seed = 42;
    req.fixed_seed = true;
    req.use_cuda = std::getenv("F5_CUDA") != nullptr;
    req.cuda_device = 1;

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
    return 0;
}
