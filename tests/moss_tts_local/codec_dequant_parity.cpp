// Parity harness for the MOSS-Audio-Tokenizer-v2 RLFQ dequantizer: runs the
// dequant on a fixed code matrix and dumps the latent for comparison against
// the Python reference (scripts/codec_dequant_ref.py).

#include "engine/models/moss/shared/audio_tokenizer_quantizer.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    const std::filesystem::path codec_dir =
        argc > 1 ? argv[1]
                 : "C:/Users/justi/.cache/huggingface/hub/models--OpenMOSS-Team--MOSS-Audio-Tokenizer-v2/"
                   "snapshots/f6e20e543b33d2c252a7ef71bdf8aa71e5ff9169";
    const std::filesystem::path out_path =
        argc > 2 ? argv[2]
                 : "C:/Users/justi/AppData/Local/Temp/claude/E--REPOS-audio-cpp/"
                   "62af4e53-c9e0-4e66-ac0e-27e93cec72c9/scratchpad/cpp_latent.txt";

    constexpr int64_t kNumQuantizers = 12;
    constexpr int64_t kSteps = 8;

    std::vector<std::vector<int32_t>> codes(kNumQuantizers, std::vector<int32_t>(kSteps));
    for (int64_t i = 0; i < kNumQuantizers; ++i) {
        for (int64_t t = 0; t < kSteps; ++t) {
            codes[static_cast<size_t>(i)][static_cast<size_t>(t)] = static_cast<int32_t>((i * 37 + t * 5) % 1024);
        }
    }

    try {
        engine::models::moss::MossAudioTokenizerQuantizer dequantizer(codec_dir, kNumQuantizers);
        const std::vector<float> latent = dequantizer.decode(codes);  // [code_dim, steps]
        const int64_t code_dim = dequantizer.code_dim();

        double mean = 0.0;
        for (const float value : latent) {
            mean += value;
        }
        mean /= static_cast<double>(latent.size());
        double var = 0.0;
        for (const float value : latent) {
            var += (value - mean) * (value - mean);
        }
        var /= static_cast<double>(latent.size());

        std::printf("shape %lld %lld\n", static_cast<long long>(code_dim), static_cast<long long>(kSteps));
        std::printf("first16");
        for (int i = 0; i < 16; ++i) {
            std::printf(" %.6f", latent[static_cast<size_t>(i)]);
        }
        std::printf("\n");
        std::printf("mean %.6f std %.6f\n", mean, std::sqrt(var));

        std::ofstream out(out_path);
        out.precision(8);
        for (const float value : latent) {
            out << value << "\n";
        }
        std::printf("wrote %lld values to %s\n", static_cast<long long>(latent.size()), out_path.string().c_str());
    } catch (const std::exception & error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
    return 0;
}
