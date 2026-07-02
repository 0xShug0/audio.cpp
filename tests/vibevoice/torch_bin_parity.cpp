#include "engine/framework/assets/torch_bin.h"

#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    const std::string path = argc > 1
        ? argv[1]
        : "E:/REPOS/VIBEVOICE-MAIN/VibeVoice/loras/mp1/semantic_connector/pytorch_model.bin";
    try {
        const auto source = engine::assets::open_torch_bin_tensor_source(path);
        std::printf("torch .bin: %s\n", path.c_str());
        for (const auto & meta : source->tensors()) {
            const auto values = source->require_f32(meta.name);
            double sum = 0.0;
            for (const float value : values) {
                sum += value;
            }
            std::printf("%-16s dtype=%-4s shape=[", meta.name.c_str(), meta.dtype.c_str());
            for (size_t i = 0; i < meta.shape.size(); ++i) {
                std::printf("%s%lld", i == 0 ? "" : ",", static_cast<long long>(meta.shape[i]));
            }
            std::printf("] mean=%.8f first8=", sum / static_cast<double>(values.size()));
            for (size_t i = 0; i < values.size() && i < 8; ++i) {
                std::printf("%s%.6f", i == 0 ? "" : ",", values[i]);
            }
            std::printf("\n");
        }
    } catch (const std::exception & error) {
        std::fprintf(stderr, "torch_bin_parity failed: %s\n", error.what());
        return 1;
    }
    return 0;
}
