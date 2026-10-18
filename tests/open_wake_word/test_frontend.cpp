#include "engine/framework/assets/tensor_source.h"
#include "engine/models/open_wake_word/frontend.h"

#include <cmath>
#include <iostream>

int main(int argc, char **argv) {
    try {
        if (argc != 2) {
            throw std::runtime_error("usage: open_wake_word_frontend_test TRACE_GGUF");
        }
        const auto source = engine::assets::open_tensor_source(argv[1]);
        engine::models::open_wake_word::Frontend frontend(
            engine::models::open_wake_word::load_frontend_weights(*source));
        double max_error = 0, error = 0, energy = 0;
        for (int index = 0; index < 8; ++index) {
            const auto prefix = "reference.mel." + std::to_string(index);
            const auto input = source->require_f32(prefix + ".input");
            auto expected = source->require_f32(prefix + ".output");
            const auto actual = frontend.extract(input, 1);
            if (actual.values.size() != expected.size()) {
                throw std::runtime_error("frontend shape mismatch");
            }
            for (size_t i = 0; i < expected.size(); ++i) {
                expected[i] = expected[i] / 10.0F + 2.0F;
                const double delta = double(actual.values[i]) - expected[i];
                if (!std::isfinite(delta)) {
                    throw std::runtime_error("non-finite frontend output");
                }
                max_error = std::max(max_error, std::abs(delta));
                error += delta * delta;
                energy += double(expected[i]) * expected[i];
            }
        }
        std::cout << "frontend.max_abs=" << max_error << " relative_l2=" << std::sqrt(error / energy) << '\n';
        if (max_error > 1e-3 || std::sqrt(error / energy) > 1e-4) {
            throw std::runtime_error("frontend parity failed");
        }
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
