#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/io/json.h"
#include "engine/models/open_wake_word/runtime.h"
#include <gguf.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>

int main(int argc, char **argv) {
    try {
        if (argc != 3) {
            throw std::runtime_error("usage: open_wake_word_runtime_test TRACE_GGUF THREADS");
        }
        auto source = engine::assets::open_tensor_source(argv[1]);
        std::unique_ptr<gguf_context, decltype(&gguf_free)> metadata(gguf_init_from_file(argv[1], {true, nullptr}),
                                                                     gguf_free);
        if (!metadata) {
            throw std::runtime_error("invalid trace GGUF");
        }
        const auto config_key = gguf_find_key(metadata.get(), "open_wake_word.config");
        const auto count_key = gguf_find_key(metadata.get(), "reference.embedding.count");
        if (config_key < 0 || count_key < 0) {
            throw std::runtime_error("missing embedding trace");
        }
        const auto config = engine::io::json::parse(gguf_get_val_str(metadata.get(), config_key));
        const auto count = gguf_get_val_u32(metadata.get(), count_key);
        engine::core::BackendConfig backend;
        backend.type = engine::core::BackendType::Cpu;
        backend.threads = std::stoi(argv[2]);
        engine::core::ExecutionContext execution(backend);
        const auto weights = engine::models::open_wake_word::load_weights(*source, config);
#ifndef ENGINE_HAVE_NEON
        bool rejected = false;
        try {
            engine::models::open_wake_word::Runtime unavailable(*weights, execution, true);
        } catch (const std::runtime_error &) {
            rejected = true;
        }
        if (!rejected) throw std::runtime_error("disabled native embedding was accepted");
#endif
        for (const bool native : {false, true}) {
#ifndef ENGINE_HAVE_NEON
            if (native) continue;
#endif
            engine::models::open_wake_word::Runtime runtime(*weights, execution, native);
            double maximum = 0, error = 0, energy = 0;
            for (uint32_t i = 0; i < count; ++i) {
                const auto prefix = "reference.embedding." + std::to_string(i);
                const auto input = source->require_f32(prefix + ".input");
                const auto expected = source->require_f32(prefix + ".output");
                const auto actual = runtime.encode(input);
                if (actual.size() != expected.size()) {
                    throw std::runtime_error("embedding shape mismatch");
                }
                for (size_t j = 0; j < actual.size(); ++j) {
                    const double delta = double(actual[j]) - expected[j];
                    if (!std::isfinite(delta)) {
                        throw std::runtime_error("non-finite embedding");
                    }
                    maximum = std::max(maximum, std::abs(delta));
                    error += delta * delta;
                    energy += double(expected[j]) * expected[j];
                }
            }
            std::cout << "embedding." << (native ? "native" : "graph") << " max_abs=" << maximum
                      << " relative_l2=" << std::sqrt(error / energy) << '\n';
            if (maximum > 1e-3 || std::sqrt(error / energy) > 1e-4) {
                throw std::runtime_error("embedding parity failed");
            }
            const auto audio = source->require_f32("reference.probabilities.0.input");
            const auto expected = source->require_f32("reference.probabilities.0.output");
            auto scores = runtime.push(audio);
            const auto tail = runtime.finish();
            scores.insert(scores.end(), tail.begin(), tail.end());
            if (scores.size() != expected.size()) {
                throw std::runtime_error("pipeline score count mismatch");
            }
            maximum = 0;
            for (size_t i = 0; i < scores.size(); ++i) {
                maximum = std::max(maximum, std::abs(double(scores[i].probability) - expected[i]));
            }
            std::cout << "pipeline.max_score_error=" << maximum << '\n';
            if (maximum > 1e-4) {
                throw std::runtime_error("pipeline probability parity failed");
            }
            runtime.reset();
            scores.clear();
            for (size_t offset = 0; offset < audio.size(); offset += 137) {
                const auto end = std::min(audio.size(), offset + 137);
                const auto part = runtime.push(std::vector<float>(audio.begin() + offset, audio.begin() + end));
                scores.insert(scores.end(), part.begin(), part.end());
            }
            const auto streamed_tail = runtime.finish();
            scores.insert(scores.end(), streamed_tail.begin(), streamed_tail.end());
            if (scores.size() != expected.size()) {
                throw std::runtime_error("streaming score count mismatch");
            }
            for (size_t i = 0; i < scores.size(); ++i) {
                if (std::abs(double(scores[i].probability) - expected[i]) > 1e-4) {
                    throw std::runtime_error("streaming probability parity failed");
                }
            }
        }
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
