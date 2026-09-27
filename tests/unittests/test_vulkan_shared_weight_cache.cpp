#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/core/shared_weight_cache.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <future>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {

struct Weights {
    std::shared_ptr<engine::core::BackendWeightStore> store;
    engine::core::TensorValue matrix;
};

struct Buffer {
    ggml_backend_buffer_t value = nullptr;
    ~Buffer() { if (value != nullptr) ggml_backend_buffer_free(value); }
};

void infer(engine::core::BackendConfig config, const std::shared_ptr<const Weights> & weights) {
    engine::core::ExecutionContext execution(config);
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(
        ggml_init({2 * 1024 * 1024, nullptr, true}), ggml_free);
    if (ctx == nullptr) throw std::runtime_error("graph context allocation failed");
    auto * input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 32, 1);
    ggml_set_input(input);
    auto * output = ggml_mul_mat(ctx.get(), weights->matrix.tensor, input);
    ggml_set_output(output);
    auto * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, output);
    Buffer buffer{ggml_backend_alloc_ctx_tensors(ctx.get(), execution.backend())};
    if (buffer.value == nullptr) throw std::runtime_error("graph buffer allocation failed");
    std::vector<float> values(32), actual(32);
    for (int run = 0; run < 16; ++run) {
        for (int k = 0; k < 32; ++k) values[k] = float((k + run) % 5 - 2);
        ggml_backend_tensor_set(input, values.data(), 0, values.size() * sizeof(float));
        if (engine::core::compute_backend_graph(execution.backend(), graph) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("shared-weight graph compute failed");
        ggml_backend_synchronize(execution.backend());
        ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
        for (int row = 0; row < 32; ++row) {
            float expected = 0.0F;
            for (int k = 0; k < 32; ++k) expected += float((row * 32 + k) % 7 - 3) * 0.125F * values[k];
            if (!std::isfinite(actual[row]) || std::abs(actual[row] - expected) > 1e-5F)
                throw std::runtime_error("shared Vulkan weights changed or lost owner-backend lifetime");
        }
    }
}

} // namespace

int main() {
    try {
        const auto devices = engine::core::list_backend_devices();
        auto device = std::find_if(devices.begin(), devices.end(), [](const auto & d) {
            return d.backend == "Vulkan" && d.type == "GPU";
        });
        if (device == devices.end()) {
            device = std::find_if(devices.begin(), devices.end(), [](const auto & d) { return d.backend == "Vulkan"; });
        }
        if (device == devices.end()) return 77;
        engine::core::BackendConfig config{engine::core::BackendType::Vulkan, device->index, 2};
        auto cache = std::make_unique<engine::core::SharedWeightCache>();
        std::weak_ptr<const Weights> retained;
        std::atomic<int> loads{0};
        {
            engine::core::ExecutionContext owner(config);
            retained = cache->get_or_load<Weights>("matrix", [&] {
                ++loads;
                Weights result;
                result.store = std::make_shared<engine::core::BackendWeightStore>(
                    owner.backend(), config.type, "test.shared.weights", 1024 * 1024);
                std::vector<float> values(32 * 32);
                for (int i = 0; i < 32 * 32; ++i) values[i] = float(i % 7 - 3) * 0.125F;
                result.matrix = result.store->make_f32(engine::core::TensorShape::from_dims({32, 32}), std::move(values));
                result.store->upload();
                return result;
            });
        } // Destroy the uploading backend before either independent graph runs.
        const auto reuse = [&] {
            auto weights = cache->get_or_load<Weights>("matrix", []() -> Weights {
                throw std::runtime_error("weights uploaded again");
            });
            infer(config, weights);
        };
        auto first = std::async(std::launch::async, reuse);
        auto second = std::async(std::launch::async, reuse);
        first.get(); second.get();
        if (loads != 1 || retained.expired()) throw std::runtime_error("weight cache ownership failed");
        cache.reset();
        if (!retained.expired()) throw std::runtime_error("package unload did not release Vulkan weights");
        std::cout << "vulkan_shared_weight_cache_test: ok on " << device->name << '\n';
        return 0;
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
