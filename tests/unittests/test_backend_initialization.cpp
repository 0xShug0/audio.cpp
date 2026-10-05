#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"

#include <algorithm>
#include <cmath>
#include <future>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using engine::core::ExecutionContext;
struct Buffer {
    ggml_backend_buffer_t value;
    ~Buffer() { if (value) ggml_backend_buffer_free(value); }
};
void infer(ExecutionContext & execution, ggml_tensor * weights) {
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(
        ggml_init({2 * 1024 * 1024, nullptr, true}), ggml_free);
    if (!ctx) throw std::runtime_error("graph context allocation failed");
    auto * input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 32, 1);
    ggml_set_input(input);
    auto * output = ggml_mul_mat(ctx.get(), weights, input);
    ggml_set_output(output);
    auto * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, output);
    Buffer buffer{ggml_backend_alloc_ctx_tensors(ctx.get(), execution.backend())};
    if (!buffer.value) throw std::runtime_error("graph allocation failed");
    std::vector<float> values(32), actual(32);
    for (int run = 0; run < 16; ++run) {
        for (int k = 0; k < 32; ++k) values[k] = float((k + run) % 5 - 2);
        ggml_backend_tensor_set(input, values.data(), 0, values.size() * sizeof(float));
        if (engine::core::compute_backend_graph(execution.backend(), graph) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("graph compute failed");
        ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
        for (int row = 0; row < 32; ++row) {
            float expected = 0.0F;
            for (int k = 0; k < 32; ++k) expected += float((row * 32 + k) % 7 - 3) * 0.125F * values[k];
            if (!std::isfinite(actual[row]) || std::abs(actual[row] - expected) > 1e-5F)
                throw std::runtime_error("independent backend could not use the shared device weights");
        }
    }
}
}

int main(int argc, char ** argv) {
    try {
        using engine::core::BackendType;
        engine::core::BackendConfig config{BackendType::Cpu, 0, 2};
        if (argc > 1 && std::string(argv[1]) == "vulkan") {
            const auto devices = engine::core::list_backend_devices();
            auto device = std::find_if(devices.begin(), devices.end(), [](const auto & d) {
                return d.backend == "Vulkan" && d.type == "GPU";
            });
            if (device == devices.end())
                device = std::find_if(devices.begin(), devices.end(), [](const auto & d) { return d.backend == "Vulkan"; });
            if (device == devices.end()) return 77;
            config = {BackendType::Vulkan, device->index, 2};
            std::cout << "device=" << device->name << '\n';
        }
        // A failure must release the initialization guard before the cold wave.
        bool rejected = false;
        try { ExecutionContext invalid({BackendType::Vulkan, -1, 2}); }
        catch (const std::runtime_error &) { rejected = true; }
        if (!rejected) throw std::runtime_error("negative device index was accepted");

        std::promise<void> start;
        auto ready = start.get_future().share();
        std::vector<std::future<std::unique_ptr<ExecutionContext>>> cold;
        for (int i = 0; i < 8; ++i) cold.push_back(std::async(std::launch::async, [=] {
            ready.wait();
            return std::make_unique<ExecutionContext>(config);
        }));
        start.set_value();
        std::vector<std::unique_ptr<ExecutionContext>> contexts;
        for (auto & job : cold) contexts.push_back(job.get());
        auto store = std::make_shared<engine::core::BackendWeightStore>(
            contexts.front()->backend(), config.type, "cold-initialization-test", 1024 * 1024);
        std::vector<float> values(32 * 32);
        for (int i = 0; i < 32 * 32; ++i) values[i] = float(i % 7 - 3) * 0.125F;
        auto matrix = store->make_f32(engine::core::TensorShape::from_dims({32, 32}), std::move(values));
        store->upload();
        std::vector<std::future<void>> jobs;
        for (auto & context : contexts) {
            auto * execution = context.get();
            jobs.push_back(std::async(std::launch::async, [&, execution] { infer(*execution, matrix.tensor); }));
        }
        for (auto & job : jobs) job.get();
        store.reset(); // Release uploaded weights before the backend owners.
        contexts.clear();
        ExecutionContext retry(config);
        if (!retry.backend()) throw std::runtime_error("backend initialization retry failed");
        std::cout << "PASS: eight cold backends, 128 parallel shared-weight computes, failure/retry\n";
        return 0;
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
