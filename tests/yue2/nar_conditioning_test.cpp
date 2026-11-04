#include "../../src/models/yue2/nar_conditioning.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
using engine::core::BackendType;
using engine::models::yue2::detail::add_time_conditioning;
using Context = std::unique_ptr<ggml_context, decltype(&ggml_free)>;

void check(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

int count(ggml_cgraph * graph, ggml_op op) {
    int n = 0;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) n += ggml_graph_node(graph, i)->op == op;
    return n;
}

struct Result {
    std::vector<float> values;
    size_t buffer_bytes;
    int repeats;
};

// Execute SYCL-selected graphs only on a CPU backend. This proves graph/value
// equivalence and actual ggml allocator savings, not SYCL kernel performance.
Result run(int64_t frames, int64_t hidden_size, int64_t batches, BackendType policy) {
    Context ctx(ggml_init({2 * 1024 * 1024, nullptr, true}), ggml_free);
    check(ctx != nullptr, "context allocation failed");
    auto * hidden = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, hidden_size, frames, 1, batches);
    auto * time = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, hidden_size, 1, 1, 1);
    auto * pos = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, hidden_size, frames, 1, batches);
    ggml_set_input(hidden);
    ggml_set_input(time);
    ggml_set_input(pos);
    auto * sum = add_time_conditioning(ctx.get(), hidden, time, policy);
    auto * output = ggml_add(ctx.get(), sum, pos); // preserve the real operation order
    ggml_set_output(output);
    auto * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, output);
    const int repeats = count(graph, GGML_OP_REPEAT);
    check(count(graph, GGML_OP_ADD) == 2, "conditioning must retain both additions");
    check(repeats == (policy == BackendType::Sycl ? 0 : 1), "wrong backend repeat policy");
    check(policy != BackendType::Sycl || sum->src[1] == time, "broadcast must use the original time vector");
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> cpu(ggml_backend_cpu_init(), ggml_backend_free);
    check(cpu != nullptr, "CPU backend initialization failed");
    ggml_backend_cpu_set_n_threads(cpu.get(), 4);
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> alloc(
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(cpu.get())), ggml_gallocr_free);
    check(ggml_gallocr_alloc_graph(alloc.get(), graph), "graph allocation failed");
    std::vector<float> data(static_cast<size_t>(ggml_nelements(hidden)));
    std::vector<float> t(static_cast<size_t>(hidden_size));
    for (size_t i = 0; i < data.size(); ++i) data[i] = 0.25f * std::sin(float(i % 7919) * 0.013f);
    for (size_t i = 0; i < t.size(); ++i) t[i] = 0.125f * std::cos(float(i) * 0.037f);
    ggml_backend_tensor_set(hidden, data.data(), 0, data.size() * sizeof(float));
    ggml_backend_tensor_set(time, t.data(), 0, t.size() * sizeof(float));
    for (size_t i = 0; i < data.size(); ++i) data[i] = 0.0625f * std::sin(float(i % 1237) * 0.019f);
    ggml_backend_tensor_set(pos, data.data(), 0, data.size() * sizeof(float));
    check(ggml_backend_graph_compute(cpu.get(), graph) == GGML_STATUS_SUCCESS, "CPU graph execution failed");
    ggml_backend_tensor_get(output, data.data(), 0, data.size() * sizeof(float));
    for (float v : data) check(std::isfinite(v), "nonfinite result");
    return {std::move(data), ggml_gallocr_get_buffer_size(alloc.get(), 0), repeats};
}

void parity(int64_t frames, int64_t hidden, int64_t batches) {
    const auto ref = run(frames, hidden, batches, BackendType::Cpu);
    const auto candidate = run(frames, hidden, batches, BackendType::Sycl);
    check(ref.values.size() == candidate.values.size(), "size mismatch");
    check(std::memcmp(ref.values.data(), candidate.values.data(), ref.values.size() * sizeof(float)) == 0,
          "broadcast result not bit-exact on CPU");
    check(candidate.buffer_bytes <= ref.buffer_bytes, "broadcast increased allocator size");
    if (frames > 1) check(candidate.buffer_bytes < ref.buffer_bytes, "repeat removal saved no allocation");
    std::cout << "PASS frames=" << frames << " hidden=" << hidden << " batches=" << batches
              << " values=" << candidate.values.size() << " output=bit-exact(CPU) removed_repeat=1"
              << " reference_buffer=" << ref.buffer_bytes << " candidate_buffer=" << candidate.buffer_bytes
              << " saved_bytes=" << ref.buffer_bytes - candidate.buffer_bytes << '\n';
}
} // namespace

int main() {
    try {
        for (auto backend : {BackendType::Cpu, BackendType::Cuda, BackendType::Hip,
                             BackendType::Vulkan, BackendType::Metal, BackendType::BestAvailable}) {
            (void) run(33, 64, 1, backend);
        }
        for (int64_t frames : {1, 2, 31, 32, 33, 65}) parity(frames, 128, 2);
        for (int64_t frames : {2402, 6502, 12002}) parity(frames, 2048, 1);
        std::cout << "PASS backend policy + CPU conditioning parity/allocator checks\n";
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
