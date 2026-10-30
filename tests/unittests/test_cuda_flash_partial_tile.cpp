#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cuda.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
void run(ggml_backend_t backend, int steps) {
    constexpr int dim = 64, heads = 8, window = 25;
    auto * ctx = ggml_init({8 * 1024 * 1024, nullptr, true});
    if (!ctx) throw std::runtime_error("context allocation failed");
    auto * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dim, steps, heads, 1);
    auto * k = ggml_dup_tensor(ctx, q);
    auto * v = ggml_dup_tensor(ctx, q);
    auto * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, steps, steps);
    for (auto * tensor : {q, k, v, mask}) ggml_set_input(tensor);
    const float scale = 1.0F / std::sqrt(float(dim));
    auto * output = ggml_flash_attn_ext(ctx, q, k, v, mask, scale, 0.0F, 0.0F);
    ggml_flash_attn_ext_set_prec(output, GGML_PREC_F32);
    ggml_set_output(output);
    auto * graph = ggml_new_graph_custom(ctx, 32, false);
    ggml_build_forward_expand(graph, output);
    auto alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(alloc, graph)) throw std::runtime_error("graph allocation failed");
    std::vector<float> input(size_t(dim) * steps * heads);
    for (size_t i = 0; i < input.size(); ++i) input[i] = std::sin(float(i) * 0.13F);
    for (auto * tensor : {q, k, v}) ggml_backend_tensor_set(tensor, input.data(), 0, input.size() * sizeof(float));
    std::vector<ggml_fp16_t> mask_data(size_t(steps) * steps);
    for (int j = 0; j < steps; ++j) {
        for (int i = 0; i < steps; ++i) {
            mask_data[size_t(j) * steps + i] = ggml_fp32_to_fp16(i <= j && i > j - window ? 0.0F : -1.0e9F);
        }
    }
    ggml_backend_tensor_set(mask, mask_data.data(), 0, mask_data.size() * sizeof(ggml_fp16_t));
    const auto status = ggml_backend_graph_compute(backend, graph);
    std::vector<float> actual(input.size());
    if (status == GGML_STATUS_SUCCESS) ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
    ggml_backend_cuda_clear_graph(backend, graph);
    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    if (status != GGML_STATUS_SUCCESS) throw std::runtime_error("graph computation failed");

    // Compare every valid query with an independent CPU softmax reference.
    // The masked rows still require finite zero-filled K/V inside a partial tile.
    for (int head = 0; head < heads; ++head) {
        for (int query = 0; query < steps; ++query) {
            const int begin = std::max(0, query - window + 1);
            std::vector<double> scores;
            double maximum = -1.0e30;
            for (int key = begin; key <= query; ++key) {
                double score = 0;
                for (int d = 0; d < dim; ++d) {
                    score += double(input[(head * steps + query) * dim + d]) * input[(head * steps + key) * dim + d];
                }
                scores.push_back(score * scale);
                maximum = std::max(maximum, scores.back());
            }
            double total = 0;
            for (auto & score : scores) { score = std::exp(score - maximum); total += score; }
            for (int d = 0; d < dim; ++d) {
                double expected = 0;
                for (int key = begin; key <= query; ++key) expected += scores[key - begin] * input[(head * steps + key) * dim + d] / total;
                const float value = actual[(query * heads + head) * dim + d];
                if (!std::isfinite(value) || std::abs(value - expected) > 0.003) {
                    std::cerr << "steps=" << steps << " head=" << head << " query=" << query << " dim=" << d
                              << " actual=" << value << " expected=" << expected << '\n';
                    throw std::runtime_error("partial-tile flash attention differs from reference");
                }
            }
        }
    }
}
}

int main() {
    const int devices = ggml_backend_cuda_get_device_count();
    if (devices == 0) return 77;
    for (int device = 0; device < devices; ++device) {
        auto backend = ggml_backend_cuda_init(device);
        if (!backend) return 1;
        try {
            for (int steps : {17, 43, 86, 128, 212, 300, 325}) run(backend, steps);
        } catch (const std::exception & error) {
            std::cerr << "device=" << device << ": " << error.what() << '\n';
            ggml_backend_free(backend);
            return 1;
        }
        ggml_backend_free(backend);
    }
    std::cout << "PASS: partial-tile CUDA flash attention on all devices\n";
}
