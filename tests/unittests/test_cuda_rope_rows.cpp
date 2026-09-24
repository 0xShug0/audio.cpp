#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cuda.h>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

// Nine lanes force the generic kernel (54 rows); eight lanes exercise packed
// rows (48 rows). The first eight lanes have identical values and positions.
static std::vector<float> run(ggml_backend_t backend, int lanes, bool strided, int dims) {
    auto * ctx = ggml_init({4 * 1024 * 1024, nullptr, true});
    auto * storage = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, strided ? 192 : 64, 2, 3, lanes);
    auto * input = strided ? ggml_view_4d(ctx, storage, 64, 2, 3, lanes,
        storage->nb[1], storage->nb[2], storage->nb[3], 64 * sizeof(float)) : storage;
    auto * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 3);
    ggml_set_input(storage);
    ggml_set_input(pos);
    auto * output = ggml_rope_ext(ctx, input, pos, nullptr, dims, GGML_ROPE_TYPE_NORMAL,
        0, 10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    ggml_set_output(output);
    auto * graph = ggml_new_graph_custom(ctx, 32, false);
    ggml_build_forward_expand(graph, output);
    auto alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(alloc, graph)) throw std::runtime_error("allocation failed");
    std::vector<float> values(ggml_nelements(storage));
    for (size_t i = 0; i < values.size(); ++i) values[i] = std::sin(float(i) * 0.17f) * 12.0f;
    const int32_t positions[] = {0, 17, 4096};
    for (int repeat = 0; repeat < 3; ++repeat) {
        ggml_backend_tensor_set(storage, values.data(), 0, values.size() * sizeof(float));
        ggml_backend_tensor_set(pos, positions, 0, sizeof(positions));
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("graph failed");
    }
    std::vector<float> result(ggml_nelements(output));
    ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(float));
    auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
    auto clear = reinterpret_cast<void (*)(ggml_backend_t, ggml_cgraph *)>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_clear_graph"));
    if (clear) clear(backend, graph);
    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    return result;
}

int main() {
    if (ggml_backend_cuda_get_device_count() == 0) return 77;
    auto backend = ggml_backend_cuda_init(0);
    if (!backend) return 1;
    try {
        for (bool strided : {false, true}) {
            for (int dims : {32, 64}) {
                auto reference = run(backend, 9, strided, dims);
                auto packed = run(backend, 8, strided, dims);
                if (std::memcmp(reference.data(), packed.data(), packed.size() * sizeof(float)))
                    throw std::runtime_error("packed RoPE differs from generic CUDA RoPE");
            }
        }
        std::cout << "PASS: packed RoPE matches generic kernel, contiguous and strided heads, partial and full rotation\n";
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        ggml_backend_free(backend);
        return 1;
    }
    ggml_backend_free(backend);
}
