// ggml's CPU extra buffer types (its repacked matmul kernels) through the core
// helpers. cpu_extra_buffer_types and buffer_type_supports_matmul_weight find
// the buffer type llama.cpp's probe finds for each weight; a
// BackendWeightStore given that buffer type holds the weight there, and its
// matmuls match a plain store's; and
// validate_backend_graph_with_cpu_extra_buffers rejects, naming the weight, a
// view of such a weight, which validate_backend_graph_supported lets through,
// and a matmul of one with a non-contiguous input, which the shared validator
// lets through unless the buffer type's own supports_op turns it down (AMX's
// does, CPU_REPACK's does not).
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "test_assert.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>
#include <gguf.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace core = engine::core;
using engine::test::require;
using engine::test::require_eq;

constexpr int64_t kCols = 256;
constexpr auto kNative = engine::assets::TensorStorageType::Native;

struct ContextDeleter {
    void operator()(ggml_context * ctx) const noexcept { ggml_free(ctx); }
};

using Context = std::unique_ptr<ggml_context, ContextDeleter>;

// Pseudo-random values in [-1, 1), the same on every platform.
std::vector<float> random_values(size_t count, uint64_t seed) {
    std::vector<float> out(count);
    uint64_t state = seed;
    for (auto & value : out) {
        state += 0x9e3779b97f4a7c15ull;
        uint64_t z = state;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        z ^= z >> 31;
        value = static_cast<float>(z >> 40) / static_cast<float>(1ull << 23) - 1.0f;
    }
    return out;
}

struct WeightSpec {
    const char * name;
    ggml_type type;
    int64_t rows;
    // The CPU's F16 matmul rounds the input to F16, which AMX's does not.
    float tolerance;
};

// CPU_REPACK takes Q4_0 in rows of eight on x86 with AVX2 and of four on Arm,
// and no F16. ggml's AMX buffer, first where it is built, takes rows in 32s,
// F16 too. The test asks the probe where each weight goes.
const WeightSpec kWeights[] = {
    {"matmul.weight", GGML_TYPE_Q4_0, 64, 1e-5f},
    {"odd.weight", GGML_TYPE_Q4_0, 62, 1e-5f},
    {"half.weight", GGML_TYPE_F16, 64, 1e-3f},
};

void write_gguf(const std::filesystem::path & path) {
    Context ctx(ggml_init({64ull * 1024ull * 1024ull, nullptr, false}));
    std::unique_ptr<gguf_context, void (*)(gguf_context *)> gguf(gguf_init_empty(), gguf_free);
    uint64_t seed = 1;
    for (const auto & spec : kWeights) {
        auto * tensor = ggml_new_tensor_2d(ctx.get(), spec.type, kCols, spec.rows);
        const auto values = random_values(static_cast<size_t>(kCols * spec.rows), seed++);
        ggml_quantize_chunk(spec.type, values.data(), tensor->data, 0, spec.rows, kCols, nullptr);
        ggml_set_name(tensor, spec.name);
        gguf_add_tensor(gguf.get(), tensor);
    }
    require(gguf_write_to_file(gguf.get(), path.string().c_str(), false), "write " + path.string());
}

// The first CPU extra buffer type whose matmul takes a [rows, kCols] weight of
// `type`, found as llama.cpp's weight_buft_supported does, or nullptr.
ggml_backend_buffer_type_t probe_extra_buffer_type(ggml_backend_t backend, ggml_type type, int64_t rows) {
    ggml_backend_dev_t device = ggml_backend_get_device(backend);
    auto get_extra = (ggml_backend_dev_get_extra_bufts_t)ggml_backend_reg_get_proc_address(
        ggml_backend_dev_backend_reg(device), "ggml_backend_dev_get_extra_bufts");
    for (auto * candidate = get_extra != nullptr ? get_extra(device) : nullptr; candidate != nullptr && *candidate != nullptr;
         ++candidate) {
        Context ctx(ggml_init({8 * ggml_tensor_overhead(), nullptr, true}));
        auto * weight = ggml_new_tensor_2d(ctx.get(), type, kCols, rows);
        auto * op = ggml_mul_mat(ctx.get(), weight, ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, kCols, 512));
        weight->buffer = ggml_backend_buft_alloc_buffer(*candidate, 0);
        const bool supported = ggml_backend_dev_supports_op(device, op);
        ggml_backend_buffer_free(weight->buffer);
        weight->buffer = nullptr;
        if (supported) {
            return *candidate;
        }
    }
    return nullptr;
}

// The same through the core helpers.
ggml_backend_buffer_type_t core_extra_buffer_type(ggml_backend_t backend, ggml_type type, int64_t rows) {
    for (auto * buffer_type : core::cpu_extra_buffer_types(backend)) {
        if (core::buffer_type_supports_matmul_weight(backend, buffer_type, type, kCols, rows)) {
            return buffer_type;
        }
    }
    return nullptr;
}

// One op on the backend: ggml_mul_mat(weight, x) for x of `steps` random
// columns.
std::vector<float> run(ggml_backend_t backend, ggml_tensor * weight, int64_t steps) {
    Context ctx(ggml_init({16 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true}));
    ggml_tensor * input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, kCols, steps);
    ggml_set_input(input);
    ggml_tensor * out = ggml_mul_mat(ctx.get(), weight, input);
    ggml_set_output(out);
    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, out);
    core::validate_backend_graph_with_cpu_extra_buffers(backend, graph, "test graph");

    std::unique_ptr<ggml_gallocr, void (*)(ggml_gallocr_t)> allocator(
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)), ggml_gallocr_free);
    require(ggml_gallocr_alloc_graph(allocator.get(), graph), "allocate the test graph");
    const auto values = random_values(static_cast<size_t>(kCols * steps), 99);
    ggml_backend_tensor_set(input, values.data(), 0, values.size() * sizeof(float));
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "compute the test graph");
    std::vector<float> result(static_cast<size_t>(ggml_nelements(out)));
    ggml_backend_tensor_get(out, result.data(), 0, result.size() * sizeof(float));
    return result;
}

// The repacked kernels sum in another order; ggml's probe measured them within
// 5e-7 of the plain ones relative to the largest output.
void require_close_outputs(const std::vector<float> & actual, const std::vector<float> & expected, const std::string & label, float tolerance) {
    require_eq(actual.size(), expected.size(), label + " size");
    float scale = 0.0f;
    float diff = 0.0f;
    for (size_t i = 0; i < actual.size(); ++i) {
        scale = std::max(scale, std::fabs(expected[i]));
        diff = std::max(diff, std::fabs(actual[i] - expected[i]));
    }
    require(scale > 0.0f && diff <= tolerance * scale,
        label + ": max difference " + std::to_string(diff) + " against a largest output of " + std::to_string(scale));
}

using Validator = void (*)(ggml_backend_t, ggml_cgraph *, const char *);

// Whether `validate` rejects `out`'s graph, with an error that has `needle`.
bool is_rejected(Validator validate, ggml_backend_t backend, ggml_tensor * out, const char * label, const std::string & needle) {
    Context ctx(ggml_init({ggml_graph_overhead(), nullptr, true}));
    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, out);
    try {
        validate(backend, graph, label);
    } catch (const std::exception & error) {
        require(std::string(error.what()).find(needle) != std::string::npos, std::string(label) + " error: " + error.what());
        return true;
    }
    return false;
}

// A matmul of rows 2 to 33 of the weight.
bool view_is_rejected(Validator validate, ggml_backend_t backend, ggml_tensor * weight) {
    Context ctx(ggml_init({8 * ggml_tensor_overhead(), nullptr, true}));
    auto * rows = ggml_view_2d(ctx.get(), weight, weight->ne[0], 32, weight->nb[1], 2 * weight->nb[1]);
    auto * out = ggml_mul_mat(ctx.get(), rows, ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, kCols, 1));
    return is_rejected(validate, backend, out, "view graph", "view graph views weight 'matmul.weight', held in a ");
}

// A matmul of the weight with every other row of an input, which CPU_REPACK's
// kernels would read as if each row followed the last.
ggml_tensor * strided_matmul(ggml_context * ctx, ggml_tensor * weight) {
    auto * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kCols, 8);
    auto * rows = ggml_view_2d(ctx, input, kCols, 4, 2 * input->nb[1], 0);
    return ggml_mul_mat(ctx, weight, rows);
}

constexpr const char * kStridedMisuse = "strided graph multiplies a non-contiguous input by weight 'matmul.weight', held in a ";

bool strided_input_is_rejected(Validator validate, ggml_backend_t backend, ggml_tensor * weight, const std::string & needle = kStridedMisuse) {
    Context ctx(ggml_init({8 * ggml_tensor_overhead(), nullptr, true}));
    return is_rejected(validate, backend, strided_matmul(ctx.get(), weight), "strided graph", needle);
}

// Whether the backend's supports_op takes that matmul, as the shared validator asks.
bool backend_takes_strided_input(ggml_backend_t backend, ggml_tensor * weight) {
    Context ctx(ggml_init({8 * ggml_tensor_overhead(), nullptr, true}));
    return ggml_backend_supports_op(backend, strided_matmul(ctx.get(), weight));
}

void test_extra_buffers(ggml_backend_t backend, const engine::assets::TensorSource & source) {
    ggml_backend_buffer_type_t cpu = ggml_backend_get_default_buffer_type(backend);
    std::cout << "CPU extra buffer types:";
    for (auto * buffer_type : core::cpu_extra_buffer_types(backend)) {
        std::cout << ' ' << ggml_backend_buft_name(buffer_type);
    }
    std::cout << '\n';

    core::BackendWeightStore plain_store(backend, core::BackendType::Cpu, "test.plain", 1024 * 1024);
    std::vector<core::TensorValue> plain;
    for (const auto & spec : kWeights) {
        plain.push_back(plain_store.load_tensor(source, spec.name, kNative, {spec.rows, kCols}));
    }
    plain_store.upload();

    ggml_tensor * extra_matmul = nullptr;  // matmul.weight in an extra buffer, where one takes it
    std::vector<std::unique_ptr<core::BackendWeightStore>> extra_stores;
    for (size_t i = 0; i < std::size(kWeights); ++i) {
        const auto & spec = kWeights[i];
        const std::string label = std::string(spec.name) + " (" + ggml_type_name(spec.type) + ", " + std::to_string(spec.rows) + " rows)";
        require(ggml_backend_buffer_get_type(plain[i].tensor->buffer) == cpu, label + " in the plain store");
        const auto expected = probe_extra_buffer_type(backend, spec.type, spec.rows);
        const auto found = core_extra_buffer_type(backend, spec.type, spec.rows);
        std::cout << label << " goes to " << ggml_backend_buft_name(expected != nullptr ? expected : cpu) << '\n';
        require(found == expected, label + ": the core helpers find the buffer type the probe finds");
        if (expected == nullptr) {
            continue;
        }

        // An unchanged store, given the extra buffer type as its buffer type.
        extra_stores.push_back(std::make_unique<core::BackendWeightStore>(
            backend, core::BackendType::Cpu, std::string("test.") + ggml_backend_buft_name(expected), 1024 * 1024, expected));
        const auto weight = extra_stores.back()->load_tensor(source, spec.name, kNative, {spec.rows, kCols});
        ggml_set_name(weight.tensor, spec.name);
        extra_stores.back()->upload();
        require(ggml_backend_buffer_get_type(weight.tensor->buffer) == expected, label + " in the extra buffer");
        for (const int64_t steps : {1, 4, 33}) {
            require_close_outputs(run(backend, weight.tensor, steps), run(backend, plain[i].tensor, steps),
                label + " at " + std::to_string(steps) + " steps", spec.tolerance);
        }
        if (i == 0) {
            extra_matmul = weight.tensor;
        }
    }

    const Validator shared = core::validate_backend_graph_supported;
    const Validator extended = core::validate_backend_graph_with_cpu_extra_buffers;
    for (const Validator validate : {shared, extended}) {
        require(!view_is_rejected(validate, backend, plain[0].tensor), "a view of a plain weight passes validation");
        require(!strided_input_is_rejected(validate, backend, plain[0].tensor), "a plain weight takes a non-contiguous input");
    }
    if (extra_matmul != nullptr) {
        require(!view_is_rejected(shared, backend, extra_matmul), "the shared validator does not look at extra buffers");
        // AMX's supports_op turns a non-contiguous input down itself;
        // CPU_REPACK's and KleidiAI's do not look at it.
        const bool takes_strided = backend_takes_strided_input(backend, extra_matmul);
        require(strided_input_is_rejected(shared, backend, extra_matmul, "strided graph contains unsupported backend op 'MUL_MAT'") == !takes_strided,
            "the shared validator rejects a strided input only where the buffer type's supports_op does");
        require(view_is_rejected(extended, backend, extra_matmul), "a view of a weight in an extra buffer fails validation");
        require(strided_input_is_rejected(extended, backend, extra_matmul), "a weight in an extra buffer rejects a non-contiguous input");
    }
}

}  // namespace

int main() {
    try {
        const auto dir = std::filesystem::temp_directory_path() / "audiocpp_cpu_extra_buffers_test";
        std::filesystem::remove_all(dir);
        std::filesystem::create_directories(dir);
        const auto path = dir / "weights.gguf";
        write_gguf(path);
        {
            const auto source = engine::assets::open_tensor_source(path);
            ggml_backend_t backend = core::init_backend({core::BackendType::Cpu, 0, 2});
            core::set_backend_threads(backend, 2);
            try {
                test_extra_buffers(backend, *source);
            } catch (...) {
                ggml_backend_free(backend);
                throw;
            }
            ggml_backend_free(backend);
        }
        std::filesystem::remove_all(dir);
        std::cout << "cpu_extra_buffers_test: PASS\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "cpu_extra_buffers_test: " << error.what() << '\n';
        return 1;
    }
}
