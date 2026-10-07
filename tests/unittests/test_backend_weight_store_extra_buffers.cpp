// BackendWeightStore's opt-in CPU extra buffers (ggml's repacked matmul
// kernels). A MatMulOnly weight goes into one exactly when that buffer type
// runs its matmul, by the same probe llama.cpp uses; other weights stay in the
// plain CPU buffer; the matmuls match the plain store's; and a view of a weight
// in an extra buffer, or a matmul of one with a non-contiguous input, fails
// validation. Without the opt-in, or with an explicit buffer type, every
// weight stays in one plain buffer.
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
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
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
};

// CPU_REPACK takes Q4_0 in rows of eight on x86 with AVX2 and of four on Arm,
// and no F16. ggml's AMX buffer, first where it is built, takes rows in 32s,
// F16 too. The test asks the probe where each hinted weight goes.
const WeightSpec kWeights[] = {
    {"matmul.weight", GGML_TYPE_Q4_0, 64},
    {"odd.weight", GGML_TYPE_Q4_0, 62},
    {"half.weight", GGML_TYPE_F16, 64},
    {"table.weight", GGML_TYPE_Q4_0, 16},
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
    auto * norm = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, kCols);
    const auto values = random_values(static_cast<size_t>(kCols), seed);
    std::copy(values.begin(), values.end(), static_cast<float *>(norm->data));
    ggml_set_name(norm, "norm.weight");
    gguf_add_tensor(gguf.get(), norm);
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

struct Loaded {
    core::TensorValue matmul;
    core::TensorValue odd;
    core::TensorValue half;
    core::TensorValue table;
    core::TensorValue norm;
};

Loaded load(core::BackendWeightStore & store, const engine::assets::TensorSource & source) {
    const auto matmul_only = core::WeightUse::MatMulOnly;
    Loaded out;
    out.matmul = store.load_tensor(source, "matmul.weight", kNative, {64, kCols}, matmul_only);
    out.odd = store.load_tensor(source, "odd.weight", kNative, {62, kCols}, matmul_only);
    out.half = store.load_tensor(source, "half.weight", kNative, {64, kCols}, matmul_only);
    out.table = store.load_tensor(source, "table.weight", kNative, {16, kCols});
    out.norm = store.load_f32_tensor(source, "norm.weight", {kCols});
    store.upload();
    return out;
}

ggml_backend_buffer_type_t buffer_type_of(const core::TensorValue & value) {
    return ggml_backend_buffer_get_type(value.tensor->buffer);
}

// One op on the backend: ggml_mul_mat(weight, x) for x of `steps` random
// columns, or the weight's rows 3, 0 and 15 for a gather.
std::vector<float> run(ggml_backend_t backend, ggml_tensor * weight, int64_t steps, bool gather = false) {
    Context ctx(ggml_init({16 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true}));
    ggml_tensor * input = gather ? ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 3) : ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, kCols, steps);
    ggml_set_input(input);
    ggml_tensor * out = gather ? ggml_get_rows(ctx.get(), weight, input) : ggml_mul_mat(ctx.get(), weight, input);
    ggml_set_output(out);
    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, out);
    core::validate_backend_graph_supported(backend, graph, "test graph");

    std::unique_ptr<ggml_gallocr, void (*)(ggml_gallocr_t)> allocator(
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)), ggml_gallocr_free);
    require(ggml_gallocr_alloc_graph(allocator.get(), graph), "allocate the test graph");
    if (gather) {
        const std::vector<int32_t> rows = {3, 0, 15};
        ggml_backend_tensor_set(input, rows.data(), 0, rows.size() * sizeof(int32_t));
    } else {
        const auto values = random_values(static_cast<size_t>(kCols * steps), 99);
        ggml_backend_tensor_set(input, values.data(), 0, values.size() * sizeof(float));
    }
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "compute the test graph");
    std::vector<float> values(static_cast<size_t>(ggml_nelements(out)));
    ggml_backend_tensor_get(out, values.data(), 0, values.size() * sizeof(float));
    return values;
}

// The repacked kernels sum in another order; ggml's probe measured them within
// 5e-7 of the plain ones relative to the largest output.
void require_close_outputs(
    const std::vector<float> & actual, const std::vector<float> & expected, const std::string & label, float tolerance = 1e-5f) {
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

// Whether validation rejects `out`'s graph, with an error that has `needle`.
bool is_rejected(ggml_backend_t backend, ggml_tensor * out, const char * label, const std::string & needle) {
    Context ctx(ggml_init({ggml_graph_overhead(), nullptr, true}));
    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, out);
    try {
        core::validate_backend_graph_supported(backend, graph, label);
    } catch (const std::exception & error) {
        require(std::string(error.what()).find(needle) != std::string::npos, std::string(label) + " error: " + error.what());
        return true;
    }
    return false;
}

// A matmul of rows 2 to 33 of the weight.
bool view_is_rejected(ggml_backend_t backend, ggml_tensor * weight) {
    Context ctx(ggml_init({8 * ggml_tensor_overhead(), nullptr, true}));
    auto * rows = ggml_view_2d(ctx.get(), weight, weight->ne[0], 32, weight->nb[1], 2 * weight->nb[1]);
    auto * out = ggml_mul_mat(ctx.get(), rows, ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, kCols, 1));
    return is_rejected(backend, out, "view graph", "view graph views weight 'matmul.weight', held in a ");
}

// A matmul of the weight with every other row of an input, which CPU_REPACK's
// kernels would read as if each row followed the last.
bool strided_input_is_rejected(ggml_backend_t backend, ggml_tensor * weight) {
    Context ctx(ggml_init({8 * ggml_tensor_overhead(), nullptr, true}));
    auto * input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, kCols, 8);
    auto * rows = ggml_view_2d(ctx.get(), input, kCols, 4, 2 * input->nb[1], 0);
    auto * out = ggml_mul_mat(ctx.get(), weight, rows);
    return is_rejected(
        backend, out, "strided graph", "strided graph multiplies a non-contiguous input by weight 'matmul.weight', held in a ");
}

void test_extra_buffers(ggml_backend_t backend, const engine::assets::TensorSource & source) {
    ggml_backend_buffer_type_t cpu = ggml_backend_get_default_buffer_type(backend);
    // Where each hinted weight goes, or nullptr for the CPU buffer.
    const ggml_backend_buffer_type_t expected = probe_extra_buffer_type(backend, GGML_TYPE_Q4_0, 64);
    const ggml_backend_buffer_type_t expected_odd = probe_extra_buffer_type(backend, GGML_TYPE_Q4_0, 62);
    const ggml_backend_buffer_type_t expected_half = probe_extra_buffer_type(backend, GGML_TYPE_F16, 64);
    const auto or_cpu = [&](ggml_backend_buffer_type_t buffer_type) { return buffer_type != nullptr ? buffer_type : cpu; };
    // The extra buffers, in the order of their first weights, and how many
    // weights each holds.
    std::vector<std::pair<ggml_backend_buffer_type_t, size_t>> expected_extra;
    size_t moved = 0;
    for (auto * buffer_type : {expected, expected_odd, expected_half}) {
        if (buffer_type == nullptr) {
            continue;
        }
        ++moved;
        const auto known = std::find_if(
            expected_extra.begin(), expected_extra.end(), [&](const auto & extra) { return extra.first == buffer_type; });
        if (known == expected_extra.end()) {
            expected_extra.emplace_back(buffer_type, 1);
        } else {
            ++known->second;
        }
    }
    require(moved == 0 || !core::cpu_extra_buffer_types(backend).empty(), "the probe found an extra buffer type");
    std::cout << "CPU extra buffer types:";
    for (auto * buffer_type : core::cpu_extra_buffer_types(backend)) {
        std::cout << ' ' << ggml_backend_buft_name(buffer_type);
    }
    std::cout << "; Q4_0 [64 x 256] goes to " << ggml_backend_buft_name(or_cpu(expected)) << ", Q4_0 [62 x 256] to "
              << ggml_backend_buft_name(or_cpu(expected_odd)) << ", F16 [64 x 256] to " << ggml_backend_buft_name(or_cpu(expected_half))
              << '\n';

    // Without the opt-in the hint changes nothing.
    core::BackendWeightStore plain_store(backend, core::BackendType::Cpu, "test.plain", 1024 * 1024);
    const auto plain = load(plain_store, source);
    require_eq(plain_store.buffers().size(), size_t{1}, "plain store buffers");
    require_eq(plain_store.buffers()[0].tensors, size_t{5}, "plain store tensors");
    for (const auto * value : {&plain.matmul, &plain.odd, &plain.half, &plain.table, &plain.norm}) {
        require(buffer_type_of(*value) == cpu, "a plain store keeps every weight in the CPU buffer");
    }
    require(plain_store.matmul_buffer_type(GGML_TYPE_Q4_0, core::TensorShape::from_dims({64, kCols})) == nullptr,
        "a plain store has no extra buffer types");

    // An explicit buffer type wins over the opt-in.
    core::BackendWeightStoreOptions explicit_options;
    explicit_options.buffer_type = cpu;
    explicit_options.cpu_extra_buffers = true;
    core::BackendWeightStore explicit_store(backend, core::BackendType::Cpu, "test.explicit", 1024 * 1024, explicit_options);
    const auto explicit_weights = load(explicit_store, source);
    require_eq(explicit_store.buffers().size(), size_t{1}, "explicit buffer type store buffers");
    require(buffer_type_of(explicit_weights.matmul) == cpu, "an explicit buffer type keeps MatMulOnly weights in it");

    core::BackendWeightStoreOptions options;
    options.cpu_extra_buffers = true;
    core::BackendWeightStore store(backend, core::BackendType::Cpu, "test.extra", 1024 * 1024, options);
    const auto weights = load(store, source);
    require(store.matmul_buffer_type(GGML_TYPE_Q4_0, core::TensorShape::from_dims({64, kCols})) == expected,
        "matmul_buffer_type follows the probe");
    require(buffer_type_of(weights.matmul) == or_cpu(expected), "the MatMulOnly Q4_0 weight");
    require(buffer_type_of(weights.odd) == or_cpu(expected_odd), "the 62-row Q4_0 weight");
    require(buffer_type_of(weights.half) == or_cpu(expected_half), "the F16 weight");
    require(buffer_type_of(weights.table) == cpu, "a weight without the hint stays in the CPU buffer");
    require(buffer_type_of(weights.norm) == cpu, "an F32 weight stays in the CPU buffer");
    const auto buffers = store.buffers();
    require_eq(buffers.size(), 1 + expected_extra.size(), "opt-in store buffers");
    require_eq(buffers[0].tensors, 5 - moved, "opt-in store CPU buffer tensors");
    for (size_t i = 0; i < expected_extra.size() && i + 1 < buffers.size(); ++i) {
        require_eq(buffers[i + 1].buffer_type, std::string(ggml_backend_buft_name(expected_extra[i].first)), "extra buffer type name");
        require_eq(buffers[i + 1].tensors, expected_extra[i].second, "extra buffer tensors");
        if (expected_extra[i].first == expected) {
            require(buffers[i + 1].bytes >= ggml_nbytes(weights.matmul.tensor), "the extra buffer holds the weight");
        }
    }

    // A moved weight's matmul sums in another order. The CPU's F16 matmul
    // also rounds the input to F16, which AMX's does not.
    const auto compare = [&](const core::TensorValue & weight, const core::TensorValue & plain_weight, bool moved_weight, float tolerance,
                             const std::string & label) {
        for (const int64_t steps : {1, 4, 33}) {
            const auto actual = run(backend, weight.tensor, steps);
            const auto reference = run(backend, plain_weight.tensor, steps);
            const std::string at = label + " at " + std::to_string(steps) + " steps";
            if (moved_weight) {
                require_close_outputs(actual, reference, at, tolerance);
            } else {
                require(actual == reference, at);
            }
        }
    };
    compare(weights.matmul, plain.matmul, expected != nullptr, 1e-5f, "matmul");
    compare(weights.odd, plain.odd, expected_odd != nullptr, 1e-5f, "the 62-row matmul");
    compare(weights.half, plain.half, expected_half != nullptr, 1e-3f, "the F16 matmul");
    require(run(backend, weights.table.tensor, 0, true) == run(backend, plain.table.tensor, 0, true), "rows of the gather table");

    require(!view_is_rejected(backend, plain.matmul.tensor), "a view of a plain weight passes validation");
    require(view_is_rejected(backend, weights.matmul.tensor) == (expected != nullptr), "a view of a weight in an extra buffer fails validation");
    require(!strided_input_is_rejected(backend, plain.matmul.tensor), "a plain weight takes a non-contiguous input");
    require(strided_input_is_rejected(backend, weights.matmul.tensor) == (expected != nullptr),
        "a weight in an extra buffer rejects a non-contiguous input");
}

}  // namespace

int main() {
    try {
        const auto dir = std::filesystem::temp_directory_path() / "audiocpp_backend_weight_store_extra_buffers_test";
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
        std::cout << "backend_weight_store_extra_buffers_test: PASS\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "backend_weight_store_extra_buffers_test: " << error.what() << '\n';
        return 1;
    }
}
