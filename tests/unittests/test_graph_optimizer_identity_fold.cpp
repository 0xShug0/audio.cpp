// engine::runtime::optimize_graph identity-materialization folding.
//
// F2.11 of docs/reviews/02-kv-cache-attention-performance.md notes that the graph optimizer
// defaults on and has exactly one caller. This test pins two things a caller needs to know
// before wiring it into the decode/prefill graphs:
//
//   identity_cont_is_folded        a ggml_cont whose result has the same type, shape and
//                                  strides as its source is removed and its consumer is
//                                  rewired to the source, producing bit-identical values.
//   permuted_cont_is_not_folded    a ggml_cont over a permuted view is NOT folded, because
//                                  its strides genuinely differ from the source's. That is
//                                  the exact shape of the redundant copy in F2.1, so running
//                                  optimize_graph over a decode graph does not remove it --
//                                  only changing the attention lowering does.

#include "engine/framework/core/backend.h"
#include "engine/framework/runtime/graph_optimizer.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr size_t kGraphBytes = 16 * 1024 * 1024;
constexpr size_t kGraphNodes = 512;

void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::vector<float> patterned(size_t count) {
    std::vector<float> values(count, 0.0F);
    for (size_t i = 0; i < count; ++i) {
        values[i] = std::sin(0.37F + 0.019F * static_cast<float>(i));
    }
    return values;
}

std::vector<float> read_all(const ggml_tensor * tensor) {
    std::vector<float> values(static_cast<size_t>(ggml_nelements(tensor)), 0.0F);
    ggml_backend_tensor_get(tensor, values.data(), 0, values.size() * sizeof(float));
    return values;
}

void compute_or_throw(ggml_backend_t backend, ggml_cgraph * graph, const char * label) {
    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    ggml_backend_synchronize(backend);
    if (status != GGML_STATUS_SUCCESS) {
        std::ostringstream oss;
        oss << label << " graph compute failed with status " << static_cast<int>(status);
        throw std::runtime_error(oss.str());
    }
}

void run() {
    ggml_backend_t backend = engine::core::init_backend({engine::core::BackendType::Cpu, 0, 1});
    ggml_init_params params{kGraphBytes, nullptr, true};
    ggml_context * ggml = ggml_init(params);
    if (ggml == nullptr) {
        ggml_backend_free(backend);
        throw std::runtime_error("failed to initialize graph optimizer test context");
    }
    ggml_backend_buffer_t buffer = nullptr;

    try {
        // Case 1: ggml_cont over an already-packed tensor. Identity, must fold.
        ggml_tensor * source = ggml_new_tensor_2d(ggml, GGML_TYPE_F32, 16, 5);
        ggml_tensor * identity_cont = ggml_cont(ggml, source);
        ggml_tensor * folded_consumer = ggml_scale(ggml, identity_cont, 2.0F);
        ggml_tensor * reference = ggml_scale(ggml, source, 2.0F);

        // Case 2: ggml_cont over a permuted view. Real copy, must not fold.
        ggml_tensor * base = ggml_new_tensor_3d(ggml, GGML_TYPE_F32, 8, 3, 4);
        ggml_tensor * permuted = ggml_permute(ggml, base, 0, 2, 1, 3);
        ggml_tensor * permuted_cont = ggml_cont(ggml, permuted);
        ggml_tensor * permuted_consumer = ggml_scale(ggml, permuted_cont, 3.0F);

        ggml_cgraph * identity_graph = ggml_new_graph_custom(ggml, kGraphNodes, false);
        ggml_build_forward_expand(identity_graph, folded_consumer);

        ggml_cgraph * reference_graph = ggml_new_graph_custom(ggml, kGraphNodes, false);
        ggml_build_forward_expand(reference_graph, reference);

        ggml_cgraph * permuted_graph = ggml_new_graph_custom(ggml, kGraphNodes, false);
        ggml_build_forward_expand(permuted_graph, permuted_consumer);

        buffer = ggml_backend_alloc_ctx_tensors(ggml, backend);
        if (buffer == nullptr) {
            throw std::runtime_error("failed to allocate graph optimizer test tensors");
        }

        const auto source_values = patterned(static_cast<size_t>(ggml_nelements(source)));
        ggml_backend_tensor_set(source, source_values.data(), 0, source_values.size() * sizeof(float));
        const auto base_values = patterned(static_cast<size_t>(ggml_nelements(base)));
        ggml_backend_tensor_set(base, base_values.data(), 0, base_values.size() * sizeof(float));

        // Reference result first, while the identity graph is still unoptimized.
        compute_or_throw(backend, reference_graph, "graph optimizer reference");
        const auto expected = read_all(reference);

        require(
            ggml_graph_n_nodes(identity_graph) == 2,
            "identity graph should start as [cont, scale]");

        const auto report = engine::runtime::optimize_graph(
            *identity_graph, engine::runtime::GraphOptimizationBackend::Gpu, true);

        std::cout << "graph_optimizer identity_cont nodes_before=" << report.nodes_before
                  << " nodes_after=" << report.nodes_after
                  << " identity_materializations_elided=" << report.identity_materializations_elided
                  << "\n";

        require(report.nodes_before == 2, "identity fold nodes_before should be 2");
        require(report.nodes_after == 1, "identity fold should leave exactly the scale node");
        require(
            report.identity_materializations_elided == 1,
            "identity fold should report exactly one elided materialization");
        require(
            folded_consumer->src[0] == source,
            "the scale consumer should be rewired straight to the cont's source");

        compute_or_throw(backend, identity_graph, "graph optimizer folded");
        const auto actual = read_all(folded_consumer);
        require(actual.size() == expected.size(), "folded output size mismatch");
        float max_abs = 0.0F;
        for (size_t i = 0; i < actual.size(); ++i) {
            max_abs = std::max(max_abs, std::abs(actual[i] - expected[i]));
        }
        std::cout << "graph_optimizer identity_cont folded_vs_reference max_abs=" << max_abs << "\n";
        require(max_abs == 0.0F, "folding an identity cont changed the result");

        // A cont over a permuted view has different strides from its source, so
        // same_type_shape_and_layout rejects it and the copy survives. This is why running
        // optimize_graph over a decode graph does not remove the F2.1 K/V copy.
        const int permuted_nodes_before = ggml_graph_n_nodes(permuted_graph);
        const auto permuted_report = engine::runtime::optimize_graph(
            *permuted_graph, engine::runtime::GraphOptimizationBackend::Gpu, true);
        std::cout << "graph_optimizer permuted_cont nodes_before=" << permuted_report.nodes_before
                  << " nodes_after=" << permuted_report.nodes_after
                  << " identity_materializations_elided="
                  << permuted_report.identity_materializations_elided << "\n";
        require(
            permuted_report.identity_materializations_elided == 0,
            "a cont over a permuted view must not be folded");
        require(
            permuted_report.nodes_after == permuted_nodes_before,
            "the permuted graph should be left untouched by the GPU optimizer profile");
    } catch (...) {
        if (buffer != nullptr) {
            ggml_backend_buffer_free(buffer);
        }
        ggml_free(ggml);
        ggml_backend_free(backend);
        throw;
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ggml);
    ggml_backend_free(backend);
}

}  // namespace

int main() try {
    run();
    std::cout << "graph_optimizer_identity_fold passed\n";
    return 0;
} catch (const std::exception & error) {
    std::cerr << "graph_optimizer_identity_fold_test failed: " << error.what() << "\n";
    return 1;
}
