#include "engine/framework/modules/asr_helpers.h"

#include "test_assert.h"

#include <iostream>
#include <stdexcept>

namespace {

using engine::modules::asr_graph_capacity_usable;
using engine::test::require;

void test_exact_match_is_reusable() {
    require(asr_graph_capacity_usable(1000, 1000), "a graph built for the request size is reusable");
}

void test_undersized_graph_is_not_reusable() {
    require(!asr_graph_capacity_usable(999, 1000), "a graph smaller than the request cannot hold it");
    require(!asr_graph_capacity_usable(1, 1000), "a much smaller graph cannot hold the request");
}

void test_slightly_oversized_graph_is_reusable() {
    // Clip lengths that wobble by a few percent should not force a rebuild on
    // every call; the wasted compute is bounded by the same few percent.
    require(asr_graph_capacity_usable(1050, 1000), "a 5% oversized graph is worth reusing");
    require(asr_graph_capacity_usable(1100, 1000), "the tolerance is inclusive at its edge");
}

void test_oversized_graph_is_rejected() {
    // This is the issue #617 case: one long request must not leave a capacity
    // behind that every later short request pays for. The encoder zero-pads up
    // to the built capacity, so reuse here would cost ~8x on every call.
    require(!asr_graph_capacity_usable(1101, 1000), "past the tolerance, rebuilding wins");
    require(!asr_graph_capacity_usable(8000, 1000), "a graph 8x the request must be rebuilt");
}

void test_zero_sizes_are_handled_without_special_casing() {
    // Every caller rejects a non-positive frame count before reaching here, so
    // this pins down behaviour rather than guarding a reachable path.
    require(!asr_graph_capacity_usable(0, 1000), "an empty graph cannot hold a request");
    require(asr_graph_capacity_usable(0, 0), "zero capacity trivially holds a zero request");
}

}  // namespace

int main() {
    try {
        test_exact_match_is_reusable();
        test_undersized_graph_is_not_reusable();
        test_slightly_oversized_graph_is_reusable();
        test_oversized_graph_is_rejected();
        test_zero_sizes_are_handled_without_special_casing();
        std::cout << "asr_graph_capacity_test passed\n";
    } catch (const std::exception & ex) {
        std::cerr << "asr_graph_capacity_test failed: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
