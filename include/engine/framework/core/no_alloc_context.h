#pragma once

#include <ggml.h>

#include <cstddef>

namespace engine::core {

// How a runtime sizes the no_alloc ggml contexts it owns.
//
// A no_alloc context only ever holds tensor headers (and the graph object,
// when the graph is built in it). Tensor data lives in backend buffers that
// the allocators size exactly. ggml_init still allocates the whole mem_size
// up front, so a context sized from a config number reserves that number
// whether or not anything fills it: host memory that is never touched on
// Linux, commit charge on Windows.
//
// ExplicitBytes keeps the caller-supplied byte counts; behaviour is unchanged.
// FromCapacity ignores them and sizes each context from the node cap or
// tensor count the runtime already knows (each runtime documents which).
// The default is ExplicitBytes everywhere: a call site opts in by setting
// FromCapacity on the runtime's options, and nothing changes without that.
enum class ContextSizing {
    ExplicitBytes,
    FromCapacity,
};

// Bytes a no_alloc context needs to hold `tensors` tensor headers.
inline size_t no_alloc_tensor_context_bytes(size_t tensors) {
    return tensors * ggml_tensor_overhead();
}

// Bytes a no_alloc context needs when a graph of up to `nodes` nodes is
// built in it: a header for every node and every leaf the graph may hold
// (ggml_new_graph_custom reserves both arrays at `nodes`), plus the graph
// object itself. That covers every tensor reachable from the graph, which
// ggml_build_forward_expand caps at `nodes` each; tensors a runtime creates
// in the context but never links into the graph sit outside that cap, so a
// runtime that makes many of those should add its own slack.
inline size_t no_alloc_graph_context_bytes(size_t nodes) {
    return 2 * nodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(nodes, false);
}

// Capacity of a weight context in tensor headers rather than bytes, so a
// BackendWeightStore call site says what its number means.
struct TensorCapacity {
    size_t tensors = 0;
};

}  // namespace engine::core
