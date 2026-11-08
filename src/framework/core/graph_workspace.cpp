#include "engine/framework/core/graph_workspace.h"
#include <stdexcept>

namespace engine::core {

GraphWorkspace::GraphWorkspace(ggml_backend_t backend)
    : allocator_(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)), ggml_gallocr_free) {
    if (!allocator_) throw std::runtime_error("cannot create graph activation workspace");
}
GraphWorkspace::~GraphWorkspace() = default;

void GraphWorkspace::add(ggml_cgraph * graph) {
    if (prepared_) throw std::logic_error("cannot register graphs after workspace preparation");
    graphs_.push_back(graph);
}

void GraphWorkspace::prepare() {
    if (prepared_ || graphs_.empty()) throw std::logic_error("workspace preparation requires unprepared graphs");
    ggml_cgraph * largest = nullptr;
    size_t maximum = 0;
    for (auto * graph : graphs_) {
        size_t bytes = 0;
        ggml_gallocr_reserve_n_size(allocator_.get(), graph, nullptr, nullptr, &bytes);
        if (!largest || bytes > maximum) { largest = graph; maximum = bytes; }
    }
    if (!ggml_gallocr_reserve(allocator_.get(), largest)) throw std::runtime_error("cannot reserve graph activation workspace");
    for (auto * graph : graphs_) {
        if (!ggml_gallocr_reserve(allocator_.get(), graph) || !ggml_gallocr_alloc_graph(allocator_.get(), graph))
            throw std::runtime_error("cannot allocate graph activation workspace");
    }
    prepared_ = true;
}

void GraphWorkspace::reset() {
    graphs_.clear();
    prepared_ = false;
}

size_t GraphWorkspace::bytes() const { return ggml_gallocr_get_buffer_size(allocator_.get(), 0); }

} // namespace engine::core
