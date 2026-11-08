#pragma once

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <memory>
#include <type_traits>
#include <vector>

namespace engine::core {

// Shared activation allocation for graphs that execute serially. Graph owners
// retain their contexts, plans and backend captures; this owns only scratch.
class GraphWorkspace {
public:
    explicit GraphWorkspace(ggml_backend_t backend);
    ~GraphWorkspace();
    GraphWorkspace(const GraphWorkspace &) = delete;
    GraphWorkspace & operator=(const GraphWorkspace &) = delete;
    void add(ggml_cgraph * graph);
    // Register every graph before preparing; allocation addresses remain fixed
    // afterward. Graphs must stay alive through this call.
    void prepare();
    // Start a new graph plan without dropping allocation capacity. The caller
    // must release the previous graphs and their backend resources first.
    void reset();
    size_t bytes() const;
private:
    std::unique_ptr<std::remove_pointer_t<ggml_gallocr_t>, decltype(&ggml_gallocr_free)> allocator_;
    std::vector<ggml_cgraph *> graphs_;
    bool prepared_ = false;
};

} // namespace engine::core
