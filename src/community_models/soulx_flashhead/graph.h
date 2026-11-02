#pragma once
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/constant_tensor_cache.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/scaled_dot_product_attention.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"
#include <ggml-alloc.h>
#include <map>
#include <unordered_map>

namespace engine::community_models::soulx_flashhead {
using core::TensorShape;
using core::TensorValue;

// Component-owned weight bindings are shared by its cached shape-specific graphs.
struct Weights {
    core::BackendWeightStore store;
    std::unordered_map<std::string, TensorValue> values;
    std::shared_ptr<const assets::TensorSource> source;
    Weights(core::ExecutionContext & execution, std::shared_ptr<const assets::TensorSource> src,
            const std::string & name)
        : store(execution.backend(), execution.backend_type(), name, 4 * 1024 * 1024), source(std::move(src)) {
        for (const auto & tensor : source->tensors())
            get(tensor.name);
        store.upload();
    }
    const TensorValue & get(const std::string & name) {
        auto it = values.find(name);
        if (it != values.end())
            return it->second;
        auto shape = source->require_metadata(name).shape;
        TensorValue v;
        v = store.load_tensor(
            *source, name,
            shape.size() == 2 ? assets::TensorStorageType::Native
                              : (shape.size() == 4 ? assets::TensorStorageType::F16 : assets::TensorStorageType::F32),
            shape);
        return values.emplace(name, v).first->second;
    }
};

class Graph {
  public:
    Graph(core::ExecutionContext & execution, const std::string & name, ggml_gallocr_t shared_allocator = nullptr)
        : execution(execution), name(name),
          constants(execution.backend(), execution.config().threads, name + ".constants", 128 * 1024) {
        allocator = shared_allocator;
        owns_allocator = shared_allocator == nullptr;
        context.reset(ggml_init({32 * 1024 * 1024, nullptr, true}));
        if (!context)
            throw std::runtime_error("Cannot allocate FlashHead graph metadata");
        ctx = {context.get(), this->name.c_str(), execution.backend_type()};
    }
    virtual ~Graph() {
        host_plan.reset();
        if (graph)
            core::release_backend_graph_resources(execution.backend(), graph, true);
        if (allocator && owns_allocator)
            ggml_gallocr_free(allocator);
    }
    Graph(const Graph &) = delete;
    Graph & operator=(const Graph &) = delete;
    TensorValue input(const std::string & key, std::initializer_list<int64_t> shape) {
        auto v = core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims(shape));
        ggml_set_input(v.tensor);
        inputs.emplace(key, v);
        return v;
    }
    TensorValue constant(const std::string & key, std::initializer_list<int64_t> shape, std::vector<float> data) {
        auto v = constants.make_f32(TensorShape::from_dims(shape), data);
        ggml_set_name(v.tensor, key.c_str());
        return v;
    }
    void finish(const TensorValue & value, std::initializer_list<TensorValue> retained = {}) {
        output = value;
        graph = ggml_new_graph_custom(context.get(), 32768, false);
        ggml_set_output(output.tensor);
        ggml_build_forward_expand(graph, output.tensor);
        for (const auto & tensor : retained) {
            ggml_set_output(tensor.tensor);
            ggml_build_forward_expand(graph, tensor.tensor);
        }
        constants.ensure_uploaded();
        core::validate_backend_graph_supported(execution.backend(), graph, name.c_str());
        if (owns_allocator) {
            allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend()));
            if (!allocator || !ggml_gallocr_alloc_graph(allocator, graph))
                throw std::runtime_error("Cannot allocate " + name);
        }
        core::prepare_host_graph_plan(execution, graph, host_plan);
        debug::timing_log_scalar(name + ".buffer_mb", ggml_gallocr_get_buffer_size(allocator, 0) / (1024. * 1024.));
    }
    void execute(std::initializer_list<std::pair<const char *, const std::vector<float> *>> host,
                 std::initializer_list<std::pair<const char *, TensorValue>> device = {}) {
        const auto started = std::chrono::steady_clock::now();
        for (const auto & item : host)
            core::write_tensor_f32(inputs.at(item.first), *item.second);
        for (const auto & item : device) {
            auto & destination = inputs.at(item.first);
            core::validate_shape(item.second, destination.shape, item.first);
            if (item.second.type != destination.type || !ggml_is_contiguous(item.second.tensor))
                throw std::runtime_error("FlashHead device input requires matching contiguous storage");
            ggml_backend_tensor_copy_async(execution.backend(), execution.backend(), item.second.tensor,
                                          destination.tensor);
        }
        debug::timing_log_scalar(name + ".input_upload_ms", debug::elapsed_ms(started));
        const auto compute_started = std::chrono::steady_clock::now();
        if (core::compute_graph(execution, graph, host_plan, name.c_str()) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("FlashHead graph execution failed: " + name);
        debug::timing_log_scalar(name + ".graph.compute_ms", debug::elapsed_ms(compute_started));
    }
    const std::vector<float> & run(std::initializer_list<std::pair<const char *, const std::vector<float> *>> data) {
        const auto started = std::chrono::steady_clock::now();
        execute(data);
        const auto read_started = std::chrono::steady_clock::now();
        core::read_tensor_float_into(output.tensor, result);
        debug::timing_log_scalar(name + ".output_read_ms", debug::elapsed_ms(read_started));
        debug::timing_log_scalar(name + ".wall_ms", debug::elapsed_ms(started));
        return result;
    }
    TensorValue run_device(std::initializer_list<std::pair<const char *, const std::vector<float> *>> host,
                           std::initializer_list<std::pair<const char *, TensorValue>> device = {}) {
        const auto started = std::chrono::steady_clock::now();
        execute(host, device);
        debug::timing_log_scalar(name + ".wall_ms", debug::elapsed_ms(started));
        return output;
    }
    core::ExecutionContext & execution;
    std::string name;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context{nullptr, ggml_free};
    core::ModuleBuildContext ctx;
    ggml_cgraph * graph = nullptr;
    ggml_gallocr_t allocator = nullptr;
    bool owns_allocator = true;
    TensorValue output;
    std::map<std::string, TensorValue> inputs;
    core::ConstantTensorCache constants;
    core::HostGraphPlan host_plan;
    std::vector<float> result;
};
} // namespace engine::community_models::soulx_flashhead
