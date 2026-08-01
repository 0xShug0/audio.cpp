#pragma once

#include "engine/framework/core/module.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <string>
#include <vector>

namespace engine::core {

class ExecutionContext;

struct BackendConfig {
    BackendType type = BackendType::Cpu;
    int device = 0;
    int threads = 1;
};

struct BackendMemorySnapshot {
    bool available = false;
    int64_t total_bytes = 0;
    int64_t used_bytes = 0;
    int64_t free_bytes = 0;
};

ggml_backend_t init_backend(const BackendConfig & config);
void set_backend_threads(ggml_backend_t backend, int threads);
BackendType backend_type(ggml_backend_t backend);
bool is_host_backend(ggml_backend_t backend);
bool uses_host_graph_plan(BackendType type);
bool uses_host_graph_plan(ggml_backend_t backend);
bool requested_backend_uses_host_graph_plan(const BackendConfig & config);
void release_backend_graph_resources(ggml_backend_t backend, ggml_cgraph * graph);
void release_backend_graph_resources(BackendType backend_type, ggml_backend_t backend, ggml_cgraph * graph);
void validate_backend_graph_supported(ggml_backend_t backend, ggml_cgraph * graph, const char * label);
BackendMemorySnapshot query_backend_memory(ggml_backend_t backend, int device_hint);
BackendMemorySnapshot query_backend_memory(const BackendConfig & config);
ggml_backend_graph_plan_t create_backend_graph_plan_if_host(ggml_backend_t backend, ggml_cgraph * graph);
void free_backend_graph_plan(ggml_backend_t backend, ggml_backend_graph_plan_t & plan);
ggml_status compute_backend_graph(
    ggml_backend_t backend,
    ggml_cgraph * graph,
    ggml_backend_graph_plan_t plan = nullptr,
    const char * label = nullptr);

struct HostGraphPlan {
    ggml_backend_graph_plan_t plan = nullptr;
    ggml_backend_t backend = nullptr;

    ~HostGraphPlan() { reset(); }

    bool active() const noexcept { return plan != nullptr; }
    void reset() {
        if (plan != nullptr && backend != nullptr) {
            ggml_backend_graph_plan_free(backend, plan);
        }
        plan = nullptr;
        backend = nullptr;
    }
};

void prepare_host_graph_plan(const ExecutionContext & execution_context, ggml_cgraph * graph, HostGraphPlan & plan);
ggml_status compute_graph(
    const ExecutionContext & execution_context,
    ggml_cgraph * graph,
    HostGraphPlan & plan,
    const char * label = nullptr);

void write_tensor_f32(const TensorValue & tensor, const float * values, size_t count);
void write_tensor_f32_slice(const TensorValue & tensor, size_t element_offset, const float * values, size_t count);
void write_tensor_f32(const TensorValue & tensor, const std::vector<float> & values);
void write_tensor_f16(const TensorValue & tensor, const float * values, size_t count);
void write_tensor_f16(const TensorValue & tensor, const std::vector<float> & values);
void write_tensor_bf16(const TensorValue & tensor, const float * values, size_t count);
void write_tensor_bf16(const TensorValue & tensor, const std::vector<float> & values);
void write_tensor_i32(const TensorValue & tensor, const int32_t * values, size_t count);
void write_tensor_i32(const TensorValue & tensor, const std::vector<int32_t> & values);
void read_tensor_f32_into(const ggml_tensor * tensor, std::vector<float> & values);
std::vector<float> read_tensor_f32(const ggml_tensor * tensor);
void read_tensor_f16_into(const ggml_tensor * tensor, std::vector<float> & values);
std::vector<float> read_tensor_f16(const ggml_tensor * tensor);
void read_tensor_bf16_into(const ggml_tensor * tensor, std::vector<float> & values);
std::vector<float> read_tensor_bf16(const ggml_tensor * tensor);
void read_tensor_i32_into(const ggml_tensor * tensor, std::vector<int32_t> & values);
std::vector<int32_t> read_tensor_i32(const ggml_tensor * tensor);

}  // namespace engine::core
