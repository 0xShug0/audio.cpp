#include "engine/framework/core/backend.h"
#include "engine/framework/core/execution_context.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#include "ggml-cpu.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#ifdef GGML_USE_VULKAN
#include "ggml-vulkan.h"
#endif

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

namespace engine::core {

namespace {

bool is_cuda_backend_handle(ggml_backend_t backend) {
#ifdef GGML_USE_CUDA
    if (backend == nullptr) {
        return false;
    }
    ggml_backend_dev_t device = ggml_backend_get_device(backend);
    return device != nullptr && ggml_backend_dev_backend_reg(device) == ggml_backend_cuda_reg();
#else
    (void)backend;
    return false;
#endif
}

bool is_vulkan_backend_handle(ggml_backend_t backend) {
#ifdef GGML_USE_VULKAN
    if (backend == nullptr) {
        return false;
    }
    ggml_backend_dev_t device = ggml_backend_get_device(backend);
    return device != nullptr && ggml_backend_dev_backend_reg(device) == ggml_backend_vk_reg();
#else
    (void)backend;
    return false;
#endif
}

bool is_metal_backend_handle(ggml_backend_t backend) {
#ifdef GGML_USE_METAL
    if (backend == nullptr) {
        return false;
    }
    ggml_backend_dev_t device = ggml_backend_get_device(backend);
    return device != nullptr && ggml_backend_dev_backend_reg(device) == ggml_backend_metal_reg();
#else
    (void)backend;
    return false;
#endif
}

bool backend_name_has_prefix(ggml_backend_t backend, const char * prefix) {
    if (backend == nullptr || prefix == nullptr) {
        return false;
    }
    const char * name = ggml_backend_name(backend);
    return name != nullptr && std::strncmp(name, prefix, std::strlen(prefix)) == 0;
}

#ifndef NDEBUG
bool backend_graph_validation_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("ENGINE_VALIDATE_BACKEND_GRAPH");
        return value != nullptr && std::strcmp(value, "1") == 0;
    }();
    return enabled;
}
#endif

}

ggml_backend_t init_backend(const BackendConfig & config) {
    switch (config.type) {
        case BackendType::Cpu: {
            ggml_backend_t backend = ggml_backend_cpu_init();
            if (backend == nullptr) {
                throw std::runtime_error("Failed to initialize CPU backend");
            }
            return backend;
        }
        case BackendType::Cuda: {
#ifdef GGML_USE_CUDA
            ggml_backend_t backend = ggml_backend_cuda_init(config.device);
            if (backend == nullptr) {
                throw std::runtime_error("Failed to initialize CUDA backend");
            }
            return backend;
#else
            throw std::runtime_error("CUDA backend requested but this build does not include GGML_USE_CUDA");
#endif
        }
        case BackendType::Vulkan: {
#ifdef GGML_USE_VULKAN
            if (config.device < 0) {
                throw std::runtime_error("Vulkan backend requested with negative device index");
            }
            ggml_backend_t backend = ggml_backend_vk_init(static_cast<size_t>(config.device));
            if (backend == nullptr) {
                throw std::runtime_error("Failed to initialize Vulkan backend");
            }
            return backend;
#else
            throw std::runtime_error("Vulkan backend requested but this build does not include GGML_USE_VULKAN");
#endif
        }
        case BackendType::Metal: {
#ifdef GGML_USE_METAL
            ggml_backend_t backend = nullptr;
            if (config.device == 0) {
                backend = ggml_backend_metal_init();
            } else if (config.device > 0) {
                ggml_backend_dev_t device = ggml_backend_reg_dev_get(ggml_backend_metal_reg(), static_cast<size_t>(config.device));
                if (device != nullptr) {
                    backend = ggml_backend_dev_init(device, nullptr);
                }
            } else {
                throw std::runtime_error("Metal backend requested with negative device index");
            }
            if (backend == nullptr) {
                throw std::runtime_error("Failed to initialize Metal backend");
            }
            return backend;
#else
            throw std::runtime_error("Metal backend requested but this build does not include GGML_USE_METAL");
#endif
        }
        case BackendType::BestAvailable: {
            ggml_backend_t backend = ggml_backend_init_best();
            if (backend == nullptr) {
                throw std::runtime_error("Failed to initialize best backend");
            }
            return backend;
        }
        default:
            throw std::runtime_error("Unsupported backend type");
    }
}

void set_backend_threads(ggml_backend_t backend, int threads) {
    if (ggml_backend_is_cpu(backend)) {
        ggml_backend_cpu_set_n_threads(backend, threads);
    }
}

bool is_host_backend(ggml_backend_t backend) {
    if (backend == nullptr) {
        return false;
    }
    ggml_backend_dev_t device = ggml_backend_get_device(backend);
    return device != nullptr && ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_CPU;
}

BackendType backend_type(ggml_backend_t backend) {
    if (is_host_backend(backend)) {
        return BackendType::Cpu;
    }
    if (is_cuda_backend_handle(backend)) {
        return BackendType::Cuda;
    }
    if (is_vulkan_backend_handle(backend)) {
        return BackendType::Vulkan;
    }
    if (is_metal_backend_handle(backend)) {
        return BackendType::Metal;
    }
    return BackendType::BestAvailable;
}

bool uses_host_graph_plan(BackendType type) {
    return type == BackendType::Cpu;
}

bool uses_host_graph_plan(ggml_backend_t backend) {
    return is_host_backend(backend);
}

bool requested_backend_uses_host_graph_plan(const BackendConfig & config) {
    return uses_host_graph_plan(config.type);
}

void release_backend_graph_resources(ggml_backend_t backend, ggml_cgraph * graph) {
    if (backend == nullptr || graph == nullptr) {
        return;
    }
#ifdef GGML_USE_CUDA
    if (backend_name_has_prefix(backend, "CUDA")) {
        ggml_backend_cuda_clear_graph(backend, graph);
    }
#endif
}

void release_backend_graph_resources(BackendType backend_type, ggml_backend_t backend, ggml_cgraph * graph) {
    if (backend == nullptr || graph == nullptr) {
        return;
    }
#ifdef GGML_USE_CUDA
    if (backend_type == BackendType::Cuda) {
        ggml_backend_cuda_clear_graph(backend, graph);
    }
#else
    (void)backend_type;
#endif
}

void validate_backend_graph_supported(ggml_backend_t backend, ggml_cgraph * graph, const char * label) {
    if (backend == nullptr || graph == nullptr) {
        throw std::runtime_error("Cannot validate backend graph support for null backend or graph");
    }
    const int nodes = ggml_graph_n_nodes(graph);
    for (int i = 0; i < nodes; ++i) {
        ggml_tensor * node = ggml_graph_node(graph, i);
        if (node != nullptr && !ggml_backend_supports_op(backend, node)) {
            const char * graph_label = label != nullptr ? label : "graph";
            std::string op_name = ggml_op_name(node->op);
            if (node->op == GGML_OP_UNARY) {
                op_name += "/";
                op_name += ggml_unary_op_name(ggml_get_unary_op(node));
            }
            throw std::runtime_error(
                std::string(graph_label) +
                " contains unsupported backend op '" +
                op_name +
                "' at node " +
                std::to_string(i) +
                " tensor '" +
                (node->name[0] != '\0' ? node->name : "<unnamed>") +
                "'");
        }
    }
}

BackendMemorySnapshot query_backend_memory(ggml_backend_t backend, int device_hint) {
    BackendMemorySnapshot snapshot;
#ifdef GGML_USE_CUDA
    if (is_cuda_backend_handle(backend)) {
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        ggml_backend_cuda_get_device_memory(device_hint, &free_bytes, &total_bytes);
        if (total_bytes > 0 && free_bytes <= total_bytes) {
            snapshot.available = true;
            snapshot.total_bytes = static_cast<int64_t>(total_bytes);
            snapshot.free_bytes = static_cast<int64_t>(free_bytes);
            snapshot.used_bytes = static_cast<int64_t>(total_bytes - free_bytes);
        }
        return snapshot;
    }
#endif
#ifdef GGML_USE_VULKAN
    if (is_vulkan_backend_handle(backend)) {
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        ggml_backend_vk_get_device_memory(device_hint, &free_bytes, &total_bytes);
        if (total_bytes > 0 && free_bytes <= total_bytes) {
            snapshot.available = true;
            snapshot.total_bytes = static_cast<int64_t>(total_bytes);
            snapshot.free_bytes = static_cast<int64_t>(free_bytes);
            snapshot.used_bytes = static_cast<int64_t>(total_bytes - free_bytes);
        }
    }
#endif
#ifdef GGML_USE_METAL
    if (is_metal_backend_handle(backend)) {
        ggml_backend_dev_t device = ggml_backend_get_device(backend);
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        if (device != nullptr) {
            ggml_backend_dev_memory(device, &free_bytes, &total_bytes);
        }
        if (total_bytes > 0 && free_bytes <= total_bytes) {
            snapshot.available = true;
            snapshot.total_bytes = static_cast<int64_t>(total_bytes);
            snapshot.free_bytes = static_cast<int64_t>(free_bytes);
            snapshot.used_bytes = static_cast<int64_t>(total_bytes - free_bytes);
        }
        return snapshot;
    }
#endif
    (void)device_hint;
    (void)backend;
    return snapshot;
}

BackendMemorySnapshot query_backend_memory(const BackendConfig & config) {
    BackendMemorySnapshot snapshot;
    switch (config.type) {
        case BackendType::Cuda:
#ifdef GGML_USE_CUDA
        {
            size_t free_bytes = 0;
            size_t total_bytes = 0;
            ggml_backend_cuda_get_device_memory(config.device, &free_bytes, &total_bytes);
            if (total_bytes > 0 && free_bytes <= total_bytes) {
                snapshot.available = true;
                snapshot.total_bytes = static_cast<int64_t>(total_bytes);
                snapshot.free_bytes = static_cast<int64_t>(free_bytes);
                snapshot.used_bytes = static_cast<int64_t>(total_bytes - free_bytes);
            }
            return snapshot;
        }
#else
            return snapshot;
#endif
        case BackendType::Vulkan:
#ifdef GGML_USE_VULKAN
        {
            size_t free_bytes = 0;
            size_t total_bytes = 0;
            ggml_backend_vk_get_device_memory(config.device, &free_bytes, &total_bytes);
            if (total_bytes > 0 && free_bytes <= total_bytes) {
                snapshot.available = true;
                snapshot.total_bytes = static_cast<int64_t>(total_bytes);
                snapshot.free_bytes = static_cast<int64_t>(free_bytes);
                snapshot.used_bytes = static_cast<int64_t>(total_bytes - free_bytes);
            }
            return snapshot;
        }
#else
            return snapshot;
#endif
        case BackendType::Metal:
#ifdef GGML_USE_METAL
        {
            ggml_backend_dev_t device = ggml_backend_reg_dev_get(ggml_backend_metal_reg(), config.device < 0 ? 0 : static_cast<size_t>(config.device));
            if (device != nullptr) {
                size_t free_bytes = 0;
                size_t total_bytes = 0;
                ggml_backend_dev_memory(device, &free_bytes, &total_bytes);
                if (total_bytes > 0 && free_bytes <= total_bytes) {
                    snapshot.available = true;
                    snapshot.total_bytes = static_cast<int64_t>(total_bytes);
                    snapshot.free_bytes = static_cast<int64_t>(free_bytes);
                    snapshot.used_bytes = static_cast<int64_t>(total_bytes - free_bytes);
                }
            }
            return snapshot;
        }
#else
            return snapshot;
#endif
        case BackendType::Cpu:
            return snapshot;
        case BackendType::BestAvailable:
            return snapshot;
    }
    return snapshot;
}

ggml_backend_graph_plan_t create_backend_graph_plan_if_host(ggml_backend_t backend, ggml_cgraph * graph) {
    if (backend == nullptr || graph == nullptr || !is_host_backend(backend)) {
        return nullptr;
    }
    return ggml_backend_graph_plan_create(backend, graph);
}

void free_backend_graph_plan(ggml_backend_t backend, ggml_backend_graph_plan_t & plan) {
    if (plan != nullptr) {
        ggml_backend_graph_plan_free(backend, plan);
        plan = nullptr;
    }
}

ggml_status compute_backend_graph(
    ggml_backend_t backend,
    ggml_cgraph * graph,
    ggml_backend_graph_plan_t plan,
    const char * label) {
    if (backend == nullptr || graph == nullptr) {
        return GGML_STATUS_FAILED;
    }
#ifndef NDEBUG
    if (plan == nullptr && backend_graph_validation_enabled()) {
        validate_backend_graph_supported(backend, graph, label);
    } else {
        (void)label;
    }
#else
    (void)label;
#endif
    return plan != nullptr
        ? ggml_backend_graph_plan_compute(backend, plan)
        : ggml_backend_graph_compute(backend, graph);
}

void prepare_host_graph_plan(const ExecutionContext & execution_context, ggml_cgraph * graph, HostGraphPlan & plan) {
    plan.reset();
    if (!execution_context.uses_host_graph_plan()) {
        return;
    }
    plan.plan = ggml_graph_plan(graph, std::max(1, execution_context.config().threads), nullptr);
    plan.work_buffer.resize(plan.plan.work_size);
    plan.plan.work_data = plan.work_buffer.empty() ? nullptr : plan.work_buffer.data();
}

ggml_status compute_graph(
    const ExecutionContext & execution_context,
    ggml_cgraph * graph,
    HostGraphPlan & plan,
    const char * label) {
    return plan.active()
        ? ggml_graph_compute(graph, &plan.plan)
        : compute_backend_graph(execution_context.backend(), graph, nullptr, label);
}

void write_tensor_f32(const TensorValue & tensor, const float * values, size_t count) {
    if (tensor.type != GGML_TYPE_F32) {
        throw std::runtime_error("write_tensor_f32 requires GGML_TYPE_F32 tensor");
    }
    if (tensor.shape.num_elements() != static_cast<int64_t>(count)) {
        throw std::runtime_error(
            "write_tensor_f32 value count does not match tensor shape for tensor '" +
            std::string(tensor.tensor != nullptr ? tensor.tensor->name : "<null>") +
            "': expected " + std::to_string(tensor.shape.num_elements()) +
            ", got " + std::to_string(count));
    }
    ggml_backend_tensor_set(tensor.tensor, values, 0, count * sizeof(float));
}

void write_tensor_f32_slice(const TensorValue & tensor, size_t element_offset, const float * values, size_t count) {
    if (tensor.type != GGML_TYPE_F32) {
        throw std::runtime_error("write_tensor_f32_slice requires GGML_TYPE_F32 tensor");
    }
    const size_t total = static_cast<size_t>(tensor.shape.num_elements());
    if (element_offset > total || count > total - element_offset) {
        throw std::runtime_error("write_tensor_f32_slice range exceeds tensor shape");
    }
    ggml_backend_tensor_set(
        tensor.tensor,
        values,
        element_offset * sizeof(float),
        count * sizeof(float));
}

void write_tensor_f32(const TensorValue & tensor, const std::vector<float> & values) {
    write_tensor_f32(tensor, values.data(), values.size());
}

void write_tensor_f16(const TensorValue & tensor, const float * values, size_t count) {
    if (tensor.type != GGML_TYPE_F16) {
        throw std::runtime_error("write_tensor_f16 requires GGML_TYPE_F16 tensor");
    }
    if (tensor.shape.num_elements() != static_cast<int64_t>(count)) {
        throw std::runtime_error(
            "write_tensor_f16 value count does not match tensor shape for tensor '" +
            std::string(tensor.tensor != nullptr ? tensor.tensor->name : "<null>") +
            "': expected " + std::to_string(tensor.shape.num_elements()) +
            ", got " + std::to_string(count));
    }
    std::vector<ggml_fp16_t> fp16_values(count);
    ggml_fp32_to_fp16_row(values, fp16_values.data(), static_cast<int64_t>(count));
    ggml_backend_tensor_set(tensor.tensor, fp16_values.data(), 0, count * sizeof(ggml_fp16_t));
}

void write_tensor_f16(const TensorValue & tensor, const std::vector<float> & values) {
    write_tensor_f16(tensor, values.data(), values.size());
}

void write_tensor_bf16(const TensorValue & tensor, const float * values, size_t count) {
    if (tensor.type != GGML_TYPE_BF16) {
        throw std::runtime_error("write_tensor_bf16 requires GGML_TYPE_BF16 tensor");
    }
    if (tensor.shape.num_elements() != static_cast<int64_t>(count)) {
        throw std::runtime_error(
            "write_tensor_bf16 value count does not match tensor shape for tensor '" +
            std::string(tensor.tensor != nullptr ? tensor.tensor->name : "<null>") +
            "': expected " + std::to_string(tensor.shape.num_elements()) +
            ", got " + std::to_string(count));
    }
    std::vector<ggml_bf16_t> bf16_values(count);
    ggml_fp32_to_bf16_row(values, bf16_values.data(), static_cast<int64_t>(count));
    ggml_backend_tensor_set(tensor.tensor, bf16_values.data(), 0, count * sizeof(ggml_bf16_t));
}

void write_tensor_bf16(const TensorValue & tensor, const std::vector<float> & values) {
    write_tensor_bf16(tensor, values.data(), values.size());
}

void write_tensor_i32(const TensorValue & tensor, const int32_t * values, size_t count) {
    if (tensor.type != GGML_TYPE_I32) {
        throw std::runtime_error("write_tensor_i32 requires GGML_TYPE_I32 tensor");
    }
    if (tensor.shape.num_elements() != static_cast<int64_t>(count)) {
        throw std::runtime_error("write_tensor_i32 value count does not match tensor shape");
    }
    ggml_backend_tensor_set(tensor.tensor, values, 0, count * sizeof(int32_t));
}

void write_tensor_i32(const TensorValue & tensor, const std::vector<int32_t> & values) {
    write_tensor_i32(tensor, values.data(), values.size());
}

template <typename T>
void read_tensor_typed_into(const ggml_tensor * tensor, ggml_type expected_type, std::vector<T> & values) {
    if (tensor->type != expected_type) {
        throw std::runtime_error("read_tensor_typed type mismatch");
    }
    const size_t element_count = static_cast<size_t>(ggml_nelements(tensor));
    values.resize(element_count);
    const size_t byte_count = static_cast<size_t>(ggml_nbytes(tensor));
    if (ggml_is_contiguous(tensor) && tensor->nb[0] == sizeof(T)) {
        ggml_backend_tensor_get(tensor, values.data(), 0, byte_count);
        return;
    }

    std::vector<uint8_t> raw(byte_count);
    ggml_backend_tensor_get(tensor, raw.data(), 0, raw.size());
    size_t out_index = 0;
    const char * base = reinterpret_cast<const char *>(raw.data());
    for (int64_t i3 = 0; i3 < tensor->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < tensor->ne[2]; ++i2) {
            for (int64_t i1 = 0; i1 < tensor->ne[1]; ++i1) {
                for (int64_t i0 = 0; i0 < tensor->ne[0]; ++i0) {
                    const char * ptr = base
                        + i0 * tensor->nb[0]
                        + i1 * tensor->nb[1]
                        + i2 * tensor->nb[2]
                        + i3 * tensor->nb[3];
                    std::memcpy(&values[out_index++], ptr, sizeof(T));
                }
            }
        }
    }
}

template <typename T>
std::vector<T> read_tensor_typed(const ggml_tensor * tensor, ggml_type expected_type) {
    std::vector<T> values;
    read_tensor_typed_into(tensor, expected_type, values);
    return values;
}

void read_tensor_f32_into(const ggml_tensor * tensor, std::vector<float> & values) {
    read_tensor_typed_into<float>(tensor, GGML_TYPE_F32, values);
}

std::vector<float> read_tensor_f32(const ggml_tensor * tensor) {
    return read_tensor_typed<float>(tensor, GGML_TYPE_F32);
}

void read_tensor_f16_into(const ggml_tensor * tensor, std::vector<float> & values) {
    const auto fp16_values = read_tensor_typed<ggml_fp16_t>(tensor, GGML_TYPE_F16);
    values.resize(fp16_values.size());
    ggml_fp16_to_fp32_row(fp16_values.data(), values.data(), static_cast<int64_t>(values.size()));
}

std::vector<float> read_tensor_f16(const ggml_tensor * tensor) {
    std::vector<float> values;
    read_tensor_f16_into(tensor, values);
    return values;
}

void read_tensor_bf16_into(const ggml_tensor * tensor, std::vector<float> & values) {
    const auto bf16_values = read_tensor_typed<ggml_bf16_t>(tensor, GGML_TYPE_BF16);
    values.resize(bf16_values.size());
    ggml_bf16_to_fp32_row(bf16_values.data(), values.data(), static_cast<int64_t>(values.size()));
}

std::vector<float> read_tensor_bf16(const ggml_tensor * tensor) {
    std::vector<float> values;
    read_tensor_bf16_into(tensor, values);
    return values;
}

void read_tensor_i32_into(const ggml_tensor * tensor, std::vector<int32_t> & values) {
    read_tensor_typed_into<int32_t>(tensor, GGML_TYPE_I32, values);
}

std::vector<int32_t> read_tensor_i32(const ggml_tensor * tensor) {
    return read_tensor_typed<int32_t>(tensor, GGML_TYPE_I32);
}

}  // namespace engine::core
