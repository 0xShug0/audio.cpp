#include "engine/framework/core/backend.h"

#include "test_assert.h"

#include <cstring>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace {

using engine::core::BackendConfig;
using engine::core::BackendType;
using engine::test::require;
using engine::test::require_eq;

struct BackendHandle {
    ggml_backend_t handle = nullptr;

    explicit BackendHandle(ggml_backend_t backend) : handle(backend) {}
    BackendHandle(const BackendHandle &) = delete;
    BackendHandle & operator=(const BackendHandle &) = delete;
    ~BackendHandle() {
        if (handle != nullptr) {
            ggml_backend_free(handle);
        }
    }
};

const char * backend_label(BackendType type) {
    switch (type) {
        case BackendType::Cpu:           return "cpu";
        case BackendType::Cuda:          return "cuda";
        case BackendType::Hip:           return "hip";
        case BackendType::Vulkan:        return "vulkan";
        case BackendType::Metal:         return "metal";
        case BackendType::BestAvailable: return "best";
    }
    return "unknown";
}

// Mirrors the registry-name mapping in src/framework/core/backend.cpp. Duplicated on purpose:
// the test pins the contract rather than reusing the implementation's table.
std::optional<BackendType> backend_type_for_reg_name(const char * reg_name) {
    if (reg_name == nullptr) {
        return std::nullopt;
    }
    // CUDA's registry name is "CUDA" or "MUSA" depending on the ggml build.
    if (std::strcmp(reg_name, "CUDA") == 0 || std::strcmp(reg_name, "MUSA") == 0) {
        return BackendType::Cuda;
    }
    if (std::strcmp(reg_name, "ROCm") == 0) {
        return BackendType::Hip;
    }
    if (std::strcmp(reg_name, "Vulkan") == 0) {
        return BackendType::Vulkan;
    }
    if (std::strcmp(reg_name, "MTL") == 0) {
        return BackendType::Metal;
    }
    return std::nullopt;
}

void ensure_backends_loaded() {
    if (ggml_backend_reg_count() == 0) {
        ggml_backend_load_all();
    }
}

std::string init_backend_error(const BackendConfig & config) {
    try {
        BackendHandle backend(engine::core::init_backend(config));
        return {};
    } catch (const std::exception & error) {
        return error.what();
    }
}

// Every device ggml enumerates for an accelerator registry must be selectable by index, and
// the resulting handle must classify back to the requested backend. This is the regression
// guard for identifying backends by device type (Metal reports GPU, not ACCEL; Vulkan reports
// IGPU on integrated GPUs) or by a single CUDA registry name (ROCm/MUSA builds differ).
void test_every_enumerated_device_resolves() {
    ensure_backends_loaded();

    size_t checked_registries = 0;
    for (size_t i = 0; i < ggml_backend_reg_count(); ++i) {
        ggml_backend_reg_t reg = ggml_backend_reg_get(i);
        require(reg != nullptr, "registry " + std::to_string(i) + " is null");

        const char * reg_name = ggml_backend_reg_name(reg);
        const auto type = backend_type_for_reg_name(reg_name);
        if (!type.has_value()) {
            continue;
        }

        const size_t dev_count = ggml_backend_reg_dev_count(reg);
        const std::string label = std::string(reg_name) + "/" + backend_label(*type);
        require(dev_count > 0, label + " registry exposes no devices");
        ++checked_registries;

        for (size_t device = 0; device < dev_count; ++device) {
            BackendConfig config;
            config.type = *type;
            config.device = static_cast<int>(device);

            BackendHandle backend(engine::core::init_backend(config));
            require(
                backend.handle != nullptr,
                label + " device " + std::to_string(device) + " returned a null backend");
            require_eq(
                static_cast<int>(engine::core::backend_type(backend.handle)),
                static_cast<int>(*type),
                label + " device " + std::to_string(device) + " backend_type round trip");
            require(
                !engine::core::is_host_backend(backend.handle),
                label + " device " + std::to_string(device) + " must not be a host backend");
            require(
                !engine::core::uses_host_graph_plan(backend.handle),
                label + " device " + std::to_string(device) + " must not use a host graph plan");

            // Shares the device lookup with init_backend, so it must agree that the device exists.
            // Drivers are free to not report memory, but a reported snapshot must be consistent.
            const auto memory = engine::core::query_backend_memory(config);
            if (memory.available) {
                require_eq(
                    memory.used_bytes,
                    memory.total_bytes - memory.free_bytes,
                    label + " device " + std::to_string(device) + " memory snapshot");
            }
        }

        BackendConfig out_of_range;
        out_of_range.type = *type;
        out_of_range.device = static_cast<int>(dev_count);
        const std::string out_of_range_error = init_backend_error(out_of_range);
        require(
            !out_of_range_error.empty(),
            label + " accepted out-of-range device index " + std::to_string(dev_count));
        require(
            out_of_range_error.find("available:") != std::string::npos,
            label + " out-of-range error must list the available devices, got: " + out_of_range_error);

        BackendConfig negative;
        negative.type = *type;
        negative.device = -1;
        require(
            !init_backend_error(negative).empty(),
            label + " accepted a negative device index");
    }

    std::cout << "  checked " << checked_registries << " accelerator registr"
              << (checked_registries == 1 ? "y" : "ies") << '\n';
}

// A backend must never be satisfied by a device from a registry it does not own, so requesting
// one that is not registered in this build has to fail rather than pick up a stray device.
void test_unregistered_backends_are_rejected() {
    ensure_backends_loaded();

    for (const BackendType type : {
             BackendType::Cuda,
             BackendType::Hip,
             BackendType::Vulkan,
             BackendType::Metal,
         }) {
        bool registered = false;
        for (size_t i = 0; i < ggml_backend_reg_count(); ++i) {
            ggml_backend_reg_t reg = ggml_backend_reg_get(i);
            if (reg == nullptr) {
                continue;
            }
            const auto reg_type = backend_type_for_reg_name(ggml_backend_reg_name(reg));
            if (reg_type.has_value() && *reg_type == type) {
                registered = true;
                break;
            }
        }
        if (registered) {
            continue;
        }

        BackendConfig config;
        config.type = type;
        const std::string error = init_backend_error(config);
        require(
            !error.empty(),
            std::string(backend_label(type)) + " is not registered but init_backend succeeded");
        require(
            error.find("not registered in this build") != std::string::npos,
            std::string(backend_label(type)) +
                " should report that it is not registered, got: " + error);
    }
}

void test_host_and_best_backends_still_resolve() {
    BackendConfig cpu;
    cpu.type = BackendType::Cpu;
    BackendHandle cpu_backend(engine::core::init_backend(cpu));
    require(cpu_backend.handle != nullptr, "cpu backend is null");
    require_eq(
        static_cast<int>(engine::core::backend_type(cpu_backend.handle)),
        static_cast<int>(BackendType::Cpu),
        "cpu backend_type round trip");
    require(engine::core::is_host_backend(cpu_backend.handle), "cpu backend must be a host backend");
    require(
        engine::core::uses_host_graph_plan(cpu_backend.handle),
        "cpu backend must use a host graph plan");

    BackendConfig best;
    best.type = BackendType::BestAvailable;
    BackendHandle best_backend(engine::core::init_backend(best));
    require(best_backend.handle != nullptr, "best backend is null");
}

}  // namespace

int main() {
    try {
        test_every_enumerated_device_resolves();
        test_unregistered_backends_are_rejected();
        test_host_and_best_backends_still_resolve();
    } catch (const std::exception & error) {
        std::cerr << "backend_device_resolution_test failed: " << error.what() << '\n';
        return 1;
    }
    std::cout << "backend_device_resolution_test passed\n";
    return 0;
}
