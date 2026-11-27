#include "engine/framework/core/backend.h"
#include "engine/framework/io/json.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"

#include "test_assert.h"
#include "ggml-alloc.h"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
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
        case BackendType::WebGPU:        return "webgpu";
        case BackendType::Sycl:          return "sycl";
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
    if (std::strcmp(reg_name, "WebGPU") == 0) {
        return BackendType::WebGPU;
    }
    if (std::strcmp(reg_name, "SYCL") == 0) {
        return BackendType::Sycl;
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
             BackendType::WebGPU,
             BackendType::Sycl,
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

// Parsers depend on the text form of --list-devices, so pin it byte for byte against ggml.
void test_text_listing_format_is_stable() {
    ensure_backends_loaded();

    std::string expected;
    size_t count = 0;
    for (size_t i = 0; i < ggml_backend_reg_count(); ++i) {
        ggml_backend_reg_t reg = ggml_backend_reg_get(i);
        for (size_t j = 0; reg != nullptr && j < ggml_backend_reg_dev_count(reg); ++j) {
            ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, j);
            const char * description = ggml_backend_dev_description(dev);
            const std::string name = description != nullptr && description[0] != '\0' ? description : ggml_backend_dev_name(dev);
            const char * types[] = {"CPU", "GPU", "IGPU", "ACCEL", "META"};
            expected += std::string(ggml_backend_reg_name(reg)) + ":" + std::to_string(j) +
                (name.empty() ? "" : " \"" + name + "\"") + " [" + types[ggml_backend_dev_type(dev)] + "]\n";
            ++count;
        }
    }
    expected = "available_devices=" + std::to_string(count) + "\n" + expected +
        "select with: --backend <cuda|hip|vulkan|metal|webgpu|sycl|cpu> --device <index>\n";

    std::ostringstream text;
    engine::core::print_backend_devices(text);
    require_eq(text.str(), expected, "--list-devices text output");
}

std::optional<BackendType> backend_type_for_option(const std::string & option) {
    for (const BackendType type : {BackendType::Cpu, BackendType::Cuda, BackendType::Hip, BackendType::Vulkan, BackendType::Metal, BackendType::WebGPU, BackendType::Sycl}) {
        if (option == backend_label(type)) {
            return type;
        }
    }
    return std::nullopt;
}

// The --backend value for a ggml registry name, spelled out independently of backend.cpp.
std::string expected_backend_option(const std::string & registry) {
    if (registry == "CPU") {
        return "cpu";
    }
    if (registry == "CUDA" || registry == "MUSA") {
        return "cuda";
    }
    if (registry == "ROCm") {
        return "hip";
    }
    if (registry == "Vulkan") {
        return "vulkan";
    }
    if (registry == "MTL") {
        return "metal";
    }
    if (registry == "WebGPU") {
        return "webgpu";
    }
    if (registry == "SYCL") {
        return "sycl";
    }
    return {};
}

// --list-devices --json lists the text listing's devices in the same order, and each
// "backend"/"index" pair, fed to init_backend as --backend/--device are, opens that device.
void test_json_listing_matches_selection() {
    namespace json = engine::io::json;
    std::ostringstream text;
    engine::core::print_backend_devices_json(text);
    const json::Value root = json::parse(text.str());
    require_eq(root.require("schema_version").as_i64(), int64_t{1}, "schema_version");

    const auto listed = engine::core::list_backend_devices();
    const auto & devices = root.require("devices").as_array();
    require_eq(devices.size(), listed.size(), "JSON and text list the same number of devices");
    for (size_t i = 0; i < devices.size(); ++i) {
        const json::Value & device = devices[i];
        const std::string label = listed[i].backend + ":" + std::to_string(listed[i].index);
        require_eq(device.require("registry").as_string(), listed[i].backend, label + " registry");
        require_eq(device.require("index").as_i64(), int64_t{listed[i].index}, label + " index");

        ggml_backend_reg_t reg = ggml_backend_reg_by_name(listed[i].backend.c_str());
        require(reg != nullptr, label + " registry not found");
        ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, static_cast<size_t>(listed[i].index));
        require_eq(device.require("name").as_string(), std::string(ggml_backend_dev_name(dev)), label + " name");

        ggml_backend_dev_props props{};
        ggml_backend_dev_get_props(dev, &props);
        const json::Value & pci_bus_id = device.require("pci_bus_id");
        if (props.device_id == nullptr || props.device_id[0] == '\0') {
            require(pci_bus_id.is_null(), label + " pci_bus_id must be null when ggml has none");
        } else {
            require_eq(pci_bus_id.as_string(), std::string(props.device_id), label + " pci_bus_id");
        }

        const json::Value & memory = device.require("memory");
        if (!memory.is_null()) {
            size_t free_bytes = 0;
            size_t total_bytes = 0;
            ggml_backend_dev_memory(dev, &free_bytes, &total_bytes);
            require_eq(memory.require("total_bytes").as_i64(), static_cast<int64_t>(total_bytes), label + " total_bytes");
            const int64_t free = memory.require("free_bytes").as_i64();
            require(free >= 0 && free <= static_cast<int64_t>(total_bytes), label + " free_bytes out of range");
        }

        // Every registry --backend can select must name its option; the rest (BLAS, RPC) must be null.
        const std::string expected_backend = expected_backend_option(listed[i].backend);
        const json::Value & backend = device.require("backend");
        if (expected_backend.empty()) {
            require(backend.is_null(), label + " backend must be null for a registry --backend cannot select");
            continue;
        }
        require(!backend.is_null(), label + " backend must not be null");
        require_eq(backend.as_string(), expected_backend, label + " backend");
        if (listed[i].backend == "CPU") {
            require(!memory.is_null(), label + " memory must not be null for the CPU");
        }
        const auto type = backend_type_for_option(backend.as_string());
        require(type.has_value(), label + " unknown backend " + backend.as_string());
        BackendConfig config;
        config.type = *type;
        config.device = listed[i].index;
        BackendHandle handle(engine::core::init_backend(config));
        require(
            ggml_backend_get_device(handle.handle) == dev,
            label + " --backend " + backend.as_string() + " --device " + std::to_string(config.device) +
                " opened a different device");
    }
}

// Byte counts must print as exact integers, and unknown values as null.
void test_json_writer_exact_integers_and_nulls() {
    namespace json = engine::io::json;
    engine::core::BackendDeviceInfo gpu;
    gpu.backend = "CUDA";
    gpu.index = 1;
    gpu.type = "GPU";
    gpu.backend_option = "cuda";
    gpu.device_name = "CUDA1";
    gpu.description = "Card \"A\" \\ B";
    gpu.pci_bus_id = "0000:01:00.0";
    gpu.memory.available = true;
    gpu.memory.free_bytes = 9007199254740993;  // 2^53 + 1: not representable as a double
    gpu.memory.total_bytes = 9223372036854775807;
    gpu.caps.async = true;

    engine::core::BackendDeviceInfo blas;
    blas.backend = "BLAS";
    blas.type = "ACCEL";
    blas.device_name = "BLAS";

    std::ostringstream text;
    engine::core::print_backend_devices_json(text, {gpu, blas});
    const std::string out = text.str();
    require(
        out.find("\"memory\":{\"free_bytes\":9007199254740993,\"total_bytes\":9223372036854775807}") != std::string::npos,
        "byte counts must be exact integers, got: " + out);

    const json::Value root = json::parse(out);
    const auto & devices = root.require("devices").as_array();
    require_eq(devices.size(), size_t{2}, "device count");
    require_eq(devices[0].require("backend").as_string(), std::string("cuda"), "backend");
    require_eq(devices[0].require("description").as_string(), gpu.description, "escaped description");
    require_eq(devices[0].require("type").as_string(), std::string("gpu"), "type");
    require(devices[0].require("caps").require("async").as_bool(), "caps.async");
    require(!devices[0].require("caps").require("events").as_bool(), "caps.events");
    for (const char * key : {"backend", "description", "pci_bus_id", "memory"}) {
        require(devices[1].require(key).is_null(), std::string("unknown ") + key + " must be null");
    }
}

void test_backend_impls() {
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto dev = ggml_backend_dev_get(i);
        const char * const * names = ggml_backend_dev_get_impls(dev);
        require(names && names[0] && std::string(names[0]) == "default", "default implementation is discoverable");
        BackendHandle original(ggml_backend_dev_init(dev, nullptr));
        require(original.handle != nullptr, "original initialization");
        require_eq(std::string(ggml_backend_get_impl(original.handle)), std::string("default"), "default identity");
        require(ggml_backend_dev_init_with_impl(dev, "invalid-implementation", nullptr) == nullptr,
                "unknown implementation must not be ignored");
        for (size_t j = 0; names[j]; ++j) {
            BackendHandle selected(ggml_backend_dev_init_with_impl(dev, names[j], nullptr));
            require(selected.handle != nullptr, "advertised implementation initializes");
            require_eq(std::string(ggml_backend_get_impl(selected.handle)), std::string(names[j]), "selected identity");
            require_eq(std::string(ggml_backend_get_impl(original.handle)), std::string("default"), "instance isolation");
        }
        BackendHandle empty(ggml_backend_dev_init_with_impl(dev, "", nullptr));
        require(empty.handle != nullptr, "empty implementation initializes the default");
        require_eq(std::string(ggml_backend_get_impl(empty.handle)), std::string("default"), "empty identity");
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

void test_device_name_resolution_synthetic_cases() {
    auto make_dev = [](std::string reg, std::string opt, int idx, std::string dev_name, std::string desc, std::string type = "GPU") {
        engine::core::BackendDeviceInfo d;
        d.backend = std::move(reg);
        d.backend_option = std::move(opt);
        d.index = idx;
        d.device_name = std::move(dev_name);
        d.description = std::move(desc);
        d.name = !d.description.empty() ? d.description : d.device_name;
        d.type = std::move(type);
        return d;
    };

    const std::vector<engine::core::BackendDeviceInfo> devices = {
        make_dev("CUDA", "cuda", 0, "CUDA0", "NVIDIA GeForce RTX 5090"),
        make_dev("CUDA", "cuda", 1, "CUDA1", "NVIDIA GeForce RTX 4090"),
        make_dev("CUDA", "cuda", 10, "CUDA10", "NVIDIA GeForce RTX 4090 D"),
        make_dev("Vulkan", "vulkan", 0, "Vulkan0", "NVIDIA GeForce RTX 5090"),
        make_dev("Vulkan", "vulkan", 1, "Vulkan1", "Intel(R) Arc(TM) A770 Graphics", "IGPU"),
        make_dev("ROCm", "hip", 0, "ROCm0", "AMD Radeon RX 7900 XTX"),
        make_dev("MTL", "metal", 0, "MTL0", "Apple M3 Max"),
        make_dev("CPU", "cpu", 0, "CPU", "CPU", "CPU"),
        make_dev("BLAS", "", 0, "BLAS", "Accelerate", "ACCEL"),
    };

    // Tier 1: exact case-insensitive ggml device_name match (no prefix collision CUDA1 vs CUDA10).
    {
        const auto r = engine::core::resolve_backend_device("CUDA1", std::nullopt, devices);
        require_eq(static_cast<int>(r.type), static_cast<int>(BackendType::Cuda), "CUDA1 backend");
        require_eq(r.device, 1, "CUDA1 index");

        const auto lower = engine::core::resolve_backend_device("  vulkan1  ", std::nullopt, devices);
        require_eq(static_cast<int>(lower.type), static_cast<int>(BackendType::Vulkan), "trimmed lowercase Vulkan1 backend");
        require_eq(lower.device, 1, "trimmed lowercase Vulkan1 index");

        const auto rocm = engine::core::resolve_backend_device("rocm0", std::nullopt, devices);
        require_eq(static_cast<int>(rocm.type), static_cast<int>(BackendType::Hip), "ROCm0 resolves to Hip");
        require_eq(rocm.device, 0, "ROCm0 index");

        const auto cpu = engine::core::resolve_backend_device("cpu", std::nullopt, devices);
        require_eq(static_cast<int>(cpu.type), static_cast<int>(BackendType::Cpu), "CPU resolves to Cpu");
        require_eq(cpu.device, 0, "CPU index");
    }

    // Tier 2: exact description beats substring (RTX 4090 must not collide with RTX 4090 D).
    {
        const auto r = engine::core::resolve_backend_device("nvidia geforce rtx 4090", std::nullopt, devices);
        require_eq(static_cast<int>(r.type), static_cast<int>(BackendType::Cuda), "exact RTX 4090 backend");
        require_eq(r.device, 1, "exact RTX 4090 index");
    }

    // Tier 3: unique case-insensitive description substring match.
    {
        const auto intel = engine::core::resolve_backend_device("intel", std::nullopt, devices);
        require_eq(static_cast<int>(intel.type), static_cast<int>(BackendType::Vulkan), "Intel substring backend");
        require_eq(intel.device, 1, "Intel substring index");

        const auto radeon = engine::core::resolve_backend_device("7900 xtx", std::nullopt, devices);
        require_eq(static_cast<int>(radeon.type), static_cast<int>(BackendType::Hip), "Radeon substring backend");
        require_eq(radeon.device, 0, "Radeon substring index");
    }

    // Disambiguation via backend_filter when same physical GPU appears under CUDA and Vulkan.
    {
        bool ambiguous = false;
        try {
            (void) engine::core::resolve_backend_device("RTX 5090", std::nullopt, devices);
        } catch (const std::runtime_error & error) {
            const std::string msg = error.what();
            ambiguous = msg.find("ambiguous") != std::string::npos &&
                msg.find("CUDA0") != std::string::npos &&
                msg.find("Vulkan0") != std::string::npos;
        }
        require(ambiguous, "unfiltered RTX 5090 across CUDA and Vulkan must fail as ambiguous");

        const auto vulkan_5090 = engine::core::resolve_backend_device("RTX 5090", BackendType::Vulkan, devices);
        require_eq(static_cast<int>(vulkan_5090.type), static_cast<int>(BackendType::Vulkan), "filtered Vulkan RTX 5090");
        require_eq(vulkan_5090.device, 0, "filtered Vulkan RTX 5090 index");
    }

    // Identical dual GPUs on the same backend: description is ambiguous, exact ggml name selects one.
    {
        const std::vector<engine::core::BackendDeviceInfo> dual_gpu = {
            make_dev("CUDA", "cuda", 0, "CUDA0", "NVIDIA GeForce RTX 5090"),
            make_dev("CUDA", "cuda", 1, "CUDA1", "NVIDIA GeForce RTX 5090"),
        };
        for (const char * query : {"NVIDIA GeForce RTX 5090", "5090"}) {
            bool ambiguous = false;
            try {
                (void) engine::core::resolve_backend_device(query, BackendType::Cuda, dual_gpu);
            } catch (const std::runtime_error & error) {
                const std::string msg = error.what();
                ambiguous = msg.find("ambiguous") != std::string::npos &&
                    msg.find("CUDA0") != std::string::npos &&
                    msg.find("CUDA1") != std::string::npos;
            }
            require(ambiguous, std::string("dual identical GPU query must be ambiguous: ") + query);
        }
        const auto second = engine::core::resolve_backend_device("CUDA1", BackendType::Cuda, dual_gpu);
        require_eq(second.device, 1, "exact CUDA1 disambiguates identical GPUs");
    }

    // Non-selectable registries (BLAS) must never match.
    {
        for (const char * query : {"BLAS", "Accelerate"}) {
            bool rejected = false;
            try {
                (void) engine::core::resolve_backend_device(query, std::nullopt, devices);
            } catch (const std::runtime_error & error) {
                rejected = std::string(error.what()).find("available:") != std::string::npos;
            }
            require(rejected, std::string("non-selectable registry must not match: ") + query);
        }
    }

    // Empty, whitespace-only, missing, cross-backend mismatch, and BestAvailable filter must fail cleanly.
    for (const char * empty_query : {"", "   \t\n "}) {
        bool rejected = false;
        try {
            (void) engine::core::resolve_backend_device(empty_query, std::nullopt, devices);
        } catch (const std::runtime_error & error) {
            rejected = std::string(error.what()).find("empty") != std::string::npos;
        }
        require(rejected, "empty/whitespace device name must be rejected");
    }
    {
        bool rejected = false;
        try {
            (void) engine::core::resolve_backend_device("CUDA0", BackendType::Vulkan, devices);
        } catch (const std::runtime_error & error) {
            const std::string msg = error.what();
            rejected = msg.find("no vulkan device matches") != std::string::npos &&
                msg.find("available:") != std::string::npos;
        }
        require(rejected, "backend filter mismatch must fail with available devices");
    }
    {
        bool rejected = false;
        try {
            (void) engine::core::resolve_backend_device("CPU", BackendType::BestAvailable, devices);
        } catch (const std::runtime_error & error) {
            rejected = std::string(error.what()).find("best") != std::string::npos;
        }
        require(rejected, "BestAvailable backend filter with device name must be rejected");
    }
}

void test_device_name_resolution_live_round_trip() {
    ensure_backends_loaded();
    const auto listed = engine::core::list_backend_devices(false);
    for (const auto & device : listed) {
        if (device.backend_option.empty() || device.device_name.empty()) {
            continue;
        }
        const auto expected_type = backend_type_for_option(device.backend_option);
        require(expected_type.has_value(), "known backend option for " + device.device_name);
        const auto resolved = engine::core::resolve_backend_device(device.device_name);
        require_eq(
            static_cast<int>(resolved.type),
            static_cast<int>(*expected_type),
            device.device_name + " live resolved backend");
        require_eq(resolved.device, device.index, device.device_name + " live resolved index");

        BackendConfig config;
        config.type = resolved.type;
        config.device = resolved.device;
        BackendHandle handle(engine::core::init_backend(config));
        ggml_backend_reg_t reg = ggml_backend_reg_by_name(device.backend.c_str());
        require(reg != nullptr, device.device_name + " registry");
        ggml_backend_dev_t expected_dev = ggml_backend_reg_dev_get(reg, static_cast<size_t>(device.index));
        require(ggml_backend_get_device(handle.handle) == expected_dev,
            device.device_name + " init_backend opened the resolved ggml device");
    }
}

void test_webgpu_backend_instances_share_a_stable_device() {
    auto reg = ggml_backend_reg_by_name("WebGPU");
    if (reg == nullptr || ggml_backend_reg_dev_count(reg) == 0) return;
    BackendConfig config;
    config.type = BackendType::WebGPU;
    BackendHandle first(engine::core::init_backend(config));
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(
        ggml_init({1024 * 1024, nullptr, true}), ggml_free);
    require(ctx != nullptr, "WebGPU lifetime graph context allocation");
    auto * input = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 4);
    auto * output = ggml_scale(ctx.get(), input, 2.0F);
    ggml_set_input(input);
    ggml_set_output(output);
    auto * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, output);
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> alloc(
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(first.handle)), ggml_gallocr_free);
    require(ggml_gallocr_alloc_graph(alloc.get(), graph), "WebGPU lifetime graph allocation");
    const std::vector<float> values{1.0F, 2.0F, 3.0F, 4.0F};
    BackendHandle second(nullptr);
    for (int phase = 0; phase < 3; ++phase) {
        if (phase == 1) second.handle = engine::core::init_backend(config);
        if (phase == 2) {
            ggml_backend_free(second.handle);
            second.handle = nullptr;
        }
        ggml_backend_tensor_set(input, values.data(), 0, values.size() * sizeof(float));
        require(ggml_backend_graph_compute(first.handle, graph) == GGML_STATUS_SUCCESS,
                "WebGPU original backend remains executable");
        std::vector<float> actual(values.size());
        ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
        for (size_t i = 0; i < actual.size(); ++i) {
            require_eq(actual[i], values[i] * 2.0F,
                       "WebGPU original buffers/pipeline, phase " + std::to_string(phase));
        }
    }
}

}  // namespace

int main() {
    engine::debug::configure_logging({true});
    const auto started = std::chrono::steady_clock::now();
    try {
        test_every_enumerated_device_resolves();
        test_unregistered_backends_are_rejected();
        test_host_and_best_backends_still_resolve();
        test_device_name_resolution_synthetic_cases();
        test_device_name_resolution_live_round_trip();
        test_backend_impls();
        test_text_listing_format_is_stable();
        test_json_listing_matches_selection();
        test_json_writer_exact_integers_and_nulls();
        test_webgpu_backend_instances_share_a_stable_device();
    } catch (const std::exception & error) {
        std::cerr << "backend_device_resolution_test failed: " << error.what() << '\n';
        return 1;
    }
    engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(started));
    std::cout << "backend_device_resolution_test passed\n";
    return 0;
}
