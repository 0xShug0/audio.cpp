#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/disk_expert_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/io/safetensors.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace {
struct FixtureDirectory {
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("audiocpp_disk_experts_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    FixtureDirectory() { std::filesystem::create_directories(path); }
    ~FixtureDirectory() { std::error_code error; std::filesystem::remove_all(path, error); }
};
struct ContextDeleter { void operator()(ggml_context * ctx) const { ggml_free(ctx); } };
struct BufferDeleter { void operator()(ggml_backend_buffer_t buffer) const { ggml_backend_buffer_free(buffer); } };
struct SchedulerDeleter { void operator()(ggml_backend_sched_t scheduler) const { ggml_backend_sched_free(scheduler); } };
}

int main(int argc, char ** argv) {
    try {
        namespace core = engine::core;
        namespace assets = engine::assets;
        engine::debug::configure_logging({true, std::nullopt});
        const auto started = std::chrono::steady_clock::now();
        core::BackendConfig config;
        config.threads = 4;
        if (argc > 1 && std::string(argv[1]) == "cuda") config.type = core::BackendType::Cuda;
        if (argc > 1 && std::string(argv[1]) == "vulkan") config.type = core::BackendType::Vulkan;
        core::ExecutionContext execution(config);
        std::unique_ptr<core::ExecutionContext> host;
        if (config.type != core::BackendType::Cpu) {
            auto host_config = config;
            host_config.type = core::BackendType::Cpu;
            host = std::make_unique<core::ExecutionContext>(host_config);
        }
        FixtureDirectory directory;
        const auto input = directory.path / "weights.safetensors";
        const auto gguf = directory.path / "weights.gguf";
        const std::vector<std::string> names = {
            "model.layers.0.mlp.experts.gate.weight", "model.layers.0.mlp.experts.up.weight",
            "model.layers.0.mlp.experts.down.weight"};
        std::vector<engine::io::SafeTensorWriteEntry> entries;
        for (size_t p = 0; p < names.size(); ++p) {
            std::vector<float> values(12 * 256 * 256);
            for (size_t i = 0; i < values.size(); ++i) values[i] = std::sin(float(i) * .013F + float(p)) * .1F;
            std::vector<unsigned char> bytes(values.size() * sizeof(float));
            std::memcpy(bytes.data(), values.data(), bytes.size());
            entries.push_back({names[p], "F32", {12, 256, 256}, std::move(bytes)});
        }
        engine::io::write_safetensors_file(input, entries);
        for (const auto type : {assets::TensorStorageType::F32, assets::TensorStorageType::Q4_K}) {
            assets::GgufConversionOptions options;
            options.type_overrides.push_back({"model.layers.0.mlp.experts.*", type});
            assets::convert_tensor_sources_to_gguf({{input, ""}}, gguf, assets::TensorStorageType::F32,
                true, false, {}, {}, std::nullopt, options);
            core::DiskExpertWeightStore disk(gguf);
            const auto group = disk.add_group(names, execution.backend(), 4);
            if (disk.statistics().bytes_read != 0 || disk.statistics().expert_loads != 0)
                throw std::runtime_error("expert payloads were eagerly read");
            auto source = assets::open_tensor_source(gguf);
            const auto expected_type = type == assets::TensorStorageType::Q4_K ? "q4_k" : "f32";
            if (source->require_metadata(names[0]).dtype != expected_type)
                throw std::runtime_error("fixture expert dtype mismatch");
            core::BackendWeightStore resident(execution.backend(), execution.backend_type(), "test.resident", 1024 * 1024);
            std::vector<core::TensorValue> full;
            for (const auto & name : names) full.push_back(resident.load_tensor(*source, name,
                assets::TensorStorageType::Native, {12, 256, 256}));
            resident.upload();
            std::unique_ptr<ggml_context, ContextDeleter> inputs(ggml_init({1024 * 1024, nullptr, true}));
            std::unique_ptr<ggml_context, ContextDeleter> context(ggml_init({4 * 1024 * 1024, nullptr, true}));
            auto * ids = ggml_new_tensor_2d(inputs.get(), GGML_TYPE_I32, 2, 2);
            auto * x = ggml_new_tensor_3d(inputs.get(), GGML_TYPE_F32, 256, 1, 2);
            ggml_set_input(ids);
            ggml_set_input(x);
            std::unique_ptr<ggml_backend_buffer, BufferDeleter> input_buffer(
                ggml_backend_alloc_ctx_tensors(inputs.get(), execution.backend()));
            core::ModuleBuildContext ctx{context.get(), "disk_expert_test", execution.backend_type()};
            auto * local_ids = disk.route(ctx, ids, group);
            ggml_tensor * cached = nullptr;
            ggml_tensor * expected = nullptr;
            for (size_t p = 0; p < names.size(); ++p) {
                auto * a = ggml_mul_mat_id(context.get(), disk.tensor(group, p).tensor, x, local_ids);
                auto * b = ggml_mul_mat_id(context.get(), full[p].tensor, x, ids);
                cached = cached ? ggml_add(context.get(), cached, a) : a;
                expected = expected ? ggml_add(context.get(), expected, b) : b;
            }
            auto * graph = ggml_new_graph_custom(context.get(), 128, false);
            ggml_set_output(cached);
            ggml_set_output(expected);
            ggml_build_forward_expand(graph, cached);
            ggml_build_forward_expand(graph, expected);
            ggml_backend_t backends[] = {execution.backend(), host ? host->backend() : nullptr};
            std::unique_ptr<ggml_backend_sched, SchedulerDeleter> scheduler(
                ggml_backend_sched_new(backends, nullptr, host ? 2 : 1, 128, false, false));
            ggml_backend_sched_set_eval_callback(scheduler.get(), core::DiskExpertWeightStore::evaluate, &disk);
            if (!ggml_backend_sched_alloc_graph(scheduler.get(), graph)) throw std::runtime_error("test graph allocation failed");
            std::vector<float> values(512);
            for (size_t i = 0; i < values.size(); ++i) values[i] = std::cos(float(i) * .07F);
            ggml_backend_tensor_set(x, values.data(), 0, values.size() * sizeof(float));
            const auto cache_bytes = disk.statistics().buffer_bytes;
            size_t full_bytes = 0;
            for (const auto & weight : full) full_bytes += ggml_nbytes(weight.tensor);
            if (cache_bytes > full_bytes / 3 + 1024) throw std::runtime_error("expert cache allocation is not bounded by slots");
            int requests = 0;
            for (const auto & route : std::vector<std::vector<int32_t>>{
                     {0, 1, 2, 3}, {0, 1, 2, 3}, {4, 5, 4, 5}, {0, 1, 2, 3}, {0, 1, 2, 3}}) {
                const auto before = disk.statistics();
                ggml_backend_tensor_set(ids, route.data(), 0, route.size() * sizeof(int32_t));
                if (ggml_backend_sched_graph_compute(scheduler.get(), graph) != GGML_STATUS_SUCCESS)
                    throw std::runtime_error("expert graph compute failed");
                const auto actual = core::read_tensor_f32(cached);
                const auto reference = core::read_tensor_f32(expected);
                float maximum = 0;
                for (size_t i = 0; i < actual.size(); ++i) maximum = std::max(maximum, std::abs(actual[i] - reference[i]));
                if (!std::all_of(actual.begin(), actual.end(), [](float a) { return std::isfinite(a); }) || maximum > 1e-4F)
                    throw std::runtime_error("disk and resident expert output mismatch: " + std::to_string(maximum));
                const auto after = disk.statistics();
                if ((requests == 1 || requests == 4) && before.bytes_read != after.bytes_read)
                    throw std::runtime_error("cache hit read expert payloads again");
                if (after.buffer_bytes != cache_bytes) throw std::runtime_error("expert cache grew");
                if (core::read_tensor_i32(ids) != route) throw std::runtime_error("routing changed original expert IDs");
                std::cout << "request=" << requests++ << " max_abs=" << maximum << " expert_loads=" << after.expert_loads
                          << " bytes_read=" << after.bytes_read << " cache_bytes=" << after.buffer_bytes << '\n';
            }
            const std::vector<int32_t> invalid{12, 0, 1, 2};
            ggml_backend_tensor_set(ids, invalid.data(), 0, invalid.size() * sizeof(int32_t));
            bool rejected = false;
            try { ggml_backend_sched_graph_compute(scheduler.get(), graph); }
            catch (const std::runtime_error & e) { rejected = std::string(e.what()).find("routing ID out of range") != std::string::npos; }
            if (!rejected) throw std::runtime_error("invalid routing IDs were not rejected");
            source->release_storage();
            std::filesystem::resize_file(gguf, 0);
            const std::vector<int32_t> unread{8, 9, 8, 9};
            ggml_backend_tensor_set(ids, unread.data(), 0, unread.size() * sizeof(int32_t));
            rejected = false;
            try { ggml_backend_sched_graph_compute(scheduler.get(), graph); }
            catch (const std::runtime_error & e) { rejected = std::string(e.what()).find("failed to read selected expert") != std::string::npos; }
            if (!rejected) throw std::runtime_error("failed expert reads were not propagated");
            core::release_backend_graph_resources(execution.backend(), graph, true);
        }
        engine::debug::timing_log_scalar("session.wall_ms", std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count());
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
