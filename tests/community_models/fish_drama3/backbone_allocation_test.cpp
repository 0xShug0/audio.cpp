#include <ggml-alloc.h>
#include <ggml-backend.h>
#include "engine/framework/io/safetensors.h"
#include "engine/framework/debug/profiler.h"

#include <cmath>
#include <iostream>
#include <set>
#include <stdexcept>
#include <random>

namespace {
bool fail_buffer = false;
bool fail_graph = false;
std::set<ggml_backend_buffer_t> live_buffers;

ggml_backend_buffer_t test_alloc_buffer(ggml_context * ctx, ggml_backend_t backend) {
    if (fail_buffer) return nullptr;
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buffer) live_buffers.insert(buffer);
    return buffer;
}

void test_free_buffer(ggml_backend_buffer_t buffer) {
    live_buffers.erase(buffer);
    ggml_backend_buffer_free(buffer);
}

bool test_alloc_graph(ggml_gallocr_t allocator, ggml_cgraph * graph) {
    return !fail_graph && ggml_gallocr_alloc_graph(allocator, graph);
}

bool test_alloc_split_graph(ggml_backend_sched_t scheduler, ggml_cgraph * graph) {
    return !fail_graph && ggml_backend_sched_alloc_graph(scheduler, graph);
}
}  // namespace

// Compile the model's actual constructor with allocation failures injected only
// into this test. This avoids production fault hooks or changes to GGML.
#define ggml_backend_alloc_ctx_tensors test_alloc_buffer
#define ggml_backend_buffer_free test_free_buffer
#define ggml_gallocr_alloc_graph test_alloc_graph
#define ggml_backend_sched_alloc_graph test_alloc_split_graph
#include "../../../src/community_models/fish_drama3/backbone.cpp"
#undef ggml_backend_alloc_ctx_tensors
#undef ggml_backend_buffer_free
#undef ggml_gallocr_alloc_graph
#undef ggml_backend_sched_alloc_graph

int main(int argc, char ** argv) {
    try {
        namespace drama = engine::models::fish_drama3;
        engine::core::BackendConfig backend;
        backend.threads = 1;
        engine::core::ExecutionContext execution(backend);
        std::unique_ptr<ggml_context, drama::GgmlContextDeleter> context(
            ggml_init({1024 * 1024, nullptr, false}));
        if (!context) throw std::runtime_error("test weight context allocation failed");
        auto * norm = ggml_new_tensor_1d(context.get(), GGML_TYPE_F32, 16);
        std::fill_n(static_cast<float *>(norm->data), 16, 0.0F);
        auto weights = std::make_shared<drama::Drama3BackboneWeights>();
        weights->final_norm.weight = engine::core::wrap_tensor(
            norm, engine::core::TensorShape::from_dims({16}), GGML_TYPE_F32);
        drama::Drama3BackboneDecoderConfig config;
        config.hidden_size = 16;
        config.layers = 0;
        config.linear_key_head_dim = config.linear_value_head_dim = 4;
        config.linear_num_key_heads = config.linear_num_value_heads = 1;

        int failures = 0;
        for (bool split : {false, true}) {
            weights->cpu_execution = split ? std::make_shared<engine::core::ExecutionContext>(backend) : nullptr;
            for (bool buffer_failure : {true, false}) {
                fail_buffer = buffer_failure;
                fail_graph = !buffer_failure;
                for (int attempt = 0; attempt < 3; ++attempt) {
                    bool rejected = false;
                    try {
                        drama::Drama3BackboneCachedDecodeGraph graph(execution, weights, config, 32, 8 * 1024 * 1024);
                    } catch (const std::runtime_error & error) {
                        if (std::string(error.what()) != "failed to allocate Drama 3 backbone cached decode graph") throw;
                        rejected = true;
                    }
                    if (!rejected) throw std::runtime_error("injected allocation failure was not rejected");
                    if (!live_buffers.empty()) throw std::runtime_error("failed construction leaked a persistent buffer");
                    ++failures;
                }
            }
            fail_buffer = fail_graph = false;
            {
                drama::Drama3BackboneCachedDecodeGraph graph(execution, weights, config, 32, 8 * 1024 * 1024);
                const auto hidden = graph.run_embedding_step(std::vector<float>(16, 0.5F));
                if (hidden.size() != 16 || !std::all_of(hidden.begin(), hidden.end(),
                        [](float value) { return std::isfinite(value); })) {
                    throw std::runtime_error("decode did not recover after allocation failures");
                }
            }
            if (!live_buffers.empty()) throw std::runtime_error("normal destruction leaked a persistent buffer");
        }
        std::cout << "rejected_allocations=" << failures << " live_buffers=0 recovery_cases=2\n";

        engine::debug::configure_logging({true, std::nullopt});
        const auto started = std::chrono::steady_clock::now();
        auto test_backend = backend;
        if (argc > 1 && std::string(argv[1]) == "cuda") test_backend.type = engine::core::BackendType::Cuda;
        if (argc > 1 && std::string(argv[1]) == "vulkan") test_backend.type = engine::core::BackendType::Vulkan;
        engine::core::ExecutionContext prefill_execution(test_backend);
        struct FixtureDirectory {
            std::filesystem::path path = std::filesystem::temp_directory_path() /
                ("drama3_prefill_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
            FixtureDirectory() { std::filesystem::create_directories(path); }
            ~FixtureDirectory() { std::error_code error; std::filesystem::remove_all(path, error); }
        } fixture;
        auto c = config;
        c.vocab_size = 16;
        c.hidden_size = c.intermediate_size = 256;
        c.layers = 2;
        c.experts = 12;
        c.experts_per_token = 2;
        c.expert_cache_size = 4;
        c.heads = 4;
        c.kv_heads = 2;
        c.head_dim = 64;
        c.full_attention_interval = 2;
        c.linear_key_head_dim = c.linear_value_head_dim = 128;
        c.linear_num_key_heads = 2;
        c.linear_num_value_heads = 4;
        std::vector<engine::io::SafeTensorWriteEntry> entries;
        std::mt19937 random(1234);
        std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
        auto add = [&](const std::string & name, std::vector<int64_t> shape, float scale = .025F, float offset = 0.0F) {
            size_t count = 1;
            for (auto dim : shape) count *= static_cast<size_t>(dim);
            std::vector<float> data(count);
            for (size_t i = 0; i < count; ++i) data[i] = offset + scale * distribution(random);
            std::vector<unsigned char> bytes(count * sizeof(float));
            std::memcpy(bytes.data(), data.data(), bytes.size());
            entries.push_back({"model_weights/" + name, "F32", std::move(shape), std::move(bytes)});
        };
        add(c.weight_prefix + ".embed_tokens.weight", {16, 256});
        add("lm_head.weight", {16, 256});
        add(c.weight_prefix + ".norm.weight", {256}, 0);
        for (int layer = 0; layer < 2; ++layer) {
            const auto p = c.weight_prefix + ".layers." + std::to_string(layer);
            add(p + ".input_layernorm.weight", {256}, 0);
            add(p + ".post_attention_layernorm.weight", {256}, 0);
            if (layer == 0) {
                const auto a = p + ".linear_attn";
                add(a + ".in_proj_qkv.weight", {1024, 256});
                add(a + ".in_proj_z.weight", {512, 256});
                add(a + ".in_proj_b.weight", {4, 256});
                add(a + ".in_proj_a.weight", {4, 256});
                add(a + ".conv1d.weight", {1024, 1, 4});
                add(a + ".dt_bias", {4});
                add(a + ".A_log", {4});
                add(a + ".norm.weight", {128}, 0, 1);
                add(a + ".out_proj.weight", {256, 512});
            } else {
                const auto a = p + ".self_attn";
                add(a + ".q_proj.weight", {512, 256});
                add(a + ".k_proj.weight", {128, 256});
                add(a + ".v_proj.weight", {128, 256});
                add(a + ".o_proj.weight", {256, 256});
                add(a + ".q_norm.weight", {64}, 0);
                add(a + ".k_norm.weight", {64}, 0);
            }
            for (const auto * projection : {"gate_proj", "up_proj", "down_proj"}) {
                add(p + ".mlp.shared_expert." + projection + ".weight", {256, 256});
                add(p + ".mlp.experts." + projection + ".weight", {12, 256, 256});
            }
            add(p + ".mlp.gate.weight", {12, 256});
            add(p + ".mlp.shared_expert_gate.weight", {1, 256});
        }
        const auto safe = fixture.path / "weights.safetensors";
        const auto gguf = fixture.path / "weights.gguf";
        engine::io::write_safetensors_file(safe, entries);
        engine::assets::convert_tensor_sources_to_gguf({{safe, ""}}, gguf, engine::assets::TensorStorageType::F32,
            true, false, {}, {}, std::nullopt);
        auto source = engine::assets::open_tensor_source(gguf, "model_weights/");
        auto resident = drama::load_qwen35_weights(*source, c, prefill_execution, 4 * 1024 * 1024, engine::assets::TensorStorageType::Native);
        {
            drama::LmHeadGraph full(prefill_execution, resident, c, 4 * 1024 * 1024);
            auto selected_config = c;
            selected_config.lm_head_token_ids = {9, 2, 3, 4, 15};
            drama::LmHeadGraph selected(prefill_execution, resident, selected_config, 4 * 1024 * 1024);
            for (float value : {.2F, -.3F, .2F}) {
                const std::vector<float> hidden(256, value);
                const auto expected = full.run(hidden);
                const auto actual = selected.run(hidden);
                if (actual.size() != selected_config.lm_head_token_ids.size())
                    throw std::runtime_error("selected LM head candidate count mismatch");
                for (size_t i = 0; i < actual.size(); ++i)
                    if (std::abs(actual[i] - expected.at(selected_config.lm_head_token_ids[i])) > 1e-5F)
                        throw std::runtime_error("selected LM head changed candidate logits");
                selected.release_graph();
            }
        }
        c.disk_experts = true;
        auto disk = drama::load_qwen35_weights(*source, c, prefill_execution, 4 * 1024 * 1024, engine::assets::TensorStorageType::Native);
        for (int64_t steps : {1, 7}) {
            std::unique_ptr<ggml_context, drama::GgmlContextDeleter> routing_context(
                ggml_init({1024 * 1024, nullptr, true}));
            engine::core::ModuleBuildContext routing_ctx{routing_context.get(), "routing_layout", test_backend.type};
            auto * input = ggml_new_tensor_3d(routing_ctx.ggml, GGML_TYPE_F32, c.hidden_size, steps, 1);
            const auto routed = drama::build_moe_inputs(routing_ctx,
                engine::core::wrap_tensor(input, engine::core::TensorShape::from_dims({1, steps, c.hidden_size})),
                disk->layers.front(), c);
            if (!ggml_is_contiguous(routed.ids))
                throw std::runtime_error("Drama 3 offload routing IDs must remain compact");
            if (steps == 1 && routed.ids->op == GGML_OP_CONT)
                throw std::runtime_error("single-token routing must not copy already compact IDs");
        }
        drama::Drama3BackboneCachedDecodeGraph reference(prefill_execution, resident, c, 1024, 8 * 1024 * 1024);
        drama::Drama3BackboneCachedDecodeGraph grouped(prefill_execution, disk, c, 1024, 8 * 1024 * 1024);
        std::vector<float> first_hidden, first_next;
        for (int steps : {7, 129, 513, 7}) {
            std::vector<float> input(steps * 256);
            for (size_t i = 0; i < input.size(); ++i) input[i] = std::cos(float(i) * .091F);
            reference.reset();
            const auto reference_hidden = reference.prefill_embeddings(input, steps).hidden;
            grouped.reset(true);
            const auto actual = grouped.prefill_embeddings(input, steps);
            float maximum = 0;
            double square = 0, energy = 0;
            for (size_t i = 0; i < actual.hidden.size(); ++i) {
                maximum = std::max(maximum, std::abs(actual.hidden[i] - reference_hidden[i]));
                square += std::pow(actual.hidden[i] - reference_hidden[i], 2);
                energy += reference_hidden[i] * reference_hidden[i];
            }
            std::cout << "batched_steps=" << steps << " max_abs=" << maximum
                      << " relative_rmse=" << std::sqrt(square / energy) << '\n';
            if (!std::all_of(actual.hidden.begin(), actual.hidden.end(), [](float value) { return std::isfinite(value); }) ||
                maximum > 1e-3F || square > 1e-8 * energy)
                throw std::runtime_error("grouped and resident batched prefill mismatch");
            const std::vector<float> next(256, .3F);
            const auto valid_steps = grouped.valid_steps();
            grouped.release_prefill_graphs();
            if (grouped.valid_steps() != valid_steps)
                throw std::runtime_error("releasing prefill graphs reset continuation state");
            const auto expected_next = reference.run_embedding_step(next);
            const auto actual_next = grouped.run_embedding_step(next);
            maximum = 0;
            for (size_t i = 0; i < next.size(); ++i) maximum = std::max(maximum, std::abs(expected_next[i] - actual_next[i]));
            std::cout << "continuation_steps=" << steps << " max_abs=" << maximum << '\n';
            if (maximum > 1e-4F) throw std::runtime_error("grouped prefill corrupted continuation state");
            if (steps == 7) {
                if (first_hidden.empty()) {
                    first_hidden = actual.hidden;
                    first_next = actual_next;
                } else if (first_hidden != actual.hidden || first_next != actual_next) {
                    throw std::runtime_error("offload prefill retained state after prompt length changed");
                }
            }
        }
        engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(started));
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
