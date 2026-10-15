#include <ggml-alloc.h>
#include <ggml-backend.h>

#include <cmath>
#include <iostream>
#include <set>
#include <stdexcept>

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

int main() {
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
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
