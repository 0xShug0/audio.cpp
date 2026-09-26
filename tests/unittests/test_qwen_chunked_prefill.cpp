#include "engine/framework/modules/transformers/qwen_causal_decode_runtime.h"
#include "engine/framework/debug/trace.h"

#include <cmath>
#include <algorithm>
#include <array>
#include <iostream>
#include <stdexcept>

namespace {
std::vector<float> pattern(size_t size, float scale) {
    std::vector<float> result(size);
    for (size_t i = 0; i < size; ++i) { result[i] = scale * std::sin(static_cast<float>(i) * .17f + .3f); }
    return result;
}
void close(const std::vector<float> & actual, const std::vector<float> & expected) {
    if (actual.size() != expected.size()) { throw std::runtime_error("logit size mismatch"); }
    float maximum = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i])) { throw std::runtime_error("nonfinite logits"); }
        maximum = std::max(maximum, std::abs(actual[i] - expected[i]));
    }
    engine::debug::trace_log_scalar("qwen_chunked_prefill_test.max_error", maximum);
    if (maximum > .003f) { throw std::runtime_error("chunked prefill logits differ from reference"); }
}
}

int main(int argc, char ** argv) {
    try {
        engine::core::BackendConfig backend;
        backend.threads = 8;
        engine::debug::LoggingConfig logging;
        for (int i = 1; i < argc; ++i) {
            const std::string arg(argv[i]);
            if (arg == "--backend" && i + 1 < argc) {
                const std::string value(argv[++i]);
                if (value == "cuda") { backend.type = engine::core::BackendType::Cuda; }
                else if (value == "cpu") { backend.type = engine::core::BackendType::Cpu; }
                else { throw std::runtime_error("unsupported test backend"); }
            } else if (arg == "--log" && i + 1 < argc) {
                logging.enabled = true;
                logging.file_path = argv[++i];
            } else { throw std::runtime_error("unknown test argument"); }
        }
        engine::debug::configure_logging(logging);
        engine::core::ExecutionContext execution(backend);
        auto * context = ggml_init({2 * 1024 * 1024, nullptr, true});
        if (!context) { throw std::runtime_error("weight context allocation failed"); }
        struct ContextGuard { ggml_context * p; ~ContextGuard() { ggml_free(p); } } context_guard{context};
        engine::core::ModuleBuildContext ctx{context, "qwen_chunked_prefill_test", backend.type};
        std::vector<engine::core::TensorValue> tensors;
        auto tensor = [&](std::initializer_list<int64_t> shape) {
            auto value = engine::core::make_tensor(ctx, GGML_TYPE_F32, engine::core::TensorShape::from_dims(shape));
            tensors.push_back(value);
            return value;
        };
        engine::modules::QwenCausalDecodeRuntimeWeights weights;
        weights.token_embedding = tensor({97, 64});
        weights.final_norm = {tensor({64}), std::nullopt};
        weights.lm_head = engine::modules::LinearWeights{weights.token_embedding, std::nullopt};
        for (int layer = 0; layer < 2; ++layer) {
            engine::modules::QwenDecoderLayerWeights w;
            w.input_norm = {tensor({64}), std::nullopt};
            w.post_norm = {tensor({64}), std::nullopt};
            w.q_norm = {tensor({64}), std::nullopt};
            w.k_norm = {tensor({64}), std::nullopt};
            w.self_attention.q_weight = tensor({128, 64});
            w.self_attention.k_weight = tensor({64, 64});
            w.self_attention.v_weight = tensor({64, 64});
            w.self_attention.out_weight = tensor({64, 128});
            w.mlp.gate_proj = {tensor({128, 64}), std::nullopt};
            w.mlp.up_proj = {tensor({128, 64}), std::nullopt};
            w.mlp.down_proj = {tensor({64, 128}), std::nullopt};
            weights.stack.layers.push_back(w);
        }
        std::array<std::array<engine::core::TensorValue, 6>, 3> import_tensors;
        const std::array<ggml_type, 3> import_types{GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16};
        for (size_t type = 0; type < import_types.size(); ++type) {
            for (auto & value : import_tensors[type]) {
                value = engine::core::make_tensor(ctx, import_types[type], engine::core::TensorShape::from_dims({1, 9, 1, 4}));
            }
        }
        auto * buffer = ggml_backend_alloc_ctx_tensors(context, execution.backend());
        if (!buffer) { throw std::runtime_error("weight buffer allocation failed"); }
        struct BufferGuard { ggml_backend_buffer_t p; ~BufferGuard() { ggml_backend_buffer_free(p); } } buffer_guard{buffer};
        for (const auto & value : tensors) {
            auto values = pattern(ggml_nelements(value.tensor), .04f);
            if (value.shape.rank == 1) { std::fill(values.begin(), values.end(), 1.f); }
            engine::core::write_tensor_f32(value, values);
        }
        for (auto & values : import_tensors) {
            engine::runtime::TransformerKVCacheOptions options;
            options.allow_f16_storage = true;
            options.allow_bf16_storage = true;
            engine::runtime::TransformerKVCache legacy(9, 4, {values[0]}, {values[1]}, options);
            engine::runtime::TransformerKVCache device_zero(9, 4, {values[2]}, {values[3]}, options);
            options.lazy_import_scratch = true;
            engine::runtime::TransformerKVCache lazy(9, 4, {values[4]}, {values[5]}, options);
            for (int steps : {0, 5, 2, 0, 9}) {
                engine::runtime::TransformerKVState state;
                state.current_end = steps;
                state.layers.resize(1);
                state.layers[0].valid_steps = steps;
                state.layers[0].key = pattern(steps * 4, .4f);
                state.layers[0].value = pattern(steps * 4, -.7f);
                legacy.import_state(state);
                lazy.import_state(state);
                if (lazy.current_end() != legacy.current_end() || lazy.valid_steps() != legacy.valid_steps()) {
                    throw std::runtime_error("lazy cache import changed positions");
                }
                for (size_t kind = 0; kind < 2; ++kind) {
                    const auto bytes = ggml_nbytes(values[kind].tensor);
                    std::vector<unsigned char> expected(bytes), actual(bytes);
                    ggml_backend_tensor_get(values[kind].tensor, expected.data(), 0, bytes);
                    ggml_backend_tensor_get(values[kind + 4].tensor, actual.data(), 0, bytes);
                    if (actual != expected) { throw std::runtime_error("lazy cache import changed values or tail"); }
                }
                device_zero.import_state(state);
                device_zero.clear_on_backend();
                for (size_t kind = 0; kind < 2; ++kind) {
                    const auto bytes = ggml_nbytes(values[kind].tensor);
                    std::vector<unsigned char> expected(bytes), actual(bytes);
                    if (device_zero.current_end() != 0 || device_zero.valid_steps() != 0) {
                        throw std::runtime_error("cache clear did not reset positions");
                    }
                    ggml_backend_tensor_get(values[kind + 2].tensor, actual.data(), 0, bytes);
                    if (actual != expected) { throw std::runtime_error("cache clear left nonzero bytes"); }
                }
            }
        }
        engine::modules::QwenCausalDecodeRuntimeConfig config;
        auto & stack = config.decoder.stack;
        stack.hidden_size = 64;
        stack.num_attention_heads = 2;
        stack.num_key_value_heads = 1;
        stack.head_dim = 64;
        stack.intermediate_size = 128;
        stack.layers = 2;
        stack.runtime.static_cache.update_mode = engine::modules::QwenDecoderStaticCacheUpdateMode::DirectSetRows;
        stack.runtime.static_cache.set_rows_mode = engine::modules::QwenDecoderStaticCacheSetRowsMode::BackendViewOptimized;
        stack.runtime.attention.prefill_mode = engine::modules::QwenDecoderAttentionMode::FlashGroupedViewKV;
        stack.runtime.attention.static_mode = engine::modules::QwenDecoderAttentionMode::FlashGroupedViewKV;
        config.decoder.logits_size = 97;
        config.decoder.static_cache_type = GGML_TYPE_F16;
        config.prefill_graph_arena_bytes = 4 * 1024 * 1024;
        config.decode_graph_arena_bytes = 4 * 1024 * 1024;
        config.evict_cuda_graph_cache_on_release = true;
        engine::modules::QwenCausalDecodeRuntime reference(execution, config, weights);
        engine::modules::QwenCausalDecodeRuntime saver(execution, config, weights);
        for (int steps : {11, 5, 1, 11, 28}) {
            const auto embeddings = pattern(steps * 64, .3f);
            auto expected = reference.prefill_embeddings(embeddings, steps);
            auto actual = saver.prefill_embeddings_into_cache(embeddings, steps, 32, 4);
            close(actual.logits, expected.logits);
            reference.start_decode_tokens(expected.state, 32);
            for (int token = 0; token < 4; ++token) {
                const auto reference_logits = reference.decode_token(token + 7).logits;
                close(saver.decode_token(token + 7).logits, reference_logits);
            }
            if (saver.decode_current_end() != steps + 4) { throw std::runtime_error("cache position mismatch"); }
        }
        bool rejected = false;
        try { saver.decode_token(7); } catch (const std::runtime_error &) { rejected = true; }
        if (!rejected) { throw std::runtime_error("context overflow accepted"); }
        const auto grown_embeddings = pattern(41 * 64, .3f);
        auto grown_reference = reference.prefill_embeddings(grown_embeddings, 41);
        close(saver.prefill_embeddings_into_cache(grown_embeddings, 41, 64, 8).logits,
              grown_reference.logits);
        reference.start_decode_tokens(grown_reference.state, 64);
        close(saver.decode_token(7).logits, reference.decode_token(7).logits);
        // Kept prefix: a prompt sharing its first 24 rows with the previous
        // one recomputes only the remainder, including after decode steps
        // appended rows past the prefix, and across a block-size change.
        for (const auto & [steps, keep, chunk] : std::array<std::array<int, 3>, 4>{{{45, 24, 8}, {50, 24, 8}, {48, 47, 16}, {49, 40, 5}}}) {
            auto kept_embeddings = pattern(steps * 64, .3f);
            std::fill(kept_embeddings.begin() + 24 * 64, kept_embeddings.end(), 0.f);
            const auto tail = pattern((steps - 24) * 64, -.2f + .01f * steps);
            std::copy(tail.begin(), tail.end(), kept_embeddings.begin() + 24 * 64);
            auto kept_reference = reference.prefill_embeddings(kept_embeddings, steps);
            if (saver.retainable_prefix_steps(steps, 64, chunk, keep) != std::min(keep, steps - 1)) {
                throw std::runtime_error("resident prefix not retainable");
            }
            close(saver.prefill_embeddings_into_cache(kept_embeddings, steps, 64, chunk, keep).logits,
                  kept_reference.logits);
            if (saver.decode_current_end() != steps) { throw std::runtime_error("kept prefix cache position mismatch"); }
            reference.start_decode_tokens(kept_reference.state, 64);
            for (int token = 0; token < 3; ++token) {
                close(saver.decode_token(token + 9).logits, reference.decode_token(token + 9).logits);
            }
        }
        // A prefix longer than the resident rows, or a capacity the current
        // graph cannot hold, falls back to a full prefill.
        if (saver.retainable_prefix_steps(60, 64, 8, 55) != 0) { throw std::runtime_error("unresident prefix retained"); }
        if (saver.retainable_prefix_steps(40, 128, 8, 20) != 0) { throw std::runtime_error("prefix retained across capacity growth"); }
        const auto regrown = pattern(40 * 64, .3f);
        auto regrown_reference = reference.prefill_embeddings(regrown, 40);
        close(saver.prefill_embeddings_into_cache(regrown, 40, 128, 8, 20).logits, regrown_reference.logits);
        saver.release_runtime_graphs();
        if (saver.decode_current_end() != 0) { throw std::runtime_error("reset failed"); }
        std::cout << "PASS chunk boundaries, repeated prefill, kept prefix, decode, reset, capacity, reference parity\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
