#include "engine/framework/modules/text_encoders/t5_encoder_runtime.h"
#include "engine/framework/core/prepared_graph.h"
#include "engine/framework/debug/profiler.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace engine;

namespace {
constexpr int64_t width = 32;
constexpr int64_t vocab = 8;
constexpr float epsilon = 1.e-6F;

float embedding(int64_t token, int64_t channel) {
    return static_cast<float>((token + 1) * (channel % 7 - 3)) * .03F;
}

// Zero attention/FFN projections make each residual block the identity.
// Expected values below use scalar embedding lookup, RMSNorm and projection;
// no encoder module or alternate runtime supplies the reference.
modules::T5EncoderRuntimeWeights load_weights(core::ExecutionContext & execution, bool gemma, bool residual_only = true) {
    auto bank = std::make_shared<core::PreparedGraph>(execution,
        core::PreparedGraphConfig{"test.t5.weights", 1024 * 1024, 128, 1024 * 1024});
    auto & ctx = bank->persistent_build_context();
    std::vector<std::pair<core::TensorValue, float>> tensors;
    auto tensor = [&](std::initializer_list<int64_t> shape, float fill = 0.F) {
        auto value = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims(shape));
        tensors.emplace_back(value, fill);
        return value;
    };
    auto linear = [&](int64_t out, int64_t in) {
        return modules::LinearWeights{tensor({out, in}, residual_only ? 0.F : .01F), std::nullopt};
    };
    auto embed = tensor({vocab, width});
    modules::T5EncoderRuntimeWeights result;
    result.storage = bank;
    if (gemma) {
        modules::T5GemmaEncoderWeights weights;
        weights.embed_tokens = embed;
        weights.norm = tensor({width});
        modules::T5GemmaEncoderLayerWeights layer;
        layer.pre_self_attn_norm = tensor({width});
        layer.post_self_attn_norm = tensor({width});
        layer.pre_ff_norm = tensor({width});
        layer.post_ff_norm = tensor({width});
        layer.q_proj = linear(width, width);
        layer.k_proj = linear(width, width);
        layer.v_proj = linear(width, width);
        layer.o_proj = linear(width, width);
        layer.gate_proj = linear(64, width);
        layer.up_proj = linear(64, width);
        layer.down_proj = linear(width, 64);
        weights.layers.push_back(layer);
        result.encoder = weights;
        result.projection = modules::LinearWeights{tensor({4, width}, .02F), std::nullopt};
    } else {
        modules::T5BaseEncoderWeights weights;
        weights.embed_tokens = embed;
        weights.relative_attention_bias = tensor({32, 2});
        weights.final_layer_norm = tensor({width}, 1.F);
        modules::T5BaseEncoderLayerWeights layer;
        layer.self_attention_layer_norm = tensor({width}, 1.F);
        layer.ffn_layer_norm = tensor({width}, 1.F);
        layer.q_proj = linear(width, width);
        layer.k_proj = linear(width, width);
        layer.v_proj = linear(width, width);
        layer.o_proj = linear(width, width);
        layer.wi_proj = linear(64, width);
        layer.wo_proj = linear(width, 64);
        weights.layers.push_back(layer);
        result.encoder = weights;
    }
    bank->allocate_persistent();
    for (const auto & item : tensors) {
        core::write_tensor_f32(item.first, std::vector<float>(ggml_nelements(item.first.tensor), item.second));
    }
    std::vector<float> values;
    for (int64_t t = 0; t < vocab; ++t)
        for (int64_t c = 0; c < width; ++c) values.push_back(embedding(t, c) + (residual_only ? 0.F :
                .02F * static_cast<float>((t * 3 + c * 2) % 5 - 2)));
    core::write_tensor_f32(embed, values);
    if (!gemma && !residual_only) {
        std::vector<float> bias(64);
        for (size_t i = 0; i < bias.size(); ++i) bias[i] = static_cast<float>(i % 9) * .07F;
        core::write_tensor_f32(std::get<modules::T5BaseEncoderWeights>(result.encoder).relative_attention_bias, bias);
    }
    return result;
}

void check(const std::vector<float> & actual, const std::vector<int32_t> & ids, bool gemma) {
    std::vector<float> expected;
    for (auto token : ids) {
        float sum_squares = 0.F;
        const float scale = gemma ? std::sqrt(static_cast<float>(width)) : 1.F;
        for (int64_t c = 0; c < width; ++c) {
            const float x = embedding(token, c) * scale;
            sum_squares += x * x;
        }
        const float inverse = 1.F / std::sqrt(sum_squares / width + epsilon);
        float projected = 0.F;
        for (int64_t c = 0; c < width; ++c) {
            const float value = embedding(token, c) * scale * inverse;
            if (gemma) projected += value * .02F;
            else expected.push_back(value);
        }
        if (gemma) expected.insert(expected.end(), 4, projected);
    }
    if (actual.size() != expected.size()) throw std::runtime_error("T5 output size mismatch");
    for (size_t i = 0; i < actual.size(); ++i)
        if (!std::isfinite(actual[i]) || std::abs(actual[i] - expected[i]) > 2.e-5F)
            throw std::runtime_error("T5 output differs from independent scalar reference");
}

modules::T5EncoderArchitecture make_architecture(bool gemma) {
    modules::T5EncoderArchitecture architecture;
    if (gemma) {
        modules::T5GemmaEncoderConfig config;
        config.hidden_size = width; config.layers = 1; config.attention_heads = 2;
        config.kv_heads = 2; config.head_dim = 16; config.intermediate_size = 64; config.vocab_size = vocab;
        config.rms_norm_eps = epsilon;
        architecture = config;
    } else {
        modules::T5BaseEncoderConfig config;
        config.hidden_size = width; config.layers = 1; config.attention_heads = 2;
        config.head_dim = 16; config.intermediate_size = 64; config.vocab_size = vocab;
        config.rms_norm_eps = epsilon;
        architecture = config;
    }
    return architecture;
}

void run(core::ExecutionContext & execution, bool gemma, bool retain_weights, bool retain_graphs) {
    auto architecture = make_architecture(gemma);
    int loads = 0;
    std::weak_ptr<const void> storage;
    modules::T5EncoderRuntimeConfig config;
    config.graph_nodes = 4096;
    config.retain_graphs = retain_graphs;
    config.retain_weights = retain_weights;
    if (gemma) config.projection = modules::LinearConfig{width, 4, false};
    {
        modules::T5EncoderRuntime runtime(execution, architecture, [&]() {
            ++loads;
            auto weights = load_weights(execution, gemma);
            storage = weights.storage;
            return weights;
        }, config);
        const std::vector<std::vector<int32_t>> inputs{{1}, {1,2,3}, {1}, {2,3,4,5,6}, {1}};
        for (const auto & ids : inputs) check(runtime.encode(ids), ids, gemma);
        if (loads != (retain_weights ? 1 : static_cast<int>(inputs.size())))
            throw std::runtime_error("Unexpected T5 weight reload count");
        runtime.release_graphs();
        check(runtime.encode(inputs[0]), inputs[0], gemma);
        runtime.release_weights();
        if (!storage.expired()) throw std::runtime_error("Released T5 weights still owned");
        check(runtime.encode(inputs[0]), inputs[0], gemma);
    }
    if (!storage.expired()) throw std::runtime_error("T5 destructor retained weight storage");
}

void check_nonzero_reuse(core::ExecutionContext & execution, bool gemma) {
    auto architecture = make_architecture(gemma);
    modules::T5EncoderRuntimeConfig config;
    config.graph_nodes = 4096;
    config.label = gemma ? "test.t5gemma.nonzero" : "test.t5base.nonzero";
    config.sequence_padding_multiple = 4;
    if (gemma) config.projection = modules::LinearConfig{width, 4, false};
    auto loader = [&]() { return load_weights(execution, gemma, false); };
    modules::T5EncoderRuntime reused(execution, architecture, loader, config);
    // Same padded shape with changing valid lengths exercises both mask refresh
    // and write-once relative buckets; then replace shape and return to it.
    for (const auto & ids : std::vector<std::vector<int32_t>>{{1,2,3}, {4,2}, {1,2,3}, {1,2,3,4,5}, {1,2,3}}) {
        modules::T5EncoderRuntime fresh(execution, architecture, loader, config);
        const auto expected = fresh.encode(ids);
        const auto actual = reused.encode(ids);
        if (expected.size() != actual.size()) throw std::runtime_error("T5 reuse output shape changed");
        for (size_t i = 0; i < actual.size(); ++i)
            if (!std::isfinite(actual[i]) || std::abs(expected[i] - actual[i]) > 1.e-6F)
                throw std::runtime_error("T5 cached graph differs from identical fresh execution");
    }
    reused.release_graphs();
    modules::T5EncoderRuntime fresh(execution, architecture, loader, config);
    if (reused.encode({1,2,3}) != fresh.encode({1,2,3}))
        throw std::runtime_error("T5 graph release/reprepare changed output");
    const std::vector<std::vector<int32_t>> rows{{1,2,3,4}, {5,6,7,1}, {2,1,4,6}};
    const std::vector<int64_t> lengths{3,1,0};
    const auto before = reused.encode_batch(rows, lengths);
    for (size_t row = 0; row < rows.size(); ++row) {
        modules::T5EncoderRuntime single(execution, architecture, loader, config);
        const auto expected = single.encode_batch({rows[row]}, {lengths[row]}).front();
        if (before[row].size() != expected.size()) throw std::runtime_error("T5 batch row size changed");
        float max_error = 0.F;
        for (size_t i = 0; i < expected.size(); ++i) {
            if (!std::isfinite(before[row][i])) throw std::runtime_error("T5 batch produced non-finite output");
            max_error = (std::max)(max_error, std::abs(before[row][i] - expected[i]));
        }
        debug::timing_log_scalar(config.label + ".batch_row_max_abs", max_error);
        // Different batch shapes can select different backend matmul tilings.
        // Keep same-shape reuse/mask invariance checks below at 1e-6.
        if (max_error > 2.e-4F)
            throw std::runtime_error(config.label + " batch differs from independent row execution: " + std::to_string(max_error));
    }
    auto changed = rows;
    changed[0][3] = 7;
    changed[1][1] = 3;
    changed[2] = {7,6,5,4};
    const auto masked = reused.encode_batch(changed, lengths);
    for (size_t row = 0; row < rows.size(); ++row)
        for (size_t i = 0; i < before[row].size(); ++i)
            if (!std::isfinite(masked[row][i]) || std::abs(before[row][i] - masked[row][i]) > 1.e-6F)
                throw std::runtime_error("T5 masked suffix affected valid output");
    changed = rows;
    changed[0][1] = 7;
    const auto visible = reused.encode_batch(changed, lengths);
    bool sensitive = false;
    const size_t output_width = gemma ? 4 : width;
    for (size_t i = 0; i < output_width; ++i)
        sensitive |= std::abs(visible[0][i] - before[0][i]) > 1.e-7F;
    if (!sensitive) throw std::runtime_error("T5 mask fixture does not exercise key visibility");

    config.sequence_padding_multiple = 1;
    modules::T5EncoderRuntime singleton(execution, architecture, loader, config);
    for (int pass = 0; pass < 2; ++pass) {
        const auto actual = singleton.encode({1});
        singleton.encode({1,2}); // Same physical shape, now both keys are valid.
        const auto expected = singleton.encode_batch({{1,7}}, {1}).front();
        if (actual.size() != output_width || expected.size() != actual.size())
            throw std::runtime_error("T5 singleton returned padding rows");
        for (size_t i = 0; i < actual.size(); ++i)
            if (!std::isfinite(actual[i]) || std::abs(actual[i] - expected[i]) > 1.e-6F)
                throw std::runtime_error("T5 singleton padding leaked into valid output");
        singleton.encode({1,2,3}); // Replace the graph before returning to one token.
        const auto returned = singleton.encode({1});
        for (size_t i = 0; i < actual.size(); ++i)
            if (!std::isfinite(returned[i]) || std::abs(returned[i] - actual[i]) > 1.e-6F)
                throw std::runtime_error("T5 singleton changed after graph replacement");
        singleton.release_weights();
    }
}
void check_batch(core::ExecutionContext & execution, bool gemma, bool release) {
    auto architecture = make_architecture(gemma);
    modules::T5EncoderRuntimeConfig config;
    config.graph_nodes = 4096;
    config.prepare_host_graph_plan = true;
    config.retain_graphs = !release;
    config.retain_weights = !release;
    if (gemma) config.projection = modules::LinearConfig{width, 4, false};
    int loads = 0;
    std::weak_ptr<const void> storage;
    modules::T5EncoderRuntime runtime(execution, architecture, [&]() {
        ++loads;
        auto weights = load_weights(execution, gemma);
        storage = weights.storage;
        return weights;
    }, config);
    const std::vector<std::vector<std::vector<int32_t>>> batches{
        {{1,2,3,4},{5,6}}, {{4,3}}, {{1,2,3,4},{5,6}}, {{1}}, {{}, {5,6}}};
    for (const auto & batch : batches) {
        std::vector<int64_t> valid;
        for (const auto & row : batch) valid.push_back(row.empty() ? 0 : static_cast<int64_t>(row.size()) - 1);
        const auto output = runtime.encode_batch(batch, valid);
        if (output.size() != batch.size()) throw std::runtime_error("T5 batch output count mismatch");
        for (size_t i = 0; i < batch.size(); ++i)
            check(output[i], std::vector<int32_t>(batch[i].begin(), batch[i].begin() + valid[i]), gemma);
        if (release && !storage.expired()) throw std::runtime_error("T5 batch retained released weights");
    }
    if (loads != (release ? static_cast<int>(batches.size()) : 1)) throw std::runtime_error("T5 batch weight reload count mismatch");
    runtime.release_weights();
    if (!storage.expired()) throw std::runtime_error("T5 batch explicit release retained weights");
    check(runtime.encode_batch({{1,2,3}}).front(), {1,2,3}, gemma);
    const auto singletons = runtime.encode_batch({{1}, {2}});
    check(singletons[0], {1}, gemma);
    check(singletons[1], {2}, gemma);
    const auto empty = runtime.encode_batch({{}, {}});
    if (empty.size() != 2 || !empty[0].empty() || !empty[1].empty())
        throw std::runtime_error("T5 empty batch rows produced values");
    if (release && !storage.expired()) throw std::runtime_error("T5 empty batch retained released weights");
}
} // namespace

int main(int argc, char ** argv) {
    try {
        core::BackendConfig config;
        config.threads = 8;
        for (int i = 1; i < argc; ++i) {
            const std::string flag = argv[i];
            if (++i >= argc) throw std::runtime_error("Missing argument");
            const std::string value = argv[i];
            if (flag == "--log") debug::configure_logging({true, value});
            else if (flag == "--backend") {
                if (value == "cuda") config.type = core::BackendType::Cuda;
                else if (value == "vulkan") config.type = core::BackendType::Vulkan;
                else if (value != "cpu") throw std::runtime_error("Unknown backend");
            } else throw std::runtime_error("Unknown argument");
        }
        const auto started = std::chrono::steady_clock::now();
        core::ExecutionContext execution(config);
        for (bool gemma : {false, true}) {
            run(execution, gemma, true, true);
            run(execution, gemma, true, false);
            run(execution, gemma, false, false);
            check_batch(execution, gemma, false);
            check_batch(execution, gemma, true);
        }
        check_nonzero_reuse(execution, false);
        check_nonzero_reuse(execution, true);
        debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(started));
        std::cout << "T5 scalar reference, A/B/A reuse, release/reload and ownership: passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
