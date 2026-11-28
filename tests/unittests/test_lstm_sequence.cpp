#include "engine/framework/core/backend.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/core/module.h"
#include "engine/framework/modules/recurrent_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "test_assert.h"
#include "engine/framework/debug/profiler.h"
#include "ggml-alloc.h"

#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

namespace {
using namespace engine;

std::vector<float> check(int64_t frames, int64_t batch, bool reverse,
    core::BackendType backend = core::BackendType::Cpu, int64_t features = 3, int64_t width = 4, bool reuse = false) {
    core::ExecutionContext execution({backend, 0, 1});
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context(
        ggml_init({32 * 1024 * 1024, nullptr, true}), ggml_free);
    core::ModuleBuildContext ctx{context.get(), "lstm_sequence_test", backend};
    using Shape = core::TensorShape;
    std::vector<core::TensorValue> inputs;
    auto tensor = [&](Shape shape) {
        auto value = core::make_tensor(ctx, GGML_TYPE_F32, shape);
        inputs.push_back(value);
        ggml_set_input(value.tensor);
        return value;
    };
    const auto input = tensor(Shape::from_dims({frames, batch, features}));
    const auto hidden = tensor(Shape::from_dims({batch, width}));
    const auto cell = tensor(Shape::from_dims({batch, width}));
    const modules::LSTMSequenceWeights weights{{
        tensor(Shape::from_dims({4 * width, features})), tensor(Shape::from_dims({4 * width, width})),
        tensor(Shape::from_dims({4 * width})), tensor(Shape::from_dims({4 * width}))}};
    const auto batched = modules::LSTMSequenceModule({features, width, reverse, true}).build(ctx, input, hidden, cell, weights);
    const auto fusion_friendly = modules::LSTMSequenceModule({features, width, reverse, true, true})
        .build(ctx, input, hidden, cell, weights);
    const auto bidirectional = modules::BidirectionalLSTMModule({features, width, false, true}).build(
        ctx, input, hidden, cell, hidden, cell, {weights.cell, weights.cell});
    std::vector<modules::LSTMSequenceOutputs> single;
    auto * graph = ggml_new_graph_custom(context.get(), 8192, false);
    ggml_set_output(bidirectional.sequence.tensor);
    ggml_build_forward_expand(graph, bidirectional.sequence.tensor);
    for (const auto & value : {batched.sequence, batched.hidden, batched.cell}) {
        ggml_set_output(value.tensor);
        ggml_build_forward_expand(graph, value.tensor);
    }
    for (const auto & value : {fusion_friendly.sequence, fusion_friendly.hidden, fusion_friendly.cell}) {
        ggml_set_output(value.tensor);
        ggml_build_forward_expand(graph, value.tensor);
    }
    for (int64_t b = 0; b < batch; ++b) {
        auto x = modules::SliceModule({1, b, 1}).build(ctx, input);
        x = core::ensure_backend_addressable_layout(ctx, x);
        x = core::reshape_tensor(ctx, x, Shape::from_dims({frames, features}));
        single.push_back(modules::LSTMSequenceModule({features, width, reverse}).build(ctx, x,
            modules::SliceModule({0, b, 1}).build(ctx, hidden),
            modules::SliceModule({0, b, 1}).build(ctx, cell), weights));
        for (const auto & value : {single.back().sequence, single.back().hidden, single.back().cell}) {
            ggml_set_output(value.tensor);
            ggml_build_forward_expand(graph, value.tensor);
        }
    }
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
        reuse ? nullptr : ggml_backend_alloc_ctx_tensors(context.get(), execution.backend()), ggml_backend_buffer_free);
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> alloc(
        reuse ? ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend())) : nullptr, ggml_gallocr_free);
    if (reuse) test::require(alloc && ggml_gallocr_reserve(alloc.get(), graph) && ggml_gallocr_alloc_graph(alloc.get(), graph), "LSTM graph allocation failed");
    else test::require(buffer != nullptr, "LSTM allocation failed");
    for (size_t i = 0; i < inputs.size(); ++i) {
        std::vector<float> values(ggml_nelements(inputs[i].tensor));
        for (size_t j = 0; j < values.size(); ++j) values[j] = 0.2f * std::sin(static_cast<float>(j + 7 * i));
        core::write_tensor_f32(inputs[i], values);
    }
    test::require(core::compute_backend_graph(execution.backend(), graph) == GGML_STATUS_SUCCESS, "LSTM compute failed");
    const auto sequence = core::read_tensor_f32(batched.sequence.tensor);
    const auto both_directions = core::read_tensor_f32(bidirectional.sequence.tensor);
    const auto final_hidden = core::read_tensor_f32(batched.hidden.tensor);
    const auto final_cell = core::read_tensor_f32(batched.cell.tensor);
    const auto fusion_sequence = core::read_tensor_f32(fusion_friendly.sequence.tensor);
    const auto fusion_hidden = core::read_tensor_f32(fusion_friendly.hidden.tensor);
    const auto fusion_cell = core::read_tensor_f32(fusion_friendly.cell.tensor);
    for (size_t i = 0; i < sequence.size(); ++i)
        test::require_close(fusion_sequence[i], sequence[i], 0.0f, "fusion-friendly sequence");
    for (size_t i = 0; i < final_hidden.size(); ++i) {
        test::require_close(fusion_hidden[i], final_hidden[i], 0.0f, "fusion-friendly hidden");
        test::require_close(fusion_cell[i], final_cell[i], 0.0f, "fusion-friendly cell");
    }
    for (int64_t b = 0; b < batch; ++b) {
        const auto expected = core::read_tensor_f32(single[b].sequence.tensor);
        const auto expected_hidden = core::read_tensor_f32(single[b].hidden.tensor);
        const auto expected_cell = core::read_tensor_f32(single[b].cell.tensor);
        for (int64_t t = 0; t < frames; ++t)
            for (int64_t c = 0; c < width; ++c) {
                test::require_close(sequence[(t * batch + b) * width + c], expected[t * width + c], 1e-6f, "batched sequence");
                test::require_close(both_directions[(t * batch + b) * (2 * width) + (reverse ? width : 0) + c],
                                    expected[t * width + c], 1e-6f, "bidirectional sequence");
            }
        for (int64_t c = 0; c < width; ++c) {
            test::require_close(final_hidden[b * width + c], expected_hidden[c], 1e-6f, "batched hidden");
            test::require_close(final_cell[b * width + c], expected_cell[c], 1e-6f, "batched cell");
        }
    }
    core::release_backend_graph_resources(execution.backend(), graph, true);
    return sequence;
}
}  // namespace

int main(int argc, char ** argv) {
    try {
        engine::debug::configure_logging({true});
        const auto started = std::chrono::steady_clock::now();
        if (argc > 2 && std::string(argv[1]) == "--backend" && std::string(argv[2]) == "webgpu") {
            const auto expected = check(1, 1, false, core::BackendType::Cpu, 640, 640, true);
            const auto actual = check(1, 1, false, core::BackendType::WebGPU, 640, 640, true);
            for (size_t i = 0; i < expected.size(); ++i)
                test::require_close(actual[i], expected[i], 1e-5f, "WebGPU LSTM CPU reference");
            debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(started));
            std::cout << "WebGPU 640-wide LSTM matches CPU with graph buffer reuse\n";
            return 0;
        }
        for (const auto frames : {1, 5, 6})
            for (const auto batch : {1, 3})
                for (const bool reverse : {false, true}) check(frames, batch, reverse);
        debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(started));
        std::cout << "Batched LSTM matches independent sequences in all 12 cases\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
