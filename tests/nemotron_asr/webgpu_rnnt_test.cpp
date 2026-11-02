#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/core/module.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/recurrent_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "test_assert.h"
#include "ggml-alloc.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace engine;

// A fixed initial RNNT predictor/joint step, independent of autoregressive
// control flow. Real native weights exercise the recurrent and Q8 graph together.
std::vector<float> check_rnnt(const assets::TensorSource & source, core::BackendType backend) {
    constexpr int64_t width = 640, vocabulary = 13088;
    core::ExecutionContext execution({backend, 0, 8});
    core::BackendWeightStore store(execution.backend(), backend, "nemotron_asr_webgpu_test.rnnt.weights", 4 * 1024 * 1024);
    const auto storage = assets::TensorStorageType::Native;
    auto embedding = store.load_tensor(source, "decoder.embedding.weight", storage, {vocabulary, width});
    std::vector<modules::LSTMCellWeights> weights;
    for (int layer = 0; layer < 2; ++layer) {
        const auto suffix = std::to_string(layer);
        weights.push_back({
            store.load_tensor(source, "decoder.lstm.weight_ih_l" + suffix, storage, {4 * width, width}),
            store.load_tensor(source, "decoder.lstm.weight_hh_l" + suffix, storage, {4 * width, width}),
            store.load_f32_tensor(source, "decoder.lstm.bias_ih_l" + suffix, {4 * width}),
            store.load_f32_tensor(source, "decoder.lstm.bias_hh_l" + suffix, {4 * width})});
    }
    const modules::LinearWeights projection{
        store.load_tensor(source, "decoder.decoder_projector.weight", storage, {width, width}),
        store.load_f32_tensor(source, "decoder.decoder_projector.bias", {width})};
    const modules::LinearWeights head{
        store.load_tensor(source, "joint.head.weight", storage, {vocabulary, width}),
        store.load_f32_tensor(source, "joint.head.bias", {vocabulary})};
    store.upload();
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context(ggml_init({8 * 1024 * 1024, nullptr, true}), ggml_free);
    core::ModuleBuildContext ctx{context.get(), "nemotron_asr_webgpu_test.rnnt", backend};
    auto token = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({1}));
    ggml_set_input(token.tensor);
    auto hidden = modules::EmbeddingModule({vocabulary, width}).build(ctx, token, embedding);
    hidden = core::reshape_tensor(ctx, hidden, core::TensorShape::from_dims({1, width}));
    std::vector<core::TensorValue> initial;
    std::vector<core::TensorValue> outputs;
    for (const auto & layer : weights) {
        auto h = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, width}));
        auto c = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, width}));
        ggml_set_input(h.tensor); ggml_set_input(c.tensor);
        initial.push_back(h); initial.push_back(c);
        const auto state = modules::LSTMCellModule({width, width}).build(ctx, hidden, h, c, layer);
        hidden = state.hidden;
        outputs.push_back(state.hidden); outputs.push_back(state.cell);
    }
    auto projected = modules::LinearModule({width, width, true}).build(ctx, hidden, projection);
    auto encoder = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, width}));
    ggml_set_input(encoder.tensor);
    initial.push_back(encoder);
    auto joint = modules::ReluModule().build(ctx, modules::AddModule().build(ctx, encoder, projected));
    auto logits = modules::LinearModule({width, vocabulary, true}).build(ctx, joint, head);
    ggml_set_output(logits.tensor);
    outputs.push_back(projected);
    for (const auto & output : outputs) {
        ggml_set_output(output.tensor);
        if (backend == core::BackendType::WebGPU)
            for (auto * base = output.tensor->view_src; base; base = base->view_src) ggml_set_output(base);
    }
    auto * graph = ggml_new_graph_custom(context.get(), 8192, false);
    ggml_build_forward_expand(graph, logits.tensor);
    for (const auto & output : outputs) ggml_build_forward_expand(graph, output.tensor);
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> alloc(
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend())), ggml_gallocr_free);
    test::require(alloc && ggml_gallocr_reserve(alloc.get(), graph) && ggml_gallocr_alloc_graph(alloc.get(), graph), "RNNT allocation failed");
    core::write_tensor_i32(token, std::vector<int32_t>{static_cast<int32_t>(vocabulary - 1)});
    for (const auto & input : initial) core::write_tensor_f32(input, std::vector<float>(width, 0.0f));
    test::require(core::compute_backend_graph(execution.backend(), graph) == GGML_STATUS_SUCCESS, "RNNT step failed");
    auto values = core::read_tensor_f32(logits.tensor);
    debug::trace_log_f32("nemotron_asr_webgpu_test.rnnt.logits", {vocabulary}, values);
    const auto cache = core::read_tensor_f32(projected.tensor);
    debug::trace_log_f32("nemotron_asr_webgpu_test.rnnt.cache", {width}, cache);
    std::unique_ptr<ggml_context, decltype(&ggml_free)> joint_context(ggml_init({1024 * 1024, nullptr, true}), ggml_free);
    core::ModuleBuildContext joint_ctx{joint_context.get(), "nemotron_asr_webgpu_test.rnnt_joint", backend};
    auto enc_input = core::make_tensor(joint_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, width}));
    auto pred_input = core::make_tensor(joint_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, width}));
    ggml_set_input(enc_input.tensor); ggml_set_input(pred_input.tensor);
    auto joint_only = modules::ReluModule().build(joint_ctx, modules::AddModule().build(joint_ctx, enc_input, pred_input));
    auto joint_logits = modules::LinearModule({width, vocabulary, true}).build(joint_ctx, joint_only, head);
    ggml_set_output(joint_logits.tensor);
    auto * joint_graph = ggml_new_graph_custom(joint_context.get(), 1024, false);
    ggml_build_forward_expand(joint_graph, joint_logits.tensor);
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> joint_alloc(
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend())), ggml_gallocr_free);
    test::require(joint_alloc && ggml_gallocr_reserve(joint_alloc.get(), joint_graph) && ggml_gallocr_alloc_graph(joint_alloc.get(), joint_graph), "RNNT joint allocation failed");
    core::write_tensor_f32(enc_input, std::vector<float>(width, 0.0f));
    core::write_tensor_f32(pred_input, cache);
    test::require(core::compute_backend_graph(execution.backend(), joint_graph) == GGML_STATUS_SUCCESS, "RNNT joint failed");
    const auto joint_values = core::read_tensor_f32(joint_logits.tensor);
    debug::trace_log_f32("nemotron_asr_webgpu_test.rnnt.joint_logits", {vocabulary}, joint_values);
    for (size_t i = 0; i < values.size(); ++i)
        test::require_close(joint_values[i], values[i], 1e-3f, "RNNT cached joint logits");
    core::release_backend_graph_resources(execution.backend(), joint_graph, true);

    core::release_backend_graph_resources(execution.backend(), graph, true);
    return values;
}
}  // namespace

int main(int argc, char ** argv) {
    try {
        engine::debug::configure_logging({true});
        const auto started = std::chrono::steady_clock::now();
        if (argc < 3 || std::string(argv[1]) != "--model") {
            throw std::runtime_error("usage: nemotron_asr_webgpu_rnnt_test --model <Nemotron GGUF> --log");
        }
        const auto source = assets::open_tensor_source(argv[2]);
        const auto expected = check_rnnt(*source, core::BackendType::Cpu);
        const auto actual = check_rnnt(*source, core::BackendType::WebGPU);
        double error = 0, energy = 0;
        for (size_t i = 0; i < expected.size(); ++i) {
            test::require(std::isfinite(actual[i]) && std::isfinite(expected[i]), "RNNT nonfinite logits");
            error += std::pow(actual[i] - expected[i], 2);
            energy += std::pow(expected[i], 2);
        }
        debug::trace_log_scalar("nemotron_asr_webgpu_test.rnnt.nmse", error / energy);
        test::require(error / energy < 5e-4, "RNNT CPU reference mismatch");
        test::require(std::max_element(actual.begin(), actual.end()) - actual.begin() ==
            std::max_element(expected.begin(), expected.end()) - expected.begin(), "RNNT initial token mismatch");
        debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(started));
        std::cout << "Fixed RNNT step matches CPU reference\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
