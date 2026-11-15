#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/prepared_graph.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/modules/moe_feed_forward_module.h"

#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
namespace core = engine::core;
namespace modules = engine::modules;

void test_gated_defaults(core::ExecutionContext & execution, bool silu, bool fused, bool gate_first) {
    constexpr int h = 8, width = 16, tokens = 3;
    core::BackendWeightStore store(execution.backend(), execution.backend_type(), "test.ffn.defaults", 1024 * 1024);
    std::vector<float> x_data(tokens * h), packed_data(2 * width * h), down_data(h * width);
    for (size_t i = 0; i < x_data.size(); ++i) x_data[i] = .2F * std::sin(float(i));
    for (size_t i = 0; i < packed_data.size(); ++i) packed_data[i] = .1F * std::cos(float(i) * .3F);
    for (size_t i = 0; i < down_data.size(); ++i) down_data[i] = .15F * std::sin(float(i) * .2F);
    const auto x = store.make_f32(core::TensorShape::from_dims({tokens, h}), x_data);
    const auto packed = store.make_f32(core::TensorShape::from_dims({2 * width, h}), packed_data);
    const auto down = store.make_f32(core::TensorShape::from_dims({h, width}), down_data);
    store.upload();
    core::PreparedGraph graph(execution, {"test.ffn.defaults", 1024 * 1024, 256});
    auto & ctx = graph.build_context();
    auto * g = ctx.ggml;
    modules::GatedFeedForwardConfig config;
    config.hidden_size = h; config.intermediate_size = width;
    config.activation = silu ? modules::GatedFeedForwardActivation::Silu : modules::GatedFeedForwardActivation::Gelu;
    config.fused_swiglu = fused; config.packed_gate_first = gate_first;
    modules::GatedFeedForwardWeights weights;
    weights.up_gate_proj = {packed, std::nullopt}; weights.down_proj = {down, std::nullopt};
    const auto actual_packed = modules::GatedFeedForwardModule(config).build(ctx, x, weights);
    weights.up_gate_proj = {};
    weights.up_proj = {core::wrap_tensor(ggml_view_2d(g, packed.tensor, h, width, h * sizeof(float), gate_first ? h * width * sizeof(float) : 0),
        core::TensorShape::from_dims({width, h})), std::nullopt};
    weights.gate_proj = {core::wrap_tensor(ggml_view_2d(g, packed.tensor, h, width, h * sizeof(float), gate_first ? 0 : h * width * sizeof(float)),
        core::TensorShape::from_dims({width, h})), std::nullopt};
    const auto actual_split = modules::GatedFeedForwardModule(config).build(ctx, x, weights);
    auto * up = ggml_mul_mat(g, weights.up_proj.weight.tensor, x.tensor);
    auto * gate = ggml_mul_mat(g, weights.gate_proj.weight.tensor, x.tensor);
    gate = silu ? ggml_silu(g, gate) : ggml_gelu(g, gate);
    auto * expected = ggml_mul_mat(g, down.tensor, ggml_mul(g, gate, up));
    for (auto * result : {actual_packed.tensor, actual_split.tensor, expected}) {
        ggml_set_output(result); ggml_build_forward_expand(graph.graph(), result);
    }
    graph.allocate(); graph.compute();
    const auto reference = core::read_tensor_f32(expected);
    for (auto * result : {actual_packed.tensor, actual_split.tensor}) {
        const auto values = core::read_tensor_f32(result);
        for (size_t i = 0; i < values.size(); ++i) {
            if (!std::isfinite(values[i]) || std::abs(values[i] - reference[i]) > 1e-6F)
                throw std::runtime_error("Gated FFN differs from independent activation reference");
        }
    }
}

void test_composition(core::ExecutionContext & execution, int mode) {
    constexpr int h = 8, width = 16, experts = 4, tokens = 3;
    const int latent = mode == 2 ? 6 : h;
    core::BackendWeightStore store(execution.backend(), execution.backend_type(), "test.ffn", 1024 * 1024);
    int offset = 0;
    const auto tensor = [&](core::TensorShape shape) {
        std::vector<float> data(static_cast<size_t>(shape.prefix_elements() * shape.last_dim()));
        for (auto & value : data) value = .2F * std::sin(float(++offset) * .37F);
        return store.make_f32(shape, std::move(data));
    };
    auto x = tensor(core::TensorShape::from_dims({tokens, h}));
    auto router = tensor(core::TensorShape::from_dims({experts, h}));
    auto up = tensor(core::TensorShape::from_dims({experts, width, latent}));
    auto gate = tensor(core::TensorShape::from_dims({experts, width, latent}));
    auto down = tensor(core::TensorShape::from_dims({experts, latent, width}));
    auto shared_up = tensor(core::TensorShape::from_dims({2 * width, h}));
    auto shared_down = tensor(core::TensorShape::from_dims({h, width}));
    auto shared_gate = tensor(core::TensorShape::from_dims({1, h}));
    auto latent_down = tensor(core::TensorShape::from_dims({latent, h}));
    auto latent_up = tensor(core::TensorShape::from_dims({h, latent}));
    auto router_norm = tensor(core::TensorShape::from_dims({h}));
    modules::MoeFeedForwardConfig config;
    config.routing.experts = experts;
    config.routing.selected = 2;
    modules::MoeFeedForwardWeights weights;
    weights.router = router;
    weights.experts = {up.tensor, gate.tensor, down.tensor};
    if (mode == 0) {
        modules::GatedFeedForwardConfig shared;
        shared.hidden_size = h; shared.intermediate_size = width;
        shared.activation = modules::GatedFeedForwardActivation::Silu;
        shared.packed_gate_first = true;
        shared.round_projections = shared.round_activation = shared.round_product = shared.round_output = true;
        config.shared = shared;
        config.experts = {modules::MoeActivation::SwiGlu, false, true, true, true, true};
        config.round_routed_output = config.round_output = true;
        modules::GatedFeedForwardWeights w;
        w.up_gate_proj = {shared_up, std::nullopt}; w.down_proj = {shared_down, std::nullopt};
        weights.shared = w;
        weights.shared_gate = {shared_gate, std::nullopt};
    } else if (mode == 1) {
        modules::FeedForwardConfig shared{h, width, false};
        shared.activation = modules::FeedForwardActivation::ReluSquared;
        config.shared = shared;
        config.experts.activation = modules::MoeActivation::ReluSquared;
        weights.experts.gate = nullptr;
        modules::FeedForwardWeights w;
        w.fc2_weight = shared_down;
        weights.shared = w;
    } else {
        weights.latent_down = {latent_down, std::nullopt};
        weights.latent_up = {latent_up, std::nullopt};
        config.router_norm = modules::NormConfig{h, 1e-5F, true, false};
        weights.router_norm.weight = router_norm;
        config.routing.scoring = modules::MoeScoring::SelectedSoftmax;
        config.routing.normalization = modules::MoeNormalization::None;
    }
    store.upload();
    core::PreparedGraph graph(execution, {"test.ffn", 4 * 1024 * 1024, 1024});
    auto & ctx = graph.build_context();
    // Shared ReLU uses an independent view of the projection's first width rows.
    if (mode == 1) {
        auto & shared = std::get<modules::FeedForwardWeights>(weights.shared);
        shared.fc1_weight = core::wrap_tensor(ggml_view_2d(ctx.ggml, shared_up.tensor, h, width, shared_up.tensor->nb[1], 0),
            core::TensorShape::from_dims({width, h}));
    }
    const modules::MoeFeedForwardModule module(config);
    const auto prepared = module.build_inputs(ctx, x.tensor, weights);
    auto * actual = module.build_output(ctx, prepared, weights);
    auto * g = ctx.ggml;
    auto * expert_input = mode == 2 ? ggml_mul_mat(g, latent_down.tensor, x.tensor) : x.tensor;
    auto * routing_input = mode == 2 ? ggml_mul(g, ggml_rms_norm(g, x.tensor, 1e-5F), router_norm.tensor) : x.tensor;
    const auto routing = modules::MoeModule::build_routing(ctx, routing_input, router.tensor, nullptr, config.routing);
    auto * routed = modules::MoeModule(config.experts).build(ctx, expert_input, routing, weights.experts);
    if (config.round_routed_output) routed = ggml_round_bf16(g, routed);
    auto * expected = mode == 2 ? ggml_mul_mat(g, latent_up.tensor, routed) : routed;
    if (mode != 2) {
        auto * projected = ggml_mul_mat(g, shared_up.tensor, x.tensor);
        auto * first = ggml_cont(g, ggml_view_2d(g, projected, width, tokens, projected->nb[1], 0));
        ggml_tensor * activated;
        if (mode == 0) {
            auto * second = ggml_cont(g, ggml_view_2d(g, projected, width, tokens, projected->nb[1], width * sizeof(float)));
            first = ggml_round_bf16(g, first); second = ggml_round_bf16(g, second);
            activated = ggml_round_bf16(g, ggml_silu(g, first));
            activated = ggml_round_bf16(g, ggml_mul(g, activated, second));
        } else activated = ggml_sqr(g, ggml_relu(g, first));
        auto * shared = ggml_mul_mat(g, shared_down.tensor, activated);
        if (mode == 0) {
            shared = ggml_round_bf16(g, shared);
            shared = ggml_mul(g, shared, ggml_sigmoid(g, ggml_mul_mat(g, shared_gate.tensor, x.tensor)));
        }
        expected = ggml_add(g, expected, shared);
    }
    if (config.round_output) expected = ggml_round_bf16(g, expected);
    // The staged boundary must compose the same output without owning execution.
    auto * expert_rows = modules::MoeModule(config.experts).build_experts(ctx,
        ggml_reshape_3d(g, prepared.experts, latent, 1, tokens), prepared.routing.ids, weights.experts);
    auto * combined = modules::MoeModule::combine(ctx, expert_rows, prepared.routing.weights);
    auto * split = module.finish(ctx, combined, prepared.shared, weights);
    for (auto * result : {actual, expected, split}) { ggml_set_output(result); ggml_build_forward_expand(graph.graph(), result); }
    graph.allocate(); graph.compute();
    const auto reference = core::read_tensor_f32(expected);
    for (auto * result : {actual, split}) {
        const auto values = core::read_tensor_f32(result);
        for (size_t i = 0; i < values.size(); ++i) {
            if (!std::isfinite(values[i]) || std::abs(values[i] - reference[i]) > 1e-6F * (1 + std::abs(reference[i])))
                throw std::runtime_error("FFN composition mismatch: mode=" + std::to_string(mode));
        }
    }
}

void test_external_routing(core::ExecutionContext & execution) {
    constexpr int h = 8, width = 16, experts = 4, tokens = 3;
    core::BackendWeightStore store(execution.backend(), execution.backend_type(), "test.ffn.external", 1024 * 1024);
    std::vector<float> input(tokens * h), router(experts * h), up(experts * width * h), down(experts * h * width);
    for (size_t i = 0; i < input.size(); ++i) input[i] = .2F * std::sin(float(i));
    for (size_t i = 0; i < router.size(); ++i) router[i] = .3F * std::cos(float(i));
    for (size_t i = 0; i < up.size(); ++i) up[i] = .2F * std::sin(float(i) * .13F);
    for (size_t i = 0; i < down.size(); ++i) down[i] = .1F * std::cos(float(i) * .17F);
    auto reversed_up = up, reversed_down = down;
    for (int e = 0; e < experts; ++e) for (int j = 0; j < width * h; ++j) {
        reversed_up[e * width * h + j] = up[(experts - 1 - e) * width * h + j];
        reversed_down[e * width * h + j] = down[(experts - 1 - e) * width * h + j];
    }
    const auto x = store.make_f32(core::TensorShape::from_dims({tokens, h}), input);
    const auto r = store.make_f32(core::TensorShape::from_dims({experts, h}), router);
    const auto u = store.make_f32(core::TensorShape::from_dims({experts, width, h}), up);
    const auto d = store.make_f32(core::TensorShape::from_dims({experts, h, width}), down);
    const auto bank_u = store.make_f32(core::TensorShape::from_dims({experts, width, h}), reversed_up);
    const auto bank_d = store.make_f32(core::TensorShape::from_dims({experts, h, width}), reversed_down);
    store.upload();
    core::PreparedGraph graph(execution, {"test.ffn.external", 1024 * 1024, 512});
    auto & ctx = graph.build_context(); auto * g = ctx.ggml;
    modules::MoeRoutingConfig routing_config; routing_config.experts = experts; routing_config.selected = 2;
    auto routing = modules::MoeModule::build_routing(ctx, x.tensor, r.tensor, nullptr, routing_config);
    auto * bank_ids = ggml_cast(g, ggml_scale_bias(g, ggml_cast(g, routing.ids, GGML_TYPE_F32), -1.0F, experts - 1), GGML_TYPE_I32);
    modules::MoeFeedForwardWeights weights; weights.experts = {bank_u.tensor, bank_u.tensor, bank_d.tensor};
    // No router weights/config: external routing must bypass the internal router entirely.
    const modules::MoeFeedForwardModule module(modules::MoeFeedForwardConfig{});
    const auto inputs = module.build_inputs(ctx, x.tensor, weights, &routing);
    if (inputs.routing.ids != routing.ids || inputs.routing.weights != routing.weights || inputs.experts != x.tensor)
        throw std::runtime_error("External routing identity was not preserved");
    auto * actual = module.build_output(ctx, inputs, weights, bank_ids);
    auto * expected = modules::MoeModule().build(ctx, x.tensor, routing, {u.tensor, u.tensor, d.tensor});
    for (auto * t : {actual, expected}) { ggml_set_output(t); ggml_build_forward_expand(graph.graph(), t); }
    graph.allocate(); graph.compute();
    const auto a = core::read_tensor_f32(actual), b = core::read_tensor_f32(expected);
    for (size_t i = 0; i < a.size(); ++i)
        if (!std::isfinite(a[i]) || std::abs(a[i] - b[i]) > 1e-6F)
            throw std::runtime_error("External routing with remapped bank IDs changed output");
    bool rejected = false;
    try { (void) module.finish(ctx, actual, x.tensor, weights); }
    catch (const std::invalid_argument &) { rejected = true; }
    if (!rejected) throw std::runtime_error("Unexpected shared branch was silently accepted");
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        engine::debug::configure_logging({true, std::nullopt});
        const auto start = std::chrono::steady_clock::now();
        core::BackendConfig config; config.threads = 8;
        if (argc > 1 && std::string(argv[1]) == "cuda") config.type = core::BackendType::Cuda;
        if (argc > 1 && std::string(argv[1]) == "vulkan") config.type = core::BackendType::Vulkan;
        core::ExecutionContext execution(config);
        test_external_routing(execution);
        test_gated_defaults(execution, false, false, false);
        test_gated_defaults(execution, true, false, false);
        test_gated_defaults(execution, true, true, false);
        test_gated_defaults(execution, true, true, true);
        bool rejected = false;
        try {
            modules::GatedFeedForwardConfig invalid;
            invalid.hidden_size = 8; invalid.intermediate_size = 16; invalid.fused_swiglu = true;
            (void) modules::GatedFeedForwardModule(invalid);
        } catch (const std::invalid_argument &) { rejected = true; }
        if (!rejected) throw std::runtime_error("Fused GELU must be rejected");
        for (int mode = 0; mode < 3; ++mode) test_composition(execution, mode);
        engine::debug::timing_log_scalar("session.wall_ms", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
        std::cout << "MoE FFN composition tests passed\n";
    } catch (const std::exception & error) { std::cerr << error.what() << '\n'; return 1; }
}
