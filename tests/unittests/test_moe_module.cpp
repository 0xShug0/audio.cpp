#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/modules/moe_module.h"
#include "engine/framework/modules/feed_forward_modules.h"
#include <ggml-alloc.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace {
namespace core = engine::core;
namespace modules = engine::modules;

void routing_and_experts(core::ExecutionContext & execution) {
    constexpr int h = 8, tokens = 3, experts = 8, width = 16, selected = 2;
    for (const auto scoring : {modules::MoeScoring::Softmax, modules::MoeScoring::Sigmoid}) {
        for (const auto normalization : {modules::MoeNormalization::Divide, modules::MoeNormalization::Clamp,
                 modules::MoeNormalization::AddEpsilon}) {
            core::BackendWeightStore store(execution.backend(), execution.backend_type(), "test.moe", 1024 * 1024);
            std::vector<float> input(h * tokens), router(h * experts), bias(experts), up(16 * width * experts), down(32 * h * experts);
            for (size_t i = 0; i < input.size(); ++i) input[i] = .2F * std::cos(float(i) * .37F);
            for (size_t i = 0; i < router.size(); ++i) router[i] = .3F * std::sin(float(i) * .43F);
            for (int e = 0; e < experts; ++e) bias[e] = float(e % 3) * .04F;
            for (size_t i = 0; i < up.size(); ++i) up[i] = .1F * std::sin(float(i) * .23F);
            for (size_t i = 0; i < down.size(); ++i) down[i] = .1F * std::cos(float(i) * .13F);
            auto * x = store.make_f32(core::TensorShape::from_dims({tokens, h}), input).tensor;
            auto * r = store.make_f32(core::TensorShape::from_dims({experts, h}), router).tensor;
            auto * b = store.make_f32(core::TensorShape::from_dims({experts}), bias).tensor;
            auto * u = store.make_f32(core::TensorShape::from_dims({experts, width, 16}), up).tensor;
            auto * d = store.make_f32(core::TensorShape::from_dims({experts, h, 32}), down).tensor;
            store.upload();
            std::unique_ptr<ggml_context, decltype(&ggml_free)> context(ggml_init({4 * 1024 * 1024, nullptr, true}), ggml_free);
            core::ModuleBuildContext ctx{context.get(), "test.moe", execution.backend_type()};
            modules::MoeRoutingConfig config;
            config.experts = experts;
            config.selected = selected;
            config.scoring = scoring;
            config.normalization = normalization;
            config.contiguous_ids = normalization == modules::MoeNormalization::Clamp;
            config.denominator_epsilon = normalization == modules::MoeNormalization::Clamp ? 2.0F : .1F;
            config.scale = 1.7F;
            config.precision = GGML_PREC_F32;
            // Exercise top-two group scoring as well as ordinary routing.
            if (normalization == modules::MoeNormalization::AddEpsilon) {
                config.groups = 2;
                config.selected_groups = 1;
            }
            auto routing = modules::MoeModule::build_routing(ctx, x, r, b, config);
            if (config.contiguous_ids && !ggml_is_contiguous(routing.ids))
                throw std::runtime_error("MoE selected IDs were not materialized");
            auto * graph = ggml_new_graph_custom(ctx.ggml, 256, false);
            auto * ids_output = ggml_cont(ctx.ggml, routing.ids);
            ggml_set_output(ids_output);
            ggml_set_output(routing.weights);
            ggml_build_forward_expand(graph, ids_output);
            ggml_build_forward_expand(graph, routing.weights);
            std::vector<std::pair<ggml_tensor *, ggml_tensor *>> comparisons;
            for (bool relu : {false, true}) {
                for (bool rounded : {false, true}) {
                    modules::MoeExpertConfig c;
                    c.activation = relu ? modules::MoeActivation::ReluSquared : modules::MoeActivation::SwiGlu;
                    c.round_projections = rounded;
                    c.round_activation = rounded;
                    c.round_product = rounded;
                    c.round_output = rounded;
                    modules::MoeModule module(c);
                    auto * actual = module.build(ctx, x, routing, {u, relu ? nullptr : u, d});
                    // Independent legacy graph: padding, each rounding boundary,
                    // activation, then the original selected-lane reduction.
                    auto * rows = ggml_reshape_3d(ctx.ggml, x, h, 1, tokens);
                    rows = ggml_pad(ctx.ggml, rows, 16 - h, 0, 0, 0);
                    auto * first = ggml_mul_mat_id(ctx.ggml, u, rows, routing.ids);
                    if (rounded) first = ggml_round_bf16(ctx.ggml, first);
                    auto * activated = relu ? ggml_relu(ctx.ggml, first) : ggml_silu(ctx.ggml, first);
                    if (rounded) activated = ggml_round_bf16(ctx.ggml, activated);
                    if (relu) activated = ggml_sqr(ctx.ggml, activated);
                    else {
                        auto * second = ggml_mul_mat_id(ctx.ggml, u, rows, routing.ids);
                        if (rounded) second = ggml_round_bf16(ctx.ggml, second);
                        activated = ggml_mul(ctx.ggml, activated, second);
                    }
                    if (rounded) activated = ggml_round_bf16(ctx.ggml, activated);
                    activated = ggml_pad(ctx.ggml, activated, 32 - width, 0, 0, 0);
                    auto * output = ggml_mul_mat_id(ctx.ggml, d, activated, routing.ids);
                    if (rounded) output = ggml_round_bf16(ctx.ggml, output);
                    output = ggml_mul(ctx.ggml, output, ggml_reshape_3d(ctx.ggml, routing.weights, 1, selected, tokens));
                    output = ggml_cont(ctx.ggml, ggml_permute(ctx.ggml, output, 1, 0, 2, 3));
                    output = ggml_reshape_2d(ctx.ggml, ggml_sum_rows(ctx.ggml, output), h, tokens);
                    for (auto * t : {actual, output}) { ggml_set_output(t); ggml_build_forward_expand(graph, t); }
                    comparisons.emplace_back(actual, output);
                }
            }
            modules::FeedForwardConfig ff_config{h, h, false};
            ff_config.activation = modules::FeedForwardActivation::ReluSquared;
            ctx.elide_identity_reshapes = true;
            const auto dense_weight = core::wrap_tensor(r, core::TensorShape::from_dims({h, h}));
            const auto shared = modules::FeedForwardModule(ff_config).build(ctx,
                core::wrap_tensor(x, core::TensorShape::from_dims({tokens, h})),
                {dense_weight, std::nullopt, dense_weight, std::nullopt});
            auto * legacy_shared = ggml_mul_mat(ctx.ggml, r, x);
            legacy_shared = ggml_sqr(ctx.ggml, ggml_relu(ctx.ggml, legacy_shared));
            legacy_shared = ggml_mul_mat(ctx.ggml, r, legacy_shared);
            for (auto * tensor : {shared.tensor, legacy_shared}) {
                ggml_set_output(tensor);
                ggml_build_forward_expand(graph, tensor);
            }
            comparisons.emplace_back(shared.tensor, legacy_shared);
            std::unique_ptr<std::remove_pointer_t<ggml_gallocr_t>, decltype(&ggml_gallocr_free)> allocator(
                ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend())), ggml_gallocr_free);
            if (!ggml_gallocr_alloc_graph(allocator.get(), graph)) throw std::runtime_error("MoE test graph allocation failed");
            core::HostGraphPlan plan;
            core::prepare_host_graph_plan(execution, graph, plan);
            if (core::compute_graph(execution, graph, plan, "test.moe") != GGML_STATUS_SUCCESS) throw std::runtime_error("MoE test execution failed");
            const auto ids = core::read_tensor_i32(ids_output);
            const auto weights = core::read_tensor_f32(routing.weights);
            for (int t = 0; t < tokens; ++t) {
                std::vector<float> scores(experts), selection(experts);
                float sum = 0;
                for (int e = 0; e < experts; ++e) {
                    float dot = 0;
                    for (int i = 0; i < h; ++i) dot += input[t * h + i] * router[e * h + i];
                    scores[e] = scoring == modules::MoeScoring::Softmax ? std::exp(dot) : 1.0F / (1.0F + std::exp(-dot));
                    sum += scores[e];
                }
                if (scoring == modules::MoeScoring::Softmax) for (auto & score : scores) score /= sum;
                for (int e = 0; e < experts; ++e) selection[e] = scores[e] + bias[e];
                std::vector<int> order(experts); std::iota(order.begin(), order.end(), 0);
                if (config.groups > 1) {
                    float group_scores[2]{};
                    for (int group = 0; group < 2; ++group) {
                        std::vector<float> v(selection.begin() + group * 4, selection.begin() + group * 4 + 4);
                        std::sort(v.begin(), v.end(), std::greater<float>());
                        group_scores[group] = v[0] + v[1];
                    }
                    const int group = group_scores[0] >= group_scores[1] ? 0 : 1;
                    order.erase(std::remove_if(order.begin(), order.end(), [&](int e) { return e / 4 != group; }), order.end());
                }
                std::sort(order.begin(), order.end(), [&](int a, int b) { return selection[a] > selection[b]; });
                float denominator = scores[order[0]] + scores[order[1]];
                if (normalization == modules::MoeNormalization::Clamp) denominator = std::max(denominator, config.denominator_epsilon);
                if (normalization == modules::MoeNormalization::AddEpsilon) denominator += config.denominator_epsilon;
                for (int k = 0; k < selected; ++k) {
                    if (ids[t * selected + k] != order[k] || std::abs(weights[t * selected + k] - scores[order[k]] / denominator * config.scale) > 2e-6F)
                        throw std::runtime_error("MoE routing differs from host mathematical reference: scoring=" + std::to_string(static_cast<int>(scoring)) + " normalization=" + std::to_string(static_cast<int>(normalization)) + " token=" + std::to_string(t) + " lane=" + std::to_string(k) + " actual_id=" + std::to_string(ids[t * selected + k]) + " expected_id=" + std::to_string(order[k]) + " actual_weight=" + std::to_string(weights[t * selected + k]) + " expected_weight=" + std::to_string(scores[order[k]] / denominator * config.scale));
                }
            }
            for (const auto & pair : comparisons) {
                const auto actual = core::read_tensor_f32(pair.first), expected = core::read_tensor_f32(pair.second);
                for (size_t i = 0; i < actual.size(); ++i)
                    if (!std::isfinite(actual[i]) || std::abs(actual[i] - expected[i]) > 1e-6F)
                        throw std::runtime_error("MoE expert math/rounding differs from legacy graph");
            }
            core::release_backend_graph_resources(execution.backend(), graph, true);
        }
    }
}
}

int main(int argc, char ** argv) {
    try {
        engine::debug::configure_logging({true, std::nullopt});
        const auto start = std::chrono::steady_clock::now();
        core::BackendConfig config; config.threads = 8;
        if (argc > 1 && std::string(argv[1]) == "cuda") config.type = core::BackendType::Cuda;
        if (argc > 1 && std::string(argv[1]) == "vulkan") config.type = core::BackendType::Vulkan;
        core::ExecutionContext execution(config);
        routing_and_experts(execution);
        engine::debug::timing_log_scalar("session.wall_ms", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
        std::cout << "MoE routing and expert tests passed\n";
    } catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
}
