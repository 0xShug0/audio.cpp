#include "engine/framework/modules/transformers/dit_runtime.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/io/safetensors.h"
#include "engine/framework/sampling/flow_sampler_runtime.h"
#include "engine/framework/core/prepared_graph.h"
#include "test_assert.h"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>

using namespace engine;
using test::require;

namespace {

void check_solvers() {
    for (const auto rule : {modules::FlowSamplerUpdateRule::Euler, modules::FlowSamplerUpdateRule::Midpoint,
             modules::FlowSamplerUpdateRule::Rk4}) {
        modules::FlowSamplerRuntimeConfig config;
        config.label = "test.solver"; config.latent_shape = {2}; config.branches = {{"velocity", 1}};
        config.update_rule = rule;
        for (int i = 0; i < 4; ++i) config.schedule.push_back({i, i * .25f, (i + 1) * .25f, 0, 0});
        std::vector<float> observed;
        modules::FlowSamplerSingleBranchDenoiserConfig denoiser;
        denoiser.label = config.label; denoiser.branch_name = "velocity";
        denoiser.sampler_mode = [](const modules::FlowSamplerStepState &) { return "test"; };
        denoiser.predict = [&](const modules::FlowSamplerDenoiserInput & input) {
            require(input.state.graph_key.sampler_mode == "test", "solver must forward prepared graph key");
            observed.push_back(input.state.schedule.t);
            return std::vector<float>{input.latent[0] + input.state.schedule.t, input.latent[1] + input.state.schedule.t};
        };
        modules::FlowSamplerRuntime runtime(config, modules::make_flow_sampler_single_branch_denoiser(std::move(denoiser)));
        // w = y+t+1 obeys w'=w. The three stability polynomials provide an
        // independent closed-form result for all intermediate solver stages.
        const double h = .25;
        const double amplification = 1 + h + (rule == modules::FlowSamplerUpdateRule::Euler ? 0 : h*h/2) +
            (rule == modules::FlowSamplerUpdateRule::Rk4 ? h*h*h/6 + h*h*h*h/24 : 0);
        for (const float initial : {1.f, 2.f, 1.f}) {
            observed.clear(); runtime.run_sequence({initial, -initial});
            require(observed == modules::flow_sampler_evaluation_times(config.schedule, rule), "solver evaluation order");
            test::require_close(runtime.latent()[0], float((initial + 1) * std::pow(amplification, 4) - 2), 2e-6f, "solver positive result");
            test::require_close(runtime.latent()[1], float((1 - initial) * std::pow(amplification, 4) - 2), 2e-6f, "solver negative result");
        }
    }
    modules::FlowSamplerRuntimeConfig fixed;
    fixed.label = "test.fixed_step"; fixed.latent_shape = {1}; fixed.branches = {{"velocity", 1}};
    fixed.step_size = .1f;
    for (int i = 0; i < 10; ++i) fixed.schedule.push_back({i, i*.1f, i*.1f+.1f, 0, 0});
    modules::FlowSamplerSingleBranchDenoiserConfig denoiser;
    denoiser.label = fixed.label; denoiser.branch_name = "velocity";
    denoiser.sampler_mode = [](const modules::FlowSamplerStepState &) { return "fixed"; };
    denoiser.predict = [](const modules::FlowSamplerDenoiserInput & input) {
        require(input.state.step_size == .1f, "fixed step must not be recovered from rounded endpoint labels");
        return std::vector<float>{1};
    };
    modules::FlowSamplerRuntime runtime(fixed, modules::make_flow_sampler_single_branch_denoiser(std::move(denoiser)));
    runtime.run_sequence({0});
    float expected = 0;
    for (int i = 0; i < 10; ++i) expected += .1f;
    require(runtime.latent()[0] == expected, "fixed-step Euler result");
}

void check_apg(core::ExecutionContext & execution) {
    constexpr int B = 2, T = 3, C = 4, N = B*T*C;
    std::vector<float> positive(N), negative(N), reference(N), mask{1,0,1, 0,1,1};
    for (int i = 0; i < N; ++i) {
        positive[i] = std::sin(i * .37f); negative[i] = std::cos(i * .41f); reference[i] = std::cos(i * .23f);
    }
    for (const bool channels : {false, true}) for (const bool masked : {false, true}) {
        core::PreparedGraph graph(execution, {"test.apg", 1024*1024, 256});
        auto & ctx = graph.build_context();
        const auto input = [&](core::TensorShape shape) {
            auto value = core::make_tensor(ctx, GGML_TYPE_F32, shape);
            ggml_set_input(value.tensor); ggml_set_output(value.tensor); return value;
        };
        const auto shape = core::TensorShape::from_dims({B,T,C});
        auto cond = input(shape), uncond = input(shape), direction = input(shape);
        auto scale = input(core::TensorShape::from_dims({1})), strength = input(core::TensorShape::from_dims({1}));
        core::TensorValue padding;
        if (masked) padding = input(core::TensorShape::from_dims({B,T,1}));
        const auto out = modules::FlowSamplerApgGraphModule(channels ? std::vector<int>{1,2} : std::vector<int>{1}).build(
            ctx, cond, uncond, direction, scale, strength, padding);
        ggml_set_output(out.tensor); ggml_build_forward_expand(graph.graph(), out.tensor); graph.allocate();
        core::write_tensor_f32(cond, positive); core::write_tensor_f32(uncond, negative); core::write_tensor_f32(direction, reference);
        core::write_tensor_f32(scale, {4}); if (masked) core::write_tensor_f32(padding, mask);
        for (const float amount : {0.f, .4f, 1.f}) {
            core::write_tensor_f32(strength, {amount}); graph.compute();
            const auto actual = core::read_tensor_f32(out.tensor);
            for (int b = 0; b < B; ++b) for (int group = 0; group < (channels ? 1 : C); ++group) {
                double dot = 0, norm = 0;
                for (int t = 0; t < T; ++t) for (int c = 0; c < C; ++c) if (channels || group == c) {
                    const auto i = (b*T+t)*C+c;
                    const double r = reference[i] * (masked ? mask[b*T+t] : 1);
                    dot += (positive[i] - negative[i]) * r; norm += r*r;
                }
                for (int t = 0; t < T; ++t) for (int c = 0; c < C; ++c) if (channels || group == c) {
                    const auto i = (b*T+t)*C+c;
                    const double m = masked ? mask[b*T+t] : 1, diff = positive[i] - negative[i];
                    const double orthogonal = (diff - dot / (norm + 1e-16) * reference[i] * m) * m;
                    const double expected = positive[i] + 3 * (amount * orthogonal + (1-amount)*diff);
                    test::require_close(actual[i], float(expected), 3e-6f, "APG masked projection");
                    // Independent denoised-space equation used by Stable Audio:
                    // x0_cond - x0_uncond = -sigma * (cond - uncond).
                    // Verify the velocity simplification across noise levels.
                    for (const double sigma : {.01, .3, 1.}) {
                        const double x = reference[i] + sigma * positive[i];
                        const double x0_difference = -sigma * diff;
                        const double direction = reference[i] * m / std::sqrt(norm + 1e-16);
                        const double projection = -sigma * dot / std::sqrt(norm + 1e-16) * direction;
                        const double x0_orthogonal = (x0_difference - projection) * m;
                        const double guided_x0 = reference[i] + 3 *
                            (amount * x0_orthogonal + (1-amount) * x0_difference);
                        test::require_close(actual[i], float((x-guided_x0)/sigma), 3e-6f,
                            "APG denoised-space equivalence");
                    }
                }
            }
        }
    }
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 3 || std::string(argv[1]) != "--log") return 2;
    debug::configure_logging({true, argv[2]});
    const auto started = std::chrono::steady_clock::now();
    const auto path = std::filesystem::temp_directory_path() /
        ("audiocpp_dit_runtime_" + std::to_string(started.time_since_epoch().count()) + ".safetensors");
    try {
        check_solvers();
        // Small, model-independent weights exercise all conditioning channels.
        std::vector<io::SafeTensorWriteEntry> tensors;
        const auto add = [&](std::string name, int64_t rows, int64_t columns, const std::vector<float> & values) {
            io::SafeTensorWriteEntry entry{std::move(name), "F32", {rows, columns}, {}};
            require(values.size() == static_cast<size_t>(rows * columns), "fixture matrix shape");
            entry.data.resize(values.size() * sizeof(float));
            std::memcpy(entry.data.data(), values.data(), entry.data.size());
            tensors.push_back(std::move(entry));
        };
        const std::vector<float> eye{1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1}, zero(16, 0);
        for (const auto * name : {"input", "self.v", "self.out", "cross.v", "cross.out", "local"}) add(name, 4, 4, eye);
        for (const auto * name : {"self.q", "self.k", "cross.q", "cross.k", "up", "down"}) add(name, 4, 4, zero);
        add("output", 2, 4, {0,0,1,0, 0,0,0,1});
        add("modulation", 24, 4, std::vector<float>(96, 0));
        std::vector<float> gates(24, 0), final(32, 0);
        for (int i = 0; i < 4; ++i) { gates[8 + i] = 1; gates[20 + i] = 1; final[5 * i] = .03f; }
        add("gates", 1, 24, gates);
        // Bias is a vector in the checkpoint, unlike the matrix helper above.
        tensors.back().shape = {24};
        add("final", 8, 4, final);
        add("final_table", 2, 4, {.1f,.2f,.3f,.4f, .03f,.02f,.01f,.04f});
        tensors.back().shape = {1,2,4};
        io::write_safetensors_file(path, tensors);
        {
            const auto source = assets::open_tensor_source(path);
            core::ExecutionContext execution({core::BackendType::Cpu, 0, 8});
            check_apg(execution);
            modules::DitRuntimeConfig config;
            config.label = "test.dit";
            config.graph_metadata_bytes = 2 * 1024 * 1024;
            config.graph_nodes = 2048;
            auto & model = config.model;
            model.input_channels = 4; model.output_channels = 2; model.hidden_size = 4;
            model.intermediate_size = 4; model.layers = 1; model.time_features = 4;
            model.modulation = modules::DitModulation::PerLayer;
            model.context_dim = 4; model.global_dim = 4; model.local_condition_dim = 4;
            model.projection_precision = GGML_PREC_F32;
            model.self_attention = {1, 1, 4, modules::DitQKNorm::None, 1e-6f, {}, modules::ScaledDotProductAttentionLowering::Explicit};
            model.cross_attention = model.self_attention;
            modules::DitWeightNames names;
            names.input.weight = "input"; names.output.weight = "output";
            names.final_modulation.weight = "final";
            names.layer.modulation = {"modulation", "gates"};
            names.layer.feed_forward_in.weight = "up"; names.layer.feed_forward_out.weight = "down";
            names.layer.self_attention = {{"self.q", ""}, {"self.k", ""}, {"self.v", ""}, {"self.out", ""}, {}, {}, {}, {}};
            names.layer.cross_attention = {{"cross.q", ""}, {"cross.k", ""}, {"cross.v", ""}, {"cross.out", ""}, {}, {}, {}, {}};
            names.layer.local_embedding = {{"local", ""}};
            const auto values = [](size_t count, float scale) {
                std::vector<float> out(count);
                for (size_t i = 0; i < count; ++i) out[i] = scale * static_cast<float>(static_cast<int>(i % 17) - 8);
                return out;
            };
            const auto equal = [](const std::vector<float> & a, const std::vector<float> & b) {
                require(a.size() == b.size(), "prediction size");
                for (size_t i = 0; i < a.size(); ++i) {
                    require(std::isfinite(a[i]) && std::isfinite(b[i]), "non-finite prediction");
                    test::require_close(a[i], b[i], 1e-5f, "conditioning/guidance result");
                }
            };
            for (const auto cache : {modules::DitConditioningCache::ProjectedKV,
                     modules::DitConditioningCache::Unprojected}) {
                {
                    const auto guidance = modules::DitGuidance::Linear;
                    config.conditioning_cache = cache; config.guidance = guidance;
                    modules::DitRuntime runtime(execution, *source, config, names);
                    const auto latent = values(12, .03f);
                    for (int request = 0; request < 2; ++request) {
                        modules::DitConditioning conditions;
                        conditions.batch = 2; conditions.branches = 2; conditions.frames = 3; conditions.context_tokens = 2;
                        const float scale = .02f * (request + 1);
                        conditions.context = values(32, scale); conditions.global = values(16, scale);
                        conditions.local_condition = values(48, scale);
                        conditions.frame_channels = {1}; conditions.vector_channels = {1};
                        conditions.frame_features = {values(12, scale)}; conditions.vector_features = {values(4, scale)};
                        conditions.value_mask = {1,1,0, 1,0,1, 0,1,1, 1,1,1};
                        runtime.prepare_conditioning(conditions);
                        std::vector<std::vector<float>> expected;
                        for (int64_t branch = 0; branch < 2; ++branch) {
                            auto single = conditions; single.branches = 1;
                            for (auto * data : {&single.context, &single.global, &single.local_condition,
                                     &single.frame_features[0], &single.vector_features[0], &single.value_mask}) {
                                const auto width = data->size() / 2;
                                *data = std::vector<float>(data->begin() + branch * width, data->begin() + (branch + 1) * width);
                            }
                            modules::DitRuntime reference(execution, *source, config, names);
                            reference.prepare_conditioning(std::move(single));
                            expected.push_back(reference.predict(latent, .375f));
                        }
                        require(expected[0] != expected[1], "fixture must distinguish positive and negative conditions");
                        for (const int64_t branch : {int64_t{0}, int64_t{1}, int64_t{0}, int64_t{1}})
                            equal(runtime.predict(latent, .375f, branch), expected[static_cast<size_t>(branch)]);
                        modules::DitSamplingRequest sampling;
                        sampling.noise = latent; sampling.steps = 1; sampling.schedule = {.375f, .125f, .375f, .125f};
                        for (const float gain : {1.f, 1.5f}) {
                            sampling.guidance = gain;
                            auto target = latent;
                            for (size_t i = 0; i < target.size(); ++i) {
                                const float base = expected[1][i];
                                target[i] += -.25f * (base + gain * (expected[0][i] - expected[1][i]));
                            }
                            equal(runtime.sample(sampling), target);
                        }
                        sampling.steps = 3; sampling.schedule = {.75f,.5f,.25f,0, .75f,.5f,.25f,0};
                        sampling.guidance = 1.5f;
                        auto target = latent;
                        for (const float time : {.75f, .5f, .25f}) {
                            const auto positive = runtime.predict(target, time, 0);
                            const auto negative = runtime.predict(target, time, 1);
                            for (size_t i = 0; i < target.size(); ++i)
                                target[i] += -.25f * (negative[i] + 1.5f * (positive[i] - negative[i]));
                        }
                        equal(runtime.sample(sampling), target);
                        equal(runtime.sample(sampling), target);
                        runtime.prepare_conditioning(conditions);
                        equal(runtime.sample(sampling), target);
                    }
                }
            }
            // Exercise projected/table-based final modulation, B=1/2, and
            // singleton-scale layouts with guidance disabled.
            config.model.global_dim = 0;
            for (const bool table : {false, true}) for (const int64_t batch : {int64_t{1}, int64_t{2}}) {
                names.final_modulation.weight = table ? "" : "final";
                names.final_modulation_table = table ? "final_table" : "";
                modules::DitRuntime runtime(execution, *source, config, names);
                modules::DitConditioning conditions;
                conditions.batch = batch; conditions.branches = 2; conditions.frames = 3; conditions.context_tokens = 2;
                conditions.context = values(static_cast<size_t>(batch*16), .02f);
                conditions.local_condition = values(static_cast<size_t>(batch*24), .02f);
                conditions.frame_channels = {1}; conditions.vector_channels = {1};
                conditions.frame_features = {values(static_cast<size_t>(batch*6), .02f)};
                conditions.vector_features = {values(static_cast<size_t>(batch*2), .02f)};
                runtime.prepare_conditioning(conditions);
                modules::DitSamplingRequest sampling;
                sampling.noise = values(static_cast<size_t>(batch*6), .03f);
                sampling.steps = 3;
                for (const float gain : {1.f, 1.5f}) {
                    sampling.guidance = gain;
                    const auto first = runtime.sample(sampling);
                    equal(runtime.sample(sampling), first);
                }
            }
        }
        std::filesystem::remove(path);
        debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(started));
        std::cout << "dit_runtime: PASS\n";
        return 0;
    } catch (const std::exception & error) {
        std::filesystem::remove(path);
        debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(started));
        std::cerr << error.what() << '\n';
        return 1;
    }
}
