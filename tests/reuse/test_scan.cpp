#include "app/cli/args.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/debug/profiler.h"

#include <ggml-alloc.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

int main(int argc, char ** argv) {
    try {
        const auto backend_name = minitts::cli::find_arg(argc, argv, "--backend").value_or("cuda");
        const auto log = minitts::cli::find_arg(argc, argv, "--log-file").value();
        const int batch = minitts::cli::parse_int_arg(argc, argv, "--batch", 2);
        const bool contiguous = minitts::cli::has_arg(argc, argv, "--contiguous-bc");
        if (batch < 1 || batch > 442) {
            throw std::runtime_error("--batch must be between 1 and 442");
        }
        if (!minitts::cli::has_arg(argc, argv, "--log")) {
            throw std::runtime_error("--log is required");
        }
        minitts::cli::require_known_args(argc, argv);
        engine::debug::configure_logging({true, log});
        engine::core::ExecutionContext execution({minitts::cli::parse_backend(backend_name), 0, 8});
        constexpr int states = 16, channels = 256, tokens = 257;
        std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init({4 * 1024 * 1024, nullptr, true}), ggml_free);
        auto * s = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, states, 1, channels, batch);
        auto * x = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, 1, channels, tokens, batch);
        auto * dt = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, channels, tokens, batch);
        auto * a = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, states, channels);
        auto * bc = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 36, tokens, batch);
        auto * b = ggml_view_4d(ctx.get(), bc, states, 1, tokens, batch,
            states * sizeof(float), bc->nb[1], bc->nb[2], 4 * sizeof(float));
        auto * c = ggml_view_4d(ctx.get(), bc, states, 1, tokens, batch,
            states * sizeof(float), bc->nb[1], bc->nb[2], 20 * sizeof(float));
        auto * ids = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, batch);
        auto * out = ggml_ssm_scan(ctx.get(), s, x, dt, a,
            contiguous ? ggml_cont(ctx.get(), b) : b, contiguous ? ggml_cont(ctx.get(), c) : c, ids);
        auto * graph = ggml_new_graph(ctx.get());
        ggml_build_forward_expand(graph, out);
        std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
            ggml_backend_alloc_ctx_tensors(ctx.get(), execution.backend()), ggml_backend_buffer_free);
        if (!buffer) {
            throw std::runtime_error("Scan test allocation failed");
        }
        std::vector<float> sv(states * channels * batch, 0), xv(channels * tokens * batch), dtv(xv.size()),
            av(states * channels), bv(states * tokens * batch), cv(bv.size());
        for (size_t i = 0; i < xv.size(); ++i) {
            xv[i] = std::sin(static_cast<float>(i) * 0.013f);
            dtv[i] = -3.0f + std::cos(static_cast<float>(i) * 0.007f);
        }
        for (size_t i = 0; i < av.size(); ++i) {
            av[i] = -0.2f - 0.03f * static_cast<float>(i % states);
        }
        for (size_t i = 0; i < bv.size(); ++i) {
            bv[i] = std::sin(static_cast<float>(i) * 0.017f);
            cv[i] = std::cos(static_cast<float>(i) * 0.011f);
        }
        ggml_backend_tensor_set(s, sv.data(), 0, ggml_nbytes(s));
        ggml_backend_tensor_set(x, xv.data(), 0, ggml_nbytes(x));
        ggml_backend_tensor_set(dt, dtv.data(), 0, ggml_nbytes(dt));
        ggml_backend_tensor_set(a, av.data(), 0, ggml_nbytes(a));
        std::vector<float> packed(36 * tokens * batch, 0);
        for (int i = 0; i < tokens * batch; ++i) {
            std::copy_n(bv.data() + i * states, states, packed.data() + i * 36 + 4);
            std::copy_n(cv.data() + i * states, states, packed.data() + i * 36 + 20);
        }
        ggml_backend_tensor_set(bc, packed.data(), 0, ggml_nbytes(bc));
        std::vector<int32_t> id_values(batch);
        for (int i = 0; i < batch; ++i) {
            id_values[i] = i;
        }
        ggml_backend_tensor_set(ids, id_values.data(), 0, ggml_nbytes(ids));
        std::vector<float> expected(xv.size() + sv.size(), 0);
        for (int seq = 0; seq < batch; ++seq) {
            for (int channel = 0; channel < channels; ++channel) {
                float state[states] = {};
                for (int t = 0; t < tokens; ++t) {
                    const int index = (seq * tokens + t) * channels + channel;
                    const float delta = std::log1p(std::exp(dtv[index]));
                    float value = 0;
                    for (int j = 0; j < states; ++j) {
                        const int bc = (seq * tokens + t) * states + j;
                        state[j] = state[j] * std::exp(delta * av[channel * states + j]) + bv[bc] * (xv[index] * delta);
                        value += state[j] * cv[bc];
                    }
                    expected[index] = value;
                }
                for (int j = 0; j < states; ++j) {
                    expected[xv.size() + (seq * channels + channel) * states + j] = state[j];
                }
            }
        }
        std::vector<float> previous, actual(expected.size());
        bool failed = false;
        for (int iteration = 0; iteration < 5; ++iteration) {
            if (engine::core::compute_backend_graph(execution.backend(), graph) != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("Scan execution failed");
            }
            ggml_backend_tensor_get(out, actual.data(), 0, ggml_nbytes(out));
            double error = 0, repeat = 0;
            for (size_t i = 0; i < actual.size(); ++i) {
                if (!std::isfinite(actual[i])) {
                    throw std::runtime_error("Scan returned non-finite values");
                }
                error = std::max(error, std::abs(static_cast<double>(actual[i]) - expected[i]));
                if (!previous.empty()) {
                    repeat = std::max(repeat, std::abs(static_cast<double>(actual[i]) - previous[i]));
                }
            }
            engine::debug::log_message("scan iteration=" + std::to_string(iteration) +
                " max_abs=" + std::to_string(error) + " repeat_max_abs=" + std::to_string(repeat));
            failed |= error > 2e-5 || repeat != 0;
            previous = actual;
        }
        engine::core::release_backend_graph_resources(execution.backend(), graph, true);
        if (failed) {
            throw std::runtime_error("Scan recurrence or repeatability check failed");
        }
        engine::debug::log_message("PASS RE-USE scan recurrence and repeatability");
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
