#include "engine/community_models/niagara_asr/assets.h"
#include "engine/community_models/niagara_asr/encoder.h"
#include "engine/community_models/niagara_asr/weights.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/execution_context.h"

#include "ggml-alloc.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int64_t kStateKernelSize = 128;

struct Args {
    std::filesystem::path model;
    int threads = 8;
    int frames = 8;
    int warmup = 1;
    int iterations = 3;
};

Args parse_args(int argc, char ** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto require_value = [&](const char * name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string(name) + " requires a value");
            }
            return argv[++i];
        };
        if (key == "--model") {
            args.model = require_value("--model");
        } else if (key == "--threads") {
            args.threads = std::stoi(require_value("--threads"));
        } else if (key == "--frames") {
            args.frames = std::stoi(require_value("--frames"));
        } else if (key == "--warmup") {
            args.warmup = std::stoi(require_value("--warmup"));
        } else if (key == "--iterations") {
            args.iterations = std::stoi(require_value("--iterations"));
        } else {
            throw std::runtime_error("unknown argument: " + key);
        }
    }
    if (args.model.empty()) {
        throw std::runtime_error("--model is required");
    }
    if (args.frames <= 0 || args.threads <= 0 || args.warmup < 0 || args.iterations <= 0) {
        throw std::runtime_error("invalid numeric argument");
    }
    return args;
}

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct GgmlGallocrDeleter {
    void operator()(ggml_gallocr * gallocr) const noexcept {
        if (gallocr != nullptr) {
            ggml_gallocr_free(gallocr);
        }
    }
};

std::string lmuformer_prefix(int64_t layer) {
    if (layer == 0) {
        return "p__torch_params_encoder_lmuformer_";
    }
    return "p__torch_params_encoder_lmuformer_" + std::to_string(layer) + "_";
}

std::string state_space_basis_name(int64_t layer) {
    return "c_lifted_tensor_" + std::to_string(44 + 40 * layer);
}

std::vector<float> make_input(int64_t elements) {
    std::vector<float> values(static_cast<size_t>(elements), 0.0f);
    for (int64_t i = 0; i < elements; ++i) {
        values[static_cast<size_t>(i)] = std::sin(static_cast<float>(i) * 0.013f);
    }
    return values;
}

std::vector<float> dense(
    const std::vector<float> & input,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    const std::vector<float> & kernel,
    const std::vector<float> & bias) {
    std::vector<float> output(static_cast<size_t>(rows * out_features), 0.0f);
    for (int64_t row = 0; row < rows; ++row) {
        for (int64_t out = 0; out < out_features; ++out) {
            float value = bias[static_cast<size_t>(out)];
            for (int64_t in = 0; in < in_features; ++in) {
                value += input[static_cast<size_t>(row * in_features + in)] *
                    kernel[static_cast<size_t>(in * out_features + out)];
            }
            output[static_cast<size_t>(row * out_features + out)] = value;
        }
    }
    return output;
}

std::vector<float> l1_norm(
    const engine::community_models::niagara_asr::NiagaraAsrAssets & assets,
    const std::string & prefix,
    const std::vector<float> & input,
    int64_t frames) {
    const auto & source = *assets.source;
    const auto hidden = assets.config.encoder.hidden_size;
    const auto gamma = source.require_f32(prefix + "l1norm_gamma", {hidden});
    const auto beta = source.require_f32(prefix + "l1norm_beta", {hidden});
    std::vector<float> output(static_cast<size_t>(frames * hidden), 0.0f);
    for (int64_t frame = 0; frame < frames; ++frame) {
        const float * row = input.data() + frame * hidden;
        const float mean = std::accumulate(row, row + hidden, 0.0f) / static_cast<float>(hidden);
        float mean_abs = 0.0f;
        for (int64_t h = 0; h < hidden; ++h) {
            mean_abs += std::abs(row[h] - mean);
        }
        const float inv_den = 1.0f / (mean_abs / static_cast<float>(hidden) + 1.0e-6f);
        for (int64_t h = 0; h < hidden; ++h) {
            output[static_cast<size_t>(frame * hidden + h)] =
                ((row[h] - mean) * inv_den) * gamma[static_cast<size_t>(h)] + beta[static_cast<size_t>(h)];
        }
    }
    return output;
}

std::vector<float> state_conv_kernel(
    const engine::community_models::niagara_asr::NiagaraAsrAssets & assets,
    int64_t layer) {
    const auto & source = *assets.source;
    const auto & config = assets.config.encoder;
    const auto prefix = lmuformer_prefix(layer);
    const auto basis = source.require_f32(
        state_space_basis_name(layer),
        {1, kStateKernelSize, config.state_size});
    const auto c = source.require_f32(
        prefix + "state_space_state_space_c",
        {config.state_channels, config.state_size, 1});
    std::vector<float> kernel(static_cast<size_t>(config.state_channels * kStateKernelSize), 0.0f);
    for (int64_t channel = 0; channel < config.state_channels; ++channel) {
        for (int64_t k = 0; k < kStateKernelSize; ++k) {
            double sum = 0.0;
            const int64_t source_k = kStateKernelSize - 1 - k;
            for (int64_t s = 0; s < config.state_size; ++s) {
                sum += static_cast<double>(basis[static_cast<size_t>(source_k * config.state_size + s)]) *
                    static_cast<double>(c[static_cast<size_t>(channel * config.state_size + s)]);
            }
            kernel[static_cast<size_t>(channel * kStateKernelSize + k)] = static_cast<float>(sum);
        }
    }
    return kernel;
}

std::vector<float> compute_reference_state_space(
    const engine::community_models::niagara_asr::NiagaraAsrAssets & assets,
    const std::vector<float> & input,
    int64_t frames) {
    const auto & source = *assets.source;
    const auto & config = assets.config.encoder;
    const auto prefix = lmuformer_prefix(0);
    auto x = l1_norm(assets, prefix + "state_space_", input, frames);
    x = dense(
        x,
        frames,
        config.hidden_size,
        config.state_channels,
        source.require_f32(prefix + "state_space_dense_1_kernel", {config.hidden_size, config.state_channels}),
        source.require_f32(prefix + "state_space_dense_1_bias", {config.state_channels}));

    const auto kernel = state_conv_kernel(assets, 0);
    std::vector<float> conv(static_cast<size_t>(frames * config.state_channels), 0.0f);
    for (int64_t frame = 0; frame < frames; ++frame) {
        for (int64_t channel = 0; channel < config.state_channels; ++channel) {
            double sum = 0.0;
            for (int64_t k = 0; k < kStateKernelSize; ++k) {
                const int64_t source_frame = frame + k - (kStateKernelSize - 1);
                if (source_frame >= 0) {
                    sum += static_cast<double>(x[static_cast<size_t>(source_frame * config.state_channels + channel)]) *
                        static_cast<double>(kernel[static_cast<size_t>(channel * kStateKernelSize + k)]);
                }
            }
            conv[static_cast<size_t>(frame * config.state_channels + channel)] = static_cast<float>(sum);
        }
    }

    if (config.state_channels == config.hidden_size * 2) {
        std::vector<float> gated(static_cast<size_t>(frames * config.hidden_size), 0.0f);
        for (int64_t i = 0; i < frames * config.hidden_size; ++i) {
            const int64_t frame = i / config.hidden_size;
            const int64_t channel = i % config.hidden_size;
            const float value = conv[static_cast<size_t>(frame * config.state_channels + channel)];
            const float gate = conv[static_cast<size_t>(frame * config.state_channels + config.hidden_size + channel)];
            gated[static_cast<size_t>(i)] = value / (1.0f + std::exp(-gate));
        }
        x = std::move(gated);
    } else if (config.state_channels == config.hidden_size) {
        for (float & value : conv) {
            value = value / (1.0f + std::exp(-value));
        }
        x = std::move(conv);
    } else {
        throw std::runtime_error("unsupported Niagara state-space channel count");
    }

    return dense(
        x,
        frames,
        config.hidden_size,
        config.hidden_size,
        source.require_f32(prefix + "state_space_dense_2_kernel", {config.hidden_size, config.hidden_size}),
        source.require_f32(prefix + "state_space_dense_2_bias", {config.hidden_size}));
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const auto args = parse_args(argc, argv);
        auto assets = engine::community_models::niagara_asr::load_niagara_asr_assets(args.model);
        engine::core::ExecutionContext execution({engine::core::BackendType::Cpu, 0, args.threads});
        const auto weights = engine::community_models::niagara_asr::load_niagara_weights(
            *assets,
            execution,
            engine::assets::TensorStorageType::F32,
            512ull * 1024ull * 1024ull);

        ggml_init_params params{96ull * 1024ull * 1024ull, nullptr, true};
        std::unique_ptr<ggml_context, GgmlContextDeleter> graph_ctx(ggml_init(params));
        if (graph_ctx == nullptr) {
            throw std::runtime_error("failed to create Niagara native state-space probe graph context");
        }

        const auto & config = assets->config.encoder;
        engine::core::ModuleBuildContext build_ctx{graph_ctx.get(), "niagara_asr.native_state_space_probe", execution.backend_type()};
        auto input = engine::core::make_tensor(
            build_ctx,
            GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({1, args.frames, config.hidden_size}));
        ggml_set_input(input.tensor);
        auto output = engine::community_models::niagara_asr::build_niagara_state_space(
            build_ctx,
            input,
            weights->layers.front().state_space,
            config);
        ggml_set_output(output.tensor);

        ggml_cgraph * graph = ggml_new_graph_custom(graph_ctx.get(), 8192, false);
        ggml_build_forward_expand(graph, output.tensor);
        std::unique_ptr<ggml_gallocr, GgmlGallocrDeleter> gallocr(
            ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend())));
        if (gallocr == nullptr ||
            !ggml_gallocr_reserve(gallocr.get(), graph) ||
            !ggml_gallocr_alloc_graph(gallocr.get(), graph)) {
            throw std::runtime_error("failed to allocate Niagara native state-space probe graph tensors");
        }

        const auto input_values = make_input(input.shape.num_elements());
        engine::core::write_tensor_f32(input, input_values);
        for (int i = 0; i < args.warmup; ++i) {
            const auto status = engine::core::compute_backend_graph(execution.backend(), graph, nullptr, "Niagara native state-space probe");
            if (status != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("native state-space probe warmup failed");
            }
        }
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < std::max(1, args.iterations); ++i) {
            const auto status = engine::core::compute_backend_graph(execution.backend(), graph, nullptr, "Niagara native state-space probe");
            if (status != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("native state-space probe compute failed");
            }
        }
        const auto end = std::chrono::steady_clock::now();
        const double average_ms =
            std::chrono::duration<double, std::milli>(end - start).count() / std::max(1, args.iterations);

        const auto values = engine::core::read_tensor_f32(output.tensor);
        const auto reference = compute_reference_state_space(*assets, input_values, args.frames);
        if (reference.size() != values.size()) {
            throw std::runtime_error("native state-space probe reference shape mismatch");
        }
        double sum_sq = 0.0;
        float max_abs_diff = 0.0f;
        for (size_t i = 0; i < values.size(); ++i) {
            const float diff = std::abs(values[i] - reference[i]);
            max_abs_diff = std::max(max_abs_diff, diff);
            sum_sq += static_cast<double>(diff) * static_cast<double>(diff);
        }
        const double rmse = std::sqrt(sum_sq / static_cast<double>(values.size()));

        std::cout << "family=niagara_asr\n";
        std::cout << "frames=" << args.frames << "\n";
        std::cout << "hidden_size=" << config.hidden_size << "\n";
        std::cout << "state_channels=" << config.state_channels << "\n";
        std::cout << "output_values=" << values.size() << "\n";
        std::cout << "average_ms=" << average_ms << "\n";
        std::cout << "max_abs_diff=" << max_abs_diff << "\n";
        std::cout << "rmse=" << rmse << "\n";
        if (max_abs_diff > 1.0e-2f) {
            throw std::runtime_error("native state-space probe exceeded reference tolerance");
        }
        return 0;
    } catch (const std::exception & ex) {
        std::cerr << "niagara_asr_native_state_space_probe failed: " << ex.what() << "\n";
        return 1;
    }
}
