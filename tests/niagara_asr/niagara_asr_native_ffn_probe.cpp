#include "engine/community_models/niagara_asr/assets.h"
#include "engine/community_models/niagara_asr/encoder.h"
#include "engine/community_models/niagara_asr/weights.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/linear_module.h"

#include "ggml-alloc.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

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

std::vector<float> make_input(int64_t elements) {
    std::vector<float> values(static_cast<size_t>(elements), 0.0f);
    for (int64_t i = 0; i < elements; ++i) {
        values[static_cast<size_t>(i)] = std::sin(static_cast<float>(i) * 0.013f);
    }
    return values;
}

std::vector<float> compute_reference_l1(
    const engine::community_models::niagara_asr::NiagaraAsrAssets & assets,
    const std::vector<float> & input,
    int64_t frames) {
    if (assets.source == nullptr) {
        throw std::runtime_error("native FFN probe requires GGUF tensor source");
    }
    const auto & source = *assets.source;
    const auto & config = assets.config.encoder;
    const int64_t hidden = config.hidden_size;
    const std::string prefix = "p__torch_params_encoder_lmuformer_ffn_";
    const auto gamma = source.require_f32(prefix + "l1norm_gamma", {hidden});
    const auto beta = source.require_f32(prefix + "l1norm_beta", {hidden});

    std::vector<float> output(static_cast<size_t>(frames * hidden), 0.0f);
    for (int64_t frame = 0; frame < frames; ++frame) {
        const float * row = input.data() + frame * hidden;
        float mean = std::accumulate(row, row + hidden, 0.0f) / static_cast<float>(hidden);
        float mean_abs = 0.0f;
        for (int64_t h = 0; h < hidden; ++h) {
            mean_abs += std::abs(row[h] - mean);
        }
        const float inv_den = 1.0f / (mean_abs / static_cast<float>(hidden) + 1.0e-6f);
        float * out_row = output.data() + frame * hidden;
        for (int64_t h = 0; h < hidden; ++h) {
            out_row[h] =
                ((row[h] - mean) * inv_den) * gamma[static_cast<size_t>(h)] + beta[static_cast<size_t>(h)];
        }
    }
    return output;
}

std::vector<float> compute_reference_ffn(
    const engine::community_models::niagara_asr::NiagaraAsrAssets & assets,
    const std::vector<float> & input,
    int64_t frames) {
    if (assets.source == nullptr) {
        throw std::runtime_error("native FFN probe requires GGUF tensor source");
    }
    const auto & source = *assets.source;
    const auto & config = assets.config.encoder;
    const int64_t hidden = config.hidden_size;
    const int64_t intermediate = config.intermediate_size;
    const std::string prefix = "p__torch_params_encoder_lmuformer_ffn_";
    const auto dense1_kernel = source.require_f32(prefix + "dense_1_kernel", {hidden, intermediate});
    const auto dense1_bias = source.require_f32(prefix + "dense_1_bias", {intermediate});
    const auto dense2_kernel = source.require_f32(prefix + "dense_2_kernel", {intermediate, hidden});
    const auto dense2_bias = source.require_f32(prefix + "dense_2_bias", {hidden});

    const auto norm = compute_reference_l1(assets, input, frames);
    std::vector<float> activation(static_cast<size_t>(intermediate), 0.0f);
    std::vector<float> output(static_cast<size_t>(frames * hidden), 0.0f);
    for (int64_t frame = 0; frame < frames; ++frame) {
        const float * norm_row = norm.data() + frame * hidden;
        for (int64_t out = 0; out < intermediate; ++out) {
            float value = dense1_bias[static_cast<size_t>(out)];
            for (int64_t in = 0; in < hidden; ++in) {
                value += norm_row[in] *
                    dense1_kernel[static_cast<size_t>(in * intermediate + out)];
            }
            activation[static_cast<size_t>(out)] = value / (1.0f + std::exp(-value));
        }

        float * out_row = output.data() + frame * hidden;
        for (int64_t out = 0; out < hidden; ++out) {
            float value = dense2_bias[static_cast<size_t>(out)];
            for (int64_t in = 0; in < intermediate; ++in) {
                value += activation[static_cast<size_t>(in)] *
                    dense2_kernel[static_cast<size_t>(in * hidden + out)];
            }
            out_row[out] = value;
        }
    }
    return output;
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

        ggml_init_params params{64ull * 1024ull * 1024ull, nullptr, true};
        std::unique_ptr<ggml_context, GgmlContextDeleter> graph_ctx(ggml_init(params));
        if (graph_ctx == nullptr) {
            throw std::runtime_error("failed to create Niagara native FFN probe graph context");
        }

        const auto & config = assets->config.encoder;
        engine::core::ModuleBuildContext build_ctx{graph_ctx.get(), "niagara_asr.native_ffn_probe", execution.backend_type()};
        auto input = engine::core::make_tensor(
            build_ctx,
            GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({1, args.frames, config.hidden_size}));
        ggml_set_input(input.tensor);
        auto l1 = engine::community_models::niagara_asr::build_niagara_l1_norm(
            build_ctx,
            input,
            weights->layers.front().ffn.norm,
            config);
        ggml_set_output(l1.tensor);
        auto hidden = engine::modules::LinearModule({config.hidden_size, config.intermediate_size, true})
                          .build(build_ctx, l1, weights->layers.front().ffn.dense1);
        hidden = engine::modules::SiluModule{}.build(build_ctx, hidden);
        auto output = engine::modules::LinearModule({config.intermediate_size, config.hidden_size, true})
                          .build(build_ctx, hidden, weights->layers.front().ffn.dense2);
        ggml_set_output(output.tensor);

        ggml_cgraph * graph = ggml_new_graph_custom(graph_ctx.get(), 4096, false);
        ggml_build_forward_expand(graph, output.tensor);
        std::unique_ptr<ggml_gallocr, GgmlGallocrDeleter> gallocr(
            ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend())));
        if (gallocr == nullptr ||
            !ggml_gallocr_reserve(gallocr.get(), graph) ||
            !ggml_gallocr_alloc_graph(gallocr.get(), graph)) {
            throw std::runtime_error("failed to allocate Niagara native FFN probe graph tensors");
        }

        const auto input_values = make_input(input.shape.num_elements());
        engine::core::write_tensor_f32(input, input_values);

        for (int i = 0; i < args.warmup; ++i) {
            const auto status = engine::core::compute_backend_graph(execution.backend(), graph, nullptr, "Niagara native FFN probe");
            if (status != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("native FFN probe warmup failed");
            }
        }
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < std::max(1, args.iterations); ++i) {
            const auto status = engine::core::compute_backend_graph(execution.backend(), graph, nullptr, "Niagara native FFN probe");
            if (status != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("native FFN probe compute failed");
            }
        }
        const auto end = std::chrono::steady_clock::now();
        const double average_ms =
            std::chrono::duration<double, std::milli>(end - start).count() / std::max(1, args.iterations);
        const auto values = engine::core::read_tensor_f32(output.tensor);
        const auto l1_values = engine::core::read_tensor_f32(l1.tensor);
        if (values.empty()) {
            throw std::runtime_error("native FFN probe produced empty output");
        }
        const auto l1_reference = compute_reference_l1(*assets, input_values, args.frames);
        const auto reference = compute_reference_ffn(*assets, input_values, args.frames);
        if (reference.size() != values.size()) {
            throw std::runtime_error("native FFN probe reference shape mismatch");
        }
        double l1_sum_sq = 0.0;
        float l1_max_abs_diff = 0.0f;
        for (size_t i = 0; i < l1_values.size(); ++i) {
            const float diff = std::abs(l1_values[i] - l1_reference[i]);
            l1_max_abs_diff = std::max(l1_max_abs_diff, diff);
            l1_sum_sq += static_cast<double>(diff) * static_cast<double>(diff);
        }
        const double l1_rmse = std::sqrt(l1_sum_sq / static_cast<double>(l1_values.size()));
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
        std::cout << "output_values=" << values.size() << "\n";
        std::cout << "average_ms=" << average_ms << "\n";
        std::cout << "l1_max_abs_diff=" << l1_max_abs_diff << "\n";
        std::cout << "l1_rmse=" << l1_rmse << "\n";
        std::cout << "max_abs_diff=" << max_abs_diff << "\n";
        std::cout << "rmse=" << rmse << "\n";
        if (max_abs_diff > 1.0e-2f) {
            throw std::runtime_error("native FFN probe exceeded reference tolerance");
        }
        return 0;
    } catch (const std::exception & ex) {
        std::cerr << "niagara_asr_native_ffn_probe failed: " << ex.what() << "\n";
        return 1;
    }
}
