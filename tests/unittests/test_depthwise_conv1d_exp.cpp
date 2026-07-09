#include "engine/framework/core/backend.h"
#include "engine/framework/modules/streaming_conv_modules.h"

#include "ggml-backend.h"

#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr size_t kGraphBytes = 512ull * 1024ull * 1024ull;
constexpr size_t kGraphNodes = 4096;
constexpr int64_t kBatch = 2;
constexpr int64_t kChannels = 256;
constexpr int64_t kOutputFrames = 96;
constexpr int64_t kKernel = 5;
constexpr int kStride = 1;
constexpr int kPadding = 0;
constexpr int kWarmupRounds = 5;
constexpr int kMeasureRounds = 25;

using Clock = std::chrono::steady_clock;

void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::vector<float> make_patterned_f32(size_t count, float phase, float scale) {
    std::vector<float> values(count, 0.0f);
    for (size_t i = 0; i < count; ++i) {
        const float x = static_cast<float>(i);
        values[i] = scale * (std::sin(phase + 0.173f * x) + 0.5f * std::cos(phase * 0.7f + 0.097f * x));
    }
    return values;
}

double elapsed_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void require_parity(
    const std::vector<float> & actual,
    const std::vector<float> & expected,
    float max_allowed,
    double mean_allowed,
    const std::string & label) {
    require(actual.size() == expected.size(), label + " size mismatch");
    float max_diff = 0.0f;
    size_t max_index = 0;
    double mean_diff = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const float diff = std::fabs(actual[i] - expected[i]);
        mean_diff += diff;
        if (diff > max_diff) {
            max_diff = diff;
            max_index = i;
        }
    }
    mean_diff /= static_cast<double>(actual.size());
    std::cout << label << " max_diff=" << max_diff << " mean_diff=" << mean_diff << "\n";
    if (max_diff > max_allowed || mean_diff > mean_allowed) {
        std::ostringstream oss;
        oss << label << " parity failed: max_diff=" << max_diff << " limit=" << max_allowed
            << " mean_diff=" << mean_diff << " limit=" << mean_allowed
            << " at=" << max_index << " actual=" << actual[max_index]
            << " expected=" << expected[max_index];
        throw std::runtime_error(oss.str());
    }
}

enum class Variant {
    Original,
    Exp2d,
};

struct RunResult {
    std::vector<float> values;
    double mean_ms = 0.0;
};

class CaseGraph {
public:
    CaseGraph(Variant variant, int dilation, engine::core::BackendType backend_type)
        : backend_config_{backend_type, 0, 8},
          backend_(engine::core::init_backend(backend_config_)),
          dilation_(dilation) {
        ggml_init_params params{};
        params.mem_size = kGraphBytes;
        params.mem_buffer = nullptr;
        params.no_alloc = true;
        ggml_ = ggml_init(params);
        if (ggml_ == nullptr) {
            throw std::runtime_error("failed to initialize ggml context");
        }

        engine::core::ModuleBuildContext ctx{ggml_, "depthwise_conv1d_exp_test", backend_type};
        const int64_t input_frames = kOutputFrames + static_cast<int64_t>(dilation) * (kKernel - 1);
        input_ = engine::core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({kBatch, kChannels, input_frames}));
        weight_ = engine::core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({kChannels, 1, kKernel}));
        bias_ = engine::core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({kChannels}));

        const engine::modules::DepthwiseConv1dConfig config{
            kChannels,
            kKernel,
            kStride,
            kPadding,
            dilation,
            true,
        };
        if (variant == Variant::Original) {
            output_ = engine::modules::DepthwiseConv1dModule(config).build(ctx, input_, {weight_, bias_});
        } else {
            output_ = engine::modules::DepthwiseConv1dModuleExp(config).build(ctx, input_, {weight_, bias_});
        }
        require(
            output_.shape.num_elements() == kBatch * kChannels * kOutputFrames,
            "unexpected depthwise output shape");

        graph_ = ggml_new_graph_custom(ggml_, kGraphNodes, false);
        ggml_build_forward_expand(graph_, output_.tensor);
        buffer_ = ggml_backend_alloc_ctx_tensors(ggml_, backend_);
        if (buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate backend tensors");
        }

        engine::core::write_tensor_f32(
            input_,
            make_patterned_f32(static_cast<size_t>(input_.shape.num_elements()), 0.11f + 0.03f * dilation, 0.07f));
        engine::core::write_tensor_f32(
            weight_,
            make_patterned_f32(static_cast<size_t>(weight_.shape.num_elements()), 0.37f + 0.05f * dilation, 0.013f));
        engine::core::write_tensor_f32(
            bias_,
            make_patterned_f32(static_cast<size_t>(bias_.shape.num_elements()), 0.73f + 0.07f * dilation, 0.003f));
    }

    ~CaseGraph() {
        if (buffer_ != nullptr) {
            ggml_backend_buffer_free(buffer_);
        }
        if (ggml_ != nullptr) {
            ggml_free(ggml_);
        }
        if (backend_ != nullptr) {
            ggml_backend_free(backend_);
        }
    }

    RunResult run() {
        for (int i = 0; i < kWarmupRounds; ++i) {
            compute();
        }

        std::vector<double> times;
        times.reserve(kMeasureRounds);
        for (int i = 0; i < kMeasureRounds; ++i) {
            const auto start = Clock::now();
            compute();
            times.push_back(elapsed_ms(start, Clock::now()));
        }

        RunResult result;
        result.mean_ms = std::accumulate(times.begin(), times.end(), 0.0) / static_cast<double>(times.size());
        engine::core::read_tensor_f32_into(output_.tensor, result.values);
        return result;
    }

private:
    void compute() {
        const ggml_status status = ggml_backend_graph_compute(backend_, graph_);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("depthwise graph compute failed for dilation " + std::to_string(dilation_));
        }
        ggml_backend_synchronize(backend_);
    }

    engine::core::BackendConfig backend_config_;
    ggml_backend_t backend_ = nullptr;
    ggml_context * ggml_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    engine::core::TensorValue input_{};
    engine::core::TensorValue weight_{};
    engine::core::TensorValue bias_{};
    engine::core::TensorValue output_{};
    int dilation_ = 1;
};

engine::core::BackendType parse_backend(int argc, char ** argv) {
    engine::core::BackendType backend = engine::core::BackendType::Cpu;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--backend") {
            require(i + 1 < argc, "--backend requires a value");
            const std::string value = argv[++i];
            if (value == "cpu") {
                backend = engine::core::BackendType::Cpu;
            } else if (value == "cuda") {
                backend = engine::core::BackendType::Cuda;
            } else {
                throw std::runtime_error("unsupported backend for depthwise test: " + value);
            }
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    return backend;
}

const char * backend_name(engine::core::BackendType backend) {
    switch (backend) {
        case engine::core::BackendType::Cpu:
            return "cpu";
        case engine::core::BackendType::Cuda:
            return "cuda";
        case engine::core::BackendType::Vulkan:
            return "vulkan";
        case engine::core::BackendType::Metal:
            return "metal";
        case engine::core::BackendType::BestAvailable:
            return "best";
    }
    return "unknown";
}

void run_case(int dilation, engine::core::BackendType backend) {
    CaseGraph original(Variant::Original, dilation, backend);
    CaseGraph exp(Variant::Exp2d, dilation, backend);

    const auto original_result = original.run();
    const auto exp_result = exp.run();
    require_parity(
        exp_result.values,
        original_result.values,
        2.0e-5f,
        2.0e-6,
        "dilation " + std::to_string(dilation));

    const double speedup = original_result.mean_ms / std::max(exp_result.mean_ms, std::numeric_limits<double>::min());
    if (backend == engine::core::BackendType::Cpu) {
        require(
            exp_result.mean_ms < original_result.mean_ms,
            "dilation " + std::to_string(dilation) + " exp2d path is not faster than original path");
    }
    std::cout << "backend=" << backend_name(backend)
              << " dilation=" << dilation
              << " original_ms=" << original_result.mean_ms
              << " exp2d_ms=" << exp_result.mean_ms
              << " speedup=" << speedup << "x\n";
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const auto backend = parse_backend(argc, argv);
        for (int dilation : {1, 2, 4, 8}) {
            run_case(dilation, backend);
        }
        std::cout << "DepthwiseConv1dModuleExp parity/perf test passed\n";
        return 0;
    } catch (const std::exception & ex) {
        std::cerr << "DepthwiseConv1dModuleExp parity/perf test failed: " << ex.what() << "\n";
        return 1;
    }
}
