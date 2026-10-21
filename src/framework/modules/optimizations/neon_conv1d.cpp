#include "engine/framework/modules/optimizations/neon_conv1d.h"
#include "engine/framework/core/backend.h"

#include <stdexcept>

#ifdef ENGINE_HAVE_NEON
#include "engine/framework/neon/execution.h"
#include <cstring>
#include <limits>
#include <optional>
#endif

namespace engine::modules {

struct NeonConv1dGraph::Impl {
#ifdef ENGINE_HAVE_NEON
    struct Convolution {
        std::unique_ptr<neon::Conv1d> operation;
        Impl * owner;
        int threads;

        static void compute(ggml_tensor * output, int ith, int, void * userdata) {
            // GGML dispatches custom callbacks on its worker team; the NEON
            // plan owns its own team and must be entered exactly once.
            if (ith != 0) return;
            auto & self = *static_cast<Convolution *>(userdata);
            const auto * input = output->src[0];
            self.owner->plan.replace(*self.owner->output, *self.operation);
            const auto & result = self.owner->plan.run({neon::TensorView(
                static_cast<const float *>(input->data), ggml_nelements(input),
                {static_cast<int>(input->ne[1]), static_cast<int>(input->ne[0])})}, self.threads);
            std::memcpy(output->data, result.values.data(), ggml_nbytes(output));
        }
    };
    neon::ExecutionPlan plan;
    neon::ExecutionPlan::Value input = plan.input();
    std::optional<neon::ExecutionPlan::Value> output;
    std::vector<std::unique_ptr<Convolution>> convolutions;
#endif
    int threads;
};

NeonConv1dGraph::NeonConv1dGraph(int threads) : impl_(std::make_unique<Impl>()) {
#ifdef ENGINE_HAVE_NEON
    if (!neon::available()) throw std::runtime_error("NEON convolution requires an AArch64 NEON CPU");
    if (threads < 1) throw std::invalid_argument("NEON convolution requires positive threads");
    impl_->threads = threads;
#else
    (void)threads;
    throw std::runtime_error("NEON convolution requires ENGINE_ENABLE_NEON=ON");
#endif
}

NeonConv1dGraph::~NeonConv1dGraph() = default;

core::TensorValue NeonConv1dGraph::build(core::ModuleBuildContext & ctx, const core::TensorValue & input,
                                        const Conv1dWeights & weights, const Conv1dConfig & config) {
#ifdef ENGINE_HAVE_NEON
    core::validate_shape(input, core::TensorShape::from_dims({1, config.in_channels, input.shape.last_dim()}),
                         "NEON convolution input");
    core::validate_shape(weights.weight,
        core::TensorShape::from_dims({config.out_channels, config.in_channels, config.kernel_size}),
        "NEON convolution weight");
    if (config.use_bias && !weights.bias) throw std::invalid_argument("NEON convolution requires bias weights");
    if (ctx.backend_type != core::BackendType::Cpu || input.type != GGML_TYPE_F32 || config.bf16_compute) {
        throw std::invalid_argument("NEON convolution requires CPU F32 activations");
    }
    for (auto dimension : {config.in_channels, config.out_channels, config.kernel_size, input.shape.last_dim()}) {
        if (dimension <= 0 || dimension > std::numeric_limits<int>::max()) {
            throw std::invalid_argument("NEON convolution dimension out of range");
        }
    }
    if (config.stride <= 0 || config.dilation <= 0 || config.padding < 0) {
        throw std::invalid_argument("NEON convolution invalid configuration");
    }
    const auto extent = input.shape.last_dim() + 2LL * config.padding -
        config.dilation * (config.kernel_size - 1) - 1;
    if (extent < 0) {
        throw std::invalid_argument("NEON convolution invalid extent");
    }
    const auto frames = extent / config.stride + 1;
    if (frames > std::numeric_limits<int>::max()) throw std::invalid_argument("NEON convolution output out of range");
    auto convolution = std::make_unique<Impl::Convolution>();
    convolution->threads = impl_->threads;
    convolution->owner = impl_.get();
    const neon::Conv1dConfig parameters{static_cast<int>(config.in_channels), static_cast<int>(config.out_channels),
        static_cast<int>(config.kernel_size), config.stride, config.dilation, config.padding, config.padding};
    const auto bias = config.use_bias && weights.bias ? core::read_tensor_float(weights.bias->tensor) : std::vector<float>{};
    convolution->operation = std::make_unique<neon::Conv1d>(
        parameters, core::read_tensor_float(weights.weight.tensor), bias);
    if (!impl_->output) {
        impl_->output = impl_->plan.add(*convolution->operation, {impl_->input});
        impl_->plan.set_output(*impl_->output);
    }
    auto * source = ggml_is_contiguous(input.tensor) ? input.tensor : ggml_cont(ctx.ggml, input.tensor);
    ggml_tensor * args[] = {source};
    auto * output = ggml_custom_4d(ctx.ggml, GGML_TYPE_F32, frames, config.out_channels, 1, 1,
        args, 1, Impl::Convolution::compute, 1, convolution.get());
    ggml_set_name(output, "neon.conv1d");
    impl_->convolutions.push_back(std::move(convolution));
    return core::wrap_tensor(output, core::TensorShape::from_dims({1, config.out_channels, frames}));
#else
    (void)ctx; (void)input; (void)weights; (void)config;
    throw std::runtime_error("NEON convolution requires ENGINE_ENABLE_NEON=ON");
#endif
}

} // namespace engine::modules
