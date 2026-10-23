#include "engine/framework/neon/blocks.h"

#include <limits>
#include <optional>
#include <stdexcept>

namespace engine::neon {
namespace {

ExecutionPlan::Value snake_residual(ExecutionPlan &plan, ExecutionPlan::Value x, int channels,
    int width, const std::vector<int> &dilations, float epsilon, const SnakeResidualWeights &w) {
    if (width <= 0 || width % 2 == 0 || dilations.empty() || w.first.size() != dilations.size() ||
        w.second.size() != dilations.size() || w.alpha_first.size() != dilations.size() ||
        w.alpha_second.size() != dilations.size())
        throw std::invalid_argument("NEON HiFT residual geometry or weights invalid");
    for (size_t i = 0; i < dilations.size(); ++i) {
        if (dilations[i] <= 0 || int64_t(width - 1) * dilations[i] > std::numeric_limits<int>::max())
            throw std::invalid_argument("NEON HiFT residual dilation invalid");
        const int padding = (width - 1) * dilations[i] / 2;
        auto h = plan.add(Snake(w.alpha_first[i], epsilon), {x});
        h = plan.add(Conv1d({channels, channels, width, 1, dilations[i], padding, padding},
            w.first[i].weight, w.first[i].bias), {h});
        h = plan.add(Snake(w.alpha_second[i], epsilon), {h});
        h = plan.add(Conv1d({channels, channels, width, 1, 1, width / 2, width / 2},
            w.second[i].weight, w.second[i].bias), {h});
        x = plan.add(ScaleAdd({1.0F}), {x, h});
    }
    return x;
}

} // namespace

ExecutionPlan::Value build_hift(ExecutionPlan &plan, ExecutionPlan::Value mel,
    ExecutionPlan::Value source_stft, const HiftConfig &c, const HiftWeights &w) {
    const size_t stages = c.upsample_rates.size(), kernels = c.residual_kernels.size();
    if (c.input_channels <= 0 || c.initial_channels <= 0 || c.source_channels <= 0 || c.output_channels <= 0 ||
        c.pre_kernel <= 0 || c.pre_kernel % 2 == 0 || c.post_kernel <= 0 || c.post_kernel % 2 == 0 ||
        stages == 0 || kernels == 0 || c.upsample_kernels.size() != stages ||
        c.residual_dilations.size() != kernels || c.source_residual_kernels.size() != stages ||
        c.source_residual_dilations.size() != stages || w.upsample.size() != stages ||
        w.source_down.size() != stages || w.source_residual.size() != stages || w.residual.size() != stages * kernels)
        throw std::invalid_argument("NEON HiFT configuration or stage count mismatch");
    auto x = plan.add(Conv1d({c.input_channels, c.initial_channels, c.pre_kernel, 1, 1,
        c.pre_kernel / 2, c.pre_kernel / 2}, w.pre.weight, w.pre.bias), {mel});
    int channels = c.initial_channels;
    for (size_t stage = 0; stage < stages; ++stage) {
        const int output = channels / 2, width = c.upsample_kernels[stage], rate = c.upsample_rates[stage];
        if (channels % 2 || output == 0 || rate <= 0 || width < rate)
            throw std::invalid_argument("NEON HiFT upsample geometry invalid");
        x = plan.add(Activation(ActivationKind::LeakyReLU, c.negative_slope), {x});
        x = build_conv_transpose1d(plan, x, {channels, output, width, rate, (width - rate) / 2},
            w.upsample[stage].weight, w.upsample[stage].bias);
        if (stage + 1 == stages) x = plan.add(ReflectPad1d(1, 0), {x});
        int stride = 1;
        for (size_t next = stage + 1; next < stages; ++next) {
            if (c.upsample_rates[next] <= 0 || c.upsample_rates[next] > std::numeric_limits<int>::max() / stride)
                throw std::invalid_argument("NEON HiFT source stride invalid");
            stride *= c.upsample_rates[next];
        }
        if (stride > std::numeric_limits<int>::max() / 2)
            throw std::invalid_argument("NEON HiFT source kernel too large");
        auto source = plan.add(Conv1d({c.source_channels, output, stride == 1 ? 1 : stride * 2,
            stride, 1, stride == 1 ? 0 : stride / 2, stride == 1 ? 0 : stride / 2},
            w.source_down[stage].weight, w.source_down[stage].bias), {source_stft});
        source = snake_residual(plan, source, output, c.source_residual_kernels[stage],
            c.source_residual_dilations[stage], c.snake_epsilon, w.source_residual[stage]);
        x = plan.add(ScaleAdd({1.0F}), {x, source});
        std::optional<ExecutionPlan::Value> sum;
        for (size_t k = 0; k < kernels; ++k) {
            auto residual = snake_residual(plan, x, output, c.residual_kernels[k],
                c.residual_dilations[k], c.snake_epsilon, w.residual[stage * kernels + k]);
            sum = sum ? plan.add(ScaleAdd({1.0F}), {*sum, residual}) : residual;
        }
        x = plan.add(Scale(1.0F / static_cast<float>(kernels)), {*sum});
        channels = output;
    }
    x = plan.add(Activation(ActivationKind::LeakyReLU, 0.01F), {x});
    return plan.add(Conv1d({channels, c.output_channels, c.post_kernel, 1, 1,
        c.post_kernel / 2, c.post_kernel / 2}, w.post.weight, w.post.bias), {x});
}

} // namespace engine::neon
