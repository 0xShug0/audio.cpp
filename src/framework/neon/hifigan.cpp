#include "engine/framework/neon/blocks.h"

#include <limits>
#include <stdexcept>

namespace engine::neon {

ExecutionPlan::Value build_hifigan(ExecutionPlan &plan, ExecutionPlan::Value input,
    std::optional<ExecutionPlan::Value> conditioning, const HifiGanConfig &c, const HifiGanWeights &w) {
    const size_t stages = c.upsample_rates.size(), kernels = c.residual_kernels.size();
    if (c.input_channels <= 0 || c.initial_channels <= 0 || c.output_channels <= 0 ||
        stages == 0 || kernels == 0 || c.upsample_kernels.size() != stages ||
        c.residual_dilations.size() != kernels || w.upsample.size() != stages ||
        w.residual.size() != stages * kernels || bool(conditioning) != (c.conditioning_channels > 0)) {
        throw std::invalid_argument("NEON HiFi-GAN configuration or stage count mismatch");
    }
    auto x = plan.add(Conv1d({c.input_channels, c.initial_channels, 7, 1, 1, 3, 3}, w.pre.weight, w.pre.bias), {input});
    if (conditioning) {
        auto g = plan.add(Conv1d({c.conditioning_channels, c.initial_channels, 1},
            w.conditioning.weight, w.conditioning.bias), {*conditioning});
        x = plan.add(BroadcastAdd(), {x, g});
    }
    int channels = c.initial_channels;
    for (size_t stage = 0; stage < stages; ++stage) {
        const int output = channels / 2, kernel = c.upsample_kernels[stage], rate = c.upsample_rates[stage];
        if (channels % 2 || output == 0 || rate <= 0 || kernel < rate)
            throw std::invalid_argument("NEON HiFi-GAN upsample geometry invalid");
        x = plan.add(Activation(ActivationKind::LeakyReLU, c.negative_slope), {x});
        x = build_conv_transpose1d(plan, x, {channels, output, kernel, rate, (kernel - rate) / 2},
            w.upsample[stage].weight, w.upsample[stage].bias);
        std::optional<ExecutionPlan::Value> sum;
        for (size_t k = 0; k < kernels; ++k) {
            auto residual = x;
            const auto &block = w.residual[stage * kernels + k];
            const auto &dilations = c.residual_dilations[k];
            const int width = c.residual_kernels[k];
            if (width <= 0 || width % 2 == 0 || dilations.empty() || block.first.size() != dilations.size() ||
                block.second.size() != (c.paired_convolutions ? dilations.size() : 0))
                throw std::invalid_argument("NEON HiFi-GAN residual configuration invalid");
            for (size_t layer = 0; layer < dilations.size(); ++layer) {
                const int dilation = dilations[layer];
                if (dilation <= 0 || int64_t(width - 1) * dilation > std::numeric_limits<int>::max())
                    throw std::invalid_argument("NEON HiFi-GAN residual dilation invalid");
                const int padding = (width - 1) * dilation / 2;
                auto h = plan.add(Activation(ActivationKind::LeakyReLU, c.negative_slope), {residual});
                h = plan.add(Conv1d({output, output, width, 1, dilation, padding, padding},
                    block.first[layer].weight, block.first[layer].bias), {h});
                if (c.paired_convolutions) {
                    h = plan.add(Activation(ActivationKind::LeakyReLU, c.negative_slope), {h});
                    h = plan.add(Conv1d({output, output, width, 1, 1, width / 2, width / 2},
                        block.second[layer].weight, block.second[layer].bias), {h});
                }
                residual = plan.add(ScaleAdd({1.0F}), {residual, h});
            }
            sum = sum ? plan.add(ScaleAdd({1.0F}), {*sum, residual}) : residual;
        }
        x = plan.add(Scale(1.0F / static_cast<float>(kernels)), {*sum});
        channels = output;
    }
    x = plan.add(Activation(ActivationKind::LeakyReLU, c.output_negative_slope), {x});
    x = plan.add(Conv1d({channels, c.output_channels, 7, 1, 1, 3, 3}, w.post.weight, w.post.bias), {x});
    return plan.add(Activation(ActivationKind::Tanh), {x});
}

} // namespace engine::neon
