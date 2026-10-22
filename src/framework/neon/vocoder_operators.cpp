#include "engine/framework/neon/blocks.h"
#include "operator_kernel.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace engine::neon {
namespace {

const Shape &sequence(const std::vector<Shape> &inputs, size_t count = 1) {
    if (inputs.size() != count || inputs[0].rank != 2 || inputs[0].layout != Layout::ChannelFirst) {
        throw std::invalid_argument("NEON sequence shape mismatch");
    }
    return inputs[0];
}

class ScaleKernel final : public detail::Kernel {
    float scale_;
public:
    explicit ScaleKernel(float scale) : scale_(scale) {
        if (!std::isfinite(scale)) throw std::invalid_argument("NEON scale must be finite");
    }
    Shape output_shape(const std::vector<Shape> &inputs) const override {
        if (inputs.size() != 1) throw std::invalid_argument("NEON scale expects one input");
        return inputs[0];
    }
    bool inplace() const noexcept override { return true; }
    void compute(const std::vector<detail::BufferView> &inputs, float *out, const Shape &shape,
        detail::Workspace &, detail::WorkerContext &worker) const noexcept override {
        int begin, end;
        detail::split_range(static_cast<int>(shape.elements()), worker.index, worker.count, begin, end);
        for (int i = begin; i < end; ++i) out[i] = inputs[0].data[i] * scale_;
    }
};

class BroadcastAddKernel final : public detail::Kernel {
public:
    Shape output_shape(const std::vector<Shape> &inputs) const override {
        const auto &s = sequence(inputs, 2);
        if (inputs[1].rank != 2 || inputs[1].layout != Layout::ChannelFirst ||
            inputs[1].dimensions[0] != s.dimensions[0] ||
            (inputs[1].dimensions[1] != 1 && inputs[1].dimensions[1] != s.dimensions[1])) {
            throw std::invalid_argument("NEON broadcast addition shape mismatch");
        }
        return s;
    }
    bool inplace() const noexcept override { return true; }
    void compute(const std::vector<detail::BufferView> &inputs, float *out, const Shape &s,
        detail::Workspace &, detail::WorkerContext &worker) const noexcept override {
        int begin, end;
        detail::split_range(s.dimensions[0], worker.index, worker.count, begin, end);
        const int frames = s.dimensions[1], other = inputs[1].shape->dimensions[1];
        for (int c = begin; c < end; ++c)
            for (int t = 0; t < frames; ++t)
                out[c * frames + t] = inputs[0].data[c * frames + t] + inputs[1].data[c * other + (other == 1 ? 0 : t)];
    }
};

class ReflectPadKernel final : public detail::Kernel {
    int left_, right_;
public:
    ReflectPadKernel(int left, int right) : left_(left), right_(right) {
        if (left < 0 || right < 0) throw std::invalid_argument("NEON reflection padding must be nonnegative");
    }
    Shape output_shape(const std::vector<Shape> &inputs) const override {
        auto s = sequence(inputs);
        if (left_ >= s.dimensions[1] || right_ >= s.dimensions[1] ||
            int64_t(s.dimensions[1]) + left_ + right_ > std::numeric_limits<int>::max())
            throw std::invalid_argument("NEON reflection padding exceeds sequence");
        s.dimensions[1] += left_ + right_;
        return s;
    }
    void compute(const std::vector<detail::BufferView> &inputs, float *out, const Shape &s,
        detail::Workspace &, detail::WorkerContext &worker) const noexcept override {
        int begin, end;
        detail::split_range(s.dimensions[0], worker.index, worker.count, begin, end);
        const int frames = inputs[0].shape->dimensions[1];
        for (int c = begin; c < end; ++c) {
            for (int t = 0; t < s.dimensions[1]; ++t) {
                int source = t - left_;
                if (source < 0) source = -source;
                if (source >= frames) source = 2 * frames - 2 - source;
                out[c * s.dimensions[1] + t] = inputs[0].data[c * frames + source];
            }
        }
    }
};

class AdaInKernel final : public detail::Kernel {
    float epsilon_;
public:
    explicit AdaInKernel(float epsilon) : epsilon_(epsilon) {
        if (!std::isfinite(epsilon) || epsilon <= 0) throw std::invalid_argument("NEON normalization epsilon invalid");
    }
    Shape output_shape(const std::vector<Shape> &inputs) const override {
        const auto &s = sequence(inputs, 2);
        if (!(inputs[1] == Shape{2 * s.dimensions[0], 1}))
            throw std::invalid_argument("NEON adaptive normalization affine shape mismatch");
        return s;
    }
    bool inplace() const noexcept override { return true; }
    void compute(const std::vector<detail::BufferView> &inputs, float *out, const Shape &s,
        detail::Workspace &, detail::WorkerContext &worker) const noexcept override {
        int begin, end;
        detail::split_range(s.dimensions[0], worker.index, worker.count, begin, end);
        const int frames = s.dimensions[1];
        for (int c = begin; c < end; ++c) {
            const float *x = inputs[0].data + c * frames;
            double sum = 0;
            for (int t = 0; t < frames; ++t) sum += x[t];
            const float mean = static_cast<float>(sum) / static_cast<float>(frames);
            double variance = 0;
            for (int t = 0; t < frames; ++t) { const float d = x[t] - mean; variance += static_cast<double>(d * d); }
            const float stddev = std::sqrt(static_cast<float>(variance) / static_cast<float>(frames) + epsilon_);
            const float gamma = 1.0F + inputs[1].data[c], beta = inputs[1].data[c + s.dimensions[0]];
            for (int t = 0; t < frames; ++t) out[c * frames + t] = ((x[t] - mean) / stddev) * gamma + beta;
        }
    }
};

class SnakeKernel final : public detail::Kernel {
    std::vector<float> alpha_;
public:
    explicit SnakeKernel(std::vector<float> alpha) : alpha_(std::move(alpha)) {
        for (float a : alpha_) if (!std::isfinite(a) || a == 0) throw std::invalid_argument("NEON Snake alpha invalid");
    }
    Shape output_shape(const std::vector<Shape> &inputs) const override {
        const auto &s = sequence(inputs);
        if (alpha_.size() != size_t(s.dimensions[0])) throw std::invalid_argument("NEON Snake channel mismatch");
        return s;
    }
    bool inplace() const noexcept override { return true; }
    void compute(const std::vector<detail::BufferView> &inputs, float *out, const Shape &s,
        detail::Workspace &, detail::WorkerContext &worker) const noexcept override {
        int begin, end;
        detail::split_range(s.dimensions[0], worker.index, worker.count, begin, end);
        for (int c = begin; c < end; ++c) for (int t = 0; t < s.dimensions[1]; ++t) {
            const int i = c * s.dimensions[1] + t;
            const float x = inputs[0].data[i], a = alpha_[c], v = std::sin(a * x);
            out[i] = x + (v * v) / a;
        }
    }
};

class SnakeEpsilonKernel final : public detail::Kernel {
    std::vector<float> alpha_;
    float epsilon_;
public:
    SnakeEpsilonKernel(std::vector<float> alpha, float epsilon) : alpha_(std::move(alpha)), epsilon_(epsilon) {
        if (!std::isfinite(epsilon) || epsilon < 0) throw std::invalid_argument("NEON Snake epsilon invalid");
        for (float a : alpha_) if (!std::isfinite(a) || (a == 0 && epsilon == 0)) throw std::invalid_argument("NEON Snake alpha invalid");
        for (float a : alpha_) if (a + epsilon == 0) throw std::invalid_argument("NEON Snake denominator invalid");
    }
    Shape output_shape(const std::vector<Shape> &inputs) const override {
        const auto &s = sequence(inputs);
        if (alpha_.size() != size_t(s.dimensions[0])) throw std::invalid_argument("NEON Snake channel mismatch");
        return s;
    }
    bool inplace() const noexcept override { return true; }
    void compute(const std::vector<detail::BufferView> &inputs, float *out, const Shape &s,
        detail::Workspace &, detail::WorkerContext &worker) const noexcept override {
        int begin, end;
        detail::split_range(s.dimensions[0], worker.index, worker.count, begin, end);
        for (int c = begin; c < end; ++c) {
            const float a = alpha_[c];
            if (epsilon_ == 0.0F) {
                for (int t = 0; t < s.dimensions[1]; ++t) {
                    const int i = c * s.dimensions[1] + t;
                    const float x = inputs[0].data[i], v = std::sin(a * x);
                    out[i] = x + (v * v) / a;
                }
            } else {
                const float inverse = 1.0F / (a + epsilon_);
                for (int t = 0; t < s.dimensions[1]; ++t) {
                    const int i = c * s.dimensions[1] + t;
                    const float x = inputs[0].data[i], v = std::sin(a * x);
                    out[i] = x + (v * v) * inverse;
                }
            }
        }
    }
};

// Internal second half of the polyphase transposed convolution. The preceding
// ordinary convolution stores [output_channel * stride + phase, frame].
class PhaseInterleaveKernel final : public detail::Kernel {
    ConvTranspose1dConfig config_;
    int taps_;
    std::vector<float> bias_;
public:
    PhaseInterleaveKernel(ConvTranspose1dConfig config, std::vector<float> bias) : config_(config),
        taps_((config.kernel_size + config.stride - 1) / config.stride), bias_(std::move(bias)) {}
    Shape output_shape(const std::vector<Shape> &inputs) const override {
        const auto &s = sequence(inputs);
        const int64_t frames = int64_t(s.dimensions[1] - taps_) * config_.stride +
            config_.kernel_size - 2LL * config_.padding + config_.output_padding;
        if (s.dimensions[0] != config_.output_channels * config_.stride || frames <= 0 ||
            frames > std::numeric_limits<int>::max()) throw std::invalid_argument("NEON transposed convolution output invalid");
        return {config_.output_channels, static_cast<int>(frames)};
    }
    void compute(const std::vector<detail::BufferView> &inputs, float *out, const Shape &s,
        detail::Workspace &, detail::WorkerContext &worker) const noexcept override {
        int begin, end;
        detail::split_range(s.dimensions[0], worker.index, worker.count, begin, end);
        const int frames = inputs[0].shape->dimensions[1];
        for (int c = begin; c < end; ++c) for (int t = 0; t < s.dimensions[1]; ++t) {
            const int phase = t % config_.stride;
            const int source = t / config_.stride + (phase + config_.padding) / config_.stride;
            out[c * s.dimensions[1] + t] = source < frames ?
                inputs[0].data[(c * config_.stride + phase) * frames + source] : (bias_.empty() ? 0.0F : bias_[c]);
        }
    }
};

class PhaseInterleave final : public Operator {
public:
    PhaseInterleave(ConvTranspose1dConfig config, const std::vector<float> &bias)
        : Operator(std::make_shared<PhaseInterleaveKernel>(config, bias)) {}
};

} // namespace

Scale::Scale(float scale) : Operator(std::make_shared<ScaleKernel>(scale)) {}
BroadcastAdd::BroadcastAdd() : Operator(std::make_shared<BroadcastAddKernel>()) {}
ReflectPad1d::ReflectPad1d(int left, int right) : Operator(std::make_shared<ReflectPadKernel>(left, right)) {}
AdaptiveInstanceNorm::AdaptiveInstanceNorm(float epsilon) : Operator(std::make_shared<AdaInKernel>(epsilon)) {}
Snake::Snake(std::vector<float> alpha) : Operator(std::make_shared<SnakeKernel>(std::move(alpha))) {}
Snake::Snake(std::vector<float> alpha, float epsilon)
    : Operator(std::make_shared<SnakeEpsilonKernel>(std::move(alpha), epsilon)) {}

ExecutionPlan::Value build_conv_transpose1d(ExecutionPlan &plan, ExecutionPlan::Value input,
    ConvTranspose1dConfig c, const std::vector<float> &weight, const std::vector<float> &bias) {
    if (c.input_channels <= 0 || c.output_channels <= 0 || c.kernel_size <= 0 || c.stride <= 0 ||
        c.padding < 0 || c.padding >= c.kernel_size || c.output_padding < 0 || c.output_padding >= c.stride ||
        int64_t(c.output_channels) * c.stride > std::numeric_limits<int>::max() ||
        weight.size() != Shape{c.input_channels, c.output_channels, c.kernel_size}.elements() ||
        (!bias.empty() && bias.size() != size_t(c.output_channels))) {
        throw std::invalid_argument("NEON transposed convolution configuration or weights invalid");
    }
    const int taps = (c.kernel_size + c.stride - 1) / c.stride;
    const int channels = c.output_channels * c.stride;
    std::vector<float> phases(Shape{channels, c.input_channels, taps}.elements(), 0.0F);
    std::vector<float> biases(channels, 0.0F);
    for (int out = 0; out < c.output_channels; ++out) for (int phase = 0; phase < c.stride; ++phase) {
        if (!bias.empty()) biases[out * c.stride + phase] = bias[out];
        for (int in = 0; in < c.input_channels; ++in) for (int tap = 0; tap < taps; ++tap) {
            const int k = (phase + c.padding) % c.stride + (taps - 1 - tap) * c.stride;
            if (k < c.kernel_size) phases[((out * c.stride + phase) * c.input_channels + in) * taps + tap] =
                weight[(in * c.output_channels + out) * c.kernel_size + k];
        }
    }
    auto result = plan.add(Conv1d({c.input_channels, channels, taps, 1, 1, taps - 1, taps - 1}, phases, biases), {input});
    return plan.add(PhaseInterleave(c, bias), {result});
}

} // namespace engine::neon
