#include "engine/framework/neon/operators.h"
#include "operator_kernel.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <variant>

namespace engine::neon {
namespace {

void require_size(const std::vector<float> &values, size_t size, bool optional = false) {
    if (values.size() != size && !(optional && values.empty())) {
        throw std::invalid_argument("ARM operator weight size mismatch");
    }
}

const Shape &unary_shape(const std::vector<Shape> &inputs, int rank = 0) {
    if (inputs.size() != 1 || (rank && inputs[0].rank != static_cast<size_t>(rank))) {
        throw std::invalid_argument("ARM operator input count or rank mismatch");
    }
    if (inputs[0].layout != Layout::ChannelFirst) {
        throw std::invalid_argument("ARM operator requires channel-first input");
    }
    return inputs[0];
}

int output_extent(int input, int kernel, int stride, int dilation, int left, int right) {
    const int64_t extent =
        static_cast<int64_t>(input) + left + right - static_cast<int64_t>(dilation) * (kernel - 1) - 1;
    if (extent < 0 || extent / stride + 1 > std::numeric_limits<int>::max() - 15) {
        throw std::invalid_argument("ARM convolution or pooling output extent invalid");
    }
    return static_cast<int>(extent / stride + 1);
}

class LinearKernel final : public detail::Kernel {
public:
    LinearKernel(int input, int output, std::vector<float> weight, std::vector<float> bias)
        : input_(input), output_(output), weight_(std::move(weight)), bias_(std::move(bias)) {
        require_size(weight_, Shape{output, input}.elements());
        require_size(bias_, output, true);
    }
    Shape output_shape(const std::vector<Shape> &inputs) const override {
        if (inputs.size() != 1 || inputs[0].rank != 2 || inputs[0].dimensions[0] != input_) {
            throw std::invalid_argument("ARM Linear input shape mismatch");
        }
        return Shape({output_, inputs[0].dimensions[1]}, inputs[0].layout);
    }
    bool supports_frame_output() const noexcept override {
        return true;
    }
    void compute(const std::vector<detail::BufferView> &inputs, float *output, const Shape &shape, detail::Workspace &,
                 detail::WorkerContext &worker) const noexcept override {
        detail::linear(weight_.data(), bias_.empty() ? nullptr : bias_.data(), inputs[0].data, output, input_, output_,
                       shape.dimensions[1], inputs[0].shape->layout == Layout::FrameMajor,
                       shape.layout == Layout::FrameMajor, worker);
    }

private:
    int input_, output_;
    std::vector<float> weight_, bias_;
};

class QuantizedLinearKernel final : public detail::Kernel {
public:
    QuantizedLinearKernel(int input, int output, BlockInt8Weights weight, std::vector<float> bias)
        : input_(input), output_(output), bias_(std::move(bias)) {
        const size_t count = Shape{output, input}.elements();
        if (input % 32 || weight.values.size() != count || weight.scales.size() != count / 32) {
            throw std::invalid_argument("ARM QuantizedLinear block shape mismatch");
        }
        for (float scale : weight.scales) {
            if (!std::isfinite(scale) || scale < 0) {
                throw std::invalid_argument("ARM QuantizedLinear invalid scale");
            }
        }
        require_size(bias_, output, true);
        weight_ = detail::pack_int8_rows(weight, input, output);
    }
    Shape output_shape(const std::vector<Shape> &inputs) const override {
        if (inputs.size() != 1 || inputs[0].rank != 2 || inputs[0].dimensions[0] != input_) {
            throw std::invalid_argument("ARM QuantizedLinear input shape mismatch");
        }
        return Shape({output_, inputs[0].dimensions[1]}, inputs[0].layout);
    }
    detail::WorkspaceRequirements workspace(const std::vector<Shape> &inputs) const override {
        detail::WorkspaceRequirements result;
        result.quantized_elements = static_cast<int>(inputs[0].elements());
        return result;
    }
    bool supports_frame_output() const noexcept override {
        return true;
    }
    void compute(const std::vector<detail::BufferView> &inputs, float *output, const Shape &shape,
                 detail::Workspace &workspace, detail::WorkerContext &worker) const noexcept override {
        detail::quantized_linear(weight_, bias_.empty() ? nullptr : bias_.data(), inputs[0].data, output, input_,
                                 output_, shape.dimensions[1], inputs[0].shape->layout == Layout::FrameMajor,
                                 shape.layout == Layout::FrameMajor, workspace, worker);
    }

private:
    int input_, output_;
    detail::PackedInt8Rows weight_;
    std::vector<float> bias_;
};

class EmbeddingKernel final : public detail::Kernel {
public:
    EmbeddingKernel(int features, int rows, std::variant<std::vector<float>, BlockInt8Weights> weight)
        : features_(features), rows_(rows), weight_(std::move(weight)) {
        const size_t count = Shape{features, rows}.elements();
        if (const auto *plain = std::get_if<std::vector<float>>(&weight_)) {
            require_size(*plain, count);
        } else {
            const auto &quantized = std::get<BlockInt8Weights>(weight_);
            if (features % 32 || quantized.values.size() != count || quantized.scales.size() != count / 32) {
                throw std::invalid_argument("ARM embedding block layout invalid");
            }
            for (float scale : quantized.scales) {
                if (!std::isfinite(scale) || scale < 0) {
                    throw std::invalid_argument("ARM embedding scale invalid");
                }
            }
        }
    }
    void validate_types(const std::vector<DataType> &inputs) const override {
        if (inputs.size() != 1 || inputs[0] != DataType::Int32) {
            throw std::invalid_argument("ARM embedding expects Int32 index");
        }
    }
    Shape output_shape(const std::vector<Shape> &inputs) const override {
        if (inputs.size() != 1 || inputs[0].elements() != 1) {
            throw std::invalid_argument("ARM embedding expects one index");
        }
        return {features_, 1};
    }
    void validate_values(const std::vector<detail::BufferView> &inputs) const override {
        if (inputs[0].indices[0] < 0 || inputs[0].indices[0] >= rows_) {
            throw std::invalid_argument("ARM embedding index outside table");
        }
    }
    void compute(const std::vector<detail::BufferView> &inputs, float *output, const Shape &, detail::Workspace &,
                 detail::WorkerContext &worker) const noexcept override {
        const size_t row = inputs[0].indices[0];
        int begin, end;
        if (const auto *plain = std::get_if<std::vector<float>>(&weight_)) {
            detail::split_range(features_, worker.index, worker.count, begin, end);
            std::copy(plain->data() + row * features_ + begin, plain->data() + row * features_ + end, output + begin);
        } else {
            const auto &quantized = std::get<BlockInt8Weights>(weight_);
            detail::split_range(features_ / 32, worker.index, worker.count, begin, end);
            for (int block = begin; block < end; ++block) {
                const size_t offset = row * features_ + block * 32;
                const float scale = quantized.scales[row * (features_ / 32) + block];
                for (int i = 0; i < 32; ++i) {
                    output[block * 32 + i] = quantized.values[offset + i] * scale;
                }
            }
        }
    }

private:
    int features_, rows_;
    std::variant<std::vector<float>, BlockInt8Weights> weight_;
};

class Conv1dKernel final : public detail::Kernel {
public:
    Conv1dKernel(Conv1dConfig config, const std::vector<float> &weight, const std::vector<float> &bias)
        : config_(config) {
        if (config.input_channels <= 0 || config.output_channels <= 0 || config.kernel_size <= 0 ||
            config.stride <= 0 || config.dilation <= 0 || config.padding_left < 0 || config.padding_right < 0) {
            throw std::invalid_argument("ARM Conv1d configuration invalid");
        }
        require_size(weight, Shape{config.input_channels, config.output_channels, config.kernel_size}.elements());
        require_size(bias, config.output_channels, true);
        weights_ = detail::pack_conv_weights(weight.data(), bias.empty() ? nullptr : bias.data(), config.input_channels,
                                             config.output_channels, config.kernel_size);
        same_padding_ = config.stride == 1 && config.dilation == 1 && config.kernel_size % 2 == 1 &&
                        config.padding_left == config.kernel_size / 2 && config.padding_right == config.kernel_size / 2;
    }
    Shape output_shape(const std::vector<Shape> &inputs) const override {
        const auto &shape = unary_shape(inputs, 2);
        if (shape.dimensions[0] != config_.input_channels) {
            throw std::invalid_argument("ARM Conv1d channels mismatch");
        }
        return {config_.output_channels, output_extent(shape.dimensions[1], config_.kernel_size, config_.stride,
                                                       config_.dilation, config_.padding_left, config_.padding_right)};
    }
    detail::WorkspaceRequirements workspace(const std::vector<Shape> &inputs) const override {
        return {inputs[0].dimensions[1], config_.input_channels, same_padding_ ? config_.kernel_size : 1,
                same_padding_ ? 0 : config_.input_channels * config_.kernel_size};
    }
    bool supports_frame_output() const noexcept override {
        return same_padding_;
    }
    void compute(const std::vector<detail::BufferView> &inputs, float *output, const Shape &shape,
                 detail::Workspace &workspace, detail::WorkerContext &worker) const noexcept override {
        if (same_padding_) {
            detail::convolve_1d(weights_, inputs[0].data, output,
                                shape.layout == Layout::FrameMajor ? config_.output_channels : 0,
                                inputs[0].shape->dimensions[1], workspace, worker);
        } else {
            detail::convolve_1d_strided(weights_, inputs[0].data, inputs[0].shape->dimensions[1], output,
                                        config_.stride, config_.dilation, config_.padding_left, shape.dimensions[1],
                                        workspace, worker);
        }
    }

private:
    Conv1dConfig config_;
    detail::PackedConvWeights weights_;
    bool same_padding_;
};

class Conv2dKernel final : public detail::Kernel {
public:
    Conv2dKernel(Conv2dConfig config, const std::vector<float> &weight, const std::vector<float> &bias)
        : config_(config) {
        if (config.stride_height <= 0 || config.stride_width <= 0 || config.dilation_height <= 0 ||
            config.dilation_width <= 0 || config.padding_height < 0 || config.padding_width < 0) {
            throw std::invalid_argument("ARM Conv2d configuration invalid");
        }
        require_size(
            weight,
            Shape{config.input_channels, config.output_channels, config.kernel_height, config.kernel_width}.elements());
        require_size(bias, config.output_channels, true);
        weights_ = detail::pack_conv_weights(weight.data(), bias.empty() ? nullptr : bias.data(),
                                             config.input_channels * config.kernel_height * config.kernel_width,
                                             config.output_channels, 1);
    }
    Shape output_shape(const std::vector<Shape> &inputs) const override {
        const auto &shape = unary_shape(inputs, 3);
        if (shape.dimensions[0] != config_.input_channels) {
            throw std::invalid_argument("ARM Conv2d channels mismatch");
        }
        return {config_.output_channels,
                output_extent(shape.dimensions[1], config_.kernel_height, config_.stride_height,
                              config_.dilation_height, config_.padding_height, config_.padding_height),
                output_extent(shape.dimensions[2], config_.kernel_width, config_.stride_width, config_.dilation_width,
                              config_.padding_width, config_.padding_width)};
    }
    detail::WorkspaceRequirements workspace(const std::vector<Shape> &) const override {
        return {1, 1, 1, config_.input_channels * config_.kernel_height * config_.kernel_width};
    }
    void compute(const std::vector<detail::BufferView> &inputs, float *output, const Shape &,
                 detail::Workspace &workspace, detail::WorkerContext &worker) const noexcept override {
        detail::convolve_2d(weights_, config_, inputs[0].data, inputs[0].shape->dimensions[1],
                            inputs[0].shape->dimensions[2], output, workspace, worker);
    }

private:
    Conv2dConfig config_;
    detail::PackedConvWeights weights_;
};

class DepthwiseConv1dKernel final : public detail::Kernel {
public:
    DepthwiseConv1dKernel(int channels, int kernel, std::vector<float> weight, std::vector<float> bias)
        : channels_(channels), kernel_size_(kernel), weight_(std::move(weight)), bias_(std::move(bias)) {
        if (channels <= 0 || kernel <= 0 || kernel % 2 == 0) {
            throw std::invalid_argument("ARM depthwise configuration invalid");
        }
        require_size(weight_, static_cast<size_t>(channels) * kernel);
        require_size(bias_, channels);
    }
    Shape output_shape(const std::vector<Shape> &inputs) const override {
        const auto &shape = unary_shape(inputs, 2);
        if (shape.dimensions[0] != channels_) {
            throw std::invalid_argument("ARM depthwise channels mismatch");
        }
        return shape;
    }
    void compute(const std::vector<detail::BufferView> &inputs, float *output, const Shape &shape, detail::Workspace &,
                 detail::WorkerContext &worker) const noexcept override {
        detail::convolve_depthwise_1d(inputs[0].data, weight_.data(), bias_.data(), output, channels_,
                                      shape.dimensions[1], kernel_size_, worker);
    }

private:
    int channels_, kernel_size_;
    std::vector<float> weight_, bias_;
};

class LayerNormKernel final : public detail::Kernel {
public:
    LayerNormKernel(std::vector<float> weight, std::vector<float> bias, float epsilon)
        : weight_(std::move(weight)), bias_(std::move(bias)), epsilon_(epsilon) {
        if (weight_.empty() || !std::isfinite(epsilon) || epsilon <= 0) {
            throw std::invalid_argument("ARM LayerNorm configuration invalid");
        }
        if (bias_.empty()) {
            bias_.assign(weight_.size(), 0.0F);
        }
        require_size(bias_, weight_.size());
    }
    Shape output_shape(const std::vector<Shape> &inputs) const override {
        const auto &shape = unary_shape(inputs, 2);
        if (static_cast<size_t>(shape.dimensions[0]) != weight_.size()) {
            throw std::invalid_argument("ARM LayerNorm channels mismatch");
        }
        return shape;
    }
    detail::WorkspaceRequirements workspace(const std::vector<Shape> &inputs) const override {
        return {inputs[0].dimensions[1], 1, 1, 0};
    }
    void compute(const std::vector<detail::BufferView> &inputs, float *output, const Shape &shape,
                 detail::Workspace &workspace, detail::WorkerContext &worker) const noexcept override {
        detail::normalize_channels(inputs[0].data, weight_.data(), bias_.data(), output, shape.dimensions[0],
                                   shape.dimensions[1], epsilon_, workspace, worker);
    }

private:
    std::vector<float> weight_, bias_;
    float epsilon_;
};

class ActivationKernel final : public detail::Kernel {
public:
    ActivationKernel(ActivationKind kind, float slope) : kind_(kind), slope_(slope) {
        if ((kind != ActivationKind::SiLU && kind != ActivationKind::GELU && kind != ActivationKind::LeakyReLU) ||
            !std::isfinite(slope)) {
            throw std::invalid_argument("ARM activation configuration invalid");
        }
    }
    Shape output_shape(const std::vector<Shape> &inputs) const override {
        return unary_shape(inputs);
    }
    bool inplace() const noexcept override {
        return true;
    }
    void compute(const std::vector<detail::BufferView> &inputs, float *output, const Shape &shape, detail::Workspace &,
                 detail::WorkerContext &worker) const noexcept override {
        const int count = static_cast<int>(shape.elements());
        int begin, end;
        detail::split_range(kind_ == ActivationKind::GELU ? count / 4 : count, worker.index, worker.count, begin, end);
        if (kind_ == ActivationKind::GELU) {
            begin *= 4;
            end = worker.index == worker.count - 1 ? count : end * 4;
        }
        if (inputs[0].data != output) {
            std::copy(inputs[0].data + begin, inputs[0].data + end, output + begin);
        }
        if (kind_ == ActivationKind::GELU) {
            detail::apply_gelu(output, begin, end);
        } else if (kind_ == ActivationKind::SiLU) {
            detail::apply_silu(output, begin, end);
        } else {
            detail::apply_leaky_relu_clamp(output, begin, end, slope_, -std::numeric_limits<float>::infinity());
        }
    }

private:
    ActivationKind kind_;
    float slope_;
};

class ClampMinKernel final : public detail::Kernel {
public:
    explicit ClampMinKernel(float minimum) : minimum_(minimum) {
        if (!std::isfinite(minimum)) {
            throw std::invalid_argument("ARM clamp bound invalid");
        }
    }
    Shape output_shape(const std::vector<Shape> &inputs) const override {
        return unary_shape(inputs);
    }
    bool inplace() const noexcept override {
        return true;
    }
    void compute(const std::vector<detail::BufferView> &inputs, float *output, const Shape &shape, detail::Workspace &,
                 detail::WorkerContext &worker) const noexcept override {
        int begin, end;
        detail::split_range(static_cast<int>(shape.elements()), worker.index, worker.count, begin, end);
        for (int index = begin; index < end; ++index) {
            output[index] = std::max(inputs[0].data[index], minimum_);
        }
    }

private:
    float minimum_;
};

class MaxPool2dKernel final : public detail::Kernel {
public:
    MaxPool2dKernel(int kh, int kw, int sh, int sw) : kh_(kh), kw_(kw), sh_(sh), sw_(sw) {
        if (kh <= 0 || kw <= 0 || sh <= 0 || sw <= 0) {
            throw std::invalid_argument("ARM pooling geometry invalid");
        }
    }
    Shape output_shape(const std::vector<Shape> &inputs) const override {
        const auto &shape = unary_shape(inputs, 3);
        return {shape.dimensions[0], output_extent(shape.dimensions[1], kh_, sh_, 1, 0, 0),
                output_extent(shape.dimensions[2], kw_, sw_, 1, 0, 0)};
    }
    void compute(const std::vector<detail::BufferView> &inputs, float *output, const Shape &, detail::Workspace &,
                 detail::WorkerContext &worker) const noexcept override {
        const auto &dims = inputs[0].shape->dimensions;
        detail::pool_max_2d(inputs[0].data, output, dims[0], dims[1], dims[2], kh_, kw_, sh_, sw_, worker);
    }

private:
    int kh_, kw_, sh_, sw_;
};

class ScaleAddKernel final : public detail::Kernel {
public:
    explicit ScaleAddKernel(std::vector<float> scale) : scale_(std::move(scale)) {
        if (scale_.empty()) {
            throw std::invalid_argument("ARM scale cannot be empty");
        }
    }
    Shape output_shape(const std::vector<Shape> &inputs) const override {
        if (inputs.size() != 2 || !(inputs[0] == inputs[1]) || inputs[0].layout != Layout::ChannelFirst ||
            (scale_.size() != 1 && scale_.size() != static_cast<size_t>(inputs[0].dimensions[0]))) {
            throw std::invalid_argument("ARM scaled addition shape mismatch");
        }
        return inputs[0];
    }
    bool inplace() const noexcept override {
        return true;
    }
    void compute(const std::vector<detail::BufferView> &inputs, float *output, const Shape &shape, detail::Workspace &,
                 detail::WorkerContext &worker) const noexcept override {
        const int channels = shape.dimensions[0];
        const size_t spatial = shape.elements() / channels;
        int begin, end;
        detail::split_range(channels, worker.index, worker.count, begin, end);
        for (int channel = begin; channel < end; ++channel) {
            const float scale = scale_[scale_.size() == 1 ? 0 : channel];
            for (size_t index = channel * spatial; index < (channel + 1) * spatial; ++index) {
                output[index] = inputs[0].data[index] + scale * inputs[1].data[index];
            }
        }
    }

private:
    std::vector<float> scale_;
};

class ConcatChannelsKernel final : public detail::Kernel {
public:
    Shape output_shape(const std::vector<Shape> &inputs) const override {
        if (inputs.empty()) {
            throw std::invalid_argument("ARM concatenation requires inputs");
        }
        Shape output = inputs[0];
        output.dimensions[0] = 0;
        for (const auto &shape : inputs) {
            if (shape.layout != Layout::ChannelFirst || shape.rank != output.rank ||
                !std::equal(shape.dimensions.begin() + 1, shape.dimensions.begin() + shape.rank,
                            output.dimensions.begin() + 1) ||
                shape.dimensions[0] > std::numeric_limits<int>::max() - output.dimensions[0]) {
                throw std::invalid_argument("ARM concatenation spatial dimensions mismatch");
            }
            output.dimensions[0] += shape.dimensions[0];
        }
        return output;
    }
    void compute(const std::vector<detail::BufferView> &inputs, float *output, const Shape &, detail::Workspace &,
                 detail::WorkerContext &worker) const noexcept override {
        size_t offset = 0;
        for (const auto &input : inputs) {
            const size_t spatial = input.shape->elements() / input.shape->dimensions[0];
            int begin, end;
            detail::split_range(input.shape->dimensions[0], worker.index, worker.count, begin, end);
            std::copy(input.data + begin * spatial, input.data + end * spatial, output + offset + begin * spatial);
            offset += input.shape->elements();
        }
    }
};

class ConstantChannelsKernel final : public detail::Kernel {
public:
    ConstantChannelsKernel(int channels, float value) : channels_(channels), value_(value) {
        if (channels <= 0 || !std::isfinite(value)) {
            throw std::invalid_argument("ARM constant configuration invalid");
        }
    }
    Shape output_shape(const std::vector<Shape> &inputs) const override {
        Shape shape = unary_shape(inputs);
        shape.dimensions[0] = channels_;
        return shape;
    }
    void compute(const std::vector<detail::BufferView> &, float *output, const Shape &shape, detail::Workspace &,
                 detail::WorkerContext &worker) const noexcept override {
        int begin, end;
        detail::split_range(static_cast<int>(shape.elements()), worker.index, worker.count, begin, end);
        std::fill(output + begin, output + end, value_);
    }

private:
    int channels_;
    float value_;
};


class RotaryKernel final : public detail::Kernel {
public:
    RotaryKernel(int head, int rotary, float theta) : head_(head) {
        if (head <= 0 || rotary <= 0 || rotary > head || rotary % 2 || !std::isfinite(theta) || theta <= 0) {
            throw std::invalid_argument("ARM rotary configuration invalid");
        }
        frequencies_.resize(rotary / 2);
        for (int i = 0; i < rotary / 2; ++i) {
            frequencies_[i] = std::pow(theta, -2.0F * i / rotary);
        }
    }
    void validate_types(const std::vector<DataType> &inputs) const override {
        if (inputs != std::vector<DataType>{DataType::Float32, DataType::Int32}) {
            throw std::invalid_argument("ARM rotary expects Float32 features and Int32 position");
        }
    }
    Shape output_shape(const std::vector<Shape> &inputs) const override {
        if (inputs.size() != 2 || inputs[0].rank != 2 || inputs[0].dimensions[0] % head_ || inputs[1].elements() != 1) {
            throw std::invalid_argument("ARM rotary input shape mismatch");
        }
        return inputs[0];
    }
    void validate_values(const std::vector<detail::BufferView> &inputs) const override {
        if (inputs[1].indices[0] < 0 || int64_t(inputs[1].indices[0]) + inputs[0].shape->dimensions[1] > INT32_MAX) {
            throw std::invalid_argument("ARM rotary position invalid");
        }
    }
    void compute(const std::vector<detail::BufferView> &inputs, float *output, const Shape &shape, detail::Workspace &,
                 detail::WorkerContext &worker) const noexcept override {
        detail::rotary_embedding(inputs[0].data, output, shape.dimensions[0], shape.dimensions[1],
                                 shape.layout == Layout::FrameMajor, head_, frequencies_, inputs[1].indices[0], worker);
    }

private:
    int head_;
    std::vector<float> frequencies_;
};

class AttentionKernel final : public detail::Kernel {
public:
    AttentionKernel(int heads, int kv_heads, int dimension)
        : heads_(heads), kv_heads_(kv_heads), dimension_(dimension) {
        if (heads <= 0 || kv_heads <= 0 || heads % kv_heads || dimension <= 0 || heads > INT32_MAX / dimension) {
            throw std::invalid_argument("ARM attention dimensions invalid");
        }
    }
    void validate_types(const std::vector<DataType> &inputs) const override {
        if (inputs.size() != 3 && inputs.size() != 6) {
            throw std::invalid_argument("ARM attention input count invalid");
        }
        for (size_t i = 0; i < inputs.size(); ++i) {
            if (inputs[i] != (i == 5 ? DataType::Int32 : DataType::Float32)) {
                throw std::invalid_argument("ARM attention input type invalid");
            }
        }
    }
    Shape output_shape(const std::vector<Shape> &inputs) const override {
        if ((inputs.size() != 3 && inputs.size() != 6) || inputs[0].rank != 2 ||
            inputs[0].dimensions[0] != heads_ * dimension_ ||
            !((inputs[1].rank == 2 && inputs[1].layout == Layout::FrameMajor &&
               inputs[1].dimensions[0] == kv_heads_ * dimension_) ||
              (inputs[1].rank == 3 && inputs[1].layout == Layout::ChannelFirst &&
               inputs[1].dimensions[0] == kv_heads_ && inputs[1].dimensions[2] == dimension_)) ||
            !(inputs[1] == inputs[2])) {
            throw std::invalid_argument("ARM attention Q/K/V shape mismatch");
        }
        if (inputs.size() == 6 &&
            (inputs[0].dimensions[1] != 1 || inputs[3].rank != 2 || inputs[3].dimensions[0] != kv_heads_ * dimension_ ||
             inputs[3].dimensions[1] != 1 || !(inputs[3] == inputs[4]) || inputs[5].elements() != 1)) {
            throw std::invalid_argument("ARM cached attention current K/V shape mismatch");
        }
        return inputs[0];
    }
    void validate_values(const std::vector<detail::BufferView> &inputs) const override {
        if (inputs.size() == 6 &&
            (inputs[5].indices[0] < 0 || inputs[5].indices[0] >= inputs[1].shape->dimensions[1])) {
            throw std::invalid_argument("ARM attention position exceeds cache capacity");
        }
    }
    detail::WorkspaceRequirements workspace(const std::vector<Shape> &inputs) const override {
        detail::WorkspaceRequirements result;
        const int64_t rows = int64_t(inputs[1].dimensions[1]) + dimension_;
        if (rows > INT32_MAX) {
            throw std::invalid_argument("ARM attention workspace exceeds index range");
        }
        result.attention_rows = static_cast<int>(rows);
        return result;
    }
    bool supports_frame_output() const noexcept override {
        return true;
    }
    void compute(const std::vector<detail::BufferView> &inputs, float *output, const Shape &shape,
                 detail::Workspace &workspace, detail::WorkerContext &worker) const noexcept override {
        const bool cached = inputs.size() == 6;
        detail::attention(inputs[0].data, inputs[1].data, inputs[2].data, cached ? inputs[3].data : nullptr,
                          cached ? inputs[4].data : nullptr, output, shape.dimensions[1],
                          cached ? inputs[5].indices[0] + 1 : inputs[1].shape->dimensions[1], heads_, kv_heads_,
                          dimension_, inputs[1].shape->dimensions[1], inputs[1].shape->rank == 3,
                          inputs[0].shape->layout == Layout::FrameMajor, shape.layout == Layout::FrameMajor, workspace,
                          worker);
    }

private:
    int heads_, kv_heads_, dimension_;
};

class SwiGLUKernel final : public detail::Kernel {
public:
    Shape output_shape(const std::vector<Shape> &inputs) const override {
        auto shape = unary_shape(inputs, 2);
        if (shape.dimensions[0] % 2) {
            throw std::invalid_argument("ARM SwiGLU requires two equal channel groups");
        }
        shape.dimensions[0] /= 2;
        return shape;
    }
    void compute(const std::vector<detail::BufferView> &inputs, float *output, const Shape &shape, detail::Workspace &,
                 detail::WorkerContext &worker) const noexcept override {
        int begin, end;
        const int count = static_cast<int>(shape.elements());
        detail::split_range(count, worker.index, worker.count, begin, end);
        for (int i = begin; i < end; ++i) {
            const float gate = inputs[0].data[count + i];
            output[i] = inputs[0].data[i] * (gate / (1.0F + std::exp(-gate)));
        }
    }
};

class SplitHeadsKernel final : public detail::Kernel {
public:
    explicit SplitHeadsKernel(int heads) : heads_(heads) {
        if (heads <= 0) {
            throw std::invalid_argument("ARM head count must be positive");
        }
    }
    Shape output_shape(const std::vector<Shape> &inputs) const override {
        if (inputs.size() != 1 || inputs[0].rank != 2 || inputs[0].layout != Layout::FrameMajor ||
            inputs[0].dimensions[0] % heads_) {
            throw std::invalid_argument("ARM head packing shape mismatch");
        }
        return {heads_, inputs[0].dimensions[1], inputs[0].dimensions[0] / heads_};
    }
    void compute(const std::vector<detail::BufferView> &inputs, float *output, const Shape &shape, detail::Workspace &,
                 detail::WorkerContext &worker) const noexcept override {
        const int frames = shape.dimensions[1], dimension = shape.dimensions[2];
        int begin, end;
        detail::split_range(heads_ * frames, worker.index, worker.count, begin, end);
        for (int row = begin; row < end; ++row) {
            const int head = row / frames, frame = row % frames;
            std::copy_n(inputs[0].data + (static_cast<size_t>(frame) * heads_ + head) * dimension, dimension,
                        output + static_cast<size_t>(row) * dimension);
        }
    }

private:
    int heads_;
};

} // namespace

Operator::Operator(std::shared_ptr<const detail::Kernel> kernel) : kernel_(std::move(kernel)) {}
Operator::~Operator() = default;

Linear::Linear(int input, int output, std::vector<float> weight, std::vector<float> bias)
    : Operator(std::make_shared<LinearKernel>(input, output, std::move(weight), std::move(bias))) {}

QuantizedLinear::QuantizedLinear(int input, int output, BlockInt8Weights weight, std::vector<float> bias)
    : Operator(std::make_shared<QuantizedLinearKernel>(input, output, std::move(weight), std::move(bias))) {}
Embedding::Embedding(int features, int rows, std::vector<float> weight)
    : Operator(std::make_shared<EmbeddingKernel>(features, rows, std::move(weight))) {}
Embedding::Embedding(int features, int rows, BlockInt8Weights weight)
    : Operator(std::make_shared<EmbeddingKernel>(features, rows, std::move(weight))) {}
Conv1d::Conv1d(Conv1dConfig config, const std::vector<float> &weight, const std::vector<float> &bias)
    : Operator(std::make_shared<Conv1dKernel>(config, weight, bias)) {}
Conv2d::Conv2d(Conv2dConfig config, const std::vector<float> &weight, const std::vector<float> &bias)
    : Operator(std::make_shared<Conv2dKernel>(config, weight, bias)) {}
DepthwiseConv1d::DepthwiseConv1d(int channels, int kernel, const std::vector<float> &weight,
                                 const std::vector<float> &bias)
    : Operator(std::make_shared<DepthwiseConv1dKernel>(channels, kernel, weight, bias)) {}
LayerNorm::LayerNorm(const std::vector<float> &weight, const std::vector<float> &bias, float epsilon)
    : Operator(std::make_shared<LayerNormKernel>(weight, bias, epsilon)) {}
Activation::Activation(ActivationKind kind, float slope) : Operator(std::make_shared<ActivationKernel>(kind, slope)) {}
ClampMin::ClampMin(float minimum) : Operator(std::make_shared<ClampMinKernel>(minimum)) {}
MaxPool2d::MaxPool2d(int kh, int kw, int sh, int sw) : Operator(std::make_shared<MaxPool2dKernel>(kh, kw, sh, sw)) {}
ScaleAdd::ScaleAdd(const std::vector<float> &scale) : Operator(std::make_shared<ScaleAddKernel>(scale)) {}
ConcatChannels::ConcatChannels() : Operator(std::make_shared<ConcatChannelsKernel>()) {}
ConstantChannels::ConstantChannels(int channels, float value)
    : Operator(std::make_shared<ConstantChannelsKernel>(channels, value)) {}
RotaryEmbedding::RotaryEmbedding(int head, int rotary, float theta)
    : Operator(std::make_shared<RotaryKernel>(head, rotary, theta)) {}
Attention::Attention(int heads, int kv_heads, int dimension)
    : Operator(std::make_shared<AttentionKernel>(heads, kv_heads, dimension)) {}
SwiGLU::SwiGLU() : Operator(std::make_shared<SwiGLUKernel>()) {}
SplitHeads::SplitHeads(int heads) : Operator(std::make_shared<SplitHeadsKernel>(heads)) {}

} // namespace engine::neon
