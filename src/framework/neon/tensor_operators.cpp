#include "engine/framework/neon/operators.h"
#include "operator_kernel.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace engine::neon {
namespace {

const Shape &input_shape(const std::vector<Shape> &inputs) {
    if (inputs.size() != 1 || inputs[0].layout != Layout::ChannelFirst) {
        throw std::invalid_argument("NEON tensor operator requires one channel-first input");
    }
    return inputs[0];
}

size_t inner_size(const Shape &shape, int axis) {
    size_t size = 1;
    for (size_t i = axis + 1; i < shape.rank; ++i) size *= shape.dimensions[i];
    return size;
}

class ConcatKernel final : public detail::Kernel {
public:
    explicit ConcatKernel(int axis) : axis_(axis) {}
    Shape output_shape(const std::vector<Shape> &inputs) const override {
        if (inputs.empty() || axis_ < 0 || size_t(axis_) >= inputs[0].rank) {
            throw std::invalid_argument("NEON concat axis or input count invalid");
        }
        auto output = inputs[0];
        output.dimensions[axis_] = 0;
        for (const auto &shape : inputs) {
            if (shape.layout != Layout::ChannelFirst || shape.rank != output.rank) {
                throw std::invalid_argument("NEON concat layout or rank mismatch");
            }
            for (size_t i = 0; i < shape.rank; ++i) {
                if (int(i) != axis_ && shape.dimensions[i] != output.dimensions[i]) {
                    throw std::invalid_argument("NEON concat non-axis dimension mismatch");
                }
            }
            if (shape.dimensions[axis_] > std::numeric_limits<int>::max() - output.dimensions[axis_]) {
                throw std::invalid_argument("NEON concat axis overflow");
            }
            output.dimensions[axis_] += shape.dimensions[axis_];
        }
        return output;
    }
    void compute(const std::vector<detail::BufferView> &inputs, float *output, const Shape &shape,
                 detail::Workspace &, detail::WorkerContext &worker) const noexcept override {
        const size_t inner = inner_size(shape, axis_);
        const size_t width = inner * shape.dimensions[axis_];
        const size_t outer = shape.elements() / width;
        for (size_t row = outer * worker.index / worker.count; row < outer * (worker.index + 1) / worker.count; ++row) {
            size_t offset = row * width;
            for (const auto &input : inputs) {
                const size_t count = inner * input.shape->dimensions[axis_];
                std::copy_n(input.data + row * count, count, output + offset);
                offset += count;
            }
        }
    }
private:
    int axis_;
};

class SliceKernel final : public detail::Kernel {
public:
    SliceKernel(int axis, int start, int length) : axis_(axis), start_(start), length_(length) {}
    Shape output_shape(const std::vector<Shape> &inputs) const override {
        auto output = input_shape(inputs);
        if (axis_ < 0 || size_t(axis_) >= output.rank || start_ < 0 || length_ <= 0 ||
            start_ > output.dimensions[axis_] || length_ > output.dimensions[axis_] - start_) {
            throw std::invalid_argument("NEON slice bounds invalid");
        }
        output.dimensions[axis_] = length_;
        return output;
    }
    void compute(const std::vector<detail::BufferView> &inputs, float *output, const Shape &shape,
                 detail::Workspace &, detail::WorkerContext &worker) const noexcept override {
        const size_t inner = inner_size(shape, axis_);
        const size_t count = inner * length_;
        const size_t source_width = inner * inputs[0].shape->dimensions[axis_];
        const size_t outer = shape.elements() / count;
        for (size_t row = outer * worker.index / worker.count; row < outer * (worker.index + 1) / worker.count; ++row) {
            std::copy_n(inputs[0].data + row * source_width + start_ * inner, count, output + row * count);
        }
    }
private:
    int axis_, start_, length_;
};

class ReshapeKernel final : public detail::Kernel {
public:
    explicit ReshapeKernel(Shape shape) : shape_(shape) { shape_.elements(); }
    Shape output_shape(const std::vector<Shape> &inputs) const override {
        if (input_shape(inputs).elements() != shape_.elements() || shape_.layout != Layout::ChannelFirst) {
            throw std::invalid_argument("NEON reshape element count or layout mismatch");
        }
        return shape_;
    }
    void compute(const std::vector<detail::BufferView> &inputs, float *output, const Shape &shape,
                 detail::Workspace &, detail::WorkerContext &worker) const noexcept override {
        const auto begin = shape.elements() * worker.index / worker.count;
        const auto end = shape.elements() * (worker.index + 1) / worker.count;
        std::copy(inputs[0].data + begin, inputs[0].data + end, output + begin);
    }
private:
    Shape shape_;
};

class Transpose2dKernel final : public detail::Kernel {
public:
    Shape output_shape(const std::vector<Shape> &inputs) const override {
        const auto &shape = input_shape(inputs);
        if (shape.rank != 2) throw std::invalid_argument("NEON matrix transpose requires rank two");
        return Shape({shape.dimensions[1], shape.dimensions[0]});
    }
    void compute(const std::vector<detail::BufferView> &inputs, float *output, const Shape &shape,
                 detail::Workspace &, detail::WorkerContext &worker) const noexcept override {
        const int rows = shape.dimensions[0], columns = shape.dimensions[1];
        int begin, end;
        detail::split_range(rows, worker.index, worker.count, begin, end);
        for (int row = begin; row < end; ++row) {
            for (int col = 0; col < columns; ++col) output[row * columns + col] = inputs[0].data[col * rows + row];
        }
    }
};

class ClampKernel final : public detail::Kernel {
public:
    ClampKernel(float minimum, float maximum) : minimum_(minimum), maximum_(maximum) {
        if (std::isnan(minimum) || std::isnan(maximum) || minimum > maximum) {
            throw std::invalid_argument("NEON clamp bounds invalid");
        }
    }
    Shape output_shape(const std::vector<Shape> &inputs) const override { return input_shape(inputs); }
    bool inplace() const noexcept override { return true; }
    void compute(const std::vector<detail::BufferView> &inputs, float *output, const Shape &shape,
                 detail::Workspace &, detail::WorkerContext &worker) const noexcept override {
        const auto begin = shape.elements() * worker.index / worker.count;
        const auto end = shape.elements() * (worker.index + 1) / worker.count;
        for (auto i = begin; i < end; ++i) output[i] = std::clamp(inputs[0].data[i], minimum_, maximum_);
    }
private:
    float minimum_, maximum_;
};

class QuantizeDequantizeKernel final : public detail::Kernel {
public:
    QuantizeDequantizeKernel(float scale, int zero_point, int minimum, int maximum)
        : scale_(scale), minimum_(float(minimum) - float(zero_point)), maximum_(float(maximum) - float(zero_point)) {
        if (!std::isfinite(scale) || scale <= 0 || !std::isfinite(1.0F / scale) || minimum > maximum ||
            zero_point < minimum || zero_point > maximum) {
            throw std::invalid_argument("NEON quantization parameters invalid");
        }
    }
    Shape output_shape(const std::vector<Shape> &inputs) const override { return input_shape(inputs); }
    bool inplace() const noexcept override { return true; }
    void compute(const std::vector<detail::BufferView> &inputs, float *output, const Shape &shape,
                 detail::Workspace &, detail::WorkerContext &worker) const noexcept override {
        const auto begin = shape.elements() * worker.index / worker.count;
        const auto end = shape.elements() * (worker.index + 1) / worker.count;
        detail::quantize_dequantize(inputs[0].data + begin, output + begin, end - begin, scale_, minimum_, maximum_);
    }
private:
    float scale_, minimum_, maximum_;
};

} // namespace

Concat::Concat(int axis) : Operator(std::make_shared<ConcatKernel>(axis)) {}
Slice::Slice(int axis, int start, int length) : Operator(std::make_shared<SliceKernel>(axis, start, length)) {}
Reshape::Reshape(Shape shape) : Operator(std::make_shared<ReshapeKernel>(shape)) {}
Transpose2d::Transpose2d() : Operator(std::make_shared<Transpose2dKernel>()) {}
Clamp::Clamp(float minimum, float maximum) : Operator(std::make_shared<ClampKernel>(minimum, maximum)) {}
QuantizeDequantize::QuantizeDequantize(float scale, int zero_point, int minimum, int maximum)
    : Operator(std::make_shared<QuantizeDequantizeKernel>(scale, zero_point, minimum, maximum)) {}
} // namespace engine::neon
