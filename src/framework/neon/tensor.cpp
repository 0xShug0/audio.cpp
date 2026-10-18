#include "engine/framework/neon/tensor.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace engine::neon {

Shape::Shape(std::initializer_list<int> dims, Layout order) : rank(dims.size()), layout(order) {
    if (rank > dimensions.size()) {
        throw std::invalid_argument("ARM tensor rank exceeds four dimensions");
    }
    std::copy(dims.begin(), dims.end(), dimensions.begin());
    elements();
}

size_t Shape::elements() const {
    if (rank == 0 || rank > dimensions.size() || (layout != Layout::ChannelFirst && layout != Layout::FrameMajor) ||
        (layout == Layout::FrameMajor && rank != 2)) {
        throw std::invalid_argument("ARM tensor layout or rank invalid");
    }
    size_t count = 1;
    for (size_t axis = 0; axis < rank; ++axis) {
        const int dimension = dimensions[axis];
        if (dimension <= 0 || count > static_cast<size_t>(std::numeric_limits<int>::max()) / dimension) {
            throw std::invalid_argument("ARM tensor dimensions exceed the worker index range");
        }
        count *= dimension;
    }
    return count;
}

bool Shape::operator==(const Shape &other) const noexcept {
    return rank == other.rank && layout == other.layout && dimensions == other.dimensions;
}

TensorView::TensorView(const std::vector<float> &values, Shape dimensions)
    : TensorView(values.data(), values.size(), std::move(dimensions)) {}

TensorView::TensorView(const float *values, size_t count, Shape dimensions)
    : data(values), size(count), shape(std::move(dimensions)) {
    if (!data || size != shape.elements()) {
        throw std::invalid_argument("ARM tensor data size does not match shape");
    }
}

TensorView::TensorView(const int32_t *values, size_t count, Shape dimensions)
    : data(nullptr), size(count), shape(std::move(dimensions)), indices(values), type(DataType::Int32) {
    if (!indices || size != shape.elements()) {
        throw std::invalid_argument("ARM integer tensor size does not match shape");
    }
}

} // namespace engine::neon
