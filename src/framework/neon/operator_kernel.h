#pragma once

#include "engine/framework/neon/tensor.h"
#include "kernels.h"
#include "worker_team.h"
#include <stdexcept>

namespace engine::neon::detail {

struct WorkspaceRequirements {
    int frames = 1;
    int channels = 1;
    int kernel_size = 1;
    int panel_rows = 0;
    int attention_rows = 0;
    int quantized_elements = 0;
};

struct BufferView {
    const float *data;
    const Shape *shape;
    const int32_t *indices = nullptr;
};

// Validation, shape propagation and workspace sizing happen on the caller thread.
// compute() only executes validated, allocation-free work and cannot throw.
class Kernel {
public:
    virtual ~Kernel() = default;
    virtual Shape output_shape(const std::vector<Shape> &inputs) const = 0;
    virtual void validate_types(const std::vector<DataType> &inputs) const {
        for (auto type : inputs) {
            if (type != DataType::Float32) {
                throw std::invalid_argument("ARM arithmetic requires Float32 inputs");
            }
        }
    }
    virtual void validate_values(const std::vector<BufferView> &) const {}
    virtual WorkspaceRequirements workspace(const std::vector<Shape> &inputs) const {
        return {};
    }
    virtual bool inplace() const noexcept {
        return false;
    }
    virtual bool supports_frame_output() const noexcept {
        return false;
    }
    virtual void compute(const std::vector<BufferView> &inputs, float *output, const Shape &shape, Workspace &workspace,
                         WorkerContext &worker) const noexcept = 0;
};

} // namespace engine::neon::detail
