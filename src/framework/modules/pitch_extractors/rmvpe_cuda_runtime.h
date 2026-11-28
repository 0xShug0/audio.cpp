#pragma once

#include "ggml-backend.h"
#include <memory>

namespace engine::modules::detail {

// Native 3x3/stride-2 transpose convolution used by the RMVPE U-Net.
void rmvpe_cuda_conv_transpose(ggml_backend_t backend, ggml_tensor * output);

// Borrows weights and fixed-address tensors from the session's PreparedGraph.
// Owns only the captured native recurrence; no per-step host work or allocation.
class RmvpeCudaGru {
public:
    RmvpeCudaGru(ggml_backend_t backend, int device, int frames, int hidden,
        bool reverse, const ggml_tensor * projected, const ggml_tensor * weight,
        const ggml_tensor * bias, ggml_tensor * state, const ggml_tensor * keep,
        ggml_tensor * sequence);
    ~RmvpeCudaGru();
    void compute();
    void prepare_pair(const RmvpeCudaGru & reverse);
    void compute_pair();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace engine::modules::detail
