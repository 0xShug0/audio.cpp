#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"

#include <memory>
#include <vector>

namespace engine::models::auk {

class VAEEncoderRuntime {
public:
    VAEEncoderRuntime(core::ExecutionContext & execution, const assets::TensorSource & source, int64_t samples);
    ~VAEEncoderRuntime();
    VAEEncoderRuntime(const VAEEncoderRuntime &) = delete;
    VAEEncoderRuntime & operator=(const VAEEncoderRuntime &) = delete;

    void prepare(int64_t samples);
    int64_t frames() const;
    // Mono 24 kHz input, noise in [1, 64, frames]; normalized output is [1, frames, 64].
    // Optional statistics capture is [1, 128, frames], before sampling.
    std::vector<float> encode(const std::vector<float> & audio, const std::vector<float> & noise,
                              std::vector<float> * statistics = nullptr);

private:
    struct State;
    std::unique_ptr<State> state_;
};

class VAEDecoderRuntime {
public:
    VAEDecoderRuntime(core::ExecutionContext & execution, const assets::TensorSource & source, int64_t frames);
    ~VAEDecoderRuntime();
    VAEDecoderRuntime(const VAEDecoderRuntime &) = delete;
    VAEDecoderRuntime & operator=(const VAEDecoderRuntime &) = delete;

    // Reuses the prepared shape; a different shape replaces only the graph workspace.
    void prepare(int64_t frames);

    // Normalized FM output in [1, frames, 64] order; returns mono 24 kHz samples.
    std::vector<float> decode(const std::vector<float> & latents);

private:
    struct State;
    std::unique_ptr<State> state_;
};

}  // namespace engine::models::auk
