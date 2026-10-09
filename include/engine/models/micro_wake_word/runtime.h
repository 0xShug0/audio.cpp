#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/io/json.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::micro_wake_word {

struct ModelConfig {
    int sample_rate = 16000;
    int feature_dim = 40;
    int input_frames = 0;
    int sliding_window_size = 5;
    float probability_cutoff = 0.5F;
    std::string wake_phrase;
};

class Runtime {
public:
    Runtime(std::shared_ptr<const assets::TensorSource> source,
            const io::json::Value & config,
            core::ExecutionContext & execution);
    ~Runtime();

    Runtime(const Runtime &) = delete;
    Runtime & operator=(const Runtime &) = delete;

    const ModelConfig & config() const noexcept;
    float infer(const std::vector<float> & features);
    void reset();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::micro_wake_word
