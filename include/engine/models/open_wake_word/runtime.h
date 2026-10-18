#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/models/open_wake_word/weights.h"

namespace engine::models::open_wake_word {

struct Score {
    int64_t end_sample = 0;
    float probability = 0;
};

class Runtime {
public:
    Runtime(const Weights &weights, const core::ExecutionContext &execution, bool arm_embedding);
    ~Runtime();
    void reset();
    std::vector<Score> push(const std::vector<float> &samples);
    std::vector<Score> finish();
    std::vector<float> encode(const std::vector<float> &mel);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace engine::models::open_wake_word
