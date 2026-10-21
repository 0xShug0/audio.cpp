#pragma once

#include "engine/models/ming3_audio/assets.h"
#include "engine/framework/core/execution_context.h"

namespace engine::models::ming3_audio {

class AudioEncoder {
public:
    AudioEncoder(core::ExecutionContext & execution, std::shared_ptr<const assets::TensorSource> source,
                 AudioConfig config, int64_t hidden_size);
    ~AudioEncoder();

    std::vector<float> encode(const std::vector<float> & log_mel, int64_t frames);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::ming3_audio
