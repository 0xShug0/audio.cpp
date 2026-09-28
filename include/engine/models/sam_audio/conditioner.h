#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"

#include <memory>
#include <string>
#include <vector>

namespace engine::models::sam_audio {

struct TextConditioning {
    std::vector<int32_t> tokens;
    std::vector<float> features;
};

class T5TextEncoder {
public:
    T5TextEncoder(std::shared_ptr<const assets::TensorSource> source,
                core::ExecutionContext & execution,
                const std::filesystem::path & config,
                const std::filesystem::path & tokenizer,
                int64_t max_length = 512);
    ~T5TextEncoder();
    TextConditioning encode(const std::string & text);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::sam_audio
