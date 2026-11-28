#pragma once

#include "engine/models/moss_soundeffect/assets.h"
#include "engine/framework/core/execution_context.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::moss_soundeffect {

// Session-owned tokenizer, text weights and cached Qwen graphs. Execution must outlive this object.
// Cleaning follows upstream's valid-Unicode ftfy/HTML/whitespace path; malformed UTF-8
// is rejected and mojibake/legacy-encoding repair is intentionally not implemented.
class TextEncoderRuntime {
public:
    TextEncoderRuntime(core::ExecutionContext & execution, std::shared_ptr<const Assets> assets,
                       assets::TensorStorageType storage);
    ~TextEncoderRuntime();
    TextEncoderRuntime(const TextEncoderRuntime &) = delete;
    TextEncoderRuntime & operator=(const TextEncoderRuntime &) = delete;

    std::vector<int32_t> tokenize(const std::string & prompt) const;
    // Fixed text_length prefill shape; flattened hidden output is zero after the valid prefix.
    // Empty prompts return zeros without executing the graph.
    std::vector<float> encode(const std::string & prompt);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::moss_soundeffect
