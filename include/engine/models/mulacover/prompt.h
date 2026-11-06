#pragma once

#include "engine/models/mulacover/assets.h"
#include "engine/framework/core/execution_context.h"

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::mulacover {

struct PromptWeights {
    ggml_type compute_type = GGML_TYPE_F32;
    core::TensorValue text, unconditional;
    modules::LinearWeights qwen, muq;
};

struct PromptEmbeddings {
    std::vector<float> values;
    int64_t batch_size = 0, steps = 0, dims = 0;
};

// Session-owned; execution must outlive this encoder. A single graph is reused at
// the current (prompt steps, CFG batch) shape. Inputs are text contents, not filesystem paths.
class PromptEncoder final {
public:
    // Borrows the session's embedding/projection tensors. Their store and execution
    // context must outlive this encoder.
    PromptEncoder(core::ExecutionContext & execution,
                  const PromptWeights & weights,
                  std::shared_ptr<const assets::TensorSource> source,
                  const std::filesystem::path & tokenizer_json,
                  MuLaCoverGenerationConfig config);
    ~PromptEncoder();
    PromptEncoder(const PromptEncoder &) = delete;
    PromptEncoder & operator=(const PromptEncoder &) = delete;

    static std::string normalize_tags(const std::string & tags);

    PromptEmbeddings encode(
        const std::string & lyrics, const std::string & tags,
        const std::vector<float> & qwen_style, float cfg_scale);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::mulacover
