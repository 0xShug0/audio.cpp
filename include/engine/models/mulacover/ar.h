#pragma once

#include "engine/models/mulacover/assets.h"
#include "engine/models/mulacover/prompt.h"
#include "engine/models/mulacover/symbolic.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/sampling/torch_random.h"

#include <memory>
#include <vector>

namespace engine::models::mulacover {

struct GenerationOptions {
    float temperature = 1.0F;
    float guidance_scale = 1.5F;
    int64_t top_k = 250;
};

struct GeneratedFrames {
    std::vector<int32_t> codes;
    int64_t frames = 0, codebooks = 0;
    uint64_t rng_offset_blocks = 0;
    sampling::TorchCudaSamplingPolicy rng_policy;
};

// One session owns weights, both decoder runtimes and the small audio I/O graphs.
// Conditioning shape changes invalidate only the affected backbone graphs.
class AutoregressiveRuntime final {
public:
    AutoregressiveRuntime(core::ExecutionContext & execution, const MuLaCoverAssets & assets,
        assets::TensorStorageType storage = assets::TensorStorageType::Native);
    ~AutoregressiveRuntime();
    AutoregressiveRuntime(const AutoregressiveRuntime &) = delete;
    AutoregressiveRuntime & operator=(const AutoregressiveRuntime &) = delete;

    const PromptWeights & prompt_weights() const;
    void prepare(const SymbolicInputs & symbolic, const std::vector<float> & style, bool cfg_enabled);
    GeneratedFrames generate(const PromptEmbeddings & prompt, const GenerationOptions & options,
        uint64_t seed, int64_t max_frames, const std::vector<int32_t> * teacher_codes = nullptr);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::mulacover
