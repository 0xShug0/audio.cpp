#pragma once
#include "engine/community_models/soulx_flashhead/assets.h"
#include "engine/framework/core/execution_context.h"
#include <cstddef>

namespace engine::community_models::soulx_flashhead {
struct GenerateOptions {
    std::filesystem::path reference_image;
    int64_t height = 512, width = 512, steps = 4;
    uint64_t seed = 42;
    float color_strength = 1, audio_guidance_scale = 3;
};
struct VideoChunk {
    int64_t height = 0, width = 0, frames = 0, index = 0;
    std::vector<std::byte> rgb;
};
class Pipeline {
  public:
    Pipeline(core::ExecutionContext & execution, std::shared_ptr<const Assets> assets, bool cuda_cudnn = false);
    ~Pipeline();
    void begin(const GenerateOptions & options);
    std::vector<VideoChunk> push(const std::vector<float> & mono16k, bool final);
    void reset();

  private:
    struct State;
    std::unique_ptr<State> state_;
};
} // namespace engine::community_models::soulx_flashhead
