#pragma once
#include "engine/community_models/soulx_flashhead/assets.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/core/module.h"
namespace engine::community_models::soulx_flashhead {
class VideoVae {
  public:
    VideoVae(core::ExecutionContext & execution, const Assets & assets, bool cuda_cudnn = false);
    ~VideoVae();
    void reset(uint64_t seed);
    std::vector<float> encode(const std::vector<float> & pixels, int64_t frames, int64_t height, int64_t width);
    // Valid until the next decode for this shape or destruction.
    const std::vector<float> & decode(const std::vector<float> & latents, int64_t frames, int64_t height,
                                      int64_t width);

    core::TensorValue decode_device(const std::vector<float> & latents, int64_t frames,
        int64_t height, int64_t width);
    std::vector<float> encode_device(const core::TensorValue & pixels, int64_t frames,
        int64_t height, int64_t width);

  private:
    struct State;
    std::unique_ptr<State> state_;
};
} // namespace engine::community_models::soulx_flashhead
