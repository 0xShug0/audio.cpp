#pragma once
#include "engine/framework/core/execution_context.h"
#include "engine/framework/core/module.h"
#include <memory>
#include <vector>
namespace engine::community_models::soulx_flashhead {
struct ColorMatchResult {
    core::TensorValue video;
    core::TensorValue motion;
};

class ColorMatcher {
  public:
    explicit ColorMatcher(core::ExecutionContext & execution);
    ~ColorMatcher();
    // Borrows the input when disabled; otherwise the cached graph output.
    const std::vector<float> & match(const std::vector<float> & video, const std::vector<float> & portrait,
                                     int64_t height, int64_t width, float strength);

    // Keeps corrected video and a contiguous motion tail on the backend.
    ColorMatchResult match_device(const core::TensorValue & video, const std::vector<float> & portrait,
        int64_t height, int64_t width, float strength, int64_t motion_frames);

  private:
    struct State;
    std::unique_ptr<State> state_;
};
} // namespace engine::community_models::soulx_flashhead
