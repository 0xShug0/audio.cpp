#pragma once
#include "engine/community_models/soulx_flashhead/assets.h"
#include "engine/framework/core/execution_context.h"

namespace engine::community_models::soulx_flashhead {
class Denoiser {
  public:
    Denoiser(core::ExecutionContext & execution, const Assets & assets);
    ~Denoiser();
    void prepare_audio(const std::vector<float> & audio, int64_t height, int64_t width);
    std::vector<float> predict(const std::vector<float> & noisy, const std::vector<float> & reference,
                               float timestep, int64_t height, int64_t width);

  private:
    struct State;
    std::unique_ptr<State> state_;
};
} // namespace engine::community_models::soulx_flashhead
