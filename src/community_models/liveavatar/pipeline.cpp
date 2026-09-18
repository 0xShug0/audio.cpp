#include "pipeline_internal.h"

#include <memory>
#include <stdexcept>
#include <utility>

namespace engine::community_models::liveavatar {

LiveAvatarPipelineRuntime::LiveAvatarPipelineRuntime(
    std::shared_ptr<const LiveAvatarAssets> assets,
    engine::core::ExecutionContext & execution,
    bool denoiser_weight_streaming)
    : assets_(require_assets(std::move(assets))),
      execution_(execution),
      impl_(std::make_unique<LiveAvatarPipelineState>(assets_, execution_, denoiser_weight_streaming)),
      denoiser_weight_streaming_(denoiser_weight_streaming) {}

LiveAvatarPipelineRuntime::~LiveAvatarPipelineRuntime() = default;

LiveAvatarVideoResult LiveAvatarPipelineRuntime::generate(
    const LiveAvatarGenerateRequest & request,
    const LiveAvatarVideoChunkCallback & chunk_callback) {
    if (denoiser_weight_streaming_ && (!request.blockwise_generation || !request.denoiser_layerwise)) {
        throw std::runtime_error(
            "liveavatar.denoiser_weight_streaming requires generation_mode=liveavatar and denoiser_layerwise=true");
    }
    auto shared = prepare_liveavatar_generate_shared(*impl_, execution_, assets_, request);
    if (request.blockwise_generation) {
        return generate_liveavatar_blockwise(*impl_, execution_, assets_, shared, chunk_callback);
    }
    return generate_liveavatar_offline(*impl_, execution_, assets_, shared);
}

}  // namespace engine::community_models::liveavatar
