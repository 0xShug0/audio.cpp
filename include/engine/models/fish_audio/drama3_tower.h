#pragma once

#include "engine/models/fish_audio/drama3.h"
#include "engine/framework/core/execution_context.h"

namespace engine::models::fish_audio {

class Drama3Tower {
public:
    Drama3Tower(std::shared_ptr<const FishAudioAssets> assets, core::ExecutionContext & execution,
        size_t weight_context_bytes, assets::TensorStorageType storage);
    ~Drama3Tower();
    void reset(int64_t capacity);
    std::vector<float> embed_backbone(const std::vector<int32_t> & frame,
        const std::vector<float> & token_embedding) const;
    std::vector<float> run(const std::vector<float> & hidden, const std::vector<int32_t> & current,
        const std::vector<int32_t> & next, int64_t position, bool teacher_force,
        const std::function<int32_t(const std::vector<float> &)> & sample,
        std::vector<int32_t> & output_codes, bool return_hidden = true);
    void release_graphs();
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::fish_audio
