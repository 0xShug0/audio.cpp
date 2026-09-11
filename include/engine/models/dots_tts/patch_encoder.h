#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/models/dots_tts/types.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace engine::models::dots_tts {

struct DotsPatchEmbeddings {
    std::vector<float> values;
    int64_t patches = 0;
    int64_t hidden_size = 0;
};

class DotsPatchEncoderState {
public:
    DotsPatchEncoderState();
    ~DotsPatchEncoderState();
    DotsPatchEncoderState(DotsPatchEncoderState &&) noexcept;
    DotsPatchEncoderState & operator=(DotsPatchEncoderState &&) noexcept;
    DotsPatchEncoderState(const DotsPatchEncoderState &) = delete;
    DotsPatchEncoderState & operator=(const DotsPatchEncoderState &) = delete;

    int64_t seq_len() const noexcept;
    int64_t capacity() const noexcept;

private:
    friend class DotsPatchEncoderComponent;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class DotsPatchEncoderComponent {
public:
    static DotsPatchEncoderComponent load_from_tensor_source(
        std::shared_ptr<const assets::TensorSource> source,
        core::BackendConfig backend,
        DotsConfig config,
        assets::TensorStorageType weight_storage_type);

    DotsPatchEncoderComponent();
    DotsPatchEncoderComponent(DotsPatchEncoderComponent &&) noexcept;
    DotsPatchEncoderComponent & operator=(DotsPatchEncoderComponent &&) noexcept;
    DotsPatchEncoderComponent(const DotsPatchEncoderComponent &) = delete;
    DotsPatchEncoderComponent & operator=(const DotsPatchEncoderComponent &) = delete;
    ~DotsPatchEncoderComponent();

    bool is_loaded() const noexcept;
    DotsPatchEncoderState create_state(int64_t max_audio_patch_count) const;
    DotsPatchEmbeddings prefill(
        const std::vector<float> & normalized_latents,
        int64_t frames,
        DotsPatchEncoderState & state) const;
    DotsPatchEmbeddings decode_patch(
        const std::vector<float> & normalized_latents,
        DotsPatchEncoderState & state) const;

    void release_runtime_graphs();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::dots_tts
