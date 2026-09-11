#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/models/rvc/assets.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace engine::models::rvc {

struct RvcSynthesizerInput {
    std::vector<float> features;
    std::vector<int32_t> pitch;
    std::vector<float> pitchf;
    std::vector<float> sine_source;
    int64_t frames = 0;
    int64_t feature_dim = 0;
    int speaker_id = 0;
};

struct RvcSynthesizerOutput {
    std::vector<float> audio;
    int sample_rate = 0;
};

class RvcSynthesizer {
public:
    RvcSynthesizer() = default;
    RvcSynthesizer(
        std::shared_ptr<const engine::assets::TensorSource> source,
        engine::core::BackendConfig backend,
        engine::assets::TensorStorageType storage_type,
        int sample_rate,
        RvcSynthesizerLayout layout,
        bool v1,
        bool has_f0);
    ~RvcSynthesizer();

    RvcSynthesizer(RvcSynthesizer &&) noexcept;
    RvcSynthesizer & operator=(RvcSynthesizer &&) noexcept;
    RvcSynthesizer(const RvcSynthesizer &) = delete;
    RvcSynthesizer & operator=(const RvcSynthesizer &) = delete;

    RvcSynthesizerOutput infer(const RvcSynthesizerInput & input) const;

private:
    struct State;
    std::shared_ptr<State> state_;
};

}  // namespace engine::models::rvc
