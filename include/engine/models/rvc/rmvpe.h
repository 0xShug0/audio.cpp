#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace engine::models::rvc {

class RvcRmvpeF0Extractor {
public:
    RvcRmvpeF0Extractor() = default;
    RvcRmvpeF0Extractor(
        std::shared_ptr<const engine::assets::TensorSource> source,
        engine::core::BackendConfig backend,
        engine::assets::TensorStorageType storage_type);
    ~RvcRmvpeF0Extractor();

    RvcRmvpeF0Extractor(RvcRmvpeF0Extractor &&) noexcept;
    RvcRmvpeF0Extractor & operator=(RvcRmvpeF0Extractor &&) noexcept;
    RvcRmvpeF0Extractor(const RvcRmvpeF0Extractor &) = delete;
    RvcRmvpeF0Extractor & operator=(const RvcRmvpeF0Extractor &) = delete;

    std::vector<float> infer_16k_mono(
        const std::vector<float> & waveform_16k,
        float threshold,
        size_t threads) const;

private:
    struct State;

    std::shared_ptr<State> state_;
};

}  // namespace engine::models::rvc
