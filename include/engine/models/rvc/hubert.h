#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace engine::models::rvc {

struct RvcHubertFeatures {
    std::vector<float> values;
    int64_t frames = 0;
    int64_t dim = 0;
};

class RvcHubertEncoder {
public:
    RvcHubertEncoder() = default;
    RvcHubertEncoder(
        std::shared_ptr<const engine::assets::TensorSource> source,
        engine::core::BackendConfig backend,
        engine::assets::TensorStorageType storage_type);
    ~RvcHubertEncoder();

    RvcHubertEncoder(RvcHubertEncoder &&) noexcept;
    RvcHubertEncoder & operator=(RvcHubertEncoder &&) noexcept;
    RvcHubertEncoder(const RvcHubertEncoder &) = delete;
    RvcHubertEncoder & operator=(const RvcHubertEncoder &) = delete;

    RvcHubertFeatures encode_16k_mono(
        const std::vector<float> & waveform_16k,
        bool v1_features) const;

private:
    struct State;
    std::shared_ptr<State> state_;
};

}  // namespace engine::models::rvc
