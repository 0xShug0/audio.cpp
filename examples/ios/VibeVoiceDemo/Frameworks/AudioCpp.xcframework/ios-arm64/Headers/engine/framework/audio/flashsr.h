#pragma once

#include "engine/framework/core/backend.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

namespace engine::audio {

struct FlashSrWeights;
class FlashSrGraph;

struct FlashSrOutput {
    int sample_rate = 48000;
    std::vector<float> samples;
};

class FlashSrModel {
public:
    static FlashSrModel load_from_directory(const std::filesystem::path & model_dir);
    static FlashSrModel load_from_directory(const std::filesystem::path & model_dir, const core::BackendConfig & backend_config);

    FlashSrModel();
    ~FlashSrModel();
    FlashSrModel(FlashSrModel &&) noexcept;
    FlashSrModel & operator=(FlashSrModel &&) noexcept;
    FlashSrModel(const FlashSrModel &) = delete;
    FlashSrModel & operator=(const FlashSrModel &) = delete;

    FlashSrOutput super_resolve_mono_16k(const std::vector<float> & waveform) const;

private:
    explicit FlashSrModel(std::shared_ptr<FlashSrWeights> weights);

    std::shared_ptr<FlashSrWeights> weights_;
    mutable std::unique_ptr<FlashSrGraph> graph_;
};

}  // namespace engine::audio
