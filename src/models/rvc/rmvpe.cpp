#include "engine/models/rvc/rmvpe.h"

#include "engine/framework/modules/pitch_extractors/rmvpe_pitch_extractor.h"

#include <stdexcept>
#include <utility>

namespace engine::models::rvc {

struct RvcRmvpeF0Extractor::State {
    explicit State(
        std::shared_ptr<const engine::assets::TensorSource> source,
        engine::core::BackendConfig backend,
        engine::assets::TensorStorageType storage_type)
        : component(
              std::move(source),
              std::move(backend),
              storage_type,
              engine::modules::RmvpePitchExtractorConfig{"rvc.rmvpe"}) {}

    engine::modules::RmvpePitchExtractorComponent component;
};

RvcRmvpeF0Extractor::RvcRmvpeF0Extractor(
    std::shared_ptr<const engine::assets::TensorSource> source,
    engine::core::BackendConfig backend,
    engine::assets::TensorStorageType storage_type)
    : state_(std::make_shared<State>(std::move(source), std::move(backend), storage_type)) {}

RvcRmvpeF0Extractor::~RvcRmvpeF0Extractor() = default;
RvcRmvpeF0Extractor::RvcRmvpeF0Extractor(RvcRmvpeF0Extractor &&) noexcept = default;
RvcRmvpeF0Extractor & RvcRmvpeF0Extractor::operator=(RvcRmvpeF0Extractor &&) noexcept = default;

std::vector<float> RvcRmvpeF0Extractor::infer_16k_mono(
    const std::vector<float> & waveform_16k,
    float threshold,
    size_t threads) const {
    if (state_ == nullptr) {
        throw std::runtime_error("RVC RMVPE is not initialized");
    }
    return state_->component.infer_16k_mono(waveform_16k, threshold, threads);
}

}  // namespace engine::models::rvc
