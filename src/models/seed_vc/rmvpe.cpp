#include "engine/models/seed_vc/rmvpe.h"

#include "engine/framework/modules/pitch_extractors/rmvpe_pitch_extractor.h"

#include <stdexcept>
#include <utility>

namespace engine::models::seed_vc {

struct SeedVcRmvpeF0Extractor::State {
    explicit State(
        std::shared_ptr<const engine::assets::TensorSource> source,
        engine::core::BackendConfig backend,
        engine::assets::TensorStorageType storage_type)
        : component(
              std::move(source),
              std::move(backend),
              storage_type,
              engine::modules::RmvpePitchExtractorConfig{"seed_vc.rmvpe"}) {}

    engine::modules::RmvpePitchExtractorComponent component;
};

SeedVcRmvpeF0Extractor::SeedVcRmvpeF0Extractor(
    std::shared_ptr<const engine::assets::TensorSource> source,
    engine::core::BackendConfig backend,
    engine::assets::TensorStorageType storage_type)
    : state_(std::make_shared<State>(std::move(source), std::move(backend), storage_type)) {}

SeedVcRmvpeF0Extractor::~SeedVcRmvpeF0Extractor() = default;
SeedVcRmvpeF0Extractor::SeedVcRmvpeF0Extractor(SeedVcRmvpeF0Extractor &&) noexcept = default;
SeedVcRmvpeF0Extractor & SeedVcRmvpeF0Extractor::operator=(SeedVcRmvpeF0Extractor &&) noexcept = default;

std::vector<float> SeedVcRmvpeF0Extractor::infer_16k_mono(
    const std::vector<float> & waveform_16k,
    float threshold,
    size_t threads) const {
    if (state_ == nullptr) {
        throw std::runtime_error("Seed-VC RMVPE is not initialized");
    }
    return state_->component.infer_16k_mono(waveform_16k, threshold, threads);
}

}  // namespace engine::models::seed_vc
