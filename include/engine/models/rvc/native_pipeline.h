#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/runtime/session.h"
#include "engine/models/rvc/assets.h"

#include <memory>
#include <string>
#include <vector>

namespace engine::models::rvc {

struct RvcInferenceConfig {
    int semitone_shift = 0;
    float retrieval_blend = 0.0F;
    int pitch_filter_radius = 3;
    int output_sample_rate = 0;
    float rms_mix_rate = 0.25F;
    float unvoiced_protection = 0.33F;
    std::string pitch_extractor = "rmvpe";
    std::string pitch_path;
    std::string retrieval_index_path;
    int speaker_id = 0;
    int audio_pad_duration_sec = 1;
    int split_query_sec = 5;
    int split_center_sec = 30;
    int split_threshold_sec = 32;
};

class RvcNativePipeline {
public:
    RvcNativePipeline(
        std::shared_ptr<const RvcAssets> assets,
        engine::core::BackendConfig backend,
        engine::assets::TensorStorageType storage_type);
    ~RvcNativePipeline();

    RvcNativePipeline(RvcNativePipeline &&) noexcept;
    RvcNativePipeline & operator=(RvcNativePipeline &&) noexcept;
    RvcNativePipeline(const RvcNativePipeline &) = delete;
    RvcNativePipeline & operator=(const RvcNativePipeline &) = delete;

    runtime::AudioBuffer infer(
        const runtime::AudioBuffer & source,
        const RvcVoiceModel & voice,
        const RvcInferenceConfig & config,
        size_t threads);

private:
    struct State;
    std::shared_ptr<State> state_;
};

}  // namespace engine::models::rvc
