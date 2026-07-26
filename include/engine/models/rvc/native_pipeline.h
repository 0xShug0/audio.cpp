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
    int f0_up_key = 0;
    float index_rate = 0.0F;
    int filter_radius = 3;
    int resample_sr = 0;
    float rms_mix_rate = 0.25F;
    float protect = 0.33F;
    std::string f0_method = "rmvpe";
    std::string f0_file;
    std::string file_index;
    int speaker_id = 0;
    int x_pad = 1;
    int x_query = 5;
    int x_center = 30;
    int x_max = 32;
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
