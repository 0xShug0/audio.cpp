#pragma once

#include "engine/models/kugelaudio/assets.h"

#include <filesystem>
#include <functional>
#include <memory>
#include <string>

namespace engine::models::kugelaudio {

struct GenerationOptions {
    int64_t max_tokens = 2048;
    int steps = 20;
    float guidance_scale = 3.0F;
    uint32_t seed = 1234;
    bool do_sample = false;
    float temperature = 1.0F;
};

class Generator {
public:
    Generator(const assets::TensorSource & source, const std::filesystem::path & tokenizer_dir,
              core::ExecutionContext & execution, assets::TensorStorageType storage,
              ArConfig ar_config = {}, DiffusionConfig diffusion_config = {}, CodecConfig codec_config = {});
    ~Generator();
    std::vector<float> generate(const std::string & text, const VoiceFeatures & voice,
        const GenerationOptions & options,
        const std::function<void(const std::vector<float> &)> & on_audio = {});

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::kugelaudio
