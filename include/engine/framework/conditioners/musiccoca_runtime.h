#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/io/json.h"

#include <array>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace engine::conditioners {

class MusicCoCaRuntime {
public:
    MusicCoCaRuntime(core::ExecutionContext & execution,
                    std::shared_ptr<const assets::TensorSource> weights,
                    io::json::Value graphs, const std::filesystem::path & tokenizer);
    ~MusicCoCaRuntime();

    std::vector<float> embed_text(const std::string & text);
    std::vector<float> map_text(const std::vector<float> & embedding, const std::vector<float> & noise);
    std::vector<float> map_text(const std::vector<float> & embedding, uint32_t seed = 0);
    std::vector<float> embed_features(const std::vector<float> & log_mel);
    std::vector<float> preprocess_audio(const std::vector<float> & mono_16khz);
    std::vector<float> embed_audio(const std::vector<float> & mono_16khz);
    std::array<int32_t, 12> quantize(const std::vector<float> & embedding);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::conditioners
