#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"

#include <memory>
#include <vector>

namespace engine::models::kugelaudio {

struct CodecConfig {
    int64_t latent_size = 64;
    int64_t filters = 32;
    std::vector<int> ratios = {8, 5, 5, 4, 2, 2};
    std::vector<int> depths = {8, 3, 3, 3, 3, 3, 3};
    float norm_eps = 1e-5F;
};

class AcousticDecoder {
public:
    AcousticDecoder(const assets::TensorSource & source, core::ExecutionContext & execution,
                    assets::TensorStorageType storage, CodecConfig config = {});
    ~AcousticDecoder();
    void reset();
    std::vector<float> decode_frame(const std::vector<float> & latent);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::kugelaudio
