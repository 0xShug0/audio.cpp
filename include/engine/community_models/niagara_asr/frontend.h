#pragma once

#include "engine/community_models/niagara_asr/assets.h"
#include "engine/framework/runtime/session.h"

#include <memory>
#include <vector>

namespace engine::community_models::niagara_asr {

struct NiagaraFeatures {
    std::vector<float> values;
    int64_t frames = 0;
    int64_t feature_dim = 80;
};

class NiagaraFrontend {
public:
    explicit NiagaraFrontend(std::shared_ptr<const NiagaraAsrAssets> assets);

    NiagaraFeatures extract(const runtime::AudioBuffer & audio) const;

private:
    std::shared_ptr<const NiagaraAsrAssets> assets_;
    std::vector<float> window_;
    std::vector<float> mel_filterbank_;
};

}  // namespace engine::community_models::niagara_asr
