#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/module.h"

#include <array>
#include <memory>
#include <vector>

namespace engine::community_models::parakeet_tdt {

struct ParakeetTDTAssets;

struct PredictorOutput {
    core::TensorValue hidden;
    std::vector<core::TensorValue> recurrent;
    std::vector<core::TensorValue> auxiliary;
};

// Selected at load time; build() runs only when creating the cached step graph.
class Predictor {
public:
    virtual ~Predictor() = default;
    virtual std::array<int64_t, 2> state_widths() const = 0;
    virtual bool requires_resident_state() const { return false; }
    virtual PredictorOutput build(
        core::ModuleBuildContext & ctx, const core::TensorValue & input,
        const std::vector<core::TensorValue> & recurrent,
        const std::vector<core::TensorValue> & auxiliary) const = 0;
};

std::shared_ptr<const Predictor> load_lstm_predictor(
    core::BackendWeightStore & store, const ParakeetTDTAssets & assets,
    engine::assets::TensorStorageType storage_type);

}  // namespace engine::community_models::parakeet_tdt
