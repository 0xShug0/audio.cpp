#include "engine/community_models/parakeet_tdt/predictor.h"
#include "engine/community_models/parakeet_tdt/assets.h"
#include "engine/framework/modules/recurrent_modules.h"

namespace engine::community_models::parakeet_tdt {
namespace {

class LstmPredictor final : public Predictor {
public:
    LstmPredictor(core::BackendWeightStore & store, const ParakeetTDTAssets & assets,
                  engine::assets::TensorStorageType storage_type)
        : hidden_size_(assets.config.decoder_hidden_size) {
        const auto & source = *assets.source;
        const int64_t H = hidden_size_;
        layers_.reserve(static_cast<size_t>(assets.config.decoder_layers));
        for (int64_t l = 0; l < assets.config.decoder_layers; ++l) {
            const std::string p = "decoder.lstm";
            layers_.push_back({
                store.load_tensor(source, p + ".weight_ih_l" + std::to_string(l), storage_type, {4*H, H}),
                store.load_tensor(source, p + ".weight_hh_l" + std::to_string(l), storage_type, {4*H, H}),
                store.load_f32_tensor(source, p + ".bias_ih_l" + std::to_string(l), {4*H}),
                store.load_f32_tensor(source, p + ".bias_hh_l" + std::to_string(l), {4*H}),
            });
        }
    }

    std::array<int64_t, 2> state_widths() const override { return {hidden_size_, hidden_size_}; }

    PredictorOutput build(
        core::ModuleBuildContext & ctx, const core::TensorValue & input,
        const std::vector<core::TensorValue> & recurrent,
        const std::vector<core::TensorValue> & auxiliary) const override {
        PredictorOutput result;
        auto hidden = input;
        for (size_t l = 0; l < layers_.size(); ++l) {
            auto out = modules::LSTMCellModule({hidden_size_, hidden_size_})
                .build(ctx, hidden, recurrent[l], auxiliary[l], layers_[l]);
            result.recurrent.push_back(out.hidden);
            result.auxiliary.push_back(out.cell);
            hidden = out.hidden;
        }
        result.hidden = hidden;
        return result;
    }

private:
    int64_t hidden_size_;
    std::vector<modules::LSTMCellWeights> layers_;
};

}  // namespace

std::shared_ptr<const Predictor> load_lstm_predictor(
    core::BackendWeightStore & store, const ParakeetTDTAssets & assets,
    engine::assets::TensorStorageType storage_type) {
    return std::make_shared<LstmPredictor>(store, assets, storage_type);
}

}  // namespace engine::community_models::parakeet_tdt
