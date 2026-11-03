#include "engine/community_models/parakeet_tdt/kda_predictor.h"
#include "engine/community_models/parakeet_tdt/assets.h"
#include "engine/framework/modules/weight_binding.h"
#include "engine/framework/io/json.h"

#include <stdexcept>
#include <utility>

#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"

namespace engine::community_models::parakeet_tdt {
namespace core = engine::core;
namespace modules = engine::modules;

PredictorOutput build_kda_predictor(
    core::ModuleBuildContext & ctx, const KdaPredictorConfig & config,
    const KdaPredictorWeights & weights, const core::TensorValue & input,
    const std::vector<core::TensorValue> & recurrent,
    const std::vector<core::TensorValue> & convolution) {
    const int64_t H = config.decoder_hidden_size;
    const int64_t N = config.kda_heads;
    const int64_t D = config.kda_head_dim;
    const int64_t C = N * D;
    const int64_t K = config.kda_conv_size;
    PredictorOutput result;
    auto hidden = input;
    for (size_t l = 0; l < weights.layers.size(); ++l) {
        const auto & w = weights.layers[l];
        auto x = modules::LayerNormModule({H, 1.0e-5F, true, true}).build(ctx, hidden, w.norm);
        std::array<core::TensorValue, 3> qkv;
        core::TensorValue next_conv;
        for (size_t i = 0; i < qkv.size(); ++i) {
            auto projected = modules::LinearModule({H, C, false}).build(ctx, x, w.qkv[i]);
            auto history = modules::SliceModule({1, static_cast<int64_t>(i) * C * K, C * K})
                .build(ctx, convolution[l]);
            history = core::reshape_tensor(ctx, history, core::TensorShape::from_dims({1, C, K}));
            auto current = core::reshape_tensor(ctx, projected, core::TensorShape::from_dims({1, C, 1}));
            auto window = current;
            if (K > 1) {
                auto tail = modules::SliceModule({2, 1, K - 1}).build(ctx, history);
                window = modules::ConcatModule({2}).build(ctx, tail, current);
            }
            auto flat = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, window),
                core::TensorShape::from_dims({1, C * K}));
            next_conv = i == 0 ? flat : modules::ConcatModule({1}).build(ctx, next_conv, flat);
            auto convolved = modules::DepthwiseConv1dModule({C, K, 1, 0, 1, false}).build(ctx, window, w.conv[i]);
            convolved = modules::SiluModule{}.build(ctx, convolved);
            qkv[i] = core::reshape_tensor(ctx, convolved, core::TensorShape::from_dims({1, 1, N, D}));
        }
        for (size_t i = 0; i < 2; ++i) {
            // fla normalizes with rsqrt(sum(x*x) + eps), whereas ggml_l2_norm
            // clamps the denominator. Keep the source's additive epsilon.
            auto square = modules::MulModule{}.build(ctx, qkv[i], qkv[i]);
            auto sum = modules::ReduceSumModule({3}).build(ctx, square);
            sum = modules::ScaleBiasModule({1.0F, 1.0e-6F}).build(ctx, sum);
            auto denominator = modules::SqrtModule{}.build(ctx, sum);
            qkv[i] = core::wrap_tensor(ggml_div(ctx.ggml, qkv[i].tensor, denominator.tensor),
                qkv[i].shape, GGML_TYPE_F32);
        }
        auto forget = modules::LinearModule({H, D, false}).build(ctx, x, w.forget_in);
        forget = modules::LinearModule({D, C, false}).build(ctx, forget, w.forget_out);
        auto bias = core::reshape_tensor(ctx, w.dt_bias, core::TensorShape::from_dims({1, C}));
        forget = modules::AddModule{}.build(ctx, forget, bias);
        forget = core::wrap_tensor(ggml_softplus(ctx.ggml, forget.tensor), forget.shape, GGML_TYPE_F32);
        forget = core::reshape_tensor(ctx, forget, core::TensorShape::from_dims({N, D}));
        auto a = core::wrap_tensor(ggml_exp(ctx.ggml, w.a_log.tensor), w.a_log.shape, GGML_TYPE_F32);
        a = core::reshape_tensor(ctx, a, core::TensorShape::from_dims({N, 1}));
        forget = modules::MulModule(true).build(ctx, forget, a);
        forget = modules::ScaleBiasModule({-1.0F, 0.0F}).build(ctx, forget);
        forget = core::reshape_tensor(ctx, forget, core::TensorShape::from_dims({1, 1, N, D}));
        auto beta = modules::LinearModule({H, N, false}).build(ctx, x, w.beta);
        beta = modules::SigmoidModule{}.build(ctx, beta);
        beta = core::reshape_tensor(ctx, beta, core::TensorShape::from_dims({1, 1, N, 1}));
        auto state = core::reshape_tensor(ctx, recurrent[l], core::TensorShape::from_dims({1, 1, C * D}));
        auto delta = core::wrap_tensor(ggml_gated_delta_net(ctx.ggml, qkv[0].tensor, qkv[1].tensor,
            qkv[2].tensor, forget.tensor, beta.tensor, state.tensor),
            core::TensorShape::from_dims({1 + D, C}), GGML_TYPE_F32);
        auto output = modules::SliceModule({0, 0, 1}).build(ctx, delta);
        auto next_state = modules::SliceModule({0, 1, D}).build(ctx, delta);
        result.recurrent.push_back(core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, next_state),
            recurrent[l].shape));
        result.auxiliary.push_back(next_conv);
        output = core::reshape_tensor(ctx, output, core::TensorShape::from_dims({N, D}));
        output = modules::RMSNormModule({D, 1.0e-5F, true, false}).build(ctx, output, w.output_norm);
        auto gate = modules::LinearModule({H, D, false}).build(ctx, x, w.gate_in);
        gate = modules::LinearModule({D, C, true}).build(ctx, gate, w.gate_out);
        gate = modules::SigmoidModule{}.build(ctx, gate);
        gate = core::reshape_tensor(ctx, gate, output.shape);
        output = modules::MulModule{}.build(ctx, output, gate);
        output = core::reshape_tensor(ctx, output, core::TensorShape::from_dims({1, C}));
        output = modules::LinearModule({C, H, false}).build(ctx, output, w.output);
        hidden = modules::AddModule{}.build(ctx, hidden, output);
    }
    result.hidden = config.kda_final_norm
        ? modules::LayerNormModule({H, 1.0e-5F, true, true}).build(ctx, hidden, weights.final_norm)
        : hidden;
    return result;
}

namespace {
class KdaPredictor final : public Predictor {
public:
    KdaPredictor(core::BackendWeightStore & store, const ParakeetTDTAssets & assets,
                 engine::assets::TensorStorageType storage_type) {
        namespace json = engine::io::json;
        namespace binding = engine::modules::binding;
        const auto root = assets.resources.parse_json("config");
        if (json::require_string(root, "predictor_type") != "kda")
            throw std::runtime_error("Haas requires a KDA predictor");
        const auto & kda = root.require("kda_config");
        config_ = {assets.config.decoder_hidden_size, json::require_i64(kda, "num_heads"),
            json::require_i64(kda, "head_dim"), json::require_i64(kda, "conv_size"),
            json::require_bool(kda, "final_norm")};
        if (config_.kda_heads <= 0 || config_.kda_head_dim <= 0 || config_.kda_conv_size <= 0)
            throw std::runtime_error("Haas invalid KDA predictor geometry");
        const auto & source = *assets.source;
        const int64_t H = config_.decoder_hidden_size;
        const int64_t N = config_.kda_heads, D = config_.kda_head_dim, C = N * D;
        for (int64_t l = 0; l < assets.config.decoder_layers; ++l) {
            const auto p = "decoder.kda.layers." + std::to_string(l);
            KdaLayerWeights layer;
            layer.norm = binding::norm_from_source(store, source, "decoder.kda.norms." + std::to_string(l), H);
            const std::array<std::string, 3> names = {"q", "k", "v"};
            for (size_t i = 0; i < names.size(); ++i) {
                layer.qkv[i] = binding::linear_from_source(store, source, p + "." + names[i] + "_proj", storage_type, C, H, false);
                layer.conv[i].weight = store.load_f32_tensor(source, p + "." + names[i] + "_conv1d.weight",
                    {C, 1, config_.kda_conv_size});
            }
            layer.forget_in = binding::linear_from_source(store, source, p + ".f_proj.0", storage_type, D, H, false);
            layer.forget_out = binding::linear_from_source(store, source, p + ".f_proj.1", storage_type, C, D, false);
            layer.beta = binding::linear_from_source(store, source, p + ".b_proj", storage_type, N, H, false);
            layer.gate_in = binding::linear_from_source(store, source, p + ".g_proj.0", storage_type, D, H, false);
            layer.gate_out = binding::linear_from_source(store, source, p + ".g_proj.1", storage_type, C, D, true);
            layer.output = binding::linear_from_source(store, source, p + ".o_proj", storage_type, H, C, false);
            layer.output_norm = binding::norm_weight_from_source(store, source, p + ".o_norm", D);
            layer.a_log = store.load_f32_tensor(source, p + ".A_log", {N});
            layer.dt_bias = store.load_f32_tensor(source, p + ".dt_bias", {C});
            weights_.layers.push_back(std::move(layer));
        }
        if (config_.kda_final_norm)
            weights_.final_norm = binding::norm_from_source(store, source, "decoder.kda.final_norm", H);
    }

    std::array<int64_t, 2> state_widths() const override {
        return {config_.kda_heads * config_.kda_head_dim * config_.kda_head_dim,
                3 * config_.kda_heads * config_.kda_head_dim * config_.kda_conv_size};
    }
    bool requires_resident_state() const override { return true; }
    PredictorOutput build(core::ModuleBuildContext & ctx, const core::TensorValue & input,
        const std::vector<core::TensorValue> & recurrent,
        const std::vector<core::TensorValue> & auxiliary) const override {
        return build_kda_predictor(ctx, config_, weights_, input, recurrent, auxiliary);
    }

private:
    KdaPredictorConfig config_;
    KdaPredictorWeights weights_;
};
}  // namespace

std::shared_ptr<const Predictor> load_kda_predictor(
    core::BackendWeightStore & store, const ParakeetTDTAssets & assets,
    engine::assets::TensorStorageType storage_type) {
    return std::make_shared<KdaPredictor>(store, assets, storage_type);
}

}  // namespace engine::community_models::parakeet_tdt
