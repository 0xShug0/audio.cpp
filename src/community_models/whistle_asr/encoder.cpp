#include "engine/community_models/whistle_asr/encoder.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/positional_modules.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::community_models::whistle_asr {
namespace {

constexpr size_t kGraphNodes = 16384;
// 30 s of 10 ms mel frames through three stride-2 stem stages.
constexpr int64_t kMaximumFrames = 375;
constexpr float kNormEpsilon = 1.0e-6f;
constexpr int64_t kMels = 80;
constexpr int64_t kStemChannels = 128;
constexpr int64_t kStemWidth = 10;
constexpr int64_t kHeads = 8;
constexpr int64_t kKvHeads = 2;
constexpr int64_t kQkDim = 48;
constexpr int64_t kVDim = 64;
constexpr int64_t kConvTaps = 9;
constexpr int kSinkhornIterations = 20;
constexpr float kRopeTheta = 100000.0f;
constexpr const char * kEncoderLayer = "encoder/layers/block/";

using core::TensorShape;
using core::TensorValue;

TensorValue rows_2d(ggml_tensor * tensor) {
    return core::wrap_tensor(tensor, TensorShape::from_dims({tensor->ne[1], tensor->ne[0]}), GGML_TYPE_F32);
}

// Reads checkpoint tensors in their JAX layouts and uploads them in the shapes the
// framework modules expect. Layer-stacked tensors carry a leading [8] dimension.
class WeightLoader {
public:
    WeightLoader(core::BackendWeightStore & store, const assets::TensorSource & source)
        : store_(store), source_(source) {}

    std::vector<float> values(const std::string & name, std::vector<int64_t> shape, int layer = -1) const {
        if (layer < 0) {
            return source_.require_f32(name, shape);
        }
        shape.insert(shape.begin(), kWhistleLayers);
        const auto stacked = source_.require_f32(name, shape);
        const size_t stride = stacked.size() / kWhistleLayers;
        return {stacked.begin() + static_cast<std::ptrdiff_t>(layer * stride),
            stacked.begin() + static_cast<std::ptrdiff_t>((layer + 1) * stride)};
    }

    float scalar(const std::string & name, int layer = -1) const {
        return values(name, {}, layer).front();
    }

    TensorValue vector(const std::string & name, int64_t size, int layer = -1) {
        return store_.make_f32(TensorShape::from_dims({size}), values(name, {size}, layer));
    }

    modules::NormWeights norm(const std::string & name, int64_t size, int layer = -1) {
        return {vector(name + "/scale", size, layer), std::nullopt};
    }

    // JAX kernels are [in, out]; LinearModule and ggml_mul_mat take [out, in].
    TensorValue transposed(const std::string & name, int64_t in, int64_t out, int layer = -1) {
        const auto source = values(name, {in, out}, layer);
        std::vector<float> weight(source.size());
        for (int64_t row = 0; row < in; ++row) {
            for (int64_t col = 0; col < out; ++col) {
                weight[static_cast<size_t>(col * in + row)] = source[static_cast<size_t>(row * out + col)];
            }
        }
        return store_.make_f32(TensorShape::from_dims({out, in}), std::move(weight));
    }

    modules::LinearWeights linear(const std::string & name, int64_t in, int64_t out, int layer = -1) {
        return {transposed(name + "/kernel", in, out, layer), std::nullopt};
    }

    // Pointwise stem projections as 1x1 convolutions, [out, in, 1, 1].
    modules::Conv2dWeights pointwise(const std::string & name) {
        const auto source = values(name + "/kernel", {kStemChannels, kStemChannels});
        std::vector<float> weight(source.size());
        for (int64_t in = 0; in < kStemChannels; ++in) {
            for (int64_t out = 0; out < kStemChannels; ++out) {
                weight[static_cast<size_t>(out * kStemChannels + in)] = source[static_cast<size_t>(in * kStemChannels + out)];
            }
        }
        return {store_.make_f32(TensorShape::from_dims({kStemChannels, kStemChannels, 1, 1}), std::move(weight)), std::nullopt};
    }

    // JAX HWIO [3, 3, 1, channels] to the module's [channels, 1, 3, 3].
    modules::Conv2dWeights stem_conv(const std::string & name) {
        const auto source = values(name, {3, 3, 1, kStemChannels});
        std::vector<float> weight(source.size());
        for (int64_t kh = 0; kh < 3; ++kh) {
            for (int64_t kw = 0; kw < 3; ++kw) {
                for (int64_t channel = 0; channel < kStemChannels; ++channel) {
                    weight[static_cast<size_t>((channel * 3 + kh) * 3 + kw)] =
                        source[static_cast<size_t>((kh * 3 + kw) * kStemChannels + channel)];
                }
            }
        }
        return {store_.make_f32(TensorShape::from_dims({kStemChannels, 1, 3, 3}), std::move(weight)), std::nullopt};
    }

    // JAX [taps, 1, channels] to the module's [channels, 1, taps].
    modules::DepthwiseConv1dWeights depthwise(const std::string & name, int layer) {
        const auto source = values(name, {kConvTaps, 1, kWhistleDimension}, layer);
        std::vector<float> weight(source.size());
        for (int64_t tap = 0; tap < kConvTaps; ++tap) {
            for (int64_t channel = 0; channel < kWhistleDimension; ++channel) {
                weight[static_cast<size_t>(channel * kConvTaps + tap)] =
                    source[static_cast<size_t>(tap * kWhistleDimension + channel)];
            }
        }
        return {store_.make_f32(TensorShape::from_dims({kWhistleDimension, 1, kConvTaps}), std::move(weight)),
            std::nullopt};
    }

    WhistleHadamardWeights hadamard(const std::string & prefix, int layer) {
        WhistleHadamardWeights weights;
        weights.d1 = vector(prefix + "d1", kWhistleDimension, layer);
        weights.d2 = vector(prefix + "d2", kWhistleDimension, layer);
        weights.d3 = vector(prefix + "d3", kWhistleDimension, layer);
        weights.d4 = vector(prefix + "d4", kWhistleDimension, layer);
        weights.b2 = vector(prefix + "b2", kWhistleDimension, layer);
        weights.cond_v = {transposed(prefix + "cond_v", kWhistleDimension, 8, layer), std::nullopt};
        weights.cond_u = {transposed(prefix + "cond_u", 8, kWhistleDimension, layer), std::nullopt};
        weights.w1a = transposed(prefix + "w1a", 16, 16, layer);
        weights.w1b = transposed(prefix + "w1b", 32, 32, layer);
        weights.w2a = transposed(prefix + "w2a", 16, 16, layer);
        weights.w2b = transposed(prefix + "w2b", 32, 32, layer);
        weights.w3a = transposed(prefix + "w3a", 16, 16, layer);
        weights.w3b = transposed(prefix + "w3b", 32, 32, layer);
        return weights;
    }

    WhistleMhcWeights mhc(const std::string & prefix, int layer) {
        WhistleMhcWeights weights;
        const int64_t width = kWhistleLanes * kWhistleDimension;
        weights.phi_pre = {transposed(prefix + "phi_pre", width, kWhistleLanes, layer), std::nullopt};
        weights.phi_post = {transposed(prefix + "phi_post", width, kWhistleLanes, layer), std::nullopt};
        weights.phi_res = {transposed(prefix + "phi_res", width, kWhistleLanes * kWhistleLanes, layer), std::nullopt};
        weights.a_pre = scalar(prefix + "a_pre", layer);
        weights.a_post = scalar(prefix + "a_post", layer);
        weights.a_res = scalar(prefix + "a_res", layer);
        // The lane matching the layer index is the active lane; the others are
        // pushed towards the sigmoid floor by a fixed logit offset.
        auto pre_bias = values(prefix + "b_pre", {kWhistleLanes}, layer);
        auto post_bias = values(prefix + "b_post", {kWhistleLanes}, layer);
        for (int64_t lane = 0; lane < kWhistleLanes; ++lane) {
            const bool active = lane == layer % kWhistleLanes;
            pre_bias[static_cast<size_t>(lane)] += active ? 4.0f : -4.0f;
            post_bias[static_cast<size_t>(lane)] += active ? 0.0f : -4.0f;
        }
        weights.pre_bias = store_.make_f32(TensorShape::from_dims({kWhistleLanes}), std::move(pre_bias));
        weights.post_bias = store_.make_f32(TensorShape::from_dims({kWhistleLanes}), std::move(post_bias));
        weights.res_bias = store_.make_f32(
            TensorShape::from_dims({kWhistleLanes * kWhistleLanes}),
            values(prefix + "b_res", {kWhistleLanes, kWhistleLanes}, layer));
        return weights;
    }

    // Gather indices repeated per frame, consumed through a leading-frame view.
    TensorValue permutation(const std::array<uint16_t, 512> & permutation) {
        std::vector<int32_t> indices(static_cast<size_t>(kMaximumFrames * kWhistleDimension));
        for (int64_t frame = 0; frame < kMaximumFrames; ++frame) {
            for (int64_t index = 0; index < kWhistleDimension; ++index) {
                indices[static_cast<size_t>(frame * kWhistleDimension + index)] = permutation[static_cast<size_t>(index)];
            }
        }
        return store_.make_tensor(
            TensorShape::from_dims({kMaximumFrames, kWhistleDimension}), GGML_TYPE_I32,
            indices.data(), indices.size() * sizeof(int32_t));
    }

    TensorValue positional(float gate) {
        std::vector<float> table(static_cast<size_t>(kMaximumFrames * kWhistleDimension));
        const int64_t half = kWhistleDimension / 2;
        for (int64_t frame = 0; frame < kMaximumFrames; ++frame) {
            for (int64_t index = 0; index < half; ++index) {
                const float angle = static_cast<float>(frame) *
                    std::exp(static_cast<float>(index) * (-std::log(10000.0f) / static_cast<float>(half - 1)));
                table[static_cast<size_t>(frame * kWhistleDimension + index)] = gate * std::sin(angle);
                table[static_cast<size_t>(frame * kWhistleDimension + half + index)] = gate * std::cos(angle);
            }
        }
        return store_.make_f32(TensorShape::from_dims({kMaximumFrames, kWhistleDimension}), std::move(table));
    }

    // Flattened 4x4 transpose as a 16x16 permutation: out[c * 4 + r] = in[r * 4 + c].
    TensorValue transpose_4x4() {
        constexpr int64_t n = kWhistleLanes;
        std::vector<float> matrix(static_cast<size_t>(n * n * n * n), 0.0f);
        for (int64_t out = 0; out < n * n; ++out) {
            const int64_t source = (out % n) * n + out / n;
            matrix[static_cast<size_t>(out * n * n + source)] = 1.0f;
        }
        return store_.make_f32(TensorShape::from_dims({n * n, n * n}), std::move(matrix));
    }

    // Identity repeated across frames, consumed through a leading-frame view.
    TensorValue identity(int64_t size) {
        std::vector<float> values(static_cast<size_t>(kMaximumFrames * size * size), 0.0f);
        for (int64_t frame = 0; frame < kMaximumFrames; ++frame) {
            for (int64_t index = 0; index < size; ++index) {
                values[static_cast<size_t>((frame * size + index) * size + index)] = 1.0f;
            }
        }
        return store_.make_f32(TensorShape::from_dims({kMaximumFrames, size, size}), std::move(values));
    }

    TensorValue lane_mean() {
        return store_.make_f32(
            TensorShape::from_dims({1, kWhistleLanes}),
            std::vector<float>(static_cast<size_t>(kWhistleLanes), 1.0f / static_cast<float>(kWhistleLanes)));
    }

private:
    core::BackendWeightStore & store_;
    const assets::TensorSource & source_;
};

float sigmoid(float value) {
    return 1.0f / (1.0f + std::exp(-value));
}

WhistleEncoderWeights load_weights(core::BackendWeightStore & store, const WhistleAssets & assets) {
    WeightLoader loader(store, *assets.weights);
    WhistleEncoderWeights weights;
    weights.stem_conv = loader.stem_conv("stem/w");
    weights.stem_dw_1 = loader.stem_conv("stem/dw_1");
    weights.stem_pw_1 = loader.pointwise("stem/pw_1");
    weights.stem_dw_2 = loader.stem_conv("stem/dw_2");
    weights.stem_pw_2 = loader.pointwise("stem/pw_2");
    weights.stem_out = loader.linear("stem/out", kStemChannels * kStemWidth, kWhistleDimension);
    for (int layer = 0; layer < kWhistleLayers; ++layer) {
        const std::string block = kEncoderLayer;
        auto & target = weights.layers[static_cast<size_t>(layer)];
        target.mhc = loader.mhc("encoder/mhc_", layer);
        target.pre_hada_norm_0 = loader.norm(block + "pre_hada_norm_0", kWhistleDimension, layer);
        target.hadamard_0 = loader.hadamard(block + "hadamard_mlp_0/", layer);
        target.attn_norm = loader.norm(block + "ZCRMSNorm_0", kWhistleDimension, layer);
        target.q_proj = loader.linear(block + "self_attn/q_proj", kWhistleDimension, kHeads * kQkDim, layer);
        target.k_proj = loader.linear(block + "self_attn/k_proj", kWhistleDimension, kKvHeads * kQkDim, layer);
        target.v_proj = loader.linear(block + "self_attn/v_proj", kWhistleDimension, kKvHeads * kVDim, layer);
        target.q_norm = loader.norm(block + "self_attn/q_norm", kQkDim, layer);
        target.k_norm = loader.norm(block + "self_attn/k_norm", kQkDim, layer);
        target.gate_proj = loader.linear(block + "self_attn/gate_proj", kWhistleDimension, kWhistleDimension, layer);
        target.out_proj = loader.linear(block + "self_attn/out_proj", kWhistleDimension, kWhistleDimension, layer);
        target.post_attn_norm = loader.norm(block + "post_attn_norm", kWhistleDimension, layer);
        target.attn_gate = sigmoid(loader.scalar(block + "attn_gate", layer));
        target.conv_norm = loader.norm(block + "conv_norm", kWhistleDimension, layer);
        target.pw1 = loader.linear(block + "pw1", kWhistleDimension, 2 * kWhistleDimension, layer);
        target.dw = loader.depthwise(block + "dw", layer);
        target.conv_out_norm = loader.norm(block + "conv_out_norm", kWhistleDimension, layer);
        target.pw2 = loader.linear(block + "pw2", kWhistleDimension, kWhistleDimension, layer);
        target.pre_hada_norm = loader.norm(block + "pre_hada_norm", kWhistleDimension, layer);
        target.hadamard = loader.hadamard(block + "hadamard_mlp/", layer);

        const std::string cross = "stack/layers/block/cross_attn/";
        auto & projection = weights.cross[static_cast<size_t>(layer)];
        projection.k_proj = loader.linear(cross + "k_proj", kWhistleDimension, kHeads * kQkDim, layer);
        projection.k_norm = loader.norm(cross + "k_norm", kQkDim, layer);
        projection.v_proj = loader.linear(cross + "v_proj", kWhistleDimension, kHeads * kVDim, layer);
    }
    weights.final_norm = loader.norm("encoder/final_norm", kWhistleDimension);
    weights.positional = loader.positional(sigmoid(loader.scalar("pe_gate")));
    weights.permutation_1 = loader.permutation(assets.hadamard_permutations[0]);
    weights.permutation_2 = loader.permutation(assets.hadamard_permutations[1]);
    weights.lane_mean = loader.lane_mean();
    weights.transpose_4x4 = loader.transpose_4x4();
    weights.identity_16 = loader.identity(16);
    weights.identity_32 = loader.identity(32);
    store.upload();
    return weights;
}

// Graph construction over ggml [features, frames] activations.
class GraphBuilder {
public:
    GraphBuilder(core::ModuleBuildContext & ctx, const WhistleEncoderWeights & weights, ggml_tensor * positions)
        : ctx_(ctx), weights_(weights), positions_(positions) {}

    ggml_tensor * stem(ggml_tensor * mel) {
        const int64_t mel_frames = mel->ne[1];
        auto grid = core::wrap_tensor(
            ggml_reshape_4d(ctx_.ggml, mel, kMels, mel_frames, 1, 1),
            TensorShape::from_dims({1, 1, mel_frames, kMels}), GGML_TYPE_F32);
        grid = modules::Conv2dModule({1, kStemChannels, 3, 3, 2, 2, 1, 1, 1, 1, false})
            .build(ctx_, grid, weights_.stem_conv);
        grid = modules::SiluModule().build(ctx_, grid);
        grid = depthwise_pointwise(grid, weights_.stem_dw_1, weights_.stem_pw_1);
        grid = depthwise_pointwise(grid, weights_.stem_dw_2, weights_.stem_pw_2);
        const int64_t frames = grid.shape.dims[2];
        if (grid.shape.dims[3] != kStemWidth || frames > kMaximumFrames) {
            throw std::runtime_error("Whistle stem produced an unexpected grid shape");
        }
        // The checkpoint flattens channels before mel-width bins: [width, channels, frames].
        ggml_tensor * grid_3d = ggml_reshape_3d(ctx_.ggml, grid.tensor, kStemWidth, frames, kStemChannels);
        ggml_tensor * flattened = ggml_reshape_2d(
            ctx_.ggml, ggml_cont(ctx_.ggml, ggml_permute(ctx_.ggml, grid_3d, 0, 2, 1, 3)),
            kStemChannels * kStemWidth, frames);
        return linear(flattened, weights_.stem_out, kStemChannels * kStemWidth, kWhistleDimension);
    }

    ggml_tensor * mhc(ggml_tensor * state, const WhistleEncoderLayerWeights & layer) {
        const int64_t frames = state->ne[1];
        const int64_t width = kWhistleLanes * kWhistleDimension;
        const auto & w = layer.mhc;
        ggml_tensor * normalized = rms_norm(state, width);
        ggml_tensor * pre = linear(normalized, w.phi_pre, width, kWhistleLanes);
        ggml_tensor * post = linear(normalized, w.phi_post, width, kWhistleLanes);
        ggml_tensor * res = linear(normalized, w.phi_res, width, kWhistleLanes * kWhistleLanes);
        ggml_tensor * h_pre = sigmoid(ggml_add(ctx_.ggml, ggml_scale(ctx_.ggml, pre, w.a_pre), w.pre_bias.tensor));
        ggml_tensor * h_post = ggml_scale(ctx_.ggml,
            sigmoid(ggml_add(ctx_.ggml, ggml_scale(ctx_.ggml, post, w.a_post), w.post_bias.tensor)), 2.0f);
        ggml_tensor * h_res = sinkhorn(
            ggml_add(ctx_.ggml, ggml_scale(ctx_.ggml, res, w.a_res), w.res_bias.tensor));

        // [lanes, features, frames] so the lane axis is the contraction axis.
        ggml_tensor * lanes = ggml_cont(ctx_.ggml, ggml_permute(ctx_.ggml,
            ggml_reshape_3d(ctx_.ggml, state, kWhistleDimension, kWhistleLanes, frames), 1, 0, 2, 3));
        ggml_tensor * mixed = ggml_reshape_2d(ctx_.ggml,
            ggml_mul_mat(ctx_.ggml, lanes, ggml_reshape_3d(ctx_.ggml, h_pre, kWhistleLanes, 1, frames)),
            kWhistleDimension, frames);
        ggml_tensor * delta = block(mixed, layer);
        ggml_tensor * difference = ggml_reshape_3d(ctx_.ggml,
            ggml_sub(ctx_.ggml, delta, mixed), kWhistleDimension, 1, frames);
        ggml_tensor * update = ggml_mul(ctx_.ggml,
            ggml_repeat_4d(ctx_.ggml, difference, kWhistleDimension, kWhistleLanes, frames, 1),
            ggml_reshape_3d(ctx_.ggml, h_post, 1, kWhistleLanes, frames));
        ggml_tensor * residual = ggml_mul_mat(ctx_.ggml, lanes,
            ggml_reshape_3d(ctx_.ggml, h_res, kWhistleLanes, kWhistleLanes, frames));
        return ggml_reshape_2d(ctx_.ggml, ggml_add(ctx_.ggml, update, residual), width, frames);
    }

    ggml_tensor * finish(ggml_tensor * state) {
        const int64_t frames = state->ne[1];
        ggml_tensor * lanes = ggml_cont(ctx_.ggml, ggml_permute(ctx_.ggml,
            ggml_reshape_3d(ctx_.ggml, state, kWhistleDimension, kWhistleLanes, frames), 1, 0, 2, 3));
        ggml_tensor * memory = ggml_reshape_2d(ctx_.ggml,
            ggml_mul_mat(ctx_.ggml, weights_.lane_mean.tensor, lanes), kWhistleDimension, frames);
        memory = gemma_norm(memory, weights_.final_norm, kWhistleDimension);
        ggml_tensor * positional = ggml_view_2d(ctx_.ggml, weights_.positional.tensor,
            kWhistleDimension, frames, weights_.positional.tensor->nb[1], 0);
        return ggml_add(ctx_.ggml, memory, positional);
    }

    ggml_tensor * cross_keys(ggml_tensor * memory, const WhistleCrossProjectionWeights & w) {
        return head_norm(linear(memory, w.k_proj, kWhistleDimension, kHeads * kQkDim), kHeads, w.k_norm);
    }

    ggml_tensor * cross_values(ggml_tensor * memory, const WhistleCrossProjectionWeights & w) {
        return linear(memory, w.v_proj, kWhistleDimension, kHeads * kVDim);
    }

private:
    TensorValue depthwise_pointwise(
        const TensorValue & grid, const modules::Conv2dWeights & depthwise, const modules::Conv2dWeights & pointwise) {
        auto conv = modules::DepthwiseConv2dModule({kStemChannels, 3, 3, 2, 2, 1, 1, 1, 1, false})
            .build(ctx_, grid, depthwise);
        conv = modules::Conv2dModule({kStemChannels, kStemChannels, 1, 1, 1, 1, 0, 0, 1, 1, false})
            .build(ctx_, conv, pointwise);
        return modules::SiluModule().build(ctx_, conv);
    }

    ggml_tensor * linear(ggml_tensor * input, const modules::LinearWeights & weights, int64_t in, int64_t out) {
        return modules::LinearModule({in, out, false}).build(ctx_, rows_2d(input), weights).tensor;
    }

    ggml_tensor * silu(ggml_tensor * input) {
        return modules::SiluModule().build(ctx_, rows_2d(input)).tensor;
    }

    ggml_tensor * sigmoid(ggml_tensor * input) {
        return modules::SigmoidModule().build(ctx_, rows_2d(input)).tensor;
    }

    ggml_tensor * softmax(ggml_tensor * input) {
        return modules::SoftmaxModule().build(ctx_, rows_2d(input)).tensor;
    }

    ggml_tensor * rms_norm(ggml_tensor * input, int64_t size) {
        return modules::RMSNormModule({size, kNormEpsilon, false, false}).build(ctx_, rows_2d(input), {}).tensor;
    }

    ggml_tensor * gemma_norm(ggml_tensor * input, const modules::NormWeights & weights, int64_t size) {
        return modules::GemmaRMSNormModule({size, kNormEpsilon, true, false}).build(ctx_, rows_2d(input), weights).tensor;
    }

    // Per-head Gemma RMS norm over [heads * dim, frames], returned flat again.
    ggml_tensor * head_norm(ggml_tensor * input, int64_t heads, const modules::NormWeights & weights) {
        const int64_t frames = input->ne[1];
        auto split = core::wrap_tensor(
            ggml_reshape_4d(ctx_.ggml, input, kQkDim, heads, frames, 1),
            TensorShape::from_dims({1, frames, heads, kQkDim}), GGML_TYPE_F32);
        auto normalized = modules::GemmaRMSNormModule({kQkDim, kNormEpsilon, true, false}).build(ctx_, split, weights);
        return ggml_reshape_2d(ctx_.ggml, normalized.tensor, heads * kQkDim, frames);
    }

    TensorValue rope(ggml_tensor * input, int64_t heads) {
        const int64_t frames = input->ne[1];
        auto split = core::wrap_tensor(
            ggml_reshape_4d(ctx_.ggml, input, kQkDim, heads, frames, 1),
            TensorShape::from_dims({1, frames, heads, kQkDim}), GGML_TYPE_F32);
        const auto positions = core::wrap_tensor(positions_, TensorShape::from_dims({frames}), GGML_TYPE_I32);
        return modules::RoPEModule({kQkDim, GGML_ROPE_TYPE_NEOX, kRopeTheta, 1.0f}).build(ctx_, split, positions);
    }

    // Subtracts each row's log-sum-exp. ggml has no max reduction, so the log-sum-exp
    // is recovered from the softmax as sum(p * (x - log p)): every term with a
    // non-underflowed p equals the log-sum-exp, and the clamp keeps underflowed terms
    // at zero instead of 0 * -inf. Entries far below the row maximum keep their exact
    // x - lse value, as in the host implementation.
    ggml_tensor * log_normalize_rows(ggml_tensor * x) {
        const auto rows = core::wrap_tensor(x, TensorShape::from_dims({x->ne[2], x->ne[1], x->ne[0]}), GGML_TYPE_F32);
        ggml_tensor * p = ggml_clamp(ctx_.ggml, modules::SoftmaxModule().build(ctx_, rows).tensor,
            std::numeric_limits<float>::min(), 1.0f);
        ggml_tensor * lse = ggml_sum_rows(ctx_.ggml,
            ggml_mul(ctx_.ggml, p, ggml_sub(ctx_.ggml, x, ggml_log(ctx_.ggml, p))));
        return ggml_sub(ctx_.ggml, x, lse);
    }

    // Iterative row/column log-normalization of the 4x4 lane-mixing logits.
    ggml_tensor * sinkhorn(ggml_tensor * logits) {
        const int64_t frames = logits->ne[1];
        const int64_t size = kWhistleLanes * kWhistleLanes;
        ggml_tensor * x = logits;
        for (int iteration = 0; iteration < kSinkhornIterations; ++iteration) {
            for (int pass = 0; pass < 2; ++pass) {
                x = log_normalize_rows(ggml_reshape_3d(ctx_.ggml, x, kWhistleLanes, kWhistleLanes, frames));
                x = ggml_mul_mat(ctx_.ggml, weights_.transpose_4x4.tensor, ggml_reshape_2d(ctx_.ggml, x, size, frames));
            }
        }
        return ggml_exp(ctx_.ggml, x);
    }

    ggml_tensor * identity_view(const TensorValue & identity, int64_t frames) {
        ggml_tensor * tensor = identity.tensor;
        return ggml_view_3d(ctx_.ggml, tensor, tensor->ne[0], tensor->ne[1], frames, tensor->nb[1], tensor->nb[2], 0);
    }

    // (A kron B) applied to each 512-vector viewed as a 16x32 matrix: A mixes rows,
    // B mixes columns. ggml_mul_mat contracts over the leading axis, so the matrix is
    // transposed between the two products by multiplying with a batched identity.
    ggml_tensor * kronecker(ggml_tensor * input, const TensorValue & a, const TensorValue & b) {
        const int64_t frames = input->ne[1];
        ggml_tensor * columns_mixed = ggml_mul_mat(ctx_.ggml, b.tensor, ggml_reshape_3d(ctx_.ggml, input, 32, 16, frames));
        ggml_tensor * by_row = ggml_mul_mat(ctx_.ggml, columns_mixed, identity_view(weights_.identity_32, frames));
        ggml_tensor * rows_mixed = ggml_mul_mat(ctx_.ggml, a.tensor, by_row);
        ggml_tensor * by_column = ggml_mul_mat(ctx_.ggml, rows_mixed, identity_view(weights_.identity_16, frames));
        return ggml_reshape_2d(ctx_.ggml, by_column, kWhistleDimension, frames);
    }

    // Permutes the feature axis of a [features, frames] activation: each feature is a
    // one-element row, gathered per frame with a batched index tensor.
    ggml_tensor * permute_features(ggml_tensor * input, const TensorValue & indices) {
        const int64_t frames = input->ne[1];
        ggml_tensor * index_view = ggml_view_2d(
            ctx_.ggml, indices.tensor, kWhistleDimension, frames, indices.tensor->nb[1], 0);
        ggml_tensor * gathered = ggml_get_rows(
            ctx_.ggml, ggml_reshape_3d(ctx_.ggml, input, 1, kWhistleDimension, frames), index_view);
        return ggml_reshape_2d(ctx_.ggml, gathered, kWhistleDimension, frames);
    }

    ggml_tensor * hadamard(ggml_tensor * input, const WhistleHadamardWeights & w) {
        ggml_tensor * condition = softmax(linear(input, w.cond_v, kWhistleDimension, 8));
        condition = ggml_scale_bias(ctx_.ggml, linear(condition, w.cond_u, 8, kWhistleDimension), 1.0f, 1.0f);
        ggml_tensor * z = ggml_mul(ctx_.ggml, input, w.d1.tensor);
        z = permute_features(kronecker(z, w.w1a, w.w1b), weights_.permutation_1);
        z = ggml_add(ctx_.ggml, ggml_mul(ctx_.ggml, ggml_mul(ctx_.ggml, z, condition), w.d2.tensor), w.b2.tensor);
        z = silu(z);
        z = permute_features(kronecker(z, w.w2a, w.w2b), weights_.permutation_2);
        z = ggml_mul(ctx_.ggml, z, w.d3.tensor);
        return ggml_mul(ctx_.ggml, kronecker(z, w.w3a, w.w3b), w.d4.tensor);
    }

    // Grouped-query attention with 48-wide queries/keys and 64-wide values, so the
    // shared attention modules (which require one head size) do not apply.
    ggml_tensor * attention(ggml_tensor * input, const WhistleEncoderLayerWeights & w) {
        const int64_t frames = input->ne[1];
        const int64_t group = kHeads / kKvHeads;
        auto q = rope(head_norm(linear(input, w.q_proj, kWhistleDimension, kHeads * kQkDim), kHeads, w.q_norm), kHeads);
        auto k = rope(head_norm(linear(input, w.k_proj, kWhistleDimension, kKvHeads * kQkDim), kKvHeads, w.k_norm), kKvHeads);
        ggml_tensor * v = linear(input, w.v_proj, kWhistleDimension, kKvHeads * kVDim);

        ggml_tensor * queries = ggml_cont(ctx_.ggml, ggml_permute(ctx_.ggml,
            ggml_reshape_4d(ctx_.ggml, q.tensor, kQkDim, group, kKvHeads, frames), 0, 2, 3, 1));
        ggml_tensor * keys = ggml_cont(ctx_.ggml, ggml_permute(ctx_.ggml,
            ggml_reshape_4d(ctx_.ggml, k.tensor, kQkDim, 1, kKvHeads, frames), 0, 2, 3, 1));
        ggml_tensor * scores = ggml_mul_mat(ctx_.ggml, keys, queries);
        ggml_tensor * probabilities = ggml_soft_max_ext(
            ctx_.ggml, scores, nullptr, 1.0f / std::sqrt(static_cast<float>(kQkDim)), 0.0f);
        ggml_tensor * values = ggml_cont(ctx_.ggml, ggml_permute(ctx_.ggml,
            ggml_reshape_4d(ctx_.ggml, v, kVDim, 1, kKvHeads, frames), 1, 2, 3, 0));
        ggml_tensor * context = ggml_mul_mat(ctx_.ggml, values, probabilities);
        context = ggml_reshape_2d(ctx_.ggml,
            ggml_cont(ctx_.ggml, ggml_permute(ctx_.ggml, context, 0, 3, 1, 2)), kHeads * kVDim, frames);
        ggml_tensor * gate = sigmoid(linear(input, w.gate_proj, kWhistleDimension, kWhistleDimension));
        return linear(ggml_mul(ctx_.ggml, context, gate), w.out_proj, kWhistleDimension, kWhistleDimension);
    }

    ggml_tensor * convolution(ggml_tensor * input, const WhistleEncoderLayerWeights & w) {
        const int64_t frames = input->ne[1];
        ggml_tensor * c = gemma_norm(input, w.conv_norm, kWhistleDimension);
        c = linear(c, w.pw1, kWhistleDimension, 2 * kWhistleDimension);
        c = modules::GLUModule().build(ctx_, rows_2d(c)).tensor;
        auto time_major = core::wrap_tensor(
            ggml_cont(ctx_.ggml, ggml_transpose(ctx_.ggml, c)),
            TensorShape::from_dims({1, kWhistleDimension, frames}), GGML_TYPE_F32);
        auto conv = modules::DepthwiseConv1dModule({kWhistleDimension, kConvTaps, 1, (kConvTaps - 1) / 2, 1, false})
            .build(ctx_, time_major, w.dw);
        c = ggml_cont(ctx_.ggml, ggml_transpose(ctx_.ggml,
            ggml_reshape_2d(ctx_.ggml, conv.tensor, frames, kWhistleDimension)));
        c = silu(gemma_norm(c, w.conv_out_norm, kWhistleDimension));
        return linear(c, w.pw2, kWhistleDimension, kWhistleDimension);
    }

    ggml_tensor * block(ggml_tensor * input, const WhistleEncoderLayerWeights & w) {
        ggml_tensor * h = input;
        h = ggml_add(ctx_.ggml, h, ggml_scale(ctx_.ggml,
            hadamard(gemma_norm(h, w.pre_hada_norm_0, kWhistleDimension), w.hadamard_0), 0.5f));
        ggml_tensor * attended = attention(gemma_norm(h, w.attn_norm, kWhistleDimension), w);
        attended = gemma_norm(attended, w.post_attn_norm, kWhistleDimension);
        h = ggml_add(ctx_.ggml, h, ggml_scale(ctx_.ggml, attended, w.attn_gate));
        h = ggml_add(ctx_.ggml, h, convolution(h, w));
        return ggml_add(ctx_.ggml, h, ggml_scale(ctx_.ggml,
            hadamard(gemma_norm(h, w.pre_hada_norm, kWhistleDimension), w.hadamard), 0.5f));
    }

    core::ModuleBuildContext & ctx_;
    const WhistleEncoderWeights & weights_;
    ggml_tensor * positions_;
};

// Output views share storage with their producing node; the allocator keeps a
// node alive for readback only when the node itself carries the output flag.
ggml_tensor * mark_output(ggml_tensor * tensor) {
    ggml_tensor * base = tensor;
    while (base->view_src != nullptr) {
        base = base->view_src;
    }
    ggml_set_output(base);
    return base;
}

std::vector<float> read_output(const ggml_tensor * tensor) {
    std::vector<float> values(static_cast<size_t>(ggml_nelements(tensor)));
    ggml_backend_tensor_get(tensor, values.data(), 0, values.size() * sizeof(float));
    return values;
}

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept { ggml_free(ctx); }
};

struct GallocrDeleter {
    void operator()(ggml_gallocr * galloc) const noexcept { ggml_gallocr_free(galloc); }
};

}  // namespace

WhistleEncoderRuntime::WhistleEncoderRuntime(
    std::shared_ptr<const WhistleAssets> assets, core::ExecutionContext & execution_context)
    : assets_(std::move(assets)),
      execution_context_(&execution_context),
      weight_store_(
          execution_context.backend(), execution_context.backend_type(),
          "Whistle encoder weights", 8ull * 1024ull * 1024ull) {
    if (!assets_ || !assets_->weights) {
        throw std::invalid_argument("Whistle encoder needs verified model assets");
    }
    weights_ = load_weights(weight_store_, *assets_);
}

WhistleEncoderOutput WhistleEncoderRuntime::encode(const MelFeatures & mel) {
    if (mel.frames == 0 || mel.values.size() != mel.frames * static_cast<size_t>(kMels)) {
        throw std::invalid_argument("Whistle encoder needs non-empty 80-bin mel features");
    }
    ggml_init_params params{};
    params.mem_size = ggml_tensor_overhead() * kGraphNodes + ggml_graph_overhead_custom(kGraphNodes, false);
    params.no_alloc = true;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ggml_ctx(ggml_init(params));
    if (!ggml_ctx) {
        throw std::runtime_error("Failed to initialize the Whistle encoder graph context");
    }
    std::unique_ptr<ggml_gallocr, GallocrDeleter> galloc(
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_context_->backend())));
    if (!galloc) {
        throw std::runtime_error("Failed to initialize the Whistle encoder graph allocator");
    }

    core::ModuleBuildContext ctx{ggml_ctx.get(), "whistle_asr_encoder", execution_context_->backend_type()};
    ggml_tensor * mel_input = ggml_new_tensor_2d(ctx.ggml, GGML_TYPE_F32, kMels, static_cast<int64_t>(mel.frames));
    ggml_set_input(mel_input);
    int64_t frames = static_cast<int64_t>(mel.frames);
    for (int stage = 0; stage < 3; ++stage) {
        frames = (frames + 1) / 2;
    }
    ggml_tensor * positions = ggml_new_tensor_1d(ctx.ggml, GGML_TYPE_I32, frames);
    ggml_set_input(positions);

    GraphBuilder builder(ctx, weights_, positions);
    ggml_tensor * projected = builder.stem(mel_input);
    ggml_tensor * state = ggml_reshape_2d(ctx.ggml,
        ggml_repeat_4d(ctx.ggml, ggml_reshape_3d(ctx.ggml, projected, kWhistleDimension, 1, frames),
            kWhistleDimension, kWhistleLanes, frames, 1),
        kWhistleLanes * kWhistleDimension, frames);
    for (int layer = 0; layer < kWhistleLayers; ++layer) {
        state = builder.mhc(state, weights_.layers[static_cast<size_t>(layer)]);
    }
    ggml_tensor * memory = mark_output(builder.finish(state));
    std::array<ggml_tensor *, kWhistleLayers> cross_k{};
    std::array<ggml_tensor *, kWhistleLayers> cross_v{};
    for (int layer = 0; layer < kWhistleLayers; ++layer) {
        const auto & projection = weights_.cross[static_cast<size_t>(layer)];
        cross_k[static_cast<size_t>(layer)] = mark_output(builder.cross_keys(memory, projection));
        cross_v[static_cast<size_t>(layer)] = mark_output(builder.cross_values(memory, projection));
    }

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.ggml, kGraphNodes, false);
    ggml_build_forward_expand(graph, memory);
    for (int layer = 0; layer < kWhistleLayers; ++layer) {
        ggml_build_forward_expand(graph, cross_k[static_cast<size_t>(layer)]);
        ggml_build_forward_expand(graph, cross_v[static_cast<size_t>(layer)]);
    }
    core::validate_backend_graph_supported(execution_context_->backend(), graph, "Whistle encoder");
    if (!ggml_gallocr_alloc_graph(galloc.get(), graph)) {
        throw std::runtime_error("Failed to allocate the Whistle encoder graph");
    }
    ggml_backend_tensor_set(mel_input, mel.values.data(), 0, mel.values.size() * sizeof(float));
    std::vector<int32_t> position_values(static_cast<size_t>(frames));
    std::iota(position_values.begin(), position_values.end(), 0);
    ggml_backend_tensor_set(positions, position_values.data(), 0, position_values.size() * sizeof(int32_t));
    if (ggml_backend_graph_compute(execution_context_->backend(), graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("Failed to compute the Whistle encoder graph");
    }

    WhistleEncoderOutput output;
    output.frames = static_cast<size_t>(frames);
    output.memory = read_output(memory);
    for (int layer = 0; layer < kWhistleLayers; ++layer) {
        output.cross_k[static_cast<size_t>(layer)] = read_output(cross_k[static_cast<size_t>(layer)]);
        output.cross_v[static_cast<size_t>(layer)] = read_output(cross_v[static_cast<size_t>(layer)]);
    }
    return output;
}

}  // namespace engine::community_models::whistle_asr
