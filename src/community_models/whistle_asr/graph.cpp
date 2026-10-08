#include "engine/community_models/whistle_asr/graph.h"

#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/positional_modules.h"

#include <ggml.h>

#include <cstddef>
#include <limits>
#include <optional>
#include <utility>

namespace engine::community_models::whistle_asr {
namespace {

constexpr float kNormEpsilon = 1.0e-6f;
constexpr int kSinkhornIterations = 20;
constexpr float kRopeTheta = 100000.0f;

using core::TensorShape;
using core::TensorValue;

TensorValue rows_2d(ggml_tensor * tensor) {
    return core::wrap_tensor(tensor, TensorShape::from_dims({tensor->ne[1], tensor->ne[0]}), GGML_TYPE_F32);
}

}  // namespace

WhistleWeightLoader::WhistleWeightLoader(core::BackendWeightStore & store, const assets::TensorSource & source)
    : store_(store), source_(source) {}

std::vector<float> WhistleWeightLoader::values(const std::string & name, std::vector<int64_t> shape, int layer) const {
    if (layer < 0) {
        return source_.require_f32(name, shape);
    }
    shape.insert(shape.begin(), kWhistleLayers);
    const auto stacked = source_.require_f32(name, shape);
    const size_t stride = stacked.size() / kWhistleLayers;
    return {stacked.begin() + static_cast<std::ptrdiff_t>(layer * stride),
        stacked.begin() + static_cast<std::ptrdiff_t>((layer + 1) * stride)};
}

float WhistleWeightLoader::scalar(const std::string & name, int layer) const {
    return values(name, {}, layer).front();
}

TensorValue WhistleWeightLoader::vector(const std::string & name, int64_t size, int layer) {
    return store_.make_f32(TensorShape::from_dims({size}), values(name, {size}, layer));
}

modules::NormWeights WhistleWeightLoader::norm(const std::string & name, int64_t size, int layer) {
    return {vector(name + "/scale", size, layer), std::nullopt};
}

TensorValue WhistleWeightLoader::transposed(const std::string & name, int64_t in, int64_t out, int layer) {
    const auto source = values(name, {in, out}, layer);
    std::vector<float> weight(source.size());
    for (int64_t row = 0; row < in; ++row) {
        for (int64_t col = 0; col < out; ++col) {
            weight[static_cast<size_t>(col * in + row)] = source[static_cast<size_t>(row * out + col)];
        }
    }
    return store_.make_f32(TensorShape::from_dims({out, in}), std::move(weight));
}

modules::LinearWeights WhistleWeightLoader::linear(const std::string & name, int64_t in, int64_t out, int layer) {
    return {transposed(name + "/kernel", in, out, layer), std::nullopt};
}

WhistleHadamardWeights WhistleWeightLoader::hadamard(const std::string & prefix, int layer) {
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

WhistleMhcWeights WhistleWeightLoader::mhc(const std::string & prefix, int layer) {
    WhistleMhcWeights weights;
    const int64_t width = kWhistleLanes * kWhistleDimension;
    weights.phi_pre = {transposed(prefix + "phi_pre", width, kWhistleLanes, layer), std::nullopt};
    weights.phi_post = {transposed(prefix + "phi_post", width, kWhistleLanes, layer), std::nullopt};
    weights.phi_res = {transposed(prefix + "phi_res", width, kWhistleLanes * kWhistleLanes, layer), std::nullopt};
    weights.a_pre = scalar(prefix + "a_pre", layer);
    weights.a_post = scalar(prefix + "a_post", layer);
    weights.a_res = scalar(prefix + "a_res", layer);
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

// Gather indices repeated per row, consumed through a leading-row view.
TensorValue WhistleWeightLoader::permutation(const std::array<uint16_t, 512> & permutation, int64_t rows) {
    std::vector<int32_t> indices(static_cast<size_t>(rows * kWhistleDimension));
    for (int64_t row = 0; row < rows; ++row) {
        for (int64_t index = 0; index < kWhistleDimension; ++index) {
            indices[static_cast<size_t>(row * kWhistleDimension + index)] = permutation[static_cast<size_t>(index)];
        }
    }
    return store_.make_tensor(
        TensorShape::from_dims({rows, kWhistleDimension}), GGML_TYPE_I32,
        indices.data(), indices.size() * sizeof(int32_t));
}

// Identity repeated across rows, consumed through a leading-row view.
TensorValue WhistleWeightLoader::identity(int64_t size, int64_t rows) {
    std::vector<float> values(static_cast<size_t>(rows * size * size), 0.0f);
    for (int64_t row = 0; row < rows; ++row) {
        for (int64_t index = 0; index < size; ++index) {
            values[static_cast<size_t>((row * size + index) * size + index)] = 1.0f;
        }
    }
    return store_.make_f32(TensorShape::from_dims({rows, size, size}), std::move(values));
}

WhistleGraphConstants WhistleWeightLoader::constants(const WhistleAssets & assets, int64_t rows) {
    WhistleGraphConstants constants;
    constants.permutation_1 = permutation(assets.hadamard_permutations[0], rows);
    constants.permutation_2 = permutation(assets.hadamard_permutations[1], rows);
    constants.lane_mean = store_.make_f32(
        TensorShape::from_dims({1, kWhistleLanes}),
        std::vector<float>(static_cast<size_t>(kWhistleLanes), 1.0f / static_cast<float>(kWhistleLanes)));
    // Flattened 4x4 transpose as a 16x16 permutation: out[c * 4 + r] = in[r * 4 + c].
    constexpr int64_t n = kWhistleLanes;
    std::vector<float> transpose(static_cast<size_t>(n * n * n * n), 0.0f);
    for (int64_t out = 0; out < n * n; ++out) {
        const int64_t source = (out % n) * n + out / n;
        transpose[static_cast<size_t>(out * n * n + source)] = 1.0f;
    }
    constants.transpose_4x4 = store_.make_f32(TensorShape::from_dims({n * n, n * n}), std::move(transpose));
    constants.identity_16 = identity(16, rows);
    constants.identity_32 = identity(32, rows);
    return constants;
}

WhistleGraphOps::WhistleGraphOps(core::ModuleBuildContext & ctx, const WhistleGraphConstants & constants)
    : ctx_(ctx), constants_(constants) {}

ggml_tensor * WhistleGraphOps::linear(
    ggml_tensor * input, const modules::LinearWeights & weights, int64_t in, int64_t out) {
    return modules::LinearModule({in, out, false}).build(ctx_, rows_2d(input), weights).tensor;
}

ggml_tensor * WhistleGraphOps::silu(ggml_tensor * input) {
    return modules::SiluModule().build(ctx_, rows_2d(input)).tensor;
}

ggml_tensor * WhistleGraphOps::sigmoid(ggml_tensor * input) {
    return modules::SigmoidModule().build(ctx_, rows_2d(input)).tensor;
}

ggml_tensor * WhistleGraphOps::softmax(ggml_tensor * input) {
    return modules::SoftmaxModule().build(ctx_, rows_2d(input)).tensor;
}

ggml_tensor * WhistleGraphOps::rms_norm(ggml_tensor * input, int64_t size) {
    return modules::RMSNormModule({size, kNormEpsilon, false, false}).build(ctx_, rows_2d(input), {}).tensor;
}

ggml_tensor * WhistleGraphOps::gemma_norm(ggml_tensor * input, const modules::NormWeights & weights, int64_t size) {
    return modules::GemmaRMSNormModule({size, kNormEpsilon, true, false}).build(ctx_, rows_2d(input), weights).tensor;
}

ggml_tensor * WhistleGraphOps::head_norm(ggml_tensor * input, int64_t heads, const modules::NormWeights & weights) {
    const int64_t rows = input->ne[1];
    auto split = core::wrap_tensor(
        ggml_reshape_4d(ctx_.ggml, input, kWhistleQkDim, heads, rows, 1),
        TensorShape::from_dims({1, rows, heads, kWhistleQkDim}), GGML_TYPE_F32);
    auto normalized = modules::GemmaRMSNormModule({kWhistleQkDim, kNormEpsilon, true, false}).build(ctx_, split, weights);
    return ggml_reshape_2d(ctx_.ggml, normalized.tensor, heads * kWhistleQkDim, rows);
}

TensorValue WhistleGraphOps::rope(ggml_tensor * input, int64_t heads, ggml_tensor * positions) {
    const int64_t rows = input->ne[1];
    auto split = core::wrap_tensor(
        ggml_reshape_4d(ctx_.ggml, input, kWhistleQkDim, heads, rows, 1),
        TensorShape::from_dims({1, rows, heads, kWhistleQkDim}), GGML_TYPE_F32);
    const auto position_values = core::wrap_tensor(positions, TensorShape::from_dims({rows}), GGML_TYPE_I32);
    return modules::RoPEModule({kWhistleQkDim, GGML_ROPE_TYPE_NEOX, kRopeTheta, 1.0f}).build(ctx_, split, position_values);
}

// Subtracts each row's log-sum-exp. ggml has no max reduction, so the log-sum-exp
// is recovered from the softmax as sum(p * (x - log p)). Every term whose p has not
// underflowed equals the log-sum-exp. The clamp keeps the other terms at zero
// instead of 0 * -inf. Entries far below the row maximum keep their exact
// x - lse value.
ggml_tensor * WhistleGraphOps::log_normalize_rows(ggml_tensor * x) {
    const auto rows = core::wrap_tensor(x, TensorShape::from_dims({x->ne[2], x->ne[1], x->ne[0]}), GGML_TYPE_F32);
    ggml_tensor * p = ggml_clamp(ctx_.ggml, modules::SoftmaxModule().build(ctx_, rows).tensor,
        std::numeric_limits<float>::min(), 1.0f);
    ggml_tensor * lse = ggml_sum_rows(ctx_.ggml,
        ggml_mul(ctx_.ggml, p, ggml_sub(ctx_.ggml, x, ggml_log(ctx_.ggml, p))));
    return ggml_sub(ctx_.ggml, x, lse);
}

// Twenty rounds of row then column log-normalization of the 4x4 lane-mixing
// logits, followed by exp. Each pass normalizes the rows and then transposes with
// the transpose_4x4 constant, so the row pass and the column pass share one code
// path. Each pass subtracts a per-row log-sum-exp in log space, so no whole row
// or column can underflow.
ggml_tensor * WhistleGraphOps::sinkhorn(ggml_tensor * logits) {
    const int64_t rows = logits->ne[1];
    const int64_t size = kWhistleLanes * kWhistleLanes;
    ggml_tensor * x = logits;
    for (int iteration = 0; iteration < kSinkhornIterations; ++iteration) {
        for (int pass = 0; pass < 2; ++pass) {
            x = log_normalize_rows(ggml_reshape_3d(ctx_.ggml, x, kWhistleLanes, kWhistleLanes, rows));
            x = ggml_mul_mat(ctx_.ggml, constants_.transpose_4x4.tensor, ggml_reshape_2d(ctx_.ggml, x, size, rows));
        }
    }
    return ggml_exp(ctx_.ggml, x);
}

ggml_tensor * WhistleGraphOps::identity_view(const TensorValue & identity, int64_t rows) {
    ggml_tensor * tensor = identity.tensor;
    return ggml_view_3d(ctx_.ggml, tensor, tensor->ne[0], tensor->ne[1], rows, tensor->nb[1], tensor->nb[2], 0);
}

// (A kron B) applied to each 512-vector viewed as a 16x32 matrix: A mixes rows,
// B mixes columns. ggml_mul_mat contracts over the leading axis, so the matrix is
// transposed between the two products by multiplying with a batched identity.
ggml_tensor * WhistleGraphOps::kronecker(ggml_tensor * input, const TensorValue & a, const TensorValue & b) {
    const int64_t rows = input->ne[1];
    ggml_tensor * columns_mixed = ggml_mul_mat(ctx_.ggml, b.tensor, ggml_reshape_3d(ctx_.ggml, input, 32, 16, rows));
    ggml_tensor * by_row = ggml_mul_mat(ctx_.ggml, columns_mixed, identity_view(constants_.identity_32, rows));
    ggml_tensor * rows_mixed = ggml_mul_mat(ctx_.ggml, a.tensor, by_row);
    ggml_tensor * by_column = ggml_mul_mat(ctx_.ggml, rows_mixed, identity_view(constants_.identity_16, rows));
    return ggml_reshape_2d(ctx_.ggml, by_column, kWhistleDimension, rows);
}

// Permutes the feature axis of a [features, rows] activation: each feature is a
// one-element row, gathered per row with a batched index tensor.
ggml_tensor * WhistleGraphOps::permute_features(ggml_tensor * input, const TensorValue & indices) {
    const int64_t rows = input->ne[1];
    ggml_tensor * index_view = ggml_view_2d(
        ctx_.ggml, indices.tensor, kWhistleDimension, rows, indices.tensor->nb[1], 0);
    ggml_tensor * gathered = ggml_get_rows(
        ctx_.ggml, ggml_reshape_3d(ctx_.ggml, input, 1, kWhistleDimension, rows), index_view);
    return ggml_reshape_2d(ctx_.ggml, gathered, kWhistleDimension, rows);
}

ggml_tensor * WhistleGraphOps::hadamard(ggml_tensor * input, const WhistleHadamardWeights & w) {
    ggml_tensor * condition = softmax(linear(input, w.cond_v, kWhistleDimension, 8));
    condition = ggml_scale_bias(ctx_.ggml, linear(condition, w.cond_u, 8, kWhistleDimension), 1.0f, 1.0f);
    ggml_tensor * z = ggml_mul(ctx_.ggml, input, w.d1.tensor);
    z = permute_features(kronecker(z, w.w1a, w.w1b), constants_.permutation_1);
    z = ggml_add(ctx_.ggml, ggml_mul(ctx_.ggml, ggml_mul(ctx_.ggml, z, condition), w.d2.tensor), w.b2.tensor);
    z = silu(z);
    z = permute_features(kronecker(z, w.w2a, w.w2b), constants_.permutation_2);
    z = ggml_mul(ctx_.ggml, z, w.d3.tensor);
    return ggml_mul(ctx_.ggml, kronecker(z, w.w3a, w.w3b), w.d4.tensor);
}

ggml_tensor * WhistleGraphOps::mhc(
    ggml_tensor * state, const WhistleMhcWeights & w, const std::function<ggml_tensor *(ggml_tensor *)> & block) {
    const int64_t rows = state->ne[1];
    const int64_t width = kWhistleLanes * kWhistleDimension;
    ggml_tensor * normalized = rms_norm(state, width);
    ggml_tensor * pre = linear(normalized, w.phi_pre, width, kWhistleLanes);
    ggml_tensor * post = linear(normalized, w.phi_post, width, kWhistleLanes);
    ggml_tensor * res = linear(normalized, w.phi_res, width, kWhistleLanes * kWhistleLanes);
    ggml_tensor * h_pre = sigmoid(ggml_add(ctx_.ggml, ggml_scale(ctx_.ggml, pre, w.a_pre), w.pre_bias.tensor));
    ggml_tensor * h_post = ggml_scale(ctx_.ggml,
        sigmoid(ggml_add(ctx_.ggml, ggml_scale(ctx_.ggml, post, w.a_post), w.post_bias.tensor)), 2.0f);
    ggml_tensor * h_res = sinkhorn(
        ggml_add(ctx_.ggml, ggml_scale(ctx_.ggml, res, w.a_res), w.res_bias.tensor));

    // [lanes, features, rows] so the lane axis is the contraction axis.
    ggml_tensor * lanes = ggml_cont(ctx_.ggml, ggml_permute(ctx_.ggml,
        ggml_reshape_3d(ctx_.ggml, state, kWhistleDimension, kWhistleLanes, rows), 1, 0, 2, 3));
    ggml_tensor * mixed = ggml_reshape_2d(ctx_.ggml,
        ggml_mul_mat(ctx_.ggml, lanes, ggml_reshape_3d(ctx_.ggml, h_pre, kWhistleLanes, 1, rows)),
        kWhistleDimension, rows);
    ggml_tensor * delta = block(mixed);
    ggml_tensor * difference = ggml_reshape_3d(ctx_.ggml,
        ggml_sub(ctx_.ggml, delta, mixed), kWhistleDimension, 1, rows);
    ggml_tensor * update = ggml_mul(ctx_.ggml,
        ggml_repeat_4d(ctx_.ggml, difference, kWhistleDimension, kWhistleLanes, rows, 1),
        ggml_reshape_3d(ctx_.ggml, h_post, 1, kWhistleLanes, rows));
    ggml_tensor * residual = ggml_mul_mat(ctx_.ggml, lanes,
        ggml_reshape_3d(ctx_.ggml, h_res, kWhistleLanes, kWhistleLanes, rows));
    return ggml_reshape_2d(ctx_.ggml, ggml_add(ctx_.ggml, update, residual), width, rows);
}

ggml_tensor * WhistleGraphOps::lane_mean(ggml_tensor * state) {
    const int64_t rows = state->ne[1];
    ggml_tensor * lanes = ggml_cont(ctx_.ggml, ggml_permute(ctx_.ggml,
        ggml_reshape_3d(ctx_.ggml, state, kWhistleDimension, kWhistleLanes, rows), 1, 0, 2, 3));
    return ggml_reshape_2d(ctx_.ggml,
        ggml_mul_mat(ctx_.ggml, constants_.lane_mean.tensor, lanes), kWhistleDimension, rows);
}

}  // namespace engine::community_models::whistle_asr
