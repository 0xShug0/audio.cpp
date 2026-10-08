#include "engine/community_models/whistle_asr/encoder.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/modules/activation_modules.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

#include <cmath>
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
constexpr int64_t kMaximumFrames = kWhistleMaximumFrames;
constexpr int64_t kMels = 80;
constexpr int64_t kStemChannels = 128;
constexpr int64_t kStemWidth = 10;
constexpr int64_t kHeads = kWhistleHeads;
constexpr int64_t kKvHeads = kWhistleKvHeads;
constexpr int64_t kQkDim = kWhistleQkDim;
constexpr int64_t kVDim = kWhistleVDim;
constexpr int64_t kConvTaps = 9;
constexpr const char * kEncoderLayer = "encoder/layers/block/";

using core::TensorShape;
using core::TensorValue;

TensorValue rows_2d(ggml_tensor * tensor) {
    return core::wrap_tensor(tensor, TensorShape::from_dims({tensor->ne[1], tensor->ne[0]}), GGML_TYPE_F32);
}

// Adds the encoder-only stem, depthwise convolution and positional tables to the
// shared loader.
class EncoderWeightLoader : public WhistleWeightLoader {
public:
    using WhistleWeightLoader::WhistleWeightLoader;

    // Pointwise stem projections as 1x1 convolutions, [out, in, 1, 1].
    modules::Conv2dWeights pointwise(const std::string & name) {
        const auto source = values(name + "/kernel", {kStemChannels, kStemChannels});
        std::vector<float> weight(source.size());
        for (int64_t in = 0; in < kStemChannels; ++in) {
            for (int64_t out = 0; out < kStemChannels; ++out) {
                weight[static_cast<size_t>(out * kStemChannels + in)] = source[static_cast<size_t>(in * kStemChannels + out)];
            }
        }
        return {store().make_f32(TensorShape::from_dims({kStemChannels, kStemChannels, 1, 1}), std::move(weight)), std::nullopt};
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
        return {store().make_f32(TensorShape::from_dims({kStemChannels, 1, 3, 3}), std::move(weight)), std::nullopt};
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
        return {store().make_f32(TensorShape::from_dims({kWhistleDimension, 1, kConvTaps}), std::move(weight)),
            std::nullopt};
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
        return store().make_f32(TensorShape::from_dims({kMaximumFrames, kWhistleDimension}), std::move(table));
    }
};

float sigmoid(float value) {
    return 1.0f / (1.0f + std::exp(-value));
}

WhistleEncoderWeights load_weights(core::BackendWeightStore & store, const WhistleAssets & assets) {
    EncoderWeightLoader loader(store, *assets.weights);
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
    weights.constants = loader.constants(assets, kMaximumFrames);
    store.upload();
    return weights;
}

// Graph construction over ggml [features, frames] activations.
class GraphBuilder {
public:
    GraphBuilder(core::ModuleBuildContext & ctx, const WhistleEncoderWeights & weights, ggml_tensor * positions)
        : ctx_(ctx), ops_(ctx, weights.constants), weights_(weights), positions_(positions) {}

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
        return ops_.linear(flattened, weights_.stem_out, kStemChannels * kStemWidth, kWhistleDimension);
    }

    ggml_tensor * layer(ggml_tensor * state, const WhistleEncoderLayerWeights & layer) {
        return ops_.mhc(state, layer.mhc, [&](ggml_tensor * mixed) { return block(mixed, layer); });
    }

    ggml_tensor * finish(ggml_tensor * state) {
        const int64_t frames = state->ne[1];
        ggml_tensor * memory = ops_.gemma_norm(ops_.lane_mean(state), weights_.final_norm, kWhistleDimension);
        ggml_tensor * positional = ggml_view_2d(ctx_.ggml, weights_.positional.tensor,
            kWhistleDimension, frames, weights_.positional.tensor->nb[1], 0);
        return ggml_add(ctx_.ggml, memory, positional);
    }

    ggml_tensor * cross_keys(ggml_tensor * memory, const WhistleCrossProjectionWeights & w) {
        return ops_.head_norm(ops_.linear(memory, w.k_proj, kWhistleDimension, kHeads * kQkDim), kHeads, w.k_norm);
    }

    ggml_tensor * cross_values(ggml_tensor * memory, const WhistleCrossProjectionWeights & w) {
        return ops_.linear(memory, w.v_proj, kWhistleDimension, kHeads * kVDim);
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

    // Grouped-query attention with 48-wide queries/keys and 64-wide values, so the
    // shared attention modules (which require one head size) do not apply.
    ggml_tensor * attention(ggml_tensor * input, const WhistleEncoderLayerWeights & w) {
        const int64_t frames = input->ne[1];
        const int64_t group = kHeads / kKvHeads;
        auto q = ops_.rope(ops_.head_norm(
            ops_.linear(input, w.q_proj, kWhistleDimension, kHeads * kQkDim), kHeads, w.q_norm), kHeads, positions_);
        auto k = ops_.rope(ops_.head_norm(
            ops_.linear(input, w.k_proj, kWhistleDimension, kKvHeads * kQkDim), kKvHeads, w.k_norm), kKvHeads, positions_);
        ggml_tensor * v = ops_.linear(input, w.v_proj, kWhistleDimension, kKvHeads * kVDim);

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
        ggml_tensor * gate = ops_.sigmoid(ops_.linear(input, w.gate_proj, kWhistleDimension, kWhistleDimension));
        return ops_.linear(ggml_mul(ctx_.ggml, context, gate), w.out_proj, kWhistleDimension, kWhistleDimension);
    }

    ggml_tensor * convolution(ggml_tensor * input, const WhistleEncoderLayerWeights & w) {
        const int64_t frames = input->ne[1];
        ggml_tensor * c = ops_.gemma_norm(input, w.conv_norm, kWhistleDimension);
        c = ops_.linear(c, w.pw1, kWhistleDimension, 2 * kWhistleDimension);
        c = modules::GLUModule().build(ctx_, rows_2d(c)).tensor;
        auto time_major = core::wrap_tensor(
            ggml_cont(ctx_.ggml, ggml_transpose(ctx_.ggml, c)),
            TensorShape::from_dims({1, kWhistleDimension, frames}), GGML_TYPE_F32);
        auto conv = modules::DepthwiseConv1dModule({kWhistleDimension, kConvTaps, 1, (kConvTaps - 1) / 2, 1, false})
            .build(ctx_, time_major, w.dw);
        c = ggml_cont(ctx_.ggml, ggml_transpose(ctx_.ggml,
            ggml_reshape_2d(ctx_.ggml, conv.tensor, frames, kWhistleDimension)));
        c = ops_.silu(ops_.gemma_norm(c, w.conv_out_norm, kWhistleDimension));
        return ops_.linear(c, w.pw2, kWhistleDimension, kWhistleDimension);
    }

    ggml_tensor * block(ggml_tensor * input, const WhistleEncoderLayerWeights & w) {
        ggml_tensor * h = input;
        h = ggml_add(ctx_.ggml, h, ggml_scale(ctx_.ggml,
            ops_.hadamard(ops_.gemma_norm(h, w.pre_hada_norm_0, kWhistleDimension), w.hadamard_0), 0.5f));
        ggml_tensor * attended = attention(ops_.gemma_norm(h, w.attn_norm, kWhistleDimension), w);
        attended = ops_.gemma_norm(attended, w.post_attn_norm, kWhistleDimension);
        h = ggml_add(ctx_.ggml, h, ggml_scale(ctx_.ggml, attended, w.attn_gate));
        h = ggml_add(ctx_.ggml, h, convolution(h, w));
        return ggml_add(ctx_.ggml, h, ggml_scale(ctx_.ggml,
            ops_.hadamard(ops_.gemma_norm(h, w.pre_hada_norm, kWhistleDimension), w.hadamard), 0.5f));
    }

    core::ModuleBuildContext & ctx_;
    WhistleGraphOps ops_;
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
        state = builder.layer(state, weights_.layers[static_cast<size_t>(layer)]);
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
