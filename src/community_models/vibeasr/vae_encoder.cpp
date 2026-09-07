#include "engine/community_models/vibeasr/vae_encoder.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// The in-band tensor scale of GGML_TYPE_I8_S is internal to ggml, so the
// waveform quantizer and the feature dequantizer reach for the same declarations
// the implementation uses rather than re-deriving the layout here. Buffer sizes
// still come from the public ggml_nbytes(), which already accounts for the
// trailing scale.
extern "C" {
void   ggml_i8_s_to_float  (const void  * x, float * y, int64_t n);
size_t ggml_i8_s_from_float(const float * x, void  * y, int64_t n);
}

namespace engine::community_models::vibeasr {
namespace {

// The graph is ~530 nodes for the published 7-stage encoder; leave headroom for
// deeper stage stacks without making the arena reservation depend on the config.
constexpr size_t kGraphNodes = 8192;

// Activation layout inside this file is described in ggml `ne` order, which is
// the reverse of core::TensorShape. The encoder alternates between two layouts:
//
//   channel-major  ne [C, L] -- what a matmul produces, what the norms and the
//                               FFN want, since they reduce over ne[0]
//   length-major   ne [L, C] -- what im2col wants, since it slides over ne[0]
//
// VibeASR's graph flips between them with permute + cont in exactly the places
// reproduced below.

core::TensorValue load_i8_s_tensor(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & name,
    const std::vector<int64_t> & expected_shape) {
    const auto metadata = source.require_metadata(name);
    if (metadata.dtype != "i8_s") {
        throw std::runtime_error("VibeASR VAE tensor " + name + " is " + metadata.dtype + ", expected i8_s");
    }
    if (metadata.shape != expected_shape) {
        throw std::runtime_error("VibeASR VAE tensor " + name + " has an unexpected shape");
    }

    core::TensorShape shape;
    shape.rank = expected_shape.size();
    for (size_t i = 0; i < shape.rank; ++i) {
        shape.dims[i] = expected_shape[i];
    }

    // I8_S is a whole-tensor quantization: the GGUF payload is the int8 values
    // followed by one padded F32 scale, which is exactly what ggml_nbytes()
    // expects, so the bytes go to the backend untouched.
    const auto raw = source.require_tensor_data(name);
    return store.make_tensor(shape, GGML_TYPE_I8_S, raw.bytes.data(), raw.bytes.size());
}

VaeBlockWeights load_block_weights(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    const VaeBlockConfig & config) {
    const int64_t channels = config.channels;
    const int64_t hidden = config.ffn_hidden;

    VaeBlockWeights weights;
    weights.mixer_norm = store.load_f32_tensor(source, prefix + ".norm.weight", {channels});
    weights.mixer_conv_weight = load_i8_s_tensor(
        store, source, prefix + ".mixer.conv.conv.conv.weight", {channels, 1, config.kernel_size});
    weights.mixer_conv_bias = store.load_f32_tensor(source, prefix + ".mixer.conv.conv.conv.bias", {channels});
    weights.mixer_gamma = store.load_f32_tensor(source, prefix + ".gamma", {channels});
    weights.ffn_norm = store.load_f32_tensor(source, prefix + ".ffn_norm.weight", {channels});
    weights.ffn_fc1_weight = load_i8_s_tensor(store, source, prefix + ".ffn.linear1.weight", {hidden, channels});
    weights.ffn_fc1_bias = store.load_f32_tensor(source, prefix + ".ffn.linear1.bias", {hidden});
    weights.ffn_fc2_weight = load_i8_s_tensor(store, source, prefix + ".ffn.linear2.weight", {channels, hidden});
    weights.ffn_fc2_bias = store.load_f32_tensor(source, prefix + ".ffn.linear2.bias", {channels});
    weights.ffn_gamma = store.load_f32_tensor(source, prefix + ".ffn_gamma", {channels});
    return weights;
}

VaeBranchWeights load_branch_weights(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const VaeBranchConfig & config) {
    VaeBranchWeights weights;
    weights.stages.reserve(config.stages.size());

    for (size_t stage = 0; stage < config.stages.size(); ++stage) {
        const auto & stage_config = config.stages[stage];
        const std::string stage_prefix = config.prefix + ".stages." + std::to_string(stage);
        const std::string downsample_prefix =
            config.prefix + ".downsample_layers." + std::to_string(stage) + ".0.conv.conv";

        VaeStageWeights stage_weights;
        stage_weights.downsample_weight = load_i8_s_tensor(
            store,
            source,
            downsample_prefix + ".weight",
            {stage_config.out_channels, stage_config.in_channels, stage_config.downsample_kernel_size});
        stage_weights.downsample_bias =
            store.load_f32_tensor(source, downsample_prefix + ".bias", {stage_config.out_channels});
        stage_weights.blocks.reserve(stage_config.blocks.size());
        for (size_t block = 0; block < stage_config.blocks.size(); ++block) {
            stage_weights.blocks.push_back(load_block_weights(
                store, source, stage_prefix + "." + std::to_string(block), stage_config.blocks[block]));
        }
        weights.stages.push_back(std::move(stage_weights));
    }

    const int64_t last_channels = config.stages.back().out_channels;
    weights.head_weight = load_i8_s_tensor(
        store, source, config.prefix + ".head.conv.conv.weight",
        {config.latent_dim, last_channels, config.head_kernel_size});
    weights.head_bias = store.load_f32_tensor(source, config.prefix + ".head.conv.conv.bias", {config.latent_dim});

    const std::string connector = config.prefix + "_connector";
    weights.connector_fc1_weight = load_i8_s_tensor(
        store, source, connector + ".fc1.weight", {config.connector_hidden, config.latent_dim});
    weights.connector_fc1_bias = store.load_f32_tensor(source, connector + ".fc1.bias", {config.connector_hidden});
    weights.connector_norm = store.load_f32_tensor(source, connector + ".norm.weight", {config.connector_hidden});
    weights.connector_fc2_weight = load_i8_s_tensor(
        store, source, connector + ".fc2.weight", {config.connector_hidden, config.connector_hidden});
    weights.connector_fc2_bias = store.load_f32_tensor(source, connector + ".fc2.bias", {config.connector_hidden});
    return weights;
}

// channel-major [C, L] -> length-major [L, C], and back.
ggml_tensor * transpose_layout(core::ModuleBuildContext & ctx, ggml_tensor * x) {
    return ggml_cont(ctx.ggml, ggml_permute(ctx.ggml, x, 1, 0, 2, 3));
}

// x [C, L], gamma [C] -> [C, L]. Reduces over the channel axis, matching the
// channels-last RMSNorm of the reference implementation.
ggml_tensor * rms_norm(core::ModuleBuildContext & ctx, ggml_tensor * x, ggml_tensor * gamma, float eps) {
    return ggml_rms_norm_scaled(ctx.ggml, x, gamma, eps);
}

// x [IC, L], w [IC, OC], bias [OC] -> [OC, L].
//
// Everything is flattened to 2D for the matmul, so the trailing ne of x carry no
// information beyond the total number of positions.
ggml_tensor * linear(
    core::ModuleBuildContext & ctx,
    ggml_tensor * x,
    ggml_tensor * w,
    ggml_tensor * bias,
    bool fuse_relu) {
    GGML_ASSERT(x->ne[3] == 1);
    const int64_t in_features = x->ne[0];
    const int64_t out_features = w->ne[1];
    const int64_t positions = x->ne[1] * x->ne[2];

    ggml_tensor * flat = ggml_reshape_2d(ctx.ggml, x, in_features, positions);
    ggml_tensor * out = fuse_relu ? ggml_mul_mat_add_relu(ctx.ggml, w, flat, bias)
                                  : ggml_mul_mat_add(ctx.ggml, w, flat, bias);
    return ggml_reshape_2d(ctx.ggml, out, out_features, positions);
}

// Causal Conv1d. x [L, IC, 1], w [K, IC, OC], bias [OC] -> [OC, OW].
//
// The left pad is K - stride and the right pad is zero, which is what makes the
// stack causal; the converter left-pads short kernels with zeros so the padded
// K stays exact.
ggml_tensor * conv1d_causal(
    core::ModuleBuildContext & ctx,
    ggml_tensor * x,
    ggml_tensor * w,
    ggml_tensor * bias,
    int stride) {
    const int64_t kernel_size = w->ne[0];
    const int64_t in_channels = w->ne[1];
    const int64_t out_channels = w->ne[2];
    const int left_pad = static_cast<int>(kernel_size) - stride;
    GGML_ASSERT(left_pad >= 0);

    // im2col gives [IC*K, OW, N].
    ggml_tensor * cols = ggml_im2col_asym(
        ctx.ggml, w, x, stride, 0, /*lp0=*/left_pad, /*rp0=*/0, /*p1=*/0, /*d0=*/1, /*d1=*/0,
        /*is_2D=*/false, GGML_TYPE_I8_S);

    ggml_tensor * w2d = ggml_reshape_2d(ctx.ggml, w, kernel_size * in_channels, out_channels);
    ggml_tensor * cols2d = ggml_reshape_2d(ctx.ggml, cols, cols->ne[0], cols->ne[1] * cols->ne[2]);
    return ggml_mul_mat_add(ctx.ggml, w2d, cols2d, bias);
}

// Causal depthwise Conv1d. x [L, C], w [K, 1, C], bias [C] -> [C, L].
//
// ggml_mul_mat_add takes its depthwise contraction path when the weight is
// [K, 1, C], producing [1, L, C]; the trailing reshape and permute fold that
// back to channel-major.
ggml_tensor * conv1d_dw_causal(
    core::ModuleBuildContext & ctx,
    ggml_tensor * x,
    ggml_tensor * w,
    ggml_tensor * bias) {
    const int64_t kernel_size = w->ne[0];

    ggml_tensor * x4d = ggml_reshape_4d(ctx.ggml, x, x->ne[0], 1, x->ne[1], 1);
    ggml_tensor * cols = ggml_im2col_asym(
        ctx.ggml, w, x4d, /*s0=*/1, 0, /*lp0=*/static_cast<int>(kernel_size) - 1, /*rp0=*/0, /*p1=*/0,
        /*d0=*/1, /*d1=*/0, /*is_2D=*/false, GGML_TYPE_I8_S);

    ggml_tensor * out = ggml_mul_mat_add(ctx.ggml, w, cols, bias);
    out = ggml_reshape_3d(ctx.ggml, out, out->ne[1], out->ne[2], 1);
    return transpose_layout(ctx, out);
}

// One ConvNeXt block. x [C, L] -> [C, L].
ggml_tensor * build_block(
    core::ModuleBuildContext & ctx,
    ggml_tensor * x,
    const VaeBlockWeights & weights,
    float eps) {
    ggml_tensor * residual = x;
    ggml_tensor * h = rms_norm(ctx, x, weights.mixer_norm.tensor, eps);
    h = transpose_layout(ctx, h);
    h = conv1d_dw_causal(ctx, h, weights.mixer_conv_weight.tensor, weights.mixer_conv_bias.tensor);
    // LayerScale folded into the residual add: h * gamma + residual.
    x = ggml_add_scaled(ctx.ggml, h, residual, weights.mixer_gamma.tensor);

    residual = x;
    h = rms_norm(ctx, x, weights.ffn_norm.tensor, eps);
    // The I8_S FFN uses ReLU, fused into the first matmul. VibeASR's F32
    // fallback uses GELU instead; only the quantized path has published weights,
    // so only ReLU is ported.
    h = linear(ctx, h, weights.ffn_fc1_weight.tensor, weights.ffn_fc1_bias.tensor, /*fuse_relu=*/true);
    h = linear(ctx, h, weights.ffn_fc2_weight.tensor, weights.ffn_fc2_bias.tensor, /*fuse_relu=*/false);
    return ggml_add_scaled(ctx.ggml, h, residual, weights.ffn_gamma.tensor);
}

// waveform [n_samples, 1, 1] -> features [connector_hidden, frames].
ggml_tensor * build_branch(
    core::ModuleBuildContext & ctx,
    ggml_tensor * waveform,
    const VaeBranchConfig & config,
    const VaeBranchWeights & weights,
    float eps) {
    ggml_tensor * x = waveform;

    for (size_t stage = 0; stage < config.stages.size(); ++stage) {
        const auto & stage_weights = weights.stages[stage];
        x = conv1d_causal(
            ctx,
            x,
            stage_weights.downsample_weight.tensor,
            stage_weights.downsample_bias.tensor,
            static_cast<int>(config.stages[stage].downsample_stride));
        for (const auto & block : stage_weights.blocks) {
            x = build_block(ctx, x, block, eps);
        }
        // Back to length-major for the next stage's im2col (and for the head).
        x = transpose_layout(ctx, x);
    }

    x = conv1d_causal(ctx, x, weights.head_weight.tensor, weights.head_bias.tensor, /*stride=*/1);

    x = linear(ctx, x, weights.connector_fc1_weight.tensor, weights.connector_fc1_bias.tensor, false);
    x = rms_norm(ctx, x, weights.connector_norm.tensor, eps);
    return linear(ctx, x, weights.connector_fc2_weight.tensor, weights.connector_fc2_bias.tensor, false);
}

}  // namespace

VibeASRVaeEncoderRuntime::VibeASRVaeEncoderRuntime(
    std::shared_ptr<const VibeASRVaeAssets> assets,
    engine::core::ExecutionContext & execution_context,
    size_t graph_arena_bytes)
    : assets_(std::move(assets)),
      execution_context_(&execution_context),
      weight_store_(
          execution_context.backend(),
          execution_context.backend_type(),
          "VibeASR VAE encoder weights",
          256ull * 1024ull * 1024ull),
      graph_arena_bytes_(graph_arena_bytes) {
    if (assets_ == nullptr) {
        throw std::runtime_error("VibeASR VAE encoder runtime requires assets");
    }
    weights_.acoustic = load_branch_weights(weight_store_, *assets_->source, assets_->config.acoustic);
    weights_.semantic = load_branch_weights(weight_store_, *assets_->source, assets_->config.semantic);
    weight_store_.upload();
}

VaeEncoderFeatures VibeASRVaeEncoderRuntime::encode_acoustic(const std::vector<float> & samples) {
    return encode(assets_->config.acoustic, weights_.acoustic, samples);
}

VaeEncoderFeatures VibeASRVaeEncoderRuntime::encode_semantic(const std::vector<float> & samples) {
    return encode(assets_->config.semantic, weights_.semantic, samples);
}

VaeEncoderFeatures VibeASRVaeEncoderRuntime::encode(
    const VaeBranchConfig & config,
    const VaeBranchWeights & weights,
    const std::vector<float> & samples) {
    const int64_t num_samples = static_cast<int64_t>(samples.size());
    const int64_t expected_frames = config.frames_for_samples(num_samples);
    if (expected_frames <= 0) {
        // Shorter than one encoder frame: the first stage's im2col would have no
        // output column at all.
        return {};
    }

    ggml_init_params params{};
    params.mem_size = graph_arena_bytes_;
    params.mem_buffer = nullptr;
    params.no_alloc = true;

    ggml_context * ggml_ctx = ggml_init(params);
    if (ggml_ctx == nullptr) {
        throw std::runtime_error("Failed to initialize GGML context for the VibeASR VAE encoder");
    }

    ggml_gallocr * galloc = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(execution_context_->backend()));
    if (galloc == nullptr) {
        ggml_free(ggml_ctx);
        throw std::runtime_error("Failed to initialize GGML allocator for the VibeASR VAE encoder");
    }

    VaeEncoderFeatures features;

    try {
        core::ModuleBuildContext ctx{ggml_ctx, "vibeasr_vae_encoder", execution_context_->backend_type()};

        // The waveform enters the graph already quantized: the encoder never
        // touches F32 activations, so there is no leading quantize node.
        ggml_tensor * waveform = ggml_new_tensor_3d(ggml_ctx, GGML_TYPE_I8_S, num_samples, 1, 1);
        ggml_set_input(waveform);

        ggml_tensor * out = build_branch(ctx, waveform, config, weights, assets_->config.rms_norm_eps);
        ggml_set_output(out);

        ggml_cgraph * gf = ggml_new_graph_custom(ggml_ctx, kGraphNodes, false);
        ggml_build_forward_expand(gf, out);

        if (!ggml_gallocr_alloc_graph(galloc, gf)) {
            throw std::runtime_error("Failed to allocate the GGML graph for the VibeASR VAE encoder");
        }

        std::vector<std::byte> quantized(ggml_nbytes(waveform));
        ggml_i8_s_from_float(samples.data(), quantized.data(), num_samples);
        ggml_backend_tensor_set(waveform, quantized.data(), 0, quantized.size());

        if (ggml_backend_graph_compute(execution_context_->backend(), gf) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to compute the GGML graph for the VibeASR VAE encoder");
        }

        features.dim = out->ne[0];
        features.frames = out->ne[1];
        if (features.frames != expected_frames) {
            throw std::runtime_error("VibeASR VAE encoder produced an unexpected frame count");
        }

        // The result is still I8_S, one scale for the whole feature block.
        std::vector<std::byte> raw(ggml_nbytes(out));
        ggml_backend_tensor_get(out, raw.data(), 0, raw.size());
        features.values.resize(static_cast<size_t>(features.dim * features.frames));
        ggml_i8_s_to_float(raw.data(), features.values.data(), features.dim * features.frames);
    } catch (...) {
        ggml_gallocr_free(galloc);
        ggml_free(ggml_ctx);
        throw;
    }

    ggml_gallocr_free(galloc);
    ggml_free(ggml_ctx);
    return features;
}

}  // namespace engine::community_models::vibeasr
