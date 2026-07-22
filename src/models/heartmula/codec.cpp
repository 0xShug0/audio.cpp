#include "engine/models/heartmula/codec.h"

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"
#include "engine/framework/sampling/torch_random.h"
#include "engine/models/heartmula/tokenizer_text.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::heartmula {
namespace {

namespace binding = engine::modules::binding;

constexpr float kProjectLayerScale = 0.5773502691896257F;
constexpr float kFlowTimestepScale = 1000.0F;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct FlowEstimatorBuildResult {
    core::TensorValue output;
    ggml_tensor * first_stage_state = nullptr;
};

void require_assets(const HeartMuLaAssets & assets) {
    if (assets.codec_weights == nullptr) {
        throw std::runtime_error("HeartCodec requires codec weights");
    }
}

int64_t pow2_i64(size_t exponent) {
    int64_t value = 1;
    for (size_t i = 0; i < exponent; ++i) {
        value *= 2;
    }
    return value;
}

int64_t llama_mlp_hidden_dim(int64_t dim) {
    const int64_t multiple_of = 256;
    int64_t hidden_dim = 4 * dim;
    hidden_dim = 2 * hidden_dim / 3;
    return multiple_of * ((hidden_dim + multiple_of - 1) / multiple_of);
}

std::vector<float> effective_weight_norm_dim0(
    const assets::TensorSource & source,
    const std::string & prefix,
    const std::vector<int64_t> & shape) {
    if (shape.empty()) {
        throw std::runtime_error("HeartCodec weight-norm tensor shape is empty");
    }
    const auto g = source.require_f32(prefix + ".parametrizations.weight.original0", {shape[0], 1, 1});
    const auto v = source.require_f32(prefix + ".parametrizations.weight.original1", shape);
    std::vector<float> out(v.size(), 0.0F);
    int64_t inner = 1;
    for (size_t axis = 1; axis < shape.size(); ++axis) {
        inner *= shape[axis];
    }
    for (int64_t row = 0; row < shape[0]; ++row) {
        double sum = 0.0;
        const size_t base = static_cast<size_t>(row * inner);
        for (int64_t index = 0; index < inner; ++index) {
            const float value = v[base + static_cast<size_t>(index)];
            sum += static_cast<double>(value) * static_cast<double>(value);
        }
        const double norm = std::sqrt(sum);
        if (norm == 0.0) {
            throw std::runtime_error("HeartCodec weight-norm tensor has zero norm: " + prefix);
        }
        const float scale = static_cast<float>(static_cast<double>(g[static_cast<size_t>(row)]) / norm);
        for (int64_t index = 0; index < inner; ++index) {
            out[base + static_cast<size_t>(index)] = v[base + static_cast<size_t>(index)] * scale;
        }
    }
    return out;
}

modules::Conv1dWeights load_weight_norm_conv1d(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    int64_t out_channels,
    int64_t in_channels,
    int64_t kernel_size,
    bool use_bias = true) {
    modules::Conv1dWeights weights;
    const auto shape = core::TensorShape::from_dims({out_channels, in_channels, kernel_size});
    weights.weight = store.make_from_f32(
        shape,
        storage_type,
        effective_weight_norm_dim0(source, prefix, {out_channels, in_channels, kernel_size}));
    if (use_bias) {
        weights.bias = store.load_f32_tensor(source, prefix + ".bias", {out_channels});
    }
    return weights;
}

modules::ConvTranspose1dWeights load_weight_norm_conv_transpose1d(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel_size,
    bool use_bias = true) {
    modules::ConvTranspose1dWeights weights;
    const auto shape = core::TensorShape::from_dims({in_channels, out_channels, kernel_size});
    weights.weight = store.make_from_f32(
        shape,
        storage_type,
        effective_weight_norm_dim0(source, prefix, {in_channels, out_channels, kernel_size}));
    if (use_bias) {
        weights.bias = store.load_f32_tensor(source, prefix + ".bias", {out_channels});
    }
    return weights;
}

HeartCodecPreluWeights load_prelu(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix) {
    return {store.load_f32_tensor(source, prefix + ".weight", {1})};
}

HeartCodecProjectLayerWeights load_project_layer(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    int64_t in_channels,
    int64_t out_channels) {
    HeartCodecProjectLayerWeights weights;
    weights.ffn_1 = binding::conv1d_from_source(
        store,
        source,
        prefix + ".ffn_1",
        storage_type,
        out_channels,
        in_channels,
        3,
        true);
    weights.ffn_2 = binding::linear_from_source(
        store,
        source,
        prefix + ".ffn_2",
        storage_type,
        out_channels,
        out_channels,
        true);
    return weights;
}

HeartCodecAdaLayerNormWeights load_adaln(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    int64_t hidden_size) {
    HeartCodecAdaLayerNormWeights weights;
    weights.timestep_linear_1 = binding::linear_from_source(
        store,
        source,
        prefix + ".emb.timestep_embedder.linear_1",
        storage_type,
        hidden_size,
        512,
        true);
    weights.timestep_linear_2 = binding::linear_from_source(
        store,
        source,
        prefix + ".emb.timestep_embedder.linear_2",
        storage_type,
        hidden_size,
        hidden_size,
        true);
    weights.linear = binding::linear_from_source(
        store,
        source,
        prefix + ".linear",
        storage_type,
        6 * hidden_size,
        hidden_size,
        true);
    return weights;
}

HeartCodecTransformerBlockWeights load_transformer_block(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    int64_t hidden_size,
    int64_t mlp_hidden_size) {
    HeartCodecTransformerBlockWeights weights;
    weights.attn_norm = binding::norm_weight_from_source(store, source, prefix + ".attn_norm", hidden_size);
    weights.q_proj = binding::linear_from_source(store, source, prefix + ".attn.q_proj", storage_type, hidden_size, hidden_size, false);
    weights.k_proj = binding::linear_from_source(store, source, prefix + ".attn.k_proj", storage_type, hidden_size, hidden_size, false);
    weights.v_proj = binding::linear_from_source(store, source, prefix + ".attn.v_proj", storage_type, hidden_size, hidden_size, false);
    weights.o_proj = binding::linear_from_source(store, source, prefix + ".attn.o_proj", storage_type, hidden_size, hidden_size, false);
    weights.mlp_norm = binding::norm_weight_from_source(store, source, prefix + ".mlp_norm", hidden_size);
    weights.mlp_gate = binding::linear_from_source(store, source, prefix + ".mlp.gate", storage_type, mlp_hidden_size, hidden_size, false);
    weights.mlp_up = binding::linear_from_source(store, source, prefix + ".mlp.up", storage_type, mlp_hidden_size, hidden_size, false);
    weights.mlp_down = binding::linear_from_source(store, source, prefix + ".mlp.down", storage_type, hidden_size, mlp_hidden_size, false);
    weights.scale_shift_table = store.load_f32_tensor(source, prefix + ".scale_shift_table", {6, hidden_size});
    return weights;
}

std::vector<HeartCodecTransformerBlockWeights> load_transformer_blocks(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    int64_t layers,
    int64_t hidden_size) {
    std::vector<HeartCodecTransformerBlockWeights> blocks;
    blocks.reserve(static_cast<size_t>(layers));
    const int64_t mlp_hidden_size = llama_mlp_hidden_dim(hidden_size);
    for (int64_t layer = 0; layer < layers; ++layer) {
        blocks.push_back(load_transformer_block(
            store,
            source,
            prefix + "." + std::to_string(layer),
            storage_type,
            hidden_size,
            mlp_hidden_size));
    }
    return blocks;
}

HeartCodecFlowWeights load_flow_weights(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const HeartCodecConfig & config,
    assets::TensorStorageType storage_type) {
    const int64_t estimator_dim = config.num_attention_heads * config.attention_head_dim;
    const int64_t estimator_dim_2 = 2 * estimator_dim;
    HeartCodecFlowWeights weights;
    weights.vq_project_out = binding::linear_from_source(
        store,
        source,
        "flow_matching.vq_embed.project_out",
        storage_type,
        config.dim,
        config.codebook_dim,
        true);
    weights.vq_codebooks.reserve(static_cast<size_t>(config.num_quantizers));
    for (int64_t quantizer = 0; quantizer < config.num_quantizers; ++quantizer) {
        const std::string name = "flow_matching.vq_embed.layers." + std::to_string(quantizer) + "._codebook.embed";
        weights.vq_codebooks.push_back(store.load_tensor_as_shape(
            source,
            name,
            storage_type,
            {1, config.codebook_size, config.codebook_dim},
            core::TensorShape::from_dims({config.codebook_size, config.codebook_dim})));
    }
    weights.cond_feature_emb = binding::linear_from_source(
        store,
        source,
        "flow_matching.cond_feature_emb",
        storage_type,
        config.dim,
        config.dim,
        true);
    weights.zero_cond_embedding = store.load_f32_tensor(
        source,
        "flow_matching.zero_cond_embedding1",
        {config.dim});
    weights.zero_cond_embedding_host = source.require_f32_tensor("flow_matching.zero_cond_embedding1", {config.dim});

    auto & estimator = weights.estimator;
    estimator.proj_in = load_project_layer(
        store,
        source,
        "flow_matching.estimator.proj_in",
        storage_type,
        config.in_channels,
        estimator_dim);
    estimator.transformer_blocks = load_transformer_blocks(
        store,
        source,
        "flow_matching.estimator.transformer_blocks",
        storage_type,
        config.num_layers,
        estimator_dim);
    estimator.connection_proj = load_project_layer(
        store,
        source,
        "flow_matching.estimator.connection_proj",
        storage_type,
        config.in_channels + estimator_dim,
        estimator_dim_2);
    estimator.transformer_blocks_2 = load_transformer_blocks(
        store,
        source,
        "flow_matching.estimator.transformer_blocks_2",
        storage_type,
        config.num_layers_2,
        estimator_dim_2);
    estimator.scale_shift_table = store.load_f32_tensor(
        source,
        "flow_matching.estimator.scale_shift_table",
        {2, estimator_dim});
    estimator.scale_shift_table_2 = store.load_f32_tensor(
        source,
        "flow_matching.estimator.scale_shift_table_2",
        {2, estimator_dim_2});
    estimator.proj_out = load_project_layer(
        store,
        source,
        "flow_matching.estimator.proj_out",
        storage_type,
        estimator_dim_2,
        config.out_channels);
    estimator.adaln_single = load_adaln(
        store,
        source,
        "flow_matching.estimator.adaln_single",
        storage_type,
        estimator_dim);
    estimator.adaln_single_2 = load_adaln(
        store,
        source,
        "flow_matching.estimator.adaln_single_2",
        storage_type,
        estimator_dim_2);
    return weights;
}

HeartCodecResidualUnitWeights load_residual_unit(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    int64_t channels,
    int64_t res_kernel_size) {
    HeartCodecResidualUnitWeights weights;
    weights.conv1 = load_weight_norm_conv1d(
        store,
        source,
        prefix + ".conv1",
        storage_type,
        channels,
        channels,
        res_kernel_size);
    weights.conv2 = load_weight_norm_conv1d(
        store,
        source,
        prefix + ".conv2",
        storage_type,
        channels,
        channels,
        1);
    weights.activation1 = load_prelu(store, source, prefix + ".activation1");
    weights.activation2 = load_prelu(store, source, prefix + ".activation2");
    return weights;
}

HeartCodecScalarDecoderWeights load_scalar_decoder_weights(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const HeartCodecConfig & config,
    assets::TensorStorageType storage_type) {
    HeartCodecScalarDecoderWeights weights;
    const int64_t first_channels = config.init_channel * pow2_i64(config.upsample_factors.size());
    weights.input_conv = load_weight_norm_conv1d(
        store,
        source,
        "scalar_model.decoder.0",
        storage_type,
        first_channels,
        config.latent_hidden_dim,
        config.delay_kernel_size);
    weights.blocks.reserve(config.upsample_factors.size());
    int64_t in_channels = first_channels;
    for (size_t block_index = 0; block_index < config.upsample_factors.size(); ++block_index) {
        const int64_t out_channels = in_channels / 2;
        const std::string prefix = "scalar_model.decoder." + std::to_string(block_index + 1);
        HeartCodecDecoderBlockWeights block;
        block.up_conv = load_weight_norm_conv_transpose1d(
            store,
            source,
            prefix + ".up_conv.layer",
            storage_type,
            in_channels,
            out_channels,
            config.upsample_kernel_sizes[block_index]);
        block.residual_units.reserve(5);
        for (int64_t residual = 0; residual < 5; ++residual) {
            block.residual_units.push_back(load_residual_unit(
                store,
                source,
                prefix + ".convs." + std::to_string(residual),
                storage_type,
                out_channels,
                config.res_kernel_size));
        }
        weights.blocks.push_back(std::move(block));
        in_channels = out_channels;
    }
    const int64_t post_index = 1 + static_cast<int64_t>(config.upsample_factors.size());
    weights.post_processor.conv = binding::conv1d_from_source(
        store,
        source,
        "scalar_model.decoder." + std::to_string(post_index) + ".conv",
        storage_type,
        config.init_channel,
        config.init_channel,
        config.default_kernel_size,
        true);
    weights.post_processor.activation =
        load_prelu(store, source, "scalar_model.decoder." + std::to_string(post_index) + ".activation");
    weights.output_conv = load_weight_norm_conv1d(
        store,
        source,
        "scalar_model.decoder." + std::to_string(post_index + 1),
        storage_type,
        config.num_bands,
        config.init_channel,
        config.default_kernel_size);
    return weights;
}

core::TensorValue scale(core::ModuleBuildContext & ctx, const core::TensorValue & input, float value) {
    return core::wrap_tensor(ggml_scale(ctx.ggml, core::ensure_backend_addressable_layout(ctx, input).tensor, value), input.shape, GGML_TYPE_F32);
}

core::TensorValue add_one(core::ModuleBuildContext & ctx, const core::TensorValue & input) {
    return core::wrap_tensor(ggml_scale_bias(ctx.ggml, core::ensure_backend_addressable_layout(ctx, input).tensor, 1.0F, 1.0F), input.shape, GGML_TYPE_F32);
}

core::TensorValue quantize_scalar_latent(core::ModuleBuildContext & ctx, const core::TensorValue & input) {
    auto scaled = scale(ctx, input, 9.0F);
    auto rounded = core::wrap_tensor(ggml_round(ctx.ggml, core::ensure_backend_addressable_layout(ctx, scaled).tensor), scaled.shape, GGML_TYPE_F32);
    return scale(ctx, rounded, 1.0F / 9.0F);
}

core::TensorValue causal_left_pad_bct(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t left_pad) {
    if (left_pad == 0) {
        return input;
    }
    auto * padded = ggml_pad_ext(
        ctx.ggml,
        core::ensure_backend_addressable_layout(ctx, input).tensor,
        static_cast<int>(left_pad),
        0,
        0,
        0,
        0,
        0,
        0,
        0);
    return core::wrap_tensor(
        padded,
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], input.shape.dims[2] + left_pad}),
        GGML_TYPE_F32);
}

core::TensorValue causal_conv1d(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const modules::Conv1dWeights & weights,
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel_size,
    int64_t dilation = 1) {
    const int64_t left_pad = dilation * (kernel_size - 1);
    auto padded = causal_left_pad_bct(ctx, input, left_pad);
    return modules::Conv1dModule({
        in_channels,
        out_channels,
        kernel_size,
        1,
        0,
        static_cast<int>(dilation),
        weights.bias.has_value(),
    }).build(ctx, padded, weights);
}

core::TensorValue causal_conv_transpose1d(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const modules::ConvTranspose1dWeights & weights,
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel_size,
    int64_t stride) {
    auto output = modules::ConvTranspose1dModule({
        in_channels,
        out_channels,
        kernel_size,
        static_cast<int>(stride),
        0,
        1,
        weights.bias.has_value(),
    }).build(ctx, input, weights);
    return modules::SliceModule({2, 0, output.shape.dims[2] - stride}).build(ctx, output);
}

core::TensorValue prelu(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const HeartCodecPreluWeights & weights) {
    auto positive = core::wrap_tensor(ggml_relu(ctx.ggml, core::ensure_backend_addressable_layout(ctx, input).tensor), input.shape, GGML_TYPE_F32);
    auto negative = core::wrap_tensor(ggml_sub(ctx.ggml, input.tensor, positive.tensor), input.shape, GGML_TYPE_F32);
    auto alpha = core::reshape_tensor(ctx, weights.weight, core::TensorShape::from_dims({1, 1, 1}));
    auto alpha_rep = core::wrap_tensor(ggml_repeat(ctx.ggml, alpha.tensor, input.tensor), input.shape, GGML_TYPE_F32);
    return modules::AddModule{}.build(ctx, positive, modules::MulModule{}.build(ctx, negative, alpha_rep));
}

core::TensorValue adjacent_repeat_frames_bct(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t repeat) {
    auto btc = modules::TransposeModule({{0, 2, 1}, 3}).build(ctx, input);
    btc = core::ensure_backend_addressable_layout(ctx, btc);
    auto bt1c = core::reshape_tensor(
        ctx,
        btc,
        core::TensorShape::from_dims({btc.shape.dims[0], btc.shape.dims[1], 1, btc.shape.dims[2]}));
    auto repeated = modules::RepeatModule(
        {core::TensorShape::from_dims({btc.shape.dims[0], btc.shape.dims[1], repeat, btc.shape.dims[2]})})
                        .build(ctx, bt1c);
    auto btc_repeated = core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, repeated),
        core::TensorShape::from_dims({btc.shape.dims[0], btc.shape.dims[1] * repeat, btc.shape.dims[2]}));
    return modules::TransposeModule({{0, 2, 1}, 3}).build(ctx, btc_repeated);
}

core::TensorValue adjacent_repeat_frames_btc(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t repeat) {
    auto bct = modules::TransposeModule({{0, 2, 1}, 3}).build(ctx, input);
    bct = adjacent_repeat_frames_bct(ctx, bct, repeat);
    return modules::TransposeModule({{0, 2, 1}, 3}).build(ctx, bct);
}

core::TensorValue reshape_codebook_indices(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & codes_bqt,
    int64_t quantizer) {
    auto selected = modules::SliceModule({1, quantizer, 1}).build(ctx, codes_bqt);
    selected = core::ensure_backend_addressable_layout(ctx, modules::TransposeModule({{0, 2, 1}, 3}).build(ctx, selected));
    return core::reshape_tensor(
        ctx,
        selected,
        core::TensorShape::from_dims({codes_bqt.shape.dims[0], codes_bqt.shape.dims[2]}));
}

core::TensorValue conditioning_from_codes(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & codes_bqt,
    const HeartCodecFlowWeights & weights,
    const HeartCodecConfig & config) {
    core::TensorValue summed;
    for (int64_t quantizer = 0; quantizer < config.num_quantizers; ++quantizer) {
        auto indices = reshape_codebook_indices(ctx, codes_bqt, quantizer);
        auto looked_up = modules::CodebookLookupModule({config.codebook_size, config.codebook_dim})
                             .build(ctx, indices, weights.vq_codebooks[static_cast<size_t>(quantizer)]);
        summed = summed.valid() ? modules::AddModule{}.build(ctx, summed, looked_up) : looked_up;
    }
    auto projected = modules::LinearModule({config.codebook_dim, config.dim, true, GGML_PREC_F32})
                         .build(ctx, summed, weights.vq_project_out);
    projected = modules::LinearModule({config.dim, config.dim, true, GGML_PREC_F32})
                    .build(ctx, projected, weights.cond_feature_emb);
    return adjacent_repeat_frames_btc(ctx, projected, 2);
}

core::TensorValue scalar_residual_unit(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const HeartCodecResidualUnitWeights & weights,
    int64_t channels,
    int64_t kernel_size,
    int64_t dilation) {
    auto x = causal_conv1d(ctx, input, weights.conv1, channels, channels, kernel_size, dilation);
    x = prelu(ctx, x, weights.activation1);
    x = causal_conv1d(ctx, x, weights.conv2, channels, channels, 1);
    x = prelu(ctx, x, weights.activation2);
    return modules::AddModule{}.build(ctx, input, x);
}

core::TensorValue scalar_decoder(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & latents,
    const HeartCodecScalarDecoderWeights & weights,
    const HeartCodecConfig & config) {
    auto x = quantize_scalar_latent(ctx, latents);
    int64_t channels = config.init_channel * pow2_i64(config.upsample_factors.size());
    x = modules::Conv1dModule({
        config.latent_hidden_dim,
        channels,
        config.delay_kernel_size,
        1,
        static_cast<int>(config.delay_kernel_size / 2),
        1,
        weights.input_conv.bias.has_value(),
    }).build(ctx, x, weights.input_conv);
    constexpr int64_t dilations[] = {1, 3, 5, 7, 9};
    for (size_t block_index = 0; block_index < weights.blocks.size(); ++block_index) {
        const auto & block = weights.blocks[block_index];
        const int64_t out_channels = channels / 2;
        x = causal_conv_transpose1d(
            ctx,
            x,
            block.up_conv,
            channels,
            out_channels,
            config.upsample_kernel_sizes[block_index],
            config.upsample_factors[block_index]);
        for (size_t residual = 0; residual < block.residual_units.size(); ++residual) {
            x = scalar_residual_unit(
                ctx,
                x,
                block.residual_units[residual],
                out_channels,
                config.res_kernel_size,
                dilations[residual]);
        }
        channels = out_channels;
    }
    if (config.num_samples > 1) {
        x = adjacent_repeat_frames_bct(ctx, x, config.num_samples);
        x = causal_conv1d(
            ctx,
            x,
            weights.post_processor.conv,
            config.init_channel,
            config.init_channel,
            config.default_kernel_size);
        x = prelu(ctx, x, weights.post_processor.activation);
    }
    return causal_conv1d(
        ctx,
        x,
        weights.output_conv,
        config.init_channel,
        config.num_bands,
        config.default_kernel_size);
}

core::TensorValue expand_batch_token(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input_bd,
    int64_t tokens,
    int64_t dim) {
    const int64_t batch = input_bd.shape.dims[0];
    const auto view = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, input_bd), core::TensorShape::from_dims({batch, 1, dim}));
    return modules::RepeatModule({core::TensorShape::from_dims({batch, tokens, dim})}).build(ctx, view);
}

core::TensorValue project_layer(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input_btc,
    const HeartCodecProjectLayerWeights & weights,
    int64_t in_channels,
    int64_t out_channels) {
    auto x = modules::TransposeModule({{0, 2, 1}, 3}).build(ctx, input_btc);
    x = modules::Conv1dModule({in_channels, out_channels, 3, 1, 1, 1, true}).build(ctx, x, weights.ffn_1);
    x = scale(ctx, x, kProjectLayerScale);
    x = modules::TransposeModule({{0, 2, 1}, 3}).build(ctx, x);
    return modules::LinearModule({out_channels, out_channels, true, GGML_PREC_F32}).build(ctx, x, weights.ffn_2);
}

core::TensorValue timestep_embedding(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & timesteps,
    const core::TensorValue & freqs) {
    const int64_t batch = timesteps.shape.dims[0];
    auto t = core::reshape_tensor(ctx, timesteps, core::TensorShape::from_dims({batch, 1}));
    t = modules::RepeatModule({core::TensorShape::from_dims({batch, 256})}).build(ctx, t);
    auto f = modules::RepeatModule({core::TensorShape::from_dims({batch, 256})}).build(ctx, freqs);
    auto args = scale(ctx, modules::MulModule{}.build(ctx, t, f), kFlowTimestepScale);
    auto cos_part = core::wrap_tensor(ggml_cos(ctx.ggml, core::ensure_backend_addressable_layout(ctx, args).tensor), args.shape, GGML_TYPE_F32);
    auto sin_part = core::wrap_tensor(ggml_sin(ctx.ggml, args.tensor), args.shape, GGML_TYPE_F32);
    return modules::ConcatModule({1}).build(ctx, cos_part, sin_part);
}

core::TensorValue adaln_embedding(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & timesteps,
    const core::TensorValue & freqs,
    const HeartCodecAdaLayerNormWeights & weights,
    int64_t hidden_size) {
    auto embedded = timestep_embedding(ctx, timesteps, freqs);
    embedded = modules::LinearModule({512, hidden_size, true, GGML_PREC_F32}).build(ctx, embedded, weights.timestep_linear_1);
    embedded = modules::SiluModule{}.build(ctx, embedded);
    return modules::LinearModule({hidden_size, hidden_size, true, GGML_PREC_F32})
        .build(ctx, embedded, weights.timestep_linear_2);
}

struct AdaNormParts {
    core::TensorValue shift_msa;
    core::TensorValue scale_msa;
    core::TensorValue gate_msa;
    core::TensorValue shift_mlp;
    core::TensorValue scale_mlp;
    core::TensorValue gate_mlp;
};

AdaNormParts adaptive_block_parts(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & timestep_mod,
    const core::TensorValue & scale_shift_table,
    int64_t tokens,
    int64_t hidden_size) {
    const int64_t batch = timestep_mod.shape.dims[0];
    auto scale_shift = core::reshape_tensor(ctx, scale_shift_table, core::TensorShape::from_dims({1, 6, hidden_size}));
    scale_shift = modules::RepeatModule({core::TensorShape::from_dims({batch, 6, hidden_size})}).build(ctx, scale_shift);
    auto parts = modules::AddModule{}.build(
        ctx,
        core::reshape_tensor(
            ctx,
            core::ensure_backend_addressable_layout(ctx, timestep_mod),
            core::TensorShape::from_dims({batch, 6, hidden_size})),
        scale_shift);
    AdaNormParts out{
        expand_batch_token(ctx, modules::SliceModule({1, 0, 1}).build(ctx, parts), tokens, hidden_size),
        expand_batch_token(ctx, modules::SliceModule({1, 1, 1}).build(ctx, parts), tokens, hidden_size),
        expand_batch_token(ctx, modules::SliceModule({1, 2, 1}).build(ctx, parts), tokens, hidden_size),
        expand_batch_token(ctx, modules::SliceModule({1, 3, 1}).build(ctx, parts), tokens, hidden_size),
        expand_batch_token(ctx, modules::SliceModule({1, 4, 1}).build(ctx, parts), tokens, hidden_size),
        expand_batch_token(ctx, modules::SliceModule({1, 5, 1}).build(ctx, parts), tokens, hidden_size),
    };
    return out;
}

core::TensorValue flow_attention(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const HeartCodecTransformerBlockWeights & weights,
    int64_t heads,
    int64_t head_dim) {
    const int64_t batch = input.shape.dims[0];
    const int64_t frames = input.shape.dims[1];
    const int64_t hidden_size = heads * head_dim;
    auto q = modules::LinearModule({hidden_size, hidden_size, false, GGML_PREC_F32}).build(ctx, input, weights.q_proj);
    auto k = modules::LinearModule({hidden_size, hidden_size, false, GGML_PREC_F32}).build(ctx, input, weights.k_proj);
    auto v = modules::LinearModule({hidden_size, hidden_size, false, GGML_PREC_F32}).build(ctx, input, weights.v_proj);
    q = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, q), core::TensorShape::from_dims({batch, frames, heads, head_dim}));
    k = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, k), core::TensorShape::from_dims({batch, frames, heads, head_dim}));
    v = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, v), core::TensorShape::from_dims({batch, frames, heads, head_dim}));
    q = modules::RoPEModule({head_dim, GGML_ROPE_TYPE_NORMAL, 10000.0F}).build(ctx, q, positions);
    k = modules::RoPEModule({head_dim, GGML_ROPE_TYPE_NORMAL, 10000.0F}).build(ctx, k, positions);
    q = core::ensure_backend_addressable_layout(ctx, modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, q));
    k = core::ensure_backend_addressable_layout(ctx, modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, k));
    v = core::ensure_backend_addressable_layout(ctx, modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, v));
    auto * flash = ggml_flash_attn_ext(
        ctx.ggml,
        q.tensor,
        k.tensor,
        v.tensor,
        nullptr,
        1.0F / std::sqrt(static_cast<float>(head_dim)),
        0.0F,
        0.0F);
    ggml_flash_attn_ext_set_prec(flash, GGML_PREC_F32);
    auto context = core::wrap_tensor(
        flash,
        core::TensorShape::from_dims({batch, frames, heads, head_dim}),
        GGML_TYPE_F32);
    context = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, context), core::TensorShape::from_dims({batch, frames, hidden_size}));
    return modules::LinearModule({hidden_size, hidden_size, false, GGML_PREC_F32}).build(ctx, context, weights.o_proj);
}

core::TensorValue flow_mlp(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const HeartCodecTransformerBlockWeights & weights,
    int64_t hidden_size) {
    const int64_t mlp_hidden_size = weights.mlp_gate.weight.shape.dims[0];
    auto gate = modules::LinearModule({hidden_size, mlp_hidden_size, false, GGML_PREC_F32}).build(ctx, input, weights.mlp_gate);
    gate = modules::SiluModule{}.build(ctx, gate);
    auto up = modules::LinearModule({hidden_size, mlp_hidden_size, false, GGML_PREC_F32}).build(ctx, input, weights.mlp_up);
    auto hidden = modules::MulModule{}.build(ctx, gate, up);
    return modules::LinearModule({mlp_hidden_size, hidden_size, false, GGML_PREC_F32}).build(ctx, hidden, weights.mlp_down);
}

core::TensorValue flow_transformer_block(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & timestep_mod,
    const core::TensorValue & positions,
    const HeartCodecTransformerBlockWeights & weights,
    int64_t heads,
    int64_t head_dim) {
    const int64_t hidden_size = heads * head_dim;
    const int64_t frames = input.shape.dims[1];
    const auto parts = adaptive_block_parts(ctx, timestep_mod, weights.scale_shift_table, frames, hidden_size);

    auto normed = modules::RMSNormModule({hidden_size, 1.0e-6F, true, false}).build(ctx, input, weights.attn_norm);
    normed = modules::AddModule{}.build(
        ctx,
        modules::MulModule{}.build(ctx, normed, add_one(ctx, parts.scale_msa)),
        parts.shift_msa);
    auto attn = flow_attention(ctx, normed, positions, weights, heads, head_dim);
    auto x = modules::AddModule{}.build(ctx, input, modules::MulModule{}.build(ctx, parts.gate_msa, attn));

    normed = modules::RMSNormModule({hidden_size, 1.0e-6F, true, false}).build(ctx, x, weights.mlp_norm);
    normed = modules::AddModule{}.build(
        ctx,
        modules::MulModule{}.build(ctx, normed, add_one(ctx, parts.scale_mlp)),
        parts.shift_mlp);
    auto mlp = flow_mlp(ctx, normed, weights, hidden_size);
    return modules::AddModule{}.build(ctx, x, modules::MulModule{}.build(ctx, parts.gate_mlp, mlp));
}

core::TensorValue final_layer_norm_modulation(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & embedded_timestep,
    const core::TensorValue & scale_shift_table,
    int64_t hidden_size) {
    const int64_t batch = input.shape.dims[0];
    const int64_t frames = input.shape.dims[1];
    auto table = core::reshape_tensor(ctx, scale_shift_table, core::TensorShape::from_dims({1, 2, hidden_size}));
    table = modules::RepeatModule({core::TensorShape::from_dims({batch, 2, hidden_size})}).build(ctx, table);
    auto embedded = core::reshape_tensor(ctx, embedded_timestep, core::TensorShape::from_dims({batch, 1, hidden_size}));
    embedded = modules::RepeatModule({core::TensorShape::from_dims({batch, 2, hidden_size})}).build(ctx, embedded);
    auto parts = modules::AddModule{}.build(ctx, embedded, table);
    auto shift = expand_batch_token(ctx, modules::SliceModule({1, 0, 1}).build(ctx, parts), frames, hidden_size);
    auto scale_part = expand_batch_token(ctx, modules::SliceModule({1, 1, 1}).build(ctx, parts), frames, hidden_size);
    auto normalized = modules::LayerNormModule({hidden_size, 1.0e-6F, false, false}).build(ctx, input, {});
    return modules::AddModule{}.build(
        ctx,
        modules::MulModule{}.build(ctx, normalized, add_one(ctx, scale_part)),
        shift);
}

FlowEstimatorBuildResult flow_estimator(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & hidden_states,
    const core::TensorValue & timesteps,
    const core::TensorValue & freqs,
    const core::TensorValue & positions,
    const HeartCodecFlowEstimatorWeights & weights,
    const HeartCodecConfig & config) {
    const int64_t estimator_dim = config.num_attention_heads * config.attention_head_dim;
    const int64_t estimator_dim_2 = 2 * estimator_dim;
    auto s = project_layer(ctx, hidden_states, weights.proj_in, config.in_channels, estimator_dim);
    auto embedded_timestep = adaln_embedding(ctx, timesteps, freqs, weights.adaln_single, estimator_dim);
    auto timestep_mod = modules::LinearModule({estimator_dim, 6 * estimator_dim, true, GGML_PREC_F32})
                            .build(ctx, modules::SiluModule{}.build(ctx, embedded_timestep), weights.adaln_single.linear);
    for (size_t index = 0; index < weights.transformer_blocks.size(); ++index) {
        const auto & block = weights.transformer_blocks[index];
        s = flow_transformer_block(ctx, s, timestep_mod, positions, block, config.num_attention_heads, config.attention_head_dim);
    }
    s = final_layer_norm_modulation(ctx, s, embedded_timestep, weights.scale_shift_table, estimator_dim);
    ggml_tensor * first_stage_state = s.tensor;

    auto x = modules::ConcatModule({2}).build(ctx, hidden_states, s);
    x = project_layer(ctx, x, weights.connection_proj, config.in_channels + estimator_dim, estimator_dim_2);
    auto embedded_timestep_2 = adaln_embedding(ctx, timesteps, freqs, weights.adaln_single_2, estimator_dim_2);
    auto timestep_mod_2 = modules::LinearModule({estimator_dim_2, 6 * estimator_dim_2, true, GGML_PREC_F32})
                              .build(ctx, modules::SiluModule{}.build(ctx, embedded_timestep_2), weights.adaln_single_2.linear);
    for (size_t index = 0; index < weights.transformer_blocks_2.size(); ++index) {
        const auto & block = weights.transformer_blocks_2[index];
        x = flow_transformer_block(ctx, x, timestep_mod_2, positions, block, config.num_attention_heads, config.attention_head_dim * 2);
    }
    x = final_layer_norm_modulation(ctx, x, embedded_timestep_2, weights.scale_shift_table_2, estimator_dim_2);
    return {project_layer(ctx, x, weights.proj_out, estimator_dim_2, config.out_channels), first_stage_state};
}

}  // namespace

class HeartCodecFlowEstimatorGraph {
public:
    HeartCodecFlowEstimatorGraph(
        const HeartCodecWeightsRuntime & runtime,
        int64_t batch_size,
        int64_t frames)
        : runtime_(&runtime),
          batch_size_(batch_size),
          frames_(frames) {
        if (batch_size_ <= 0 || frames_ <= 0) {
            throw std::runtime_error("HeartCodec flow estimator graph shape is invalid");
        }
        const auto & config = runtime_->assets().codec_config;
        ggml_init_params params{runtime_->flow_estimator_graph_arena_bytes(), nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize HeartCodec flow estimator graph context");
        }
        core::ModuleBuildContext ctx{ctx_.get(), "heartcodec.flow_estimator", runtime_->backend_type()};
        auto input = core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            core::TensorShape::from_dims({batch_size_, frames_, config.in_channels}));
        input_ = input.tensor;
        auto timesteps = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch_size_}));
        timesteps_ = timesteps.tensor;
        freqs_ = ggml_new_tensor_2d(ctx_.get(), GGML_TYPE_F32, 256, 1);
        auto freqs = core::wrap_tensor(freqs_, core::TensorShape::from_dims({1, 256}), GGML_TYPE_F32);
        positions_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, frames_);
        auto positions = core::wrap_tensor(positions_, core::TensorShape::from_dims({frames_}), GGML_TYPE_I32);
        auto out = flow_estimator(
            ctx,
            input,
            timesteps,
            freqs,
            positions,
            runtime_->weights().flow.estimator,
            config);
        output_ = out.output.tensor;
        first_stage_state_ = out.first_stage_state;
        ggml_set_output(output_);
        ggml_set_output(first_stage_state_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 262144, false);
        ggml_build_forward_expand(graph_, first_stage_state_);
        ggml_build_forward_expand(graph_, output_);
        allocate_workspace();
    }

    ~HeartCodecFlowEstimatorGraph() {
        release_workspace();
    }

    void release_workspace() {
        if (gallocr_ != nullptr) {
            engine::core::release_backend_graph_resources(runtime_->backend(), graph_);
            ggml_gallocr_free(gallocr_);
            gallocr_ = nullptr;
        }
    }

    bool matches(const HeartCodecWeightsRuntime & runtime, int64_t batch_size, int64_t frames) const {
        return runtime_ == &runtime && batch_size_ == batch_size && frames_ == frames;
    }

    HeartCodecFlowEstimate run(
        const std::vector<float> & hidden_states,
        const std::vector<float> & timesteps) {
        allocate_workspace();
        const auto & config = runtime_->assets().codec_config;
        const size_t expected_hidden = static_cast<size_t>(batch_size_ * frames_ * config.in_channels);
        if (hidden_states.size() != expected_hidden) {
            throw std::runtime_error("HeartCodec flow estimator hidden-state payload size mismatch");
        }
        if (static_cast<int64_t>(timesteps.size()) != batch_size_) {
            throw std::runtime_error("HeartCodec flow estimator timestep payload size mismatch");
        }
        ggml_backend_tensor_set(input_, hidden_states.data(), 0, hidden_states.size() * sizeof(float));
        ggml_backend_tensor_set(timesteps_, timesteps.data(), 0, timesteps.size() * sizeof(float));
        core::set_backend_threads(runtime_->backend(), runtime_->threads());
        const ggml_status status = core::compute_backend_graph(runtime_->backend(), graph_);
        ggml_backend_synchronize(runtime_->backend());
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("HeartCodec flow estimator graph compute failed");
        }
        HeartCodecFlowEstimate out;
        out.batch_size = batch_size_;
        out.frames = frames_;
        out.channels = config.out_channels;
        out.values.resize(static_cast<size_t>(out.batch_size * out.frames * out.channels));
        ggml_backend_tensor_get(output_, out.values.data(), 0, out.values.size() * sizeof(float));
        return out;
    }

private:
    void allocate_workspace() {
        if (gallocr_ != nullptr) {
            return;
        }
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(runtime_->backend()));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            engine::core::release_backend_graph_resources(runtime_->backend(), graph_);
            if (gallocr_ != nullptr) {
                ggml_gallocr_free(gallocr_);
                gallocr_ = nullptr;
            }
            throw std::runtime_error("failed to allocate HeartCodec flow estimator graph");
        }
        upload_static_inputs();
    }

    void upload_static_inputs() {
        std::vector<int32_t> position_values(static_cast<size_t>(frames_), 0);
        for (int64_t index = 0; index < frames_; ++index) {
            position_values[static_cast<size_t>(index)] = static_cast<int32_t>(index);
        }
        ggml_backend_tensor_set(positions_, position_values.data(), 0, position_values.size() * sizeof(int32_t));
        std::vector<float> freq_values(256, 0.0F);
        constexpr float max_period = 10000.0F;
        for (int64_t index = 0; index < 256; ++index) {
            freq_values[static_cast<size_t>(index)] =
                std::exp(-std::log(max_period) * static_cast<float>(index) / 256.0F);
        }
        ggml_backend_tensor_set(freqs_, freq_values.data(), 0, freq_values.size() * sizeof(float));
    }
    const HeartCodecWeightsRuntime * runtime_ = nullptr;
    int64_t batch_size_ = 0;
    int64_t frames_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * timesteps_ = nullptr;
    ggml_tensor * freqs_ = nullptr;
    ggml_tensor * positions_ = nullptr;
    ggml_tensor * output_ = nullptr;
    ggml_tensor * first_stage_state_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
};

class HeartCodecConditioningGraph {
public:
    HeartCodecConditioningGraph(
        const HeartCodecWeightsRuntime & runtime,
        int64_t batch_size,
        int64_t code_frames)
        : runtime_(&runtime),
          batch_size_(batch_size),
          code_frames_(code_frames) {
        if (batch_size_ <= 0 || code_frames_ <= 0) {
            throw std::runtime_error("HeartCodec conditioning graph shape is invalid");
        }
        const auto & config = runtime_->assets().codec_config;
        ggml_init_params params{runtime_->conditioning_graph_arena_bytes(), nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize HeartCodec conditioning graph context");
        }
        core::ModuleBuildContext ctx{ctx_.get(), "heartcodec.conditioning", runtime_->backend_type()};
        auto input = core::make_tensor(
            ctx,
            GGML_TYPE_I32,
            core::TensorShape::from_dims({batch_size_, config.num_quantizers, code_frames_}));
        input_ = input.tensor;
        auto out = conditioning_from_codes(ctx, input, runtime_->weights().flow, config);
        output_ = out.tensor;
        output_frames_ = out.shape.dims[1];
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 65536, false);
        ggml_build_forward_expand(graph_, output_);
        allocate_workspace();
    }

    ~HeartCodecConditioningGraph() {
        release_workspace();
    }

    void release_workspace() {
        if (gallocr_ != nullptr) {
            engine::core::release_backend_graph_resources(runtime_->backend(), graph_);
            ggml_gallocr_free(gallocr_);
            gallocr_ = nullptr;
        }
    }

    bool matches(const HeartCodecWeightsRuntime & runtime, int64_t batch_size, int64_t code_frames) const {
        return runtime_ == &runtime && batch_size_ == batch_size && code_frames_ == code_frames;
    }

    HeartCodecConditioning run(const std::vector<int32_t> & codes) {
        allocate_workspace();
        const auto & config = runtime_->assets().codec_config;
        const size_t expected_codes = static_cast<size_t>(batch_size_ * config.num_quantizers * code_frames_);
        if (codes.size() != expected_codes) {
            throw std::runtime_error("HeartCodec conditioning code payload size mismatch");
        }
        ggml_backend_tensor_set(input_, codes.data(), 0, codes.size() * sizeof(int32_t));
        core::set_backend_threads(runtime_->backend(), runtime_->threads());
        const ggml_status status = core::compute_backend_graph(runtime_->backend(), graph_);
        ggml_backend_synchronize(runtime_->backend());
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("HeartCodec conditioning graph compute failed");
        }
        HeartCodecConditioning out;
        out.batch_size = batch_size_;
        out.frames = output_frames_;
        out.channels = config.dim;
        out.values.resize(static_cast<size_t>(out.batch_size * out.frames * out.channels));
        ggml_backend_tensor_get(output_, out.values.data(), 0, out.values.size() * sizeof(float));
        return out;
    }

private:
    void allocate_workspace() {
        if (gallocr_ != nullptr) {
            return;
        }
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(runtime_->backend()));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            engine::core::release_backend_graph_resources(runtime_->backend(), graph_);
            if (gallocr_ != nullptr) {
                ggml_gallocr_free(gallocr_);
                gallocr_ = nullptr;
            }
            throw std::runtime_error("failed to allocate HeartCodec conditioning graph");
        }
    }
    const HeartCodecWeightsRuntime * runtime_ = nullptr;
    int64_t batch_size_ = 0;
    int64_t code_frames_ = 0;
    int64_t output_frames_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
};

class HeartCodecScalarDecoderGraph {
public:
    HeartCodecScalarDecoderGraph(
        const HeartCodecWeightsRuntime & runtime,
        int64_t batch_size,
        int64_t frames)
        : runtime_(&runtime),
          batch_size_(batch_size),
          frames_(frames) {
        if (batch_size_ <= 0 || frames_ <= 0) {
            throw std::runtime_error("HeartCodec scalar decoder graph shape is invalid");
        }
        const auto & config = runtime_->assets().codec_config;
        ggml_init_params params{runtime_->scalar_decoder_graph_arena_bytes(), nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize HeartCodec scalar decoder graph context");
        }
        core::ModuleBuildContext ctx{ctx_.get(), "heartcodec.scalar_decoder", runtime_->backend_type()};
        auto input = core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            core::TensorShape::from_dims({batch_size_, config.latent_hidden_dim, frames_}));
        input_ = input.tensor;
        auto out = scalar_decoder(ctx, input, runtime_->weights().scalar_decoder, config);
        output_ = out.tensor;
        output_samples_ = out.shape.dims[2];
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 262144, false);
        ggml_build_forward_expand(graph_, output_);
        allocate_workspace();
    }

    ~HeartCodecScalarDecoderGraph() {
        release_workspace();
    }

    void release_workspace() {
        if (gallocr_ != nullptr) {
            engine::core::release_backend_graph_resources(runtime_->backend(), graph_);
            ggml_gallocr_free(gallocr_);
            gallocr_ = nullptr;
        }
    }

    bool matches(const HeartCodecWeightsRuntime & runtime, int64_t batch_size, int64_t frames) const {
        return runtime_ == &runtime && batch_size_ == batch_size && frames_ == frames;
    }

    HeartCodecDecodedAudio run(const std::vector<float> & latents) {
        allocate_workspace();
        const auto & config = runtime_->assets().codec_config;
        const size_t expected_latents = static_cast<size_t>(batch_size_ * config.latent_hidden_dim * frames_);
        if (latents.size() != expected_latents) {
            throw std::runtime_error("HeartCodec scalar decoder latent payload size mismatch");
        }
        ggml_backend_tensor_set(input_, latents.data(), 0, latents.size() * sizeof(float));
        core::set_backend_threads(runtime_->backend(), runtime_->threads());
        const ggml_status status = core::compute_backend_graph(runtime_->backend(), graph_);
        ggml_backend_synchronize(runtime_->backend());
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("HeartCodec scalar decoder graph compute failed");
        }
        HeartCodecDecodedAudio out;
        out.batch_size = batch_size_;
        out.channels = config.num_bands;
        out.samples = output_samples_;
        out.values.resize(static_cast<size_t>(out.batch_size * out.channels * out.samples));
        ggml_backend_tensor_get(output_, out.values.data(), 0, out.values.size() * sizeof(float));
        return out;
    }

private:
    void allocate_workspace() {
        if (gallocr_ != nullptr) {
            return;
        }
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(runtime_->backend()));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            engine::core::release_backend_graph_resources(runtime_->backend(), graph_);
            if (gallocr_ != nullptr) {
                ggml_gallocr_free(gallocr_);
                gallocr_ = nullptr;
            }
            throw std::runtime_error("failed to allocate HeartCodec scalar decoder graph");
        }
    }
    const HeartCodecWeightsRuntime * runtime_ = nullptr;
    int64_t batch_size_ = 0;
    int64_t frames_ = 0;
    int64_t output_samples_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
};

HeartCodecWeights load_heartcodec_weights(
    const HeartMuLaAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t weight_context_bytes,
    assets::TensorStorageType weight_storage_type) {
    require_assets(assets);
    HeartCodecWeights weights;
    weights.store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "heartmula.codec.weights",
        weight_context_bytes);
    const auto & source = *assets.codec_weights;
    weights.flow = load_flow_weights(*weights.store, source, assets.codec_config, weight_storage_type);
    weights.scalar_decoder =
        load_scalar_decoder_weights(*weights.store, source, assets.codec_config, weight_storage_type);
    weights.store->upload();
    return weights;
}

HeartCodecWeightsRuntime::HeartCodecWeightsRuntime(
    std::shared_ptr<const HeartMuLaAssets> assets,
    core::ExecutionContext & execution_context,
    size_t weight_context_bytes,
    size_t flow_estimator_graph_arena_bytes,
    size_t conditioning_graph_arena_bytes,
    size_t scalar_decoder_graph_arena_bytes,
    assets::TensorStorageType weight_storage_type)
    : assets_(std::move(assets)),
      execution_context_(&execution_context),
      flow_estimator_graph_arena_bytes_(flow_estimator_graph_arena_bytes),
      conditioning_graph_arena_bytes_(conditioning_graph_arena_bytes),
      scalar_decoder_graph_arena_bytes_(scalar_decoder_graph_arena_bytes) {
    if (assets_ == nullptr) {
        throw std::runtime_error("HeartCodec weights runtime requires assets");
    }
    if (execution_context_ == nullptr || execution_context_->backend() == nullptr) {
        throw std::runtime_error("HeartCodec weights runtime requires initialized execution context");
    }
    if (execution_context_->config().threads <= 0) {
        throw std::runtime_error("HeartCodec weights runtime requires positive thread count");
    }
    weights_ = std::make_shared<HeartCodecWeights>(
        load_heartcodec_weights(
            *assets_,
            execution_context_->backend(),
            execution_context_->backend_type(),
            weight_context_bytes,
            weight_storage_type));
    assets_->codec_weights->release_storage();
}

HeartCodecWeightsRuntime::~HeartCodecWeightsRuntime() {
    scalar_decoder_graph_.reset();
    conditioning_graph_.reset();
    flow_estimator_graph_.reset();
    weights_.reset();
}

const HeartMuLaAssets & HeartCodecWeightsRuntime::assets() const noexcept {
    return *assets_;
}

const HeartCodecWeights & HeartCodecWeightsRuntime::weights() const noexcept {
    return *weights_;
}

ggml_backend_t HeartCodecWeightsRuntime::backend() const noexcept {
    return execution_context_->backend();
}

core::BackendType HeartCodecWeightsRuntime::backend_type() const noexcept {
    return execution_context_->backend_type();
}

int HeartCodecWeightsRuntime::device() const noexcept {
    return execution_context_->config().device;
}

int HeartCodecWeightsRuntime::threads() const noexcept {
    return std::max(1, execution_context_->config().threads);
}

size_t HeartCodecWeightsRuntime::flow_estimator_graph_arena_bytes() const noexcept {
    return flow_estimator_graph_arena_bytes_;
}

size_t HeartCodecWeightsRuntime::conditioning_graph_arena_bytes() const noexcept {
    return conditioning_graph_arena_bytes_;
}

size_t HeartCodecWeightsRuntime::scalar_decoder_graph_arena_bytes() const noexcept {
    return scalar_decoder_graph_arena_bytes_;
}

HeartCodecFlowEstimate HeartCodecWeightsRuntime::estimate_flow(
    const std::vector<float> & hidden_states,
    int64_t batch_size,
    int64_t frames,
    const std::vector<float> & timesteps) const {
    if (batch_size <= 0 || frames <= 0) {
        throw std::runtime_error("HeartCodec flow estimator request shape is invalid");
    }
    if (scalar_decoder_graph_ != nullptr) {
        scalar_decoder_graph_->release_workspace();
    }
    if (flow_estimator_graph_ == nullptr || !flow_estimator_graph_->matches(*this, batch_size, frames)) {
        flow_estimator_graph_ = std::make_unique<HeartCodecFlowEstimatorGraph>(*this, batch_size, frames);
    }
    return flow_estimator_graph_->run(hidden_states, timesteps);
}

HeartCodecDecodedAudio HeartCodecWeightsRuntime::decode_scalar_latents(
    const std::vector<float> & latents,
    int64_t batch_size,
    int64_t frames) const {
    if (batch_size <= 0 || frames <= 0) {
        throw std::runtime_error("HeartCodec scalar decoder request shape is invalid");
    }
    if (flow_estimator_graph_ != nullptr) {
        flow_estimator_graph_->release_workspace();
    }
    if (conditioning_graph_ != nullptr) {
        conditioning_graph_->release_workspace();
    }
    if (scalar_decoder_graph_ == nullptr || !scalar_decoder_graph_->matches(*this, batch_size, frames)) {
        scalar_decoder_graph_ = std::make_unique<HeartCodecScalarDecoderGraph>(*this, batch_size, frames);
    }
    return scalar_decoder_graph_->run(latents);
}

HeartCodecConditioning HeartCodecWeightsRuntime::build_conditioning(
    const std::vector<int32_t> & codes,
    int64_t batch_size,
    int64_t code_frames) const {
    if (batch_size <= 0 || code_frames <= 0) {
        throw std::runtime_error("HeartCodec conditioning request shape is invalid");
    }
    if (scalar_decoder_graph_ != nullptr) {
        scalar_decoder_graph_->release_workspace();
    }
    if (conditioning_graph_ == nullptr || !conditioning_graph_->matches(*this, batch_size, code_frames)) {
        conditioning_graph_ = std::make_unique<HeartCodecConditioningGraph>(*this, batch_size, code_frames);
    }
    return conditioning_graph_->run(codes);
}

void HeartCodecWeightsRuntime::clear_graph_cache() const {
    flow_estimator_graph_.reset();
    conditioning_graph_.reset();
    scalar_decoder_graph_.reset();
}

namespace {

std::vector<int32_t> frame_major_to_bqt_codes(
    const std::vector<int32_t> & codes,
    int64_t frames,
    int64_t codebooks) {
    if (frames <= 0 || codebooks <= 0 || static_cast<int64_t>(codes.size()) != frames * codebooks) {
        throw std::runtime_error("HeartCodec detokenize code shape mismatch");
    }
    std::vector<int32_t> out(static_cast<size_t>(codebooks * frames), 0);
    for (int64_t t = 0; t < frames; ++t) {
        for (int64_t q = 0; q < codebooks; ++q) {
            out[static_cast<size_t>(q * frames + t)] = codes[static_cast<size_t>(t * codebooks + q)];
        }
    }
    return out;
}

std::vector<int32_t> repeat_codes_to_length(
    const std::vector<int32_t> & codes_bqt,
    int64_t codebooks,
    int64_t frames,
    int64_t target_frames) {
    if (codebooks <= 0 || frames <= 0 || target_frames <= 0 ||
        static_cast<int64_t>(codes_bqt.size()) != codebooks * frames) {
        throw std::runtime_error("HeartCodec code repeat shape mismatch");
    }
    std::vector<int32_t> out(static_cast<size_t>(codebooks * target_frames), 0);
    for (int64_t q = 0; q < codebooks; ++q) {
        for (int64_t t = 0; t < target_frames; ++t) {
            out[static_cast<size_t>(q * target_frames + t)] =
                codes_bqt[static_cast<size_t>(q * frames + (t % frames))];
        }
    }
    return out;
}

std::vector<int32_t> slice_codes_bqt(
    const std::vector<int32_t> & codes_bqt,
    int64_t codebooks,
    int64_t frames,
    int64_t start,
    int64_t length) {
    if (start < 0 || length <= 0 || start + length > frames ||
        static_cast<int64_t>(codes_bqt.size()) != codebooks * frames) {
        throw std::runtime_error("HeartCodec code slice range mismatch");
    }
    std::vector<int32_t> out(static_cast<size_t>(codebooks * length), 0);
    for (int64_t q = 0; q < codebooks; ++q) {
        std::copy(
            codes_bqt.begin() + static_cast<std::ptrdiff_t>(q * frames + start),
            codes_bqt.begin() + static_cast<std::ptrdiff_t>(q * frames + start + length),
            out.begin() + static_cast<std::ptrdiff_t>(q * length));
    }
    return out;
}

void apply_inactive_conditioning(
    std::vector<float> & mu,
    int64_t frames,
    int64_t dim,
    int64_t latent_length,
    const std::vector<float> & zero_cond) {
    if (static_cast<int64_t>(mu.size()) != frames * dim || static_cast<int64_t>(zero_cond.size()) != dim) {
        throw std::runtime_error("HeartCodec conditioning mask shape mismatch");
    }
    for (int64_t t = latent_length; t < frames; ++t) {
        std::copy(
            zero_cond.begin(),
            zero_cond.end(),
            mu.begin() + static_cast<std::ptrdiff_t>(t * dim));
    }
}

std::vector<float> make_flow_input(
    const std::vector<float> & x,
    const std::vector<float> & incontext,
    const std::vector<float> & mu,
    int64_t batch,
    int64_t frames,
    int64_t latent_dim,
    int64_t cond_dim,
    bool conditional_mu) {
    if (static_cast<int64_t>(x.size()) != batch * frames * latent_dim ||
        static_cast<int64_t>(incontext.size()) != batch * frames * latent_dim ||
        static_cast<int64_t>(mu.size()) != batch * frames * cond_dim) {
        throw std::runtime_error("HeartCodec flow input shape mismatch");
    }
    std::vector<float> out(static_cast<size_t>(batch * frames * (2 * latent_dim + cond_dim)), 0.0F);
    const int64_t channels = 2 * latent_dim + cond_dim;
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t t = 0; t < frames; ++t) {
            const size_t x_base = static_cast<size_t>((b * frames + t) * latent_dim);
            const size_t mu_base = static_cast<size_t>((b * frames + t) * cond_dim);
            const size_t dst = static_cast<size_t>((b * frames + t) * channels);
            std::copy(x.begin() + static_cast<std::ptrdiff_t>(x_base),
                      x.begin() + static_cast<std::ptrdiff_t>(x_base + latent_dim),
                      out.begin() + static_cast<std::ptrdiff_t>(dst));
            std::copy(incontext.begin() + static_cast<std::ptrdiff_t>(x_base),
                      incontext.begin() + static_cast<std::ptrdiff_t>(x_base + latent_dim),
                      out.begin() + static_cast<std::ptrdiff_t>(dst + latent_dim));
            if (conditional_mu) {
                std::copy(mu.begin() + static_cast<std::ptrdiff_t>(mu_base),
                          mu.begin() + static_cast<std::ptrdiff_t>(mu_base + cond_dim),
                          out.begin() + static_cast<std::ptrdiff_t>(dst + 2 * latent_dim));
            }
        }
    }
    return out;
}

std::vector<float> solve_euler(
    const HeartCodecWeightsRuntime & runtime,
    std::vector<float> x,
    const std::vector<float> & incontext_x,
    int64_t incontext_length,
    const std::vector<float> & mu,
    int64_t num_steps,
    float guidance_scale) {
    const auto & config = runtime.assets().codec_config;
    const int64_t batch = 1;
    const int64_t frames = static_cast<int64_t>(mu.size()) / config.dim;
    const int64_t latent_dim = config.out_channels;
    if (num_steps <= 0 || frames <= 0 ||
        static_cast<int64_t>(x.size()) != batch * frames * latent_dim ||
        static_cast<int64_t>(incontext_x.size()) != batch * frames * latent_dim) {
        throw std::runtime_error("HeartCodec Euler solver shape mismatch");
    }
    const std::vector<float> noise = x;
    float t = 0.0F;
    float dt = 1.0F / static_cast<float>(num_steps);
    for (int64_t step = 1; step <= num_steps; ++step) {
        for (int64_t frame = 0; frame < incontext_length; ++frame) {
            for (int64_t channel = 0; channel < latent_dim; ++channel) {
                const size_t index = static_cast<size_t>(frame * latent_dim + channel);
                x[index] = (1.0F - (1.0F - 1.0e-6F) * t) * noise[index] + t * incontext_x[index];
            }
        }
        std::vector<float> dphi;
        if (guidance_scale > 1.0F) {
            auto uncond = make_flow_input(x, incontext_x, mu, batch, frames, latent_dim, config.dim, false);
            auto cond = make_flow_input(x, incontext_x, mu, batch, frames, latent_dim, config.dim, true);
            uncond.insert(uncond.end(), cond.begin(), cond.end());
            const std::vector<float> timesteps = {t, t};
            auto estimate = runtime.estimate_flow(uncond, 2, frames, timesteps);
            dphi.resize(static_cast<size_t>(frames * latent_dim), 0.0F);
            const size_t row_elems = static_cast<size_t>(frames * latent_dim);
            for (size_t i = 0; i < row_elems; ++i) {
                const float uncond_value = estimate.values[i];
                const float cond_value = estimate.values[row_elems + i];
                dphi[i] = uncond_value + guidance_scale * (cond_value - uncond_value);
            }
        } else {
            auto input = make_flow_input(x, incontext_x, mu, batch, frames, latent_dim, config.dim, true);
            const std::vector<float> timesteps = {t};
            dphi = runtime.estimate_flow(input, 1, frames, timesteps).values;
        }
        for (size_t i = 0; i < x.size(); ++i) {
            x[i] += dt * dphi[i];
        }
        t += dt;
        if (step < num_steps) {
            dt = static_cast<float>(step + 1) / static_cast<float>(num_steps) - t;
        }
    }
    return x;
}

std::vector<float> run_flow_chunk(
    const HeartCodecWeightsRuntime & runtime,
    const std::vector<int32_t> & codes_bqt,
    int64_t code_frames,
    int64_t latent_length,
    int64_t incontext_length,
    const std::vector<float> & true_latents,
    int64_t num_steps,
    float guidance_scale,
    uint64_t seed,
    uint64_t & randn_philox_offset,
    uint64_t randn_call_offset_blocks,
    const engine::sampling::TorchCudaSamplingPolicy & sampling_policy) {
    const auto & config = runtime.assets().codec_config;
    auto conditioning = runtime.build_conditioning(codes_bqt, 1, code_frames);
    if (conditioning.frames <= 0 || conditioning.channels != config.dim) {
        throw std::runtime_error("HeartCodec conditioning output shape mismatch");
    }
    apply_inactive_conditioning(
        conditioning.values,
        conditioning.frames,
        conditioning.channels,
        latent_length,
        runtime.weights().flow.zero_cond_embedding_host.values);
    if (randn_call_offset_blocks == 0) {
        throw std::runtime_error("HeartCodec CUDA randn call offset must be positive");
    }
    auto latents = engine::sampling::generate_torch_cuda_tensor_iterator_randn(
        static_cast<size_t>(conditioning.frames * config.out_channels),
        seed,
        randn_philox_offset,
        sampling_policy);
    randn_philox_offset += randn_call_offset_blocks;
    std::vector<float> incontext(static_cast<size_t>(conditioning.frames * config.out_channels), 0.0F);
    if (static_cast<int64_t>(true_latents.size()) != latent_length * config.out_channels) {
        throw std::runtime_error("HeartCodec true-latent payload shape mismatch");
    }
    for (int64_t t = 0; t < incontext_length; ++t) {
        std::copy(
            true_latents.begin() + static_cast<std::ptrdiff_t>(t * config.out_channels),
            true_latents.begin() + static_cast<std::ptrdiff_t>((t + 1) * config.out_channels),
            incontext.begin() + static_cast<std::ptrdiff_t>(t * config.out_channels));
    }
    auto solved = solve_euler(
        runtime,
        std::move(latents),
        incontext,
        incontext_length,
        conditioning.values,
        num_steps,
        guidance_scale);
    for (int64_t t = 0; t < incontext_length; ++t) {
        std::copy(
            incontext.begin() + static_cast<std::ptrdiff_t>(t * config.out_channels),
            incontext.begin() + static_cast<std::ptrdiff_t>((t + 1) * config.out_channels),
            solved.begin() + static_cast<std::ptrdiff_t>(t * config.out_channels));
    }
    return solved;
}

std::vector<float> latent_btc_to_scalar_bct(const std::vector<float> & latent_btc, int64_t frames) {
    const int64_t channels = 2;
    const int64_t latent_hidden = 128;
    if (static_cast<int64_t>(latent_btc.size()) != frames * channels * latent_hidden) {
        throw std::runtime_error("HeartCodec latent reshape payload mismatch");
    }
    std::vector<float> out(static_cast<size_t>(channels * latent_hidden * frames), 0.0F);
    for (int64_t t = 0; t < frames; ++t) {
        for (int64_t stream = 0; stream < channels; ++stream) {
            for (int64_t dim = 0; dim < latent_hidden; ++dim) {
                const size_t src = static_cast<size_t>(t * channels * latent_hidden + stream * latent_hidden + dim);
                const size_t dst = static_cast<size_t>((stream * latent_hidden + dim) * frames + t);
                out[dst] = latent_btc[src];
            }
        }
    }
    return out;
}

std::vector<float> decoded_batch_to_stereo(
    const HeartCodecDecodedAudio & decoded,
    int64_t samples) {
    if (decoded.batch_size != 2 || decoded.channels != 1 || decoded.samples < samples) {
        throw std::runtime_error("HeartCodec scalar decoder output shape mismatch");
    }
    std::vector<float> out(static_cast<size_t>(2 * samples), 0.0F);
    for (int64_t ch = 0; ch < 2; ++ch) {
        std::copy(
            decoded.values.begin() + static_cast<std::ptrdiff_t>(ch * decoded.samples),
            decoded.values.begin() + static_cast<std::ptrdiff_t>(ch * decoded.samples + samples),
            out.begin() + static_cast<std::ptrdiff_t>(ch * samples));
    }
    return out;
}

std::vector<float> planar_stereo_to_interleaved(
    const std::vector<float> & planar,
    int64_t samples) {
    if (samples < 0 || static_cast<int64_t>(planar.size()) != 2 * samples) {
        throw std::runtime_error("HeartCodec stereo output shape mismatch");
    }
    std::vector<float> interleaved(static_cast<size_t>(2 * samples), 0.0F);
    for (int64_t t = 0; t < samples; ++t) {
        interleaved[static_cast<size_t>(2 * t)] = planar[static_cast<size_t>(t)];
        interleaved[static_cast<size_t>(2 * t + 1)] = planar[static_cast<size_t>(samples + t)];
    }
    return interleaved;
}

}  // namespace

HeartCodecDecodedAudio HeartCodecWeightsRuntime::detokenize_codes(
    const std::vector<int32_t> & codes,
    int64_t frames,
    int64_t codebooks,
    const HeartMuLaGenerationOptions & options,
    uint64_t seed,
    uint64_t randn_philox_offset,
    uint64_t randn_call_offset_blocks) const {
    const auto & config = assets().codec_config;
    if (codebooks != config.num_quantizers) {
        throw std::runtime_error("HeartCodec detokenize codebook count mismatch");
    }
    if (!(options.codec_duration > 0.0F) || options.num_inference_steps <= 0 || !(options.codec_guidance_scale > 0.0F)) {
        throw std::runtime_error("HeartCodec detokenize options are invalid");
    }
    const auto sampling_policy = engine::sampling::resolve_torch_cuda_sampling_policy(
        backend_type(),
        device(),
        "heartmula.codec.cuda_sampling_policy",
        "HeartCodec",
        engine::sampling::TorchCudaSamplingPolicyFailureMode::StrictCuda);
    const int64_t original_frames = frames;
    auto working_codes = frame_major_to_bqt_codes(codes, frames, codebooks);
    const int64_t first_latent_length = 0;
    const int64_t first_latent_codes_length = 0;
    const int64_t min_code_samples = static_cast<int64_t>(options.codec_duration * 12.5F);
    const int64_t hop_code_samples = (min_code_samples / 93) * 80;
    const int64_t ovlp_code_samples = min_code_samples - hop_code_samples;
    const int64_t ovlp_frames = ovlp_code_samples * 2;
    if (min_code_samples <= 0 || hop_code_samples <= 0) {
        throw std::runtime_error("HeartCodec detokenize duration creates invalid chunking");
    }
    const int64_t target_len = static_cast<int64_t>(
        (static_cast<float>(original_frames - first_latent_codes_length) / 12.5F) * static_cast<float>(config.sample_rate));
    if (frames < min_code_samples) {
        working_codes = repeat_codes_to_length(working_codes, codebooks, frames, min_code_samples);
        frames = min_code_samples;
    }
    if ((frames - ovlp_frames) % hop_code_samples > 0) {
        const int64_t padded_frames = static_cast<int64_t>(
            std::ceil(static_cast<double>(frames - ovlp_code_samples) / static_cast<double>(hop_code_samples))) *
                hop_code_samples +
            ovlp_code_samples;
        working_codes = repeat_codes_to_length(working_codes, codebooks, frames, padded_frames);
        frames = padded_frames;
    }
    const int64_t latent_length = static_cast<int64_t>(options.codec_duration * 25.0F);
    std::vector<std::vector<float>> latent_list;
    for (int64_t sinx = 0; sinx < frames - hop_code_samples + 1; sinx += hop_code_samples) {
        const auto chunk_codes = slice_codes_bqt(working_codes, codebooks, frames, sinx, min_code_samples);
        int64_t incontext_length = first_latent_length;
        std::vector<float> true_latent(static_cast<size_t>(latent_length * config.out_channels), 0.0F);
        if (sinx != 0 && ovlp_frames != 0) {
            const auto & prev = latent_list.back();
            incontext_length = ovlp_frames;
            if (static_cast<int64_t>(prev.size()) < ovlp_frames * config.out_channels) {
                throw std::runtime_error("HeartCodec overlap latent shape mismatch");
            }
            std::copy(
                prev.end() - static_cast<std::ptrdiff_t>(ovlp_frames * config.out_channels),
                prev.end(),
                true_latent.begin());
        }
        auto latents = run_flow_chunk(
            *this,
            chunk_codes,
            min_code_samples,
            latent_length,
            incontext_length,
            true_latent,
            options.num_inference_steps,
            options.codec_guidance_scale,
            seed,
            randn_philox_offset,
            randn_call_offset_blocks,
            sampling_policy);
        latent_list.push_back(std::move(latents));
    }
    if (latent_list.empty()) {
        throw std::runtime_error("HeartCodec detokenize produced no latent chunks");
    }
    if (first_latent_length > 0) {
        auto & first = latent_list.front();
        first.erase(first.begin(), first.begin() + static_cast<std::ptrdiff_t>(first_latent_length * config.out_channels));
    }

    const int64_t audio_chunk_samples = static_cast<int64_t>(options.codec_duration * static_cast<float>(config.sample_rate));
    const int64_t audio_hop_samples = (audio_chunk_samples / 93) * 80;
    const int64_t audio_ovlp_samples = audio_chunk_samples - audio_hop_samples;
    std::vector<float> output;
    int64_t output_samples = 0;
    for (size_t chunk = 0; chunk < latent_list.size(); ++chunk) {
        const int64_t chunk_latent_frames = static_cast<int64_t>(latent_list[chunk].size()) / config.out_channels;
        auto scalar_input = latent_btc_to_scalar_bct(latent_list[chunk], chunk_latent_frames);
        auto decoded = decode_scalar_latents(scalar_input, 2, chunk_latent_frames);
        auto current = decoded_batch_to_stereo(decoded, audio_chunk_samples);
        if (output.empty()) {
            output = std::move(current);
            output_samples = audio_chunk_samples;
        } else if (audio_ovlp_samples == 0) {
            const int64_t new_samples = output_samples + audio_chunk_samples;
            std::vector<float> merged(static_cast<size_t>(2 * new_samples), 0.0F);
            for (int64_t ch = 0; ch < 2; ++ch) {
                std::copy(
                    output.begin() + static_cast<std::ptrdiff_t>(ch * output_samples),
                    output.begin() + static_cast<std::ptrdiff_t>((ch + 1) * output_samples),
                    merged.begin() + static_cast<std::ptrdiff_t>(ch * new_samples));
                std::copy(
                    current.begin() + static_cast<std::ptrdiff_t>(ch * audio_chunk_samples),
                    current.begin() + static_cast<std::ptrdiff_t>((ch + 1) * audio_chunk_samples),
                    merged.begin() + static_cast<std::ptrdiff_t>(ch * new_samples + output_samples));
            }
            output = std::move(merged);
            output_samples = new_samples;
        } else {
            const int64_t new_samples = output_samples + audio_chunk_samples - audio_ovlp_samples;
            std::vector<float> merged(static_cast<size_t>(2 * new_samples), 0.0F);
            for (int64_t ch = 0; ch < 2; ++ch) {
                std::copy(
                    output.begin() + static_cast<std::ptrdiff_t>(ch * output_samples),
                    output.begin() + static_cast<std::ptrdiff_t>((ch + 1) * output_samples),
                    merged.begin() + static_cast<std::ptrdiff_t>(ch * new_samples));
                for (int64_t t = 0; t < audio_ovlp_samples; ++t) {
                    const float left_weight = static_cast<float>(t) / static_cast<float>(audio_ovlp_samples - 1);
                    const float right_weight = 1.0F - left_weight;
                    const size_t dst = static_cast<size_t>(ch * new_samples + output_samples - audio_ovlp_samples + t);
                    const size_t old_src = static_cast<size_t>(ch * output_samples + output_samples - audio_ovlp_samples + t);
                    const size_t cur_src = static_cast<size_t>(ch * audio_chunk_samples + t);
                    merged[dst] = output[old_src] * right_weight + current[cur_src] * left_weight;
                }
                std::copy(
                    current.begin() + static_cast<std::ptrdiff_t>(ch * audio_chunk_samples + audio_ovlp_samples),
                    current.begin() + static_cast<std::ptrdiff_t>((ch + 1) * audio_chunk_samples),
                    merged.begin() + static_cast<std::ptrdiff_t>(ch * new_samples + output_samples));
            }
            output = std::move(merged);
            output_samples = new_samples;
        }
    }
    const int64_t final_samples = std::min<int64_t>(target_len, output_samples);
    std::vector<float> trimmed(static_cast<size_t>(2 * final_samples), 0.0F);
    for (int64_t ch = 0; ch < 2; ++ch) {
        std::copy(
            output.begin() + static_cast<std::ptrdiff_t>(ch * output_samples),
            output.begin() + static_cast<std::ptrdiff_t>(ch * output_samples + final_samples),
            trimmed.begin() + static_cast<std::ptrdiff_t>(ch * final_samples));
    }
    HeartCodecDecodedAudio out;
    out.batch_size = 1;
    out.channels = 2;
    out.samples = final_samples;
    out.values = planar_stereo_to_interleaved(trimmed, final_samples);
    return out;
}

}  // namespace engine::models::heartmula
