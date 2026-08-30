#include "engine/community_models/sopro_tts/acoustic.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/scaled_dot_product_attention.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::community_models::sopro_tts {

struct SoproDiTBlockWeights {
    engine::modules::LinearWeights modulation;  // attn_norm.mod.1: dim -> 6 * dim
    engine::modules::LinearWeights to_q;
    engine::modules::LinearWeights to_k;
    engine::modules::LinearWeights to_v;
    engine::modules::LinearWeights to_out;
    engine::modules::LinearWeights ff_in;
    engine::modules::LinearWeights ff_out;
};

// A Conv1d with `groups` > 1 and groups != channels, expressed as one
// independent convolution per group (ggml has no grouped conv1d primitive).
struct SoproGroupedConvWeights {
    std::vector<engine::modules::Conv1dWeights> groups;
    int64_t group_in_channels = 0;
    int64_t group_out_channels = 0;
    int64_t kernel_size = 0;
};

struct SoproAcousticWeights {
    std::shared_ptr<engine::core::BackendWeightStore> store;
    engine::core::TensorValue semantic_token_emb;  // [semantic_vocab, latent_dim]
    engine::modules::Conv1dWeights prelook_conv1;
    engine::modules::Conv1dWeights prelook_conv2;
    engine::modules::Conv1dWeights upsampler_in;
    engine::modules::Conv1dWeights upsampler_mix;
    engine::modules::Conv1dWeights upsampler_out;
    engine::modules::Conv1dWeights mu_proj;
    engine::modules::LinearWeights input_proj;
    engine::modules::LinearWeights cond_mask_proj;
    SoproGroupedConvWeights pos_conv1;
    SoproGroupedConvWeights pos_conv2;
    std::vector<SoproDiTBlockWeights> blocks;
    engine::modules::LinearWeights out_modulation;  // out_norm.mod.1: dim -> 2 * dim
    engine::modules::LinearWeights out_proj;
    // Host-side conditioning projections.
    std::vector<float> time_mlp_w0;  // [dim, time_embed_dim]
    std::vector<float> time_mlp_b0;
    std::vector<float> time_mlp_w2;  // [dim, dim]
    std::vector<float> time_mlp_b2;
    std::vector<float> spk_proj_w;   // [spk_dim, cond_hidden_dim]
    std::vector<float> spk_proj_b;
};

namespace {

namespace binding = engine::modules::binding;
namespace mod = engine::modules;

constexpr float kDiTLayerNormEps = 1.0e-6F;

// Set SOPRO_DUMP_DIR to write the solver's intermediates as raw f32 for
// stage-by-stage comparison against the reference implementation.
void dump(const std::string & name, const std::vector<float> & values) {
    const char * dir = std::getenv("SOPRO_DUMP_DIR");
    if (dir == nullptr) {
        return;
    }
    const std::string path = std::string(dir) + "/" + name + ".f32";
    std::FILE * fh = std::fopen(path.c_str(), "wb");
    if (fh != nullptr) {
        std::fwrite(values.data(), sizeof(float), values.size(), fh);
        std::fclose(fh);
    }
}

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

engine::core::TensorValue dense(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & value) {
    return engine::core::wrap_tensor(ggml_cont(ctx.ggml, value.tensor), value.shape, GGML_TYPE_F32);
}

mod::TransposeConfig swap_channel_time() {
    return mod::TransposeConfig{{0, 2, 1, 3}, 3};
}

// Zero-pad the time axis of a [B, C, T] tensor.
engine::core::TensorValue pad_time(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & value,
    int left,
    int right) {
    if (left == 0 && right == 0) {
        return value;
    }
    auto contiguous = engine::core::ensure_backend_addressable_layout(ctx, value);
    auto shape = contiguous.shape;
    shape.dims[2] += left + right;
    return engine::core::wrap_tensor(
        ggml_pad_ext(ctx.ggml, contiguous.tensor, left, right, 0, 0, 0, 0, 0, 0),
        shape,
        GGML_TYPE_F32);
}

// mish(x) = x * tanh(softplus(x))
engine::core::TensorValue mish(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & value) {
    auto * gate = ggml_tanh(ctx.ggml, ggml_softplus(ctx.ggml, value.tensor));
    return engine::core::wrap_tensor(
        ggml_mul(ctx.ggml, value.tensor, gate), value.shape, GGML_TYPE_F32);
}

// x * (1 + scale) + shift, with scale/shift broadcast over the time axis.
engine::core::TensorValue modulate(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & x,
    const engine::core::TensorValue & scale,
    const engine::core::TensorValue & shift) {
    auto scaled = mod::MulModule{}.build(ctx, x, mod::RepeatModule({x.shape}).build(ctx, scale));
    auto sum = mod::AddModule{}.build(ctx, x, scaled);
    return mod::AddModule{}.build(ctx, sum, mod::RepeatModule({x.shape}).build(ctx, shift));
}

SoproGroupedConvWeights load_grouped_conv(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    engine::assets::TensorStorageType storage_type,
    int64_t channels,
    int64_t kernel_size,
    int64_t groups) {
    if (channels % groups != 0) {
        throw std::runtime_error(prefix + ": channel count is not divisible by the group count");
    }
    SoproGroupedConvWeights out;
    out.group_in_channels = channels / groups;
    out.group_out_channels = channels / groups;
    out.kernel_size = kernel_size;
    const auto weight = source.require_f32(
        prefix + ".weight", {channels, out.group_in_channels, kernel_size});
    const auto bias = source.require_f32(prefix + ".bias", {channels});
    const auto group_weight_elements =
        static_cast<size_t>(out.group_out_channels * out.group_in_channels * kernel_size);
    out.groups.reserve(static_cast<size_t>(groups));
    for (int64_t group = 0; group < groups; ++group) {
        const size_t weight_offset = static_cast<size_t>(group) * group_weight_elements;
        std::vector<float> group_weight(
            weight.begin() + static_cast<ptrdiff_t>(weight_offset),
            weight.begin() + static_cast<ptrdiff_t>(weight_offset + group_weight_elements));
        const size_t bias_offset = static_cast<size_t>(group * out.group_out_channels);
        std::vector<float> group_bias(
            bias.begin() + static_cast<ptrdiff_t>(bias_offset),
            bias.begin() + static_cast<ptrdiff_t>(bias_offset + out.group_out_channels));
        engine::modules::Conv1dWeights conv;
        conv.weight = store.make_from_f32(
            engine::core::TensorShape::from_dims(
                {out.group_out_channels, out.group_in_channels, kernel_size}),
            storage_type,
            std::move(group_weight));
        conv.bias = store.make_from_f32(
            engine::core::TensorShape::from_dims({out.group_out_channels}),
            engine::assets::TensorStorageType::F32,
            std::move(group_bias));
        out.groups.push_back(std::move(conv));
    }
    return out;
}

engine::core::TensorValue build_grouped_conv(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    const SoproGroupedConvWeights & weights) {
    engine::core::TensorValue result;
    for (size_t group = 0; group < weights.groups.size(); ++group) {
        auto slice = mod::SliceModule({
            1,
            static_cast<int64_t>(group) * weights.group_in_channels,
            weights.group_in_channels,
        }).build(ctx, input);
        auto convolved = mod::Conv1dModule({
            weights.group_in_channels, weights.group_out_channels, weights.kernel_size,
            1, 0, 1, true,
        }).build(ctx, dense(ctx, slice), weights.groups[group]);
        result = group == 0 ? convolved : mod::ConcatModule({1}).build(ctx, result, convolved);
    }
    return result;
}

std::shared_ptr<const SoproAcousticWeights> load_acoustic_weights(
    ggml_backend_t backend,
    engine::core::BackendType backend_type,
    const engine::assets::TensorSource & source,
    const SoproModelConfig & config,
    size_t weight_context_bytes,
    engine::assets::TensorStorageType matmul_storage_type,
    engine::assets::TensorStorageType conv_storage_type) {
    auto weights = std::make_shared<SoproAcousticWeights>();
    weights->store = std::make_shared<engine::core::BackendWeightStore>(
        backend, backend_type, "sopro_tts.acoustic.weights", weight_context_bytes);
    auto & store = *weights->store;
    const std::string root = "acoustic_head.";
    const int64_t latent = config.latent_dim;
    const int64_t dim = config.acoustic_dit_dim;
    const int64_t mel = config.acoustic_mel_n_mels;

    weights->semantic_token_emb = store.load_f32_tensor(
        source, root + "semantic_token_emb.weight", {config.semantic_vocab_size, latent});
    weights->prelook_conv1 = binding::conv1d_from_source(
        store, source, root + "semantic_prelook.conv1", conv_storage_type,
        latent, latent, config.acoustic_pre_lookahead_frames + 1, true);
    weights->prelook_conv2 = binding::conv1d_from_source(
        store, source, root + "semantic_prelook.conv2", conv_storage_type, latent, latent, 3, true);
    // LearnedCausalUpsampler hidden width is max(8, channels).
    const int64_t upsampler_hidden = std::max<int64_t>(8, latent);
    weights->upsampler_in = binding::conv1d_from_source(
        store, source, root + "semantic_upsampler.in_proj", conv_storage_type,
        upsampler_hidden, latent, 1, true);
    weights->upsampler_mix = binding::conv1d_from_source(
        store, source, root + "semantic_upsampler.mix.conv", conv_storage_type,
        upsampler_hidden, upsampler_hidden, config.acoustic_upsampler_kernel_size, true);
    weights->upsampler_out = binding::conv1d_from_source(
        store, source, root + "semantic_upsampler.out_proj", conv_storage_type,
        latent, upsampler_hidden, 1, true);
    weights->mu_proj = binding::conv1d_from_source(
        store, source, root + "mu_proj", conv_storage_type, config.acoustic_mu_dim, latent, 1, true);

    const int64_t proj_in = mel * 2 + config.acoustic_mu_dim + config.acoustic_spk_dim;
    weights->input_proj = binding::linear_from_source(
        store, source, root + "input_embed.proj", matmul_storage_type, dim, proj_in, true);
    weights->cond_mask_proj = binding::linear_from_source(
        store, source, root + "input_embed.cond_mask_proj", matmul_storage_type, dim, 1, false);
    weights->pos_conv1 = load_grouped_conv(
        store, source, root + "input_embed.pos.conv1", conv_storage_type,
        dim, config.acoustic_pos_kernel_size, 16);
    weights->pos_conv2 = load_grouped_conv(
        store, source, root + "input_embed.pos.conv2", conv_storage_type,
        dim, config.acoustic_pos_kernel_size, 16);

    const int64_t inner = config.acoustic_dit_heads * config.acoustic_dit_dim_head;
    const int64_t ff_dim = config.acoustic_dit_ff_dim();
    weights->blocks.reserve(static_cast<size_t>(config.acoustic_dit_depth));
    for (int64_t index = 0; index < config.acoustic_dit_depth; ++index) {
        const std::string prefix = root + "blocks." + std::to_string(index);
        SoproDiTBlockWeights block;
        block.modulation = binding::linear_from_source(
            store, source, prefix + ".attn_norm.mod.1", matmul_storage_type, dim * 6, dim, true);
        block.to_q = binding::linear_from_source(
            store, source, prefix + ".attn.to_q", matmul_storage_type, inner, dim, true);
        block.to_k = binding::linear_from_source(
            store, source, prefix + ".attn.to_k", matmul_storage_type, inner, dim, true);
        block.to_v = binding::linear_from_source(
            store, source, prefix + ".attn.to_v", matmul_storage_type, inner, dim, true);
        block.to_out = binding::linear_from_source(
            store, source, prefix + ".attn.to_out.0", matmul_storage_type, dim, inner, true);
        block.ff_in = binding::linear_from_source(
            store, source, prefix + ".ff.0", matmul_storage_type, ff_dim, dim, true);
        block.ff_out = binding::linear_from_source(
            store, source, prefix + ".ff.3", matmul_storage_type, dim, ff_dim, true);
        weights->blocks.push_back(std::move(block));
    }
    weights->out_modulation = binding::linear_from_source(
        store, source, root + "out_norm.mod.1", matmul_storage_type, dim * 2, dim, true);
    weights->out_proj = binding::linear_from_source(
        store, source, root + "out_proj", matmul_storage_type, mel, dim, true);

    weights->time_mlp_w0 = source.require_f32(
        root + "time_mlp.0.weight", {dim, config.acoustic_time_embed_dim});
    weights->time_mlp_b0 = source.require_f32(root + "time_mlp.0.bias", {dim});
    weights->time_mlp_w2 = source.require_f32(root + "time_mlp.2.weight", {dim, dim});
    weights->time_mlp_b2 = source.require_f32(root + "time_mlp.2.bias", {dim});
    weights->spk_proj_w = source.require_f32(
        root + "spk_proj.weight", {config.acoustic_spk_dim, config.cond_hidden_dim});
    weights->spk_proj_b = source.require_f32(root + "spk_proj.bias", {config.acoustic_spk_dim});

    store.upload();
    return weights;
}

std::vector<float> affine(
    const std::vector<float> & weight,
    const std::vector<float> & bias,
    const std::vector<float> & input,
    int64_t in_dim,
    int64_t out_dim) {
    std::vector<float> out(static_cast<size_t>(out_dim), 0.0F);
    for (int64_t o = 0; o < out_dim; ++o) {
        const float * row = weight.data() + static_cast<size_t>(o * in_dim);
        double sum = bias[static_cast<size_t>(o)];
        for (int64_t i = 0; i < in_dim; ++i) {
            sum += static_cast<double>(row[i]) * static_cast<double>(input[static_cast<size_t>(i)]);
        }
        out[static_cast<size_t>(o)] = static_cast<float>(sum);
    }
    return out;
}

}  // namespace

std::vector<float> build_time_grid(int64_t steps, float sway_coefficient) {
    if (steps < 1) {
        throw std::runtime_error("Sopro acoustic solver requires at least one step");
    }
    std::vector<float> times(static_cast<size_t>(steps + 1), 0.0F);
    for (int64_t i = 0; i <= steps; ++i) {
        times[static_cast<size_t>(i)] = static_cast<float>(i) / static_cast<float>(steps);
    }
    if (std::fabs(sway_coefficient) <= 1.0e-8F) {
        return times;
    }
    for (auto & value : times) {
        value += sway_coefficient *
                 (std::cos(0.5F * static_cast<float>(M_PI) * value) - 1.0F + value);
    }
    return times;
}

std::vector<float> sinusoidal_time_embedding(float t, int64_t dim, float scale) {
    const int64_t half = std::max<int64_t>(1, dim / 2);
    const int64_t denominator = std::max<int64_t>(1, half - 1);
    std::vector<float> out(static_cast<size_t>(half * 2), 0.0F);
    for (int64_t i = 0; i < half; ++i) {
        const float frequency = std::exp(
            static_cast<float>(i) * (-(std::log(10000.0F) / static_cast<float>(denominator))));
        const float argument = scale * t * frequency;
        out[static_cast<size_t>(i)] = std::sin(argument);
        out[static_cast<size_t>(half + i)] = std::cos(argument);
    }
    return out;
}

// One graph produces mu (constant across solver steps); the other evaluates the
// DiT velocity field and is replayed once per Euler step.
struct SoproAcousticGraphs {
    SoproAcousticGraphs(
        ggml_backend_t backend_in,
        engine::core::BackendType backend_type,
        size_t graph_context_bytes,
        const SoproModelConfig & config_in,
        std::shared_ptr<const SoproAcousticWeights> weights_in,
        int64_t token_count,
        int64_t frame_count)
        : backend(backend_in),
          weights(std::move(weights_in)),
          tokens(token_count),
          frames(frame_count),
          config(&config_in) {
        if (backend == nullptr || weights == nullptr) {
            throw std::runtime_error("Sopro acoustic graphs require a backend and weights");
        }
        if (tokens <= 0 || frames <= 0) {
            throw std::runtime_error("Sopro acoustic graphs require positive lengths");
        }
        build_conditioning(backend_type, graph_context_bytes, config_in);
        build_velocity(backend_type, graph_context_bytes, config_in);
    }

    ~SoproAcousticGraphs() {
        if (conditioning_allocr != nullptr) {
            ggml_gallocr_free(conditioning_allocr);
        }
        if (velocity_allocr != nullptr) {
            ggml_gallocr_free(velocity_allocr);
        }
    }

    bool matches(const SoproAcousticWeights & other, int64_t token_count, int64_t frame_count) const noexcept {
        return weights.get() == &other && tokens == token_count && frames == frame_count;
    }

    void build_conditioning(
        engine::core::BackendType backend_type,
        size_t graph_context_bytes,
        const SoproModelConfig & config_in) {
        ggml_init_params params{graph_context_bytes, nullptr, true};
        conditioning_ctx.reset(ggml_init(params));
        if (conditioning_ctx == nullptr) {
            throw std::runtime_error("failed to initialize the Sopro conditioning graph context");
        }
        engine::core::ModuleBuildContext ctx{
            conditioning_ctx.get(), "sopro_tts.acoustic.conditioning", backend_type};
        const int64_t latent = config_in.latent_dim;
        const int64_t upsampler_hidden = std::max<int64_t>(8, latent);

        token_input = ggml_new_tensor_1d(ctx.ggml, GGML_TYPE_I32, tokens);
        ggml_set_input(token_input);
        expand_index = ggml_new_tensor_1d(ctx.ggml, GGML_TYPE_I32, frames);
        ggml_set_input(expand_index);

        // semantic_latents: [1, latent, tokens]
        auto * rows = ggml_get_rows(ctx.ggml, weights->semantic_token_emb.tensor, token_input);
        auto latents = engine::core::wrap_tensor(
            rows, engine::core::TensorShape::from_dims({1, tokens, latent}), GGML_TYPE_F32);
        latents = dense(ctx, mod::TransposeModule(swap_channel_time()).build(ctx, latents));

        // PreLookahead: right-pad the lookahead, conv, then a causal 3-tap conv.
        {
            const auto lookahead = static_cast<int>(config_in.acoustic_pre_lookahead_frames);
            auto y = pad_time(ctx, latents, 0, lookahead);
            y = mod::Conv1dModule({latent, latent, lookahead + 1, 1, 0, 1, true})
                    .build(ctx, y, weights->prelook_conv1);
            y = mod::LeakyReluModule({0.1F}).build(ctx, y);
            y = pad_time(ctx, y, 2, 0);
            y = mod::Conv1dModule({latent, latent, 3, 1, 0, 1, true})
                    .build(ctx, y, weights->prelook_conv2);
            latents = mod::AddModule{}.build(ctx, latents, y);
        }

        // LearnedCausalUpsampler: nearest-index expansion onto the mel grid,
        // then a residual causal mixer.
        engine::core::TensorValue expanded;
        {
            auto btc = dense(ctx, mod::TransposeModule(swap_channel_time()).build(ctx, latents));
            auto * gathered = ggml_get_rows(
                ctx.ggml,
                ggml_reshape_2d(ctx.ggml, btc.tensor, latent, tokens),
                expand_index);
            expanded = dense(ctx, mod::TransposeModule(swap_channel_time()).build(
                ctx,
                engine::core::wrap_tensor(
                    gathered, engine::core::TensorShape::from_dims({1, frames, latent}),
                    GGML_TYPE_F32)));
        }
        const auto mix_kernel = static_cast<int>(config_in.acoustic_upsampler_kernel_size);
        auto hidden = mod::Conv1dModule({latent, upsampler_hidden, 1, 1, 0, 1, true})
                          .build(ctx, expanded, weights->upsampler_in);
        hidden = mod::SiluModule{}.build(ctx, hidden);
        hidden = pad_time(ctx, hidden, mix_kernel - 1, 0);
        hidden = mod::Conv1dModule({upsampler_hidden, upsampler_hidden, mix_kernel, 1, 0, 1, true})
                     .build(ctx, hidden, weights->upsampler_mix);
        hidden = mod::SiluModule{}.build(ctx, hidden);
        hidden = mod::Conv1dModule({upsampler_hidden, latent, 1, 1, 0, 1, true})
                     .build(ctx, hidden, weights->upsampler_out);
        hidden = mod::AddModule{}.build(ctx, expanded, hidden);
        hidden = mod::Conv1dModule({latent, config_in.acoustic_mu_dim, 1, 1, 0, 1, true})
                     .build(ctx, hidden, weights->mu_proj);
        hidden = engine::core::ensure_backend_addressable_layout(ctx, hidden);
        mu_output = hidden.tensor;
        ggml_set_output(mu_output);
        conditioning_graph = ggml_new_graph_custom(conditioning_ctx.get(), 65536, false);
        ggml_build_forward_expand(conditioning_graph, mu_output);
        conditioning_allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (conditioning_allocr == nullptr ||
            !ggml_gallocr_reserve(conditioning_allocr, conditioning_graph) ||
            !ggml_gallocr_alloc_graph(conditioning_allocr, conditioning_graph)) {
            throw std::runtime_error("failed to allocate the Sopro conditioning graph");
        }
    }

    void build_velocity(
        engine::core::BackendType backend_type,
        size_t graph_context_bytes,
        const SoproModelConfig & config_in) {
        ggml_init_params params{graph_context_bytes, nullptr, true};
        velocity_ctx.reset(ggml_init(params));
        if (velocity_ctx == nullptr) {
            throw std::runtime_error("failed to initialize the Sopro velocity graph context");
        }
        engine::core::ModuleBuildContext ctx{
            velocity_ctx.get(), "sopro_tts.acoustic.velocity", backend_type};
        const int64_t mel = config_in.acoustic_mel_n_mels;
        const int64_t mu_dim = config_in.acoustic_mu_dim;
        const int64_t spk_dim = config_in.acoustic_spk_dim;
        const int64_t dim = config_in.acoustic_dit_dim;
        const int64_t heads = config_in.acoustic_dit_heads;
        const int64_t head_dim = config_in.acoustic_dit_dim_head;
        const int64_t inner = heads * head_dim;
        const int64_t ff_dim = config_in.acoustic_dit_ff_dim();

        const auto mel_shape = engine::core::TensorShape::from_dims({1, mel, frames});
        const auto mu_shape = engine::core::TensorShape::from_dims({1, mu_dim, frames});
        const auto mask_shape = engine::core::TensorShape::from_dims({1, 1, frames});
        x_input = engine::core::make_tensor(ctx, GGML_TYPE_F32, mel_shape).tensor;
        cond_mel_input = engine::core::make_tensor(ctx, GGML_TYPE_F32, mel_shape).tensor;
        cond_mask_input = engine::core::make_tensor(ctx, GGML_TYPE_F32, mask_shape).tensor;
        mu_input = engine::core::make_tensor(ctx, GGML_TYPE_F32, mu_shape).tensor;
        spk_input = engine::core::make_tensor(
            ctx, GGML_TYPE_F32, engine::core::TensorShape::from_dims({1, 1, spk_dim})).tensor;
        emb_input = engine::core::make_tensor(
            ctx, GGML_TYPE_F32, engine::core::TensorShape::from_dims({1, 1, dim})).tensor;
        // The solver replays this graph once per Euler step and only re-uploads
        // x_t and the time embedding, so every leaf must survive the allocator.
        for (ggml_tensor * leaf :
             {x_input, cond_mel_input, cond_mask_input, mu_input, spk_input, emb_input}) {
            ggml_set_input(leaf);
        }
        positions = ggml_new_tensor_1d(ctx.ggml, GGML_TYPE_I32, frames);
        ggml_set_input(positions);

        auto to_btc = [&](ggml_tensor * tensor, int64_t channels) {
            auto value = engine::core::wrap_tensor(
                tensor, engine::core::TensorShape::from_dims({1, channels, frames}), GGML_TYPE_F32);
            return dense(ctx, mod::TransposeModule(swap_channel_time()).build(ctx, value));
        };
        auto emb = engine::core::wrap_tensor(
            emb_input, engine::core::TensorShape::from_dims({1, 1, dim}), GGML_TYPE_F32);

        // InputEmbedding.proj over [x_t | cond_mel | mu | spk].
        auto features = mod::ConcatModule({2}).build(ctx, to_btc(x_input, mel), to_btc(cond_mel_input, mel));
        features = mod::ConcatModule({2}).build(ctx, features, to_btc(mu_input, mu_dim));
        {
            auto spk = engine::core::wrap_tensor(
                spk_input, engine::core::TensorShape::from_dims({1, 1, spk_dim}), GGML_TYPE_F32);
            auto broadcast = mod::RepeatModule({
                engine::core::TensorShape::from_dims({1, frames, spk_dim})}).build(ctx, spk);
            features = mod::ConcatModule({2}).build(ctx, features, broadcast);
        }
        const int64_t proj_in = mel * 2 + mu_dim + spk_dim;
        auto hidden = mod::LinearModule({proj_in, dim, true, GGML_PREC_F32})
                          .build(ctx, dense(ctx, features), weights->input_proj);
        hidden = mod::AddModule{}.build(
            ctx, hidden,
            mod::LinearModule({1, dim, false, GGML_PREC_F32})
                .build(ctx, to_btc(cond_mask_input, 1), weights->cond_mask_proj));
        {
            // CausalConvPositionEmbedding: two causal grouped convolutions.
            const auto kernel = static_cast<int>(config_in.acoustic_pos_kernel_size);
            auto y = dense(ctx, mod::TransposeModule(swap_channel_time()).build(ctx, hidden));
            y = build_grouped_conv(ctx, pad_time(ctx, y, kernel - 1, 0), weights->pos_conv1);
            y = mish(ctx, y);
            y = build_grouped_conv(ctx, pad_time(ctx, y, kernel - 1, 0), weights->pos_conv2);
            y = mish(ctx, y);
            y = dense(ctx, mod::TransposeModule(swap_channel_time()).build(ctx, y));
            hidden = mod::AddModule{}.build(ctx, hidden, y);
        }

        auto silu_emb = mod::SiluModule{}.build(ctx, emb);
        for (const auto & block : weights->blocks) {
            auto modulation = mod::LinearModule({dim, dim * 6, true, GGML_PREC_F32})
                                  .build(ctx, silu_emb, block.modulation);
            auto chunk = [&](int64_t index) {
                return mod::SliceModule({2, index * dim, dim}).build(ctx, modulation);
            };
            const auto shift_msa = chunk(0);
            const auto scale_msa = chunk(1);
            const auto gate_msa = chunk(2);
            const auto shift_mlp = chunk(3);
            const auto scale_mlp = chunk(4);
            const auto gate_mlp = chunk(5);

            auto norm = modulate(
                ctx,
                mod::LayerNormModule({dim, kDiTLayerNormEps, false, false})
                    .build(ctx, hidden, mod::NormWeights{}),
                scale_msa, shift_msa);
            auto q = mod::LinearModule({dim, inner, true, GGML_PREC_F32}).build(ctx, norm, block.to_q);
            auto k = mod::LinearModule({dim, inner, true, GGML_PREC_F32}).build(ctx, norm, block.to_k);
            auto v = mod::LinearModule({dim, inner, true, GGML_PREC_F32}).build(ctx, norm, block.to_v);
            const auto head_shape = engine::core::TensorShape::from_dims({1, frames, heads, head_dim});
            auto reshape_heads = [&](const engine::core::TensorValue & value) {
                return dense(ctx, engine::core::reshape_tensor(
                    ctx, engine::core::ensure_backend_addressable_layout(ctx, value), head_shape));
            };
            // Half-rotation RoPE over the frame index, applied per head.
            auto rope = [&](engine::core::TensorValue value) {
                return mod::RoPEModule({head_dim, GGML_ROPE_TYPE_NEOX, 10000.0F})
                    .build(ctx, value, engine::core::wrap_tensor(
                        positions, engine::core::TensorShape::from_dims({frames}), GGML_TYPE_I32));
            };
            auto to_flash = [&](const engine::core::TensorValue & value) {
                return dense(ctx, mod::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, value));
            };
            auto q_heads = to_flash(rope(reshape_heads(q)));
            auto k_heads = to_flash(rope(reshape_heads(k)));
            auto v_heads = to_flash(reshape_heads(v));
            auto attention = mod::ScaledDotProductAttentionModule({
                head_dim,
                mod::ScaledDotProductAttentionLowering::Flash,
                GGML_PREC_F32,
                mod::AttentionCausality::NonCausal,
            }).build(ctx, q_heads, k_heads, v_heads);
            auto flat = engine::core::reshape_tensor(
                ctx, engine::core::ensure_backend_addressable_layout(ctx, attention),
                engine::core::TensorShape::from_dims({1, frames, inner}));
            auto projected = mod::LinearModule({inner, dim, true, GGML_PREC_F32})
                                 .build(ctx, flat, block.to_out);
            hidden = mod::AddModule{}.build(
                ctx, hidden,
                mod::MulModule{}.build(
                    ctx, projected, mod::RepeatModule({projected.shape}).build(ctx, gate_msa)));

            auto feed = modulate(
                ctx,
                mod::LayerNormModule({dim, kDiTLayerNormEps, false, false})
                    .build(ctx, hidden, mod::NormWeights{}),
                scale_mlp, shift_mlp);
            feed = mod::LinearModule({dim, ff_dim, true, GGML_PREC_F32}).build(ctx, feed, block.ff_in);
            feed = mod::GeluModule({mod::GeluApproximation::Tanh}).build(ctx, feed);
            feed = mod::LinearModule({ff_dim, dim, true, GGML_PREC_F32}).build(ctx, feed, block.ff_out);
            hidden = mod::AddModule{}.build(
                ctx, hidden,
                mod::MulModule{}.build(
                    ctx, feed, mod::RepeatModule({feed.shape}).build(ctx, gate_mlp)));
        }

        {
            // AdaLayerNormFinal emits (scale, shift) in that order.
            auto modulation = mod::LinearModule({dim, dim * 2, true, GGML_PREC_F32})
                                  .build(ctx, silu_emb, weights->out_modulation);
            const auto scale = mod::SliceModule({2, 0, dim}).build(ctx, modulation);
            const auto shift = mod::SliceModule({2, dim, dim}).build(ctx, modulation);
            hidden = modulate(
                ctx,
                mod::LayerNormModule({dim, kDiTLayerNormEps, false, false})
                    .build(ctx, hidden, mod::NormWeights{}),
                scale, shift);
        }
        hidden = mod::LinearModule({dim, mel, true, GGML_PREC_F32}).build(ctx, hidden, weights->out_proj);
        // Back to [1, mel, frames] so the Euler update sees the solver layout.
        hidden = dense(ctx, mod::TransposeModule(swap_channel_time()).build(ctx, hidden));
        velocity_output = hidden.tensor;
        ggml_set_output(velocity_output);
        velocity_graph = ggml_new_graph_custom(velocity_ctx.get(), 262144, false);
        ggml_build_forward_expand(velocity_graph, velocity_output);
        velocity_allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (velocity_allocr == nullptr ||
            !ggml_gallocr_reserve(velocity_allocr, velocity_graph) ||
            !ggml_gallocr_alloc_graph(velocity_allocr, velocity_graph)) {
            throw std::runtime_error("failed to allocate the Sopro velocity graph");
        }
    }

    std::vector<float> run_conditioning(const std::vector<int32_t> & token_ids) {
        std::vector<int32_t> index(static_cast<size_t>(frames), 0);
        for (int64_t frame = 0; frame < frames; ++frame) {
            index[static_cast<size_t>(frame)] = static_cast<int32_t>(
                std::min<int64_t>(frame * tokens / frames, tokens - 1));
        }
        ggml_backend_tensor_set(token_input, token_ids.data(), 0, token_ids.size() * sizeof(int32_t));
        ggml_backend_tensor_set(expand_index, index.data(), 0, index.size() * sizeof(int32_t));
        const ggml_status status = engine::core::compute_backend_graph(backend, conditioning_graph);
        ggml_backend_synchronize(backend);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Sopro acoustic conditioning graph compute failed");
        }
        std::vector<float> mu(static_cast<size_t>(config->acoustic_mu_dim * frames), 0.0F);
        ggml_backend_tensor_get(mu_output, mu.data(), 0, mu.size() * sizeof(float));
        return mu;
    }

    // ggml_gallocr only exempts GGML_TENSOR_FLAG_OUTPUT tensors from being
    // freed and reused (ggml-alloc.c, ggml_gallocr_free_node); an input leaf's
    // arena space is handed to a later intermediate once its last consumer has
    // run. That is fine for a one-shot graph, but the solver replays this one
    // per Euler step, so every leaf has to be re-uploaded before each compute
    // rather than staged once.
    void set_constants(
        std::vector<float> cond_mel,
        std::vector<float> cond_mask,
        std::vector<float> mu,
        std::vector<float> spk) {
        cond_mel_host = std::move(cond_mel);
        cond_mask_host = std::move(cond_mask);
        mu_host = std::move(mu);
        spk_host = std::move(spk);
        position_host.assign(static_cast<size_t>(frames), 0);
        for (int64_t frame = 0; frame < frames; ++frame) {
            position_host[static_cast<size_t>(frame)] = static_cast<int32_t>(frame);
        }
    }

    void upload_constants() {
        ggml_backend_tensor_set(positions, position_host.data(), 0,
                                position_host.size() * sizeof(int32_t));
        ggml_backend_tensor_set(cond_mel_input, cond_mel_host.data(), 0,
                                cond_mel_host.size() * sizeof(float));
        ggml_backend_tensor_set(cond_mask_input, cond_mask_host.data(), 0,
                                cond_mask_host.size() * sizeof(float));
        ggml_backend_tensor_set(mu_input, mu_host.data(), 0, mu_host.size() * sizeof(float));
        ggml_backend_tensor_set(spk_input, spk_host.data(), 0, spk_host.size() * sizeof(float));
    }

    std::vector<float> run_velocity(const std::vector<float> & x, const std::vector<float> & emb) {
        upload_constants();
        ggml_backend_tensor_set(x_input, x.data(), 0, x.size() * sizeof(float));
        ggml_backend_tensor_set(emb_input, emb.data(), 0, emb.size() * sizeof(float));
        const ggml_status status = engine::core::compute_backend_graph(backend, velocity_graph);
        ggml_backend_synchronize(backend);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Sopro acoustic velocity graph compute failed");
        }
        std::vector<float> velocity(x.size(), 0.0F);
        ggml_backend_tensor_get(velocity_output, velocity.data(), 0, velocity.size() * sizeof(float));
        return velocity;
    }

    ggml_backend_t backend = nullptr;
    std::shared_ptr<const SoproAcousticWeights> weights;
    int64_t tokens = 0;
    int64_t frames = 0;
    const SoproModelConfig * config = nullptr;

    std::unique_ptr<ggml_context, GgmlContextDeleter> conditioning_ctx;
    ggml_tensor * token_input = nullptr;
    ggml_tensor * expand_index = nullptr;
    ggml_tensor * mu_output = nullptr;
    ggml_cgraph * conditioning_graph = nullptr;
    ggml_gallocr_t conditioning_allocr = nullptr;

    std::unique_ptr<ggml_context, GgmlContextDeleter> velocity_ctx;
    ggml_tensor * x_input = nullptr;
    ggml_tensor * cond_mel_input = nullptr;
    ggml_tensor * cond_mask_input = nullptr;
    ggml_tensor * mu_input = nullptr;
    ggml_tensor * spk_input = nullptr;
    ggml_tensor * emb_input = nullptr;
    ggml_tensor * positions = nullptr;
    ggml_tensor * velocity_output = nullptr;
    std::vector<float> cond_mel_host;
    std::vector<float> cond_mask_host;
    std::vector<float> mu_host;
    std::vector<float> spk_host;
    std::vector<int32_t> position_host;
    ggml_cgraph * velocity_graph = nullptr;
    ggml_gallocr_t velocity_allocr = nullptr;
};

SoproAcousticRuntime::SoproAcousticRuntime(
    const SoproTTSAssets & assets,
    engine::core::ExecutionContext & execution_context,
    size_t weight_context_bytes,
    size_t graph_context_bytes,
    engine::assets::TensorStorageType matmul_storage_type,
    engine::assets::TensorStorageType conv_storage_type)
    : config_(assets.config.model),
      execution_context_(execution_context),
      graph_context_bytes_(graph_context_bytes),
      weights_(load_acoustic_weights(
          execution_context.backend(),
          execution_context.backend_type(),
          *assets.model_weights,
          assets.config.model,
          weight_context_bytes,
          matmul_storage_type,
          conv_storage_type)) {}

SoproAcousticRuntime::~SoproAcousticRuntime() = default;

std::vector<float> SoproAcousticRuntime::solve(const SoproAcousticRequest & request) const {
    const int64_t mel = config_.acoustic_mel_n_mels;
    const int64_t frames = request.total_frames;
    const auto tokens = static_cast<int64_t>(request.semantic_tokens.size());
    if (frames <= 0 || tokens <= 0) {
        throw std::runtime_error("Sopro acoustic solve requires tokens and frames");
    }
    if (request.prompt_frames < 0 || request.prompt_frames > frames) {
        throw std::runtime_error("Sopro acoustic prompt frame count is out of range");
    }
    if (static_cast<int64_t>(request.prompt_mel.size()) != mel * request.prompt_frames) {
        throw std::runtime_error("Sopro acoustic prompt mel shape mismatch");
    }
    if (static_cast<int64_t>(request.cond_vec.size()) != config_.cond_hidden_dim) {
        throw std::runtime_error("Sopro acoustic conditioning vector shape mismatch");
    }
    const int64_t steps = std::max<int64_t>(1, request.steps);

    if (graphs_ == nullptr || !graphs_->matches(*weights_, tokens, frames)) {
        graphs_ = std::make_unique<SoproAcousticGraphs>(
            execution_context_.backend(),
            execution_context_.backend_type(),
            graph_context_bytes_,
            config_,
            weights_,
            tokens,
            frames);
    }

    // _row_has_signal: an all-zero conditioning vector disables the speaker
    // branch entirely (the reference uses it for unconditional batches).
    float cond_peak = 0.0F;
    for (const float value : request.cond_vec) {
        cond_peak = std::max(cond_peak, std::fabs(value));
    }
    std::vector<float> spk(static_cast<size_t>(config_.acoustic_spk_dim), 0.0F);
    if (cond_peak > 0.0F) {
        double norm = 0.0;
        for (const float value : request.cond_vec) {
            norm += static_cast<double>(value) * static_cast<double>(value);
        }
        // F.normalize uses an epsilon floor rather than a plain division.
        const auto inv = static_cast<float>(1.0 / std::max(std::sqrt(norm), 1.0e-12));
        std::vector<float> normalised(request.cond_vec);
        for (auto & value : normalised) {
            value *= inv;
        }
        spk = affine(
            weights_->spk_proj_w, weights_->spk_proj_b, normalised,
            config_.cond_hidden_dim, config_.acoustic_spk_dim);
    }

    auto mu = graphs_->run_conditioning(request.semantic_tokens);
    dump("mu", mu);
    dump("spk", spk);

    std::vector<float> cond_mel(static_cast<size_t>(mel * frames), 0.0F);
    std::vector<float> cond_mask(static_cast<size_t>(frames), 0.0F);
    for (int64_t c = 0; c < mel; ++c) {
        std::copy(
            request.prompt_mel.begin() + static_cast<ptrdiff_t>(c * request.prompt_frames),
            request.prompt_mel.begin() + static_cast<ptrdiff_t>((c + 1) * request.prompt_frames),
            cond_mel.begin() + static_cast<ptrdiff_t>(c * frames));
    }
    std::fill(cond_mask.begin(), cond_mask.begin() + static_cast<ptrdiff_t>(request.prompt_frames), 1.0F);
    graphs_->set_constants(cond_mel, cond_mask, mu, spk);

    std::mt19937_64 rng(request.seed);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::vector<float> x_init(static_cast<size_t>(mel * frames), 0.0F);
    for (auto & value : x_init) {
        value = normal(rng);
    }
    std::vector<float> x(x_init);
    dump("x_init", x_init);
    dump("cond_mel", cond_mel);

    const auto grid = build_time_grid(steps, config_.acoustic_sway_sampling_coef);
    const float sigma_min = config_.acoustic_sigma_min;
    for (int64_t step = 0; step < steps; ++step) {
        const float t0 = grid[static_cast<size_t>(step)];
        const float t1 = grid[static_cast<size_t>(step + 1)];
        const auto raw = sinusoidal_time_embedding(t0, config_.acoustic_time_embed_dim);
        auto emb = affine(
            weights_->time_mlp_w0, weights_->time_mlp_b0, raw,
            config_.acoustic_time_embed_dim, config_.acoustic_dit_dim);
        for (auto & value : emb) {
            value = value / (1.0F + std::exp(-value));  // SiLU
        }
        emb = affine(
            weights_->time_mlp_w2, weights_->time_mlp_b2, emb,
            config_.acoustic_dit_dim, config_.acoustic_dit_dim);

        const std::string tag = std::to_string(step);
        if (step == 0) {
            dump("emb0", emb);
            dump("x_step0", x);
        }
        dump(("traj_x_" + tag).c_str(), x);
        const auto velocity = graphs_->run_velocity(x, emb);
        if (step == 0) {
            dump("velocity0", velocity);
        }
        dump(("traj_v_" + tag).c_str(), velocity);
        const float dt = t1 - t0;
        for (size_t i = 0; i < x.size(); ++i) {
            x[i] += dt * velocity[i];
        }
        // Re-pin the prompt span to the reference mel's flow at t1.
        const float prompt_scale = 1.0F - (1.0F - sigma_min) * t1;
        for (int64_t c = 0; c < mel; ++c) {
            const size_t base = static_cast<size_t>(c * frames);
            for (int64_t t = 0; t < request.prompt_frames; ++t) {
                const size_t index = base + static_cast<size_t>(t);
                x[index] = prompt_scale * x_init[index] + t1 * cond_mel[index];
            }
        }
    }
    for (int64_t c = 0; c < mel; ++c) {
        const size_t base = static_cast<size_t>(c * frames);
        for (int64_t t = 0; t < request.prompt_frames; ++t) {
            x[base + static_cast<size_t>(t)] = cond_mel[base + static_cast<size_t>(t)];
        }
    }
    dump("solved", x);
    return x;
}

}  // namespace engine::community_models::sopro_tts
