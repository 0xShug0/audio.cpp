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
#include <memory>
#include <optional>
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

// MSVC does not define M_PI; use our own constant, as f5_tts does for the
// same sway-time grid.
constexpr float kPi = 3.14159265358979323846F;

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
                 (std::cos(0.5F * kPi * value) - 1.0F + value);
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

// semantic tokens -> mu (PreLookahead, LearnedCausalUpsampler, mu_proj). It is
// constant across solver steps, so it runs once per solve or streaming render.
struct SoproConditioningGraph {
    SoproConditioningGraph(
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
        ggml_init_params params{graph_context_bytes, nullptr, true};
        ctx_holder.reset(ggml_init(params));
        if (ctx_holder == nullptr) {
            throw std::runtime_error("failed to initialize the Sopro conditioning graph context");
        }
        engine::core::ModuleBuildContext ctx{
            ctx_holder.get(), "sopro_tts.acoustic.conditioning", backend_type};
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
        graph = ggml_new_graph_custom(ctx_holder.get(), 65536, false);
        ggml_build_forward_expand(graph, mu_output);
        allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (allocr == nullptr || !ggml_gallocr_reserve(allocr, graph) ||
            !ggml_gallocr_alloc_graph(allocr, graph)) {
            throw std::runtime_error("failed to allocate the Sopro conditioning graph");
        }
    }

    ~SoproConditioningGraph() {
        if (allocr != nullptr) {
            ggml_gallocr_free(allocr);
        }
    }

    bool matches(int64_t token_count, int64_t frame_count) const noexcept {
        return tokens == token_count && frames == frame_count;
    }

    // Returns mu as [mu_dim, frames], channel-major.
    std::vector<float> run(const int32_t * token_ids) {
        std::vector<int32_t> index(static_cast<size_t>(frames), 0);
        for (int64_t frame = 0; frame < frames; ++frame) {
            index[static_cast<size_t>(frame)] = static_cast<int32_t>(
                std::min<int64_t>(frame * tokens / frames, tokens - 1));
        }
        ggml_backend_tensor_set(token_input, token_ids, 0, static_cast<size_t>(tokens) * sizeof(int32_t));
        ggml_backend_tensor_set(expand_index, index.data(), 0, index.size() * sizeof(int32_t));
        const ggml_status status = engine::core::compute_backend_graph(backend, graph);
        ggml_backend_synchronize(backend);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Sopro acoustic conditioning graph compute failed");
        }
        std::vector<float> mu(static_cast<size_t>(config->acoustic_mu_dim * frames), 0.0F);
        ggml_backend_tensor_get(mu_output, mu.data(), 0, mu.size() * sizeof(float));
        return mu;
    }

    ggml_backend_t backend = nullptr;
    std::shared_ptr<const SoproAcousticWeights> weights;
    int64_t tokens = 0;
    int64_t frames = 0;
    const SoproModelConfig * config = nullptr;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_holder;
    ggml_tensor * token_input = nullptr;
    ggml_tensor * expand_index = nullptr;
    ggml_tensor * mu_output = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_gallocr_t allocr = nullptr;
};

// One velocity pass. The input embedding sees `window` frames; the first
// `context` of them only feed the causal positional convs, the rest are the
// frames this pass solves. Those sit at positions starting at `cached` and
// attend to the cached frames before them through per-layer key/value inputs.
// The offline solve is one pass over the whole canvas with nothing cached.
struct SoproVelocityShape {
    int64_t window = 0;
    int64_t context = 0;
    int64_t cached = 0;
    bool masked = false;   // block-causal attention mask input
    bool emit_kv = false;  // this pass's keys and values as outputs

    int64_t frames() const noexcept { return window - context; }
    // The positional convs are zero-padded only when the window starts at frame 0.
    bool padded() const noexcept { return cached == context; }
    bool operator==(const SoproVelocityShape & other) const noexcept {
        return window == other.window && context == other.context && cached == other.cached &&
               masked == other.masked && emit_kv == other.emit_kv;
    }
};

struct SoproVelocityInputs {
    const std::vector<float> * x = nullptr;          // [n_mels, window]
    const std::vector<float> * cond_mel = nullptr;   // [n_mels, window]
    const std::vector<float> * cond_mask = nullptr;  // [window]
    const std::vector<float> * mu = nullptr;         // [mu_dim, window]
    const std::vector<float> * spk = nullptr;        // [spk_dim]
    const std::vector<float> * emb = nullptr;        // [dit_dim]
    const std::vector<ggml_fp16_t> * mask = nullptr;           // [frames, cached + frames]
    const std::vector<std::vector<float>> * keys = nullptr;    // per layer [cached, heads * head_dim]
    const std::vector<std::vector<float>> * values = nullptr;
};

struct SoproVelocityGraph {
    SoproVelocityGraph(
        ggml_backend_t backend_in,
        engine::core::BackendType backend_type,
        size_t graph_context_bytes,
        const SoproModelConfig & config_in,
        std::shared_ptr<const SoproAcousticWeights> weights_in,
        const SoproVelocityShape & shape_in)
        : backend(backend_in), weights(std::move(weights_in)), shape(shape_in), config(&config_in) {
        if (backend == nullptr || weights == nullptr) {
            throw std::runtime_error("Sopro acoustic graphs require a backend and weights");
        }
        const int64_t frames = shape.frames();
        const int64_t total = shape.cached + frames;
        if (frames <= 0 || shape.context < 0 || shape.cached < shape.context) {
            throw std::runtime_error("Sopro acoustic velocity pass has an invalid shape");
        }
        ggml_init_params params{graph_context_bytes, nullptr, true};
        ctx_holder.reset(ggml_init(params));
        if (ctx_holder == nullptr) {
            throw std::runtime_error("failed to initialize the Sopro velocity graph context");
        }
        engine::core::ModuleBuildContext ctx{ctx_holder.get(), "sopro_tts.acoustic.velocity", backend_type};
        const int64_t window = shape.window;
        const int64_t mel = config_in.acoustic_mel_n_mels;
        const int64_t mu_dim = config_in.acoustic_mu_dim;
        const int64_t spk_dim = config_in.acoustic_spk_dim;
        const int64_t dim = config_in.acoustic_dit_dim;
        const int64_t heads = config_in.acoustic_dit_heads;
        const int64_t head_dim = config_in.acoustic_dit_dim_head;
        const int64_t inner = heads * head_dim;
        const int64_t ff_dim = config_in.acoustic_dit_ff_dim();

        auto input = [&](ggml_type type, std::initializer_list<int64_t> dims) {
            auto * tensor = engine::core::make_tensor(ctx, type, engine::core::TensorShape::from_dims(dims)).tensor;
            ggml_set_input(tensor);
            return tensor;
        };
        x_input = input(GGML_TYPE_F32, {1, mel, window});
        cond_mel_input = input(GGML_TYPE_F32, {1, mel, window});
        cond_mask_input = input(GGML_TYPE_F32, {1, 1, window});
        mu_input = input(GGML_TYPE_F32, {1, mu_dim, window});
        spk_input = input(GGML_TYPE_F32, {1, 1, spk_dim});
        emb_input = input(GGML_TYPE_F32, {1, 1, dim});
        positions = input(GGML_TYPE_I32, {frames});
        if (shape.masked) {
            mask_input = input(GGML_TYPE_F16, {1, 1, frames, total});
        }
        if (shape.cached > 0) {
            for (size_t layer = 0; layer < weights->blocks.size(); ++layer) {
                key_inputs.push_back(input(GGML_TYPE_F32, {1, shape.cached, heads, head_dim}));
                value_inputs.push_back(input(GGML_TYPE_F32, {1, shape.cached, heads, head_dim}));
            }
        }

        auto to_btc = [&](ggml_tensor * tensor, int64_t channels) {
            auto value = engine::core::wrap_tensor(
                tensor, engine::core::TensorShape::from_dims({1, channels, window}), GGML_TYPE_F32);
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
                engine::core::TensorShape::from_dims({1, window, spk_dim})}).build(ctx, spk);
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
            // Past the start of the canvas the window carries their receptive
            // field instead of zero padding.
            const auto kernel = static_cast<int>(config_in.acoustic_pos_kernel_size);
            const int pad = shape.padded() ? kernel - 1 : 0;
            auto y = dense(ctx, mod::TransposeModule(swap_channel_time()).build(ctx, hidden));
            y = build_grouped_conv(ctx, pad_time(ctx, y, pad, 0), weights->pos_conv1);
            y = mish(ctx, y);
            y = build_grouped_conv(ctx, pad_time(ctx, y, pad, 0), weights->pos_conv2);
            y = mish(ctx, y);
            if (shape.padded() && shape.context > 0) {
                y = dense(ctx, mod::SliceModule({2, shape.context, frames}).build(ctx, y));
            }
            y = dense(ctx, mod::TransposeModule(swap_channel_time()).build(ctx, y));
            if (shape.context > 0) {
                hidden = dense(ctx, mod::SliceModule({1, shape.context, frames}).build(ctx, hidden));
            }
            hidden = mod::AddModule{}.build(ctx, hidden, y);
        }

        std::optional<engine::core::TensorValue> mask;
        if (mask_input != nullptr) {
            mask = engine::core::wrap_tensor(
                mask_input, engine::core::TensorShape::from_dims({1, 1, frames, total}), GGML_TYPE_F16);
        }
        auto silu_emb = mod::SiluModule{}.build(ctx, emb);
        for (size_t layer = 0; layer < weights->blocks.size(); ++layer) {
            const auto & block = weights->blocks[layer];
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
                return dense(ctx, mod::RoPEModule({head_dim, GGML_ROPE_TYPE_NEOX, 10000.0F})
                    .build(ctx, value, engine::core::wrap_tensor(
                        positions, engine::core::TensorShape::from_dims({frames}), GGML_TYPE_I32)));
            };
            auto to_flash = [&](const engine::core::TensorValue & value) {
                return dense(ctx, mod::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, value));
            };
            auto keys = rope(reshape_heads(k));
            auto values = reshape_heads(v);
            if (shape.emit_kv) {
                ggml_set_output(keys.tensor);
                ggml_set_output(values.tensor);
                key_outputs.push_back(keys.tensor);
                value_outputs.push_back(values.tensor);
            }
            if (shape.cached > 0) {
                const auto cache_shape = engine::core::TensorShape::from_dims({1, shape.cached, heads, head_dim});
                keys = mod::ConcatModule({1}).build(
                    ctx, engine::core::wrap_tensor(key_inputs[layer], cache_shape, GGML_TYPE_F32), keys);
                values = mod::ConcatModule({1}).build(
                    ctx, engine::core::wrap_tensor(value_inputs[layer], cache_shape, GGML_TYPE_F32), values);
            }
            auto attention = mod::ScaledDotProductAttentionModule({
                head_dim,
                mod::ScaledDotProductAttentionLowering::Flash,
                GGML_PREC_F32,
                mod::AttentionCausality::NonCausal,
            }).build(ctx, to_flash(rope(reshape_heads(q))), to_flash(keys), to_flash(values), mask);
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
        graph = ggml_new_graph_custom(ctx_holder.get(), 262144, false);
        ggml_build_forward_expand(graph, velocity_output);
        for (size_t layer = 0; layer < key_outputs.size(); ++layer) {
            ggml_build_forward_expand(graph, key_outputs[layer]);
            ggml_build_forward_expand(graph, value_outputs[layer]);
        }
        allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (allocr == nullptr || !ggml_gallocr_reserve(allocr, graph) ||
            !ggml_gallocr_alloc_graph(allocr, graph)) {
            throw std::runtime_error("failed to allocate the Sopro velocity graph");
        }
    }

    ~SoproVelocityGraph() {
        if (allocr != nullptr) {
            ggml_gallocr_free(allocr);
        }
    }

    // ggml_gallocr only exempts GGML_TENSOR_FLAG_OUTPUT tensors from being
    // freed and reused (ggml-alloc.c, ggml_gallocr_free_node); an input leaf's
    // arena space is handed to a later intermediate once its last consumer has
    // run. The solver replays this graph once per Euler step, so every leaf is
    // uploaded before each compute rather than staged once.
    std::vector<float> run(
        const SoproVelocityInputs & in,
        std::vector<std::vector<float>> * new_keys = nullptr,
        std::vector<std::vector<float>> * new_values = nullptr) {
        auto upload = [](ggml_tensor * tensor, const void * data, size_t bytes) {
            if (ggml_nbytes(tensor) != bytes) {
                throw std::runtime_error("Sopro acoustic velocity input has the wrong size");
            }
            ggml_backend_tensor_set(tensor, data, 0, bytes);
        };
        const int64_t frames = shape.frames();
        std::vector<int32_t> position(static_cast<size_t>(frames));
        for (int64_t i = 0; i < frames; ++i) {
            position[static_cast<size_t>(i)] = static_cast<int32_t>(shape.cached + i);
        }
        upload(x_input, in.x->data(), in.x->size() * sizeof(float));
        upload(cond_mel_input, in.cond_mel->data(), in.cond_mel->size() * sizeof(float));
        upload(cond_mask_input, in.cond_mask->data(), in.cond_mask->size() * sizeof(float));
        upload(mu_input, in.mu->data(), in.mu->size() * sizeof(float));
        upload(spk_input, in.spk->data(), in.spk->size() * sizeof(float));
        upload(emb_input, in.emb->data(), in.emb->size() * sizeof(float));
        upload(positions, position.data(), position.size() * sizeof(int32_t));
        if (mask_input != nullptr) {
            upload(mask_input, in.mask->data(), in.mask->size() * sizeof(ggml_fp16_t));
        }
        for (size_t layer = 0; layer < key_inputs.size(); ++layer) {
            const auto & keys = (*in.keys)[layer];
            const auto & values = (*in.values)[layer];
            upload(key_inputs[layer], keys.data(), keys.size() * sizeof(float));
            upload(value_inputs[layer], values.data(), values.size() * sizeof(float));
        }
        const ggml_status status = engine::core::compute_backend_graph(backend, graph);
        ggml_backend_synchronize(backend);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Sopro acoustic velocity graph compute failed");
        }
        std::vector<float> velocity(ggml_nelements(velocity_output));
        ggml_backend_tensor_get(velocity_output, velocity.data(), 0, velocity.size() * sizeof(float));
        if (new_keys != nullptr && new_values != nullptr) {
            new_keys->assign(key_outputs.size(), {});
            new_values->assign(value_outputs.size(), {});
            for (size_t layer = 0; layer < key_outputs.size(); ++layer) {
                (*new_keys)[layer].resize(ggml_nelements(key_outputs[layer]));
                (*new_values)[layer].resize(ggml_nelements(value_outputs[layer]));
                ggml_backend_tensor_get(key_outputs[layer], (*new_keys)[layer].data(), 0, ggml_nbytes(key_outputs[layer]));
                ggml_backend_tensor_get(value_outputs[layer], (*new_values)[layer].data(), 0, ggml_nbytes(value_outputs[layer]));
            }
        }
        return velocity;
    }

    ggml_backend_t backend = nullptr;
    std::shared_ptr<const SoproAcousticWeights> weights;
    SoproVelocityShape shape;
    const SoproModelConfig * config = nullptr;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_holder;
    ggml_tensor * x_input = nullptr;
    ggml_tensor * cond_mel_input = nullptr;
    ggml_tensor * cond_mask_input = nullptr;
    ggml_tensor * mu_input = nullptr;
    ggml_tensor * spk_input = nullptr;
    ggml_tensor * emb_input = nullptr;
    ggml_tensor * positions = nullptr;
    ggml_tensor * mask_input = nullptr;
    std::vector<ggml_tensor *> key_inputs;
    std::vector<ggml_tensor *> value_inputs;
    std::vector<ggml_tensor *> key_outputs;
    std::vector<ggml_tensor *> value_outputs;
    ggml_tensor * velocity_output = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_gallocr_t allocr = nullptr;
};

struct SoproAcousticGraphs {
    std::unique_ptr<SoproConditioningGraph> conditioning;
    std::unique_ptr<SoproVelocityGraph> velocity;
};

namespace {

// _speaker_embedding: spk_proj(normalize(cond_vec)); an all-zero conditioning
// vector disables the speaker branch (_row_has_signal).
std::vector<float> speaker_embedding(
    const SoproAcousticWeights & weights, const SoproModelConfig & config, const std::vector<float> & cond_vec) {
    float peak = 0.0F;
    double norm = 0.0;
    for (const float value : cond_vec) {
        peak = std::max(peak, std::fabs(value));
        norm += static_cast<double>(value) * static_cast<double>(value);
    }
    if (peak <= 0.0F) {
        return std::vector<float>(static_cast<size_t>(config.acoustic_spk_dim), 0.0F);
    }
    // F.normalize uses an epsilon floor rather than a plain division.
    const auto inv = static_cast<float>(1.0 / std::max(std::sqrt(norm), 1.0e-12));
    std::vector<float> normalised(cond_vec);
    for (auto & value : normalised) {
        value *= inv;
    }
    return affine(weights.spk_proj_w, weights.spk_proj_b, normalised, config.cond_hidden_dim, config.acoustic_spk_dim);
}

// time_mlp(sinusoidal_time_embedding(t)).
std::vector<float> time_embedding(const SoproAcousticWeights & weights, const SoproModelConfig & config, float t) {
    const auto raw = sinusoidal_time_embedding(t, config.acoustic_time_embed_dim);
    auto emb = affine(weights.time_mlp_w0, weights.time_mlp_b0, raw, config.acoustic_time_embed_dim, config.acoustic_dit_dim);
    for (auto & value : emb) {
        value = value / (1.0F + std::exp(-value));  // SiLU
    }
    return affine(weights.time_mlp_w2, weights.time_mlp_b2, emb, config.acoustic_dit_dim, config.acoustic_dit_dim);
}

}  // namespace

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
          conv_storage_type)),
      graphs_(std::make_unique<SoproAcousticGraphs>()) {}

SoproAcousticRuntime::~SoproAcousticRuntime() = default;

std::vector<float> SoproAcousticRuntime::conditioning(const int32_t * tokens, int64_t token_count, int64_t frames) const {
    auto & graph = graphs_->conditioning;
    if (graph == nullptr || !graph->matches(token_count, frames)) {
        // Free the previous arena first so the two are never resident together.
        graph.reset();
        graph = std::make_unique<SoproConditioningGraph>(
            execution_context_.backend(), execution_context_.backend_type(), graph_context_bytes_,
            config_, weights_, token_count, frames);
    }
    return graph->run(tokens);
}

SoproVelocityGraph & SoproAcousticRuntime::velocity_graph(const SoproVelocityShape & shape) const {
    auto & graph = graphs_->velocity;
    if (graph == nullptr || !(graph->shape == shape)) {
        graph.reset();
        graph = std::make_unique<SoproVelocityGraph>(
            execution_context_.backend(), execution_context_.backend_type(), graph_context_bytes_,
            config_, weights_, shape);
    }
    return *graph;
}

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

    const auto spk = speaker_embedding(*weights_, config_, request.cond_vec);
    const auto mu = conditioning(request.semantic_tokens.data(), tokens, frames);
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

    std::mt19937_64 rng(request.seed);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::vector<float> x_init(static_cast<size_t>(mel * frames), 0.0F);
    for (auto & value : x_init) {
        value = normal(rng);
    }
    std::vector<float> x(x_init);
    dump("x_init", x_init);
    dump("cond_mel", cond_mel);

    auto & graph = velocity_graph({frames, 0, 0, false, false});
    const auto grid = build_time_grid(steps, config_.acoustic_sway_sampling_coef);
    const float sigma_min = config_.acoustic_sigma_min;
    for (int64_t step = 0; step < steps; ++step) {
        const float t0 = grid[static_cast<size_t>(step)];
        const float t1 = grid[static_cast<size_t>(step + 1)];
        const auto emb = time_embedding(*weights_, config_, t0);
        SoproVelocityInputs inputs;
        inputs.x = &x;
        inputs.cond_mel = &cond_mel;
        inputs.cond_mask = &cond_mask;
        inputs.mu = &mu;
        inputs.spk = &spk;
        inputs.emb = &emb;
        const auto velocity = graph.run(inputs);
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

std::vector<float> SoproAcousticRuntime::solve_chunked(
    SoproChunkedSolveState & state,
    const SoproChunkedRequest & request,
    int64_t keep_end) const {
    const int64_t mel = config_.acoustic_mel_n_mels;
    const int64_t mu_dim = config_.acoustic_mu_dim;
    const auto & tokens = request.semantic_tokens;
    const auto token_count = static_cast<int64_t>(tokens.size());
    const auto canvas = static_cast<int64_t>(request.x0.size()) / mel;
    if (token_count <= 0 || canvas % token_count != 0 || static_cast<int64_t>(request.x0.size()) != mel * canvas) {
        throw std::runtime_error("Sopro streaming canvas must be a whole number of frames per token");
    }
    if (static_cast<int64_t>(request.prompt_mel.size()) != mel * request.prompt_frames) {
        throw std::runtime_error("Sopro acoustic prompt mel shape mismatch");
    }
    const int64_t ratio = canvas / token_count;
    const int64_t steps = std::max<int64_t>(1, request.steps);
    const int64_t cached = state.cached;
    const int64_t end = std::min(keep_end, canvas);
    const int64_t frames = end - cached;
    if (frames <= 0) {
        return {};
    }
    const auto layers = weights_->blocks.size();
    if (state.x.empty()) {
        state.x.assign(static_cast<size_t>(steps), {});
        state.keys.assign(static_cast<size_t>(steps), std::vector<std::vector<float>>(layers));
        state.values.assign(static_cast<size_t>(steps), std::vector<std::vector<float>>(layers));
    }

    // _extend_mu: mu for frames [have, end). It is recomputed over a token
    // window that starts early enough to carry the receptive field of the
    // pre-lookahead (2 tokens) and the upsampler mix (kernel - 1 frames).
    const auto have = static_cast<int64_t>(state.mu.size()) / mu_dim;
    if (end > have) {
        const int64_t mix_context = config_.acoustic_upsampler_kernel_size - 1;
        const int64_t first = std::max<int64_t>(0, (have - mix_context) / ratio - 2);
        const int64_t window_frames = ratio * (token_count - first);
        const auto mu = conditioning(tokens.data() + first, token_count - first, window_frames);
        for (int64_t f = have; f < end; ++f) {
            for (int64_t c = 0; c < mu_dim; ++c) {
                state.mu.push_back(mu[static_cast<size_t>(c * window_frames + f - ratio * first)]);
            }
        }
    }

    // Window: the new frames plus the causal positional convs' receptive field.
    const int64_t pos_context = 2 * (config_.acoustic_pos_kernel_size - 1);
    const int64_t start = std::max<int64_t>(0, cached - pos_context);
    const int64_t window = end - start;
    std::vector<float> cond_mel(static_cast<size_t>(mel * window), 0.0F);
    std::vector<float> cond_mask(static_cast<size_t>(window), 0.0F);
    std::vector<float> mu(static_cast<size_t>(mu_dim * window), 0.0F);
    for (int64_t t = 0; t < window; ++t) {
        const int64_t frame = start + t;
        if (frame < request.prompt_frames) {
            cond_mask[static_cast<size_t>(t)] = 1.0F;
            for (int64_t c = 0; c < mel; ++c) {
                cond_mel[static_cast<size_t>(c * window + t)] =
                    request.prompt_mel[static_cast<size_t>(c * request.prompt_frames + frame)];
            }
        }
        for (int64_t c = 0; c < mu_dim; ++c) {
            mu[static_cast<size_t>(c * window + t)] = state.mu[static_cast<size_t>(frame * mu_dim + c)];
        }
    }

    // build_chunk_mask: a frame sees every frame up to the end of its chunk
    // (and from num_left_chunks chunks back, when that is not -1).
    const int64_t chunk = std::max<int64_t>(1, request.chunk_frames);
    const int64_t left = config_.acoustic_num_left_chunks;
    std::vector<ggml_fp16_t> mask(static_cast<size_t>(frames * end), ggml_fp32_to_fp16(-INFINITY));
    for (int64_t q = 0; q < frames; ++q) {
        const int64_t index = (cached + q) / chunk;
        const int64_t visible_end = std::min(end, (index + 1) * chunk);
        const int64_t visible_start = left < 0 ? 0 : std::min(end, std::max<int64_t>(0, index - left) * chunk);
        for (int64_t key = visible_start; key < visible_end; ++key) {
            mask[static_cast<size_t>(q * end + key)] = ggml_fp32_to_fp16(0.0F);
        }
    }

    std::vector<float> x0(static_cast<size_t>(mel * frames));
    for (int64_t c = 0; c < mel; ++c) {
        std::copy_n(
            request.x0.begin() + static_cast<ptrdiff_t>(c * canvas + cached),
            frames,
            x0.begin() + static_cast<ptrdiff_t>(c * frames));
    }
    std::vector<float> x(x0);
    const auto spk = speaker_embedding(*weights_, config_, request.cond_vec);
    const auto grid = build_time_grid(steps, config_.acoustic_sway_sampling_coef);
    const float sigma_min = config_.acoustic_sigma_min;
    auto & graph = velocity_graph({window, cached - start, cached, true, true});
    std::vector<float> x_window(static_cast<size_t>(mel * window));
    for (int64_t step = 0; step < steps; ++step) {
        const float t0 = grid[static_cast<size_t>(step)];
        const float t1 = grid[static_cast<size_t>(step + 1)];
        const auto emb = time_embedding(*weights_, config_, t0);
        auto & x_cache = state.x[static_cast<size_t>(step)];
        for (int64_t c = 0; c < mel; ++c) {
            for (int64_t t = 0; t < window; ++t) {
                const int64_t frame = start + t;
                x_window[static_cast<size_t>(c * window + t)] = frame < cached
                    ? x_cache[static_cast<size_t>(frame * mel + c)]
                    : x[static_cast<size_t>(c * frames + frame - cached)];
            }
        }
        SoproVelocityInputs inputs;
        inputs.x = &x_window;
        inputs.cond_mel = &cond_mel;
        inputs.cond_mask = &cond_mask;
        inputs.mu = &mu;
        inputs.spk = &spk;
        inputs.emb = &emb;
        inputs.mask = &mask;
        inputs.keys = &state.keys[static_cast<size_t>(step)];
        inputs.values = &state.values[static_cast<size_t>(step)];
        std::vector<std::vector<float>> new_keys;
        std::vector<std::vector<float>> new_values;
        const auto velocity = graph.run(inputs, &new_keys, &new_values);
        for (size_t layer = 0; layer < layers; ++layer) {
            auto & keys = state.keys[static_cast<size_t>(step)][layer];
            auto & values = state.values[static_cast<size_t>(step)][layer];
            keys.insert(keys.end(), new_keys[layer].begin(), new_keys[layer].end());
            values.insert(values.end(), new_values[layer].begin(), new_values[layer].end());
        }
        for (int64_t t = 0; t < frames; ++t) {
            for (int64_t c = 0; c < mel; ++c) {
                x_cache.push_back(x[static_cast<size_t>(c * frames + t)]);
            }
        }
        const float dt = t1 - t0;
        const float prompt_scale = 1.0F - (1.0F - sigma_min) * t1;
        for (int64_t c = 0; c < mel; ++c) {
            for (int64_t t = 0; t < frames; ++t) {
                const auto index = static_cast<size_t>(c * frames + t);
                x[index] += dt * velocity[index];
                const auto window_index = static_cast<size_t>(c * window + cached - start + t);
                if (cond_mask[static_cast<size_t>(cached - start + t)] > 0.0F) {
                    x[index] = prompt_scale * x0[index] + t1 * cond_mel[window_index];
                }
            }
        }
    }
    for (int64_t c = 0; c < mel; ++c) {
        for (int64_t t = 0; t < frames; ++t) {
            if (cond_mask[static_cast<size_t>(cached - start + t)] > 0.0F) {
                x[static_cast<size_t>(c * frames + t)] = cond_mel[static_cast<size_t>(c * window + cached - start + t)];
            }
        }
    }
    state.cached = end;
    return x;
}

}  // namespace engine::community_models::sopro_tts
