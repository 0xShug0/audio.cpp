#include "engine/models/vevo2/fm.h"

#include "audio_cache.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/dsp.h"
#include "engine/framework/audio/waveform_ops.h"
#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/io/binary.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"
#include "engine/framework/sampling/torch_random.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace engine::models::vevo2 {

using Clock = std::chrono::steady_clock;

constexpr size_t kMaxTimbreMelCacheEntries = 4;

struct Vevo2FMWeights {
    std::shared_ptr<engine::core::BackendWeightStore> store;
    engine::core::TensorValue cond_emb;
    std::vector<engine::modules::ConvTranspose1dWeights> resampling_layers;

    engine::modules::LinearWeights cond_mlp_0;
    engine::modules::LinearWeights cond_mlp_2;
    engine::modules::LinearWeights diff_step_mlp_0;
    engine::modules::LinearWeights diff_step_mlp_2;
    engine::modules::LinearWeights mel_mlp_0;
    engine::modules::LinearWeights mel_mlp_2;
    engine::modules::LinearWeights mel_out_mlp_0;
    engine::modules::LinearWeights mel_out_mlp_2;
    engine::modules::LinearWeights final_norm_to_weight;

    struct Layer {
        engine::modules::LinearWeights input_norm_to_weight;
        engine::modules::LinearWeights post_norm_to_weight;
        engine::modules::LinearWeights q_proj;
        engine::modules::LinearWeights k_proj;
        engine::modules::LinearWeights v_proj;
        engine::modules::LinearWeights o_proj;
        engine::modules::LinearWeights gate_proj;
        engine::modules::LinearWeights up_proj;
        engine::modules::LinearWeights down_proj;
    };
    std::vector<Layer> layers;
};

namespace {

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

int64_t require_power_of_two_log2(int64_t value, const char * name) {
    if (value <= 0 || (value & (value - 1)) != 0) {
        throw std::runtime_error(std::string(name) + " must be a positive power of two");
    }
    int64_t log2 = 0;
    while (value > 1) {
        value >>= 1;
        ++log2;
    }
    return log2;
}

engine::core::TensorValue transpose_btc_to_bct(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input) {
    return engine::modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, input);
}

engine::core::TensorValue transpose_bct_to_btc(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input) {
    return engine::modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, input);
}

engine::core::TensorValue ensure_contiguous(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & value) {
    return engine::core::ensure_backend_addressable_layout(ctx, value);
}

engine::core::TensorValue conv_transpose1d_pytorch_padding_1(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    const engine::modules::ConvTranspose1dWeights & weights,
    int64_t channels) {
    auto padded_output = engine::modules::ConvTranspose1dModule({
        channels,
        channels,
        4,
        2,
        0,
        1,
        weights.bias.has_value(),
    }).build(ctx, input, weights);
    return engine::modules::SliceModule({
        2,
        1,
        padded_output.shape.dims[2] - 2,
    }).build(ctx, padded_output);
}

engine::core::TensorValue reshape_heads(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    int64_t heads,
    int64_t dim) {
    return engine::core::reshape_tensor(
        ctx,
        ensure_contiguous(ctx, input),
        engine::core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], heads, dim}));
}

engine::core::TensorValue attention_from_heads(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & q_heads,
    const engine::core::TensorValue & k_heads,
    const engine::core::TensorValue & v_heads,
    int64_t dim) {
    const engine::modules::MatMulModule matmul;
    auto scores = matmul.build(
        ctx,
        q_heads,
        engine::modules::TransposeModule({{0, 1, 3, 2}, k_heads.shape.rank}).build(ctx, k_heads));
    scores = engine::core::wrap_tensor(
        ggml_scale(ctx.ggml, scores.tensor, 1.0F / std::sqrt(static_cast<float>(dim))),
        scores.shape,
        GGML_TYPE_F32);
    scores = ensure_contiguous(ctx, scores);
    auto attn = engine::core::wrap_tensor(
        ggml_soft_max(ctx.ggml, scores.tensor),
        scores.shape,
        GGML_TYPE_F32);
    return matmul.build(ctx, attn, v_heads);
}

engine::core::TensorValue adaptive_rms_norm(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    const engine::core::TensorValue & timestep_embedding,
    const engine::modules::LinearWeights & to_weight,
    const Vevo2FMConfig & config) {
    auto normalized = engine::modules::RMSNormModule({config.hidden_size, 1.0e-6F, false, false})
                          .build(ctx, input, {});
    auto weight = engine::modules::LinearModule({
        config.hidden_size,
        config.hidden_size,
        to_weight.bias.has_value(),
        GGML_PREC_F32,
    }).build(ctx, timestep_embedding, to_weight);
    weight = engine::core::reshape_tensor(
        ctx,
        weight,
        engine::core::TensorShape::from_dims({1, 1, config.hidden_size}));
    auto repeated = engine::core::wrap_tensor(
        ggml_repeat(ctx.ggml, weight.tensor, normalized.tensor),
        normalized.shape,
        GGML_TYPE_F32);
    return engine::core::wrap_tensor(
        ggml_mul(ctx.ggml, normalized.tensor, repeated.tensor),
        normalized.shape,
        GGML_TYPE_F32);
}

engine::core::TensorValue mlp(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    const Vevo2FMWeights::Layer & weights,
    const Vevo2FMConfig & config) {
    auto gate = engine::modules::LinearModule({
        config.hidden_size,
        config.hidden_size * 4,
        weights.gate_proj.bias.has_value(),
        GGML_PREC_F32,
    }).build(ctx, input, weights.gate_proj);
    gate = engine::modules::SiluModule{}.build(ctx, gate);
    auto up = engine::modules::LinearModule({
        config.hidden_size,
        config.hidden_size * 4,
        weights.up_proj.bias.has_value(),
        GGML_PREC_F32,
    }).build(ctx, input, weights.up_proj);
    auto hidden = engine::modules::MulModule{}.build(ctx, gate, up);
    return engine::modules::LinearModule({
        config.hidden_size * 4,
        config.hidden_size,
        weights.down_proj.bias.has_value(),
        GGML_PREC_F32,
    }).build(ctx, hidden, weights.down_proj);
}

engine::core::TensorValue diff_llama_layer(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    const engine::core::TensorValue & timestep_embedding,
    const engine::core::TensorValue & positions,
    const Vevo2FMWeights::Layer & weights,
    const Vevo2FMConfig & config) {
    const int64_t head_dim = config.hidden_size / config.num_heads;
    auto hidden = adaptive_rms_norm(ctx, input, timestep_embedding, weights.input_norm_to_weight, config);
    auto q = engine::modules::LinearModule({
        config.hidden_size,
        config.hidden_size,
        weights.q_proj.bias.has_value(),
        GGML_PREC_F32,
    }).build(ctx, hidden, weights.q_proj);
    auto k = engine::modules::LinearModule({
        config.hidden_size,
        config.hidden_size,
        weights.k_proj.bias.has_value(),
        GGML_PREC_F32,
    }).build(ctx, hidden, weights.k_proj);
    auto v = engine::modules::LinearModule({
        config.hidden_size,
        config.hidden_size,
        weights.v_proj.bias.has_value(),
        GGML_PREC_F32,
    }).build(ctx, hidden, weights.v_proj);
    q = engine::modules::RoPEModule({head_dim, GGML_ROPE_TYPE_NEOX, 10000.0F})
            .build(ctx, reshape_heads(ctx, q, config.num_heads, head_dim), positions);
    k = engine::modules::RoPEModule({head_dim, GGML_ROPE_TYPE_NEOX, 10000.0F})
            .build(ctx, reshape_heads(ctx, k, config.num_heads, head_dim), positions);
    v = reshape_heads(ctx, v, config.num_heads, head_dim);

    auto q_heads = engine::modules::TransposeModule({{0, 2, 1, 3}, q.shape.rank}).build(ctx, q);
    auto k_heads = engine::modules::TransposeModule({{0, 2, 1, 3}, k.shape.rank}).build(ctx, k);
    auto v_heads = engine::modules::TransposeModule({{0, 2, 1, 3}, v.shape.rank}).build(ctx, v);
    auto context = attention_from_heads(ctx, q_heads, k_heads, v_heads, head_dim);
    context = engine::modules::TransposeModule({{0, 2, 1, 3}, context.shape.rank}).build(ctx, context);
    context = engine::core::reshape_tensor(
        ctx,
        ensure_contiguous(ctx, context),
        engine::core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], config.hidden_size}));
    hidden = engine::modules::LinearModule({
        config.hidden_size,
        config.hidden_size,
        weights.o_proj.bias.has_value(),
        GGML_PREC_F32,
    }).build(ctx, context, weights.o_proj);
    hidden = engine::modules::AddModule{}.build(ctx, input, hidden);

    auto ff_in = adaptive_rms_norm(ctx, hidden, timestep_embedding, weights.post_norm_to_weight, config);
    return engine::modules::AddModule{}.build(ctx, hidden, mlp(ctx, ff_in, weights, config));
}

std::vector<float> timestep_embedding(float timestep, int64_t hidden_size) {
    const int64_t half = hidden_size / 2;
    std::vector<float> out(static_cast<size_t>(hidden_size), 0.0F);
    const double scale = std::log(10000.0) / static_cast<double>(half - 1);
    for (int64_t index = 0; index < half; ++index) {
        const double freq = std::exp(static_cast<double>(index) * -scale);
        const double arg = static_cast<double>(timestep) * freq;
        out[static_cast<size_t>(index)] = static_cast<float>(std::sin(arg));
        out[static_cast<size_t>(half + index)] = static_cast<float>(std::cos(arg));
    }
    return out;
}

std::vector<float> load_initial_noise_or_sample(const std::string & noise_file, size_t count, uint32_t seed) {
    if (noise_file.empty()) {
        return engine::sampling::generate_torch_cuda_randn(
            count,
            seed,
            engine::sampling::TorchRandnPrecision::Float32);
    }
    auto values = engine::io::read_f32_file(noise_file);
    if (values.size() < count) {
        throw std::runtime_error(
            "Vevo2 FM noise file is too short: expected at least " +
            std::to_string(count) + " floats, got " + std::to_string(values.size()));
    }
    values.resize(count);
    return values;
}

engine::core::TensorValue tensor_std_correction1(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & values) {
    const int64_t count = values.shape.num_elements();
    if (count < 2) {
        throw std::runtime_error("Vevo2 FM CFG std requires at least two values");
    }
    auto sum = engine::core::wrap_tensor(
        ggml_sum(ctx.ggml, values.tensor),
        engine::core::TensorShape::from_dims({1}),
        GGML_TYPE_F32);
    auto mean_scalar = engine::core::wrap_tensor(
        ggml_scale(ctx.ggml, sum.tensor, 1.0F / static_cast<float>(count)),
        sum.shape,
        GGML_TYPE_F32);
    auto mean = engine::core::wrap_tensor(
        ggml_repeat(ctx.ggml, mean_scalar.tensor, values.tensor),
        values.shape,
        GGML_TYPE_F32);
    auto centered = engine::core::wrap_tensor(
        ggml_sub(ctx.ggml, values.tensor, mean.tensor),
        values.shape,
        GGML_TYPE_F32);
    auto variance = engine::core::wrap_tensor(
        ggml_scale(
            ctx.ggml,
            ggml_sum(ctx.ggml, ggml_sqr(ctx.ggml, centered.tensor)),
            1.0F / static_cast<float>(count - 1)),
        engine::core::TensorShape::from_dims({1}),
        GGML_TYPE_F32);
    return engine::core::wrap_tensor(ggml_sqrt(ctx.ggml, variance.tensor), variance.shape, GGML_TYPE_F32);
}

Vevo2MelSequence extract_timbre_mel(
    const runtime::AudioBuffer & timbre_ref_audio,
    const Vevo2AcousticPreprocessConfig & config,
    size_t threads) {
    const int64_t pad = (config.n_fft - config.hop_size) / 2;
    auto waveform = engine::audio::reflect_pad_samples(
        engine::audio::convert_interleaved_audio_to_mono_linear_resampled(
            timbre_ref_audio.samples,
            timbre_ref_audio.sample_rate,
            timbre_ref_audio.channels,
            24000),
        pad,
        pad);
    const engine::audio::STFTConfig stft_config{
        config.n_fft,
        config.hop_size,
        config.win_size,
        false,
        engine::audio::STFTPadMode::Reflect,
        engine::audio::STFTFamily::Default,
    };
    const auto & window = engine::audio::get_cached_stft_window(stft_config);
    const auto complex = engine::audio::STFT().compute_complex(
        waveform,
        window,
        1,
        static_cast<int64_t>(waveform.size()),
        stft_config,
        threads);
    const int64_t freq_bins = config.n_fft / 2 + 1;
    const int64_t frames = complex.shape[2];
    std::vector<float> magnitude(static_cast<size_t>(freq_bins * frames), 0.0F);
    for (int64_t freq = 0; freq < freq_bins; ++freq) {
        for (int64_t frame = 0; frame < frames; ++frame) {
            const size_t src = static_cast<size_t>((freq * frames + frame) * 2);
            const float re = complex.values[src];
            const float im = complex.values[src + 1];
            magnitude[static_cast<size_t>(freq * frames + frame)] = std::sqrt(re * re + im * im + 1.0e-9F);
        }
    }

    const auto mel_filter = engine::audio::MelFilterbank().build({
        config.sample_rate,
        config.n_fft,
        config.num_mels,
        config.fmin,
        config.fmax,
        true,
    });
    const auto mel = engine::audio::MelFilterbank().compute_custom(
        magnitude,
        1,
        freq_bins,
        frames,
        mel_filter);
    Vevo2MelSequence out;
    out.frames = frames;
    out.mel_bins = config.num_mels;
    out.values.assign(static_cast<size_t>(frames * config.num_mels), 0.0F);
    const float inv_std = 1.0F / std::sqrt(config.mel_var);
    for (int64_t frame = 0; frame < frames; ++frame) {
        for (int64_t mel_bin = 0; mel_bin < config.num_mels; ++mel_bin) {
            const float logged = std::log(std::max(
                mel.values[static_cast<size_t>(mel_bin * frames + frame)],
                1.0e-5F));
            out.values[static_cast<size_t>(frame * config.num_mels + mel_bin)] =
                (logged - config.mel_mean) * inv_std;
        }
    }
    return out;
}

std::shared_ptr<const Vevo2FMWeights> load_fm_weights(
    ggml_backend_t backend,
    engine::core::BackendType backend_type,
    const engine::assets::TensorSource & source,
    const Vevo2FMConfig & config,
    size_t weight_context_bytes,
    engine::assets::TensorStorageType matmul_storage_type,
    engine::assets::TensorStorageType conv_storage_type) {
    auto weights = std::make_shared<Vevo2FMWeights>();
    weights->store = std::make_shared<engine::core::BackendWeightStore>(
        backend,
        backend_type,
        "vevo2.fm.weights",
        weight_context_bytes);
    weights->cond_emb = weights->store->load_tensor(
        source,
        "cond_emb.weight",
        matmul_storage_type,
        {config.cond_codebook_size, config.hidden_size});

    const int64_t resampling_layers = require_power_of_two_log2(config.cond_scale_factor, "Vevo2 FM cond_scale_factor");
    weights->resampling_layers.reserve(static_cast<size_t>(resampling_layers));
    for (int64_t layer = 0; layer < resampling_layers; ++layer) {
        weights->resampling_layers.push_back(engine::modules::binding::conv_transpose1d_from_source(
            *weights->store,
            source,
            "resampling_layers." + std::to_string(layer * 2),
            conv_storage_type,
            config.hidden_size,
            config.hidden_size,
            4,
            true));
    }

    weights->cond_mlp_0 = engine::modules::binding::linear_from_source(
        *weights->store,
        source,
        "diff_estimator.cond_mlp.0",
        matmul_storage_type,
        config.hidden_size * 4,
        config.hidden_size,
        true);
    weights->cond_mlp_2 = engine::modules::binding::linear_from_source(
        *weights->store,
        source,
        "diff_estimator.cond_mlp.2",
        matmul_storage_type,
        config.hidden_size,
        config.hidden_size * 4,
        true);
    weights->diff_step_mlp_0 = engine::modules::binding::linear_from_source(
        *weights->store,
        source,
        "diff_estimator.diff_step_mlp.0",
        matmul_storage_type,
        config.hidden_size * 4,
        config.hidden_size,
        true);
    weights->diff_step_mlp_2 = engine::modules::binding::linear_from_source(
        *weights->store,
        source,
        "diff_estimator.diff_step_mlp.2",
        matmul_storage_type,
        config.hidden_size,
        config.hidden_size * 4,
        true);
    weights->mel_mlp_0 = engine::modules::binding::linear_from_source(
        *weights->store,
        source,
        "diff_estimator.mel_mlp.0",
        matmul_storage_type,
        config.hidden_size * 4,
        config.mel_dim,
        true);
    weights->mel_mlp_2 = engine::modules::binding::linear_from_source(
        *weights->store,
        source,
        "diff_estimator.mel_mlp.2",
        matmul_storage_type,
        config.hidden_size,
        config.hidden_size * 4,
        true);
    weights->mel_out_mlp_0 = engine::modules::binding::linear_from_source(
        *weights->store,
        source,
        "diff_estimator.mel_out_mlp.0",
        matmul_storage_type,
        config.hidden_size * 4,
        config.hidden_size,
        true);
    weights->mel_out_mlp_2 = engine::modules::binding::linear_from_source(
        *weights->store,
        source,
        "diff_estimator.mel_out_mlp.2",
        matmul_storage_type,
        config.mel_dim,
        config.hidden_size * 4,
        true);
    weights->final_norm_to_weight = engine::modules::binding::linear_from_source(
        *weights->store,
        source,
        "diff_estimator.norm.to_weight",
        matmul_storage_type,
        config.hidden_size,
        config.hidden_size,
        true);

    weights->layers.reserve(static_cast<size_t>(config.num_layers));
    for (int64_t layer = 0; layer < config.num_layers; ++layer) {
        const std::string prefix = "diff_estimator.layers." + std::to_string(layer);
        Vevo2FMWeights::Layer layer_weights;
        layer_weights.input_norm_to_weight = engine::modules::binding::linear_from_source(
            *weights->store,
            source,
            prefix + ".input_layernorm.to_weight",
            matmul_storage_type,
            config.hidden_size,
            config.hidden_size,
            true);
        layer_weights.post_norm_to_weight = engine::modules::binding::linear_from_source(
            *weights->store,
            source,
            prefix + ".post_attention_layernorm.to_weight",
            matmul_storage_type,
            config.hidden_size,
            config.hidden_size,
            true);
        layer_weights.q_proj = engine::modules::binding::linear_from_source(
            *weights->store,
            source,
            prefix + ".self_attn.q_proj",
            matmul_storage_type,
            config.hidden_size,
            config.hidden_size,
            false);
        layer_weights.k_proj = engine::modules::binding::linear_from_source(
            *weights->store,
            source,
            prefix + ".self_attn.k_proj",
            matmul_storage_type,
            config.hidden_size,
            config.hidden_size,
            false);
        layer_weights.v_proj = engine::modules::binding::linear_from_source(
            *weights->store,
            source,
            prefix + ".self_attn.v_proj",
            matmul_storage_type,
            config.hidden_size,
            config.hidden_size,
            false);
        layer_weights.o_proj = engine::modules::binding::linear_from_source(
            *weights->store,
            source,
            prefix + ".self_attn.o_proj",
            matmul_storage_type,
            config.hidden_size,
            config.hidden_size,
            false);
        layer_weights.gate_proj = engine::modules::binding::linear_from_source(
            *weights->store,
            source,
            prefix + ".mlp.gate_proj",
            matmul_storage_type,
            config.hidden_size * 4,
            config.hidden_size,
            false);
        layer_weights.up_proj = engine::modules::binding::linear_from_source(
            *weights->store,
            source,
            prefix + ".mlp.up_proj",
            matmul_storage_type,
            config.hidden_size * 4,
            config.hidden_size,
            false);
        layer_weights.down_proj = engine::modules::binding::linear_from_source(
            *weights->store,
            source,
            prefix + ".mlp.down_proj",
            matmul_storage_type,
            config.hidden_size,
            config.hidden_size * 4,
            false);
        weights->layers.push_back(std::move(layer_weights));
    }
    weights->store->upload();
    return weights;
}

engine::core::TensorValue build_diff_llama(
    engine::core::ModuleBuildContext & build_ctx,
    const engine::core::TensorValue & x_input,
    const engine::core::TensorValue & cond_input,
    const engine::core::TensorValue & timestep_input,
    const engine::core::TensorValue & positions,
    const Vevo2FMWeights & weights,
    const Vevo2FMConfig & config) {
    auto cond_embedding = engine::modules::LinearModule({
        config.hidden_size,
        config.hidden_size * 4,
        weights.cond_mlp_0.bias.has_value(),
        GGML_PREC_F32,
    }).build(build_ctx, cond_input, weights.cond_mlp_0);
    cond_embedding = engine::modules::SiluModule{}.build(build_ctx, cond_embedding);
    cond_embedding = engine::modules::LinearModule({
        config.hidden_size * 4,
        config.hidden_size,
        weights.cond_mlp_2.bias.has_value(),
        GGML_PREC_F32,
    }).build(build_ctx, cond_embedding, weights.cond_mlp_2);

    auto x = engine::modules::LinearModule({
        config.mel_dim,
        config.hidden_size * 4,
        weights.mel_mlp_0.bias.has_value(),
        GGML_PREC_F32,
    }).build(build_ctx, x_input, weights.mel_mlp_0);
    x = engine::modules::SiluModule{}.build(build_ctx, x);
    x = engine::modules::LinearModule({
        config.hidden_size * 4,
        config.hidden_size,
        weights.mel_mlp_2.bias.has_value(),
        GGML_PREC_F32,
    }).build(build_ctx, x, weights.mel_mlp_2);

    auto diffusion_step = engine::modules::LinearModule({
        config.hidden_size,
        config.hidden_size * 4,
        weights.diff_step_mlp_0.bias.has_value(),
        GGML_PREC_F32,
    }).build(build_ctx, timestep_input, weights.diff_step_mlp_0);
    diffusion_step = engine::modules::SiluModule{}.build(build_ctx, diffusion_step);
    diffusion_step = engine::modules::LinearModule({
        config.hidden_size * 4,
        config.hidden_size,
        weights.diff_step_mlp_2.bias.has_value(),
        GGML_PREC_F32,
    }).build(build_ctx, diffusion_step, weights.diff_step_mlp_2);

    x = engine::modules::AddModule{}.build(build_ctx, x, cond_embedding);
    for (const auto & layer : weights.layers) {
        x = diff_llama_layer(build_ctx, x, diffusion_step, positions, layer, config);
    }
    x = adaptive_rms_norm(build_ctx, x, diffusion_step, weights.final_norm_to_weight, config);
    x = engine::modules::LinearModule({
        config.hidden_size,
        config.hidden_size * 4,
        weights.mel_out_mlp_0.bias.has_value(),
        GGML_PREC_F32,
    }).build(build_ctx, x, weights.mel_out_mlp_0);
    x = engine::modules::SiluModule{}.build(build_ctx, x);
    x = engine::modules::LinearModule({
        config.hidden_size * 4,
        config.mel_dim,
        weights.mel_out_mlp_2.bias.has_value(),
        GGML_PREC_F32,
    }).build(build_ctx, x, weights.mel_out_mlp_2);
    return ensure_contiguous(build_ctx, x);
}

}  // namespace

struct Vevo2FMGraph {
    Vevo2FMGraph(
        ggml_backend_t backend,
        engine::core::BackendType backend_type,
        size_t graph_context_bytes,
        const Vevo2FMConfig & config,
        std::shared_ptr<const Vevo2FMWeights> weights,
        int64_t code_tokens)
        : backend(backend),
          weights(std::move(weights)),
          code_tokens(code_tokens) {
        if (backend == nullptr || this->weights == nullptr) {
            throw std::runtime_error("Vevo2 FM graph requires backend and weights");
        }
        if (code_tokens <= 0) {
            throw std::runtime_error("Vevo2 FM graph requires positive token count");
        }
        ggml_init_params params{graph_context_bytes, nullptr, true};
        ctx.reset(ggml_init(params));
        if (ctx == nullptr) {
            throw std::runtime_error("failed to initialize Vevo2 FM graph context");
        }

        engine::core::ModuleBuildContext build_ctx{ctx.get(), "vevo2.fm", backend_type};
        code_input = engine::core::make_tensor(
            build_ctx,
            GGML_TYPE_I32,
            engine::core::TensorShape::from_dims({1, code_tokens})).tensor;
        ggml_set_input(code_input);
        auto codes = engine::core::wrap_tensor(
            code_input,
            engine::core::TensorShape::from_dims({1, code_tokens}),
            GGML_TYPE_I32);
        auto cond = engine::modules::EmbeddingModule({config.cond_codebook_size, config.hidden_size})
                        .build(build_ctx, codes, this->weights->cond_emb);
        if (config.cond_scale_factor != 1) {
            auto cond_bct = transpose_btc_to_bct(build_ctx, cond);
            for (const auto & resampler : this->weights->resampling_layers) {
                cond_bct = conv_transpose1d_pytorch_padding_1(build_ctx, cond_bct, resampler, config.hidden_size);
                cond_bct = engine::modules::GeluModule({engine::modules::GeluApproximation::ExactErf})
                               .build(build_ctx, cond_bct);
            }
            cond = transpose_bct_to_btc(build_ctx, cond_bct);
        }
        cond = ensure_contiguous(build_ctx, cond);
        cond_output = cond.tensor;
        cond_frames = cond.shape.dims[1];
        ggml_set_output(cond_output);

        graph = ggml_new_graph_custom(ctx.get(), 65536, false);
        ggml_build_forward_expand(graph, cond_output);
        gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (gallocr == nullptr || !ggml_gallocr_reserve(gallocr, graph) || !ggml_gallocr_alloc_graph(gallocr, graph)) {
            throw std::runtime_error("failed to allocate Vevo2 FM graph");
        }
    }

    ~Vevo2FMGraph() {
        if (gallocr != nullptr) {
            ggml_gallocr_free(gallocr);
            gallocr = nullptr;
        }
    }

    bool matches(const Vevo2FMWeights & other_weights, int64_t other_code_tokens) const noexcept {
        return weights.get() == &other_weights && code_tokens == other_code_tokens;
    }

    std::vector<float> run_conditioning(const std::vector<int32_t> & tokens, const Vevo2FMConfig & config) {
        if (static_cast<int64_t>(tokens.size()) != code_tokens) {
            throw std::runtime_error("Vevo2 FM conditioning token count mismatch");
        }
        ggml_backend_tensor_set(code_input, tokens.data(), 0, tokens.size() * sizeof(int32_t));
        const ggml_status status = engine::core::compute_backend_graph(backend, graph);
        ggml_backend_synchronize(backend);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Vevo2 FM conditioning graph compute failed");
        }
        std::vector<float> out(static_cast<size_t>(cond_frames * config.hidden_size), 0.0F);
        ggml_backend_tensor_get(cond_output, out.data(), 0, out.size() * sizeof(float));
        return out;
    }

    ggml_backend_t backend = nullptr;
    std::shared_ptr<const Vevo2FMWeights> weights;
    int64_t code_tokens = 0;
    int64_t cond_frames = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx;
    ggml_tensor * code_input = nullptr;
    ggml_tensor * cond_output = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_gallocr_t gallocr = nullptr;
};

struct Vevo2FMStepGraph {
    Vevo2FMStepGraph(
        ggml_backend_t backend,
        engine::core::BackendType backend_type,
        size_t graph_context_bytes,
        const Vevo2FMConfig & config,
        std::shared_ptr<const Vevo2FMWeights> weights,
        int64_t cond_frames,
        int64_t prompt_frames,
        int64_t target_frames,
        float step_size)
        : backend(backend),
          weights(std::move(weights)),
          cond_frames(cond_frames),
          prompt_frames(prompt_frames),
          target_frames(target_frames) {
        if (backend == nullptr || this->weights == nullptr) {
            throw std::runtime_error("Vevo2 FM step graph requires backend and weights");
        }
        if (cond_frames <= 0 || prompt_frames <= 0 || target_frames <= 0 || prompt_frames + target_frames != cond_frames) {
            throw std::runtime_error("Vevo2 FM step graph requires positive prompt/target frames");
        }
        if (config.hidden_size % config.num_heads != 0) {
            throw std::runtime_error("Vevo2 FM hidden_size must be divisible by num_heads");
        }
        ggml_init_params params{graph_context_bytes, nullptr, true};
        ctx.reset(ggml_init(params));
        if (ctx == nullptr) {
            throw std::runtime_error("failed to initialize Vevo2 FM step graph context");
        }

        engine::core::ModuleBuildContext build_ctx{ctx.get(), "vevo2.fm.step", backend_type};
        prompt_input = engine::core::make_tensor(
            build_ctx,
            GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({1, prompt_frames, config.mel_dim})).tensor;
        xt_input = engine::core::make_tensor(
            build_ctx,
            GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({1, target_frames, config.mel_dim})).tensor;
        cond_input = engine::core::make_tensor(
            build_ctx,
            GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({1, cond_frames, config.hidden_size})).tensor;
        uncond_cond_input = engine::core::make_tensor(
            build_ctx,
            GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({1, target_frames, config.hidden_size})).tensor;
        timestep_input = engine::core::make_tensor(
            build_ctx,
            GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({1, config.hidden_size})).tensor;
        cond_position_input = engine::core::make_tensor(
            build_ctx,
            GGML_TYPE_I32,
            engine::core::TensorShape::from_dims({cond_frames})).tensor;
        uncond_position_input = engine::core::make_tensor(
            build_ctx,
            GGML_TYPE_I32,
            engine::core::TensorShape::from_dims({target_frames})).tensor;
        ggml_set_input(prompt_input);
        ggml_set_input(xt_input);
        ggml_set_input(cond_input);
        ggml_set_input(uncond_cond_input);
        ggml_set_input(timestep_input);
        ggml_set_input(cond_position_input);
        ggml_set_input(uncond_position_input);

        auto prompt = engine::core::wrap_tensor(
            prompt_input,
            engine::core::TensorShape::from_dims({1, prompt_frames, config.mel_dim}),
            GGML_TYPE_F32);
        auto xt = engine::core::wrap_tensor(
            xt_input,
            engine::core::TensorShape::from_dims({1, target_frames, config.mel_dim}),
            GGML_TYPE_F32);
        auto cond = engine::core::wrap_tensor(
            cond_input,
            engine::core::TensorShape::from_dims({1, cond_frames, config.hidden_size}),
            GGML_TYPE_F32);
        auto uncond_cond = engine::core::wrap_tensor(
            uncond_cond_input,
            engine::core::TensorShape::from_dims({1, target_frames, config.hidden_size}),
            GGML_TYPE_F32);
        auto timestep = engine::core::wrap_tensor(
            timestep_input,
            engine::core::TensorShape::from_dims({1, config.hidden_size}),
            GGML_TYPE_F32);
        auto cond_positions = engine::core::wrap_tensor(
            cond_position_input,
            engine::core::TensorShape::from_dims({cond_frames}),
            GGML_TYPE_I32);
        auto uncond_positions = engine::core::wrap_tensor(
            uncond_position_input,
            engine::core::TensorShape::from_dims({target_frames}),
            GGML_TYPE_I32);

        auto full_x = engine::modules::ConcatModule({1}).build(build_ctx, prompt, xt);
        auto cond_flow = build_diff_llama(build_ctx, full_x, cond, timestep, cond_positions, *this->weights, config);
        auto flow_pred = engine::modules::SliceModule({1, prompt_frames, target_frames}).build(build_ctx, cond_flow);
        auto uncond_flow = build_diff_llama(build_ctx, xt, uncond_cond, timestep, uncond_positions, *this->weights, config);

        constexpr float kCfg = 1.0F;
        constexpr float kRescaleCfg = 0.75F;
        const auto pos_std = tensor_std_correction1(build_ctx, flow_pred);
        auto guidance = engine::core::wrap_tensor(
            ggml_sub(build_ctx.ggml, flow_pred.tensor, uncond_flow.tensor),
            flow_pred.shape,
            GGML_TYPE_F32);
        guidance = engine::core::wrap_tensor(
            ggml_scale(build_ctx.ggml, guidance.tensor, kCfg),
            guidance.shape,
            GGML_TYPE_F32);
        auto flow_cfg = engine::modules::AddModule{}.build(build_ctx, flow_pred, guidance);
        const auto cfg_std = tensor_std_correction1(build_ctx, flow_cfg);
        auto std_ratio = engine::core::wrap_tensor(
            ggml_div(build_ctx.ggml, pos_std.tensor, cfg_std.tensor),
            pos_std.shape,
            GGML_TYPE_F32);
        std_ratio = engine::core::wrap_tensor(
            ggml_repeat(build_ctx.ggml, std_ratio.tensor, flow_cfg.tensor),
            flow_cfg.shape,
            GGML_TYPE_F32);
        auto rescaled = engine::modules::MulModule{}.build(build_ctx, flow_cfg, std_ratio);
        rescaled = engine::core::wrap_tensor(
            ggml_scale(build_ctx.ggml, rescaled.tensor, kRescaleCfg),
            rescaled.shape,
            GGML_TYPE_F32);
        flow_cfg = engine::core::wrap_tensor(
            ggml_scale(build_ctx.ggml, flow_cfg.tensor, 1.0F - kRescaleCfg),
            flow_cfg.shape,
            GGML_TYPE_F32);
        auto flow = engine::modules::AddModule{}.build(build_ctx, rescaled, flow_cfg);
        flow = engine::core::wrap_tensor(
            ggml_scale(build_ctx.ggml, flow.tensor, step_size),
            flow.shape,
            GGML_TYPE_F32);
        auto xt_next = engine::modules::AddModule{}.build(build_ctx, xt, flow);
        xt_next = ensure_contiguous(build_ctx, xt_next);
        output = xt_next.tensor;
        ggml_set_output(output);

        graph = ggml_new_graph_custom(ctx.get(), 524288, false);
        ggml_build_forward_expand(graph, output);
        buffer = ggml_backend_alloc_ctx_tensors(ctx.get(), backend);
        if (buffer == nullptr) {
            throw std::runtime_error("failed to allocate Vevo2 FM step graph");
        }

        std::vector<int32_t> positions_data(static_cast<size_t>(cond_frames));
        for (int64_t frame = 0; frame < cond_frames; ++frame) {
            positions_data[static_cast<size_t>(frame)] = static_cast<int32_t>(frame);
        }
        ggml_backend_tensor_set(
            cond_position_input,
            positions_data.data(),
            0,
            positions_data.size() * sizeof(int32_t));
        positions_data.resize(static_cast<size_t>(target_frames));
        for (int64_t frame = 0; frame < target_frames; ++frame) {
            positions_data[static_cast<size_t>(frame)] = static_cast<int32_t>(frame);
        }
        ggml_backend_tensor_set(
            uncond_position_input,
            positions_data.data(),
            0,
            positions_data.size() * sizeof(int32_t));
    }

    ~Vevo2FMStepGraph() {
        if (buffer != nullptr) {
            ggml_backend_buffer_free(buffer);
            buffer = nullptr;
        }
    }

    bool matches(
        const Vevo2FMWeights & other_weights,
        int64_t other_cond_frames,
        int64_t other_prompt_frames,
        int64_t other_target_frames) const noexcept {
        return weights.get() == &other_weights && cond_frames == other_cond_frames &&
               prompt_frames == other_prompt_frames && target_frames == other_target_frames;
    }

    void initialize(
        const std::vector<float> & prompt,
        const std::vector<float> & cond,
        const std::vector<float> & xt,
        const Vevo2FMConfig & config) {
        if (static_cast<int64_t>(prompt.size()) != prompt_frames * config.mel_dim) {
            throw std::runtime_error("Vevo2 FM step prompt shape mismatch");
        }
        if (static_cast<int64_t>(xt.size()) != target_frames * config.mel_dim) {
            throw std::runtime_error("Vevo2 FM step xt shape mismatch");
        }
        if (static_cast<int64_t>(cond.size()) != cond_frames * config.hidden_size) {
            throw std::runtime_error("Vevo2 FM step cond shape mismatch");
        }
        ggml_backend_tensor_set(prompt_input, prompt.data(), 0, prompt.size() * sizeof(float));
        ggml_backend_tensor_set(xt_input, xt.data(), 0, xt.size() * sizeof(float));
        ggml_backend_tensor_set(cond_input, cond.data(), 0, cond.size() * sizeof(float));
        std::vector<float> zeros(static_cast<size_t>(target_frames * config.hidden_size), 0.0F);
        ggml_backend_tensor_set(uncond_cond_input, zeros.data(), 0, zeros.size() * sizeof(float));
    }

    void run_step(
        const std::vector<float> & timestep,
        const Vevo2FMConfig & config) {
        if (static_cast<int64_t>(timestep.size()) != config.hidden_size) {
            throw std::runtime_error("Vevo2 FM step timestep shape mismatch");
        }
        ggml_backend_tensor_set(timestep_input, timestep.data(), 0, timestep.size() * sizeof(float));
        const ggml_status status = engine::core::compute_backend_graph(backend, graph);
        ggml_backend_synchronize(backend);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Vevo2 FM step graph compute failed");
        }
        ggml_backend_tensor_copy(output, xt_input);
    }

    std::vector<float> read_output(const Vevo2FMConfig & config) const {
        std::vector<float> out(static_cast<size_t>(target_frames * config.mel_dim), 0.0F);
        ggml_backend_tensor_get(output, out.data(), 0, out.size() * sizeof(float));
        return out;
    }

    ggml_backend_t backend = nullptr;
    std::shared_ptr<const Vevo2FMWeights> weights;
    int64_t cond_frames = 0;
    int64_t prompt_frames = 0;
    int64_t target_frames = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx;
    ggml_tensor * prompt_input = nullptr;
    ggml_tensor * xt_input = nullptr;
    ggml_tensor * cond_input = nullptr;
    ggml_tensor * uncond_cond_input = nullptr;
    ggml_tensor * timestep_input = nullptr;
    ggml_tensor * cond_position_input = nullptr;
    ggml_tensor * uncond_position_input = nullptr;
    ggml_tensor * output = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
};

Vevo2FlowMatchingRuntime::Vevo2FlowMatchingRuntime(
    const Vevo2Assets & assets,
    engine::core::ExecutionContext & execution_context,
    size_t weight_context_bytes,
    size_t graph_context_bytes,
    engine::assets::TensorStorageType matmul_weight_storage_type,
    engine::assets::TensorStorageType conv_weight_storage_type)
    : config_(assets.config.fm),
      execution_context_(execution_context),
      graph_context_bytes_(graph_context_bytes),
      weight_source_(assets.fm_weights),
      weights_(load_fm_weights(
          execution_context.backend(),
          execution_context.backend_type(),
          *weight_source_,
          config_,
          weight_context_bytes,
          matmul_weight_storage_type,
          conv_weight_storage_type)),
      name_("flow_matching_model") {
    weight_source_->release_storage();
}

Vevo2FlowMatchingRuntime::~Vevo2FlowMatchingRuntime() = default;

Vevo2MelSequence Vevo2FlowMatchingRuntime::cached_timbre_mel(const runtime::AudioBuffer & timbre_ref_audio) const {
    const uint64_t key = audio_buffer_key(timbre_ref_audio);
    for (const auto & entry : timbre_mel_cache_) {
        if (entry.key == key &&
            entry.sample_rate == timbre_ref_audio.sample_rate &&
            entry.channels == timbre_ref_audio.channels &&
            entry.samples == timbre_ref_audio.samples.size()) {
            return entry.mel;
        }
    }
    auto mel = extract_timbre_mel(
        timbre_ref_audio,
        config_.preprocess,
        static_cast<size_t>(execution_context_.config().threads));
    if (timbre_mel_cache_.size() >= kMaxTimbreMelCacheEntries) {
        timbre_mel_cache_.erase(timbre_mel_cache_.begin());
    }
    timbre_mel_cache_.push_back({
        key,
        timbre_ref_audio.sample_rate,
        timbre_ref_audio.channels,
        timbre_ref_audio.samples.size(),
        mel,
    });
    return mel;
}

Vevo2MelSequence Vevo2FlowMatchingRuntime::generate_mel(
    const runtime::AudioBuffer & timbre_ref_audio,
    const Vevo2TokenSequence & timbre_tokens,
    const Vevo2TokenSequence & generated_tokens,
    const Vevo2GenerationOptions & generation) const {
    const auto total_start = Clock::now();
    const auto timbre_mel_start = Clock::now();
    const auto timbre_prompt_mel = cached_timbre_mel(timbre_ref_audio);
    const double timbre_mel_ms = engine::debug::elapsed_ms(timbre_mel_start);
    std::vector<int32_t> diffusion_tokens;
    diffusion_tokens.reserve(timbre_tokens.ids.size() + generated_tokens.ids.size());
    diffusion_tokens.insert(diffusion_tokens.end(), timbre_tokens.ids.begin(), timbre_tokens.ids.end());
    diffusion_tokens.insert(diffusion_tokens.end(), generated_tokens.ids.begin(), generated_tokens.ids.end());
    if (diffusion_tokens.empty()) {
        throw std::runtime_error("Vevo2 FM requires non-empty content-style conditioning tokens");
    }
    double cond_graph_build_ms = 0.0;
    if (graph_ == nullptr || !graph_->matches(*weights_, static_cast<int64_t>(diffusion_tokens.size()))) {
        const auto build_start = Clock::now();
        graph_ = std::make_unique<Vevo2FMGraph>(
            execution_context_.backend(),
            execution_context_.backend_type(),
            graph_context_bytes_,
            config_,
            weights_,
            static_cast<int64_t>(diffusion_tokens.size()));
        cond_graph_build_ms = engine::debug::elapsed_ms(build_start);
    }
    const auto cond_run_start = Clock::now();
    const auto diffusion_cond = graph_->run_conditioning(diffusion_tokens, config_);
    const double cond_run_ms = engine::debug::elapsed_ms(cond_run_start);
    const int64_t cond_frames = static_cast<int64_t>(diffusion_cond.size()) / config_.hidden_size;
    if (cond_frames * config_.hidden_size != static_cast<int64_t>(diffusion_cond.size())) {
        throw std::runtime_error("Vevo2 FM conditioning output shape mismatch");
    }
    const int64_t prompt_len = timbre_prompt_mel.frames;
    const int64_t target_len = cond_frames - prompt_len;
    if (target_len <= 0) {
        throw std::runtime_error("Vevo2 FM target mel length must be positive");
    }
    if (static_cast<int64_t>(timbre_prompt_mel.values.size()) != prompt_len * config_.mel_dim) {
        throw std::runtime_error("Vevo2 FM timbre prompt mel shape mismatch");
    }
    if (generation.num_inference_steps <= 0) {
        throw std::runtime_error("Vevo2 FM num_inference_steps must be positive");
    }

    std::vector<float> xt = load_initial_noise_or_sample(
        generation.fm_noise_file,
        static_cast<size_t>(target_len * config_.mel_dim),
        generation.seed);
    const float h = 1.0F / static_cast<float>(generation.num_inference_steps);

    double step_graph_build_ms = 0.0;
    if (step_graph_ == nullptr || !step_graph_->matches(*weights_, cond_frames, prompt_len, target_len)) {
        const auto build_start = Clock::now();
        step_graph_ = std::make_unique<Vevo2FMStepGraph>(
            execution_context_.backend(),
            execution_context_.backend_type(),
            graph_context_bytes_,
            config_,
            weights_,
            cond_frames,
            prompt_len,
            target_len,
            h);
        step_graph_build_ms = engine::debug::elapsed_ms(build_start);
    }
    const auto init_start = Clock::now();
    step_graph_->initialize(timbre_prompt_mel.values, diffusion_cond, xt, config_);
    const double step_graph_init_ms = engine::debug::elapsed_ms(init_start);

    double timestep_ms = 0.0;
    double step_compute_ms = 0.0;
    for (int step = 0; step < generation.num_inference_steps; ++step) {
        const float t = (static_cast<float>(step) + 0.5F) * h;
        const auto timestep_start = Clock::now();
        const auto t_embedding = timestep_embedding(t, config_.hidden_size);
        timestep_ms += engine::debug::elapsed_ms(timestep_start);
        const auto step_start = Clock::now();
        step_graph_->run_step(t_embedding, config_);
        step_compute_ms += engine::debug::elapsed_ms(step_start);
    }

    Vevo2MelSequence out;
    out.frames = target_len;
    out.mel_bins = config_.mel_dim;
    const auto read_start = Clock::now();
    out.values = step_graph_->read_output(config_);
    const double final_read_ms = engine::debug::elapsed_ms(read_start);
    engine::debug::timing_log_scalar("vevo2.fm.timbre_mel_ms", timbre_mel_ms);
    engine::debug::timing_log_scalar("vevo2.fm.cond.graph.build_ms", cond_graph_build_ms);
    engine::debug::timing_log_scalar("vevo2.fm.cond_run_ms", cond_run_ms);
    engine::debug::timing_log_scalar("vevo2.fm.step.graph.build_ms", step_graph_build_ms);
    engine::debug::timing_log_scalar("vevo2.fm.step.graph.init_ms", step_graph_init_ms);
    engine::debug::timing_log_scalar("vevo2.fm.timestep_embedding_ms", timestep_ms);
    engine::debug::timing_log_scalar("vevo2.fm.step_compute_ms", step_compute_ms);
    engine::debug::timing_log_scalar("vevo2.fm.final_read_ms", final_read_ms);
    engine::debug::trace_log_scalar("vevo2.fm.steps", generation.num_inference_steps);
    engine::debug::trace_log_scalar("vevo2.fm.prompt_frames", prompt_len);
    engine::debug::trace_log_scalar("vevo2.fm.target_frames", target_len);
    engine::debug::timing_log_scalar("vevo2.fm.total_ms", engine::debug::elapsed_ms(total_start));
    return out;
}

}  // namespace engine::models::vevo2
