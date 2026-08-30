#include "engine/community_models/sopro_tts/semantic_encoder.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/audio/dsp.h"
#include "engine/framework/audio/resampling.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/scaled_dot_product_attention.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::community_models::sopro_tts {

struct SoproSemanticLayerWeights {
    engine::modules::NormWeights attn_norm;
    engine::modules::LinearWeights q_proj;
    engine::modules::LinearWeights k_proj;  // bias=False upstream
    engine::modules::LinearWeights v_proj;
    engine::modules::LinearWeights out_proj;
    engine::modules::NormWeights ffn_norm;
    engine::modules::LinearWeights fc1;
    engine::modules::LinearWeights fc2;
};

struct SoproSemanticEncoderWeights {
    std::shared_ptr<engine::core::BackendWeightStore> store;
    engine::modules::Conv1dWeights conv1;
    engine::modules::Conv1dWeights conv2;
    engine::core::TensorValue pos_emb;
    std::vector<SoproSemanticLayerWeights> layers;
    engine::modules::NormWeights final_norm;
    // Host-side quantiser head.
    std::vector<float> pre_head_norm_weight;
    std::vector<float> pre_head_norm_bias;
    std::vector<float> digit_head_weight;  // [digit_dim, d_model]
    std::vector<float> digit_head_bias;
    // torchaudio MelSpectrogram buffers.
    std::vector<float> analysis_window;
    std::vector<float> mel_filterbank;
};

namespace {

namespace binding = engine::modules::binding;

constexpr float kLayerNormEps = 1.0e-5F;  // torch.nn.LayerNorm default
constexpr int64_t kConvRightContextFrames = 2;  // semantic.CONV_RIGHT_CONTEXT_FRAMES

// SOPRO_DUMP_DIR: raw f32/i32 dumps for stage comparison against the reference.
template <typename T>
void dump(const std::string & name, const std::vector<T> & values) {
    const char * dir = std::getenv("SOPRO_DUMP_DIR");
    if (dir == nullptr) {
        return;
    }
    std::FILE * fh = std::fopen((std::string(dir) + "/" + name).c_str(), "wb");
    if (fh != nullptr) {
        std::fwrite(values.data(), sizeof(T), values.size(), fh);
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

std::shared_ptr<const SoproSemanticEncoderWeights> load_weights(
    ggml_backend_t backend,
    engine::core::BackendType backend_type,
    const engine::assets::TensorSource & source,
    const SoproSemanticEncoderConfig & config,
    size_t weight_context_bytes,
    engine::assets::TensorStorageType matmul_storage_type,
    engine::assets::TensorStorageType conv_storage_type) {
    auto weights = std::make_shared<SoproSemanticEncoderWeights>();
    require_frontend_buffers(
        source, "semantic encoder",
        {"frontend.mel.spectrogram.window",
         "frontend.mel.mel_scale.fb"});
    weights->store = std::make_shared<engine::core::BackendWeightStore>(
        backend, backend_type, "sopro_tts.semantic_encoder.weights", weight_context_bytes);
    auto & store = *weights->store;
    weights->conv1 = binding::conv1d_from_source(
        store, source, "conv1", conv_storage_type, config.d_model, config.n_mels, 3, true);
    weights->conv2 = binding::conv1d_from_source(
        store, source, "conv2", conv_storage_type, config.d_model, config.d_model, 3, true);
    weights->pos_emb = store.load_f32_tensor(
        source, "pos_emb", {config.max_positions, config.d_model});
    weights->layers.reserve(static_cast<size_t>(config.layers));
    for (int64_t layer = 0; layer < config.layers; ++layer) {
        const std::string prefix = "layers." + std::to_string(layer);
        SoproSemanticLayerWeights out;
        out.attn_norm = binding::norm_from_source(
            store, source, prefix + ".self_attn_layer_norm", config.d_model);
        out.q_proj = binding::linear_from_source(
            store, source, prefix + ".self_attn.q_proj", matmul_storage_type,
            config.d_model, config.d_model, true);
        out.k_proj = binding::linear_from_source(
            store, source, prefix + ".self_attn.k_proj", matmul_storage_type,
            config.d_model, config.d_model, false);
        out.v_proj = binding::linear_from_source(
            store, source, prefix + ".self_attn.v_proj", matmul_storage_type,
            config.d_model, config.d_model, true);
        out.out_proj = binding::linear_from_source(
            store, source, prefix + ".self_attn.out_proj", matmul_storage_type,
            config.d_model, config.d_model, true);
        out.ffn_norm = binding::norm_from_source(
            store, source, prefix + ".final_layer_norm", config.d_model);
        out.fc1 = binding::linear_from_source(
            store, source, prefix + ".fc1", matmul_storage_type, config.ffn_dim, config.d_model, true);
        out.fc2 = binding::linear_from_source(
            store, source, prefix + ".fc2", matmul_storage_type, config.d_model, config.ffn_dim, true);
        weights->layers.push_back(std::move(out));
    }
    weights->final_norm = binding::norm_from_source(store, source, "final_norm", config.d_model);
    weights->pre_head_norm_weight = source.require_f32("pre_head_norm.weight", {config.d_model});
    weights->pre_head_norm_bias = source.require_f32("pre_head_norm.bias", {config.d_model});
    weights->digit_head_weight = source.require_f32(
        "digit_head.weight", {config.digit_dim(), config.d_model});
    weights->digit_head_bias = source.require_f32("digit_head.bias", {config.digit_dim()});
    weights->analysis_window = source.require_f32(
        "frontend.mel.spectrogram.window", {config.n_fft});
    weights->mel_filterbank = source.require_f32(
        "frontend.mel.mel_scale.fb", {config.n_fft / 2 + 1, config.n_mels});
    store.upload();
    return weights;
}

}  // namespace

struct SoproSemanticEncoderGraph {
    SoproSemanticEncoderGraph(
        ggml_backend_t backend_in,
        engine::core::BackendType backend_type,
        size_t graph_context_bytes,
        const SoproSemanticEncoderConfig & config,
        std::shared_ptr<const SoproSemanticEncoderWeights> weights_in,
        int64_t mel_frames_in,
        int64_t keep_steps)
        : backend(backend_in),
          weights(std::move(weights_in)),
          mel_frames(mel_frames_in),
          steps(keep_steps),
          d_model(config.d_model) {
        if (backend == nullptr || weights == nullptr) {
            throw std::runtime_error("Sopro semantic encoder graph requires a backend and weights");
        }
        if (mel_frames <= 0 || steps <= 0) {
            throw std::runtime_error("Sopro semantic encoder graph requires positive lengths");
        }
        ggml_init_params params{graph_context_bytes, nullptr, true};
        ctx.reset(ggml_init(params));
        if (ctx == nullptr) {
            throw std::runtime_error("failed to initialize the Sopro semantic encoder graph context");
        }
        engine::core::ModuleBuildContext build_ctx{ctx.get(), "sopro_tts.semantic_encoder", backend_type};
        namespace mod = engine::modules;
        const auto shape = engine::core::TensorShape::from_dims({1, config.n_mels, mel_frames});
        input = engine::core::make_tensor(build_ctx, GGML_TYPE_F32, shape).tensor;
        ggml_set_input(input);

        auto hidden = mod::Conv1dModule({config.n_mels, d_model, 3, 1, 1, 1, true})
                          .build(build_ctx, engine::core::wrap_tensor(input, shape, GGML_TYPE_F32),
                                 weights->conv1);
        hidden = mod::GeluModule({mod::GeluApproximation::ExactErf}).build(build_ctx, hidden);
        hidden = mod::Conv1dModule({d_model, d_model, 3, 2, 1, 1, true})
                     .build(build_ctx, hidden, weights->conv2);
        hidden = mod::GeluModule({mod::GeluApproximation::ExactErf}).build(build_ctx, hidden);
        // [1, C, T] -> [1, T, C]
        hidden = mod::TransposeModule({{0, 2, 1, 3}, 3}).build(build_ctx, hidden);
        const int64_t conv_steps = hidden.shape.dims[1];
        if (conv_steps < steps) {
            throw std::runtime_error("Sopro semantic encoder produced fewer frames than expected");
        }
        {
            auto positions = mod::SliceModule({0, 0, conv_steps}).build(build_ctx, weights->pos_emb);
            positions = engine::core::reshape_tensor(
                build_ctx, engine::core::ensure_backend_addressable_layout(build_ctx, positions),
                engine::core::TensorShape::from_dims({1, conv_steps, d_model}));
            hidden = mod::AddModule{}.build(build_ctx, hidden, positions);
        }
        hidden = mod::SliceModule({1, 0, steps}).build(build_ctx, hidden);
        hidden = dense(build_ctx, hidden);

        const int64_t heads = config.heads;
        const int64_t head_dim = config.head_dim();
        for (const auto & layer : weights->layers) {
            auto norm = mod::LayerNormModule({d_model, kLayerNormEps, true, true})
                            .build(build_ctx, hidden, layer.attn_norm);
            auto q = mod::LinearModule({d_model, d_model, true, GGML_PREC_F32})
                         .build(build_ctx, norm, layer.q_proj);
            auto k = mod::LinearModule({d_model, d_model, false, GGML_PREC_F32})
                         .build(build_ctx, norm, layer.k_proj);
            auto v = mod::LinearModule({d_model, d_model, true, GGML_PREC_F32})
                         .build(build_ctx, norm, layer.v_proj);
            const auto head_shape = engine::core::TensorShape::from_dims({1, steps, heads, head_dim});
            auto to_heads = [&](const engine::core::TensorValue & value) {
                auto reshaped = engine::core::reshape_tensor(
                    build_ctx, engine::core::ensure_backend_addressable_layout(build_ctx, value),
                    head_shape);
                // Flash attention needs dense [1, H, T, DH] operands.
                return dense(build_ctx, mod::TransposeModule({{0, 2, 1, 3}, 4}).build(build_ctx, reshaped));
            };
            auto attn = mod::ScaledDotProductAttentionModule({
                head_dim,
                mod::ScaledDotProductAttentionLowering::Flash,
                GGML_PREC_F32,
                mod::AttentionCausality::NonCausal,
            }).build(build_ctx, to_heads(q), to_heads(k), to_heads(v));
            auto flat = engine::core::reshape_tensor(
                build_ctx, engine::core::ensure_backend_addressable_layout(build_ctx, attn),
                engine::core::TensorShape::from_dims({1, steps, d_model}));
            auto projected = mod::LinearModule({d_model, d_model, true, GGML_PREC_F32})
                                 .build(build_ctx, flat, layer.out_proj);
            hidden = mod::AddModule{}.build(build_ctx, hidden, projected);

            auto ffn = mod::LayerNormModule({d_model, kLayerNormEps, true, true})
                           .build(build_ctx, hidden, layer.ffn_norm);
            ffn = mod::LinearModule({d_model, config.ffn_dim, true, GGML_PREC_F32})
                      .build(build_ctx, ffn, layer.fc1);
            ffn = mod::GeluModule({mod::GeluApproximation::ExactErf}).build(build_ctx, ffn);
            ffn = mod::LinearModule({config.ffn_dim, d_model, true, GGML_PREC_F32})
                      .build(build_ctx, ffn, layer.fc2);
            hidden = mod::AddModule{}.build(build_ctx, hidden, ffn);
        }
        hidden = mod::LayerNormModule({d_model, kLayerNormEps, true, true})
                     .build(build_ctx, hidden, weights->final_norm);
        hidden = engine::core::ensure_backend_addressable_layout(build_ctx, hidden);
        output = hidden.tensor;
        ggml_set_output(output);
        graph = ggml_new_graph_custom(ctx.get(), 65536, false);
        ggml_build_forward_expand(graph, output);
        gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (gallocr == nullptr || !ggml_gallocr_reserve(gallocr, graph) ||
            !ggml_gallocr_alloc_graph(gallocr, graph)) {
            throw std::runtime_error("failed to allocate the Sopro semantic encoder graph");
        }
    }

    ~SoproSemanticEncoderGraph() {
        if (gallocr != nullptr) {
            ggml_gallocr_free(gallocr);
            gallocr = nullptr;
        }
    }

    bool matches(const SoproSemanticEncoderWeights & other, int64_t frames, int64_t keep) const noexcept {
        return weights.get() == &other && mel_frames == frames && steps == keep;
    }

    std::vector<float> run(const std::vector<float> & log_mel) {
        ggml_backend_tensor_set(input, log_mel.data(), 0, log_mel.size() * sizeof(float));
        const ggml_status status = engine::core::compute_backend_graph(backend, graph);
        ggml_backend_synchronize(backend);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Sopro semantic encoder graph compute failed");
        }
        std::vector<float> out(static_cast<size_t>(steps * d_model), 0.0F);
        ggml_backend_tensor_get(output, out.data(), 0, out.size() * sizeof(float));
        return out;
    }

    ggml_backend_t backend = nullptr;
    std::shared_ptr<const SoproSemanticEncoderWeights> weights;
    int64_t mel_frames = 0;
    int64_t steps = 0;
    int64_t d_model = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx;
    ggml_tensor * input = nullptr;
    ggml_tensor * output = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_gallocr_t gallocr = nullptr;
};

SoproSemanticEncoderRuntime::SoproSemanticEncoderRuntime(
    const SoproTTSAssets & assets,
    engine::core::ExecutionContext & execution_context,
    size_t weight_context_bytes,
    size_t graph_context_bytes,
    engine::assets::TensorStorageType matmul_storage_type,
    engine::assets::TensorStorageType conv_storage_type)
    : config_(assets.config.semantic_encoder),
      execution_context_(execution_context),
      graph_context_bytes_(graph_context_bytes),
      weights_(load_weights(
          execution_context.backend(),
          execution_context.backend_type(),
          *assets.semantic_encoder_weights,
          assets.config.semantic_encoder,
          weight_context_bytes,
          matmul_storage_type,
          conv_storage_type)) {}

SoproSemanticEncoderRuntime::~SoproSemanticEncoderRuntime() = default;

std::vector<int32_t> SoproSemanticEncoderRuntime::encode(const std::vector<float> & audio24) const {
    if (audio24.empty()) {
        throw std::runtime_error("Sopro semantic encoder requires a non-empty reference waveform");
    }
    const auto n24 = static_cast<int64_t>(audio24.size());
    const int64_t token_samples = config_.token_samples_24k;
    const int64_t n_tokens = (n24 + token_samples - 1) / token_samples;

    auto audio16 = engine::audio::resample_mono_torchaudio_sinc_hann(
        audio24, 24000, static_cast<int>(config_.sample_rate));
    // SemanticEncoder.encode pins the resampled length so the token grid is
    // exactly reproducible regardless of the resampler's tail behaviour.
    const int64_t n16 = (n24 * 2 + 2) / 3;
    audio16.resize(static_cast<size_t>(n16), 0.0F);

    dump("sem_wav16.f32", audio16);
    const int64_t frames = (n16 + config_.hop_length - 1) / config_.hop_length;
    const int64_t mel_frames = frames + kConvRightContextFrames;
    // WhisperMelFrontend right-pads n_fft zeros before the centred STFT.
    audio16.resize(static_cast<size_t>(n16 + config_.n_fft), 0.0F);

    const int64_t freq_bins = config_.n_fft / 2 + 1;
    engine::audio::STFTConfig stft;
    stft.n_fft = config_.n_fft;
    stft.hop_length = config_.hop_length;
    stft.win_length = config_.n_fft;
    stft.center = true;
    stft.pad_mode = engine::audio::STFTPadMode::Reflect;
    const auto magnitude = engine::audio::STFT{}.compute_magnitude(
        audio16, weights_->analysis_window, 1, static_cast<int64_t>(audio16.size()), stft,
        static_cast<size_t>(execution_context_.config().threads));
    if (magnitude.shape.size() != 3 || magnitude.shape[1] != freq_bins) {
        throw std::runtime_error("Sopro semantic encoder STFT produced an unexpected layout");
    }
    const int64_t stft_frames = magnitude.shape[2];
    if (stft_frames < mel_frames) {
        throw std::runtime_error("Sopro semantic encoder reference audio is too short");
    }

    std::vector<float> mel(static_cast<size_t>(config_.n_mels * mel_frames), 0.0F);
    for (int64_t f = 0; f < freq_bins; ++f) {
        const float * fb_row = weights_->mel_filterbank.data() + static_cast<size_t>(f * config_.n_mels);
        const float * spec_row = magnitude.values.data() + static_cast<size_t>(f * stft_frames);
        for (int64_t m = 0; m < config_.n_mels; ++m) {
            const float weight = fb_row[m];
            if (weight == 0.0F) {
                continue;
            }
            float * out_row = mel.data() + static_cast<size_t>(m * mel_frames);
            for (int64_t t = 0; t < mel_frames; ++t) {
                out_row[t] += weight * spec_row[t] * spec_row[t];  // power = 2.0
            }
        }
    }
    float peak = -std::numeric_limits<float>::infinity();
    for (auto & value : mel) {
        value = std::log10(std::max(value, 1.0e-10F));
        peak = std::max(peak, value);
    }
    const float floor_value = peak - 8.0F;
    for (auto & value : mel) {
        value = (std::max(value, floor_value) + 4.0F) / 4.0F;
    }

    dump("sem_mel.f32", mel);
    const int64_t steps = (frames + 1) / 2;  // n50
    if (steps <= 0) {
        throw std::runtime_error("Sopro semantic encoder reference audio is too short");
    }
    if (steps > config_.max_positions) {
        throw std::runtime_error(
            "Sopro semantic encoder reference audio exceeds the positional embedding table");
    }
    if (graph_ == nullptr || !graph_->matches(*weights_, mel_frames, steps)) {
        graph_ = std::make_unique<SoproSemanticEncoderGraph>(
            execution_context_.backend(),
            execution_context_.backend_type(),
            graph_context_bytes_,
            config_,
            weights_,
            mel_frames,
            steps);
    }
    const auto hidden = graph_->run(mel);
    dump("sem_hidden.f32", hidden);

    // SemanticEncoder._interpolate: half-pixel aligned linear resampling from
    // `steps` encoder frames onto the `n_tokens` output grid.
    const int64_t d_model = config_.d_model;
    const int64_t digit_dim = config_.digit_dim();
    std::vector<int32_t> tokens(static_cast<size_t>(n_tokens), 0);
    std::vector<float> frame(static_cast<size_t>(d_model), 0.0F);
    std::vector<float> logits(static_cast<size_t>(digit_dim), 0.0F);
    const float ratio = static_cast<float>(steps) / static_cast<float>(n_tokens);
    for (int64_t t = 0; t < n_tokens; ++t) {
        float source = (static_cast<float>(t) + 0.5F) * ratio - 0.5F;
        source = std::min(std::max(source, 0.0F), static_cast<float>(steps - 1));
        const auto left = static_cast<int64_t>(std::floor(source));
        const int64_t right = std::min(left + 1, steps - 1);
        const float weight = source - static_cast<float>(left);
        const float * left_row = hidden.data() + static_cast<size_t>(left * d_model);
        const float * right_row = hidden.data() + static_cast<size_t>(right * d_model);
        for (int64_t c = 0; c < d_model; ++c) {
            frame[static_cast<size_t>(c)] = left_row[c] * (1.0F - weight) + right_row[c] * weight;
        }
        // pre_head_norm
        double sum = 0.0;
        for (int64_t c = 0; c < d_model; ++c) {
            sum += frame[static_cast<size_t>(c)];
        }
        const double mean = sum / static_cast<double>(d_model);
        double variance = 0.0;
        for (int64_t c = 0; c < d_model; ++c) {
            const double centred = frame[static_cast<size_t>(c)] - mean;
            variance += centred * centred;
        }
        variance /= static_cast<double>(d_model);
        const double inv_std = 1.0 / std::sqrt(variance + kLayerNormEps);
        for (int64_t c = 0; c < d_model; ++c) {
            frame[static_cast<size_t>(c)] = static_cast<float>(
                (frame[static_cast<size_t>(c)] - mean) * inv_std *
                    weights_->pre_head_norm_weight[static_cast<size_t>(c)] +
                weights_->pre_head_norm_bias[static_cast<size_t>(c)]);
        }
        for (int64_t d = 0; d < digit_dim; ++d) {
            double value = weights_->digit_head_bias[static_cast<size_t>(d)];
            const float * row = weights_->digit_head_weight.data() + static_cast<size_t>(d * d_model);
            for (int64_t c = 0; c < d_model; ++c) {
                value += static_cast<double>(row[c]) * static_cast<double>(frame[static_cast<size_t>(c)]);
            }
            logits[static_cast<size_t>(d)] = static_cast<float>(value);
        }
        // Finite scalar quantiser: per-level arg-max, packed with mixed radix.
        int64_t token = 0;
        int64_t base = 1;
        int64_t offset = 0;
        for (const int64_t level : config_.fsq_levels) {
            int64_t best = 0;
            float best_value = logits[static_cast<size_t>(offset)];
            for (int64_t i = 1; i < level; ++i) {
                const float value = logits[static_cast<size_t>(offset + i)];
                if (value > best_value) {
                    best_value = value;
                    best = i;
                }
            }
            token += best * base;
            base *= level;
            offset += level;
        }
        tokens[static_cast<size_t>(t)] = static_cast<int32_t>(token);
    }
    dump("sem_tokens.i32", tokens);
    return tokens;
}

}  // namespace engine::community_models::sopro_tts
