#include "engine/community_models/sopro_tts/vocoder.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/audio/dsp.h"
#include "engine/framework/audio/fft.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/streaming_conv_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::community_models::sopro_tts {

struct SoproVocoderConvNeXtWeights {
    engine::modules::DepthwiseConv1dWeights dwconv;
    engine::modules::NormWeights norm;
    engine::modules::LinearWeights pwconv1;
    engine::modules::LinearWeights pwconv2;
    engine::core::TensorValue gamma;
};

struct SoproVocosWeights {
    std::shared_ptr<engine::core::BackendWeightStore> store;
    engine::modules::Conv1dWeights embed;
    engine::modules::NormWeights norm;
    std::vector<SoproVocoderConvNeXtWeights> convnext;
    engine::modules::NormWeights final_norm;
    engine::modules::LinearWeights head_out;
    std::vector<float> istft_window;      // head.istft.window, n_fft taps
    std::vector<float> analysis_window;   // MelSpectrogram STFT window
    std::vector<float> mel_filterbank;    // [freq_bins, n_mels], row-major
};

namespace {

namespace binding = engine::modules::binding;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

// Multiply the last (channel) dimension of a channel-last tensor by a vector.
engine::core::TensorValue scale_last_dim(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    const engine::core::TensorValue & scale) {
    const auto view = engine::core::reshape_tensor(
        ctx, scale, engine::core::TensorShape::from_dims({1, 1, scale.shape.dims[0]}));
    const auto repeated = engine::modules::RepeatModule({input.shape}).build(ctx, view);
    return engine::modules::MulModule{}.build(ctx, input, repeated);
}

engine::modules::TransposeConfig swap_channel_time() {
    return engine::modules::TransposeConfig{{0, 2, 1, 3}, 3};
}

constexpr int64_t kVocoderKernel = 7;

struct TimePadding {
    int left = 0;
    int right = 0;
};

// sopro/vocoder.py _conv_causal: a causal conv sees kernel - 1 - lookahead past
// frames and `lookahead` future ones; a non-causal conv is centred.
TimePadding conv_padding(const SoproVocoderConfig & config, int64_t lookahead) {
    const int64_t context = kVocoderKernel - 1;
    if (!config.causal) {
        return {static_cast<int>(context / 2), static_cast<int>(context / 2)};
    }
    if (lookahead < 0 || lookahead > context) {
        throw std::runtime_error("Sopro vocoder lookahead must be between 0 and kernel_size - 1");
    }
    return {static_cast<int>(context - lookahead), static_cast<int>(lookahead)};
}

int64_t block_lookahead(const SoproVocoderConfig & config, int64_t layer) {
    if (!config.causal || config.block_lookaheads.empty()) {
        return 0;
    }
    return config.block_lookaheads[static_cast<size_t>(layer)];
}

// Zero-pad the time axis of a [B, C, T] tensor.
engine::core::TensorValue pad_time(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & value,
    int left,
    int right) {
    auto contiguous = engine::core::ensure_backend_addressable_layout(ctx, value);
    auto shape = contiguous.shape;
    shape.dims[2] += left + right;
    return engine::core::wrap_tensor(
        ggml_pad_ext(ctx.ggml, contiguous.tensor, left, right, 0, 0, 0, 0, 0, 0),
        shape,
        GGML_TYPE_F32);
}

std::shared_ptr<const SoproVocosWeights> load_vocoder_weights(
    ggml_backend_t backend,
    engine::core::BackendType backend_type,
    const engine::assets::TensorSource & source,
    const SoproVocoderConfig & config,
    size_t weight_context_bytes,
    engine::assets::TensorStorageType matmul_storage_type,
    engine::assets::TensorStorageType conv_storage_type) {
    auto weights = std::make_shared<SoproVocosWeights>();
    require_frontend_buffers(
        source, "vocoder",
        {"feature_extractor.mel_spec.spectrogram.window",
         "feature_extractor.mel_spec.mel_scale.fb",
         "head.istft.window"});
    weights->store = std::make_shared<engine::core::BackendWeightStore>(
        backend, backend_type, "sopro_tts.vocoder.weights", weight_context_bytes);
    weights->embed = binding::conv1d_from_source(
        *weights->store, source, "backbone.embed", conv_storage_type,
        config.dim, config.n_mels, 7, true);
    weights->norm = binding::norm_from_source(
        *weights->store, source, "backbone.norm", config.dim);
    weights->convnext.reserve(static_cast<size_t>(config.num_layers));
    for (int64_t layer = 0; layer < config.num_layers; ++layer) {
        const std::string prefix = "backbone.convnext." + std::to_string(layer);
        SoproVocoderConvNeXtWeights block;
        block.dwconv = binding::depthwise_conv1d_from_source(
            *weights->store, source, prefix + ".dwconv", conv_storage_type, config.dim, 7, true);
        block.norm = binding::norm_from_source(
            *weights->store, source, prefix + ".norm", config.dim);
        block.pwconv1 = binding::linear_from_source(
            *weights->store, source, prefix + ".pwconv1", matmul_storage_type,
            config.intermediate_dim, config.dim, true);
        block.pwconv2 = binding::linear_from_source(
            *weights->store, source, prefix + ".pwconv2", matmul_storage_type,
            config.dim, config.intermediate_dim, true);
        block.gamma = weights->store->load_f32_tensor(source, prefix + ".gamma", {config.dim});
        weights->convnext.push_back(std::move(block));
    }
    weights->final_norm = binding::norm_from_source(
        *weights->store, source, "backbone.final_layer_norm", config.dim);
    weights->head_out = binding::linear_from_source(
        *weights->store, source, "head.out", matmul_storage_type,
        config.n_fft + 2, config.dim, true);
    weights->istft_window = source.require_f32("head.istft.window", {config.n_fft});
    // torchaudio MelSpectrogram keeps both of these as persistent buffers, so
    // the analysis filterbank is byte-identical to the reference pipeline's.
    weights->analysis_window = source.require_f32(
        "feature_extractor.mel_spec.spectrogram.window", {config.n_fft});
    weights->mel_filterbank = source.require_f32(
        "feature_extractor.mel_spec.mel_scale.fb", {config.n_fft / 2 + 1, config.n_mels});
    weights->store->upload();
    return weights;
}

engine::core::TensorValue build_convnext_block(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input_bct,
    const SoproVocoderConvNeXtWeights & weights,
    const SoproVocoderConfig & config,
    int64_t layer) {
    const auto padding = conv_padding(config, block_lookahead(config, layer));
    const bool centred = padding.left == padding.right;
    auto hidden = engine::modules::DepthwiseConv1dModule({
        config.dim, kVocoderKernel, 1, centred ? padding.left : 0, 1, weights.dwconv.bias.has_value(),
    }).build(ctx, centred ? input_bct : pad_time(ctx, input_bct, padding.left, padding.right), weights.dwconv);
    hidden = engine::modules::TransposeModule(swap_channel_time()).build(ctx, hidden);
    hidden = engine::modules::LayerNormModule({config.dim, 1.0e-6F, true, true})
                 .build(ctx, hidden, weights.norm);
    hidden = engine::modules::LinearModule({config.dim, config.intermediate_dim, true, GGML_PREC_F32})
                 .build(ctx, hidden, weights.pwconv1);
    hidden = engine::modules::GeluModule({engine::modules::GeluApproximation::ExactErf}).build(ctx, hidden);
    hidden = engine::modules::LinearModule({config.intermediate_dim, config.dim, true, GGML_PREC_F32})
                 .build(ctx, hidden, weights.pwconv2);
    hidden = scale_last_dim(ctx, hidden, weights.gamma);
    hidden = engine::modules::TransposeModule(swap_channel_time()).build(ctx, hidden);
    return engine::modules::AddModule{}.build(ctx, input_bct, hidden);
}

engine::core::TensorValue build_vocoder_head(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & mel_bct,
    const SoproVocosWeights & weights,
    const SoproVocoderConfig & config) {
    const auto padding = conv_padding(config, config.lookahead_frames);
    const bool centred = padding.left == padding.right;
    auto hidden = engine::modules::Conv1dModule({
        config.n_mels, config.dim, kVocoderKernel, 1, centred ? padding.left : 0, 1, weights.embed.bias.has_value(),
    }).build(ctx, centred ? mel_bct : pad_time(ctx, mel_bct, padding.left, padding.right), weights.embed);
    hidden = engine::modules::TransposeModule(swap_channel_time()).build(ctx, hidden);
    hidden = engine::modules::LayerNormModule({config.dim, 1.0e-6F, true, true})
                 .build(ctx, hidden, weights.norm);
    hidden = engine::modules::TransposeModule(swap_channel_time()).build(ctx, hidden);
    for (size_t layer = 0; layer < weights.convnext.size(); ++layer) {
        hidden = build_convnext_block(ctx, hidden, weights.convnext[layer], config, static_cast<int64_t>(layer));
    }
    hidden = engine::modules::TransposeModule(swap_channel_time()).build(ctx, hidden);
    hidden = engine::modules::LayerNormModule({config.dim, 1.0e-6F, true, true})
                 .build(ctx, hidden, weights.final_norm);
    return engine::modules::LinearModule({config.dim, config.n_fft + 2, true, GGML_PREC_F32})
        .build(ctx, hidden, weights.head_out);
}

// ISTFTHead.spectrogram, then torch.fft.irfft(norm="backward") (which scales
// by 1/n_fft) times the synthesis window: one windowed frame per head row,
// [frames, n_fft].
std::vector<float> synthesis_frames(
    const float * head,
    int64_t frames,
    const SoproVocoderConfig & config,
    const std::vector<float> & window,
    size_t threads) {
    const int64_t freq_bins = config.n_fft / 2 + 1;
    const int64_t out_dim = config.n_fft + 2;
    if (static_cast<int64_t>(window.size()) != config.n_fft) {
        throw std::runtime_error("Sopro vocoder ISTFT window shape mismatch");
    }
    std::vector<float> framed(static_cast<size_t>(frames * config.n_fft), 0.0F);
    if (frames <= 0) {
        return framed;
    }
    const float log_max_magnitude = std::log(config.max_magnitude);
    std::vector<std::complex<float>> spectrum(static_cast<size_t>(frames * freq_bins));
    const int omp_threads = static_cast<int>(std::max<size_t>(1, threads));
#ifdef _OPENMP
#pragma omp parallel for num_threads(omp_threads) if (frames >= 8)
#endif
    for (int64_t frame = 0; frame < frames; ++frame) {
        const float * row = head + static_cast<size_t>(frame * out_dim);
        for (int64_t freq = 0; freq < freq_bins; ++freq) {
            const float magnitude = std::exp(std::min(row[freq], log_max_magnitude));
            const float phase = row[freq_bins + freq];
            spectrum[static_cast<size_t>(frame * freq_bins + freq)] = {
                magnitude * std::cos(phase), magnitude * std::sin(phase)};
        }
    }
    engine::audio::real_fft_inverse(
        {static_cast<size_t>(frames), static_cast<size_t>(config.n_fft)},
        {
            static_cast<std::ptrdiff_t>(freq_bins * static_cast<int64_t>(sizeof(std::complex<float>))),
            static_cast<std::ptrdiff_t>(sizeof(std::complex<float>)),
        },
        {
            static_cast<std::ptrdiff_t>(config.n_fft * static_cast<int64_t>(sizeof(float))),
            static_cast<std::ptrdiff_t>(sizeof(float)),
        },
        1, spectrum.data(), framed.data(),
        1.0F / static_cast<float>(config.n_fft), threads);
    for (int64_t frame = 0; frame < frames; ++frame) {
        float * row = framed.data() + static_cast<size_t>(frame * config.n_fft);
        for (int64_t i = 0; i < config.n_fft; ++i) {
            row[i] *= window[static_cast<size_t>(i)];
        }
    }
    return framed;
}

}  // namespace

struct SoproVocosGraph {
    SoproVocosGraph(
        ggml_backend_t backend_in,
        engine::core::BackendType backend_type,
        size_t graph_context_bytes,
        const SoproVocoderConfig & config_in,
        std::shared_ptr<const SoproVocosWeights> weights_in,
        int64_t frames_in)
        : backend(backend_in),
          weights(std::move(weights_in)),
          frames(frames_in),
          head_dim(config_in.n_fft + 2),
          config(&config_in) {
        if (backend == nullptr || weights == nullptr) {
            throw std::runtime_error("Sopro vocoder graph requires a backend and weights");
        }
        if (frames <= 0) {
            throw std::runtime_error("Sopro vocoder graph requires a positive frame count");
        }
        ggml_init_params params{graph_context_bytes, nullptr, true};
        ctx.reset(ggml_init(params));
        if (ctx == nullptr) {
            throw std::runtime_error("failed to initialize the Sopro vocoder graph context");
        }
        engine::core::ModuleBuildContext build_ctx{ctx.get(), "sopro_tts.vocoder", backend_type};
        const auto shape = engine::core::TensorShape::from_dims({1, config_in.n_mels, frames});
        input = engine::core::make_tensor(build_ctx, GGML_TYPE_F32, shape).tensor;
        ggml_set_input(input);
        auto head = build_vocoder_head(
            build_ctx, engine::core::wrap_tensor(input, shape, GGML_TYPE_F32), *weights, config_in);
        head = engine::core::ensure_backend_addressable_layout(build_ctx, head);
        output = head.tensor;
        ggml_set_output(output);
        graph = ggml_new_graph_custom(ctx.get(), 65536, false);
        ggml_build_forward_expand(graph, output);
        gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (gallocr == nullptr || !ggml_gallocr_reserve(gallocr, graph) ||
            !ggml_gallocr_alloc_graph(gallocr, graph)) {
            throw std::runtime_error("failed to allocate the Sopro vocoder graph");
        }
    }

    ~SoproVocosGraph() {
        if (gallocr != nullptr) {
            ggml_gallocr_free(gallocr);
            gallocr = nullptr;
        }
    }

    bool matches(const SoproVocosWeights & other, int64_t other_frames) const noexcept {
        return weights.get() == &other && frames == other_frames;
    }

    // Head output, one row of n_fft + 2 values per frame.
    std::vector<float> run(const std::vector<float> & mel) {
        ggml_backend_tensor_set(input, mel.data(), 0, mel.size() * sizeof(float));
        const ggml_status status = engine::core::compute_backend_graph(backend, graph);
        ggml_backend_synchronize(backend);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Sopro vocoder graph compute failed");
        }
        std::vector<float> head(static_cast<size_t>(frames * head_dim), 0.0F);
        ggml_backend_tensor_get(output, head.data(), 0, head.size() * sizeof(float));
        return head;
    }

    ggml_backend_t backend = nullptr;
    std::shared_ptr<const SoproVocosWeights> weights;
    int64_t frames = 0;
    int64_t head_dim = 0;
    const SoproVocoderConfig * config = nullptr;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx;
    ggml_tensor * input = nullptr;
    ggml_tensor * output = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_gallocr_t gallocr = nullptr;
};

SoproVocosRuntime::SoproVocosRuntime(
    const SoproTTSAssets & assets,
    engine::core::ExecutionContext & execution_context,
    size_t weight_context_bytes,
    size_t graph_context_bytes,
    engine::assets::TensorStorageType matmul_storage_type,
    engine::assets::TensorStorageType conv_storage_type)
    : config_(assets.config.vocoder),
      execution_context_(execution_context),
      graph_context_bytes_(graph_context_bytes),
      weights_(load_vocoder_weights(
          execution_context.backend(),
          execution_context.backend_type(),
          *assets.vocoder_weights,
          assets.config.vocoder,
          weight_context_bytes,
          matmul_storage_type,
          conv_storage_type)) {}

SoproVocosRuntime::~SoproVocosRuntime() = default;

std::vector<float> SoproVocosRuntime::head(const std::vector<float> & mel, int64_t frames) const {
    if (frames <= 0 || static_cast<int64_t>(mel.size()) != frames * config_.n_mels) {
        throw std::runtime_error("Sopro vocoder requires a [n_mels, frames] input");
    }
    if (graph_ == nullptr || !graph_->matches(*weights_, frames)) {
        // Free the previous arena first; otherwise both are resident while the
        // replacement is allocated, and every segment rebuilds this graph.
        graph_.reset();
        graph_ = std::make_unique<SoproVocosGraph>(
            execution_context_.backend(),
            execution_context_.backend_type(),
            graph_context_bytes_,
            config_,
            weights_,
            frames);
    }
    return graph_->run(mel);
}

// ISTFT.forward is the streaming overlap-add fed every frame at once: the same
// sums in the same order, and its envelope never falls near the clamp inside
// the trimmed span.
std::vector<float> SoproVocosRuntime::decode(const std::vector<float> & mel, int64_t frames) const {
    if (frames < 2) {
        throw std::runtime_error("Sopro vocoder requires at least two mel frames");
    }
    return SoproVocoderStream(*this).push(mel, frames, true);
}

std::vector<float> SoproVocosRuntime::synthesis_frames(const float * head, int64_t frames) const {
    return sopro_tts::synthesis_frames(
        head, frames, config_, weights_->istft_window, static_cast<size_t>(execution_context_.config().threads));
}

const std::vector<float> & SoproVocosRuntime::synthesis_window() const noexcept {
    return weights_->istft_window;
}

int64_t SoproVocosRuntime::context_frames(bool right) const {
    int64_t total = 0;
    for (int64_t layer = -1; layer < config_.num_layers; ++layer) {
        const auto padding = conv_padding(config_, layer < 0 ? config_.lookahead_frames : block_lookahead(config_, layer));
        total += right ? padding.right : padding.left;
    }
    return total;
}

SoproVocoderStream::SoproVocoderStream(const SoproVocosRuntime & vocoder)
    : vocoder_(vocoder),
      left_(vocoder.context_frames(false)),
      right_(vocoder.context_frames(true)) {}

std::vector<float> SoproVocoderStream::push(const std::vector<float> & mel, int64_t frames, bool flush) {
    const int64_t n_mels = vocoder_.n_mels();
    if (frames < 0 || static_cast<int64_t>(mel.size()) != frames * n_mels) {
        throw std::runtime_error("Sopro streaming vocoder requires a [n_mels, frames] input");
    }
    for (int64_t t = 0; t < frames; ++t) {
        for (int64_t c = 0; c < n_mels; ++c) {
            history_.push_back(mel[static_cast<size_t>(c * frames + t)]);
        }
    }
    received_ += frames;

    // Backbone frames whose receptive field is complete, recomputed over a
    // window that carries that field on both sides.
    const int64_t ready = flush ? received_ : received_ - right_;
    std::vector<float> framed;
    int64_t new_frames = 0;
    if (ready > decoded_) {
        const int64_t start = std::max<int64_t>(0, decoded_ - left_);
        const int64_t window = received_ - start;
        std::vector<float> window_mel(static_cast<size_t>(n_mels * window));
        for (int64_t t = 0; t < window; ++t) {
            const float * row = history_.data() + static_cast<size_t>((start - history_start_ + t) * n_mels);
            for (int64_t c = 0; c < n_mels; ++c) {
                window_mel[static_cast<size_t>(c * window + t)] = row[c];
            }
        }
        const auto rows = vocoder_.head(window_mel, window);
        new_frames = ready - decoded_;
        framed = vocoder_.synthesis_frames(
            rows.data() + static_cast<size_t>((decoded_ - start) * (vocoder_.n_fft() + 2)), new_frames);
        decoded_ = ready;
        const int64_t keep_from = std::max<int64_t>(0, decoded_ - left_);
        history_.erase(history_.begin(), history_.begin() + static_cast<ptrdiff_t>((keep_from - history_start_) * n_mels));
        history_start_ = keep_from;
    }
    auto audio = overlap_add(framed, new_frames, flush);
    if (flush) {
        history_.clear();
        history_start_ = received_ = decoded_ = 0;
        processed_ = emitted_ = tail_start_ = 0;
        ola_.clear();
        envelope_.clear();
    }
    return audio;
}

// ISTFT.forward_stream: accumulate each frame's windowed signal and squared
// window, then emit every sample past the centre padding whose overlap is
// complete (everything that is left on flush).
std::vector<float> SoproVocoderStream::overlap_add(const std::vector<float> & framed, int64_t frames, bool flush) {
    const int64_t n_fft = vocoder_.n_fft();
    const int64_t hop = vocoder_.hop_length();
    const int64_t pad = n_fft / 2;
    const auto & window = vocoder_.synthesis_window();
    const int64_t offset = processed_ * hop - tail_start_;
    const int64_t required = frames > 0 ? offset + (frames - 1) * hop + n_fft : 0;
    if (static_cast<int64_t>(ola_.size()) < required) {
        ola_.resize(static_cast<size_t>(required), 0.0F);
        envelope_.resize(static_cast<size_t>(required), 0.0F);
    }
    for (int64_t frame = 0; frame < frames; ++frame) {
        const float * src = framed.data() + static_cast<size_t>(frame * n_fft);
        const int64_t base = offset + frame * hop;
        for (int64_t i = 0; i < n_fft; ++i) {
            const float w = window[static_cast<size_t>(i)];
            ola_[static_cast<size_t>(base + i)] += src[i];
            envelope_[static_cast<size_t>(base + i)] += w * w;
        }
    }
    processed_ += frames;
    int64_t target = std::max<int64_t>(0, processed_ * hop - pad);
    if (flush && processed_ > 0) {
        target = std::max(target, (processed_ - 1) * hop + n_fft - 2 * pad);
    }
    const int64_t count = std::max<int64_t>(0, target - emitted_);
    const int64_t first = emitted_ + pad - tail_start_;
    std::vector<float> out(static_cast<size_t>(count));
    for (int64_t i = 0; i < count; ++i) {
        const auto index = static_cast<size_t>(first + i);
        out[static_cast<size_t>(i)] = ola_[index] / std::max(envelope_[index], 1.0e-8F);
    }
    emitted_ = target;
    const int64_t trim = std::min<int64_t>(emitted_ + pad - tail_start_, static_cast<int64_t>(ola_.size()));
    ola_.erase(ola_.begin(), ola_.begin() + static_cast<ptrdiff_t>(trim));
    envelope_.erase(envelope_.begin(), envelope_.begin() + static_cast<ptrdiff_t>(trim));
    tail_start_ = emitted_ + pad;
    return out;
}

std::vector<float> SoproVocosRuntime::log_mel(const std::vector<float> & audio) const {
    if (audio.empty()) {
        throw std::runtime_error("Sopro vocoder mel extraction requires a non-empty waveform");
    }
    const int64_t freq_bins = config_.n_fft / 2 + 1;
    engine::audio::STFTConfig stft;
    stft.n_fft = config_.n_fft;
    stft.hop_length = config_.hop_length;
    stft.win_length = config_.n_fft;
    stft.center = true;
    stft.pad_mode = engine::audio::STFTPadMode::Reflect;
    const auto magnitude = engine::audio::STFT{}.compute_magnitude(
        audio, weights_->analysis_window, 1, static_cast<int64_t>(audio.size()), stft,
        static_cast<size_t>(execution_context_.config().threads));
    if (magnitude.shape.size() != 3 || magnitude.shape[1] != freq_bins) {
        throw std::runtime_error("Sopro vocoder STFT produced an unexpected layout");
    }
    const int64_t frames = magnitude.shape[2];
    std::vector<float> mel(static_cast<size_t>(config_.n_mels * frames), 0.0F);
    // MelScale: mel[m, t] = sum_f magnitude[f, t] * fb[f, m], then log-clamped.
    for (int64_t f = 0; f < freq_bins; ++f) {
        const float * fb_row = weights_->mel_filterbank.data() + static_cast<size_t>(f * config_.n_mels);
        const float * spec_row = magnitude.values.data() + static_cast<size_t>(f * frames);
        for (int64_t m = 0; m < config_.n_mels; ++m) {
            const float weight = fb_row[m];
            if (weight == 0.0F) {
                continue;
            }
            float * out_row = mel.data() + static_cast<size_t>(m * frames);
            for (int64_t t = 0; t < frames; ++t) {
                out_row[t] += weight * spec_row[t];
            }
        }
    }
    for (auto & value : mel) {
        value = std::log(std::max(value, 1.0e-7F));
    }
    return mel;
}

int64_t SoproVocosRuntime::mel_frames(int64_t samples) const noexcept {
    return samples / config_.hop_length + 1;  // centred STFT
}

int64_t SoproVocosRuntime::hop_length() const noexcept { return config_.hop_length; }
int64_t SoproVocosRuntime::n_fft() const noexcept { return config_.n_fft; }
int64_t SoproVocosRuntime::n_mels() const noexcept { return config_.n_mels; }

}  // namespace engine::community_models::sopro_tts
