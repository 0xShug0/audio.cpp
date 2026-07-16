#include "engine/models/vevo2/vocoder.h"

#include "engine/framework/audio/fft.h"
#include "engine/framework/assets/tensor_source.h"
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
#include <stdexcept>
#include <utility>

namespace engine::models::vevo2 {

struct Vevo2VocoderConvNeXtBlockWeights {
    engine::modules::DepthwiseConv1dWeights dwconv;
    engine::modules::NormWeights norm;
    engine::modules::LinearWeights pwconv1;
    engine::modules::LinearWeights pwconv2;
    engine::core::TensorValue gamma;
};

struct Vevo2VocoderWeights {
    std::shared_ptr<engine::core::BackendWeightStore> store;
    engine::modules::Conv1dWeights embed;
    engine::modules::NormWeights norm;
    std::vector<Vevo2VocoderConvNeXtBlockWeights> convnext;
    engine::modules::NormWeights final_norm;
    engine::modules::LinearWeights head_out;
    std::vector<float> istft_window;
};

namespace {

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

engine::core::TensorValue ensure_contiguous(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & value) {
    return engine::core::ensure_backend_addressable_layout(ctx, value);
}

engine::core::TensorValue transpose_bct_to_btc(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input) {
    return engine::modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, input);
}

engine::core::TensorValue transpose_btc_to_bct(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input) {
    return engine::modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, input);
}

engine::core::TensorValue scale_last_dim(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    const engine::core::TensorValue & scale) {
    const auto view = engine::core::reshape_tensor(
        ctx,
        scale,
        engine::core::TensorShape::from_dims({1, 1, scale.shape.dims[0]}));
    const auto repeated = engine::core::wrap_tensor(
        ggml_repeat(ctx.ggml, view.tensor, input.tensor),
        input.shape,
        GGML_TYPE_F32);
    return engine::core::wrap_tensor(
        ggml_mul(ctx.ggml, input.tensor, repeated.tensor),
        input.shape,
        GGML_TYPE_F32);
}

std::shared_ptr<const Vevo2VocoderWeights> load_vocoder_weights(
    ggml_backend_t backend,
    engine::core::BackendType backend_type,
    const engine::assets::TensorSource & source,
    const Vevo2VocoderConfig & config,
    size_t weight_context_bytes,
    engine::assets::TensorStorageType matmul_storage_type,
    engine::assets::TensorStorageType conv_storage_type) {
    auto weights = std::make_shared<Vevo2VocoderWeights>();
    weights->store = std::make_shared<engine::core::BackendWeightStore>(
        backend,
        backend_type,
        "vevo2.vocoder.weights",
        weight_context_bytes);
    weights->embed = engine::modules::binding::conv1d_from_source(
        *weights->store,
        source,
        "backbone.embed",
        conv_storage_type,
        config.dim,
        config.input_channels,
        7,
        true);
    weights->norm = engine::modules::binding::norm_from_source(*weights->store, source, "backbone.norm", config.dim);
    weights->convnext.reserve(static_cast<size_t>(config.num_layers));
    for (int64_t layer = 0; layer < config.num_layers; ++layer) {
        const std::string prefix = "backbone.convnext." + std::to_string(layer);
        Vevo2VocoderConvNeXtBlockWeights block;
        block.dwconv = engine::modules::binding::depthwise_conv1d_from_source(
            *weights->store,
            source,
            prefix + ".dwconv",
            conv_storage_type,
            config.dim,
            7,
            true);
        block.norm = engine::modules::binding::norm_from_source(*weights->store, source, prefix + ".norm", config.dim);
        block.pwconv1 = engine::modules::binding::linear_from_source(
            *weights->store,
            source,
            prefix + ".pwconv1",
            matmul_storage_type,
            config.intermediate_dim,
            config.dim,
            true);
        block.pwconv2 = engine::modules::binding::linear_from_source(
            *weights->store,
            source,
            prefix + ".pwconv2",
            matmul_storage_type,
            config.dim,
            config.intermediate_dim,
            true);
        block.gamma = weights->store->load_f32_tensor(source, prefix + ".gamma", {config.dim});
        weights->convnext.push_back(std::move(block));
    }
    weights->final_norm = engine::modules::binding::norm_from_source(*weights->store, source, "backbone.final_layer_norm", config.dim);
    weights->head_out = engine::modules::binding::linear_from_source(
        *weights->store,
        source,
        "head.out",
        matmul_storage_type,
        config.n_fft + 2,
        config.dim,
        true);
    weights->istft_window = source.require_f32("head.istft.window", {config.n_fft});
    weights->store->upload();
    return weights;
}

engine::core::TensorValue build_vocoder_convnext_block(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input_bct,
    const Vevo2VocoderConvNeXtBlockWeights & weights,
    const Vevo2VocoderConfig & config) {
    auto hidden = engine::modules::DepthwiseConv1dModule({
        config.dim,
        7,
        1,
        3,
        1,
        weights.dwconv.bias.has_value(),
    }).build(ctx, input_bct, weights.dwconv);
    hidden = transpose_bct_to_btc(ctx, hidden);
    hidden = engine::modules::LayerNormModule({config.dim, 1.0e-6F, true, true})
                 .build(ctx, hidden, weights.norm);
    hidden = engine::modules::LinearModule({
        config.dim,
        config.intermediate_dim,
        true,
        GGML_PREC_F32,
    }).build(ctx, hidden, weights.pwconv1);
    hidden = engine::modules::GeluModule({engine::modules::GeluApproximation::ExactErf}).build(ctx, hidden);
    hidden = engine::modules::LinearModule({
        config.intermediate_dim,
        config.dim,
        true,
        GGML_PREC_F32,
    }).build(ctx, hidden, weights.pwconv2);
    hidden = scale_last_dim(ctx, hidden, weights.gamma);
    hidden = transpose_btc_to_bct(ctx, hidden);
    return engine::modules::AddModule{}.build(ctx, input_bct, hidden);
}

engine::core::TensorValue build_vocoder_head(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & mel_bct,
    const Vevo2VocoderWeights & weights,
    const Vevo2VocoderConfig & config) {
    auto hidden = engine::modules::Conv1dModule({
        config.input_channels,
        config.dim,
        7,
        1,
        3,
        1,
        weights.embed.bias.has_value(),
    }).build(ctx, mel_bct, weights.embed);
    hidden = transpose_bct_to_btc(ctx, hidden);
    hidden = engine::modules::LayerNormModule({config.dim, 1.0e-6F, true, true}).build(ctx, hidden, weights.norm);
    hidden = transpose_btc_to_bct(ctx, hidden);
    for (const auto & block : weights.convnext) {
        hidden = build_vocoder_convnext_block(ctx, hidden, block, config);
    }
    hidden = transpose_bct_to_btc(ctx, hidden);
    hidden = engine::modules::LayerNormModule({config.dim, 1.0e-6F, true, true}).build(ctx, hidden, weights.final_norm);
    return engine::modules::LinearModule({
        config.dim,
        config.n_fft + 2,
        true,
        GGML_PREC_F32,
    }).build(ctx, hidden, weights.head_out);
}

std::vector<float> istft_same_from_head(
    const std::vector<float> & head,
    int64_t frames,
    const Vevo2VocoderConfig & config,
    const std::vector<float> & window,
    size_t threads) {
    const int64_t freq_bins = config.n_fft / 2 + 1;
    const int64_t out_dim = config.n_fft + 2;
    if (static_cast<int64_t>(head.size()) != frames * out_dim) {
        throw std::runtime_error("Vevo2 vocoder head output shape mismatch");
    }
    if (static_cast<int64_t>(window.size()) != config.n_fft) {
        throw std::runtime_error("Vevo2 vocoder ISTFT window shape mismatch");
    }

    std::vector<std::complex<float>> spectrum(static_cast<size_t>(frames * freq_bins));
    for (int64_t frame = 0; frame < frames; ++frame) {
        const float * row = head.data() + static_cast<size_t>(frame * out_dim);
        for (int64_t freq = 0; freq < freq_bins; ++freq) {
            const float mag = std::min(std::exp(row[freq]), 100.0F);
            const float phase = row[freq_bins + freq];
            spectrum[static_cast<size_t>(frame * freq_bins + freq)] = {
                mag * std::cos(phase),
                mag * std::sin(phase),
            };
        }
    }

    std::vector<float> framed(static_cast<size_t>(frames * config.n_fft), 0.0F);
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
        1,
        spectrum.data(),
        framed.data(),
        1.0F / static_cast<float>(config.n_fft),
        threads);

    const int64_t pad = (config.n_fft - config.hop_size) / 2;
    const int64_t output_size = (frames - 1) * config.hop_size + config.n_fft;
    std::vector<float> folded(static_cast<size_t>(output_size), 0.0F);
    std::vector<float> envelope(static_cast<size_t>(output_size), 0.0F);
    for (int64_t frame = 0; frame < frames; ++frame) {
        const int64_t start = frame * config.hop_size;
        const float * src = framed.data() + static_cast<size_t>(frame * config.n_fft);
        for (int64_t i = 0; i < config.n_fft; ++i) {
            const float w = window[static_cast<size_t>(i)];
            folded[static_cast<size_t>(start + i)] += src[i] * w;
            envelope[static_cast<size_t>(start + i)] += w * w;
        }
    }

    const int64_t samples = output_size - 2 * pad;
    if (samples <= 0) {
        throw std::runtime_error("Vevo2 vocoder ISTFT produced non-positive samples");
    }
    std::vector<float> audio(static_cast<size_t>(samples), 0.0F);
    for (int64_t i = 0; i < samples; ++i) {
        const int64_t src = i + pad;
        const float denom = envelope[static_cast<size_t>(src)];
        if (denom <= 1.0e-11F) {
            throw std::runtime_error("Vevo2 vocoder ISTFT window envelope underflow");
        }
        audio[static_cast<size_t>(i)] = folded[static_cast<size_t>(src)] / denom;
    }
    return audio;
}

}  // namespace

struct Vevo2VocoderGraph {
    Vevo2VocoderGraph(
        ggml_backend_t backend,
        engine::core::BackendType backend_type,
        size_t graph_context_bytes,
        const Vevo2VocoderConfig & config,
        std::shared_ptr<const Vevo2VocoderWeights> weights,
        int64_t frames)
        : backend(backend),
          weights(std::move(weights)),
          frames(frames),
          input_channels(config.input_channels),
          head_dim(config.n_fft + 2) {
        if (backend == nullptr || this->weights == nullptr) {
            throw std::runtime_error("Vevo2 vocoder graph requires backend and weights");
        }
        if (frames <= 0) {
            throw std::runtime_error("Vevo2 vocoder graph requires positive frame count");
        }
        ggml_init_params params{graph_context_bytes, nullptr, true};
        ctx.reset(ggml_init(params));
        if (ctx == nullptr) {
            throw std::runtime_error("failed to initialize Vevo2 vocoder graph context");
        }

        engine::core::ModuleBuildContext build_ctx{ctx.get(), "vevo2.vocoder", backend_type};
        input = engine::core::make_tensor(
            build_ctx,
            GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({1, config.input_channels, frames})).tensor;
        auto mel = engine::core::wrap_tensor(
            input,
            engine::core::TensorShape::from_dims({1, config.input_channels, frames}),
            GGML_TYPE_F32);
        auto head = build_vocoder_head(build_ctx, mel, *this->weights, config);
        head = ensure_contiguous(build_ctx, head);
        output = head.tensor;
        ggml_set_output(output);

        graph = ggml_new_graph_custom(ctx.get(), 65536, false);
        ggml_build_forward_expand(graph, output);
        gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (gallocr == nullptr || !ggml_gallocr_reserve(gallocr, graph) || !ggml_gallocr_alloc_graph(gallocr, graph)) {
            throw std::runtime_error("failed to allocate Vevo2 vocoder graph");
        }
    }

    ~Vevo2VocoderGraph() {
        if (gallocr != nullptr) {
            ggml_gallocr_free(gallocr);
            gallocr = nullptr;
        }
    }

    bool matches(const Vevo2VocoderWeights & other_weights, int64_t other_frames) const noexcept {
        return weights.get() == &other_weights && frames == other_frames;
    }

    std::vector<float> run(const Vevo2MelSequence & mel) {
        if (mel.frames != frames || mel.mel_bins != input_channels ||
            static_cast<int64_t>(mel.values.size()) != frames * mel.mel_bins) {
            throw std::runtime_error("Vevo2 vocoder mel shape mismatch");
        }
        std::vector<float> bct(static_cast<size_t>(mel.mel_bins * frames), 0.0F);
        for (int64_t frame = 0; frame < frames; ++frame) {
            for (int64_t bin = 0; bin < mel.mel_bins; ++bin) {
                bct[static_cast<size_t>(bin * frames + frame)] =
                    mel.values[static_cast<size_t>(frame * mel.mel_bins + bin)];
            }
        }
        ggml_backend_tensor_set(input, bct.data(), 0, bct.size() * sizeof(float));
        const ggml_status status = engine::core::compute_backend_graph(backend, graph);
        ggml_backend_synchronize(backend);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Vevo2 vocoder graph compute failed");
        }
        std::vector<float> head(static_cast<size_t>(frames * head_dim), 0.0F);
        ggml_backend_tensor_get(output, head.data(), 0, head.size() * sizeof(float));
        return head;
    }

    ggml_backend_t backend = nullptr;
    std::shared_ptr<const Vevo2VocoderWeights> weights;
    int64_t frames = 0;
    int64_t input_channels = 0;
    int64_t head_dim = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx;
    ggml_tensor * input = nullptr;
    ggml_tensor * output = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_gallocr_t gallocr = nullptr;
};

Vevo2VocoderRuntime::Vevo2VocoderRuntime(
    const Vevo2Assets & assets,
    engine::core::ExecutionContext & execution_context,
    size_t weight_context_bytes,
    size_t graph_context_bytes,
    engine::assets::TensorStorageType matmul_weight_storage_type,
    engine::assets::TensorStorageType conv_weight_storage_type)
    : config_(assets.config.vocoder),
      execution_context_(execution_context),
      graph_context_bytes_(graph_context_bytes),
      weight_source_(assets.vocoder_weights_0),
      weights_(load_vocoder_weights(
          execution_context.backend(),
          execution_context.backend_type(),
          *weight_source_,
          config_,
          weight_context_bytes,
          matmul_weight_storage_type,
          conv_weight_storage_type)),
      name_("vocoder") {
    weight_source_->release_storage();
}

Vevo2VocoderRuntime::~Vevo2VocoderRuntime() = default;

runtime::AudioBuffer Vevo2VocoderRuntime::decode(const Vevo2MelSequence & mel) const {
    if (mel.frames <= 0 || mel.mel_bins != config_.input_channels) {
        throw std::runtime_error("Vevo2 vocoder requires non-empty mel with configured bins");
    }
    if (graph_ == nullptr || !graph_->matches(*weights_, mel.frames)) {
        graph_ = std::make_unique<Vevo2VocoderGraph>(
            execution_context_.backend(),
            execution_context_.backend_type(),
            graph_context_bytes_,
            config_,
            weights_,
            mel.frames);
    }
    const auto head = graph_->run(mel);
    runtime::AudioBuffer out;
    out.sample_rate = static_cast<int>(config_.preprocess.sample_rate);
    out.channels = 1;
    out.samples = istft_same_from_head(
        head,
        mel.frames,
        config_,
        weights_->istft_window,
        static_cast<size_t>(execution_context_.config().threads));
    return out;
}

}  // namespace engine::models::vevo2
