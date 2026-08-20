#include "engine/models/meanvc2/vocoder.h"

#include "engine/framework/audio/fft.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/module.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <memory>
#include <stdexcept>
#include <utility>

namespace engine::models::meanvc2 {
namespace {

constexpr int64_t kSampleRate = 16000;
constexpr int64_t kMelBins = 80;
constexpr int64_t kDim = 320;
constexpr int64_t kIntermediateDim = 1536;
constexpr int64_t kLayers = 8;
constexpr int64_t kNfft = 640;
constexpr int64_t kHopSize = 160;
constexpr int64_t kHeadDim = kNfft + 2;
constexpr int64_t kStreamChunkMelFrames = 12;
constexpr int64_t kVocoderOverlapMelFrames = 2;
constexpr int64_t kVocoderOverlapSamples = (kVocoderOverlapMelFrames - 1) * kHopSize;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

}  // namespace

struct MeanVC2VocoderConvNeXtBlockWeights {
    modules::DepthwiseConv1dWeights dwconv;
    modules::NormWeights norm;
    modules::LinearWeights pwconv1;
    modules::LinearWeights pwconv2;
    core::TensorValue gamma;
};

struct MeanVC2VocoderWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    modules::Conv1dWeights embed;
    modules::NormWeights norm;
    std::vector<MeanVC2VocoderConvNeXtBlockWeights> convnext;
    modules::NormWeights final_norm;
    modules::LinearWeights head_out;
    std::vector<float> istft_window;
};

namespace {

core::TensorValue scale_last_dim(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & scale) {
    const auto view = core::reshape_tensor(
        ctx,
        scale,
        core::TensorShape::from_dims({1, 1, scale.shape.dims[0]}));
    const auto repeated = modules::RepeatModule({input.shape}).build(ctx, view);
    return modules::MulModule{}.build(ctx, input, repeated);
}

std::shared_ptr<const MeanVC2VocoderWeights> load_vocoder_weights(
    ggml_backend_t backend,
    core::BackendType backend_type,
    const assets::TensorSource & source,
    size_t weight_context_bytes,
    assets::TensorStorageType matmul_storage_type,
    assets::TensorStorageType conv_storage_type) {
    auto weights = std::make_shared<MeanVC2VocoderWeights>();
    weights->store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "meanvc2.vocoder.weights",
        weight_context_bytes);
    weights->embed = modules::binding::conv1d_from_source(
        *weights->store,
        source,
        "backbone.embed",
        conv_storage_type,
        kDim,
        kMelBins,
        7,
        true);
    weights->norm = modules::binding::norm_from_source(*weights->store, source, "backbone.norm", kDim);
    weights->convnext.reserve(static_cast<size_t>(kLayers));
    for (int64_t layer = 0; layer < kLayers; ++layer) {
        const std::string prefix = "backbone.convnext." + std::to_string(layer);
        MeanVC2VocoderConvNeXtBlockWeights block;
        block.dwconv = modules::binding::depthwise_conv1d_from_source(
            *weights->store,
            source,
            prefix + ".dwconv",
            conv_storage_type,
            kDim,
            7,
            true);
        block.norm = modules::binding::norm_from_source(*weights->store, source, prefix + ".norm", kDim);
        block.pwconv1 = modules::binding::linear_from_source(
            *weights->store,
            source,
            prefix + ".pwconv1",
            matmul_storage_type,
            kIntermediateDim,
            kDim,
            true);
        block.pwconv2 = modules::binding::linear_from_source(
            *weights->store,
            source,
            prefix + ".pwconv2",
            matmul_storage_type,
            kDim,
            kIntermediateDim,
            true);
        block.gamma = weights->store->load_f32_tensor(source, prefix + ".gamma", {kDim});
        weights->convnext.push_back(std::move(block));
    }
    weights->final_norm = modules::binding::norm_from_source(*weights->store, source, "backbone.final_layer_norm", kDim);
    weights->head_out = modules::binding::linear_from_source(
        *weights->store,
        source,
        "head.out",
        matmul_storage_type,
        kHeadDim,
        kDim,
        true);
    weights->istft_window = source.require_f32("head.istft.window", {kNfft});
    weights->store->upload();
    return weights;
}

core::TensorValue build_convnext_block(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input_bct,
    const MeanVC2VocoderConvNeXtBlockWeights & weights) {
    auto hidden = modules::DepthwiseConv1dModule({
        kDim,
        7,
        1,
        3,
        1,
        weights.dwconv.bias.has_value(),
    }).build(ctx, input_bct, weights.dwconv);
    hidden = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, hidden);
    hidden = modules::LayerNormModule({kDim, 1.0e-6F, true, true}).build(ctx, hidden, weights.norm);
    hidden = modules::LinearModule({kDim, kIntermediateDim, true}).build(ctx, hidden, weights.pwconv1);
    hidden = modules::GeluModule({modules::GeluApproximation::ExactErf}).build(ctx, hidden);
    hidden = modules::LinearModule({kIntermediateDim, kDim, true}).build(ctx, hidden, weights.pwconv2);
    hidden = scale_last_dim(ctx, hidden, weights.gamma);
    hidden = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, hidden);
    return modules::AddModule{}.build(ctx, input_bct, hidden);
}

core::TensorValue build_vocoder_head(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & mel_bct,
    const MeanVC2VocoderWeights & weights) {
    auto hidden = modules::Conv1dModule({
        kMelBins,
        kDim,
        7,
        1,
        3,
        1,
        weights.embed.bias.has_value(),
    }).build(ctx, mel_bct, weights.embed);
    hidden = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, hidden);
    hidden = modules::LayerNormModule({kDim, 1.0e-6F, true, true}).build(ctx, hidden, weights.norm);
    hidden = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, hidden);
    for (const auto & block : weights.convnext) {
        hidden = build_convnext_block(ctx, hidden, block);
    }
    hidden = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, hidden);
    hidden = modules::LayerNormModule({kDim, 1.0e-6F, true, true}).build(ctx, hidden, weights.final_norm);
    return modules::LinearModule({kDim, kHeadDim, true}).build(ctx, hidden, weights.head_out);
}

std::vector<float> istft_center_from_head(
    const std::vector<float> & head,
    int64_t frames,
    const std::vector<float> & window,
    size_t threads) {
    constexpr int64_t kFreqBins = kNfft / 2 + 1;
    if (static_cast<int64_t>(head.size()) != frames * kHeadDim) {
        throw std::runtime_error("MeanVC2 vocoder head output shape mismatch");
    }
    if (static_cast<int64_t>(window.size()) != kNfft) {
        throw std::runtime_error("MeanVC2 vocoder ISTFT window shape mismatch");
    }

    std::vector<std::complex<float>> spectrum(static_cast<size_t>(frames * kFreqBins));
    for (int64_t frame = 0; frame < frames; ++frame) {
        const float * row = head.data() + static_cast<size_t>(frame * kHeadDim);
        for (int64_t freq = 0; freq < kFreqBins; ++freq) {
            const float mag = std::min(std::exp(row[freq]), 100.0F);
            const float phase = row[kFreqBins + freq];
            spectrum[static_cast<size_t>(frame * kFreqBins + freq)] = {
                mag * std::cos(phase),
                mag * std::sin(phase),
            };
        }
    }

    std::vector<float> framed(static_cast<size_t>(frames * kNfft), 0.0F);
    audio::real_fft_inverse(
        {static_cast<size_t>(frames), static_cast<size_t>(kNfft)},
        {
            static_cast<std::ptrdiff_t>(kFreqBins * static_cast<int64_t>(sizeof(std::complex<float>))),
            static_cast<std::ptrdiff_t>(sizeof(std::complex<float>)),
        },
        {
            static_cast<std::ptrdiff_t>(kNfft * static_cast<int64_t>(sizeof(float))),
            static_cast<std::ptrdiff_t>(sizeof(float)),
        },
        1,
        spectrum.data(),
        framed.data(),
        1.0F / static_cast<float>(kNfft),
        threads);

    const int64_t output_size = (frames - 1) * kHopSize + kNfft;
    std::vector<float> folded(static_cast<size_t>(output_size), 0.0F);
    std::vector<float> envelope(static_cast<size_t>(output_size), 0.0F);
    for (int64_t frame = 0; frame < frames; ++frame) {
        const int64_t start = frame * kHopSize;
        const float * src = framed.data() + static_cast<size_t>(frame * kNfft);
        for (int64_t i = 0; i < kNfft; ++i) {
            const float w = window[static_cast<size_t>(i)];
            folded[static_cast<size_t>(start + i)] += src[i] * w;
            envelope[static_cast<size_t>(start + i)] += w * w;
        }
    }

    constexpr int64_t kCenterPad = kNfft / 2;
    const int64_t samples = output_size - 2 * kCenterPad;
    if (samples <= 0) {
        throw std::runtime_error("MeanVC2 vocoder ISTFT produced non-positive samples");
    }
    std::vector<float> audio(static_cast<size_t>(samples), 0.0F);
    for (int64_t i = 0; i < samples; ++i) {
        const int64_t src = i + kCenterPad;
        const float denom = envelope[static_cast<size_t>(src)];
        if (denom <= 1.0e-11F) {
            throw std::runtime_error("MeanVC2 vocoder ISTFT window envelope underflow");
        }
        audio[static_cast<size_t>(i)] = folded[static_cast<size_t>(src)] / denom;
    }
    return audio;
}

}  // namespace

struct MeanVC2VocoderGraph {
    MeanVC2VocoderGraph(
        ggml_backend_t backend,
        core::BackendType backend_type,
        size_t graph_context_bytes,
        std::shared_ptr<const MeanVC2VocoderWeights> weights,
        int64_t frames)
        : backend(backend),
          weights(std::move(weights)),
          frames(frames) {
        if (backend == nullptr || this->weights == nullptr) {
            throw std::runtime_error("MeanVC2 vocoder graph requires backend and weights");
        }
        if (frames <= 0) {
            throw std::runtime_error("MeanVC2 vocoder graph requires positive frame count");
        }

        ggml_init_params params{graph_context_bytes, nullptr, true};
        ctx.reset(ggml_init(params));
        if (ctx == nullptr) {
            throw std::runtime_error("failed to initialize MeanVC2 vocoder graph context");
        }

        core::ModuleBuildContext build_ctx{ctx.get(), "meanvc2.vocoder", backend_type};
        input = core::make_tensor(
            build_ctx,
            GGML_TYPE_F32,
            core::TensorShape::from_dims({1, kMelBins, frames}));
        auto mel = input;
        auto head = build_vocoder_head(build_ctx, mel, *this->weights);
        head = core::ensure_backend_addressable_layout(build_ctx, head);
        output = head;
        ggml_set_output(output.tensor);

        graph = ggml_new_graph_custom(ctx.get(), 65536, false);
        ggml_build_forward_expand(graph, output.tensor);
        gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (gallocr == nullptr || !ggml_gallocr_reserve(gallocr, graph) || !ggml_gallocr_alloc_graph(gallocr, graph)) {
            throw std::runtime_error("failed to allocate MeanVC2 vocoder graph");
        }
    }

    ~MeanVC2VocoderGraph() {
        if (gallocr != nullptr) {
            ggml_gallocr_free(gallocr);
            gallocr = nullptr;
        }
    }

    bool matches(const MeanVC2VocoderWeights & other_weights, int64_t other_frames) const noexcept {
        return weights.get() == &other_weights && frames == other_frames;
    }

    std::vector<float> run(const std::vector<float> & mel_frames) {
        if (static_cast<int64_t>(mel_frames.size()) != frames * kMelBins) {
            throw std::runtime_error("MeanVC2 vocoder mel shape mismatch");
        }
        std::vector<float> bct(static_cast<size_t>(kMelBins * frames), 0.0F);
        for (int64_t frame = 0; frame < frames; ++frame) {
            for (int64_t bin = 0; bin < kMelBins; ++bin) {
                const float scaled = (mel_frames[static_cast<size_t>(frame * kMelBins + bin)] + 1.0F) * 0.5F;
                bct[static_cast<size_t>(bin * frames + frame)] = scaled;
            }
        }
        core::write_tensor_float(input, bct);
        const ggml_status status = core::compute_backend_graph(backend, graph);
        ggml_backend_synchronize(backend);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("MeanVC2 vocoder graph compute failed");
        }
        return core::read_tensor_float(output.tensor);
    }

    ggml_backend_t backend = nullptr;
    std::shared_ptr<const MeanVC2VocoderWeights> weights;
    int64_t frames = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx;
    core::TensorValue input;
    core::TensorValue output;
    ggml_cgraph * graph = nullptr;
    ggml_gallocr_t gallocr = nullptr;
};

MeanVC2VocoderRuntime::MeanVC2VocoderRuntime(
    std::shared_ptr<const assets::TensorSource> source,
    core::ExecutionContext & execution_context,
    size_t weight_context_bytes,
    size_t graph_context_bytes,
    assets::TensorStorageType matmul_weight_storage_type,
    assets::TensorStorageType conv_weight_storage_type)
    : execution_context_(execution_context),
      graph_context_bytes_(graph_context_bytes),
      source_(std::move(source)) {
    if (source_ == nullptr) {
        throw std::runtime_error("MeanVC2 vocoder requires tensor source");
    }
    weights_ = load_vocoder_weights(
        execution_context_.backend(),
        execution_context_.backend_type(),
        *source_,
        weight_context_bytes,
        matmul_weight_storage_type,
        conv_weight_storage_type);
    source_->release_storage();
}

MeanVC2VocoderRuntime::~MeanVC2VocoderRuntime() = default;

MeanVC2VocoderGraph & MeanVC2VocoderRuntime::graph_for_frames(int64_t frames) const {
    if (frames <= 0) {
        throw std::runtime_error("MeanVC2 vocoder graph requires positive frame count");
    }
    for (const auto & graph : graphs_) {
        if (graph != nullptr && graph->matches(*weights_, frames)) {
            return *graph;
        }
    }
    graphs_.push_back(std::make_unique<MeanVC2VocoderGraph>(
            execution_context_.backend(),
            execution_context_.backend_type(),
            graph_context_bytes_,
            weights_,
            frames));
    return *graphs_.back();
}

runtime::AudioBuffer MeanVC2VocoderRuntime::decode(const std::vector<float> & mel_frames, int64_t frames) const {
    if (frames <= 0) {
        throw std::runtime_error("MeanVC2 vocoder requires positive frame count");
    }
    if (static_cast<int64_t>(mel_frames.size()) != frames * kMelBins) {
        throw std::runtime_error("MeanVC2 vocoder requires [frames, 80] mel input");
    }
    const auto head = graph_for_frames(frames).run(mel_frames);
    runtime::AudioBuffer out;
    out.sample_rate = static_cast<int>(kSampleRate);
    out.channels = 1;
    out.samples = istft_center_from_head(
        head,
        frames,
        weights_->istft_window,
        static_cast<size_t>(execution_context_.config().threads));
    return out;
}

runtime::AudioBuffer MeanVC2VocoderRuntime::decode_streaming(
    const std::vector<float> & mel_frames,
    int64_t frames) const {
    if (frames <= 0) {
        throw std::runtime_error("MeanVC2 streaming vocoder requires positive frame count");
    }
    if (static_cast<int64_t>(mel_frames.size()) != frames * kMelBins) {
        throw std::runtime_error("MeanVC2 streaming vocoder requires [frames, 80] mel input");
    }

    reset_streaming_state();
    runtime::AudioBuffer out;
    out.sample_rate = static_cast<int>(kSampleRate);
    out.channels = 1;
    out.samples.reserve(static_cast<size_t>(frames * kHopSize));
    for (int64_t start = 0; start < frames; start += kStreamChunkMelFrames) {
        const int64_t chunk_frames = std::min<int64_t>(kStreamChunkMelFrames, frames - start);
        const auto * chunk_begin = mel_frames.data() + static_cast<std::ptrdiff_t>(start * kMelBins);
        auto chunk = decode_streaming_chunk(
            std::vector<float>(
                chunk_begin,
                chunk_begin + static_cast<std::ptrdiff_t>(chunk_frames * kMelBins)),
            chunk_frames);
        runtime::append_audio_buffer(out, chunk);
    }
    runtime::append_audio_buffer(out, finish_streaming());
    return out;
}

void MeanVC2VocoderRuntime::reset_streaming_state() const {
    streaming_mel_cache_.clear();
    streaming_last_wav_.clear();
}

runtime::AudioBuffer MeanVC2VocoderRuntime::decode_streaming_chunk(
    const std::vector<float> & mel_frames,
    int64_t frames) const {
    if (frames <= 0) {
        throw std::runtime_error("MeanVC2 streaming vocoder requires positive chunk frame count");
    }
    if (static_cast<int64_t>(mel_frames.size()) != frames * kMelBins) {
        throw std::runtime_error("MeanVC2 streaming vocoder chunk requires [frames, 80] mel input");
    }

    std::vector<float> window;
    window.reserve(static_cast<size_t>((kVocoderOverlapMelFrames + frames) * kMelBins));
    if (!streaming_mel_cache_.empty()) {
        window.insert(window.end(), streaming_mel_cache_.begin(), streaming_mel_cache_.end());
    }
    window.insert(window.end(), mel_frames.begin(), mel_frames.end());
    const int64_t window_frames = static_cast<int64_t>(window.size() / static_cast<size_t>(kMelBins));

    const auto head = graph_for_frames(window_frames).run(window);
    const auto wav = istft_center_from_head(
        head,
        window_frames,
        weights_->istft_window,
        static_cast<size_t>(execution_context_.config().threads));
    if (static_cast<int64_t>(wav.size()) <= kVocoderOverlapSamples) {
        throw std::runtime_error("MeanVC2 streaming vocoder chunk is shorter than overlap");
    }

    runtime::AudioBuffer out;
    out.sample_rate = static_cast<int>(kSampleRate);
    out.channels = 1;
    if (streaming_last_wav_.empty()) {
        out.samples.insert(
            out.samples.end(),
            wav.begin(),
            wav.end() - static_cast<std::ptrdiff_t>(kVocoderOverlapSamples));
    } else {
        if (static_cast<int64_t>(streaming_last_wav_.size()) != kVocoderOverlapSamples) {
            throw std::runtime_error("MeanVC2 streaming vocoder overlap cache shape mismatch");
        }
        for (int64_t i = 0; i < kVocoderOverlapSamples; ++i) {
            const float up = static_cast<float>(i) / static_cast<float>(kVocoderOverlapSamples - 1);
            const float down = 1.0F - up;
            out.samples.push_back(
                streaming_last_wav_[static_cast<size_t>(i)] * down +
                wav[static_cast<size_t>(i)] * up);
        }
        out.samples.insert(
            out.samples.end(),
            wav.begin() + static_cast<std::ptrdiff_t>(kVocoderOverlapSamples),
            wav.end() - static_cast<std::ptrdiff_t>(kVocoderOverlapSamples));
    }

    const int64_t cache_frames = std::min<int64_t>(kVocoderOverlapMelFrames, window_frames);
    streaming_mel_cache_.assign(
        window.end() - static_cast<std::ptrdiff_t>(cache_frames * kMelBins),
        window.end());
    streaming_last_wav_.assign(
        wav.end() - static_cast<std::ptrdiff_t>(kVocoderOverlapSamples),
        wav.end());
    return out;
}

runtime::AudioBuffer MeanVC2VocoderRuntime::finish_streaming() const {
    runtime::AudioBuffer out;
    out.sample_rate = static_cast<int>(kSampleRate);
    out.channels = 1;
    out.samples = std::move(streaming_last_wav_);
    reset_streaming_state();
    return out;
}

}  // namespace engine::models::meanvc2
