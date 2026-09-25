#include "engine/community_models/lfm2_audio/audio_encoder.h"

#include "engine/framework/audio/nemo_mel_frontend.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/conformer_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/weight_binding.h"
#include "engine/framework/runtime/errors.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace engine::community_models::lfm2_audio {
namespace {

namespace modules = engine::modules;
namespace binding = engine::modules::binding;

// The "preprocessor" block of the checkpoint config (the JP checkpoint has the
// same one); the GGUF does not carry it:
// https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B/blob/c362a0625dfe45aa588dce5f0ada28a7e5707628/config.json
// window_size 0.025 s and window_stride 0.01 s are 400 and 160 samples.
constexpr int64_t kSampleRate = 16000;
constexpr int64_t kFftSize = 512;
constexpr int64_t kWindowLength = 400;
constexpr int64_t kHopLength = 160;
// Not in the config: NeMo AudioToMelSpectrogramPreprocessor's default.
constexpr float kPreemphasis = 0.97f;

constexpr size_t kWeightContextBytes = 16ull * 1024ull * 1024ull;
constexpr size_t kGraphArenaBytes = 64ull * 1024ull * 1024ull;
constexpr size_t kGraphNodes = 32768;
constexpr double kPi = 3.14159265358979323846;

// torch.hann_window(win_length, periodic=False), as NeMo builds it.
std::vector<float> symmetric_hann_window(int64_t length) {
    std::vector<float> window(static_cast<size_t>(length));
    for (int64_t i = 0; i < length; ++i) {
        window[static_cast<size_t>(i)] =
            static_cast<float>(0.5 - 0.5 * std::cos(2.0 * kPi * static_cast<double>(i) / static_cast<double>(length - 1)));
    }
    return window;
}

double slaney_hz_to_mel(double hz) {
    constexpr double kLinearSlope = 200.0 / 3.0;
    constexpr double kMinLogHz = 1000.0;
    const double min_log_mel = kMinLogHz / kLinearSlope;
    const double log_step = std::log(6.4) / 27.0;
    return hz >= kMinLogHz ? min_log_mel + std::log(hz / kMinLogHz) / log_step : hz / kLinearSlope;
}

double slaney_mel_to_hz(double mel) {
    constexpr double kLinearSlope = 200.0 / 3.0;
    constexpr double kMinLogHz = 1000.0;
    const double min_log_mel = kMinLogHz / kLinearSlope;
    const double log_step = std::log(6.4) / 27.0;
    return mel >= min_log_mel ? kMinLogHz * std::exp(log_step * (mel - min_log_mel)) : kLinearSlope * mel;
}

// librosa.filters.mel(sr, n_fft, n_mels, fmin=0, fmax=sr/2, norm="slaney"),
// computed in double like librosa and stored as [n_mels, n_fft / 2 + 1].
audio::AudioTensor slaney_filterbank(int64_t n_mels) {
    const int64_t bins = kFftSize / 2 + 1;
    const double mel_min = slaney_hz_to_mel(0.0);
    const double mel_max = slaney_hz_to_mel(static_cast<double>(kSampleRate) / 2.0);
    std::vector<double> edges(static_cast<size_t>(n_mels + 2));
    const double step = (mel_max - mel_min) / static_cast<double>(n_mels + 1);
    for (int64_t i = 0; i < n_mels + 2; ++i) {
        edges[static_cast<size_t>(i)] = slaney_mel_to_hz(mel_min + static_cast<double>(i) * step);
    }
    edges.back() = slaney_mel_to_hz(mel_max);

    audio::AudioTensor out;
    out.shape = {n_mels, bins};
    out.values.assign(static_cast<size_t>(n_mels * bins), 0.0f);
    for (int64_t m = 0; m < n_mels; ++m) {
        const double left = edges[static_cast<size_t>(m)];
        const double center = edges[static_cast<size_t>(m + 1)];
        const double right = edges[static_cast<size_t>(m + 2)];
        const double norm = 2.0 / (right - left);
        for (int64_t k = 0; k < bins; ++k) {
            const double hz = static_cast<double>(k) * static_cast<double>(kSampleRate) / static_cast<double>(kFftSize);
            const double lower = (hz - left) / (center - left);
            const double upper = (right - hz) / (right - center);
            const double weight = std::max(0.0, std::min(lower, upper));
            out.values[static_cast<size_t>(m * bins + k)] = static_cast<float>(weight * norm);
        }
    }
    return out;
}

audio::NemoMelFrontend make_frontend(int64_t n_mels) {
    audio::NemoMelFrontendConfig config;
    config.sample_rate = kSampleRate;
    config.n_mels = n_mels;
    config.stft = {kFftSize, kHopLength, kWindowLength, true, audio::STFTPadMode::Constant};
    config.preemphasis = kPreemphasis;
    config.window = audio::MelWindow::FromArgument;
    config.mel_bank = audio::MelBank::FromArgument;
    config.mel_path = audio::MelPath::LogMelSpectrogram;
    config.norm = audio::MelNorm::PerBinF32;
    config.layout = audio::MelLayout::FeatureMajor;
    return audio::NemoMelFrontend(config, symmetric_hann_window(kWindowLength), slaney_filterbank(n_mels));
}

struct EncoderWeights {
    std::unique_ptr<core::BackendWeightStore> store;
    modules::DepthwiseConvSubsamplingWeights subsampling;
    std::vector<modules::RelativeConformerBlockWeights> blocks;
    modules::NormWeights adapter_norm;
    modules::LinearWeights adapter_fc1;
    modules::LinearWeights adapter_fc2;
};

// llama.cpp's converter stores the subsampling conv biases as [C, 1, 1].
modules::Conv2dWeights load_conv2d(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t out_channels,
    int64_t in_channels,
    int64_t kernel) {
    modules::Conv2dWeights weights;
    weights.weight = store.load_tensor(
        source, prefix + ".weight", assets::TensorStorageType::F32, {out_channels, in_channels, kernel, kernel});
    const auto bias = source.require_f32(prefix + ".bias", {out_channels, 1, 1});
    weights.bias = store.make_tensor(
        core::TensorShape::from_dims({out_channels}), GGML_TYPE_F32, bias.data(), bias.size() * sizeof(float));
    return weights;
}

EncoderWeights load_encoder_weights(
    const assets::TensorSource & source, const Lfm2FastConformerEncoderConfig & config, core::ExecutionContext & execution) {
    EncoderWeights out;
    out.store = std::make_unique<core::BackendWeightStore>(
        execution.backend(), execution.backend_type(), "lfm2_audio.encoder.weights", kWeightContextBytes);
    auto & store = *out.store;
    const auto native = assets::TensorStorageType::Native;
    const auto linear = [&](const std::string & name) {
        return binding::linear_from_named_source(store, source, name + ".weight", name + ".bias", native);
    };
    const auto norm = [&](const std::string & name) {
        return binding::norm_from_named_source(store, source, name + ".weight", name + ".bias");
    };
    const int64_t channels = config.subsampling_channels;
    const int64_t d = config.hidden_size;

    // NeMo pre_encode.conv indices: 0 conv, 2/5 depthwise, 3/6 pointwise.
    out.subsampling.input_conv = load_conv2d(store, source, "a.conv1d.0", channels, 1, 3);
    out.subsampling.stages.resize(2);
    out.subsampling.stages[0].depthwise = load_conv2d(store, source, "a.conv1d.2", channels, 1, 3);
    out.subsampling.stages[0].pointwise = load_conv2d(store, source, "a.conv1d.3", channels, channels, 1);
    out.subsampling.stages[1].depthwise = load_conv2d(store, source, "a.conv1d.5", channels, 1, 3);
    out.subsampling.stages[1].pointwise = load_conv2d(store, source, "a.conv1d.6", channels, channels, 1);
    out.subsampling.projection = linear("a.pre_encode.out");

    const int64_t head_dim = d / config.num_heads;
    for (int64_t layer = 0; layer < config.num_layers; ++layer) {
        const std::string p = "a.blk." + std::to_string(layer);
        modules::RelativeConformerBlockWeights w;
        w.ffn1_norm = norm(p + ".ffn_norm");
        w.ffn1_fc1 = linear(p + ".ffn_up");
        w.ffn1_fc2 = linear(p + ".ffn_down");
        w.norm1 = norm(p + ".ln1");
        const auto q = linear(p + ".attn_q");
        const auto k = linear(p + ".attn_k");
        const auto v = linear(p + ".attn_v");
        const auto o = linear(p + ".attn_out");
        w.self_attention.attention.q_weight = q.weight;
        w.self_attention.attention.q_bias = q.bias;
        w.self_attention.attention.k_weight = k.weight;
        w.self_attention.attention.k_bias = k.bias;
        w.self_attention.attention.v_weight = v.weight;
        w.self_attention.attention.v_bias = v.bias;
        w.self_attention.attention.out_weight = o.weight;
        w.self_attention.attention.out_bias = o.bias;
        w.self_attention.pos_weight = store.load_tensor(source, p + ".linear_pos.weight", native, {d, d});
        w.self_attention.pos_bias_u = store.load_f32_tensor(source, p + ".pos_bias_u", {config.num_heads, head_dim});
        w.self_attention.pos_bias_v = store.load_f32_tensor(source, p + ".pos_bias_v", {config.num_heads, head_dim});
        w.conv.norm = norm(p + ".norm_conv");
        w.conv.pointwise_in = linear(p + ".conv_pw1");
        w.conv.pointwise_out = linear(p + ".conv_pw2");
        w.conv.depthwise.weight = store.load_tensor_as_shape(
            source, p + ".conv_dw.weight", assets::TensorStorageType::F32, {d, config.conv_kernel_size},
            core::TensorShape::from_dims({d, 1, config.conv_kernel_size}));
        w.conv.depthwise.bias = store.load_f32_tensor(source, p + ".conv_dw.bias", {d});
        // BatchNorm is already folded into a per-channel scale and bias.
        w.conv.depthwise_norm = {
            store.load_f32_tensor(source, p + ".conv_norm.weight", {d}),
            store.load_f32_tensor(source, p + ".conv_norm.bias", {d})};
        w.norm2 = norm(p + ".ffn_norm_1");
        w.ffn2_fc1 = linear(p + ".ffn_up_1");
        w.ffn2_fc2 = linear(p + ".ffn_down_1");
        w.final_norm = norm(p + ".ln2");
        out.blocks.push_back(std::move(w));
    }

    // Adapter: LayerNorm -> Linear -> GELU -> Linear (mm.a.mlp.{0,1,3}).
    out.adapter_norm = norm("mm.a.mlp.0");
    out.adapter_fc1 = linear("mm.a.mlp.1");
    out.adapter_fc2 = linear("mm.a.mlp.3");
    store.upload();
    return out;
}

// Relative positions run from +(T-1) down to -(T-1), sin and cos interleaved.
std::vector<float> relative_position_embeddings(int64_t steps, int64_t d_model) {
    const int64_t count = 2 * steps - 1;
    std::vector<float> values(static_cast<size_t>(count * d_model));
    for (int64_t p = 0; p < count; ++p) {
        for (int64_t i = 0; i < d_model / 2; ++i) {
            const double phase = static_cast<double>(steps - 1 - p) *
                std::pow(10000.0, -2.0 * static_cast<double>(i) / static_cast<double>(d_model));
            values[static_cast<size_t>(p * d_model + 2 * i)] = static_cast<float>(std::sin(phase));
            values[static_cast<size_t>(p * d_model + 2 * i + 1)] = static_cast<float>(std::cos(phase));
        }
    }
    return values;
}

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept { ggml_free(ctx); }
};

struct GgmlGallocrDeleter {
    void operator()(ggml_gallocr_t alloc) const noexcept { ggml_gallocr_free(alloc); }
};

}  // namespace

struct Lfm2FastConformerEncoderRuntime::Impl {
    Impl(std::shared_ptr<const assets::TensorSource> source_in,
         const Lfm2FastConformerEncoderConfig & config_in,
         core::ExecutionContext & execution_in)
        : source(std::move(source_in)),
          config(config_in),
          execution(execution_in),
          frontend(make_frontend(config_in.n_mels)),
          weights(load_encoder_weights(*source, config_in, execution_in)) {}

    Lfm2AudioEmbeddings encode(const Lfm2AudioFeatures & features) {
        const auto start = std::chrono::steady_clock::now();
        const int64_t frames = features.frames;
        std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_owner(ggml_init({kGraphArenaBytes, nullptr, true}));
        if (ctx_owner == nullptr) {
            throw std::runtime_error("failed to initialize the LFM2-Audio encoder graph context");
        }
        auto * gctx = ctx_owner.get();
        core::ModuleBuildContext ctx{gctx, "lfm2_audio.encoder", execution.backend_type()};

        auto input = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, frames, config.n_mels}));
        ggml_set_input(input.tensor);
        auto x = modules::DepthwiseConvSubsamplingModule({config.n_mels, config.hidden_size, config.subsampling_channels})
                     .build(ctx, input, weights.subsampling);
        const int64_t steps = x.shape.dims[1];
        auto pos = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 2 * steps - 1, config.hidden_size}));
        ggml_set_input(pos.tensor);

        modules::ConformerBlockConfig block_config{
            config.hidden_size, config.num_heads, config.intermediate_size, config.conv_kernel_size, config.layer_norm_eps};
        block_config.contiguous_glu_gate = true;
        for (const auto & block : weights.blocks) {
            x = modules::RelativeConformerBlockModule(block_config).build(ctx, x, pos, block);
        }
        x = modules::LayerNormModule({config.hidden_size, config.layer_norm_eps}).build(ctx, x, weights.adapter_norm);
        x = modules::LinearModule({config.hidden_size, config.adapter_hidden_size, true}).build(ctx, x, weights.adapter_fc1);
        x = modules::GeluModule({modules::GeluApproximation::ExactErf}).build(ctx, x);
        x = modules::LinearModule({config.adapter_hidden_size, config.output_size, true}).build(ctx, x, weights.adapter_fc2);
        x = core::ensure_backend_addressable_layout(ctx, x);
        ggml_set_output(x.tensor);

        auto * graph = ggml_new_graph_custom(gctx, kGraphNodes, false);
        ggml_build_forward_expand(graph, x.tensor);
        std::unique_ptr<std::remove_pointer_t<ggml_gallocr_t>, GgmlGallocrDeleter> allocator(
            ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend())));
        if (allocator == nullptr || !ggml_gallocr_alloc_graph(allocator.get(), graph)) {
            throw runtime::CapacityError(
                "LFM2-Audio encoder graph does not fit in device memory for " + std::to_string(frames) + " feature frames");
        }

        // The graph input is time-major, the features are feature-major.
        std::vector<float> time_major(static_cast<size_t>(frames * config.n_mels));
        for (int64_t t = 0; t < frames; ++t) {
            for (int64_t m = 0; m < config.n_mels; ++m) {
                time_major[static_cast<size_t>(t * config.n_mels + m)] = features.values[static_cast<size_t>(m * frames + t)];
            }
        }
        ggml_backend_tensor_set(input.tensor, time_major.data(), 0, time_major.size() * sizeof(float));
        const auto positions = relative_position_embeddings(steps, config.hidden_size);
        ggml_backend_tensor_set(pos.tensor, positions.data(), 0, positions.size() * sizeof(float));

        core::set_backend_threads(execution.backend(), std::max(1, execution.config().threads));
        const ggml_status status = core::compute_backend_graph(execution.backend(), graph);
        ggml_backend_synchronize(execution.backend());
        if (status != GGML_STATUS_SUCCESS) {
            core::release_backend_graph_resources(execution.backend(), graph, true);
            throw std::runtime_error("LFM2-Audio encoder graph compute failed");
        }
        Lfm2AudioEmbeddings out;
        out.tokens = steps;
        out.hidden_size = config.output_size;
        out.values.resize(static_cast<size_t>(steps * config.output_size));
        ggml_backend_tensor_get(x.tensor, out.values.data(), 0, out.values.size() * sizeof(float));
        core::release_backend_graph_resources(execution.backend(), graph, true);
        debug::timing_log_scalar("lfm2_audio.encoder.ms", engine::debug::elapsed_ms(start));
        return out;
    }

    std::shared_ptr<const assets::TensorSource> source;
    Lfm2FastConformerEncoderConfig config;
    core::ExecutionContext & execution;
    audio::NemoMelFrontend frontend;
    EncoderWeights weights;
};

Lfm2FastConformerEncoderRuntime::Lfm2FastConformerEncoderRuntime(
    std::shared_ptr<const assets::TensorSource> source,
    const Lfm2FastConformerEncoderConfig & config,
    core::ExecutionContext & execution)
    : impl_(std::make_unique<Impl>(std::move(source), config, execution)) {}

Lfm2FastConformerEncoderRuntime::~Lfm2FastConformerEncoderRuntime() = default;

Lfm2AudioFeatures Lfm2FastConformerEncoderRuntime::extract_features(const std::vector<float> & samples) const {
    if (samples.size() < static_cast<size_t>(kHopLength)) {
        throw std::runtime_error("LFM2-Audio needs at least 10 ms of audio");
    }
    const auto features = impl_->frontend.extract_mono(
        samples, {true, audio::ValidFrameRule::FloorHops}, static_cast<size_t>(std::max(1, impl_->execution.config().threads)));
    Lfm2AudioFeatures out;
    out.n_mels = impl_->config.n_mels;
    out.frames = features.raw_frames;
    out.values.assign(features.values.begin(), features.values.begin() + out.n_mels * out.frames);
    return out;
}

Lfm2AudioEmbeddings Lfm2FastConformerEncoderRuntime::encode(const Lfm2AudioFeatures & features) {
    if (features.n_mels != impl_->config.n_mels || features.frames <= 0 ||
        static_cast<int64_t>(features.values.size()) != features.n_mels * features.frames) {
        throw std::runtime_error("LFM2-Audio encoder features have an unexpected shape");
    }
    return impl_->encode(features);
}

}  // namespace engine::community_models::lfm2_audio
