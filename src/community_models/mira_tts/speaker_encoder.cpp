#include "engine/community_models/mira_tts/speaker_encoder.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/dsp.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/conditioning_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/optimizations/fast_conv_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::community_models::mira_tts {
namespace {

namespace binding = modules::binding;

constexpr int kSampleRate = 16000;
constexpr int64_t kReferenceSamples = 96000;
constexpr int64_t kMelBins = 128;
constexpr int64_t kEcapaChannels = 512;
constexpr int64_t kRes2Width = 64;
constexpr int64_t kPerceiverDim = 128;
constexpr int64_t kPerceiverLatents = 32;
constexpr int64_t kPerceiverHeads = 8;
constexpr int64_t kPerceiverInner = 512;
constexpr float kBatchNormEps = 1.0e-5F;

struct ContextDeleter {
    void operator()(ggml_context * context) const noexcept {
        if (context != nullptr) ggml_free(context);
    }
};

struct ConvWeights {
    modules::Conv1dWeights value;
    int64_t in_channels = 0;
    int64_t out_channels = 0;
    int64_t kernel = 1;
    int64_t padding = 0;
    int64_t dilation = 1;
};

struct TdnnWeights {
    ConvWeights conv;
    modules::BatchNorm1dEvalWeights norm;
};

struct SeRes2Weights {
    TdnnWeights first;
    std::vector<TdnnWeights> res2;
    TdnnWeights second;
    modules::LinearWeights se_first;
    modules::LinearWeights se_second;
};

struct PerceiverLayerWeights {
    modules::LinearWeights q;
    modules::LinearWeights kv;
    modules::LinearWeights out;
    modules::LinearWeights ff_in;
    modules::LinearWeights ff_out;
};

struct SpeakerWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    TdnnWeights input;
    std::vector<SeRes2Weights> blocks;
    ConvWeights mfa;
    modules::LinearWeights project_context;
    core::TensorValue latents;
    std::vector<PerceiverLayerWeights> perceiver;
    core::TensorValue norm_gamma;
    modules::LinearWeights quant_project;
};

std::vector<float> require_values(
    const assets::TensorSource & source,
    const std::string & name,
    int64_t size) {
    auto tensor = source.require_f32_tensor(name);
    if (static_cast<int64_t>(tensor.values.size()) != size) {
        throw std::runtime_error("MiraTTS tensor size mismatch: " + name);
    }
    return std::move(tensor.values);
}

modules::BatchNorm1dEvalWeights load_batch_norm(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t channels) {
    const auto gamma = require_values(source, prefix + ".weight", channels);
    const auto beta = require_values(source, prefix + ".bias", channels);
    const auto mean = require_values(source, prefix + ".running_mean", channels);
    const auto variance = require_values(source, prefix + ".running_var", channels);
    std::vector<float> scale(static_cast<size_t>(channels));
    std::vector<float> bias(static_cast<size_t>(channels));
    for (int64_t i = 0; i < channels; ++i) {
        scale[static_cast<size_t>(i)] = gamma[static_cast<size_t>(i)] /
            std::sqrt(variance[static_cast<size_t>(i)] + kBatchNormEps);
        bias[static_cast<size_t>(i)] = beta[static_cast<size_t>(i)] -
            mean[static_cast<size_t>(i)] * scale[static_cast<size_t>(i)];
    }
    return {
        store.make_f32(core::TensorShape::from_dims({channels}), scale),
        store.make_f32(core::TensorShape::from_dims({channels}), bias)};
}

ConvWeights load_conv(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    int64_t out_channels,
    int64_t in_channels,
    int64_t kernel,
    int64_t padding,
    int64_t dilation) {
    ConvWeights out;
    out.in_channels = in_channels;
    out.out_channels = out_channels;
    out.kernel = kernel;
    out.padding = padding;
    out.dilation = dilation;
    out.value.weight = store.load_tensor(
        source, prefix + ".weight", storage_type,
        {out_channels, in_channels, kernel});
    out.value.bias = store.load_f32_tensor(source, prefix + ".bias", {out_channels});
    return out;
}

TdnnWeights load_tdnn(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    int64_t out_channels,
    int64_t in_channels,
    int64_t kernel,
    int64_t padding,
    int64_t dilation) {
    return {
        load_conv(store, source, prefix + ".conv", storage_type,
                  out_channels, in_channels, kernel, padding, dilation),
        load_batch_norm(store, source, prefix + ".bn", out_channels)};
}

modules::LinearWeights load_linear(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    int64_t out_features,
    int64_t in_features,
    bool bias) {
    return binding::linear_from_source(
        store, source, prefix, storage_type, out_features, in_features, bias);
}

SpeakerWeights load_weights(
    const MiraTTSAssets & assets,
    core::ExecutionContext & execution,
    size_t context_bytes,
    assets::TensorStorageType linear_type,
    assets::TensorStorageType conv_type) {
    SpeakerWeights out;
    out.store = std::make_shared<core::BackendWeightStore>(
        execution.backend(), execution.backend_type(),
        "mira_tts.speaker_encoder.weights", context_bytes);
    const auto & source = *assets.speaker_encoder_weights;
    const std::string root = "speaker_encoder.speaker_encoder.";
    out.input = load_tdnn(
        *out.store, source, root + "layer1", conv_type,
        kEcapaChannels, kMelBins, 5, 2, 1);
    for (int layer = 2; layer <= 4; ++layer) {
        const std::string prefix = root + "layer" + std::to_string(layer) + ".se_res2block.";
        SeRes2Weights block;
        block.first = load_tdnn(
            *out.store, source, prefix + "0", conv_type,
            kEcapaChannels, kEcapaChannels, 1, 0, 1);
        const int64_t dilation = layer;
        for (int branch = 0; branch < 7; ++branch) {
            TdnnWeights branch_weights;
            branch_weights.conv = load_conv(
                *out.store, source,
                prefix + "1.convs." + std::to_string(branch), conv_type,
                kRes2Width, kRes2Width, 3, dilation, dilation);
            branch_weights.norm = load_batch_norm(
                *out.store, source,
                prefix + "1.bns." + std::to_string(branch), kRes2Width);
            block.res2.push_back(std::move(branch_weights));
        }
        block.second = load_tdnn(
            *out.store, source, prefix + "2", conv_type,
            kEcapaChannels, kEcapaChannels, 1, 0, 1);
        block.se_first = load_linear(
            *out.store, source, prefix + "3.linear1", linear_type, 128, 512, true);
        block.se_second = load_linear(
            *out.store, source, prefix + "3.linear2", linear_type, 512, 128, true);
        out.blocks.push_back(std::move(block));
    }
    out.mfa = load_conv(
        *out.store, source, root + "conv", conv_type,
        1536, 1536, 1, 0, 1);
    out.project_context.weight = out.store->load_tensor(
        source, "perceiver.proj_context.weight", linear_type, {128, 1536});
    out.project_context.bias = out.store->load_f32_tensor(
        source, "speaker_encoder.perceiver_sampler.proj_context.bias", {128});
    out.latents = out.store->load_f32_tensor(
        source, "perceiver.latents", {1, kPerceiverLatents, kPerceiverDim});
    for (int layer = 0; layer < 2; ++layer) {
        const std::string prefix = "perceiver.layers." + std::to_string(layer);
        const std::string bias_prefix = "speaker_encoder.perceiver_sampler.layers." +
            std::to_string(layer) + ".1.";
        PerceiverLayerWeights item;
        item.q = load_linear(*out.store, source, prefix + ".attn.q", linear_type, 512, 128, false);
        item.kv = load_linear(*out.store, source, prefix + ".attn.kv", linear_type, 1024, 128, false);
        item.out = load_linear(*out.store, source, prefix + ".attn.out", linear_type, 128, 512, false);
        item.ff_in.weight = out.store->load_tensor(
            source, prefix + ".ff.in.weight", linear_type, {682, 128});
        item.ff_in.bias = out.store->load_f32_tensor(source, bias_prefix + "0.bias", {682});
        item.ff_out.weight = out.store->load_tensor(
            source, prefix + ".ff.out.weight", linear_type, {128, 341});
        item.ff_out.bias = out.store->load_f32_tensor(source, bias_prefix + "2.bias", {128});
        out.perceiver.push_back(std::move(item));
    }
    out.norm_gamma = out.store->load_f32_tensor(
        source, "speaker_encoder.perceiver_sampler.norm.gamma", {128});
    out.quant_project.weight = out.store->load_tensor(
        source, "quantizer.project_in.weight", linear_type, {6, 128});
    out.quant_project.bias = out.store->load_f32_tensor(
        source, "speaker_encoder.quantizer.project_in.bias", {6});
    out.store->upload();
    return out;
}

core::TensorValue conv1d(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const ConvWeights & weights) {
    return modules::FastConv1dModule({
        weights.in_channels, weights.out_channels, weights.kernel, 1,
        static_cast<int>(weights.padding), static_cast<int>(weights.dilation), true},
        modules::FastConv1dKind::MinittsFast1dIm2col)
        .build(ctx, input, weights.value);
}

core::TensorValue tdnn(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const TdnnWeights & weights) {
    auto x = conv1d(ctx, input, weights.conv);
    x = modules::ReluModule{}.build(ctx, x);
    return modules::BatchNorm1dEvalModule({weights.conv.out_channels})
        .build(ctx, x, weights.norm);
}

core::TensorValue se_res2(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const SeRes2Weights & weights) {
    auto x = tdnn(ctx, input, weights.first);
    core::TensorValue merged;
    core::TensorValue previous;
    for (int branch = 0; branch < 8; ++branch) {
        auto chunk = modules::SliceModule({1, branch * kRes2Width, kRes2Width})
            .build(ctx, x);
        core::TensorValue current;
        if (branch == 7) {
            current = chunk;
        } else {
            if (branch > 0) {
                chunk = modules::AddModule{}.build(ctx, chunk, previous);
            }
            current = tdnn(ctx, chunk, weights.res2[static_cast<size_t>(branch)]);
            previous = current;
        }
        merged = merged.valid()
            ? modules::ConcatModule({1}).build(ctx, merged, current)
            : current;
    }
    x = tdnn(ctx, merged, weights.second);
    auto pooled = modules::ReduceMeanModule({2}).build(ctx, x);
    pooled = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, pooled);
    auto gate = modules::LinearModule({512, 128, true}).build(ctx, pooled, weights.se_first);
    gate = modules::ReluModule{}.build(ctx, gate);
    gate = modules::LinearModule({128, 512, true}).build(ctx, gate, weights.se_second);
    gate = modules::SigmoidModule{}.build(ctx, gate);
    gate = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, gate);
    gate = modules::RepeatModule({x.shape}).build(ctx, gate);
    return modules::AddModule{}.build(
        ctx, input, modules::MulModule{}.build(ctx, x, gate));
}

core::TensorValue reshape_heads(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input) {
    auto x = core::ensure_backend_addressable_layout(ctx, input);
    x = core::reshape_tensor(ctx, x, core::TensorShape::from_dims(
        {input.shape.dims[0], input.shape.dims[1], kPerceiverHeads,
         kPerceiverInner / kPerceiverHeads}));
    return modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, x);
}

core::TensorValue scale(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    float value) {
    return core::wrap_tensor(
        ggml_scale(ctx.ggml, input.tensor, value), input.shape, GGML_TYPE_F32);
}

core::TensorValue perceiver_attention(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & latents,
    const core::TensorValue & context,
    const PerceiverLayerWeights & weights) {
    const auto full_context = modules::ConcatModule({1}).build(ctx, latents, context);
    auto q = modules::LinearModule({128, 512, false}).build(ctx, latents, weights.q);
    auto kv = modules::LinearModule({128, 1024, false}).build(ctx, full_context, weights.kv);
    auto k = modules::SliceModule({2, 0, 512}).build(ctx, kv);
    auto v = modules::SliceModule({2, 512, 512}).build(ctx, kv);
    q = reshape_heads(ctx, q);
    k = reshape_heads(ctx, k);
    v = reshape_heads(ctx, v);
    auto scores = modules::MatMulModule{}.build(
        ctx, q, modules::TransposeModule({{0, 1, 3, 2}, 4}).build(ctx, k));
    scores = scale(ctx, scores, 1.0F / std::sqrt(64.0F));
    auto attention = core::wrap_tensor(
        ggml_soft_max(ctx.ggml,
            core::ensure_backend_addressable_layout(ctx, scores).tensor),
        scores.shape, GGML_TYPE_F32);
    auto x = modules::MatMulModule{}.build(ctx, attention, v);
    x = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, x);
    x = core::ensure_backend_addressable_layout(ctx, x);
    x = core::reshape_tensor(ctx, x, core::TensorShape::from_dims({1, 32, 512}));
    return modules::LinearModule({512, 128, false}).build(ctx, x, weights.out);
}

core::TensorValue perceiver_ff(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const PerceiverLayerWeights & weights) {
    auto x = modules::LinearModule({128, 682, true}).build(ctx, input, weights.ff_in);
    auto value = modules::SliceModule({2, 0, 341}).build(ctx, x);
    auto gate = modules::SliceModule({2, 341, 341}).build(ctx, x);
    gate = modules::GeluModule({modules::GeluApproximation::ExactErf}).build(ctx, gate);
    x = modules::MulModule{}.build(ctx, value, gate);
    return modules::LinearModule({341, 128, true}).build(ctx, x, weights.ff_out);
}

core::TensorValue rms_norm(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & gamma) {
    auto x = modules::RMSNormModule({128, 1.0e-5F, true, false})
        .build(ctx, input, {gamma, std::nullopt});
    return x;
}

core::TensorValue build_graph(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const SpeakerWeights & weights) {
    auto x = tdnn(ctx, input, weights.input);
    std::vector<core::TensorValue> outputs;
    for (const auto & block : weights.blocks) {
        x = se_res2(ctx, x, block);
        outputs.push_back(x);
    }
    x = modules::ConcatModule({1}).build(ctx, outputs[0], outputs[1]);
    x = modules::ConcatModule({1}).build(ctx, x, outputs[2]);
    x = modules::ReluModule{}.build(ctx, conv1d(ctx, x, weights.mfa));
    x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
    x = modules::LinearModule({1536, 128, true})
        .build(ctx, x, weights.project_context);
    auto latents = modules::RepeatModule({core::TensorShape::from_dims({1, 32, 128})})
        .build(ctx, weights.latents);
    for (const auto & layer : weights.perceiver) {
        latents = modules::AddModule{}.build(
            ctx, latents, perceiver_attention(ctx, latents, x, layer));
        latents = modules::AddModule{}.build(
            ctx, latents, perceiver_ff(ctx, latents, layer));
    }
    latents = rms_norm(ctx, latents, weights.norm_gamma);
    return modules::LinearModule({128, 6, true})
        .build(ctx, latents, weights.quant_project);
}

std::vector<float> prepare_reference(const runtime::AudioBuffer & audio) {
    if (audio.sample_rate <= 0 || audio.channels <= 0 || audio.samples.empty()) {
        throw std::runtime_error("MiraTTS requires non-empty reference audio");
    }
    auto mono = engine::audio::convert_interleaved_audio_to_mono_linear_resampled(
        audio.samples, audio.sample_rate, audio.channels, kSampleRate);
    if (mono.empty()) {
        throw std::runtime_error("MiraTTS reference audio contains no samples");
    }
    // librosa.load(..., duration=8, sr=16000) in the upstream encoder limits
    // the signal before volume normalization and six-second tiling/truncation.
    mono.resize(std::min<size_t>(mono.size(), 8 * kSampleRate));
    std::vector<float> magnitudes;
    magnitudes.reserve(mono.size());
    for (float sample : mono) magnitudes.push_back(std::abs(sample));
    std::sort(magnitudes.begin(), magnitudes.end());
    if (magnitudes.back() < 0.1F) {
        const float divisor = std::max(magnitudes.back(), 1.0e-3F);
        for (float & sample : mono) sample = sample / divisor * 0.1F;
        for (float & magnitude : magnitudes) magnitude = magnitude / divisor * 0.1F;
    }
    const auto first_significant = std::upper_bound(
        magnitudes.begin(), magnitudes.end(), 0.01F);
    const size_t significant = static_cast<size_t>(magnitudes.end() - first_significant);
    if (significant > 10) {
        const size_t begin = static_cast<size_t>(0.90 * significant);
        const size_t end = static_cast<size_t>(0.99 * significant);
        float sum = 0.0F;
        for (size_t i = begin; i < end; ++i) {
            sum += *(first_significant + static_cast<std::ptrdiff_t>(i));
        }
        const float volume = sum / static_cast<float>(std::max<size_t>(1, end - begin));
        const float gain = std::clamp(0.2F / volume, 0.1F, 10.0F);
        for (float & sample : mono) sample *= gain;
    }
    float peak = 0.0F;
    for (float sample : mono) peak = std::max(peak, std::abs(sample));
    if (peak > 1.0F) {
        for (float & sample : mono) sample /= peak;
    }
    std::vector<float> fixed(static_cast<size_t>(kReferenceSamples));
    for (int64_t i = 0; i < kReferenceSamples; ++i) {
        fixed[static_cast<size_t>(i)] = mono[static_cast<size_t>(i) % mono.size()];
    }
    return fixed;
}

std::vector<float> extract_mel(const runtime::AudioBuffer & audio, size_t threads) {
    const auto waveform = prepare_reference(audio);
    const engine::audio::STFTConfig stft{
        1024, 320, 640, true,
        engine::audio::STFTPadMode::Reflect,
        // torch.hann_window defaults to periodic=true in upstream MiraTTS.
        engine::audio::STFTFamily::Kokoro};
    const auto & window = engine::audio::get_cached_stft_window(stft);
    const auto magnitude = engine::audio::STFT().compute_magnitude(
        waveform, window, 1, kReferenceSamples, stft, threads);
    const int64_t frames = magnitude.shape.at(2);
    auto mel = engine::audio::MelFilterbank().compute(
        magnitude.values, 1, 513, frames,
        engine::audio::MelFilterbankConfig{
            16000, 1024, 128, 10.0F, 8000.0F, true});
    // AudioTensor is [B, mel, frames], which is already the graph's BCT layout.
    return std::move(mel.values);
}

}  // namespace

struct MiraSpeakerEncoder::Impl {
    Impl(
        const MiraTTSAssets & assets,
        core::ExecutionContext & execution,
        size_t weight_context_bytes,
        size_t graph_context_bytes,
        assets::TensorStorageType linear_type,
        assets::TensorStorageType conv_type)
        : execution(execution),
          weights(load_weights(
              assets, execution, weight_context_bytes, linear_type, conv_type)),
          graph_context_bytes(graph_context_bytes) {}

    ~Impl() {
        if (gallocr != nullptr) ggml_gallocr_free(gallocr);
    }

    void ensure_graph(int64_t frames) {
        if (ctx != nullptr && frames == graph_frames) return;
        if (gallocr != nullptr) {
            ggml_gallocr_free(gallocr);
            gallocr = nullptr;
        }
        ctx.reset();
        ggml_init_params params{graph_context_bytes, nullptr, true};
        ctx.reset(ggml_init(params));
        if (!ctx) throw std::runtime_error("MiraTTS failed to initialize speaker graph");
        core::ModuleBuildContext build_ctx{
            ctx.get(), "mira_tts.speaker_encoder", execution.backend_type()};
        auto input = core::make_tensor(
            build_ctx, GGML_TYPE_F32,
            core::TensorShape::from_dims({1, 128, frames}));
        input_tensor = input.tensor;
        output_tensor = build_graph(build_ctx, input, weights).tensor;
        ggml_set_output(output_tensor);
        graph = ggml_new_graph_custom(ctx.get(), 65536, false);
        ggml_build_forward_expand(graph, output_tensor);
        gallocr = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(execution.backend()));
        if (gallocr == nullptr || !ggml_gallocr_reserve(gallocr, graph) ||
            !ggml_gallocr_alloc_graph(gallocr, graph)) {
            throw std::runtime_error("MiraTTS failed to allocate speaker graph");
        }
        graph_frames = frames;
    }

    std::vector<int32_t> encode(const runtime::AudioBuffer & audio) {
        auto mel = extract_mel(audio, static_cast<size_t>(
            std::max(1, execution.config().threads)));
        if (mel.size() % 128 != 0) {
            throw std::runtime_error("MiraTTS mel feature shape mismatch");
        }
        const int64_t frames = static_cast<int64_t>(mel.size() / 128);
        ensure_graph(frames);
        ggml_backend_tensor_set(
            input_tensor, mel.data(), 0, mel.size() * sizeof(float));
        if (ggml_backend_graph_compute(execution.backend(), graph) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("MiraTTS speaker graph execution failed");
        }
        std::vector<float> projected(32 * 6);
        ggml_backend_tensor_get(
            output_tensor, projected.data(), 0,
            projected.size() * sizeof(float));
        std::vector<int32_t> codes(32, 0);
        for (int row = 0; row < 32; ++row) {
            int32_t code = 0;
            int32_t radix = 1;
            for (int dim = 0; dim < 6; ++dim) {
                const float value = projected[static_cast<size_t>(row * 6 + dim)];
                const float bounded = std::tanh(value + 0.3461989760398865F) *
                    1.501500129699707F - 0.5F;
                const int32_t digit = static_cast<int32_t>(std::nearbyint(bounded)) + 2;
                code += std::clamp(digit, 0, 3) * radix;
                radix *= 4;
            }
            codes[static_cast<size_t>(row)] = code;
        }
        return codes;
    }

    core::ExecutionContext & execution;
    SpeakerWeights weights;
    size_t graph_context_bytes = 0;
    int64_t graph_frames = 0;
    std::unique_ptr<ggml_context, ContextDeleter> ctx;
    ggml_cgraph * graph = nullptr;
    ggml_gallocr_t gallocr = nullptr;
    ggml_tensor * input_tensor = nullptr;
    ggml_tensor * output_tensor = nullptr;
};

MiraSpeakerEncoder::MiraSpeakerEncoder(
    const MiraTTSAssets & assets,
    core::ExecutionContext & execution,
    size_t weight_context_bytes,
    size_t graph_context_bytes,
    assets::TensorStorageType linear_storage_type,
    assets::TensorStorageType conv_storage_type)
    : impl_(std::make_unique<Impl>(
          assets, execution, weight_context_bytes, graph_context_bytes,
          linear_storage_type, conv_storage_type)) {}

MiraSpeakerEncoder::~MiraSpeakerEncoder() = default;

std::vector<int32_t> MiraSpeakerEncoder::encode(
    const runtime::AudioBuffer & reference_audio) {
    return impl_->encode(reference_audio);
}

}  // namespace engine::community_models::mira_tts
