#include "engine/models/citrinet_asr/runtime.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/dsp.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/conditioning_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/optimizations/fast_conv_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/streaming_conv_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/models/citrinet_asr/assets.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::citrinet_asr {

struct BackendConv1dWeights {
    int64_t in_channels = 0;
    int64_t out_channels = 0;
    int64_t kernel = 0;
    int64_t stride = 1;
    int64_t dilation = 1;
    int64_t padding = 0;
    int64_t groups = 1;
    bool use_bias = false;
    modules::Conv1dWeights conv;
    modules::DepthwiseConv1dWeights depthwise;
};

struct BackendSeparableConvBn {
    BackendConv1dWeights depthwise;
    BackendConv1dWeights pointwise;
};

struct BackendConvBn {
    BackendConv1dWeights conv;
};

struct BackendSqueezeExciteWeights {
    BackendConv1dWeights fc1;
    BackendConv1dWeights fc2;
};

struct BackendJasperBlockWeights {
    bool separable = false;
    std::vector<BackendSeparableConvBn> separable_repeats;
    std::vector<BackendConvBn> conv_repeats;
    bool has_residual = false;
    BackendConv1dWeights residual_conv;
    bool has_se = false;
    BackendSqueezeExciteWeights se;
};

struct CitrinetBackendWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    std::vector<BackendJasperBlockWeights> blocks;
    BackendConv1dWeights decoder;
};

namespace {

using Clock = std::chrono::steady_clock;

struct FeaturePack {
    std::vector<float> values;
    int64_t raw_frames = 0;
    int64_t padded_frames = 0;
};

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

core::TensorValue store_weight(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const Conv1dWeights & tensor,
    assets::TensorStorageType storage_type) {
    return store.load_tensor_as_shape(
        source,
        tensor.weight_name,
        storage_type,
        tensor.weight_source_shape,
        core::TensorShape::from_dims({tensor.out_channels, tensor.in_channels / tensor.groups, tensor.kernel}));
}

BackendConv1dWeights make_backend_conv(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const Conv1dWeights & conv,
    assets::TensorStorageType storage_type) {
    BackendConv1dWeights weights;
    weights.in_channels = conv.in_channels;
    weights.out_channels = conv.out_channels;
    weights.kernel = conv.kernel;
    weights.stride = conv.stride;
    weights.dilation = conv.dilation;
    weights.padding = conv.padding;
    weights.groups = conv.groups;
    weights.use_bias = conv.use_bias;
    weights.conv.weight = store_weight(store, source, conv, storage_type);
    if (conv.use_bias) {
        weights.conv.bias = store.load_f32_tensor(source, *conv.bias_name, {conv.out_channels});
    }
    weights.depthwise.weight = weights.conv.weight;
    weights.depthwise.bias = weights.conv.bias;
    return weights;
}

std::vector<float> read_f32_tensor_values(
    const assets::TensorSource & source,
    const std::string & name,
    const std::vector<int64_t> & expected_shape) {
    const auto tensor = source.require_f32_tensor(name);
    if (tensor.shape.rank != expected_shape.size()) {
        throw std::runtime_error("Citrinet tensor rank mismatch: " + name);
    }
    for (size_t i = 0; i < expected_shape.size(); ++i) {
        if (tensor.shape.dims[i] != expected_shape[i]) {
            throw std::runtime_error("Citrinet tensor shape mismatch: " + name);
        }
    }
    return tensor.values;
}

BackendConv1dWeights make_backend_conv_bn(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const Conv1dWeights & conv,
    const BatchNorm1dWeights & bn,
    assets::TensorStorageType storage_type) {
    if (bn.channels != conv.out_channels) {
        throw std::runtime_error("Citrinet batch norm channel count does not match conv output");
    }
    BackendConv1dWeights weights;
    weights.in_channels = conv.in_channels;
    weights.out_channels = conv.out_channels;
    weights.kernel = conv.kernel;
    weights.stride = conv.stride;
    weights.dilation = conv.dilation;
    weights.padding = conv.padding;
    weights.groups = conv.groups;
    weights.use_bias = true;

    const auto tensor_shape = core::TensorShape::from_dims({conv.out_channels, conv.in_channels / conv.groups, conv.kernel});
    auto values = read_f32_tensor_values(source, conv.weight_name, conv.weight_source_shape);
    if (values.size() % static_cast<size_t>(conv.out_channels) != 0) {
        throw std::runtime_error("Citrinet conv weight size is not divisible by output channels");
    }
    std::vector<float> bias(static_cast<size_t>(conv.out_channels), 0.0f);
    if (conv.use_bias) {
        bias = read_f32_tensor_values(source, *conv.bias_name, {conv.out_channels});
    }
    const size_t values_per_output_channel = values.size() / static_cast<size_t>(conv.out_channels);
    for (int64_t channel = 0; channel < conv.out_channels; ++channel) {
        const size_t idx = static_cast<size_t>(channel);
        const float scale = bn.weight[idx] / std::sqrt(bn.running_var[idx] + 1.0e-3f);
        const float folded_bias = bn.bias[idx] - bn.running_mean[idx] * scale;
        const size_t offset = idx * values_per_output_channel;
        for (size_t i = 0; i < values_per_output_channel; ++i) {
            values[offset + i] *= scale;
        }
        bias[idx] = bias[idx] * scale + folded_bias;
    }

    weights.conv.weight = store.make_from_f32(tensor_shape, storage_type, std::move(values));
    weights.conv.bias = store.make_f32(core::TensorShape::from_dims({conv.out_channels}), std::move(bias));
    weights.depthwise.weight = weights.conv.weight;
    weights.depthwise.bias = weights.conv.bias;
    return weights;
}

std::shared_ptr<const CitrinetBackendWeights> load_backend_weights(
    const CitrinetWeights & weights,
    ggml_backend_t backend,
    core::BackendType backend_type,
    assets::TensorStorageType storage_type) {
    auto out = std::make_shared<CitrinetBackendWeights>();
    out->store = std::make_shared<core::BackendWeightStore>(
        backend, backend_type, "citrinet_asr.weights", 256ull * 1024ull * 1024ull);
    auto & store = *out->store;
    if (weights.source == nullptr) {
        throw std::runtime_error("Citrinet weights require a tensor source");
    }
    const auto & source = *weights.source;
    out->blocks.reserve(weights.blocks.size());
    for (const auto & block : weights.blocks) {
        BackendJasperBlockWeights dst;
        dst.separable = block.separable;
        dst.has_residual = block.has_residual;
        dst.has_se = block.has_se;
        dst.separable_repeats.reserve(block.separable_repeats.size());
        for (const auto & repeat : block.separable_repeats) {
            dst.separable_repeats.push_back({
                make_backend_conv(store, source, repeat.depthwise, storage_type),
                make_backend_conv_bn(store, source, repeat.pointwise, repeat.bn, storage_type),
            });
        }
        dst.conv_repeats.reserve(block.conv_repeats.size());
        for (const auto & repeat : block.conv_repeats) {
            dst.conv_repeats.push_back({
                make_backend_conv_bn(store, source, repeat.conv, repeat.bn, storage_type),
            });
        }
        if (block.has_residual) {
            dst.residual_conv = make_backend_conv_bn(store, source, block.residual_conv, block.residual_bn, storage_type);
        }
        if (block.has_se) {
            dst.se.fc1 = make_backend_conv(store, source, block.se.fc1, storage_type);
            dst.se.fc2 = make_backend_conv(store, source, block.se.fc2, storage_type);
        }
        out->blocks.push_back(std::move(dst));
    }
    out->decoder = make_backend_conv(store, source, weights.decoder, storage_type);
    store.upload();
    weights.source->release_storage();
    return out;
}

std::shared_ptr<const CitrinetWeights> require_weights(std::shared_ptr<const CitrinetWeights> weights) {
    if (weights == nullptr) {
        throw std::runtime_error("Citrinet runtime requires weights");
    }
    return weights;
}

core::TensorValue conv1d(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & x,
    const BackendConv1dWeights & conv) {
    if (conv.groups == conv.in_channels && conv.out_channels == conv.in_channels) {
        return modules::DepthwiseConv1dModule({
            conv.in_channels,
            conv.kernel,
            static_cast<int>(conv.stride),
            static_cast<int>(conv.padding),
            static_cast<int>(conv.dilation),
            conv.use_bias,
        }).build(ctx, x, conv.depthwise);
    }
    if (conv.groups != 1) {
        throw std::runtime_error("Citrinet only supports regular or depthwise Conv1d groups");
    }
    return modules::FastConv1dModule({
        conv.in_channels,
        conv.out_channels,
        conv.kernel,
        static_cast<int>(conv.stride),
        static_cast<int>(conv.padding),
        static_cast<int>(conv.dilation),
        conv.use_bias,
    }, modules::FastConv1dKind::MinittsFast1dIm2col).build(ctx, x, conv.conv);
}

core::TensorValue squeeze_excite(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & x,
    const BackendSqueezeExciteWeights & se) {
    return modules::SqueezeExcite1dModule({se.fc2.out_channels, se.fc1.out_channels, false}).build(
        ctx,
        x,
        {se.fc1.conv, se.fc2.conv});
}

core::TensorValue separable_conv_bn(
    core::ModuleBuildContext & ctx,
    core::TensorValue x,
    const BackendSeparableConvBn & layer) {
    x = conv1d(ctx, x, layer.depthwise);
    return conv1d(ctx, x, layer.pointwise);
}

core::TensorValue conv_bn(
    core::ModuleBuildContext & ctx,
    core::TensorValue x,
    const BackendConvBn & layer) {
    return conv1d(ctx, x, layer.conv);
}

core::TensorValue jasper_block(
    core::ModuleBuildContext & ctx,
    core::TensorValue x,
    const BackendJasperBlockWeights & block) {
    auto residual = x;
    if (block.separable) {
        for (size_t i = 0; i < block.separable_repeats.size(); ++i) {
            x = separable_conv_bn(ctx, x, block.separable_repeats[i]);
            if (i + 1 != block.separable_repeats.size()) {
                x = modules::ReluModule{}.build(ctx, x);
            }
        }
    } else {
        for (size_t i = 0; i < block.conv_repeats.size(); ++i) {
            x = conv_bn(ctx, x, block.conv_repeats[i]);
            if (i + 1 != block.conv_repeats.size()) {
                x = modules::ReluModule{}.build(ctx, x);
            }
        }
    }
    if (block.has_se) {
        x = squeeze_excite(ctx, x, block.se);
    }
    if (block.has_residual) {
        auto res = conv1d(ctx, residual, block.residual_conv);
        x = modules::ResidualAddModule{}.build(ctx, x, res);
    }
    x = modules::ReluModule{}.build(ctx, x);
    return x;
}

core::TensorValue build_citrinet_graph(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const CitrinetBackendWeights & weights) {
    auto x = input;
    for (size_t i = 0; i < weights.blocks.size(); ++i) {
        x = jasper_block(ctx, x, weights.blocks[i]);
    }
    x = conv1d(ctx, x, weights.decoder);
    return x;
}

}  // namespace

class CitrinetRuntime::Graph {
  public:
    Graph(
        std::shared_ptr<const CitrinetWeights> weights,
        std::shared_ptr<const CitrinetBackendWeights> backend_weights,
        int64_t frames,
        core::ExecutionContext & execution_context)
        : weights_(std::move(weights)),
          backend_weights_(std::move(backend_weights)),
          frames_(frames),
          backend_(execution_context.backend()),
          compute_threads_(std::max(1, execution_context.config().threads)) {
        const auto build_start = Clock::now();
        if (frames_ <= 0) {
            throw std::runtime_error("invalid Citrinet graph shape");
        }
        if (backend_ == nullptr) {
            throw std::runtime_error("Citrinet execution backend is not initialized");
        }
        ggml_init_params params{512ull * 1024ull * 1024ull, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize ggml context");
        }
        core::ModuleBuildContext build_ctx{
            ctx_.get(),
            "citrinet_asr",
            execution_context.backend_type(),
        };
        auto input = core::make_tensor(build_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, weights_->config.n_mels, frames_}));
        input_ = input.tensor;
        output_ = build_citrinet_graph(build_ctx, input, *backend_weights_).tensor;
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 16384, false);
        ggml_build_forward_expand(graph_, output_);
        buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), backend_);
        if (buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate graph");
        }
        if (engine::core::uses_host_graph_plan(backend_)) {
            const auto plan_start = Clock::now();
            plan_ = engine::core::create_backend_graph_plan_if_host(backend_, graph_);
            plan_create_ms_ = engine::debug::elapsed_ms(plan_start, Clock::now());
            if (plan_ == nullptr) {
                throw std::runtime_error("failed to create Citrinet graph plan");
            }
        }
        const auto build_end = Clock::now();
        debug::timing_log_scalar("citrinet.graph.build_ms", engine::debug::elapsed_ms(build_start, build_end));
        debug::timing_log_scalar("citrinet.graph.plan_create_ms", plan_create_ms_);
    }

    ~Graph() {
        if (plan_ != nullptr) {
            engine::core::free_backend_graph_plan(backend_, plan_);
        }
        if (buffer_ != nullptr) {
            ggml_backend_buffer_free(buffer_);
        }
    }

    bool matches(const CitrinetWeights & weights, int64_t frames, ggml_backend_t backend, int threads) const {
        return weights_.get() == &weights && frames_ == frames && backend_ == backend && compute_threads_ == std::max(1, threads);
    }

    CitrinetInferenceResult run(const std::vector<float> & features) {
        const auto run_start = Clock::now();
        if (features.size() != static_cast<size_t>(frames_ * weights_->config.n_mels)) {
            throw std::runtime_error("feature tensor size mismatch");
        }
        if (channels_first_.size() != features.size()) {
            channels_first_.resize(features.size());
        }
        for (int64_t frame = 0; frame < frames_; ++frame) {
            for (int64_t mel = 0; mel < weights_->config.n_mels; ++mel) {
                channels_first_[static_cast<size_t>(frame + frames_ * mel)] = features[static_cast<size_t>(frame * weights_->config.n_mels + mel)];
            }
        }
        ggml_backend_tensor_set(input_, channels_first_.data(), 0, channels_first_.size() * sizeof(float));
        core::set_backend_threads(backend_, compute_threads_);
        const auto compute_start = Clock::now();
        const ggml_status status = core::compute_backend_graph(backend_, graph_, plan_);
        ggml_backend_synchronize(backend_);
        const auto compute_end = Clock::now();
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Citrinet graph compute failed");
        }
        const size_t values = ggml_nelements(output_);
        if (raw_output_.size() != values) {
            raw_output_.resize(values);
        }
        ggml_backend_tensor_get(output_, raw_output_.data(), 0, raw_output_.size() * sizeof(float));
        const int64_t output_frames = output_->ne[0];
        const int64_t num_classes = output_->ne[1];
        if (row_major_output_.size() != values) {
            row_major_output_.resize(values);
        }
        for (int64_t frame = 0; frame < output_frames; ++frame) {
            for (int64_t cls = 0; cls < num_classes; ++cls) {
                row_major_output_[static_cast<size_t>(frame * num_classes + cls)] = raw_output_[static_cast<size_t>(frame + output_frames * cls)];
            }
        }
        const auto run_end = Clock::now();
        debug::timing_log_scalar("citrinet.graph.compute_ms", engine::debug::elapsed_ms(compute_start, compute_end));
        debug::timing_log_scalar("citrinet.infer_features_ms", engine::debug::elapsed_ms(run_start, run_end));
        CitrinetInferenceResult result;
        result.input_frames = frames_;
        result.output_frames = output_frames;
        result.num_classes = num_classes;
        result.logits = row_major_output_;
        return result;
    }

  private:
    std::shared_ptr<const CitrinetWeights> weights_;
    std::shared_ptr<const CitrinetBackendWeights> backend_weights_;
    int64_t frames_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_backend_t backend_ = nullptr;
    int compute_threads_ = 1;
    ggml_backend_buffer_t buffer_ = nullptr;
    ggml_backend_graph_plan_t plan_ = nullptr;
    double plan_create_ms_ = 0.0;
    std::vector<float> channels_first_;
    std::vector<float> raw_output_;
    std::vector<float> row_major_output_;
};

namespace {

std::vector<float> to_time_major_features(const engine::audio::AudioTensor & features) {
    if (features.shape.size() != 3 || features.shape[0] != 1) {
        throw std::runtime_error("expected single-batch audio feature tensor");
    }
    const int64_t feature_dim = features.shape[1];
    const int64_t frames = features.shape[2];
    std::vector<float> out(static_cast<size_t>(frames * feature_dim), 0.0f);
    for (int64_t frame = 0; frame < frames; ++frame) {
        for (int64_t feature = 0; feature < feature_dim; ++feature) {
            out[static_cast<size_t>(frame * feature_dim + feature)] =
                features.values[static_cast<size_t>(feature * frames + frame)];
        }
    }
    return out;
}

FeaturePack compute_citrinet_features(const std::vector<float> & waveform, const CitrinetWeights & weights);
std::vector<int32_t> greedy_ctc_ids(const CitrinetInferenceResult & result, int32_t blank_id);

FeaturePack extract_feature_pack_from_audio(
    const runtime::AudioBuffer & audio,
    const CitrinetWeights & weights) {
    return compute_citrinet_features(
        engine::audio::convert_interleaved_audio_to_mono_linear_resampled(
            audio.samples,
            audio.sample_rate,
            audio.channels,
            static_cast<int>(weights.config.sample_rate)),
        weights);
}

CitrinetTranscriptionResult make_transcription_result(
    const CitrinetWeights & weights,
    CitrinetInferenceResult inference) {
    auto ids = greedy_ctc_ids(inference, static_cast<int32_t>(weights.config.blank_id));
    CitrinetTranscriptionResult result;
    result.text = tokenizers::decode_sentencepiece(weights.tokenizer_pieces, ids);
    result.token_ids = std::move(ids);
    result.inference = std::move(inference);
    return result;
}

FeaturePack compute_citrinet_features(const std::vector<float> & waveform, const CitrinetWeights & weights) {
    const auto start = Clock::now();
    if (waveform.empty()) {
        throw std::runtime_error("waveform is empty");
    }
    const auto & cfg = weights.config;
    if (weights.window.size() != static_cast<size_t>(cfg.win_length)) {
        throw std::runtime_error("unexpected window size in checkpoint");
    }
    if (weights.fb.size() != static_cast<size_t>(cfg.n_mels * (cfg.n_fft / 2 + 1))) {
        throw std::runtime_error("unexpected mel filterbank size in checkpoint");
    }
    const engine::audio::STFTConfig stft_config{
        cfg.n_fft,
        cfg.hop_length,
        cfg.win_length,
        true,
        engine::audio::STFTPadMode::Constant,
        engine::audio::STFTFamily::Default,
    };
    engine::audio::AudioTensor magnitude;
    const auto stft_start = Clock::now();
    magnitude = engine::audio::STFT().compute_magnitude(
        waveform,
        weights.window,
        1,
        static_cast<int64_t>(waveform.size()),
        stft_config);
    debug::timing_log_scalar("audio.stft_ms", engine::debug::elapsed_ms(stft_start, Clock::now()));
    auto sparse_fb = engine::audio::MelFilterbank().prepare_sparse(
        engine::audio::AudioTensor{weights.fb, {cfg.n_mels, cfg.n_fft / 2 + 1}});
    auto log_mel = engine::audio::MelFilterbank().compute_custom_sparse_from_magnitude(
        magnitude.values,
        1,
        magnitude.shape[1],
        magnitude.shape[2],
        magnitude.shape[2],
        sparse_fb);
    constexpr float kLogZeroGuard = 5.960464477539063e-8f;
    for (float & value : log_mel.values) {
        value = std::log(value + kLogZeroGuard);
    }
    const int64_t raw_frames = log_mel.shape[2];

    FeaturePack out;
    out.raw_frames = raw_frames;
    out.padded_frames = raw_frames;
    auto normalized = engine::audio::FeatureNormalizer().compute(
        log_mel.values,
        std::vector<int64_t>{raw_frames},
        1,
        cfg.n_mels,
        raw_frames,
        engine::audio::FeatureNormalizeType::PerFeature);
    out.values = to_time_major_features(normalized.normalized);
    if (cfg.pad_to > 1) {
        const int64_t padded_frames = ((raw_frames + cfg.pad_to - 1) / cfg.pad_to) * cfg.pad_to;
        if (padded_frames != raw_frames) {
            out.values.resize(static_cast<size_t>(padded_frames * cfg.n_mels), 0.0f);
            out.padded_frames = padded_frames;
        }
    }
    const auto end = Clock::now();
    debug::timing_log_scalar("citrinet.features_ms", engine::debug::elapsed_ms(start, end));
    return out;
}

int64_t compute_output_frames(const CitrinetConfig & cfg, int64_t input_frames) {
    int64_t frames = input_frames;
    for (const auto & block : cfg.jasper) {
        if (block.stride > 1) {
            frames = (frames + block.stride - 1) / block.stride;
        }
    }
    return frames;
}

CitrinetInferenceResult truncate_result(const CitrinetInferenceResult & result, int64_t valid_frames) {
    if (valid_frames < 0 || valid_frames > result.output_frames) {
        throw std::runtime_error("invalid requested output truncation");
    }
    if (valid_frames == result.output_frames) {
        return result;
    }
    CitrinetInferenceResult truncated = result;
    truncated.output_frames = valid_frames;
    truncated.logits.resize(static_cast<size_t>(valid_frames * result.num_classes));
    return truncated;
}

std::vector<int32_t> greedy_ctc_ids(const CitrinetInferenceResult & result, int32_t blank_id) {
    std::vector<int32_t> ids;
    ids.reserve(static_cast<size_t>(result.output_frames));
    int32_t prev = -1;
    for (int64_t frame = 0; frame < result.output_frames; ++frame) {
        const float * row = result.logits.data() + static_cast<size_t>(frame * result.num_classes);
        int32_t best = 0;
        float best_value = row[0];
        for (int64_t cls = 1; cls < result.num_classes; ++cls) {
            if (row[cls] > best_value) {
                best_value = row[cls];
                best = static_cast<int32_t>(cls);
            }
        }
        if (best == prev) {
            continue;
        }
        prev = best;
        if (best == blank_id) {
            continue;
        }
        ids.push_back(best);
    }
    return ids;
}

}  // namespace

CitrinetInferenceResult infer_runtime_audio(
    CitrinetRuntime & runtime,
    const CitrinetWeights & weights,
    const runtime::AudioBuffer & audio) {
    const auto pack = extract_feature_pack_from_audio(audio, weights);
    auto result = runtime.infer_features(pack.values, pack.padded_frames);
    result = truncate_result(result, compute_output_frames(weights.config, pack.raw_frames));
    result.input_frames = pack.raw_frames;
    return result;
}

CitrinetRuntime::CitrinetRuntime(
    std::shared_ptr<const CitrinetWeights> weights,
    core::ExecutionContext & execution_context,
    assets::TensorStorageType weight_storage_type)
    : weights_(require_weights(std::move(weights))),
      backend_weights_(load_backend_weights(*weights_, execution_context.backend(), execution_context.backend_type(), weight_storage_type)),
      execution_context_(&execution_context) {}

CitrinetRuntime::~CitrinetRuntime() = default;

CitrinetRuntime::Graph & CitrinetRuntime::ensure_graph(int64_t frames) {
    if (frames <= 0) {
        throw std::runtime_error("frames must be positive");
    }
    if (execution_context_ == nullptr) {
        throw std::runtime_error("Citrinet execution context is not initialized");
    }
    if (!graph_ || !graph_->matches(*weights_, frames, execution_context_->backend(), execution_context_->config().threads)) {
        graph_ = std::make_unique<Graph>(weights_, backend_weights_, frames, *execution_context_);
    }
    return *graph_;
}

CitrinetInferenceResult CitrinetRuntime::infer_features(const std::vector<float> & features, int64_t frames) {
    if (features.size() != static_cast<size_t>(frames * weights_->config.n_mels)) {
        throw std::runtime_error("feature tensor size mismatch");
    }
    return ensure_graph(frames).run(features);
}

CitrinetInferenceResult CitrinetRuntime::infer_audio(const runtime::AudioBuffer & audio) {
    return infer_runtime_audio(*this, *weights_, audio);
}

CitrinetTranscriptionResult CitrinetRuntime::transcribe_features(const std::vector<float> & features, int64_t frames) {
    return make_transcription_result(*weights_, infer_features(features, frames));
}

CitrinetTranscriptionResult CitrinetRuntime::transcribe_audio(const runtime::AudioBuffer & audio) {
    return make_transcription_result(*weights_, infer_audio(audio));
}

}  // namespace engine::models::citrinet_asr
