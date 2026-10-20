#include "engine/community_models/xtts_v2/vocoder.h"
#include "engine/community_models/xtts_v2/frontend.h"
#include "engine/framework/audio/waveform_ops.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/constant_tensor_cache.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <ggml-alloc.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <utility>

namespace engine::models::xtts_v2 {
namespace {
namespace binding = modules::binding;
using core::TensorShape;
using core::TensorValue;
constexpr auto native = assets::TensorStorageType::Native;
constexpr std::array<int, 4> rates{8, 8, 2, 2}, kernels{16, 16, 4, 4};
constexpr std::array<int, 3> residual_kernels{3, 7, 11}, dilations{1, 3, 5};

struct DecoderStage {
    int channels = 0;
    modules::ConvTranspose1dWeights up;
    modules::Conv1dWeights speaker;
    std::array<std::array<std::pair<modules::Conv1dWeights, modules::Conv1dWeights>, 3>, 3> residuals;
};

struct SpeakerBlock {
    int in_channels = 0, channels = 0, stride = 1;
    modules::Conv2dWeights conv1, conv2;
    modules::ChannelAffineWeights norm1, norm2;
    modules::LinearWeights se1, se2;
    std::optional<modules::Conv2dWeights> downsample;
    std::optional<modules::ChannelAffineWeights> downsample_norm;
};

struct VocoderWeights {
    modules::Conv1dWeights pre, post, speaker;
    std::array<DecoderStage, 4> decoder;
    modules::Conv2dWeights speaker_stem;
    modules::ChannelAffineWeights speaker_stem_norm, attention_norm;
    std::vector<SpeakerBlock> speaker_blocks;
    modules::Conv1dWeights attention1, attention2;
    modules::LinearWeights projection;
};

// Cached graphs own their constants, allocation and backend capture resources.
// Neural operations below are built through framework modules.
struct AudioGraph {
    AudioGraph(core::ExecutionContext & execution, const char * name)
        : backend(execution.backend()), constants(backend, execution.config().threads, name, 64 * 1024) {
        ctx = ggml_init({8 * 1024 * 1024, nullptr, true});
        if (!ctx) throw std::runtime_error("XTTS v2 audio graph context allocation failed");
    }
    ~AudioGraph() {
        if (graph) core::release_backend_graph_resources(backend, graph);
        if (allocator) ggml_gallocr_free(allocator);
        if (ctx) ggml_free(ctx);
    }
    void allocate() {
        graph = ggml_new_graph_custom(ctx, 8192, false);
        ggml_set_output(output.tensor);
        ggml_build_forward_expand(graph, output.tensor);
        constants.ensure_uploaded();
        allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!allocator || !ggml_gallocr_alloc_graph(allocator, graph)) {
            throw std::runtime_error("XTTS v2 audio graph allocation failed");
        }
    }
    std::vector<float> run(const std::vector<float> & input_values, const float * speaker_values = nullptr) {
        ggml_backend_tensor_set(input.tensor, input_values.data(), 0, input_values.size() * sizeof(float));
        if (speaker.valid()) ggml_backend_tensor_set(speaker.tensor, speaker_values, 0, 512 * sizeof(float));
        if (core::compute_backend_graph(backend, graph) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("XTTS v2 audio graph compute failed");
        }
        std::vector<float> values(output.shape.num_elements());
        ggml_backend_tensor_get(output.tensor, values.data(), 0, values.size() * sizeof(float));
        return values;
    }
    ggml_backend_t backend;
    ggml_context * ctx = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_gallocr_t allocator = nullptr;
    core::ConstantTensorCache constants;
    TensorValue input, speaker, output;
    int frames = 0;
};

std::unique_ptr<AudioGraph> build_decoder(core::ExecutionContext & execution, const VocoderWeights & weights, int frames) {
    auto graph = std::make_unique<AudioGraph>(execution, "xtts_v2.decoder.constants");
    graph->frames = frames;
    core::ModuleBuildContext ctx{graph->ctx, "xtts_v2.decoder", execution.backend_type()};
    graph->input = core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({1, 1024, frames}));
    graph->speaker = core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({1, 512, 1}));
    ggml_set_input(graph->input.tensor);
    ggml_set_input(graph->speaker.tensor);

    // Keep the two upstream align_corners=False interpolation stages. The first
    // is an exact integer scale. The second uses the requested 24000/22050
    // factor, rather than recomputing a scale from its rounded output length.
    const int first_frames = 4 * frames;
    const double scale = 24000.0 / 22050.0;
    const int output_frames = static_cast<int>(first_frames * scale);
    auto first = modules::Interpolate1dModule({first_frames, modules::Interpolate1dMode::Linear}).build(ctx, graph->input);
    auto rows = modules::TransposeModule({{1, 0}, 2}).build(ctx,
        core::reshape_tensor(ctx, first, TensorShape::from_dims({1024, first_frames})));
    rows = core::ensure_backend_addressable_layout(ctx, rows);
    std::vector<int32_t> left(output_frames), right(output_frames);
    std::vector<float> left_weight(output_frames), right_weight(output_frames);
    for (int t = 0; t < output_frames; ++t) {
        const float position = std::clamp(static_cast<float>((t + .5) / scale - .5), 0.f, float(first_frames - 1));
        left[t] = static_cast<int>(position);
        right[t] = std::min(left[t] + 1, first_frames - 1);
        right_weight[t] = position - left[t];
        left_weight[t] = 1.f - right_weight[t];
    }
    const auto index_shape = TensorShape::from_dims({output_frames});
    auto left_index = graph->constants.make_tensor(index_shape, GGML_TYPE_I32, left.data(), left.size() * sizeof(int32_t));
    auto right_index = graph->constants.make_tensor(index_shape, GGML_TYPE_I32, right.data(), right.size() * sizeof(int32_t));
    const auto coefficient_shape = TensorShape::from_dims({output_frames, 1});
    auto a = graph->constants.make_f32(coefficient_shape, left_weight);
    auto b = graph->constants.make_f32(coefficient_shape, right_weight);
    const modules::EmbeddingModule lookup({first_frames, 1024});
    auto left_values = lookup.build(ctx, left_index, rows);
    auto right_values = lookup.build(ctx, right_index, rows);
    a = modules::RepeatModule({left_values.shape}).build(ctx, a);
    b = modules::RepeatModule({right_values.shape}).build(ctx, b);
    auto interpolated = modules::AddModule().build(ctx,
        modules::MulModule().build(ctx, left_values, a), modules::MulModule().build(ctx, right_values, b));
    auto x = modules::TransposeModule({{1, 0}, 2}).build(ctx, interpolated);
    x = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, x), TensorShape::from_dims({1, 1024, output_frames}));
    x = modules::Conv1dModule({1024, 512, 7, 1, 3, 1}).build(ctx, x, weights.pre);
    auto speaker = modules::Conv1dModule({512, 512, 1, 1, 0, 1}).build(ctx, graph->speaker, weights.speaker);
    x = modules::AddModule().build(ctx, x, modules::RepeatModule({x.shape}).build(ctx, speaker));
    for (size_t stage = 0; stage < weights.decoder.size(); ++stage) {
        const auto & w = weights.decoder[stage];
        x = modules::LeakyReluModule({.1f}).build(ctx, x);
        const int padding = (kernels[stage] - rates[stage]) / 2;
        // Native CPU transposed convolution requires zero padding; symmetric
        // slicing is equivalent to PyTorch's padding (also used by Sam Audio).
        const int backend_padding = execution.backend_type() == core::BackendType::Cpu ? 0 : padding;
        x = modules::ConvTranspose1dModule({2 * w.channels, w.channels, kernels[stage], rates[stage],
            backend_padding}).build(ctx, x, w.up);
        if (backend_padding == 0 && padding != 0) {
            x = modules::SliceModule({2, padding, x.shape.dims[2] - 2 * padding}).build(ctx, x);
        }
        speaker = modules::Conv1dModule({512, w.channels, 1, 1, 0, 1}).build(ctx, graph->speaker, w.speaker);
        x = modules::AddModule().build(ctx, x, modules::RepeatModule({x.shape}).build(ctx, speaker));
        std::optional<TensorValue> sum;
        for (size_t kernel = 0; kernel < residual_kernels.size(); ++kernel) {
            auto block = x;
            for (size_t dilation = 0; dilation < dilations.size(); ++dilation) {
                auto residual = block;
                block = modules::LeakyReluModule({.1f}).build(ctx, block);
                block = modules::Conv1dModule({w.channels, w.channels, residual_kernels[kernel], 1,
                    dilations[dilation] * (residual_kernels[kernel] - 1) / 2, dilations[dilation]}).build(ctx, block, w.residuals[kernel][dilation].first);
                block = modules::LeakyReluModule({.1f}).build(ctx, block);
                block = modules::Conv1dModule({w.channels, w.channels, residual_kernels[kernel], 1,
                    (residual_kernels[kernel] - 1) / 2, 1}).build(ctx, block, w.residuals[kernel][dilation].second);
                block = modules::ResidualAddModule().build(ctx, block, residual);
            }
            sum = sum ? modules::AddModule().build(ctx, *sum, block) : block;
        }
        x = modules::ScaleBiasModule({1.f / 3, 0}).build(ctx, *sum);
    }
    x = modules::LeakyReluModule({.01f}).build(ctx, x);
    x = modules::Conv1dModule({32, 1, 7, 1, 3, 1, false}).build(ctx, x, weights.post);
    graph->output = modules::TanhModule().build(ctx, x);
    graph->allocate();
    return graph;
}

std::unique_ptr<AudioGraph> build_speaker(core::ExecutionContext & execution, const VocoderWeights & weights, int frames) {
    auto graph = std::make_unique<AudioGraph>(execution, "xtts_v2.speaker.constants");
    graph->frames = frames;
    core::ModuleBuildContext ctx{graph->ctx, "xtts_v2.speaker", execution.backend_type()};
    graph->input = core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({1, 64, frames}));
    ggml_set_input(graph->input.tensor);
    auto x = modules::GroupNormModule({64, 64, 1e-5f, false, false}).build(ctx, graph->input, {});
    x = core::reshape_tensor(ctx, x, TensorShape::from_dims({1, 1, 64, frames}));
    x = modules::Conv2dModule({1, 32, 3, 3, 1, 1, 1, 1}).build(ctx, x, weights.speaker_stem);
    x = modules::ReluModule().build(ctx, x);
    x = modules::BatchNorm2dEvalModule({32}).build(ctx, x, {weights.speaker_stem_norm.scale, weights.speaker_stem_norm.bias});
    for (const auto & w : weights.speaker_blocks) {
        auto residual = x;
        x = modules::Conv2dModule({w.in_channels, w.channels, 3, 3, w.stride, w.stride, 1, 1, 1, 1, false}).build(ctx, x, w.conv1);
        x = modules::ReluModule().build(ctx, x);
        x = modules::BatchNorm2dEvalModule({w.channels}).build(ctx, x, {w.norm1.scale, w.norm1.bias});
        x = modules::Conv2dModule({w.channels, w.channels, 3, 3, 1, 1, 1, 1, 1, 1, false}).build(ctx, x, w.conv2);
        x = modules::BatchNorm2dEvalModule({w.channels}).build(ctx, x, {w.norm2.scale, w.norm2.bias});
        auto pooled = core::reshape_tensor(ctx, x, TensorShape::from_dims({1, w.channels, x.shape.dims[2] * x.shape.dims[3]}));
        pooled = modules::ReduceMeanModule({2}).build(ctx, pooled);
        pooled = core::reshape_tensor(ctx, pooled, TensorShape::from_dims({1, w.channels}));
        pooled = modules::LinearModule({w.channels, w.channels / 8, true}).build(ctx, pooled, w.se1);
        pooled = modules::ReluModule().build(ctx, pooled);
        pooled = modules::LinearModule({w.channels / 8, w.channels, true}).build(ctx, pooled, w.se2);
        pooled = modules::SigmoidModule().build(ctx, pooled);
        pooled = core::reshape_tensor(ctx, pooled, TensorShape::from_dims({1, w.channels, 1, 1}));
        x = modules::MulModule().build(ctx, x, modules::RepeatModule({x.shape}).build(ctx, pooled));
        if (w.downsample) {
            residual = modules::Conv2dModule({w.in_channels, w.channels, 1, 1, 2, 2, 0, 0, 1, 1, false}).build(ctx, residual, *w.downsample);
            residual = modules::BatchNorm2dEvalModule({w.channels}).build(ctx, residual,
                {w.downsample_norm->scale, w.downsample_norm->bias});
        }
        x = modules::ReluModule().build(ctx, modules::ResidualAddModule().build(ctx, x, residual));
    }
    x = core::reshape_tensor(ctx, x, TensorShape::from_dims({1, 2048, x.shape.dims[3]}));
    auto attention = modules::Conv1dModule({2048, 128, 1}).build(ctx, x, weights.attention1);
    attention = modules::ReluModule().build(ctx, attention);
    attention = modules::BatchNorm1dEvalModule({128}).build(ctx, attention, {weights.attention_norm.scale, weights.attention_norm.bias});
    attention = modules::Conv1dModule({128, 2048, 1}).build(ctx, attention, weights.attention2);
    attention = modules::SoftmaxModule().build(ctx, attention);
    auto statistics = modules::WeightedStatisticsModule({2, 1e-5f, 1e30f}).build(ctx, x, attention);
    auto pooled = modules::ConcatModule({1}).build(ctx, statistics.mean, statistics.stddev);
    pooled = core::reshape_tensor(ctx, pooled, TensorShape::from_dims({1, 4096}));
    pooled = modules::LinearModule({4096, 512, true}).build(ctx, pooled, weights.projection);
    // The framework RMSNorm/PixelNorm epsilon rules differ from F.normalize.
    graph->output = core::wrap_tensor(ggml_l2_norm(ctx.ggml, pooled.tensor, 1e-12f), pooled.shape);
    graph->allocate();
    return graph;
}
}  // namespace

struct XttSv2Vocoder::Impl {
    core::ExecutionContext * execution = nullptr;
    std::shared_ptr<core::BackendWeightStore> store;
    VocoderWeights weights;
    std::unique_ptr<AudioGraph> decoder, speaker;
    MelFrontend frontend{MelPurpose::Speaker};
};

XttSv2Vocoder::XttSv2Vocoder() = default;
XttSv2Vocoder::~XttSv2Vocoder() = default;

bool XttSv2Vocoder::load(const XttSv2Assets & assets, core::ExecutionContext & execution) {
    impl_ = std::make_unique<Impl>();
    auto & c = *impl_;
    c.execution = &execution;
    c.store = std::make_shared<core::BackendWeightStore>(execution.backend(), execution.backend_type(), "xtts_v2.vocoder.weights", 8 * 1024 * 1024);
    const auto & source = *assets.vocoder_source;
    auto & store = *c.store;
    auto & w = c.weights;
    // This published GGUF calls the BatchNorm statistics rm/rv. Fold its
    // model-specific names locally; execution still uses framework BatchNorm.
    auto batch_norm = [&](const std::string & name, int channels) {
        const auto gamma = source.require_f32(name + ".weight", {channels});
        const auto beta = source.require_f32(name + ".bias", {channels});
        const auto mean = source.require_f32(name + ".rm", {channels});
        const auto variance = source.require_f32(name + ".rv", {channels});
        std::vector<float> scale(channels), bias(channels);
        for (int i = 0; i < channels; ++i) {
            scale[i] = gamma[i] / std::sqrt(variance[i] + 1e-5f);
            bias[i] = beta[i] - mean[i] * scale[i];
        }
        const auto shape = TensorShape::from_dims({channels});
        return modules::ChannelAffineWeights{store.make_f32(shape, std::move(scale)),
            store.make_f32(shape, std::move(bias))};
    };
    const std::string prefix = "hifigan_decoder.";
    w.pre = binding::conv1d_from_named_source(store, source, prefix + "dec.conv_pre.weight", prefix + "dec.conv_pre.bias", native);
    // Legacy GGUF omits the leading singleton output-channel dimension.
    w.post.weight = store.load_tensor_as_shape(source, prefix + "dec.conv_post.weight", native,
        {32, 7}, TensorShape::from_dims({1, 32, 7}));
    w.speaker = binding::conv1d_from_named_source(store, source, prefix + "dec.cond_layer.weight", prefix + "dec.cond_layer.bias", native);
    for (size_t stage = 0; stage < w.decoder.size(); ++stage) {
        auto & dst = w.decoder[stage];
        dst.channels = 256 >> stage;
        const auto p = prefix + "dec.ups." + std::to_string(stage);
        dst.up = binding::conv_transpose1d_from_named_source(store, source, p, p + ".bias", native);
        const auto speaker = prefix + "dec.conds." + std::to_string(stage);
        dst.speaker = binding::conv1d_from_named_source(store, source, speaker + ".weight", speaker + ".bias", native);
        for (size_t kernel = 0; kernel < residual_kernels.size(); ++kernel) {
            for (size_t dilation = 0; dilation < dilations.size(); ++dilation) {
                const auto p = prefix + "dec.resb." + std::to_string(3 * stage + kernel);
                const auto first = p + ".c1." + std::to_string(dilation), second = p + ".c2." + std::to_string(dilation);
                dst.residuals[kernel][dilation] = {
                    binding::conv1d_from_named_source(store, source, first, first + ".bias", native),
                    binding::conv1d_from_named_source(store, source, second, second + ".bias", native)};
            }
        }
    }
    w.speaker_stem = binding::conv2d_from_source(store, source, prefix + "se.conv1", native, 32, 1, 3, 3, true);
    w.speaker_stem_norm = batch_norm(prefix + "se.bn1", 32);
    const std::array<int, 4> channels{32, 64, 128, 256}, layers{3, 4, 6, 3};
    int in_channels = 32;
    for (size_t stage = 0; stage < layers.size(); ++stage) {
        for (int block = 0; block < layers[stage]; ++block) {
            SpeakerBlock dst;
            dst.in_channels = in_channels;
            dst.channels = channels[stage];
            dst.stride = stage > 0 && block == 0 ? 2 : 1;
            const auto p = prefix + "se.layer" + std::to_string(stage + 1) + "." + std::to_string(block);
            dst.conv1 = binding::conv2d_from_source(store, source, p + ".conv1", native, dst.channels, in_channels, 3, 3, false);
            dst.conv2 = binding::conv2d_from_source(store, source, p + ".conv2", native, dst.channels, dst.channels, 3, 3, false);
            dst.norm1 = batch_norm(p + ".bn1", dst.channels);
            dst.norm2 = batch_norm(p + ".bn2", dst.channels);
            dst.se1 = binding::linear_from_source(store, source, p + ".se.fc.0", native, dst.channels / 8, dst.channels, true);
            dst.se2 = binding::linear_from_source(store, source, p + ".se.fc.2", native, dst.channels, dst.channels / 8, true);
            if (dst.stride == 2) {
                dst.downsample = binding::conv2d_from_source(store, source, p + ".downsample.0", native, dst.channels, in_channels, 1, 1, false);
                dst.downsample_norm = batch_norm(p + ".downsample.1", dst.channels);
            }
            w.speaker_blocks.push_back(std::move(dst));
            in_channels = channels[stage];
        }
    }
    w.attention1 = binding::conv1d_from_named_source(store, source, prefix + "se.attention.0.weight", prefix + "se.attention.0.bias", native);
    w.attention2 = binding::conv1d_from_named_source(store, source, prefix + "se.attention.3.weight", prefix + "se.attention.3.bias", native);
    w.attention_norm = batch_norm(prefix + "se.attention.2", 128);
    w.projection = binding::linear_from_source(store, source, prefix + "se.fc", native, 512, 4096, true);
    store.upload();
    return true;
}

std::vector<float> XttSv2Vocoder::generate(const float * latent, int frames, const float * speaker) const {
    if (!impl_ || !latent || !speaker || frames <= 0) throw std::runtime_error("XTTS v2 requires decoder latents and a speaker embedding");
    auto & c = *impl_;
    if (!c.decoder || c.decoder->frames != frames) c.decoder = build_decoder(*c.execution, c.weights, frames);
    return c.decoder->run(std::vector<float>(latent, latent + 1024 * frames), speaker);
}

std::vector<float> XttSv2Vocoder::speaker_embed(const float * pcm, int samples) const {
    if (!impl_) throw std::runtime_error("XTTS v2 speaker weights are not loaded");
    auto & c = *impl_;
    auto audio = audio::resample_mono_torchaudio_sinc_hann(std::vector<float>(pcm, pcm + samples), 22050, 16000);
    if (audio.size() < 2) throw std::runtime_error("XTTS v2 speaker reference is too short");
    const float first = audio[0] - .97f * audio[1];
    audio::apply_preemphasis_in_place(audio, .97f);
    audio[0] = first;
    auto mel = c.frontend.compute(audio, c.execution->config().threads);
    const int frames = mel.size() / 64;
    debug::trace_log_f32("xtts_v2.speaker.power_mel", {64, frames}, mel);
    for (auto & value : mel) value = std::log(value + 1e-6f);
    if (!c.speaker || c.speaker->frames != frames) c.speaker = build_speaker(*c.execution, c.weights, frames);
    auto result = c.speaker->run(mel);
    debug::trace_log_f32("xtts_v2.speaker.embedding", {512}, result);
    return result;
}
}  // namespace engine::models::xtts_v2
