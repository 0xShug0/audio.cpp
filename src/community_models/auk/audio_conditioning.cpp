#include "engine/community_models/auk/conditioning.h"
#include "engine/framework/modules/attention/transformer_blocks.h"
#include "engine/framework/modules/weight_binding.h"
#include "engine/framework/runtime/graph_optimizer.h"
#include "engine/framework/audio/conversion.h"

#include <ggml-alloc.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <unordered_set>

namespace engine::models::auk {
namespace {
using core::TensorValue;
using core::TensorShape;

struct AudioWeights {
    modules::Conv1dWeights conv1;
    modules::Conv1dWeights conv2;
    std::array<modules::TransformerEncoderBlockWeights, 32> layers;
    modules::NormWeights norm;
    modules::LinearWeights projection;
    TensorValue positions;
};

AudioWeights load_audio_weights(core::BackendWeightStore & store, const assets::TensorSource & source) {
    const std::string root = "thinker.audio_tower.";
    const auto linear = [&](const std::string & name, int out, int in, bool bias = true) {
        return modules::binding::linear_from_source(store, source, root + name,
            assets::TensorStorageType::Native, out, in, bias);
    };
    AudioWeights weights;
    weights.conv1 = {store.load_tensor(source, root + "conv1.weight", assets::TensorStorageType::Native, {1280, 128, 3}),
                     store.load_f32_tensor(source, root + "conv1.bias", {1280})};
    weights.conv2 = {store.load_tensor(source, root + "conv2.weight", assets::TensorStorageType::Native, {1280, 1280, 3}),
                     store.load_f32_tensor(source, root + "conv2.bias", {1280})};
    // The generic encoder block uses one bias policy for all projections.
    // Qwen's key projection has no bias; an exact zero supplies that contract.
    const auto zero_bias = store.make_f32(TensorShape::from_dims({1280}), std::vector<float>(1280, 0.0F));
    for (int i = 0; i < 32; ++i) {
        const auto name = "layers." + std::to_string(i) + ".";
        auto & layer = weights.layers[i];
        layer.norm1 = modules::binding::norm_from_source(store, source, root + name + "self_attn_layer_norm", 1280);
        layer.norm2 = modules::binding::norm_from_source(store, source, root + name + "final_layer_norm", 1280);
        const auto q = linear(name + "self_attn.q_proj", 1280, 1280);
        const auto k = linear(name + "self_attn.k_proj", 1280, 1280, false);
        const auto v = linear(name + "self_attn.v_proj", 1280, 1280);
        const auto out = linear(name + "self_attn.out_proj", 1280, 1280);
        layer.self_attention.q_weight = q.weight;
        layer.self_attention.q_bias = q.bias;
        layer.self_attention.k_weight = k.weight;
        layer.self_attention.k_bias = zero_bias;
        layer.self_attention.v_weight = v.weight;
        layer.self_attention.v_bias = v.bias;
        layer.self_attention.out_weight = out.weight;
        layer.self_attention.out_bias = out.bias;
        const auto fc1 = linear(name + "fc1", 5120, 1280);
        const auto fc2 = linear(name + "fc2", 1280, 5120);
        layer.feed_forward = {fc1.weight, fc1.bias, fc2.weight, fc2.bias};
    }
    weights.norm = modules::binding::norm_from_source(store, source, root + "ln_post", 1280);
    weights.projection = linear("proj", 2048, 1280);
    std::vector<float> positions(100 * 1280);
    const float increment = static_cast<float>(std::log(10000.0) / 639.0);
    for (int channel = 0; channel < 640; ++channel) {
        const float frequency = std::exp(-increment * channel);
        for (int frame = 0; frame < 100; ++frame) {
            const float phase = frame * frequency;
            positions[frame * 1280 + channel] = std::sin(phase);
            positions[frame * 1280 + channel + 640] = std::cos(phase);
        }
    }
    weights.positions = store.make_f32(TensorShape::from_dims({100, 1280}), positions);
    return weights;
}

struct AudioGraph {
    core::ExecutionContext & execution;
    int64_t frames;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context{nullptr, ggml_free};
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator{nullptr, ggml_gallocr_free};
    ggml_cgraph * graph = nullptr;
    TensorValue input;
    TensorValue output;

    AudioGraph(core::ExecutionContext & execution_, const AudioWeights & weights, int64_t frames_)
        : execution(execution_), frames(frames_) {
        const size_t chunks = (frames + 199) / 200;
        const size_t groups = (frames >= 200 ? 1 : 0) + (frames % 200 != 0 ? 1 : 0);
        // Full chunks share one transformer stack; only convolutions expand per chunk.
        const size_t nodes = 4096 + groups * 8192 + chunks * 32;
        context.reset(ggml_init({nodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(nodes, false), nullptr, true}));
        if (!context) throw std::runtime_error("AuK audio conditioning context allocation failed");
        core::ModuleBuildContext ctx{context.get(), "auk.audio_conditioning", execution.backend_type()};
        input = core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({1, 128, frames}));
        ggml_set_input(input.tensor);
        ggml_tensor * joined = nullptr;
        int64_t total = 0;
        for (int64_t start = 0; start < frames;) {
            const int64_t length = std::min<int64_t>(200, frames - start);
            const int64_t batch = length == 200 ? (frames - start) / 200 : 1;
            auto * slice = ggml_view_3d(ctx.ggml, input.tensor, length, 128, batch,
                input.tensor->nb[1], length * sizeof(float), start * sizeof(float));
            auto x = core::wrap_tensor(ggml_cont(ctx.ggml, slice), TensorShape::from_dims({batch, 128, length}));
            // Processing the valid chunk alone gives zero padding after conv1,
            // matching Python's explicit mask before the stride-two conv2.
            x = modules::Conv1dModule({128, 1280, 3, 1, 1, 1, true}).build(ctx, x, weights.conv1);
            x = modules::GeluModule({modules::GeluApproximation::ExactErf}).build(ctx, x);
            x = modules::Conv1dModule({1280, 1280, 3, 2, 1, 1, true}).build(ctx, x, weights.conv2);
            x = modules::GeluModule({modules::GeluApproximation::ExactErf}).build(ctx, x);
            const int64_t tokens = (length + 1) / 2;
            auto * btc = ggml_cont(ctx.ggml, ggml_transpose(ctx.ggml, x.tensor));
            auto * position = ggml_view_2d(ctx.ggml, weights.positions.tensor, 1280, tokens,
                weights.positions.tensor->nb[1], 0);
            x = core::wrap_tensor(ggml_add(ctx.ggml, btc, position), TensorShape::from_dims({batch, tokens, 1280}));
            modules::TransformerEncoderBlockConfig config{1280, 20, 5120, 1.0e-5F, true};
            config.projection_precision = GGML_PREC_F32;
            config.attention_precision = GGML_PREC_F32;
            for (const auto & layer : weights.layers) {
                x = modules::TransformerEncoderBlockModule(config).build(ctx, x, layer);
            }
            // Batch keeps attention within each chunk while LinearModule packs projections.
            auto * packed = ggml_reshape_2d(ctx.ggml, ggml_cont(ctx.ggml, x.tensor), 1280, tokens * batch);
            joined = joined ? ggml_concat(ctx.ggml, joined, packed, 1) : packed;
            total += tokens * batch;
            start += length * batch;
        }
        const int64_t pooled = total / 2;
        auto * even = ggml_view_2d(ctx.ggml, joined, 1280, pooled, joined->nb[1] * 2, 0);
        auto * odd = ggml_view_2d(ctx.ggml, joined, 1280, pooled, joined->nb[1] * 2, joined->nb[1]);
        auto x = core::wrap_tensor(ggml_scale(ctx.ggml, ggml_add(ctx.ggml, even, odd), 0.5F),
            TensorShape::from_dims({1, pooled, 1280}));
        x = modules::LayerNormModule({1280, 1.0e-5F}).build(ctx, x, weights.norm);
        output = modules::LinearModule({1280, 2048, true, GGML_PREC_F32}).build(ctx, x, weights.projection);
        ggml_set_output(output.tensor);
        graph = ggml_new_graph_custom(context.get(), nodes, false);
        ggml_build_forward_expand(graph, output.tensor);
        auto optimization = runtime::graph_optimization_options_for_backend(runtime::GraphOptimizationBackend::Gpu);
        // Identity folding updates src edges but not the storage referenced by views.
        std::unordered_set<const ggml_tensor *> view_backings;
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            if (const auto * backing = ggml_graph_node(graph, i)->view_src) view_backings.insert(backing);
        }
        optimization.exclude_rewrite_user_data = &view_backings;
        optimization.exclude_rewrite = [](const runtime::GraphOptimizationRewriteContext & rewrite, const void * data) {
            return rewrite.kind == runtime::GraphOptimizationRewriteKind::IdentityMaterialization &&
                static_cast<const std::unordered_set<const ggml_tensor *> *>(data)->count(rewrite.node) != 0;
        };
        runtime::optimize_graph(*graph, optimization);
        allocator.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend())));
        if (!allocator || !ggml_gallocr_alloc_graph(allocator.get(), graph)) {
            throw std::runtime_error("AuK audio conditioning graph allocation failed");
        }
    }
    ~AudioGraph() { core::release_backend_graph_resources(execution.backend(), graph, true); }
};
}  // namespace

audio::WhisperLogMelFeatures extract_audio_features(
    const std::vector<float> & samples, int sample_rate, int channels, size_t threads) {
    if (sample_rate <= 0 || samples.empty()) throw std::runtime_error("AuK audio input must be nonempty with a positive sample rate");
    auto mono = audio::mixdown_interleaved_to_mono_average(samples, channels);
    audio::SoxrResampleOptions options;
    options.profile = audio::SoxrResampleProfile::ExplicitFloat32Runtime;
    options.output_length_policy = audio::SoxrOutputLengthPolicy::ExactExpected;
    options.require_full_input = true;
    options.reject_empty_output = true;
    options.warning_context = "auk";
    options.fallback_description = "AuK requires Soxr HQ; resampling will fail";
    auto resampled = audio::try_resample_mono_soxr(mono, sample_rate, 16000, options);
    if (!resampled) throw std::runtime_error("AuK audio conditioning requires Soxr HQ resampling");
    constexpr size_t maximum_samples = 4800000;
    if (resampled->size() > maximum_samples) resampled->resize(maximum_samples);
    const int64_t valid_frames = (resampled->size() + 159) / 160;
    // Python pads to 300 seconds. Past the finite 400-sample STFT support,
    // padding is silent and cannot increase the global log-mel maximum.
    resampled->resize(std::min(maximum_samples, resampled->size() + 400), 0.0F);
    // The existing Kokoro policy selects periodic Hann, matching torch.hann_window.
    static const audio::WhisperLogMelExtractor extractor({16000, 400, 160, 128, audio::STFTFamily::Kokoro});
    auto padded = extractor.compute(*resampled, threads);
    audio::WhisperLogMelFeatures output;
    output.frames = valid_frames;
    output.mel_bins = 128;
    output.values.resize(128 * valid_frames);
    for (int channel = 0; channel < 128; ++channel) {
        std::copy_n(padded.values.begin() + channel * padded.frames, valid_frames,
            output.values.begin() + channel * valid_frames);
    }
    return output;
}

struct AudioConditioningRuntime::State {
    core::ExecutionContext & execution;
    core::BackendWeightStore store;
    AudioWeights weights;
    std::unique_ptr<AudioGraph> prepared;
    State(core::ExecutionContext & execution_, const assets::TensorSource & source)
        : execution(execution_), store(execution.backend(), execution.backend_type(), "auk.audio_conditioning", 2 * 1024 * 1024),
          weights(load_audio_weights(store, source)) { store.upload(); }
};

AudioConditioningRuntime::AudioConditioningRuntime(core::ExecutionContext & execution,
    const assets::TensorSource & qwen, int64_t frames) : state_(std::make_unique<State>(execution, qwen)) { prepare(frames); }
AudioConditioningRuntime::~AudioConditioningRuntime() = default;

void AudioConditioningRuntime::prepare(int64_t frames) {
    if (frames < 3) throw std::runtime_error("AuK audio conditioning requires at least three mel frames");
    auto & state = *state_;
    if (state.prepared && state.prepared->frames == frames) return;
    state.prepared.reset();
    state.prepared = std::make_unique<AudioGraph>(state.execution, state.weights, frames);
}

std::vector<float> AudioConditioningRuntime::encode(const std::vector<float> & features) {
    if (!state_->prepared) throw std::runtime_error("AuK audio conditioning graph is not prepared");
    auto & graph = *state_->prepared;
    if (features.size() != static_cast<size_t>(graph.frames * 128)) {
        throw std::runtime_error("AuK audio feature length differs from prepared graph");
    }
    core::write_tensor_f32(graph.input, features);
    if (core::compute_backend_graph(graph.execution.backend(), graph.graph, nullptr, "auk.audio_conditioning") != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("AuK audio conditioning computation failed");
    }
    return core::read_tensor_f32(graph.output.tensor);
}
}  // namespace engine::models::auk
