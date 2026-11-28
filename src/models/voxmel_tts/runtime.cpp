#include "engine/models/voxmel_tts/runtime.h"
#include "engine/models/voxmel_tts/ar.h"

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/prepared_graph.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/conditioning_modules.h"
#include "engine/framework/modules/feed_forward_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/transformers/transformer_blocks.h"
#include "engine/framework/modules/vocoders/vocos_vocoder.h"
#include "engine/framework/modules/weight_binding.h"
#include "engine/framework/runtime/cache_slots.h"
#include "engine/framework/sampling/device_denoiser_runtime.h"
#include "engine/framework/sampling/torch_random.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <tuple>

namespace engine::models::voxmel_tts {

using core::TensorValue;
using core::TensorShape;
namespace binding = modules::binding;

std::vector<float> reference_mel(const ModelAssets & a, const std::vector<float> & mono, size_t threads) {
    const audio::STFTConfig config{1024, a.hop, 1024, true, audio::STFTPadMode::Reflect, audio::STFTFamily::Kokoro};
    auto magnitude = audio::STFT().compute_magnitude(mono, a.mel_window, 1,
        static_cast<int64_t>(mono.size()), config, threads);
    auto mel = audio::MelFilterbank().compute_custom(magnitude.values, 1, magnitude.shape.at(1),
        magnitude.shape.at(2), a.mel_filterbank);
    const int64_t frames = (mel.shape.at(2) / 2) * 2;
    std::vector<float> result(static_cast<size_t>(frames * a.mels));
    for (int64_t t = 0; t < frames; ++t) for (int64_t m = 0; m < a.mels; ++m)
        result[static_cast<size_t>(t * a.mels + m)] = std::log(std::max(
            mel.values[static_cast<size_t>(m * mel.shape[2] + t)], 1e-5f));
    return result;
}

class VoxMelFlowRuntime::Impl {
public:
    core::ExecutionContext & execution;
    std::string name;
    int64_t hidden, condition_size, channels, depth;
    core::BackendWeightStore store;
    modules::LinearWeights in, cond, out;
    modules::NormWeights final_norm;
    modules::TimestepEmbeddingWeights time;
    struct Block { modules::NormWeights norm; modules::FeedForwardWeights ffn; };
    std::vector<Block> blocks;
    struct Graphs {
        std::unique_ptr<core::PreparedGraph> conditioning;
        TensorValue input, projected, time_input, time_bank, time_index, dt, guidance;
        core::PreparedGraph::GraphId time_graph = 0;
        std::unique_ptr<modules::DeviceDenoiserRuntime> sampler;
        std::vector<int32_t> indices;
        modules::FlowSamplerGraphSequence sequence;
    };
    using Key = std::tuple<int64_t, int64_t, int64_t>;
    runtime::CacheSlots<Key, std::unique_ptr<Graphs>> graphs{2};

    Impl(const ModelAssets & a, core::ExecutionContext & e, assets::TensorStorageType storage,
        std::string prefix, int64_t cond_size)
        : execution(e), name(std::move(prefix)), hidden(a.flow_hidden), condition_size(cond_size),
          channels(a.mels / 2), depth(a.flow_layers),
          store(e.backend(), e.backend_type(), "voxmel_tts." + name + ".weights", 1024 * 1024) {
        const auto & source = *a.model;
        in = binding::linear_from_source(store, source, name + ".in_proj", storage, hidden, channels, true);
        cond = binding::linear_from_source(store, source, name + ".cond_proj", storage, hidden, condition_size, true);
        time.fc1 = binding::linear_from_source(store, source, name + ".time_mlp.0", storage, hidden, 128, true);
        time.fc2 = binding::linear_from_source(store, source, name + ".time_mlp.2", storage, hidden, hidden, true);
        std::vector<float> frequencies(64);
        for (size_t i = 0; i < frequencies.size(); ++i)
            frequencies[i] = std::exp(static_cast<float>(-std::log(10000.0)) * static_cast<float>(i) / 64.0f);
        time.freqs = store.make_f32(TensorShape::from_dims({64}), frequencies);
        for (int64_t i = 0; i < depth; ++i) {
            const auto n = name + ".blocks." + std::to_string(i);
            Block b;
            b.norm = binding::norm_from_source(store, source, n + ".0", hidden);
            const auto fc1 = binding::linear_from_source(store, source, n + ".1", storage, hidden, hidden, true);
            const auto fc2 = binding::linear_from_source(store, source, n + ".3", storage, hidden, hidden, true);
            b.ffn = {fc1.weight, fc1.bias, fc2.weight, fc2.bias};
            blocks.push_back(std::move(b));
        }
        final_norm = binding::norm_from_source(store, source, name + ".out.0", hidden);
        out = binding::linear_from_source(store, source, name + ".out.1", storage, channels, hidden, true);
        store.upload();
    }

    std::unique_ptr<Graphs> prepare(int64_t rows, int64_t branches, int64_t steps) {
        auto s = std::make_unique<Graphs>();
        s->conditioning = std::make_unique<core::PreparedGraph>(execution,
            core::PreparedGraphConfig{"voxmel_tts." + name + ".conditioning", 2 * 1024 * 1024, 512, 256 * 1024});
        auto & persistent = s->conditioning->persistent_build_context();
        s->input = core::make_tensor(persistent, GGML_TYPE_F32, TensorShape::from_dims({branches * rows, condition_size}));
        s->projected = core::make_tensor(persistent, GGML_TYPE_F32, TensorShape::from_dims({branches * rows, hidden}));
        s->time_input = core::make_tensor(persistent, GGML_TYPE_F32, TensorShape::from_dims({steps, 1}));
        s->time_bank = core::make_tensor(persistent, GGML_TYPE_F32, TensorShape::from_dims({steps, hidden}));
        s->time_index = core::make_tensor(persistent, GGML_TYPE_I32, TensorShape::from_dims({1}));
        s->dt = core::make_tensor(persistent, GGML_TYPE_F32, TensorShape::from_dims({1}));
        s->guidance = core::make_tensor(persistent, GGML_TYPE_F32, TensorShape::from_dims({2}));
        s->conditioning->allocate_persistent();
        auto & ctx = s->conditioning->build_context();
        auto projected = modules::LinearModule({condition_size, hidden, true}).build(ctx, s->input, cond);
        ggml_build_forward_expand(s->conditioning->graph(), ggml_cpy(ctx.ggml, projected.tensor, s->projected.tensor));
        s->time_graph = s->conditioning->add_graph("timesteps");
        auto timed = modules::TimestepEmbeddingModule({128, hidden, 1e-5f, modules::TimestepEmbeddingNorm::None})
            .build(ctx, s->time_input, time);
        ggml_build_forward_expand(s->conditioning->graph(s->time_graph), ggml_cpy(ctx.ggml, timed.tensor, s->time_bank.tensor));
        s->conditioning->allocate();
        std::vector<float> timesteps(static_cast<size_t>(steps));
        s->indices.resize(static_cast<size_t>(steps));
        for (int64_t i = 0; i < steps; ++i) {
            timesteps[static_cast<size_t>(i)] = static_cast<float>(static_cast<double>(i) / static_cast<double>(steps));
            s->indices[static_cast<size_t>(i)] = static_cast<int32_t>(i);
        }
        core::write_tensor_f32(s->time_input, timesteps);
        core::write_tensor_f32(s->dt, {1.0f / static_cast<float>(steps)});
        s->conditioning->compute(s->time_graph);
        Graphs & state = *s;
        s->sampler = std::make_unique<modules::DeviceDenoiserRuntime>(execution,
            core::PreparedGraphConfig{"voxmel_tts." + name + ".euler", 4 * 1024 * 1024, 4096},
            [&, rows, branches](core::ModuleBuildContext & c) {
                auto latent = core::make_tensor(c, GGML_TYPE_F32, TensorShape::from_dims({rows, channels}));
                ggml_set_input(latent.tensor);
                ggml_set_output(latent.tensor);
                auto batch = modules::RepeatModule({TensorShape::from_dims({branches, rows, channels})}).build(c,
                    core::reshape_tensor(c, latent, TensorShape::from_dims({1, rows, channels})));
                batch = core::reshape_tensor(c, batch, TensorShape::from_dims({branches * rows, channels}));
                auto h = modules::LinearModule({channels, hidden, true}).build(c, batch, in);
                h = modules::AddModule().build(c, h, state.projected);
                auto t = core::wrap_tensor(ggml_get_rows(c.ggml, state.time_bank.tensor, state.time_index.tensor),
                    TensorShape::from_dims({1, hidden}));
                h = modules::AddModule(false, true).build(c, h, t);
                modules::FeedForwardConfig f;
                f.hidden_size = f.intermediate_size = hidden;
                f.activation = modules::FeedForwardActivation::Silu;
                for (const auto & block : blocks) {
                    const auto normalized = modules::LayerNormModule({hidden, 1e-5f, true, true}).build(c, h, block.norm);
                    h = modules::AddModule().build(c, h, modules::FeedForwardModule(f).build(c, normalized, block.ffn));
                }
                h = modules::LayerNormModule({hidden, 1e-5f, true, true}).build(c, h, final_norm);
                auto v = modules::LinearModule({hidden, channels, true}).build(c, h, out);
                if (branches == 3) {
                    const auto full = modules::SliceModule({0, 0, rows}).build(c, v);
                    const auto drop_text = modules::SliceModule({0, rows, rows}).build(c, v);
                    const auto drop_both = modules::SliceModule({0, 2 * rows, rows}).build(c, v);
                    const auto text_scale = modules::SliceModule({0, 0, 1}).build(c, state.guidance);
                    const auto speaker_scale = modules::SliceModule({0, 1, 1}).build(c, state.guidance);
                    auto text_delta = core::wrap_tensor(ggml_sub(c.ggml, full.tensor, drop_text.tensor), full.shape);
                    auto speaker_delta = core::wrap_tensor(ggml_sub(c.ggml, drop_text.tensor, drop_both.tensor), full.shape);
                    text_delta = modules::MulModule(true).build(c, text_delta, text_scale);
                    speaker_delta = modules::MulModule(true).build(c, speaker_delta, speaker_scale);
                    v = modules::AddModule().build(c, modules::AddModule().build(c, drop_both, text_delta), speaker_delta);
                }
                const auto updated = modules::FlowSamplerEulerGraphModule().build(c, latent, v, state.dt);
                return modules::DeviceDenoiserGraph{latent, v, updated};
            });
        s->sequence.steps = static_cast<size_t>(steps);
        s->sequence.step_bindings.push_back({s->time_index, 0, sizeof(int32_t), s->indices.data(), s->indices.size() * sizeof(int32_t)});
        return s;
    }

    std::vector<float> sample(const std::vector<float> & conditioning, int64_t rows, int64_t branches,
        const std::vector<float> & noise, int64_t steps, float text, float speaker) {
        if (rows <= 0 || (branches != 1 && branches != 3) || steps <= 0 || steps > 256 ||
            conditioning.size() != static_cast<size_t>(rows * branches * condition_size) ||
            noise.size() != static_cast<size_t>(rows * channels))
            throw std::runtime_error("Invalid VoxMel flow shape or sampling steps");
        const Key key{rows, branches, steps};
        auto * entry = graphs.find(key);
        if (!entry) {
            graphs.put(key, prepare(rows, branches, steps));
            entry = graphs.find(key);
        }
        auto & s = **entry;
        core::write_tensor_f32(s.input, conditioning);
        core::write_tensor_f32(s.guidance, {text, speaker});
        s.conditioning->compute();
        return s.sampler->sample(noise, s.sequence);
    }
};

VoxMelFlowRuntime::VoxMelFlowRuntime(const ModelAssets & a, core::ExecutionContext & e,
    assets::TensorStorageType storage, const std::string & prefix, int64_t cond)
    : impl_(std::make_unique<Impl>(a, e, storage, prefix, cond)) {}
VoxMelFlowRuntime::~VoxMelFlowRuntime() = default;
std::vector<float> VoxMelFlowRuntime::sample(const std::vector<float> & c, int64_t rows, int64_t branches,
    const std::vector<float> & noise, int64_t steps, float text, float speaker) {
    return impl_->sample(c, rows, branches, noise, steps, text, speaker);
}

class VoxMelTierRuntime::Impl {
public:
    const ModelAssets & a;
    core::ExecutionContext & execution;
    size_t tier;
    bool flash;
    core::BackendWeightStore store;
    modules::LinearWeights input;
    modules::NormWeights norm;
    std::vector<modules::TransformerEncoderBlockWeights> blocks;
    std::vector<float> mean, scale;
    struct Graph {
        std::unique_ptr<core::PreparedGraph> owner;
        TensorValue known, positions, output;
    };
    runtime::CacheSlots<int64_t, Graph> graphs{1};

    Impl(const ModelAssets & assets, core::ExecutionContext & e, assets::TensorStorageType storage,
        core::AttentionPreference preference, size_t index)
        : a(assets), execution(e), tier(index),
          flash(a.tier_windows.at(tier) != 0 &&
              core::resolve_flash_attention(e.backend(), a.tier_hidden / a.tier_heads, preference)),
          store(e.backend(), e.backend_type(), "voxmel_tts.tier" + std::to_string(tier) + ".weights", 1024 * 1024) {
        const auto p = "up_tiers." + std::to_string(tier);
        const auto & source = *a.model;
        input = binding::linear_from_source(store, source, p + ".in_proj", storage, a.tier_hidden, a.mels / 2, true);
        norm = binding::norm_from_source(store, source, p + ".norm", a.tier_hidden);
        for (int64_t i = 0; i < a.tier_layers.at(tier); ++i) {
            const auto n = p + ".blocks." + std::to_string(i);
            modules::TransformerEncoderBlockWeights b;
            b.norm1 = binding::norm_from_source(store, source, n + ".norm1", a.tier_hidden);
            b.norm2 = binding::norm_from_source(store, source, n + ".norm2", a.tier_hidden);
            b.self_attention.qkv_weight = store.load_tensor(source, n + ".attn.in_proj_weight", storage, {3 * a.tier_hidden, a.tier_hidden});
            b.self_attention.qkv_bias = store.load_f32_tensor(source, n + ".attn.in_proj_bias", {3 * a.tier_hidden});
            b.self_attention.out_weight = store.load_tensor(source, n + ".attn.out_proj.weight", storage, {a.tier_hidden, a.tier_hidden});
            b.self_attention.out_bias = store.load_f32_tensor(source, n + ".attn.out_proj.bias", {a.tier_hidden});
            const auto fc1 = binding::linear_from_source(store, source, n + ".mlp.0", storage, 4 * a.tier_hidden, a.tier_hidden, true);
            const auto fc2 = binding::linear_from_source(store, source, n + ".mlp.2", storage, a.tier_hidden, 4 * a.tier_hidden, true);
            b.feed_forward = {fc1.weight, fc1.bias, fc2.weight, fc2.bias};
            blocks.push_back(std::move(b));
        }
        mean = source.require_f32(p + ".detail_mean", {a.mels / 2});
        scale = source.require_f32(p + ".detail_var", {a.mels / 2});
        for (auto & v : scale) v = std::sqrt(std::max(v, 1e-8f));
        store.upload();
    }

    TensorValue build_pointwise_block(core::ModuleBuildContext & ctx, const TensorValue & input,
        const modules::TransformerEncoderBlockWeights & block) const {
        // A zero-width window admits only the current position: softmax has
        // exactly one element, so attention is V. Keep the original V weights
        // and biases; Q/K and the attention graph are unnecessary on every backend.
        const auto value_weight = modules::SliceModule({0, 2 * a.tier_hidden, a.tier_hidden}).build(
            ctx, block.self_attention.qkv_weight.value());
        const auto value_bias = modules::SliceModule({0, 2 * a.tier_hidden, a.tier_hidden}).build(
            ctx, block.self_attention.qkv_bias.value());
        const modules::LayerNormModule layer_norm({a.tier_hidden, 1e-5f, true, true});
        const modules::LinearModule projection({a.tier_hidden, a.tier_hidden, true});
        const modules::ResidualAddModule residual;
        auto value = projection.build(ctx, layer_norm.build(ctx, input, block.norm1), {value_weight, value_bias});
        value = projection.build(ctx, value, {block.self_attention.out_weight, block.self_attention.out_bias});
        auto h = residual.build(ctx, input, value);
        auto ff = modules::FeedForwardModule({a.tier_hidden, 4 * a.tier_hidden, true}).build(
            ctx, layer_norm.build(ctx, h, block.norm2), block.feed_forward);
        return residual.build(ctx, h, ff);
    }

    std::vector<float> condition(const std::vector<float> & known) {
        const int64_t channels = a.mels / 2;
        const int64_t frames = static_cast<int64_t>(known.size()) / channels;
        if (frames <= 0 || known.size() != static_cast<size_t>(frames * channels))
            throw std::runtime_error("Invalid VoxMel residual input");
        auto * g = graphs.find(frames);
        if (!g) {
            Graph next;
            next.owner = std::make_unique<core::PreparedGraph>(execution,
                core::PreparedGraphConfig{"voxmel_tts.tier" + std::to_string(tier) + ".context", 4 * 1024 * 1024, 8192, 256 * 1024});
            auto & ctx = next.owner->build_context();
            auto & persistent = next.owner->persistent_build_context();
            next.known = core::make_tensor(persistent, GGML_TYPE_F32, TensorShape::from_dims({1, frames, channels}));
            next.positions = core::make_tensor(persistent, GGML_TYPE_F32, TensorShape::from_dims({1, frames, a.tier_hidden}));
            next.owner->allocate_persistent();
            auto h = modules::LinearModule({channels, a.tier_hidden, true}).build(ctx, next.known, input);
            h = modules::AddModule().build(ctx, h, next.positions);
            const bool pointwise = a.tier_windows.at(tier) == 0;
            modules::TransformerEncoderBlockConfig config;
            config.hidden_size = a.tier_hidden; config.intermediate_size = 4 * a.tier_hidden;
            config.num_heads = a.tier_heads; config.eps = 1e-5f;
            config.use_packed_qkv = true; config.use_flash_attention = flash;
            config.attention_precision = GGML_PREC_F32;
            for (const auto & block : blocks) {
                h = pointwise ? build_pointwise_block(ctx, h, block)
                    : modules::TransformerEncoderBlockModule(config).build(ctx, h, block);
            }
            next.output = modules::LayerNormModule({a.tier_hidden, 1e-5f, true, true}).build(ctx, h, norm);
            ggml_set_output(next.output.tensor);
            ggml_build_forward_expand(next.owner->graph(), next.output.tensor);
            next.owner->allocate();
            std::vector<float> positions(static_cast<size_t>(frames * a.tier_hidden));
            const int64_t half = a.tier_hidden / 2;
            for (int64_t t = 0; t < frames; ++t) for (int64_t i = 0; i < half; ++i) {
                const float freq = std::exp(static_cast<float>(-std::log(10000.0)) * static_cast<float>(i) / static_cast<float>(half));
                const float v = static_cast<float>(t) * freq;
                positions[static_cast<size_t>(t * a.tier_hidden + i)] = std::sin(v);
                positions[static_cast<size_t>(t * a.tier_hidden + half + i)] = std::cos(v);
            }
            core::write_tensor_f32(next.positions, positions);
            graphs.put(frames, std::move(next)); g = graphs.find(frames);
        }
        core::write_tensor_f32(g->known, known);
        g->owner->compute();
        return core::read_tensor_f32(g->output.tensor);
    }
};

VoxMelTierRuntime::VoxMelTierRuntime(const ModelAssets & a, core::ExecutionContext & e,
    assets::TensorStorageType storage, core::AttentionPreference p, size_t i)
    : impl_(std::make_unique<Impl>(a, e, storage, p, i)) {}
VoxMelTierRuntime::~VoxMelTierRuntime() = default;
std::vector<float> VoxMelTierRuntime::condition(const std::vector<float> & x) { return impl_->condition(x); }
std::vector<float> VoxMelTierRuntime::denormalize(const std::vector<float> & x) const {
    auto out = x;
    if (impl_->a.detail_norm) for (size_t i = 0; i < out.size(); ++i)
        out[i] = out[i] * impl_->scale[i % impl_->scale.size()] + impl_->mean[i % impl_->mean.size()];
    return out;
}

class VoxMelRuntime::Impl {
public:
    std::shared_ptr<const ModelAssets> a;
    core::ExecutionContext & execution;
    VoxMelQwenRuntime qwen;
    VoxMelFlowRuntime coarse_flow;
    std::vector<std::unique_ptr<VoxMelTierRuntime>> tiers;
    std::vector<std::unique_ptr<VoxMelFlowRuntime>> flows;
    modules::VocosVocoder vocos;
    sampling::TorchCudaSamplingPolicy rng;

    Impl(std::shared_ptr<const ModelAssets> assets, core::ExecutionContext & e,
        assets::TensorStorageType storage, core::AttentionPreference preference)
        : a(std::move(assets)), execution(e), qwen(*a, e, storage, preference),
          coarse_flow(*a, e, storage, "coarse_flow", a->hidden),
          vocos(a->vocos, e.backend(), e.config().threads, {a->mels, 1024, a->hop, true}),
          rng(sampling::resolve_torch_cuda_sampling_policy(e.backend_type(), e.config().device,
              "voxmel_tts", "VoxMel", sampling::TorchCudaSamplingPolicyFailureMode::FallbackToDefault)) {
        for (size_t i = 0; i < a->tier_layers.size(); ++i) {
            tiers.push_back(std::make_unique<VoxMelTierRuntime>(*a, e, storage, preference, i));
            flows.push_back(std::make_unique<VoxMelFlowRuntime>(*a, e, storage,
                "up_tiers." + std::to_string(i) + ".flow", a->tier_hidden));
        }
    }

    std::vector<float> synthesize(const std::vector<int32_t> & target,
        const std::vector<int32_t> & reference_text, const std::vector<float> & reference_mel,
        const SynthesisOptions & opts) {
        const auto started = std::chrono::steady_clock::now();
        const int64_t channels = a->mels / 2;
        const int64_t width = a->hidden;
        const bool cloning = !reference_mel.empty();
        if (cloning != !reference_text.empty()) throw std::runtime_error("VoxMel reference Mel and transcript must be supplied together");
        std::vector<int32_t> ids = reference_text;
        ids.insert(ids.end(), target.begin(), target.end());
        const auto words = qwen.text_embeddings(ids);
        std::vector<float> prompt_frequency, prompt_coarse;
        if (cloning) {
            const int64_t frames = static_cast<int64_t>(reference_mel.size()) / a->mels;
            if (frames < 2 || reference_mel.size() != static_cast<size_t>(frames * a->mels))
                throw std::runtime_error("VoxMel reference requires at least one complete coarse frame");
            const int64_t even = (frames / 2) * 2;
            prompt_frequency.resize(static_cast<size_t>(even * channels));
            for (int64_t t = 0; t < even; ++t) for (int64_t i = 0; i < channels; ++i) {
                const size_t j = static_cast<size_t>(t * a->mels + 2 * i);
                const float x = (reference_mel[j] - a->mel_mean[2 * i]) / a->mel_std[2 * i];
                const float y = (reference_mel[j + 1] - a->mel_mean[2 * i + 1]) / a->mel_std[2 * i + 1];
                prompt_frequency[static_cast<size_t>(t * channels + i)] = (x + y) / 2.0f;
            }
            prompt_coarse.resize(static_cast<size_t>((even / 2) * channels));
            for (int64_t t = 0; t < even / 2; ++t) for (int64_t i = 0; i < channels; ++i)
                prompt_coarse[static_cast<size_t>(t * channels + i)] =
                    (prompt_frequency[static_cast<size_t>(2 * t * channels + i)] +
                     prompt_frequency[static_cast<size_t>((2 * t + 1) * channels + i)]) / 2.0f;
        }
        const bool use_text = std::abs(opts.text_guidance) > 1e-6f;
        const bool use_speaker = cloning && std::abs(opts.speaker_guidance) > 1e-6f;
        const bool guided = use_text || use_speaker;
        const int64_t branches = guided ? (use_speaker ? 3 : 2) : 1;
        auto prime = std::vector<float>(a->special.begin(), a->special.begin() + width);
        prime.insert(prime.end(), words.begin(), words.end());
        prime.insert(prime.end(), a->special.begin() + width, a->special.begin() + 2 * width);
        std::vector<std::vector<float>> prefixes(branches, prime);
        for (int64_t b = 1; b < branches; ++b) for (size_t i = 0; i < ids.size(); ++i)
            std::copy(a->null_text.begin(), a->null_text.end(), prefixes[b].begin() + (i + 1) * static_cast<size_t>(width));
        if (cloning) {
            const auto prompt = qwen.coarse_embeddings(prompt_coarse);
            for (int64_t b = 0; b < branches; ++b) {
                if (b == 2) prefixes[b].insert(prefixes[b].end(), prompt.size(), 0.0f);
                else prefixes[b].insert(prefixes[b].end(), prompt.begin(), prompt.end());
            }
        }
        auto state = qwen.begin(prefixes, opts.max_frames);
        uint64_t offset = 0;
        const auto noise = [&](size_t count) {
            auto values = sampling::generate_torch_cuda_tensor_iterator_randn(count, opts.seed, offset, rng);
            offset += sampling::torch_cuda_tensor_iterator_offset_blocks(count, rng);
            return values;
        };
        std::vector<float> coarse;
        for (int64_t frame = 0; frame < opts.max_frames; ++frame) {
            if (state.stop_logit > 0 && frame > opts.min_frames && !coarse.empty()) break;
            auto conditioning = state.hidden;
            if (guided && branches == 2)
                conditioning.insert(conditioning.end(), state.hidden.begin() + width, state.hidden.begin() + 2 * width);
            auto values = coarse_flow.sample(conditioning, 1, guided ? 3 : 1, noise(static_cast<size_t>(channels)),
                opts.steps, opts.text_guidance, use_speaker ? opts.speaker_guidance : 0.0f);
            coarse.insert(coarse.end(), values.begin(), values.end());
            if (frame + 1 < opts.max_frames) state = qwen.advance(values);
        }
        debug::timing_log_scalar("voxmel_tts.ar_ms", debug::elapsed_ms(started));
        debug::trace_log_scalar("voxmel_tts.coarse_frames", coarse.size() / static_cast<size_t>(channels));
        const auto residual_started = std::chrono::steady_clock::now();
        auto x = std::move(coarse);
        for (size_t tier = 0; tier < tiers.size(); ++tier) {
            const auto & prompt = tier == 0 ? prompt_coarse : prompt_frequency;
            auto known = prompt;
            known.insert(known.end(), x.begin(), x.end());
            auto condition = tiers[tier]->condition(known);
            const int64_t rows = static_cast<int64_t>(known.size()) / channels;
            auto detail = tiers[tier]->denormalize(flows[tier]->sample(condition, rows, 1, noise(known.size()), opts.up_steps));
            std::vector<float> reconstructed(x.size() * 2);
            const size_t prompt_values = prompt.size();
            for (size_t i = 0; i < x.size(); ++i) {
                const float sum = x[i] + detail[prompt_values + i];
                const float difference = x[i] - detail[prompt_values + i];
                if (tier == 0) {
                    const size_t t = i / static_cast<size_t>(channels), c = i % static_cast<size_t>(channels);
                    reconstructed[(2 * t) * static_cast<size_t>(channels) + c] = sum;
                    reconstructed[(2 * t + 1) * static_cast<size_t>(channels) + c] = difference;
                } else {
                    reconstructed[2 * i] = sum; reconstructed[2 * i + 1] = difference;
                }
            }
            x = std::move(reconstructed);
        }
        for (size_t i = 0; i < x.size(); ++i) x[i] = x[i] * a->mel_std[i % a->mels] + a->mel_mean[i % a->mels];
        debug::timing_log_scalar("voxmel_tts.residual_ms", debug::elapsed_ms(residual_started));
        const auto vocoder_started = std::chrono::steady_clock::now();
        auto audio = vocos.decode(x);
        debug::timing_log_scalar("voxmel_tts.vocoder_ms", debug::elapsed_ms(vocoder_started));
        return audio;
    }
};

VoxMelRuntime::VoxMelRuntime(std::shared_ptr<const ModelAssets> a, core::ExecutionContext & e,
    assets::TensorStorageType s, core::AttentionPreference p) : impl_(std::make_unique<Impl>(std::move(a), e, s, p)) {}
VoxMelRuntime::~VoxMelRuntime() = default;
std::vector<float> VoxMelRuntime::synthesize(const std::vector<int32_t> & text,
    const std::vector<int32_t> & reference, const std::vector<float> & mel, const SynthesisOptions & options) {
    return impl_->synthesize(text, reference, mel, options);
}

} // namespace engine::models::voxmel_tts
