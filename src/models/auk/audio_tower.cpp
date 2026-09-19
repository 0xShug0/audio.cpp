#include "engine/models/auk/audio_tower.h"

#include "engine/framework/audio/dsp.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/module.h"
#include "engine/framework/modules/attention/scaled_dot_product_attention.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <ggml.h>

#include <cmath>
#include <mutex>
#include <stdexcept>

namespace engine::models::auk {
namespace {

namespace mod = engine::modules;

constexpr size_t kGraphContextBytes = 512ull * 1024ull * 1024ull;
constexpr int64_t kGraphNodeCapacity = 262144;

struct LayerWeights {
    mod::NormWeights attention_norm;
    mod::LinearWeights q, k, v, out;
    mod::NormWeights final_norm;
    mod::LinearWeights fc1, fc2;
};

struct TowerWeights {
    mod::Conv1dWeights conv1;
    mod::Conv1dWeights conv2;
    std::vector<LayerWeights> layers;
    mod::NormWeights ln_post;
    mod::LinearWeights proj;
    int64_t loaded = 0;
};

// Whisper's SinusoidsPositionEmbedding, computed rather than stored -- the checkpoint
// carries no positional tensor, which is why a port that looks for one finds nothing
// and has to be told this is deliberate.
std::vector<float> sinusoid_positions(int64_t length, int64_t channels) {
    std::vector<float> values(static_cast<size_t>(length * channels), 0.0F);
    const int64_t half = channels / 2;
    const double increment = std::log(10000.0) / static_cast<double>(half - 1);
    for (int64_t position = 0; position < length; ++position) {
        for (int64_t index = 0; index < half; ++index) {
            const double scaled =
                static_cast<double>(position) * std::exp(static_cast<double>(index) * -increment);
            values[static_cast<size_t>(position * channels + index)] = static_cast<float>(std::sin(scaled));
            values[static_cast<size_t>(position * channels + half + index)] = static_cast<float>(std::cos(scaled));
        }
    }
    return values;
}

// ⚠ ERF, not the tanh approximation. The config says activation_function: "gelu",
// which in transformers is ACT2FN["gelu"] = exact erf GELU. ggml_gelu is the tanh
// approximation, and the difference is small per element, applied twice per layer for
// 32 layers -- a diffuse ~0.3% error across every token with no local cause to find.
core::TensorValue gelu(core::ModuleBuildContext & ctx, const core::TensorValue & x) {
    return core::wrap_tensor(
        ggml_gelu_erf(ctx.ggml, core::ensure_backend_addressable_layout(ctx, x).tensor), x.shape, GGML_TYPE_F32);
}

// One pre-norm encoder layer over a single attention window.
core::TensorValue build_layer(
    core::ModuleBuildContext & ctx,
    const AukAudioTowerConfig & config,
    const LayerWeights & weights,
    const core::TensorValue & input,
    int64_t frames) {
    const int64_t head_dim = config.d_model / config.heads;
    const mod::LayerNormModule norm({config.d_model, config.layer_norm_eps, true, true});

    auto normed = norm.build(ctx, input, weights.attention_norm);
    const mod::LinearModule projection({config.d_model, config.d_model, true});
    auto to_heads = [&](const core::TensorValue & t) {
        auto shaped = core::reshape_tensor(
            ctx, core::ensure_backend_addressable_layout(ctx, t),
            core::TensorShape::from_dims({1, frames, config.heads, head_dim}));
        auto transposed = mod::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, shaped);
        return core::wrap_tensor(ggml_cont(ctx.ggml, transposed.tensor), transposed.shape, GGML_TYPE_F32);
    };
    // ⚠ k_proj alone is bias-free, so it needs its own module: a shared one with
    // use_bias=true throws "bias is required" rather than silently ignoring it, which
    // is the behaviour you want but still has to be honoured here.
    const mod::LinearModule key_projection({config.d_model, config.d_model, false});
    auto query = to_heads(projection.build(ctx, normed, weights.q));
    auto key = to_heads(key_projection.build(ctx, normed, weights.k));
    auto value = to_heads(projection.build(ctx, normed, weights.v));
    auto attended = mod::ScaledDotProductAttentionModule({
        head_dim,
        mod::ScaledDotProductAttentionLowering::Flash,
        GGML_PREC_F32,
        mod::AttentionCausality::NonCausal,
    }).build(ctx, query, key, value);
    auto flat = core::reshape_tensor(
        ctx, core::ensure_backend_addressable_layout(ctx, attended),
        core::TensorShape::from_dims({frames, config.d_model}));
    auto projected = projection.build(ctx, flat, weights.out);
    auto residual = mod::AddModule().build(ctx, input, projected);

    auto second = norm.build(ctx, residual, weights.final_norm);
    second = mod::LinearModule({config.d_model, config.ffn_dim, true}).build(ctx, second, weights.fc1);
    second = gelu(ctx, second);
    second = mod::LinearModule({config.ffn_dim, config.d_model, true}).build(ctx, second, weights.fc2);
    return mod::AddModule().build(ctx, residual, second);
}

}  // namespace

void AukAudioTowerConfig::validate() const {
    if (d_model <= 0 || heads <= 0 || layers <= 0 || window <= 0) {
        throw std::runtime_error("AuK audio tower config has non-positive dimensions");
    }
    if (d_model % heads != 0) {
        throw std::runtime_error("AuK audio tower d_model must divide by heads");
    }
}

std::vector<float> AukAudioTower::log_mel(const std::vector<float> & samples, int64_t & frames) {
    engine::audio::WhisperLogMelConfig config;
    config.sample_rate = 16000;
    config.n_fft = 400;
    config.hop_length = 160;
    config.feature_size = 128;    // Omni uses 128 bins, not Whisper's usual 80
    const engine::audio::WhisperLogMelExtractor extractor(config);
    const auto features = extractor.compute(samples);
    frames = features.frames;
    return features.values;
}

struct AukAudioTower::Impl {
    AukAudioTowerConfig config;
    core::ExecutionContext * execution = nullptr;
    std::unique_ptr<core::BackendWeightStore> store;
    TowerWeights weights;

    std::mutex mutex;
    ggml_context * ggml = nullptr;
    ggml_gallocr_t gallocr = nullptr;
    ggml_cgraph * graph = nullptr;
    core::HostGraphPlan plan;
    ggml_tensor * mel_input = nullptr;
    std::vector<ggml_tensor *> position_inputs;
    ggml_tensor * output = nullptr;
    int64_t mel_frames = 0;

    void release() {
        if (graph != nullptr) {
            core::release_backend_graph_resources(execution->backend(), graph);
        }
        if (gallocr != nullptr) { ggml_gallocr_free(gallocr); gallocr = nullptr; }
        if (ggml != nullptr) { ggml_free(ggml); ggml = nullptr; }
        plan.reset();
        graph = nullptr;
        mel_input = output = nullptr;
        position_inputs.clear();
        mel_frames = 0;
    }

    void ensure_graph(int64_t frames) {
        if (ggml != nullptr && mel_frames == frames) return;
        release();
        position_inputs.clear();
        ggml_init_params params{kGraphContextBytes, nullptr, true};
        ggml = ggml_init(params);
        if (ggml == nullptr) throw std::runtime_error("failed to initialize AuK audio tower context");
        core::ModuleBuildContext ctx{ggml, "auk.audio_tower", execution->backend_type()};

        auto mel = core::make_tensor(
            ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, config.mel_bins, frames}));
        mel_input = mel.tensor;
        ggml_set_input(mel_input);

        // ⚠ EVERYTHING here is per chunk. HF chunks the MEL into windows of
        // n_window*2 frames and runs the convolutions, the positional embedding and the
        // attention inside each one (modeling_qwen2_5_omni.py:884-905). Doing any of it
        // across the whole utterance instead is silently different:
        //   - the convolutions would see neighbouring audio across a boundary
        //   - the positions would keep counting instead of restarting at 0 per chunk
        // Both produce sensible-looking embeddings that are wrong.
        const int64_t chunk_mel = config.window * 2;
        const int64_t chunk_count = (frames + chunk_mel - 1) / chunk_mel;

        std::vector<core::TensorValue> encoded_chunks;
        int64_t total_tokens = 0;
        for (int64_t chunk = 0; chunk < chunk_count; ++chunk) {
            const int64_t offset = chunk * chunk_mel;
            const int64_t valid = std::min<int64_t>(chunk_mel, frames - offset);
            auto slice = mod::SliceModule({2, offset, valid}).build(ctx, mel);
            auto chunk_mel_value = core::ensure_backend_addressable_layout(ctx, slice);

            auto x = mod::Conv1dModule({config.mel_bins, config.d_model, 3, 1, 1, 1, true})
                         .build(ctx, chunk_mel_value, weights.conv1);
            x = gelu(ctx, x);
            x = mod::Conv1dModule({config.d_model, config.d_model, 3, 2, 1, 1, true})
                    .build(ctx, x, weights.conv2);
            x = gelu(ctx, x);

            const int64_t chunk_frames = x.shape.dims[2];
            auto sequence = core::reshape_tensor(
                ctx,
                core::ensure_backend_addressable_layout(
                    ctx, mod::TransposeModule({{0, 2, 1}, 3}).build(ctx, x)),
                core::TensorShape::from_dims({chunk_frames, config.d_model}));

            auto positions = core::make_tensor(
                ctx, GGML_TYPE_F32, core::TensorShape::from_dims({chunk_frames, config.d_model}));
            ggml_set_input(positions.tensor);
            ggml_set_name(positions.tensor, ("auk.audio_tower.positions." + std::to_string(chunk)).c_str());
            position_inputs.push_back(positions.tensor);
            sequence = mod::AddModule().build(ctx, sequence, positions);

            for (int64_t index = 0; index < config.layers; ++index) {
                sequence = build_layer(
                    ctx, config, weights.layers[static_cast<size_t>(index)], sequence, chunk_frames);
            }
            encoded_chunks.push_back(sequence);
            total_tokens += chunk_frames;
        }

        auto encoded = encoded_chunks.front();
        for (size_t index = 1; index < encoded_chunks.size(); ++index) {
            encoded = mod::ConcatModule({0}).build(ctx, encoded, encoded_chunks[index]);
        }

        // ⚠ Pool FIRST, then ln_post, then proj: `proj(ln_post(pooled))`. Normalizing
        // before the average is a different function, and the difference survives into
        // every downstream token.
        const int64_t pooled_frames = total_tokens / 2;
        auto even = mod::SliceModule({0, 0, pooled_frames * 2}).build(ctx, encoded);
        auto pairs = core::reshape_tensor(
            ctx, core::ensure_backend_addressable_layout(ctx, even),
            core::TensorShape::from_dims({pooled_frames, 2, config.d_model}));
        auto first = core::ensure_backend_addressable_layout(ctx, mod::SliceModule({1, 0, 1}).build(ctx, pairs));
        auto second = core::ensure_backend_addressable_layout(ctx, mod::SliceModule({1, 1, 1}).build(ctx, pairs));
        auto mean = core::wrap_tensor(
            ggml_scale(ctx.ggml, ggml_add(ctx.ggml, first.tensor, second.tensor), 0.5F),
            core::TensorShape::from_dims({pooled_frames, 1, config.d_model}), GGML_TYPE_F32);
        auto pooled = core::reshape_tensor(
            ctx, core::ensure_backend_addressable_layout(ctx, mean),
            core::TensorShape::from_dims({pooled_frames, config.d_model}));

        pooled = mod::LayerNormModule({config.d_model, config.layer_norm_eps, true, true})
                     .build(ctx, pooled, weights.ln_post);
        auto projected = mod::LinearModule({config.d_model, config.output_dim, true})
                             .build(ctx, pooled, weights.proj);
        output = core::ensure_backend_addressable_layout(ctx, projected).tensor;
        ggml_set_output(output);

        graph = ggml_new_graph_custom(ggml, kGraphNodeCapacity, false);
        ggml_build_forward_expand(graph, output);
        core::validate_backend_graph_supported(execution->backend(), graph, "auk.audio_tower");
        gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution->backend()));
        if (gallocr == nullptr || !ggml_gallocr_reserve(gallocr, graph) || !ggml_gallocr_alloc_graph(gallocr, graph)) {
            release();
            throw std::runtime_error("failed to allocate AuK audio tower graph memory");
        }
        core::prepare_host_graph_plan(*execution, graph, plan);
        mel_frames = frames;
    }
};

AukAudioTower::AukAudioTower(
    AukAudioTowerConfig config,
    const assets::TensorSource & source,
    core::ExecutionContext & execution,
    std::string prefix)
    : impl_(std::make_unique<Impl>()) {
    config.validate();
    impl_->config = std::move(config);
    impl_->execution = &execution;
    impl_->store = std::make_unique<core::BackendWeightStore>(
        execution.backend(), execution.backend_type(), "auk.audio_tower.weights",
        4ull * 1024ull * 1024ull * 1024ull);

    auto & store = *impl_->store;
    const auto & cfg = impl_->config;
    auto & weights = impl_->weights;
    const auto storage = cfg.weight_storage;
    const auto f32 = assets::TensorStorageType::F32;
    const std::string base = prefix + ".";

    weights.conv1.weight = store.load_tensor(source, base + "conv1.weight", f32, {cfg.d_model, cfg.mel_bins, 3});
    weights.conv1.bias = store.load_tensor(source, base + "conv1.bias", f32, {cfg.d_model});
    weights.conv2.weight = store.load_tensor(source, base + "conv2.weight", f32, {cfg.d_model, cfg.d_model, 3});
    weights.conv2.bias = store.load_tensor(source, base + "conv2.bias", f32, {cfg.d_model});
    weights.ln_post.weight = store.load_tensor(source, base + "ln_post.weight", f32, {cfg.d_model});
    weights.ln_post.bias = store.load_tensor(source, base + "ln_post.bias", f32, {cfg.d_model});
    weights.proj.weight = store.load_tensor(source, base + "proj.weight", storage, {cfg.output_dim, cfg.d_model});
    weights.proj.bias = store.load_tensor(source, base + "proj.bias", f32, {cfg.output_dim});
    weights.loaded = 8;

    weights.layers.reserve(static_cast<size_t>(cfg.layers));
    for (int64_t index = 0; index < cfg.layers; ++index) {
        const std::string layer = base + "layers." + std::to_string(index) + ".";
        LayerWeights value;
        value.attention_norm.weight = store.load_tensor(source, layer + "self_attn_layer_norm.weight", f32, {cfg.d_model});
        value.attention_norm.bias = store.load_tensor(source, layer + "self_attn_layer_norm.bias", f32, {cfg.d_model});
        value.q.weight = store.load_tensor(source, layer + "self_attn.q_proj.weight", storage, {cfg.d_model, cfg.d_model});
        value.q.bias = store.load_tensor(source, layer + "self_attn.q_proj.bias", f32, {cfg.d_model});
        // ⚠ k_proj has NO bias -- the Whisper convention, and the one place this tower
        // differs from qwen3_asr's otherwise identical encoder, whose loader requires one.
        value.k.weight = store.load_tensor(source, layer + "self_attn.k_proj.weight", storage, {cfg.d_model, cfg.d_model});
        value.v.weight = store.load_tensor(source, layer + "self_attn.v_proj.weight", storage, {cfg.d_model, cfg.d_model});
        value.v.bias = store.load_tensor(source, layer + "self_attn.v_proj.bias", f32, {cfg.d_model});
        value.out.weight = store.load_tensor(source, layer + "self_attn.out_proj.weight", storage, {cfg.d_model, cfg.d_model});
        value.out.bias = store.load_tensor(source, layer + "self_attn.out_proj.bias", f32, {cfg.d_model});
        value.final_norm.weight = store.load_tensor(source, layer + "final_layer_norm.weight", f32, {cfg.d_model});
        value.final_norm.bias = store.load_tensor(source, layer + "final_layer_norm.bias", f32, {cfg.d_model});
        value.fc1.weight = store.load_tensor(source, layer + "fc1.weight", storage, {cfg.ffn_dim, cfg.d_model});
        value.fc1.bias = store.load_tensor(source, layer + "fc1.bias", f32, {cfg.ffn_dim});
        value.fc2.weight = store.load_tensor(source, layer + "fc2.weight", storage, {cfg.d_model, cfg.ffn_dim});
        value.fc2.bias = store.load_tensor(source, layer + "fc2.bias", f32, {cfg.d_model});
        weights.loaded += 15;
        weights.layers.push_back(std::move(value));
    }
    store.upload();
}

AukAudioTower::~AukAudioTower() {
    if (impl_ != nullptr) impl_->release();
}

AukAudioEmbeddings AukAudioTower::encode(const std::vector<float> & mel, int64_t frames) {
    if (frames <= 0) throw std::runtime_error("AuK audio tower received no frames");
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->ensure_graph(frames);
    ggml_backend_tensor_set(impl_->mel_input, mel.data(), 0, mel.size() * sizeof(float));
    for (ggml_tensor * tensor : impl_->position_inputs) {
        // Each chunk gets positions from 0, which is what makes them chunk-local.
        const auto positions = sinusoid_positions(tensor->ne[1], impl_->config.d_model);
        ggml_backend_tensor_set(tensor, positions.data(), 0, positions.size() * sizeof(float));
    }
    if (core::compute_graph(*impl_->execution, impl_->graph, impl_->plan, "auk.audio_tower") != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("AuK audio tower graph compute failed");
    }
    AukAudioEmbeddings out;
    out.dim = impl_->output->ne[0];
    out.tokens = impl_->output->ne[1];
    out.values = core::read_tensor_f32(impl_->output);
    return out;
}

int64_t AukAudioTower::loaded_tensor_count() const noexcept { return impl_->weights.loaded; }

}  // namespace engine::models::auk
