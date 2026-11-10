#include "engine/models/stable_audio/rf_dit.h"

#include "engine/framework/modules/transformers/dit_runtime.h"

#include <algorithm>
#include <stdexcept>

namespace engine::models::stable_audio {
namespace {
modules::DitRuntimeConfig runtime_config(const StableAudioConfig & c, assets::TensorStorageType storage) {
    modules::DitRuntimeConfig out;
    out.label = "stable_audio.rf_dit";
    out.weight_type = storage;
    out.graph_metadata_bytes = 512ull * 1024 * 1024;
    out.graph_nodes = 1048576;
    out.time_features = modules::DitTimeFeatures::Fourier;
    out.guidance = modules::DitGuidance::AdaptiveProjected;
    out.conditioning_cache = modules::DitConditioningCache::Unprojected;
    auto & m = out.model;
    m.input_channels = c.io_channels; m.output_channels = c.io_channels;
    m.hidden_size = c.embed_dim; m.intermediate_size = 4 * c.embed_dim; m.layers = c.depth;
    m.context_dim = c.cond_token_dim; m.global_dim = c.global_cond_dim; m.time_features = 256;
    m.memory_tokens = c.num_memory_tokens; m.local_condition_dim = c.local_add_cond_dim;
    m.normalization = modules::DitNormalization::Rms;
    m.modulation = modules::DitModulation::SharedTable;
    m.residual_gate = modules::DitResidualGate::SigmoidOneMinus;
    m.norm_eps = 1e-5f; m.feed_forward = modules::DitFeedForward::SwiGlu; m.projection_precision = GGML_PREC_F32;
    const auto dim = c.embed_dim / c.num_heads;
    m.self_attention = {c.num_heads, c.num_heads, dim, modules::DitQKNorm::Heads, 1e-6f,
        {(std::max)(dim / 2, int64_t{32}), GGML_ROPE_TYPE_NEOX, 10000.f}};
    m.self_attention.differential = c.differential_attention;
    m.cross_attention = m.self_attention;
    m.cross_attention.rotary.dimensions = 0;
    return out;
}

modules::DitWeightNames weight_names() {
    modules::DitWeightNames w;
    w.input = {"model.model.transformer.project_in.weight", ""};
    w.output = {"model.model.transformer.project_out.weight", ""};
    w.preprocess = {"model.model.preprocess_conv.weight", ""};
    w.postprocess = {"model.model.postprocess_conv.weight", ""};
    w.memory_tokens = "model.model.transformer.memory_tokens";
    w.time_embedding = {{"model.model.to_timestep_embed.0.weight", "model.model.to_timestep_embed.0.bias"},
        {"model.model.to_timestep_embed.2.weight", "model.model.to_timestep_embed.2.bias"}};
    w.context_embedding = {{"model.model.to_cond_embed.0.weight", ""}, {"model.model.to_cond_embed.2.weight", ""}};
    w.global_embedding = {{"model.model.to_global_embed.0.weight", ""}, {"model.model.to_global_embed.2.weight", ""}};
    w.global_modulation = {{"model.model.transformer.global_cond_embedder.0.weight", "model.model.transformer.global_cond_embedder.0.bias"},
        {"model.model.transformer.global_cond_embedder.2.weight", "model.model.transformer.global_cond_embedder.2.bias"}};
    w.modulation_planes = {1, 0, 2, 4, 3, 5};
    w.layer.modulation_table = "model.model.transformer.layers.{layer}.to_scale_shift_gate";
    w.layer.self_norm = {"model.model.transformer.layers.{layer}.pre_norm.gamma", ""};
    w.layer.cross_norm = {"model.model.transformer.layers.{layer}.cross_attend_norm.gamma", ""};
    w.layer.feed_forward_norm = {"model.model.transformer.layers.{layer}.ff_norm.gamma", ""};
    w.layer.self_attention.qkv = {"model.model.transformer.layers.{layer}.self_attn.to_qkv.weight", ""};
    w.layer.self_attention.output = {"model.model.transformer.layers.{layer}.self_attn.to_out.weight", ""};
    w.layer.self_attention.query_norm = {"model.model.transformer.layers.{layer}.self_attn.q_norm.gamma", ""};
    w.layer.self_attention.key_norm = {"model.model.transformer.layers.{layer}.self_attn.k_norm.gamma", ""};
    w.layer.cross_attention.query = {"model.model.transformer.layers.{layer}.cross_attn.to_q.weight", ""};
    w.layer.cross_attention.kv = {"model.model.transformer.layers.{layer}.cross_attn.to_kv.weight", ""};
    w.layer.cross_attention.output = {"model.model.transformer.layers.{layer}.cross_attn.to_out.weight", ""};
    w.layer.cross_attention.query_norm = {"model.model.transformer.layers.{layer}.cross_attn.q_norm.gamma", ""};
    w.layer.cross_attention.key_norm = {"model.model.transformer.layers.{layer}.cross_attn.k_norm.gamma", ""};
    w.layer.local_embedding = {{"model.model.transformer.layers.{layer}.to_local_embed.0.weight", "model.model.transformer.layers.{layer}.to_local_embed.0.bias"},
        {"model.model.transformer.layers.{layer}.to_local_embed.2.weight", "model.model.transformer.layers.{layer}.to_local_embed.2.bias"}};
    w.layer.feed_forward_in = {"model.model.transformer.layers.{layer}.ff.ff.0.proj.weight", "model.model.transformer.layers.{layer}.ff.ff.0.proj.bias"};
    w.layer.feed_forward_out = {"model.model.transformer.layers.{layer}.ff.ff.2.weight", "model.model.transformer.layers.{layer}.ff.ff.2.bias"};
    return w;
}
} // namespace

StableAudioRFDiTRuntime::StableAudioRFDiTRuntime(core::ExecutionContext & execution,
    std::shared_ptr<const StableAudioAssets> assets, assets::TensorStorageType storage)
    : execution_(&execution), assets_(std::move(assets)), weight_storage_type_(storage) {}
StableAudioRFDiTRuntime::~StableAudioRFDiTRuntime() = default;

void StableAudioRFDiTRuntime::prepare(const StableAudioSamplingState &, float) const {
    if (!runtime_) runtime_ = std::make_unique<modules::DitRuntime>(*execution_, *assets_->model_weights,
        runtime_config(assets_->config, weight_storage_type_), weight_names());
}

std::vector<float> StableAudioRFDiTRuntime::sample(const StableAudioSamplingState & sampling,
    const StableAudioConditioningInputs & conditioning, uint64_t seed, uint64_t & rng_offset_blocks,
    const engine::sampling::TorchCudaSamplingPolicy & rng_policy, std::string_view sampler_type,
    float cfg_scale, float apg_scale) const {
    prepare(sampling, cfg_scale);
    const auto & c = assets_->config;
    const int64_t branches = cfg_scale == 1.f ? 1 : 2;
    const auto rows = branches * sampling.batch, tokens = c.prompt_max_length + 1;
    const auto & positive = conditioning.positive;
    if (positive.batch != sampling.batch || positive.cross_tokens != tokens || positive.cond_dim != c.cond_dim ||
        positive.cross_attention.size() != static_cast<size_t>(sampling.batch * tokens * c.cond_dim) ||
        positive.global.size() != static_cast<size_t>(sampling.batch * c.global_cond_dim) ||
        sampling.padding_mask.size() != static_cast<size_t>(sampling.batch * sampling.latent_sample_size) ||
        sampling.local_add_conditioning.size() != static_cast<size_t>(sampling.batch * sampling.latent_sample_size * c.local_add_cond_dim))
        throw std::invalid_argument("Stable Audio DiT conditioning shape mismatch");
    const auto & negative = conditioning.negative;
    if (conditioning.has_negative && (negative.batch != sampling.batch || negative.cross_tokens != tokens ||
        negative.cond_dim != c.cond_dim || negative.cross_attention.size() != positive.cross_attention.size() ||
        negative.cross_attention_mask.size() != static_cast<size_t>(sampling.batch * tokens)))
        throw std::invalid_argument("Stable Audio DiT negative conditioning shape mismatch");
    modules::DitConditioning in;
    in.batch = sampling.batch; in.frames = sampling.latent_sample_size; in.branches = branches; in.context_tokens = tokens;
    in.context.assign(static_cast<size_t>(rows * tokens * c.cond_dim), 0);
    std::copy(positive.cross_attention.begin(), positive.cross_attention.end(), in.context.begin());
    if (branches == 2 && conditioning.has_negative) {
        for (int64_t b = 0; b < sampling.batch; ++b) for (int64_t t = 0; t < tokens; ++t) {
            if (!negative.cross_attention_mask[static_cast<size_t>(b * tokens + t)]) continue;
            const auto offset = static_cast<size_t>((b * tokens + t) * c.cond_dim);
            std::copy_n(negative.cross_attention.begin() + static_cast<std::ptrdiff_t>(offset),
                static_cast<size_t>(c.cond_dim), in.context.begin() + static_cast<std::ptrdiff_t>(positive.cross_attention.size() + offset));
        }
    }
    in.global = positive.global;
    in.local_condition = sampling.local_add_conditioning;
    if (branches == 2) {
        in.global.insert(in.global.end(), positive.global.begin(), positive.global.end());
        in.local_condition.insert(in.local_condition.end(), sampling.local_add_conditioning.begin(), sampling.local_add_conditioning.end());
    }
    in.latent_mask = sampling.padding_mask;
    const auto full_tokens = in.frames + c.num_memory_tokens;
    in.value_mask.resize(static_cast<size_t>(rows * full_tokens), 1);
    for (int64_t b = 0; b < rows; ++b)
        std::copy_n(sampling.padding_mask.begin() + static_cast<std::ptrdiff_t>((b % sampling.batch) * in.frames),
            static_cast<size_t>(in.frames), in.value_mask.begin() + static_cast<std::ptrdiff_t>(b * full_tokens + c.num_memory_tokens));
    runtime_->prepare_conditioning(std::move(in));
    modules::DitSamplingRequest request;
    if (sampler_type.empty() || sampler_type == "pingpong") request.sampler = modules::DitSampler::PingPong;
    else if (sampler_type != "euler") throw std::invalid_argument("Stable Audio sampler_type must be pingpong or euler");
    request.noise = sampling.noise; request.schedule = sampling.schedule; request.steps = sampling.schedule_points - 1;
    request.guidance = cfg_scale; request.apg_scale = apg_scale;
    request.seed = seed; request.rng_offset_blocks = &rng_offset_blocks; request.rng_policy = rng_policy;
    return runtime_->sample(request);
}

void StableAudioRFDiTRuntime::release_runtime_graphs() const {
    if (runtime_) runtime_->release_graphs();
}
} // namespace engine::models::stable_audio
