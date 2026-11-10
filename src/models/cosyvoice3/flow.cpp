#include "engine/models/cosyvoice3/flow.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/feed_forward_modules.h"
#include "engine/framework/modules/attention/projected_grouped_self_attention.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"
#include "engine/models/cosyvoice3/rknn_flow.h"
#include "engine/framework/modules/transformers/dit_runtime.h"

#include <ggml-alloc.h>
#include <ggml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::cosyvoice3 {
namespace {

namespace binding = engine::modules::binding;
namespace core = engine::core;
namespace modules = engine::modules;

constexpr float kPi = 3.14159265358979323846F;
constexpr float kInferenceCfgRate = 0.7F;
constexpr int64_t kTimeEmbeddingSize = 256;
constexpr int64_t kConvPosGroups = 16;
constexpr int64_t kConvPosKernel = 31;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct GgmlGallocrDeleter {
    void operator()(ggml_gallocr_t alloc) const noexcept {
        if (alloc != nullptr) {
            ggml_gallocr_free(alloc);
        }
    }
};

struct GraphMemory {
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx;
    std::unique_ptr<ggml_context, GgmlContextDeleter> input_ctx;
    std::unique_ptr<std::remove_pointer_t<ggml_gallocr_t>, GgmlGallocrDeleter> gallocr;
    ggml_backend_buffer_t input_buffer = nullptr;
    ggml_cgraph * graph = nullptr;

    ~GraphMemory() {
        reset(nullptr);
    }

    void reset(ggml_backend_t backend) {
        if (graph != nullptr && backend != nullptr) {
            core::release_backend_graph_resources(backend, graph);
        }
        graph = nullptr;
        gallocr.reset();
        if (input_buffer != nullptr) {
            ggml_backend_buffer_free(input_buffer);
            input_buffer = nullptr;
        }
        input_ctx.reset();
        ctx.reset();
    }
};

std::vector<float> cosine_time_schedule(int64_t steps) {
    if (steps <= 0) {
        throw std::runtime_error("CosyVoice3 num_inference_steps must be positive");
    }
    std::vector<float> out(static_cast<size_t>(steps + 1));
    for (int64_t i = 0; i <= steps; ++i) {
        const float u = static_cast<float>(i) / static_cast<float>(steps);
        out[static_cast<size_t>(i)] = 1.0F - std::cos(u * 0.5F * kPi);
    }
    return out;
}

struct CosyDiTFlowWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    core::TensorValue token_embedding;
    modules::LinearWeights speaker_projection;
    modules::Conv1dWeights pre_lookahead_conv1;
    modules::Conv1dWeights pre_lookahead_conv2;
    core::TensorValue rand_noise;
    std::vector<float> rand_noise_host;
};

std::shared_ptr<CosyDiTFlowWeights> load_flow_weights(
    const CosyVoice3Assets & assets,
    core::ExecutionContext & execution,
    size_t weight_context_bytes,
    engine::assets::TensorStorageType storage_type) {
    auto weights = std::make_shared<CosyDiTFlowWeights>();
    weights->store = std::make_shared<core::BackendWeightStore>(
        execution.backend(),
        execution.backend_type(),
        "cosyvoice3.flow.weights",
        weight_context_bytes);
    const auto & source = *assets.flow_weights;
    const auto & c = assets.config;
    weights->token_embedding = weights->store->load_tensor(source, "input_embedding.weight", storage_type, {c.speech_token_size, c.flow_mel_channels});
    weights->speaker_projection = binding::linear_from_source(
        *weights->store, source, "spk_embed_affine_layer", storage_type, c.flow_mel_channels, c.speaker_dim, true);
    weights->pre_lookahead_conv1 = binding::conv1d_from_source(
        *weights->store, source, "pre_lookahead_layer.conv1", storage_type, c.flow_hidden_size, c.flow_mel_channels, c.pre_lookahead_len + 1, true);
    weights->pre_lookahead_conv2 = binding::conv1d_from_source(
        *weights->store, source, "pre_lookahead_layer.conv2", storage_type, c.flow_mel_channels, c.flow_hidden_size, 3, true);
    weights->rand_noise = weights->store->load_tensor(source, "decoder.rand_noise", engine::assets::TensorStorageType::F32, {1, c.flow_mel_channels, 50 * 300});
    weights->rand_noise_host = source.require_f32("decoder.rand_noise", {1, c.flow_mel_channels, 50 * 300});
    weights->store->upload();
    return weights;
}

modules::DitRuntimeConfig dit_config(const CosyVoice3Config & c, assets::TensorStorageType storage,
    size_t graph_bytes, size_t weight_bytes) {
    modules::DitRuntimeConfig out;
    out.label = "cosyvoice3.flow.dit";
    out.weight_type = storage; out.weight_metadata_bytes = weight_bytes;
    out.graph_metadata_bytes = graph_bytes; out.graph_nodes = 200000;
    out.time_features = modules::DitTimeFeatures::SinCos;
    out.time_frequency_denominator = kTimeEmbeddingSize / 2 - 1; out.time_scale = 1000.f;
    auto & m = out.model;
    m.input_channels = 4 * c.flow_mel_channels; m.output_channels = c.flow_mel_channels;
    m.hidden_size = c.flow_hidden_size; m.intermediate_size = c.flow_hidden_size * c.flow_ff_mult;
    m.layers = c.flow_layers; m.time_features = kTimeEmbeddingSize;
    m.modulation = modules::DitModulation::PerLayer;
    m.position_kernel = kConvPosKernel; m.position_groups = kConvPosGroups;
    m.self_attention = {c.flow_heads, c.flow_heads, c.flow_head_dim, modules::DitQKNorm::None, 1e-6f,
        {c.flow_head_dim, GGML_ROPE_TYPE_NORMAL, 10000.f}};
    return out;
}

modules::DitWeightNames dit_weight_names() {
    modules::DitWeightNames w;
    w.input = {"decoder.estimator.input_embed.proj.weight", "decoder.estimator.input_embed.proj.bias"};
    w.output = {"decoder.estimator.proj_out.weight", "decoder.estimator.proj_out.bias"};
    w.time_embedding = {{"decoder.estimator.time_embed.time_mlp.0.weight", "decoder.estimator.time_embed.time_mlp.0.bias"},
        {"decoder.estimator.time_embed.time_mlp.2.weight", "decoder.estimator.time_embed.time_mlp.2.bias"}};
    w.position_convolutions = {{"decoder.estimator.input_embed.conv_pos_embed.conv1.0.weight", "decoder.estimator.input_embed.conv_pos_embed.conv1.0.bias"},
        {"decoder.estimator.input_embed.conv_pos_embed.conv2.0.weight", "decoder.estimator.input_embed.conv_pos_embed.conv2.0.bias"}};
    w.layer.modulation = {"decoder.estimator.transformer_blocks.{layer}.attn_norm.linear.weight", "decoder.estimator.transformer_blocks.{layer}.attn_norm.linear.bias"};
    w.layer.self_attention = {
        {"decoder.estimator.transformer_blocks.{layer}.attn.to_q.weight", "decoder.estimator.transformer_blocks.{layer}.attn.to_q.bias"},
        {"decoder.estimator.transformer_blocks.{layer}.attn.to_k.weight", "decoder.estimator.transformer_blocks.{layer}.attn.to_k.bias"},
        {"decoder.estimator.transformer_blocks.{layer}.attn.to_v.weight", "decoder.estimator.transformer_blocks.{layer}.attn.to_v.bias"},
        {"decoder.estimator.transformer_blocks.{layer}.attn.to_out.0.weight", "decoder.estimator.transformer_blocks.{layer}.attn.to_out.0.bias"}, {}, {}, {}, {}};
    w.layer.feed_forward_in = {"decoder.estimator.transformer_blocks.{layer}.ff.ff.0.0.weight", "decoder.estimator.transformer_blocks.{layer}.ff.ff.0.0.bias"};
    w.layer.feed_forward_out = {"decoder.estimator.transformer_blocks.{layer}.ff.ff.2.weight", "decoder.estimator.transformer_blocks.{layer}.ff.ff.2.bias"};
    w.final_modulation = {"decoder.estimator.norm_out.linear.weight", "decoder.estimator.norm_out.linear.bias"};
    w.final_shift_plane = 1; w.final_scale_plane = 0;
    return w;
}

class ConditionGraph {
public:
    ConditionGraph(
        core::ExecutionContext & execution,
        std::shared_ptr<const CosyDiTFlowWeights> weights,
        CosyVoice3Config config,
        size_t graph_arena_bytes)
        : execution_(execution),
          weights_(std::move(weights)),
          config_(config),
          graph_arena_bytes_(graph_arena_bytes) {}

    ~ConditionGraph() {
        mem_.reset(execution_.backend());
    }

    std::vector<float> run(const std::vector<int32_t> & prompt_tokens, const std::vector<int32_t> & target_tokens) {
        const int64_t tokens = static_cast<int64_t>(prompt_tokens.size() + target_tokens.size());
        if (tokens <= 0) {
            throw std::runtime_error("CosyVoice3 flow requires speech tokens");
        }
        ensure(tokens);
        std::vector<int32_t> all_tokens;
        all_tokens.reserve(static_cast<size_t>(tokens));
        all_tokens.insert(all_tokens.end(), prompt_tokens.begin(), prompt_tokens.end());
        all_tokens.insert(all_tokens.end(), target_tokens.begin(), target_tokens.end());
        ggml_backend_tensor_set(token_ids_, all_tokens.data(), 0, all_tokens.size() * sizeof(int32_t));
        if (core::compute_backend_graph(execution_.backend(), mem_.graph, nullptr, "cosyvoice3.flow.condition") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("CosyVoice3 flow condition graph compute failed");
        }
        auto output = core::read_tensor_f32(output_.tensor);
        return output;
    }

    void release_graph() {
        mem_.reset(execution_.backend());
        tokens_ = 0;
        token_ids_ = nullptr;
        output_ = {};
    }

private:
    void ensure(int64_t tokens) {
        if (mem_.graph != nullptr && tokens_ == tokens) {
            return;
        }
        mem_.reset(execution_.backend());
        ggml_init_params params{graph_arena_bytes_, nullptr, true};
        mem_.ctx.reset(ggml_init(params));
        ggml_init_params input_params{16ull * 1024ull * 1024ull, nullptr, true};
        mem_.input_ctx.reset(ggml_init(input_params));
        core::ModuleBuildContext ctx{mem_.ctx.get(), "cosyvoice3.flow.condition", execution_.backend_type()};
        core::ModuleBuildContext input_ctx{mem_.input_ctx.get(), "cosyvoice3.flow.condition.inputs", execution_.backend_type()};

        auto ids = core::make_tensor(input_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({1, tokens}));
        token_ids_ = ids.tensor;
        ggml_set_input(token_ids_);
        auto h = modules::EmbeddingModule({config_.speech_token_size, config_.flow_mel_channels})
                     .build(ctx, ids, weights_->token_embedding);
        const auto residual = h;
        h = modules::TransposeModule({{0, 2, 1, 3}, h.shape.rank}).build(ctx, h);
        if (execution_.backend_type() == core::BackendType::Metal) {
            h = core::ensure_backend_addressable_layout(ctx, h);
        }
        h = core::wrap_tensor(
            ggml_pad_ext(ctx.ggml, h.tensor, 0, config_.pre_lookahead_len, 0, 0, 0, 0, 0, 0),
            core::TensorShape::from_dims({1, config_.flow_mel_channels, tokens + config_.pre_lookahead_len}),
            GGML_TYPE_F32);
        h = modules::Conv1dModule({config_.flow_mel_channels, config_.flow_hidden_size, config_.pre_lookahead_len + 1, 1, 0, 1, true})
                .build(ctx, h, weights_->pre_lookahead_conv1);
        h = modules::LeakyReluModule().build(ctx, h);
        h = core::wrap_tensor(
            ggml_pad_ext(ctx.ggml, h.tensor, 2, 0, 0, 0, 0, 0, 0, 0),
            core::TensorShape::from_dims({1, config_.flow_hidden_size, tokens + 2}),
            GGML_TYPE_F32);
        h = modules::Conv1dModule({config_.flow_hidden_size, config_.flow_mel_channels, 3, 1, 0, 1, true})
                .build(ctx, h, weights_->pre_lookahead_conv2);
        h = modules::TransposeModule({{0, 2, 1, 3}, h.shape.rank}).build(ctx, h);
        h = modules::AddModule().build(ctx, h, residual);
        h = modules::Interpolate1dModule({tokens * config_.token_mel_ratio, modules::Interpolate1dMode::Nearest})
                .build(ctx, modules::TransposeModule({{0, 2, 1, 3}, h.shape.rank}).build(ctx, h));
        h = modules::TransposeModule({{0, 2, 1, 3}, h.shape.rank}).build(ctx, h);
        output_ = core::ensure_backend_addressable_layout(ctx, h);
        ggml_set_output(output_.tensor);
        mem_.graph = ggml_new_graph_custom(mem_.ctx.get(), 20000, false);
        ggml_build_forward_expand(mem_.graph, output_.tensor);
        mem_.input_buffer = ggml_backend_alloc_ctx_tensors(mem_.input_ctx.get(), execution_.backend());
        mem_.gallocr.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend())));
        if (mem_.input_buffer == nullptr || mem_.gallocr == nullptr ||
            !ggml_gallocr_reserve(mem_.gallocr.get(), mem_.graph) ||
            !ggml_gallocr_alloc_graph(mem_.gallocr.get(), mem_.graph)) {
            mem_.reset(execution_.backend());
            throw std::runtime_error("failed to allocate CosyVoice3 flow condition graph");
        }
        tokens_ = tokens;
    }

    core::ExecutionContext & execution_;
    std::shared_ptr<const CosyDiTFlowWeights> weights_;
    CosyVoice3Config config_;
    size_t graph_arena_bytes_ = 0;
    GraphMemory mem_;
    int64_t tokens_ = 0;
    ggml_tensor * token_ids_ = nullptr;
    core::TensorValue output_;
};

std::vector<float> normalize_and_project_speaker(
    const CosyVoice3Config & config,
    const CosyDiTFlowWeights & weights,
    core::ExecutionContext & execution,
    const std::vector<float> & speaker_embedding,
    size_t graph_arena_bytes) {
    if (static_cast<int64_t>(speaker_embedding.size()) != config.speaker_dim) {
        throw std::runtime_error("CosyVoice3 speaker embedding size mismatch");
    }
    float norm_sq = 0.0F;
    for (float value : speaker_embedding) {
        norm_sq += value * value;
    }
    const float inv_norm = 1.0F / std::sqrt(std::max(norm_sq, 1.0e-12F));
    std::vector<float> normalized(speaker_embedding.size());
    for (size_t i = 0; i < speaker_embedding.size(); ++i) {
        normalized[i] = speaker_embedding[i] * inv_norm;
    }

    GraphMemory mem;
    ggml_init_params params{graph_arena_bytes, nullptr, true};
    mem.ctx.reset(ggml_init(params));
    ggml_init_params input_params{4ull * 1024ull * 1024ull, nullptr, true};
    mem.input_ctx.reset(ggml_init(input_params));
    core::ModuleBuildContext ctx{mem.ctx.get(), "cosyvoice3.flow.speaker", execution.backend_type()};
    core::ModuleBuildContext input_ctx{mem.input_ctx.get(), "cosyvoice3.flow.speaker.inputs", execution.backend_type()};
    auto input = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, config.speaker_dim}));
    ggml_set_input(input.tensor);
    auto output = modules::LinearModule({config.speaker_dim, config.flow_mel_channels, true}).build(ctx, input, weights.speaker_projection);
    output = core::ensure_backend_addressable_layout(ctx, output);
    ggml_set_output(output.tensor);
    mem.graph = ggml_new_graph_custom(mem.ctx.get(), 4096, false);
    ggml_build_forward_expand(mem.graph, output.tensor);
    mem.input_buffer = ggml_backend_alloc_ctx_tensors(mem.input_ctx.get(), execution.backend());
    mem.gallocr.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend())));
    if (mem.input_buffer == nullptr || mem.gallocr == nullptr ||
        !ggml_gallocr_reserve(mem.gallocr.get(), mem.graph) ||
        !ggml_gallocr_alloc_graph(mem.gallocr.get(), mem.graph)) {
        throw std::runtime_error("CosyVoice3 speaker projection graph failed");
    }
    ggml_backend_tensor_set(input.tensor, normalized.data(), 0, normalized.size() * sizeof(float));
    if (core::compute_backend_graph(execution.backend(), mem.graph, nullptr, "cosyvoice3.flow.speaker") != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("CosyVoice3 speaker projection graph failed");
    }
    auto out = core::read_tensor_f32(output.tensor);
    mem.reset(execution.backend());
    return out;
}

std::vector<float> read_noise_prefix(
    const CosyVoice3Config & config,
    const CosyDiTFlowWeights & weights,
    int64_t frames) {
    if (frames > 50 * 300) {
        throw std::runtime_error("CosyVoice3 requested mel frames exceed fixed flow noise capacity");
    }
    const auto & full = weights.rand_noise_host;
    std::vector<float> out(static_cast<size_t>(config.flow_mel_channels * frames));
    for (int64_t ch = 0; ch < config.flow_mel_channels; ++ch) {
        const auto src = full.begin() + static_cast<std::ptrdiff_t>(ch * 50 * 300);
        const auto dst = out.begin() + static_cast<std::ptrdiff_t>(ch * frames);
        std::copy(src, src + frames, dst);
    }
    return out;
}

}  // namespace

class CosyVoice3DiTFlowRuntime::Impl {
public:
    Impl(
        std::shared_ptr<const CosyVoice3Assets> assets,
        core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        size_t weight_context_bytes,
        engine::assets::TensorStorageType storage_type,
        const std::vector<std::string> & rknn_engine_paths)
        : assets_(std::move(assets)),
          execution_(execution),
          weights_(load_flow_weights(
              *assets_,
              execution_,
              weight_context_bytes,
              storage_type)),
          condition_(execution_, weights_, assets_->config, graph_arena_bytes),
          dit_(execution_, *assets_->flow_weights, dit_config(assets_->config, storage_type, graph_arena_bytes, weight_context_bytes), dit_weight_names()),
          graph_arena_bytes_(graph_arena_bytes) {
        if (assets_ == nullptr) {
            throw std::runtime_error("CosyVoice3 flow runtime requires assets");
        }
        if (!rknn_engine_paths.empty()) {
            // The engines' own channel count is checked against the checkpoint when an engine
            // loads, so nothing to compare here: registering the ladder loads none of it.
            rknn_ = std::make_unique<CosyVoice3RknnFlow>(
                rknn_engine_paths, assets_->config.flow_mel_channels);
            debug::trace_log_scalar("cosyvoice3.flow.dit.rknn.engines", rknn_->engine_count());
            debug::trace_log_scalar("cosyvoice3.flow.dit.rknn.max_frames", rknn_->max_frames());
        }
    }

    CosyVoice3FlowOutput generate(const CosyVoice3FlowRequest & request) {
        const auto start = std::chrono::steady_clock::now();
        const auto & c = assets_->config;
        if (request.speech_tokens.empty()) {
            throw std::runtime_error("CosyVoice3 flow requires generated speech tokens");
        }
        if (request.prompt_mel_frames <= 0 ||
            static_cast<int64_t>(request.prompt_mel.size()) != request.prompt_mel_frames * c.flow_mel_channels) {
            throw std::runtime_error("CosyVoice3 flow prompt mel size mismatch");
        }
        const auto cond_start = std::chrono::steady_clock::now();
        auto mu_btc = condition_.run(request.prompt_speech_tokens, request.speech_tokens);
        const int64_t total_frames = static_cast<int64_t>(mu_btc.size()) / c.flow_mel_channels;
        const int64_t target_frames = total_frames - request.prompt_mel_frames;
        if (target_frames <= 0) {
            throw std::runtime_error("CosyVoice3 flow target mel frames must be positive");
        }
        engine::debug::timing_log_scalar("cosyvoice3.flow.condition_ms", engine::debug::elapsed_ms(cond_start));

        std::vector<float> mu(static_cast<size_t>(2 * c.flow_mel_channels * total_frames), 0.0F);
        for (int64_t frame = 0; frame < total_frames; ++frame) {
            for (int64_t ch = 0; ch < c.flow_mel_channels; ++ch) {
                mu[static_cast<size_t>(ch * total_frames + frame)] =
                    mu_btc[static_cast<size_t>(frame * c.flow_mel_channels + ch)];
            }
        }
        std::vector<float> cond(static_cast<size_t>(2 * c.flow_mel_channels * total_frames), 0.0F);
        for (int64_t frame = 0; frame < request.prompt_mel_frames; ++frame) {
            for (int64_t ch = 0; ch < c.flow_mel_channels; ++ch) {
                cond[static_cast<size_t>(ch * total_frames + frame)] =
                    request.prompt_mel[static_cast<size_t>(frame * c.flow_mel_channels + ch)];
            }
        }
        auto x = read_noise_prefix(c, *weights_, total_frames);
        std::vector<float> x_batched(static_cast<size_t>(2 * c.flow_mel_channels * total_frames));
        std::copy(x.begin(), x.end(), x_batched.begin());
        std::copy(x.begin(), x.end(), x_batched.begin() + static_cast<std::ptrdiff_t>(x.size()));

        auto speaker_cond = normalize_and_project_speaker(
            c,
            *weights_,
            execution_,
            request.speaker_embedding,
            graph_arena_bytes_);
        std::vector<float> spks(static_cast<size_t>(2 * c.flow_mel_channels), 0.0F);
        std::copy(speaker_cond.begin(), speaker_cond.end(), spks.begin());

        const bool on_npu = rknn_ != nullptr && total_frames <= rknn_->max_frames();
        if (rknn_ != nullptr && !on_npu) debug::timing_log_scalar(
            "cosyvoice3.flow.dit.rknn.fallback_frames", static_cast<int64_t>(total_frames));
        const auto dit_start = std::chrono::steady_clock::now();
        if (on_npu) {
            const auto schedule = cosine_time_schedule(request.num_inference_steps);
            for (size_t step = 1; step < schedule.size(); ++step) {
                const float t = schedule[step - 1], dt = schedule[step] - t;
                const auto pred = rknn_->run(x_batched.data(), mu.data(), cond.data(), spks.data(), t, total_frames);
                const auto branch = static_cast<size_t>(c.flow_mel_channels * total_frames);
                for (size_t i = 0; i < branch; ++i) x_batched[i] += dt * ((1.f + kInferenceCfgRate) * pred[i] - kInferenceCfgRate * pred[branch + i]);
                std::copy_n(x_batched.begin(), branch, x_batched.begin() + static_cast<std::ptrdiff_t>(branch));
            }
        } else {
            modules::DitConditioning conditioning;
            conditioning.frames = total_frames; conditioning.branches = 2;
            conditioning.frame_channels = {c.flow_mel_channels, c.flow_mel_channels};
            conditioning.frame_features.resize(2);
            for (size_t n = 0; n < 2; ++n) {
                const auto & values = n == 0 ? cond : mu;
                auto & feature = conditioning.frame_features[n];
                feature.resize(values.size());
                for (int64_t b = 0; b < 2; ++b)
                    for (int64_t t = 0; t < total_frames; ++t)
                        for (int64_t ch = 0; ch < c.flow_mel_channels; ++ch)
                            feature[static_cast<size_t>((b * total_frames + t) * c.flow_mel_channels + ch)] = values[static_cast<size_t>((b * c.flow_mel_channels + ch) * total_frames + t)];
            }
            conditioning.vector_channels = {c.flow_mel_channels}; conditioning.vector_features = {spks};
            dit_.prepare_conditioning(std::move(conditioning));
            modules::DitSamplingRequest sampling;
            sampling.noise = x; sampling.steps = request.num_inference_steps;
            sampling.schedule_type = modules::DitSchedule::Cosine;
            sampling.guidance = 1.f + kInferenceCfgRate;
            x_batched = dit_.sample(sampling);
        }
        engine::debug::timing_log_scalar("cosyvoice3.flow.dit_ms", engine::debug::elapsed_ms(dit_start));

        CosyVoice3FlowOutput out;
        out.frames = target_frames;
        out.mel.resize(static_cast<size_t>(target_frames * c.flow_mel_channels));
        for (int64_t frame = 0; frame < target_frames; ++frame) {
            for (int64_t ch = 0; ch < c.flow_mel_channels; ++ch) {
                out.mel[static_cast<size_t>(frame * c.flow_mel_channels + ch)] =
                    x_batched[static_cast<size_t>(ch * total_frames + request.prompt_mel_frames + frame)];
            }
        }
        engine::debug::timing_log_scalar("cosyvoice3.flow.total_ms", engine::debug::elapsed_ms(start));
        engine::debug::trace_log_scalar("cosyvoice3.flow.total_frames", static_cast<double>(total_frames));
        engine::debug::trace_log_scalar("cosyvoice3.flow.target_frames", static_cast<double>(target_frames));
        return out;
    }

    void release_graphs() {
        condition_.release_graph();
        dit_.release_graphs();
    }

private:
    std::shared_ptr<const CosyVoice3Assets> assets_;
    core::ExecutionContext & execution_;
    std::shared_ptr<CosyDiTFlowWeights> weights_;
    ConditionGraph condition_;
    modules::DitRuntime dit_;
    std::unique_ptr<CosyVoice3RknnFlow> rknn_;
    size_t graph_arena_bytes_ = 0;
};

CosyVoice3DiTFlowRuntime::CosyVoice3DiTFlowRuntime(
    std::shared_ptr<const CosyVoice3Assets> assets,
    engine::core::ExecutionContext & execution,
    size_t graph_arena_bytes,
    size_t weight_context_bytes,
    engine::assets::TensorStorageType storage_type,
    const std::vector<std::string> & rknn_engine_paths)
    : impl_(std::make_unique<Impl>(
          std::move(assets),
          execution,
          graph_arena_bytes,
          weight_context_bytes,
          storage_type,
          rknn_engine_paths)) {}

CosyVoice3DiTFlowRuntime::~CosyVoice3DiTFlowRuntime() = default;

CosyVoice3FlowOutput CosyVoice3DiTFlowRuntime::generate(const CosyVoice3FlowRequest & request) {
    return impl_->generate(request);
}

void CosyVoice3DiTFlowRuntime::release_graphs() {
    impl_->release_graphs();
}

}  // namespace engine::models::cosyvoice3
