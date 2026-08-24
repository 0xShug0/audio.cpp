#include "engine/models/firered_audio/flow.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/feed_forward.h"
#include "engine/framework/modules/attention/projected_grouped_self_attention.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <ggml-alloc.h>
#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::firered_audio {
namespace {

namespace binding = engine::modules::binding;
namespace core = engine::core;
namespace modules = engine::modules;
constexpr float kPi = 3.14159265358979323846F;

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

std::vector<int32_t> position_ids(int64_t steps) {
    std::vector<int32_t> out(static_cast<size_t>(steps));
    for (int64_t i = 0; i < steps; ++i) {
        out[static_cast<size_t>(i)] = static_cast<int32_t>(i);
    }
    return out;
}

core::TensorValue mish(core::ModuleBuildContext & ctx, const core::TensorValue & x) {
    auto softplus = core::wrap_tensor(ggml_softplus(ctx.ggml, x.tensor), x.shape, GGML_TYPE_F32);
    auto t = core::wrap_tensor(ggml_tanh(ctx.ggml, softplus.tensor), x.shape, GGML_TYPE_F32);
    return core::wrap_tensor(ggml_mul(ctx.ggml, x.tensor, t.tensor), x.shape, GGML_TYPE_F32);
}

core::TensorValue mul_broadcast(core::ModuleBuildContext & ctx, const core::TensorValue & x, const core::TensorValue & scale) {
    auto s = modules::RepeatModule({x.shape}).build(ctx, scale);
    return core::wrap_tensor(ggml_mul(ctx.ggml, x.tensor, s.tensor), x.shape, GGML_TYPE_F32);
}

core::TensorValue modulate(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & x,
    const core::TensorValue & shift,
    const core::TensorValue & scale) {
    auto s = modules::RepeatModule({x.shape}).build(ctx, scale);
    auto sh = modules::RepeatModule({x.shape}).build(ctx, shift);
    auto one_plus = core::wrap_tensor(ggml_scale_bias(ctx.ggml, s.tensor, 1.0F, 1.0F), x.shape, GGML_TYPE_F32);
    return core::wrap_tensor(ggml_add(ctx.ggml, ggml_mul(ctx.ggml, x.tensor, one_plus.tensor), sh.tensor), x.shape, GGML_TYPE_F32);
}

struct FireRedDiTBlockWeights {
    modules::NormWeights norm1;
    modules::ProjectedGroupedSelfAttentionWeights attention;
    modules::NormWeights norm2;
    modules::Conv1dWeights conv1;
    modules::Conv1dWeights conv2;
    modules::NormWeights norm3;
    modules::FeedForwardWeights mlp;
    modules::LinearWeights ada;
};


struct FireRedFlowWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    modules::LinearWeights backbone_input_proj;
    modules::LinearWeights dit_in;
    modules::LinearWeights dit_time_fc1;
    modules::LinearWeights dit_time_fc2;
    std::vector<FireRedDiTBlockWeights> dit_blocks;
    modules::LinearWeights dit_final_ada;
    modules::LinearWeights dit_final_linear;
};

modules::ProjectedGroupedSelfAttentionWeights load_projected_attention(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t hidden,
    assets::TensorStorageType storage_type) {
    modules::ProjectedGroupedSelfAttentionWeights out;
    out.q_proj = binding::linear_from_source(store, source, prefix + ".to_q", storage_type, hidden, hidden, true);
    out.k_proj = binding::linear_from_source(store, source, prefix + ".to_k", storage_type, hidden, hidden, true);
    out.v_proj = binding::linear_from_source(store, source, prefix + ".to_v", storage_type, hidden, hidden, true);
    out.o_proj = binding::linear_from_source(store, source, prefix + ".to_out.0", storage_type, hidden, hidden, true);
    return out;
}

FireRedDiTBlockWeights load_dit_block(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t hidden,
    int64_t intermediate,
    assets::TensorStorageType storage_type) {
    FireRedDiTBlockWeights out;
    out.norm1 = binding::norm_weight_from_source(store, source, prefix + ".norm1", hidden);
    out.attention = load_projected_attention(store, source, prefix + ".attn", hidden, storage_type);
    out.norm2 = binding::norm_weight_from_source(store, source, prefix + ".norm2", hidden);
    out.conv1 = binding::conv1d_from_source(store, source, prefix + ".conv.block.0", storage_type, hidden, hidden, 3, true);
    out.conv2 = binding::conv1d_from_source(store, source, prefix + ".conv.block.2", storage_type, hidden, hidden, 3, true);
    out.norm3 = binding::norm_weight_from_source(store, source, prefix + ".norm3", hidden);
    out.mlp.fc1_weight = store.load_tensor(source, prefix + ".mlp.ff.0.0.weight", storage_type, {intermediate, hidden});
    out.mlp.fc1_bias = store.load_f32_tensor(source, prefix + ".mlp.ff.0.0.bias", {intermediate});
    out.mlp.fc2_weight = store.load_tensor(source, prefix + ".mlp.ff.2.weight", storage_type, {hidden, intermediate});
    out.mlp.fc2_bias = store.load_f32_tensor(source, prefix + ".mlp.ff.2.bias", {hidden});
    out.ada = binding::linear_from_source(store, source, prefix + ".adaLN_modulation.1", storage_type, 9 * hidden, hidden, true);
    return out;
}


std::shared_ptr<FireRedFlowWeights> load_flow_weights(
    const FireRedAudioAssets & assets,
    core::ExecutionContext & execution,
    size_t weight_context_bytes,
    assets::TensorStorageType storage_type) {
    auto weights = std::make_shared<FireRedFlowWeights>();
    weights->store = std::make_shared<core::BackendWeightStore>(
        execution.backend(),
        execution.backend_type(),
        "firered_audio.flow.weights",
        weight_context_bytes);
    const auto & c = assets.flow;
    const auto & source = *assets.model_weights;
    const int64_t in_channels = c.vae_channels + c.hidden_size;
    weights->backbone_input_proj = binding::linear_from_source(
        *weights->store,
        source,
        "dit.backbone_input_proj",
        storage_type,
        c.hidden_size,
        c.backbone_hidden_size,
        true);
    weights->dit_in = binding::linear_from_source(
        *weights->store,
        source,
        "dit.in_proj",
        storage_type,
        c.hidden_size,
        in_channels,
        true);
    weights->dit_time_fc1 = binding::linear_from_source(
        *weights->store, source, "dit.t_embedder.time_mlp.0", storage_type, c.hidden_size, 256, true);
    weights->dit_time_fc2 = binding::linear_from_source(
        *weights->store, source, "dit.t_embedder.time_mlp.2", storage_type, c.hidden_size, c.hidden_size, true);
    weights->dit_blocks.reserve(static_cast<size_t>(c.layers));
    for (int64_t layer = 0; layer < c.layers; ++layer) {
        weights->dit_blocks.push_back(load_dit_block(
            *weights->store,
            source,
            "dit.blocks." + std::to_string(layer),
            c.hidden_size,
            c.intermediate_size,
            storage_type));
    }
    weights->dit_final_ada = binding::linear_from_source(
        *weights->store, source, "dit.final_layer.adaLN_modulation.1", storage_type, 2 * c.hidden_size, c.hidden_size, true);
    weights->dit_final_linear = binding::linear_from_source(
        *weights->store, source, "dit.final_layer.linear", storage_type, c.vae_channels, c.hidden_size, true);
    weights->store->upload();
    return weights;
}

class DiTFlowGraph {
public:
    DiTFlowGraph(
        core::ExecutionContext & execution,
        std::shared_ptr<const FireRedFlowWeights> weights,
        FireRedAudioFlowConfig config,
        size_t graph_arena_bytes,
        int64_t input_channels)
        : execution_(execution),
          weights_(std::move(weights)),
          config_(config),
          graph_arena_bytes_(graph_arena_bytes),
          input_channels_(input_channels) {}

    ~DiTFlowGraph() {
        mem_.reset(execution_.backend());
    }

    std::vector<float> run(const std::vector<float> & x_in, const std::vector<float> & time_embed, int64_t batch) {
        const int64_t tokens = config_.history_patches * config_.patch_size + config_.patch_size;
        const int64_t in_channels = input_channels_;
        if (batch <= 0 || static_cast<int64_t>(x_in.size()) != batch * tokens * in_channels ||
            static_cast<int64_t>(time_embed.size()) != batch * 256) {
            throw std::runtime_error("FireRedAudio DiT input size mismatch");
        }
        ensure(batch);
        core::write_tensor_f32(core::wrap_tensor(input_, core::TensorShape::from_dims({batch, tokens, in_channels})), x_in);
        core::write_tensor_f32(core::wrap_tensor(time_, core::TensorShape::from_dims({batch, 256})), time_embed);
        if (core::compute_backend_graph(execution_.backend(), mem_.graph, nullptr, "firered_audio.dit") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("FireRedAudio DiT graph compute failed");
        }
        return core::read_tensor_f32(output_);
    }

    void release_graph() {
        mem_.reset(execution_.backend());
        batch_ = 0;
        input_ = nullptr;
        time_ = nullptr;
        positions_ = nullptr;
        output_ = nullptr;
    }

private:
    modules::ProjectedGroupedSelfAttentionConfig attention_config() const {
        modules::ProjectedGroupedSelfAttentionConfig cfg;
        cfg.hidden_size = config_.hidden_size;
        cfg.attention_heads = config_.heads;
        cfg.kv_heads = config_.heads;
        cfg.head_dim = config_.hidden_size / config_.heads;
        cfg.use_bias = true;
        cfg.use_rope = true;
        cfg.rope_type = GGML_ROPE_TYPE_NORMAL;
        cfg.rope_theta = 10000.0F;
        cfg.local_rope_theta = 10000.0F;
        cfg.causality = modules::AttentionCausality::NonCausal;
        cfg.lowering = modules::GroupedQueryAttentionLowering::FlashGroupedViewKV;
        cfg.attention_precision = GGML_PREC_F32;
        return cfg;
    }

    core::TensorValue dit_block(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & x,
        const core::TensorValue & c,
        const core::TensorValue & positions,
        const FireRedDiTBlockWeights & weights,
        int64_t layer) const {
        auto ada = modules::SiluModule{}.build(ctx, c);
        ada = modules::LinearModule({config_.hidden_size, 9 * config_.hidden_size, true}).build(ctx, ada, weights.ada);
        auto shift_msa = modules::SliceModule({2, 0 * config_.hidden_size, config_.hidden_size}).build(ctx, ada);
        auto scale_msa = modules::SliceModule({2, 1 * config_.hidden_size, config_.hidden_size}).build(ctx, ada);
        auto gate_msa = modules::SliceModule({2, 2 * config_.hidden_size, config_.hidden_size}).build(ctx, ada);
        auto shift_mlp = modules::SliceModule({2, 3 * config_.hidden_size, config_.hidden_size}).build(ctx, ada);
        auto scale_mlp = modules::SliceModule({2, 4 * config_.hidden_size, config_.hidden_size}).build(ctx, ada);
        auto gate_mlp = modules::SliceModule({2, 5 * config_.hidden_size, config_.hidden_size}).build(ctx, ada);
        auto shift_conv = modules::SliceModule({2, 6 * config_.hidden_size, config_.hidden_size}).build(ctx, ada);
        auto scale_conv = modules::SliceModule({2, 7 * config_.hidden_size, config_.hidden_size}).build(ctx, ada);
        auto gate_conv = modules::SliceModule({2, 8 * config_.hidden_size, config_.hidden_size}).build(ctx, ada);

        auto h = modules::RMSNormModule({config_.hidden_size, 1.0e-6F, true, false}).build(ctx, x, weights.norm1);
        h = modulate(ctx, h, shift_msa, scale_msa);
        h = modules::ProjectedGroupedSelfAttentionModule(attention_config()).build(ctx, h, positions, weights.attention, layer);
        h = mul_broadcast(ctx, h, gate_msa);
        auto out = core::wrap_tensor(ggml_add(ctx.ggml, x.tensor, h.tensor), x.shape, GGML_TYPE_F32);

        h = modules::RMSNormModule({config_.hidden_size, 1.0e-6F, true, false}).build(ctx, out, weights.norm2);
        h = modulate(ctx, h, shift_conv, scale_conv);
        h = modules::TransposeModule({{0, 2, 1, 3}, h.shape.rank}).build(ctx, h);
        h = modules::Conv1dModule({config_.hidden_size, config_.hidden_size, 3, 1, 1, 1, true}).build(ctx, h, weights.conv1);
        h = mish(ctx, h);
        h = modules::Conv1dModule({config_.hidden_size, config_.hidden_size, 3, 1, 1, 1, true}).build(ctx, h, weights.conv2);
        h = modules::TransposeModule({{0, 2, 1, 3}, h.shape.rank}).build(ctx, h);
        h = mul_broadcast(ctx, h, gate_conv);
        out = core::wrap_tensor(ggml_add(ctx.ggml, out.tensor, h.tensor), out.shape, GGML_TYPE_F32);

        h = modules::RMSNormModule({config_.hidden_size, 1.0e-6F, true, false}).build(ctx, out, weights.norm3);
        h = modulate(ctx, h, shift_mlp, scale_mlp);
        h = modules::FeedForwardModule({
            config_.hidden_size,
            config_.intermediate_size,
            true,
            modules::GeluApproximation::Tanh,
        }).build(ctx, h, weights.mlp);
        h = mul_broadcast(ctx, h, gate_mlp);
        return core::wrap_tensor(ggml_add(ctx.ggml, out.tensor, h.tensor), out.shape, GGML_TYPE_F32);
    }

    void ensure(int64_t batch) {
        if (mem_.graph != nullptr && batch_ == batch) {
            return;
        }
        mem_.reset(execution_.backend());
        const int64_t tokens = config_.history_patches * config_.patch_size + config_.patch_size;
        const int64_t in_channels = input_channels_;
        ggml_init_params params{graph_arena_bytes_, nullptr, true};
        mem_.ctx.reset(ggml_init(params));
        ggml_init_params input_params{16ull * 1024ull * 1024ull, nullptr, true};
        mem_.input_ctx.reset(ggml_init(input_params));
        core::ModuleBuildContext ctx{mem_.ctx.get(), "firered_audio.dit", execution_.backend_type()};
        core::ModuleBuildContext input_ctx{mem_.input_ctx.get(), "firered_audio.dit.inputs", execution_.backend_type()};
        auto x = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch, tokens, in_channels}));
        input_ = x.tensor;
        ggml_set_input(input_);
        auto t = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch, 256}));
        time_ = t.tensor;
        ggml_set_input(time_);
        auto positions = core::make_tensor(input_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({tokens}));
        positions_ = positions.tensor;

        auto vae = modules::SliceModule({2, 0, config_.vae_channels}).build(ctx, x);
        auto backbone = modules::SliceModule({2, config_.vae_channels, config_.backbone_hidden_size}).build(ctx, x);
        backbone = modules::LinearModule({config_.backbone_hidden_size, config_.hidden_size, true})
                       .build(ctx, backbone, weights_->backbone_input_proj);
        x = modules::ConcatModule({2}).build(ctx, vae, backbone);
        x = modules::LinearModule({config_.vae_channels + config_.hidden_size, config_.hidden_size, true}).build(ctx, x, weights_->dit_in);
        t = modules::LinearModule({256, config_.hidden_size, true}).build(ctx, t, weights_->dit_time_fc1);
        t = modules::SiluModule{}.build(ctx, t);
        t = modules::LinearModule({config_.hidden_size, config_.hidden_size, true}).build(ctx, t, weights_->dit_time_fc2);
        t = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, t), core::TensorShape::from_dims({batch, 1, config_.hidden_size}));
        for (size_t layer = 0; layer < weights_->dit_blocks.size(); ++layer) {
            x = dit_block(ctx, x, t, positions, weights_->dit_blocks[layer], static_cast<int64_t>(layer));
        }
        auto ada = modules::SiluModule{}.build(ctx, t);
        ada = modules::LinearModule({config_.hidden_size, 2 * config_.hidden_size, true}).build(ctx, ada, weights_->dit_final_ada);
        auto shift = modules::SliceModule({2, 0, config_.hidden_size}).build(ctx, ada);
        auto scale = modules::SliceModule({2, config_.hidden_size, config_.hidden_size}).build(ctx, ada);
        x = modules::LayerNormModule({config_.hidden_size, 1.0e-6F, false, false}).build(ctx, x, {});
        x = modulate(ctx, x, shift, scale);
        x = modules::LinearModule({config_.hidden_size, config_.vae_channels, true}).build(ctx, x, weights_->dit_final_linear);
        x = modules::SliceModule({1, config_.history_patches * config_.patch_size, config_.patch_size}).build(ctx, x);
        output_ = core::ensure_backend_addressable_layout(ctx, x).tensor;
        ggml_set_output(output_);
        mem_.graph = ggml_new_graph_custom(mem_.ctx.get(), 200000, false);
        ggml_build_forward_expand(mem_.graph, output_);
        mem_.input_buffer = ggml_backend_alloc_ctx_tensors(mem_.input_ctx.get(), execution_.backend());
        mem_.gallocr.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend())));
        if (mem_.input_buffer == nullptr || mem_.gallocr == nullptr ||
            !ggml_gallocr_reserve(mem_.gallocr.get(), mem_.graph) ||
            !ggml_gallocr_alloc_graph(mem_.gallocr.get(), mem_.graph)) {
            mem_.reset(execution_.backend());
            throw std::runtime_error("failed to allocate FireRedAudio DiT graph");
        }
        const auto pos = position_ids(tokens);
        ggml_backend_tensor_set(positions_, pos.data(), 0, pos.size() * sizeof(int32_t));
        batch_ = batch;
    }

    core::ExecutionContext & execution_;
    std::shared_ptr<const FireRedFlowWeights> weights_;
    FireRedAudioFlowConfig config_;
    size_t graph_arena_bytes_;
    int64_t input_channels_ = 0;
    GraphMemory mem_;
    int64_t batch_ = 0;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * time_ = nullptr;
    ggml_tensor * positions_ = nullptr;
    ggml_tensor * output_ = nullptr;
};


}  // namespace

std::vector<float> firered_cosine_time_schedule(int64_t steps) {
    if (steps <= 0) {
        throw std::runtime_error("FireRedAudio num_inference_steps must be positive");
    }
    std::vector<float> out(static_cast<size_t>(steps + 1));
    for (int64_t i = 0; i <= steps; ++i) {
        const float u = static_cast<float>(i) / static_cast<float>(steps);
        out[static_cast<size_t>(i)] = 1.0F - std::cos(u * 0.5F * kPi);
    }
    return out;
}

std::vector<float> firered_timestep_embedding(float timestep, int64_t dim) {
    if (dim % 2 != 0) {
        throw std::runtime_error("FireRedAudio timestep embedding dim must be even");
    }
    const int64_t half = dim / 2;
    const float step = std::log(10000.0F) / static_cast<float>(half - 1);
    std::vector<float> out(static_cast<size_t>(dim));
    for (int64_t i = 0; i < half; ++i) {
        const float freq = std::exp(static_cast<float>(i) * -step);
        const float arg = 1000.0F * timestep * freq;
        out[static_cast<size_t>(i)] = std::sin(arg);
        out[static_cast<size_t>(half + i)] = std::cos(arg);
    }
    return out;
}

class FireRedAudioFlowRuntime::Impl {
public:
    Impl(
        std::shared_ptr<const FireRedAudioAssets> assets,
        core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        size_t weight_context_bytes,
        assets::TensorStorageType storage_type)
        : assets_(std::move(assets)),
          execution_(execution),
          input_channels_(assets_->flow.vae_channels + assets_->flow.backbone_hidden_size),
          weights_(load_flow_weights(*assets_, execution_, weight_context_bytes, storage_type)),
          graph_(execution_, weights_, assets_->flow, graph_arena_bytes, input_channels_) {
        if (assets_ == nullptr) {
            throw std::runtime_error("FireRedAudio flow runtime requires assets");
        }
    }

    std::vector<float> run(const std::vector<float> & x_in, const std::vector<float> & time_embed, int64_t batch) {
        return graph_.run(x_in, time_embed, batch);
    }

    int64_t input_channels() const noexcept {
        return input_channels_;
    }

    void release_graph() {
        graph_.release_graph();
    }

private:
    std::shared_ptr<const FireRedAudioAssets> assets_;
    core::ExecutionContext & execution_;
    int64_t input_channels_ = 0;
    std::shared_ptr<FireRedFlowWeights> weights_;
    DiTFlowGraph graph_;
};

FireRedAudioFlowRuntime::FireRedAudioFlowRuntime(
    std::shared_ptr<const FireRedAudioAssets> assets,
    engine::core::ExecutionContext & execution,
    size_t graph_arena_bytes,
    size_t weight_context_bytes,
    engine::assets::TensorStorageType storage_type)
    : impl_(std::make_unique<Impl>(std::move(assets), execution, graph_arena_bytes, weight_context_bytes, storage_type)) {}

FireRedAudioFlowRuntime::~FireRedAudioFlowRuntime() = default;

std::vector<float> FireRedAudioFlowRuntime::run(const std::vector<float> & x_in, const std::vector<float> & time_embed, int64_t batch) {
    return impl_->run(x_in, time_embed, batch);
}

int64_t FireRedAudioFlowRuntime::input_channels() const noexcept {
    return impl_->input_channels();
}

void FireRedAudioFlowRuntime::release_graph() {
    impl_->release_graph();
}

}  // namespace engine::models::firered_audio
