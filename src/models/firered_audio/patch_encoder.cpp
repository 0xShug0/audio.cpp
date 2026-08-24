#include "engine/models/firered_audio/patch_encoder.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/feed_forward.h"
#include "engine/framework/modules/attention/projected_grouped_self_attention.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <ggml-alloc.h>
#include <ggml.h>

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

struct PatchBlockWeights {
    modules::NormWeights norm1;
    modules::ProjectedGroupedSelfAttentionWeights attention;
    modules::NormWeights norm2;
    modules::FeedForwardWeights mlp;
};

struct PatchEncoderWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    modules::LinearWeights in0;
    modules::LinearWeights in2;
    core::TensorValue cls;
    std::vector<PatchBlockWeights> blocks;
    modules::NormWeights out_norm;
    modules::LinearWeights out_linear;
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

PatchBlockWeights load_patch_block(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t hidden,
    int64_t intermediate,
    assets::TensorStorageType storage_type) {
    PatchBlockWeights out;
    out.norm1 = binding::norm_weight_from_source(store, source, prefix + ".norm1", hidden);
    out.attention = load_projected_attention(store, source, prefix + ".attn", hidden, storage_type);
    out.norm2 = binding::norm_weight_from_source(store, source, prefix + ".norm2", hidden);
    out.mlp.fc1_weight = store.load_tensor(source, prefix + ".mlp.ff.0.0.weight", storage_type, {intermediate, hidden});
    out.mlp.fc1_bias = store.load_f32_tensor(source, prefix + ".mlp.ff.0.0.bias", {intermediate});
    out.mlp.fc2_weight = store.load_tensor(source, prefix + ".mlp.ff.2.weight", storage_type, {hidden, intermediate});
    out.mlp.fc2_bias = store.load_f32_tensor(source, prefix + ".mlp.ff.2.bias", {hidden});
    return out;
}

std::shared_ptr<PatchEncoderWeights> load_patch_weights(
    const FireRedAudioAssets & assets,
    core::ExecutionContext & execution,
    size_t weight_context_bytes,
    assets::TensorStorageType storage_type) {
    auto weights = std::make_shared<PatchEncoderWeights>();
    weights->store = std::make_shared<core::BackendWeightStore>(
        execution.backend(),
        execution.backend_type(),
        "firered_audio.patch_encoder.weights",
        weight_context_bytes);
    const auto & c = assets.patch_encoder;
    const auto & source = *assets.model_weights;
    weights->in0 = binding::linear_from_source(
        *weights->store, source, "patch_encoder.in_proj.0", storage_type, c.hidden_size, c.vae_dim, true);
    weights->in2 = binding::linear_from_source(
        *weights->store, source, "patch_encoder.in_proj.2", storage_type, c.hidden_size, c.hidden_size, true);
    weights->cls = weights->store->load_f32_tensor(source, "patch_encoder.cls_tok", {1, 1, c.hidden_size});
    weights->blocks.reserve(static_cast<size_t>(c.layers));
    for (int64_t layer = 0; layer < c.layers; ++layer) {
        weights->blocks.push_back(load_patch_block(
            *weights->store,
            source,
            "patch_encoder.blocks." + std::to_string(layer),
            c.hidden_size,
            c.intermediate_size,
            storage_type));
    }
    weights->out_norm = binding::norm_weight_from_source(
        *weights->store, source, "patch_encoder.out_proj.norm_final", c.hidden_size);
    weights->out_linear = binding::linear_from_source(
        *weights->store, source, "patch_encoder.out_proj.linear", storage_type, c.out_dim, c.hidden_size, true);
    weights->store->upload();
    return weights;
}

class PatchEncoderGraph {
public:
    PatchEncoderGraph(
        core::ExecutionContext & execution,
        std::shared_ptr<const PatchEncoderWeights> weights,
        FireRedAudioPatchEncoderConfig config,
        size_t graph_arena_bytes)
        : execution_(execution),
          weights_(std::move(weights)),
          config_(config),
          graph_arena_bytes_(graph_arena_bytes) {}

    ~PatchEncoderGraph() {
        mem_.reset(execution_.backend());
    }

    std::vector<float> run(const std::vector<float> & latents) {
        if (latents.empty() || static_cast<int64_t>(latents.size()) % (config_.patch_size * config_.vae_dim) != 0) {
            throw std::runtime_error("FireRedAudio patch encoder input size mismatch");
        }
        const int64_t patches = static_cast<int64_t>(latents.size()) / (config_.patch_size * config_.vae_dim);
        ensure(patches);
        core::write_tensor_f32(
            core::wrap_tensor(input_, core::TensorShape::from_dims({patches, config_.patch_size, config_.vae_dim})),
            latents);
        if (core::compute_backend_graph(execution_.backend(), mem_.graph, nullptr, "firered_audio.patch_encoder") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("FireRedAudio patch encoder graph compute failed");
        }
        return core::read_tensor_f32(output_);
    }

    void release_graph() {
        mem_.reset(execution_.backend());
        patches_ = 0;
        input_ = nullptr;
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

    core::TensorValue build_block(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & x,
        const core::TensorValue & positions,
        const PatchBlockWeights & weights,
        int64_t layer) const {
        auto h = modules::RMSNormModule({config_.hidden_size, 1.0e-6F, true, false}).build(ctx, x, weights.norm1);
        h = modules::ProjectedGroupedSelfAttentionModule(attention_config()).build(ctx, h, positions, weights.attention, layer);
        auto out = core::wrap_tensor(ggml_add(ctx.ggml, x.tensor, h.tensor), x.shape, GGML_TYPE_F32);
        h = modules::RMSNormModule({config_.hidden_size, 1.0e-6F, true, false}).build(ctx, out, weights.norm2);
        h = modules::FeedForwardModule({
            config_.hidden_size,
            config_.intermediate_size,
            true,
            modules::GeluApproximation::Tanh,
        }).build(ctx, h, weights.mlp);
        return core::wrap_tensor(ggml_add(ctx.ggml, out.tensor, h.tensor), out.shape, GGML_TYPE_F32);
    }

    void ensure(int64_t patches) {
        if (mem_.graph != nullptr && patches_ == patches) {
            return;
        }
        mem_.reset(execution_.backend());
        ggml_init_params params{graph_arena_bytes_, nullptr, true};
        mem_.ctx.reset(ggml_init(params));
        ggml_init_params input_params{8ull * 1024ull * 1024ull, nullptr, true};
        mem_.input_ctx.reset(ggml_init(input_params));
        core::ModuleBuildContext ctx{mem_.ctx.get(), "firered_audio.patch_encoder", execution_.backend_type()};
        core::ModuleBuildContext input_ctx{mem_.input_ctx.get(), "firered_audio.patch_encoder.inputs", execution_.backend_type()};
        auto x = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({patches, config_.patch_size, config_.vae_dim}));
        input_ = x.tensor;
        ggml_set_input(input_);
        auto positions = core::make_tensor(input_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({config_.patch_size + 1}));
        positions_ = positions.tensor;
        x = modules::LinearModule({config_.vae_dim, config_.hidden_size, true}).build(ctx, x, weights_->in0);
        x = modules::GeluModule({modules::GeluApproximation::ExactErf}).build(ctx, x);
        x = modules::LinearModule({config_.hidden_size, config_.hidden_size, true}).build(ctx, x, weights_->in2);
        auto cls = modules::RepeatModule({core::TensorShape::from_dims({patches, 1, config_.hidden_size})}).build(ctx, weights_->cls);
        x = modules::ConcatModule({1}).build(ctx, cls, x);
        for (size_t layer = 0; layer < weights_->blocks.size(); ++layer) {
            x = build_block(ctx, x, positions, weights_->blocks[layer], static_cast<int64_t>(layer));
        }
        x = modules::RMSNormModule({config_.hidden_size, 1.0e-6F, true, false}).build(ctx, x, weights_->out_norm);
        x = modules::LinearModule({config_.hidden_size, config_.out_dim, true}).build(ctx, x, weights_->out_linear);
        x = modules::SliceModule({1, 0, 1}).build(ctx, x);
        x = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, x), core::TensorShape::from_dims({1, patches, config_.out_dim}));
        output_ = x.tensor;
        ggml_set_output(output_);
        mem_.graph = ggml_new_graph_custom(mem_.ctx.get(), 65536, false);
        ggml_build_forward_expand(mem_.graph, output_);
        mem_.input_buffer = ggml_backend_alloc_ctx_tensors(mem_.input_ctx.get(), execution_.backend());
        mem_.gallocr.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend())));
        if (mem_.input_buffer == nullptr || mem_.gallocr == nullptr ||
            !ggml_gallocr_reserve(mem_.gallocr.get(), mem_.graph) ||
            !ggml_gallocr_alloc_graph(mem_.gallocr.get(), mem_.graph)) {
            mem_.reset(execution_.backend());
            throw std::runtime_error("failed to allocate FireRedAudio patch encoder graph");
        }
        const auto pos = position_ids(config_.patch_size + 1);
        ggml_backend_tensor_set(positions_, pos.data(), 0, pos.size() * sizeof(int32_t));
        patches_ = patches;
    }

    core::ExecutionContext & execution_;
    std::shared_ptr<const PatchEncoderWeights> weights_;
    FireRedAudioPatchEncoderConfig config_;
    size_t graph_arena_bytes_;
    GraphMemory mem_;
    int64_t patches_ = 0;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * positions_ = nullptr;
    ggml_tensor * output_ = nullptr;
};

}  // namespace

class FireRedAudioPatchEncoderRuntime::Impl {
public:
    Impl(
        std::shared_ptr<const FireRedAudioAssets> assets,
        core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        size_t weight_context_bytes,
        assets::TensorStorageType storage_type)
        : assets_(std::move(assets)),
          weights_(load_patch_weights(*assets_, execution, weight_context_bytes, storage_type)),
          graph_(execution, weights_, assets_->patch_encoder, graph_arena_bytes) {
        if (assets_ == nullptr) {
            throw std::runtime_error("FireRedAudio patch encoder runtime requires assets");
        }
    }

    std::vector<float> encode(const std::vector<float> & latents) {
        return graph_.run(latents);
    }

    void release_graph() {
        graph_.release_graph();
    }

private:
    std::shared_ptr<const FireRedAudioAssets> assets_;
    std::shared_ptr<PatchEncoderWeights> weights_;
    PatchEncoderGraph graph_;
};

FireRedAudioPatchEncoderRuntime::FireRedAudioPatchEncoderRuntime(
    std::shared_ptr<const FireRedAudioAssets> assets,
    engine::core::ExecutionContext & execution,
    size_t graph_arena_bytes,
    size_t weight_context_bytes,
    engine::assets::TensorStorageType storage_type)
    : impl_(std::make_unique<Impl>(std::move(assets), execution, graph_arena_bytes, weight_context_bytes, storage_type)) {}

FireRedAudioPatchEncoderRuntime::~FireRedAudioPatchEncoderRuntime() = default;

std::vector<float> FireRedAudioPatchEncoderRuntime::encode(const std::vector<float> & latents) {
    return impl_->encode(latents);
}

void FireRedAudioPatchEncoderRuntime::release_graph() {
    impl_->release_graph();
}

}  // namespace engine::models::firered_audio
