#include "engine/models/fireredtts3/ar.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/attention/feed_forward.h"
#include "engine/framework/modules/attention/projected_grouped_self_attention.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <ggml-alloc.h>
#include <ggml.h>

#include <algorithm>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::fireredtts3 {
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

float sigmoid(float value) {
    if (value >= 0.0F) {
        const float z = std::exp(-value);
        return 1.0F / (1.0F + z);
    }
    const float z = std::exp(value);
    return z / (1.0F + z);
}

const modules::LinearWeights & require_linear_weight(
    const std::optional<modules::LinearWeights> & weight,
    const char * name) {
    if (!weight.has_value()) {
        throw std::runtime_error(std::string("FireRedTTS3 missing required linear weight: ") + name);
    }
    return *weight;
}

modules::QwenCausalDecodeRuntimeConfig qwen_runtime_config(
    const std::string & trace,
    int64_t hidden,
    int64_t intermediate,
    int64_t layers,
    int64_t heads,
    int64_t kv_heads,
    int64_t head_dim,
    modules::QwenCausalDecoderLogitsMode hidden_mode,
    size_t prefill_arena,
    size_t decode_arena,
    core::BackendType backend_type,
    bool bf16_autocast = false,
    int64_t sliding_window = 0) {
    modules::QwenCausalDecodeRuntimeConfig out;
    out.trace_name = trace;
    out.prefill_graph_arena_bytes = prefill_arena;
    out.decode_graph_arena_bytes = decode_arena;
    out.decoder.stack.hidden_size = hidden;
    out.decoder.stack.num_attention_heads = heads;
    out.decoder.stack.num_key_value_heads = kv_heads;
    out.decoder.stack.head_dim = head_dim;
    out.decoder.stack.intermediate_size = intermediate;
    out.decoder.stack.layers = layers;
    out.decoder.stack.rms_norm_eps = 1.0e-6F;
    out.decoder.stack.rope_theta = 1000000.0F;
    out.decoder.stack.rope_type = GGML_ROPE_TYPE_NEOX;
    out.decoder.stack.use_qk_norm = true;
    out.decoder.stack.attention_precision = GGML_PREC_F32;
    out.decoder.stack.projection_precision = GGML_PREC_DEFAULT;
    out.decoder.stack.runtime.attention.prefill_mode = modules::QwenDecoderAttentionMode::FlashGroupedViewKV;
    out.decoder.stack.runtime.attention.static_mode = modules::QwenDecoderAttentionMode::FlashGroupedViewKV;
    out.decoder.stack.runtime.static_cache.update_mode = modules::QwenDecoderStaticCacheUpdateMode::DirectSetRows;
    out.decoder.stack.runtime.static_cache.set_rows_mode = modules::QwenDecoderStaticCacheSetRowsMode::BackendViewOptimized;
    out.sliding_window = sliding_window;
    if (bf16_autocast && backend_type != core::BackendType::Cpu && backend_type != core::BackendType::Vulkan &&
        backend_type != core::BackendType::Metal) {
        out.decoder.stack.activation_cast.enabled = true;
        out.decoder.stack.activation_cast.type = GGML_TYPE_BF16;
        out.decoder.stack.activation_cast.after_input_norm = true;
        out.decoder.stack.activation_cast.after_qkv_projection = true;
        out.decoder.stack.activation_cast.after_qk_norm = true;
        out.decoder.stack.activation_cast.after_rope = true;
        out.decoder.stack.activation_cast.after_static_cache_update = true;
        out.decoder.stack.activation_cast.after_attention = true;
        out.decoder.stack.activation_cast.after_attention_output = true;
        out.decoder.stack.activation_cast.after_residual = true;
        out.decoder.stack.activation_cast.after_ffn_norm = true;
        out.decoder.stack.activation_cast.after_mlp_projection = true;
        out.decoder.stack.activation_cast.after_mlp_silu = true;
        out.decoder.stack.activation_cast.after_mlp_mul = true;
        out.decoder.stack.activation_cast.after_output = true;
        out.decoder.static_cache_type = GGML_TYPE_BF16;
    }
    out.decoder.logits_mode = hidden_mode;
    out.output_mode = modules::QwenCausalDecodeOutputMode::Hidden;
    out.return_hidden = true;
    if (bf16_autocast && backend_type != core::BackendType::Cpu && backend_type != core::BackendType::Vulkan &&
        backend_type != core::BackendType::Metal) {
        out.readback_round_type = GGML_TYPE_BF16;
    }
    return out;
}

modules::QwenDecoderLayerWeights load_qwen_layer(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    const modules::QwenCausalDecoderConfig & config,
    assets::TensorStorageType storage_type) {
    modules::QwenDecoderLayerWeights out;
    out.input_norm = binding::norm_weight_from_source(store, source, prefix + ".input_layernorm", config.stack.hidden_size);
    out.self_attention.q_weight = store.load_tensor(
        source,
        prefix + ".self_attn.q_proj.weight",
        storage_type,
        {config.stack.num_attention_heads * config.stack.head_dim, config.stack.hidden_size});
    out.self_attention.k_weight = store.load_tensor(
        source,
        prefix + ".self_attn.k_proj.weight",
        storage_type,
        {config.stack.num_key_value_heads * config.stack.head_dim, config.stack.hidden_size});
    out.self_attention.v_weight = store.load_tensor(
        source,
        prefix + ".self_attn.v_proj.weight",
        storage_type,
        {config.stack.num_key_value_heads * config.stack.head_dim, config.stack.hidden_size});
    out.self_attention.out_weight = store.load_tensor(
        source,
        prefix + ".self_attn.o_proj.weight",
        storage_type,
        {config.stack.hidden_size, config.stack.num_attention_heads * config.stack.head_dim});
    out.q_norm = binding::norm_weight_from_source(store, source, prefix + ".self_attn.q_norm", config.stack.head_dim);
    out.k_norm = binding::norm_weight_from_source(store, source, prefix + ".self_attn.k_norm", config.stack.head_dim);
    out.post_norm = binding::norm_weight_from_source(store, source, prefix + ".post_attention_layernorm", config.stack.hidden_size);
    out.mlp.gate_proj = binding::linear_from_source(
        store,
        source,
        prefix + ".mlp.gate_proj",
        storage_type,
        config.stack.intermediate_size,
        config.stack.hidden_size,
        false);
    out.mlp.up_proj = binding::linear_from_source(
        store,
        source,
        prefix + ".mlp.up_proj",
        storage_type,
        config.stack.intermediate_size,
        config.stack.hidden_size,
        false);
    out.mlp.down_proj = binding::linear_from_source(
        store,
        source,
        prefix + ".mlp.down_proj",
        storage_type,
        config.stack.hidden_size,
        config.stack.intermediate_size,
        false);
    return out;
}

modules::QwenCausalDecodeRuntimeWeights load_qwen_weights(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    const modules::QwenCausalDecodeRuntimeConfig & runtime_config,
    int64_t vocab_size,
    assets::TensorStorageType storage_type) {
    const auto & config = runtime_config.decoder;
    modules::QwenCausalDecodeRuntimeWeights out;
    out.token_embedding = store.load_tensor(
        source,
        prefix + ".embed_tokens.weight",
        storage_type,
        {vocab_size, config.stack.hidden_size});
    out.stack.layers.reserve(static_cast<size_t>(config.stack.layers));
    for (int64_t layer = 0; layer < config.stack.layers; ++layer) {
        out.stack.layers.push_back(load_qwen_layer(
            store,
            source,
            prefix + ".layers." + std::to_string(layer),
            config,
            storage_type));
    }
    out.final_norm = binding::norm_weight_from_source(store, source, prefix + ".norm", config.stack.hidden_size);
    return out;
}

struct FireRedAttentionBlockWeights {
    modules::NormWeights norm1;
    modules::ProjectedGroupedSelfAttentionWeights attention;
    modules::NormWeights norm2;
    modules::FeedForwardWeights mlp;
};

struct FireRedArWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    modules::QwenCausalDecodeRuntimeWeights backbone_qwen;
    std::optional<modules::LinearWeights> spk_proj_llm;
    std::optional<modules::LinearWeights> spk_proj_dit;
    modules::LinearWeights patch_in;
    core::TensorValue patch_cls;
    std::vector<FireRedAttentionBlockWeights> patch_blocks;
    modules::NormWeights patch_out_norm;
    modules::LinearWeights patch_out_linear;
    modules::LinearWeights dit_head;
    modules::LinearWeights stop_head;
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

FireRedAttentionBlockWeights load_patch_block(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t hidden,
    int64_t intermediate,
    assets::TensorStorageType storage_type) {
    FireRedAttentionBlockWeights out;
    out.norm1 = binding::norm_weight_from_source(store, source, prefix + ".norm1", hidden);
    out.attention = load_projected_attention(store, source, prefix + ".attn", hidden, storage_type);
    out.norm2 = binding::norm_weight_from_source(store, source, prefix + ".norm2", hidden);
    out.mlp.fc1_weight = store.load_tensor(source, prefix + ".mlp.ff.0.0.weight", storage_type, {intermediate, hidden});
    out.mlp.fc1_bias = store.load_f32_tensor(source, prefix + ".mlp.ff.0.0.bias", {intermediate});
    out.mlp.fc2_weight = store.load_tensor(source, prefix + ".mlp.ff.2.weight", storage_type, {hidden, intermediate});
    out.mlp.fc2_bias = store.load_f32_tensor(source, prefix + ".mlp.ff.2.bias", {hidden});
    return out;
}

std::shared_ptr<FireRedArWeights> load_base_ar_weights(
    const FireRedTTS3Assets & assets,
    core::ExecutionContext & execution,
    size_t weight_context_bytes,
    size_t graph_arena_bytes,
    assets::TensorStorageType storage_type) {
    auto weights = std::make_shared<FireRedArWeights>();
    weights->store = std::make_shared<core::BackendWeightStore>(
        execution.backend(),
        execution.backend_type(),
        "fireredtts3.base.weights",
        weight_context_bytes);
    const auto & c = assets.base;
    const auto & source = *assets.base_weights;
    const auto qwen_config = qwen_runtime_config(
        "fireredtts3.backbone",
        c.hidden_size,
        c.intermediate_size,
        c.layers,
        c.heads,
        c.kv_heads,
        c.head_dim,
        modules::QwenCausalDecoderLogitsMode::AllSteps,
        graph_arena_bytes,
        graph_arena_bytes,
        execution.backend_type());
    weights->backbone_qwen = load_qwen_weights(
        *weights->store,
        source,
        "backbone_llm",
        qwen_config,
        c.vocab_size,
        storage_type);
    weights->spk_proj_llm = binding::linear_from_source(
        *weights->store, source, "spk_proj_llm", storage_type, c.hidden_size, c.speaker_dim, true);
    weights->spk_proj_dit = binding::linear_from_source(
        *weights->store, source, "spk_proj_dit", storage_type, c.speaker_dim, c.speaker_dim, true);
    weights->patch_in = binding::linear_from_source(
        *weights->store, source, "patch_encoder.in_proj", storage_type, c.patch_hidden_size, c.redae_dim, true);
    weights->patch_cls = weights->store->load_f32_tensor(source, "patch_encoder.cls_tok", {1, 1, c.patch_hidden_size});
    weights->patch_blocks.reserve(static_cast<size_t>(c.patch_layers));
    for (int64_t layer = 0; layer < c.patch_layers; ++layer) {
        weights->patch_blocks.push_back(load_patch_block(
            *weights->store,
            source,
            "patch_encoder.blocks." + std::to_string(layer),
            c.patch_hidden_size,
            c.patch_intermediate_size,
            storage_type));
    }
    weights->patch_out_norm = binding::norm_weight_from_source(
        *weights->store, source, "patch_encoder.out_proj.norm_final", c.patch_hidden_size);
    weights->patch_out_linear = binding::linear_from_source(
        *weights->store, source, "patch_encoder.out_proj.linear", storage_type, c.hidden_size, c.patch_hidden_size, true);
    weights->dit_head = binding::linear_from_source(
        *weights->store, source, "dit_head", storage_type, c.dit_hidden_size, c.hidden_size, true);
    weights->stop_head = binding::linear_from_source(
        *weights->store, source, "stop_head", storage_type, 1, c.hidden_size, true);
    weights->store->upload();
    return weights;
}

std::shared_ptr<FireRedArWeights> load_instruct_ar_weights(
    const FireRedTTS3Assets & assets,
    core::ExecutionContext & execution,
    size_t weight_context_bytes,
    size_t graph_arena_bytes,
    assets::TensorStorageType storage_type) {
    auto weights = std::make_shared<FireRedArWeights>();
    weights->store = std::make_shared<core::BackendWeightStore>(
        execution.backend(),
        execution.backend_type(),
        "fireredtts3.instruct.weights",
        weight_context_bytes);
    const auto & c = assets.base;
    const auto & source = *assets.instruct_weights;
    const auto qwen_config = qwen_runtime_config(
        "fireredtts3.instruct.backbone",
        c.hidden_size,
        c.intermediate_size,
        c.layers,
        c.heads,
        c.kv_heads,
        c.head_dim,
        modules::QwenCausalDecoderLogitsMode::AllSteps,
        graph_arena_bytes,
        graph_arena_bytes,
        execution.backend_type());
    weights->backbone_qwen = load_qwen_weights(
        *weights->store,
        source,
        "backbone_llm.model",
        qwen_config,
        c.vocab_size,
        storage_type);
    weights->patch_in = binding::linear_from_source(
        *weights->store, source, "patch_encoder.in_proj", storage_type, c.patch_hidden_size, c.redae_dim, true);
    weights->patch_cls = weights->store->load_f32_tensor(source, "patch_encoder.cls_tok", {1, 1, c.patch_hidden_size});
    weights->patch_blocks.reserve(static_cast<size_t>(c.patch_layers));
    for (int64_t layer = 0; layer < c.patch_layers; ++layer) {
        weights->patch_blocks.push_back(load_patch_block(
            *weights->store,
            source,
            "patch_encoder.blocks." + std::to_string(layer),
            c.patch_hidden_size,
            c.patch_intermediate_size,
            storage_type));
    }
    weights->patch_out_norm = binding::norm_weight_from_source(
        *weights->store, source, "patch_encoder.out_proj.norm_final", c.patch_hidden_size);
    weights->patch_out_linear = binding::linear_from_source(
        *weights->store, source, "patch_encoder.out_proj.linear", storage_type, c.hidden_size, c.patch_hidden_size, true);
    weights->dit_head = binding::linear_from_source(
        *weights->store, source, "dit_head", storage_type, c.dit_hidden_size, c.hidden_size, true);
    weights->stop_head = binding::linear_from_source(
        *weights->store, source, "stop_head", storage_type, 1, c.hidden_size, true);
    weights->store->upload();
    return weights;
}

class LinearGraph {
public:
    LinearGraph(
        core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        std::string label,
        int64_t in_features,
        int64_t out_features,
        const modules::LinearWeights & weights)
        : execution_(execution),
          label_(std::move(label)),
          in_features_(in_features),
          out_features_(out_features),
          weights_(weights) {
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        mem_.ctx.reset(ggml_init(params));
        ggml_init_params input_params{8ull * 1024ull * 1024ull, nullptr, true};
        mem_.input_ctx.reset(ggml_init(input_params));
        if (mem_.ctx == nullptr || mem_.input_ctx == nullptr) {
            throw std::runtime_error("failed to initialize " + label_ + " graph context");
        }
    }

    ~LinearGraph() {
        mem_.reset(execution_.backend());
    }

    std::vector<float> run(const std::vector<float> & input, int64_t rows) {
        if (rows <= 0 || static_cast<int64_t>(input.size()) != rows * in_features_) {
            throw std::runtime_error(label_ + " input size mismatch");
        }
        ensure(rows);
        core::write_tensor_f32(core::wrap_tensor(input_, core::TensorShape::from_dims({1, rows, in_features_})), input);
        if (core::compute_backend_graph(execution_.backend(), mem_.graph, nullptr, label_.c_str()) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error(label_ + " graph compute failed");
        }
        return core::read_tensor_f32(output_);
    }

    void release_graph() {
        mem_.reset(execution_.backend());
        rows_ = 0;
        input_ = nullptr;
        output_ = nullptr;
    }

private:
    void ensure(int64_t rows) {
        if (mem_.graph != nullptr && rows_ == rows) {
            return;
        }
        mem_.reset(execution_.backend());
        ggml_init_params params{64ull * 1024ull * 1024ull, nullptr, true};
        mem_.ctx.reset(ggml_init(params));
        ggml_init_params input_params{8ull * 1024ull * 1024ull, nullptr, true};
        mem_.input_ctx.reset(ggml_init(input_params));
        core::ModuleBuildContext ctx{mem_.ctx.get(), label_.c_str(), execution_.backend_type()};
        core::ModuleBuildContext input_ctx{mem_.input_ctx.get(), label_.c_str(), execution_.backend_type()};
        auto input = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, rows, in_features_}));
        input_ = input.tensor;
        ggml_set_input(input_);
        auto output = modules::LinearModule({in_features_, out_features_, true}).build(ctx, input, weights_);
        output_ = core::ensure_backend_addressable_layout(ctx, output).tensor;
        ggml_set_output(output_);
        mem_.graph = ggml_new_graph_custom(mem_.ctx.get(), 8192, false);
        ggml_build_forward_expand(mem_.graph, output_);
        mem_.input_buffer = ggml_backend_alloc_ctx_tensors(mem_.input_ctx.get(), execution_.backend());
        mem_.gallocr.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend())));
        if (mem_.input_buffer == nullptr || mem_.gallocr == nullptr ||
            !ggml_gallocr_reserve(mem_.gallocr.get(), mem_.graph) ||
            !ggml_gallocr_alloc_graph(mem_.gallocr.get(), mem_.graph)) {
            mem_.reset(execution_.backend());
            throw std::runtime_error("failed to allocate " + label_ + " graph");
        }
        rows_ = rows;
    }

    core::ExecutionContext & execution_;
    GraphMemory mem_;
    std::string label_;
    int64_t in_features_ = 0;
    int64_t out_features_ = 0;
    modules::LinearWeights weights_;
    int64_t rows_ = 0;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * output_ = nullptr;
};

class PatchEncoderGraph {
public:
    PatchEncoderGraph(
        core::ExecutionContext & execution,
        std::shared_ptr<const FireRedArWeights> weights,
        FireRedTTS3BaseConfig config,
        size_t graph_arena_bytes)
        : execution_(execution),
          weights_(std::move(weights)),
          config_(config),
          graph_arena_bytes_(graph_arena_bytes) {}

    ~PatchEncoderGraph() {
        mem_.reset(execution_.backend());
    }

    std::vector<float> run(const std::vector<float> & latents) {
        if (latents.empty() || static_cast<int64_t>(latents.size()) % (config_.patch_size * config_.redae_dim) != 0) {
            throw std::runtime_error("FireRedTTS3 patch encoder input size mismatch");
        }
        const int64_t patches = static_cast<int64_t>(latents.size()) / (config_.patch_size * config_.redae_dim);
        ensure(patches);
        core::write_tensor_f32(core::wrap_tensor(input_, core::TensorShape::from_dims({patches, config_.patch_size, config_.redae_dim})), latents);
        if (core::compute_backend_graph(execution_.backend(), mem_.graph, nullptr, "fireredtts3.patch_encoder") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("FireRedTTS3 patch encoder graph compute failed");
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
        cfg.hidden_size = config_.patch_hidden_size;
        cfg.attention_heads = config_.patch_heads;
        cfg.kv_heads = config_.patch_heads;
        cfg.head_dim = config_.patch_hidden_size / config_.patch_heads;
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
        const FireRedAttentionBlockWeights & weights,
        int64_t layer) const {
        auto h = modules::RMSNormModule({config_.patch_hidden_size, 1.0e-6F, true, false}).build(ctx, x, weights.norm1);
        h = modules::ProjectedGroupedSelfAttentionModule(attention_config()).build(ctx, h, positions, weights.attention, layer);
        auto out = core::wrap_tensor(ggml_add(ctx.ggml, x.tensor, h.tensor), x.shape, GGML_TYPE_F32);
        h = modules::RMSNormModule({config_.patch_hidden_size, 1.0e-6F, true, false}).build(ctx, out, weights.norm2);
        h = modules::FeedForwardModule({
            config_.patch_hidden_size,
            config_.patch_intermediate_size,
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
        core::ModuleBuildContext ctx{mem_.ctx.get(), "fireredtts3.patch_encoder", execution_.backend_type()};
        core::ModuleBuildContext input_ctx{mem_.input_ctx.get(), "fireredtts3.patch_encoder.inputs", execution_.backend_type()};
        auto x = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({patches, config_.patch_size, config_.redae_dim}));
        input_ = x.tensor;
        ggml_set_input(input_);
        auto positions = core::make_tensor(input_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({config_.patch_size + 1}));
        positions_ = positions.tensor;

        x = modules::LinearModule({config_.redae_dim, config_.patch_hidden_size, true}).build(ctx, x, weights_->patch_in);
        auto cls = modules::RepeatModule({core::TensorShape::from_dims({patches, 1, config_.patch_hidden_size})}).build(ctx, weights_->patch_cls);
        x = modules::ConcatModule({1}).build(ctx, cls, x);
        for (size_t layer = 0; layer < weights_->patch_blocks.size(); ++layer) {
            x = build_block(ctx, x, positions, weights_->patch_blocks[layer], static_cast<int64_t>(layer));
        }
        x = modules::RMSNormModule({config_.patch_hidden_size, 1.0e-6F, true, false}).build(ctx, x, weights_->patch_out_norm);
        x = modules::LinearModule({config_.patch_hidden_size, config_.hidden_size, true}).build(ctx, x, weights_->patch_out_linear);
        x = modules::SliceModule({1, 0, 1}).build(ctx, x);
        x = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, x), core::TensorShape::from_dims({1, patches, config_.hidden_size}));
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
            throw std::runtime_error("failed to allocate FireRedTTS3 patch encoder graph");
        }
        const auto pos = position_ids(config_.patch_size + 1);
        ggml_backend_tensor_set(positions_, pos.data(), 0, pos.size() * sizeof(int32_t));
        patches_ = patches;
    }

    core::ExecutionContext & execution_;
    std::shared_ptr<const FireRedArWeights> weights_;
    FireRedTTS3BaseConfig config_;
    size_t graph_arena_bytes_;
    GraphMemory mem_;
    int64_t patches_ = 0;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * positions_ = nullptr;
    ggml_tensor * output_ = nullptr;
};

class StopHeadGraph {
public:
    StopHeadGraph(
        core::ExecutionContext & execution,
        std::shared_ptr<const FireRedArWeights> weights,
        FireRedTTS3BaseConfig config,
        size_t graph_arena_bytes)
        : execution_(execution),
          weights_(std::move(weights)),
          config_(config),
          graph_arena_bytes_(graph_arena_bytes) {}

    ~StopHeadGraph() {
        mem_.reset(execution_.backend());
    }

    float run(const std::vector<float> & hidden) {
        if (static_cast<int64_t>(hidden.size()) != config_.hidden_size) {
            throw std::runtime_error("FireRedTTS3 stop hidden size mismatch");
        }
        ensure();
        core::write_tensor_f32(core::wrap_tensor(input_, core::TensorShape::from_dims({1, 1, config_.hidden_size})), hidden);
        if (core::compute_backend_graph(execution_.backend(), mem_.graph, nullptr, "fireredtts3.stop_head") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("FireRedTTS3 stop head graph compute failed");
        }
        const auto logits = core::read_tensor_f32(output_);
        if (logits.size() != 1) {
            throw std::runtime_error("FireRedTTS3 stop head output size mismatch");
        }
        return sigmoid(logits[0]);
    }

    void release_graph() {
        mem_.reset(execution_.backend());
        input_ = nullptr;
        output_ = nullptr;
    }

private:
    void ensure() {
        if (mem_.graph != nullptr) {
            return;
        }
        ggml_init_params params{graph_arena_bytes_, nullptr, true};
        mem_.ctx.reset(ggml_init(params));
        ggml_init_params input_params{8ull * 1024ull * 1024ull, nullptr, true};
        mem_.input_ctx.reset(ggml_init(input_params));
        core::ModuleBuildContext ctx{mem_.ctx.get(), "fireredtts3.stop_head", execution_.backend_type()};
        core::ModuleBuildContext input_ctx{mem_.input_ctx.get(), "fireredtts3.stop_head.inputs", execution_.backend_type()};
        auto x = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 1, config_.hidden_size}));
        input_ = x.tensor;
        ggml_set_input(input_);
        x = modules::LinearModule({config_.hidden_size, 1, true}).build(ctx, x, weights_->stop_head);
        output_ = core::ensure_backend_addressable_layout(ctx, x).tensor;
        ggml_set_output(output_);
        mem_.graph = ggml_new_graph_custom(mem_.ctx.get(), 4096, false);
        ggml_build_forward_expand(mem_.graph, output_);
        mem_.input_buffer = ggml_backend_alloc_ctx_tensors(mem_.input_ctx.get(), execution_.backend());
        mem_.gallocr.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend())));
        if (mem_.input_buffer == nullptr || mem_.gallocr == nullptr ||
            !ggml_gallocr_reserve(mem_.gallocr.get(), mem_.graph) ||
            !ggml_gallocr_alloc_graph(mem_.gallocr.get(), mem_.graph)) {
            mem_.reset(execution_.backend());
            throw std::runtime_error("failed to allocate FireRedTTS3 stop head graph");
        }
    }

    core::ExecutionContext & execution_;
    std::shared_ptr<const FireRedArWeights> weights_;
    FireRedTTS3BaseConfig config_;
    size_t graph_arena_bytes_;
    GraphMemory mem_;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * output_ = nullptr;
};

class DiTHeadGraph {
public:
    DiTHeadGraph(
        core::ExecutionContext & execution,
        std::shared_ptr<const FireRedArWeights> weights,
        FireRedTTS3BaseConfig config,
        size_t graph_arena_bytes)
        : graph_(execution, graph_arena_bytes, "fireredtts3.dit_head", config.hidden_size, config.dit_hidden_size, weights->dit_head),
          config_(config) {}

    std::vector<float> run(const std::vector<float> & hidden, int64_t rows) {
        return graph_.run(hidden, rows);
    }

    void release_graph() {
        graph_.release_graph();
    }

private:
    LinearGraph graph_;
    FireRedTTS3BaseConfig config_;
};

class SpeakerProjectionGraph {
public:
    SpeakerProjectionGraph(
        core::ExecutionContext & execution,
        std::shared_ptr<const FireRedArWeights> weights,
        FireRedTTS3BaseConfig config,
        size_t graph_arena_bytes)
        : llm_(execution, graph_arena_bytes, "fireredtts3.spk_proj_llm", config.speaker_dim, config.hidden_size, require_linear_weight(weights->spk_proj_llm, "spk_proj_llm")),
          dit_(execution, graph_arena_bytes, "fireredtts3.spk_proj_dit", config.speaker_dim, config.speaker_dim, require_linear_weight(weights->spk_proj_dit, "spk_proj_dit")) {}

    std::vector<float> llm(const std::vector<float> & speaker) {
        return llm_.run(speaker, 1);
    }

    std::vector<float> dit(const std::vector<float> & speaker) {
        return dit_.run(speaker, 1);
    }

    void release_graphs() {
        llm_.release_graph();
        dit_.release_graph();
    }

private:
    LinearGraph llm_;
    LinearGraph dit_;
};

class TokenEmbeddingGraph {
public:
    TokenEmbeddingGraph(
        core::ExecutionContext & execution,
        std::shared_ptr<const FireRedArWeights> weights,
        FireRedTTS3BaseConfig config,
        size_t graph_arena_bytes)
        : execution_(execution),
          weights_(std::move(weights)),
          config_(config),
          graph_arena_bytes_(graph_arena_bytes) {}

    ~TokenEmbeddingGraph() {
        mem_.reset(execution_.backend());
    }

    std::vector<float> run(const std::vector<int32_t> & token_ids) {
        if (token_ids.empty()) {
            throw std::runtime_error("FireRedTTS3 token embedding requires tokens");
        }
        const int64_t steps = static_cast<int64_t>(token_ids.size());
        ensure(steps);
        ggml_backend_tensor_set(input_, token_ids.data(), 0, token_ids.size() * sizeof(int32_t));
        if (core::compute_backend_graph(execution_.backend(), mem_.graph, nullptr, "fireredtts3.token_embedding") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("FireRedTTS3 token embedding graph compute failed");
        }
        return core::read_tensor_f32(output_);
    }

    void release_graph() {
        mem_.reset(execution_.backend());
        steps_ = 0;
        input_ = nullptr;
        output_ = nullptr;
    }

private:
    void ensure(int64_t steps) {
        if (mem_.graph != nullptr && steps_ == steps) {
            return;
        }
        mem_.reset(execution_.backend());
        ggml_init_params params{graph_arena_bytes_, nullptr, true};
        mem_.ctx.reset(ggml_init(params));
        ggml_init_params input_params{8ull * 1024ull * 1024ull, nullptr, true};
        mem_.input_ctx.reset(ggml_init(input_params));
        core::ModuleBuildContext ctx{mem_.ctx.get(), "fireredtts3.token_embedding", execution_.backend_type()};
        core::ModuleBuildContext input_ctx{mem_.input_ctx.get(), "fireredtts3.token_embedding.inputs", execution_.backend_type()};
        auto ids = core::make_tensor(input_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({steps}));
        input_ = ids.tensor;
        ggml_set_input(input_);
        auto x = modules::EmbeddingModule({config_.vocab_size, config_.hidden_size}).build(ctx, ids, weights_->backbone_qwen.token_embedding);
        output_ = core::ensure_backend_addressable_layout(ctx, x).tensor;
        ggml_set_output(output_);
        mem_.graph = ggml_new_graph_custom(mem_.ctx.get(), 4096, false);
        ggml_build_forward_expand(mem_.graph, output_);
        mem_.input_buffer = ggml_backend_alloc_ctx_tensors(mem_.input_ctx.get(), execution_.backend());
        mem_.gallocr.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend())));
        if (mem_.input_buffer == nullptr || mem_.gallocr == nullptr ||
            !ggml_gallocr_reserve(mem_.gallocr.get(), mem_.graph) ||
            !ggml_gallocr_alloc_graph(mem_.gallocr.get(), mem_.graph)) {
            mem_.reset(execution_.backend());
            throw std::runtime_error("failed to allocate FireRedTTS3 token embedding graph");
        }
        steps_ = steps;
    }

    core::ExecutionContext & execution_;
    std::shared_ptr<const FireRedArWeights> weights_;
    FireRedTTS3BaseConfig config_;
    size_t graph_arena_bytes_;
    GraphMemory mem_;
    int64_t steps_ = 0;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * output_ = nullptr;
};

class TiedLmHeadGraph {
public:
    TiedLmHeadGraph(
        core::ExecutionContext & execution,
        std::shared_ptr<const FireRedArWeights> weights,
        FireRedTTS3BaseConfig config,
        size_t graph_arena_bytes)
        : execution_(execution),
          weights_(std::move(weights)),
          config_(config),
          graph_arena_bytes_(graph_arena_bytes) {}

    ~TiedLmHeadGraph() {
        mem_.reset(execution_.backend());
    }

    std::vector<float> run(const std::vector<float> & hidden) {
        if (static_cast<int64_t>(hidden.size()) != config_.hidden_size) {
            throw std::runtime_error("FireRedTTS3 LM head hidden size mismatch");
        }
        ensure();
        core::write_tensor_f32(core::wrap_tensor(input_, core::TensorShape::from_dims({1, 1, config_.hidden_size})), hidden);
        if (core::compute_backend_graph(execution_.backend(), mem_.graph, nullptr, "fireredtts3.lm_head") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("FireRedTTS3 LM head graph compute failed");
        }
        return core::read_tensor_f32(output_);
    }

    void release_graph() {
        mem_.reset(execution_.backend());
        input_ = nullptr;
        output_ = nullptr;
    }

private:
    void ensure() {
        if (mem_.graph != nullptr) {
            return;
        }
        ggml_init_params params{graph_arena_bytes_, nullptr, true};
        mem_.ctx.reset(ggml_init(params));
        ggml_init_params input_params{8ull * 1024ull * 1024ull, nullptr, true};
        mem_.input_ctx.reset(ggml_init(input_params));
        core::ModuleBuildContext ctx{mem_.ctx.get(), "fireredtts3.lm_head", execution_.backend_type()};
        core::ModuleBuildContext input_ctx{mem_.input_ctx.get(), "fireredtts3.lm_head.inputs", execution_.backend_type()};
        auto x = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 1, config_.hidden_size}));
        input_ = x.tensor;
        ggml_set_input(input_);
        x = modules::LinearModule({config_.hidden_size, config_.vocab_size, false}).build(
            ctx,
            x,
            {weights_->backbone_qwen.token_embedding, std::nullopt});
        output_ = core::ensure_backend_addressable_layout(ctx, x).tensor;
        ggml_set_output(output_);
        mem_.graph = ggml_new_graph_custom(mem_.ctx.get(), 4096, false);
        ggml_build_forward_expand(mem_.graph, output_);
        mem_.input_buffer = ggml_backend_alloc_ctx_tensors(mem_.input_ctx.get(), execution_.backend());
        mem_.gallocr.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend())));
        if (mem_.input_buffer == nullptr || mem_.gallocr == nullptr ||
            !ggml_gallocr_reserve(mem_.gallocr.get(), mem_.graph) ||
            !ggml_gallocr_alloc_graph(mem_.gallocr.get(), mem_.graph)) {
            mem_.reset(execution_.backend());
            throw std::runtime_error("failed to allocate FireRedTTS3 LM head graph");
        }
    }

    core::ExecutionContext & execution_;
    std::shared_ptr<const FireRedArWeights> weights_;
    FireRedTTS3BaseConfig config_;
    size_t graph_arena_bytes_;
    GraphMemory mem_;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * output_ = nullptr;
};

}  // namespace

class FireRedArRuntime::Impl {
public:
    Impl(
        std::shared_ptr<const FireRedTTS3Assets> assets,
        core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        size_t helper_graph_arena_bytes,
        size_t weight_context_bytes,
        assets::TensorStorageType storage_type,
        bool instruct)
        : assets_(std::move(assets)),
          execution_(execution),
          instruct_(instruct) {
        if (assets_ == nullptr) {
            throw std::runtime_error("FireRedTTS3 AR runtime requires assets");
        }
        weights_ = instruct_
            ? load_instruct_ar_weights(*assets_, execution_, weight_context_bytes, graph_arena_bytes, storage_type)
            : load_base_ar_weights(*assets_, execution_, weight_context_bytes, graph_arena_bytes, storage_type);
        token_embedding_ = std::make_unique<TokenEmbeddingGraph>(execution_, weights_, assets_->base, helper_graph_arena_bytes);
        patch_encoder_ = std::make_unique<PatchEncoderGraph>(execution_, weights_, assets_->base, graph_arena_bytes);
        dit_head_ = std::make_unique<DiTHeadGraph>(execution_, weights_, assets_->base, helper_graph_arena_bytes);
        stop_head_ = std::make_unique<StopHeadGraph>(execution_, weights_, assets_->base, helper_graph_arena_bytes);
        if (instruct_) {
            text_lm_head_ = std::make_unique<TiedLmHeadGraph>(execution_, weights_, assets_->base, graph_arena_bytes);
        } else {
            speaker_proj_ = std::make_unique<SpeakerProjectionGraph>(execution_, weights_, assets_->base, helper_graph_arena_bytes);
        }
        const auto backbone_config = qwen_runtime_config(
            instruct_ ? "fireredtts3.instruct.backbone" : "fireredtts3.backbone",
            assets_->base.hidden_size,
            assets_->base.intermediate_size,
            assets_->base.layers,
            assets_->base.heads,
            assets_->base.kv_heads,
            assets_->base.head_dim,
            modules::QwenCausalDecoderLogitsMode::AllSteps,
            graph_arena_bytes,
            graph_arena_bytes,
            execution_.backend_type(),
            true,
            0);
        backbone_ = std::make_unique<modules::QwenCausalDecodeRuntime>(execution_, backbone_config, weights_->backbone_qwen);
    }

    std::vector<float> token_embedding(const std::vector<int32_t> & token_ids) {
        return token_embedding_->run(token_ids);
    }

    std::vector<float> speaker_llm(const std::vector<float> & speaker) {
        if (!speaker_proj_) {
            throw std::runtime_error("FireRedTTS3 speaker LLM projection is unavailable for this variant");
        }
        return speaker_proj_->llm(speaker);
    }

    std::vector<float> speaker_dit(const std::vector<float> & speaker) {
        if (!speaker_proj_) {
            throw std::runtime_error("FireRedTTS3 speaker DiT projection is unavailable for this variant");
        }
        return speaker_proj_->dit(speaker);
    }

    std::vector<float> patch_encode(const std::vector<float> & latents) {
        return patch_encoder_->run(latents);
    }

    std::vector<float> dit_head(const std::vector<float> & hidden, int64_t rows) {
        return dit_head_->run(hidden, rows);
    }

    float stop(const std::vector<float> & hidden) {
        return stop_head_->run(hidden);
    }

    std::vector<float> text_logits(const std::vector<float> & hidden) {
        if (!text_lm_head_) {
            throw std::runtime_error("FireRedTTS3 text LM head is unavailable for this variant");
        }
        return text_lm_head_->run(hidden);
    }

    modules::QwenCausalPrefillResult prefill_embeddings(const std::vector<float> & embeddings, int64_t steps) {
        return backbone_->prefill_embeddings(embeddings, steps);
    }

    void start_decode_embeddings(const runtime::TransformerKVState & state, int64_t required_cache_steps) {
        backbone_->start_decode_embeddings(state, required_cache_steps);
    }

    modules::QwenCausalDecodeStepResult decode_embedding(const std::vector<float> & embedding) {
        return backbone_->decode_embedding(embedding);
    }

    void release_graphs() {
        if (token_embedding_) {
            token_embedding_->release_graph();
        }
        if (speaker_proj_) {
            speaker_proj_->release_graphs();
        }
        if (text_lm_head_) {
            text_lm_head_->release_graph();
        }
        if (patch_encoder_) {
            patch_encoder_->release_graph();
        }
        if (dit_head_) {
            dit_head_->release_graph();
        }
        if (stop_head_) {
            stop_head_->release_graph();
        }
        release_backbone_graphs();
    }

    void release_backbone_graphs() {
        if (backbone_) {
            backbone_->release_runtime_graphs();
        }
    }

private:
    std::shared_ptr<const FireRedTTS3Assets> assets_;
    core::ExecutionContext & execution_;
    bool instruct_ = false;
    std::shared_ptr<FireRedArWeights> weights_;
    std::unique_ptr<TokenEmbeddingGraph> token_embedding_;
    std::unique_ptr<SpeakerProjectionGraph> speaker_proj_;
    std::unique_ptr<TiedLmHeadGraph> text_lm_head_;
    std::unique_ptr<PatchEncoderGraph> patch_encoder_;
    std::unique_ptr<DiTHeadGraph> dit_head_;
    std::unique_ptr<StopHeadGraph> stop_head_;
    std::unique_ptr<modules::QwenCausalDecodeRuntime> backbone_;
};

FireRedArRuntime::FireRedArRuntime(
    std::shared_ptr<const FireRedTTS3Assets> assets,
    engine::core::ExecutionContext & execution,
    size_t graph_arena_bytes,
    size_t helper_graph_arena_bytes,
    size_t weight_context_bytes,
    engine::assets::TensorStorageType storage_type,
    bool instruct)
    : impl_(std::make_unique<Impl>(
          std::move(assets),
          execution,
          graph_arena_bytes,
          helper_graph_arena_bytes,
          weight_context_bytes,
          storage_type,
          instruct)) {}

FireRedArRuntime::~FireRedArRuntime() = default;

std::vector<float> FireRedArRuntime::token_embedding(const std::vector<int32_t> & token_ids) {
    return impl_->token_embedding(token_ids);
}

std::vector<float> FireRedArRuntime::speaker_llm(const std::vector<float> & speaker) {
    return impl_->speaker_llm(speaker);
}

std::vector<float> FireRedArRuntime::speaker_dit(const std::vector<float> & speaker) {
    return impl_->speaker_dit(speaker);
}

std::vector<float> FireRedArRuntime::patch_encode(const std::vector<float> & latents) {
    return impl_->patch_encode(latents);
}

std::vector<float> FireRedArRuntime::dit_head(const std::vector<float> & hidden, int64_t rows) {
    return impl_->dit_head(hidden, rows);
}

float FireRedArRuntime::stop(const std::vector<float> & hidden) {
    return impl_->stop(hidden);
}

std::vector<float> FireRedArRuntime::text_logits(const std::vector<float> & hidden) {
    return impl_->text_logits(hidden);
}

engine::modules::QwenCausalPrefillResult FireRedArRuntime::prefill_embeddings(
    const std::vector<float> & embeddings,
    int64_t steps) {
    return impl_->prefill_embeddings(embeddings, steps);
}

void FireRedArRuntime::start_decode_embeddings(
    const engine::runtime::TransformerKVState & state,
    int64_t required_cache_steps) {
    impl_->start_decode_embeddings(state, required_cache_steps);
}

engine::modules::QwenCausalDecodeStepResult FireRedArRuntime::decode_embedding(const std::vector<float> & embedding) {
    return impl_->decode_embedding(embedding);
}

void FireRedArRuntime::release_graphs() {
    impl_->release_graphs();
}

void FireRedArRuntime::release_backbone_graphs() {
    impl_->release_backbone_graphs();
}

}  // namespace engine::models::fireredtts3
