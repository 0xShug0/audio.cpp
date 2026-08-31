#include "engine/community_models/audio8_tts/ar.h"

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/transformers/qwen_causal_decoder.h"
#include "engine/framework/modules/transformers/qwen_decoder.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"
#include "engine/framework/sampling/torch_random.h"

#include "engine/framework/core/constant_tensor_cache.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::audio8_tts {
namespace {

namespace binding = engine::modules::binding;
using Clock = std::chrono::steady_clock;

constexpr int64_t kRasWindow = 10;
constexpr float kRasHighTemperature = 1.0F;
constexpr float kRasHighTopP = 0.9F;

struct ArkttsARProfile {
    double graph_build_prefill_ms = 0.0;
    double graph_build_step_ms = 0.0;
    double graph_build_fast_ms = 0.0;
    double slow_embedding_ms = 0.0;
    double fast_embedding_ms = 0.0;
    double prefill_input_upload_ms = 0.0;
    double prefill_graph_ms = 0.0;
    double prefill_output_read_ms = 0.0;
    double step_input_upload_ms = 0.0;
    double step_mask_upload_ms = 0.0;
    double step_graph_ms = 0.0;
    double step_output_read_ms = 0.0;
    double fast_input_upload_ms = 0.0;
    double fast_mask_upload_ms = 0.0;
    double fast_graph_ms = 0.0;
    double fast_output_read_ms = 0.0;
    double sample_bias_ms = 0.0;
    double sample_main_ms = 0.0;
    double sample_high_ms = 0.0;
    double sample_fast_ms = 0.0;
    int64_t prefill_runs = 0;
    int64_t step_runs = 0;
    int64_t fast_runs = 0;
    int64_t generated_frames = 0;
};

struct SampleCandidate {
    int32_t index = 0;
    float probability = 0.0F;
};

struct SampleDistribution {
    size_t source_size = 0;
    std::vector<SampleCandidate> candidates;
};

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct ArkttsLayerWeights {
    assets::TensorDataF32 input_norm;
    core::TensorValue qkv_proj;
    std::optional<core::TensorValue> qkv_bias;
    core::TensorValue o_proj;
    std::optional<assets::TensorDataF32> q_norm;
    std::optional<assets::TensorDataF32> k_norm;
    assets::TensorDataF32 post_norm;
    core::TensorValue gate_up_proj;
    core::TensorValue down_proj;
};

struct FalconH1LayerWeights {
    // Mirrors ../SenseVoice/runtime/llama.cpp/build/_deps/llama-src/src/models/falcon-h1.cpp
    // and /workspace/models/Audio8-TTS-Preview-0.1b/modeling_arktts.py FalconH1DecoderLayer
    assets::TensorDataF32 input_layernorm;          // slow.layers.*.input_layernorm.weight [512]
    core::TensorValue ssm_in;                        // slow.layers.*.mamba.in_proj.weight [1688,512]
    core::TensorValue ssm_conv1d;                    // slow.layers.*.mamba.conv1d.weight [896,1,4] -> [4,896] after convert
    assets::TensorDataF32 ssm_conv1d_b;              // slow.layers.*.mamba.conv1d.bias [896]
    core::TensorValue ssm_dt_b;                      // slow.layers.*.mamba.dt_bias [24]
    core::TensorValue ssm_A;                         // slow.layers.*.mamba.A_log [24] -> [1,24]
    core::TensorValue ssm_D;                         // slow.layers.*.mamba.D [24] -> [1,24]
    core::TensorValue ssm_out;                       // slow.layers.*.mamba.out_proj.weight [512,768]
    core::TensorValue attn_q_proj;                   // slow.layers.*.self_attn.q_proj.weight [512,512]
    core::TensorValue attn_k_proj;                   // slow.layers.*.self_attn.k_proj.weight [128,512]
    core::TensorValue attn_v_proj;                   // slow.layers.*.self_attn.v_proj.weight [128,512]
    core::TensorValue attn_o_proj;                   // slow.layers.*.self_attn.o_proj.weight [512,512]
    assets::TensorDataF32 pre_ff_layernorm;          // slow.layers.*.pre_ff_layernorm.weight [512]
    core::TensorValue ffn_gate;                      // slow.layers.*.feed_forward.gate_proj.weight [768,512]
    core::TensorValue ffn_up;                        // slow.layers.*.feed_forward.up_proj.weight [768,512]
    core::TensorValue ffn_down;                      // slow.layers.*.feed_forward.down_proj.weight [512,768]
};

struct ArkttsARWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    assets::TensorData text_embedding_host;
    assets::TensorData codebook_embedding_host;
    assets::TensorData fast_embedding_host;
    core::TensorValue text_embedding;
    std::vector<ArkttsLayerWeights> slow_layers;           // Qwen 0.6B path
    std::vector<FalconH1LayerWeights> falcon_layers;       // Falcon-H1 0.1B path (native ggml)
    assets::TensorDataF32 slow_norm;
    core::TensorValue falcon_lm_head;
    std::vector<ArkttsLayerWeights> fast_layers;
    assets::TensorDataF32 fast_norm;
    core::TensorValue fast_output;
};

struct SlowForwardOutput {
    std::vector<float> logits;
    std::vector<float> hidden;
};

struct SlowPrefillOutput {
    SlowForwardOutput forward;
};

struct ArkttsPrefillCacheTarget {
    std::vector<core::TensorValue> keys;
    std::vector<core::TensorValue> values;
};

modules::QwenDecoderActivationCastPolicy arktts_activation_cast_policy(core::BackendType backend_type) {
    modules::QwenDecoderActivationCastPolicy policy;
    if (backend_type == core::BackendType::Vulkan) {
        return policy;
    }
    policy.enabled = true;
    policy.type = GGML_TYPE_BF16;
    policy.after_input_norm = true;
    policy.after_qkv_projection = true;
    policy.after_qk_norm = true;
    policy.after_rope = true;
    policy.after_static_cache_update = true;
    policy.after_attention = true;
    policy.after_attention_output = true;
    policy.after_residual = true;
    policy.after_ffn_norm = true;
    policy.after_mlp_projection = true;
    policy.after_mlp_silu = true;
    policy.after_mlp_mul = true;
    policy.after_output = true;
    return policy;
}

modules::QwenCausalDecoderConfig make_slow_decoder_config(
    const Audio8TtsTextConfig & config,
    core::BackendType backend_type) {
    modules::QwenCausalDecoderConfig out;
    out.stack.hidden_size = config.dim;
    out.stack.num_attention_heads = config.n_head;
    out.stack.num_key_value_heads = config.n_local_heads;
    out.stack.head_dim = config.head_dim;
    out.stack.intermediate_size = config.intermediate_size;
    out.stack.layers = config.n_layer;
    out.stack.rms_norm_eps = config.norm_eps;
    out.stack.rope_theta = config.rope_base;
    out.stack.rope_type = GGML_ROPE_TYPE_NORMAL;
    out.stack.attention_precision = GGML_PREC_F32;
    out.stack.qkv_layout = modules::QwenDecoderQKVLayout::PackedQKV;
    out.stack.use_qk_norm = config.attention_qk_norm;
    out.stack.activation_cast = arktts_activation_cast_policy(backend_type);
    out.stack.runtime.attention.prefill_mode = modules::QwenDecoderAttentionMode::FlashGroupedViewKV;
    out.stack.runtime.attention.static_mode = modules::QwenDecoderAttentionMode::FlashGroupedViewKV;
    out.stack.runtime.static_cache.update_mode = modules::QwenDecoderStaticCacheUpdateMode::DirectSetRows;
    out.stack.runtime.static_cache.set_rows_mode = modules::QwenDecoderStaticCacheSetRowsMode::BackendViewOptimized;
    out.stack.runtime.mlp.mode = modules::QwenDecoderMLPMode::PackedGateUp;
    out.logits_size = config.vocab_size;
    out.logits_mode = modules::QwenCausalDecoderLogitsMode::LastStep;
    out.lm_head_precision = GGML_PREC_F32;
    return out;
}

modules::QwenCausalDecoderConfig make_fast_decoder_config(
    const Audio8TtsFastConfig & config,
    core::BackendType backend_type) {
    modules::QwenCausalDecoderConfig out;
    out.stack.hidden_size = config.dim;
    out.stack.num_attention_heads = config.n_head;
    out.stack.num_key_value_heads = config.n_local_heads;
    out.stack.head_dim = config.head_dim;
    out.stack.intermediate_size = config.intermediate_size;
    out.stack.layers = config.n_layer;
    out.stack.rms_norm_eps = config.norm_eps;
    out.stack.rope_theta = config.rope_base;
    out.stack.rope_type = GGML_ROPE_TYPE_NORMAL;
    out.stack.attention_precision = GGML_PREC_F32;
    out.stack.qkv_layout = modules::QwenDecoderQKVLayout::PackedQKV;
    out.stack.use_qk_norm = config.attention_qk_norm;
    out.stack.activation_cast = arktts_activation_cast_policy(backend_type);
    out.stack.runtime.attention.prefill_mode = modules::QwenDecoderAttentionMode::FlashGroupedViewKV;
    out.stack.runtime.attention.static_mode = modules::QwenDecoderAttentionMode::FlashGroupedViewKV;
    out.stack.runtime.static_cache.update_mode = modules::QwenDecoderStaticCacheUpdateMode::DirectSetRows;
    out.stack.runtime.static_cache.set_rows_mode = modules::QwenDecoderStaticCacheSetRowsMode::BackendViewOptimized;
    out.stack.runtime.mlp.mode = modules::QwenDecoderMLPMode::PackedGateUp;
    out.logits_size = config.vocab_size;
    out.logits_mode = modules::QwenCausalDecoderLogitsMode::LastStep;
    out.lm_head_precision = GGML_PREC_F32;
    return out;
}

modules::QwenDecoderLayerWeights bind_layer(
    core::ConstantTensorCache & constants,
    const ArkttsLayerWeights & weights,
    bool use_qk_norm) {
    modules::QwenDecoderLayerWeights out;
    out.input_norm = binding::norm_data(constants, weights.input_norm);
    out.self_attention.qkv_weight = weights.qkv_proj;
    if (weights.qkv_bias.has_value()) {
        out.self_attention.qkv_bias = *weights.qkv_bias;
    }
    out.self_attention.out_weight = weights.o_proj;
    if (use_qk_norm) {
        if (!weights.q_norm.has_value() || !weights.k_norm.has_value()) {
            throw std::runtime_error("Audio8 TTS q/k norm weights are missing");
        }
        out.q_norm = binding::norm_data(constants, *weights.q_norm);
        out.k_norm = binding::norm_data(constants, *weights.k_norm);
    }
    out.post_norm = binding::norm_data(constants, weights.post_norm);
    out.mlp.gate_up_proj = binding::linear_data(constants, weights.gate_up_proj);
    out.mlp.down_proj = binding::linear_data(constants, weights.down_proj);
    return out;
}

modules::QwenCausalDecoderWeights bind_slow_weights(
    core::ConstantTensorCache & constants,
    const ArkttsARWeights & weights,
    const Audio8TtsTextConfig & config) {
    modules::QwenCausalDecoderWeights out;
    out.stack.layers.reserve(weights.slow_layers.size());
    for (const auto & layer : weights.slow_layers) {
        out.stack.layers.push_back(bind_layer(constants, layer, config.attention_qk_norm));
    }
    out.final_norm = binding::norm_data(constants, weights.slow_norm);
    out.lm_head = binding::linear_data(constants, weights.text_embedding);
    return out;
}

modules::QwenDecoderLayerWeights bind_fast_layer(
    core::ConstantTensorCache & constants,
    const ArkttsLayerWeights & weights,
    const Audio8TtsFastConfig & config) {
    return bind_layer(constants, weights, config.attention_qk_norm);
}

void copy_tensor_row_to_f32(const assets::TensorData & table, int64_t row, int64_t width, float * out) {
    if (row < 0 || width <= 0 || table.shape.rank != 2 || table.shape.dims[1] != width ||
        row >= table.shape.dims[0]) {
        throw std::runtime_error("Audio8 TTS embedding row lookup shape mismatch");
    }
    const size_t row_bytes = ggml_row_size(table.type, width);
    const size_t offset = static_cast<size_t>(row) * row_bytes;
    if (offset + row_bytes > table.bytes.size()) {
        throw std::runtime_error("Audio8 TTS embedding row lookup exceeded tensor storage");
    }
    const auto * bytes = reinterpret_cast<const uint8_t *>(table.bytes.data()) + offset;
    if (table.type == GGML_TYPE_F32) {
        std::memcpy(out, bytes, static_cast<size_t>(width) * sizeof(float));
    } else if (table.type == GGML_TYPE_F16) {
        ggml_fp16_to_fp32_row(reinterpret_cast<const ggml_fp16_t *>(bytes), out, width);
    } else if (table.type == GGML_TYPE_BF16) {
        ggml_bf16_to_fp32_row(reinterpret_cast<const ggml_bf16_t *>(bytes), out, width);
    } else {
        throw std::runtime_error("Audio8 TTS host embedding lookup requires f32/f16/bf16 native embeddings");
    }
}

std::vector<float> lookup_row(const assets::TensorData & table, int64_t row, int64_t width) {
    std::vector<float> out(static_cast<size_t>(width), 0.0F);
    copy_tensor_row_to_f32(table, row, width, out.data());
    return out;
}

void add_row(const assets::TensorData & table, int64_t row, int64_t width, std::vector<float> & out) {
    std::vector<float> tmp(static_cast<size_t>(width), 0.0F);
    copy_tensor_row_to_f32(table, row, width, tmp.data());
    for (int64_t i = 0; i < width; ++i) {
        out[static_cast<size_t>(i)] += tmp[static_cast<size_t>(i)];
    }
}

bool is_semantic_token(const Audio8TtsConfig & config, int32_t token) {
    return token >= config.semantic_start_token_id && token <= config.semantic_end_token_id;
}

std::vector<float> build_slow_embeddings(
    const Audio8TtsConfig & config,
    const ArkttsARWeights & weights,
    const int32_t * matrix,
    int64_t steps) {
    const int64_t hidden = config.text.dim;
    // modeling_arktts.py:_embed — plain embeddings(token) + sum(codebook_embeddings); no scaling
    std::vector<float> out(static_cast<size_t>(steps * hidden), 0.0F);
    for (int64_t step = 0; step < steps; ++step) {
        const int32_t token = matrix[step];
        auto row = lookup_row(weights.text_embedding_host, token, hidden);
        if (is_semantic_token(config, token)) {
            for (int64_t codebook = 0; codebook < config.fast.num_codebooks; ++codebook) {
                const int32_t code = matrix[(codebook + 1) * steps + step];
                add_row(
                    weights.codebook_embedding_host,
                    codebook * config.fast.vocab_size + code,
                    hidden,
                    row);
            }
        }
        std::copy(row.begin(), row.end(), out.begin() + static_cast<std::ptrdiff_t>(step * hidden));
    }
    return out;
}

std::vector<float> build_slow_embedding_for_frame(
    const Audio8TtsConfig & config,
    const ArkttsARWeights & weights,
    const std::vector<int32_t> & frame) {
    if (static_cast<int64_t>(frame.size()) != config.fast.num_codebooks + 1) {
        throw std::runtime_error("Audio8 TTS frame size mismatch");
    }
    return build_slow_embeddings(config, weights, frame.data(), 1);
}

std::vector<float> build_fast_embedding(
    const Audio8TtsConfig & config,
    const ArkttsARWeights & weights,
    int32_t code) {
    return lookup_row(weights.fast_embedding_host, code, config.fast.dim);
}

ArkttsLayerWeights load_layer(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t hidden,
    int64_t heads,
    int64_t kv_heads,
    int64_t head_dim,
    int64_t intermediate,
    bool qk_norm,
    bool with_qkv_bias,
    assets::TensorStorageType storage_type) {
    ArkttsLayerWeights w;
    w.input_norm = source.require_f32_tensor(prefix + ".attention_norm.weight", {hidden});
    {
        // modeling_arktts.py checkpoints ship attention QKV pre-concatenated as
        // wqkv [(heads + 2 * kv_heads) * head_dim, hidden]; bind it without re-packing.
        const auto qkv = source.require_tensor(
            prefix + ".attention.wqkv.weight",
            storage_type,
            {(heads + 2 * kv_heads) * head_dim, hidden});
        w.qkv_proj = store.make_tensor(
            core::TensorShape::from_dims({(heads + 2 * kv_heads) * head_dim, hidden}),
            qkv.type,
            qkv.bytes.data(),
            qkv.bytes.size());
        if (with_qkv_bias) {
            const auto bias_rows = (heads + 2 * kv_heads) * head_dim;
            w.qkv_bias = store.make_f32(
                core::TensorShape::from_dims({bias_rows}),
                source.require_f32(prefix + ".attention.wqkv.bias", {bias_rows}));
        }
    }
    w.o_proj = store.load_tensor(source, prefix + ".attention.wo.weight", storage_type, {hidden, heads * head_dim});
    if (qk_norm) {
        w.q_norm = source.require_f32_tensor(prefix + ".attention.q_norm.weight", {head_dim});
        w.k_norm = source.require_f32_tensor(prefix + ".attention.k_norm.weight", {head_dim});
    }
    w.post_norm = source.require_f32_tensor(prefix + ".ffn_norm.weight", {hidden});
    w.down_proj = store.load_tensor(source, prefix + ".feed_forward.w2.weight", storage_type, {hidden, intermediate});
    {
        const auto gate = source.require_tensor(prefix + ".feed_forward.w1.weight", storage_type, {intermediate, hidden});
        const auto up = source.require_tensor(prefix + ".feed_forward.w3.weight", storage_type, {intermediate, hidden});
        if (gate.type != up.type) {
            throw std::runtime_error("Audio8 TTS packed gate/up weights require matching storage types");
        }
        std::vector<std::byte> packed;
        packed.reserve(gate.bytes.size() + up.bytes.size());
        packed.insert(packed.end(), gate.bytes.begin(), gate.bytes.end());
        packed.insert(packed.end(), up.bytes.begin(), up.bytes.end());
        w.gate_up_proj = store.make_tensor(
            core::TensorShape::from_dims({intermediate * 2, hidden}),
            gate.type,
            packed.data(),
            packed.size());
    }
    return w;
}

FalconH1LayerWeights load_falcon_layer(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    const Audio8TtsTextConfig & text_config,
    assets::TensorStorageType storage_type) {
    // Ported from ../SenseVoice/runtime/llama.cpp/build/_deps/llama-src/src/models/falcon-h1.cpp
    // and HF /workspace/models/Audio8-TTS-Preview-0.1b/modeling_arktts.py
    // FalconH1DecoderLayer: input_layernorm -> parallel mamba (FalconH1Mixer) + attention -> sum -> residual -> pre_ff_layernorm -> ffn
    // Shapes reflect HF safetensors (safetensors) and GGUF (after convert) – use actual metadata shape to stay compatible.
    FalconH1LayerWeights w;
    w.input_layernorm = source.require_f32_tensor(prefix + ".input_layernorm.weight", {text_config.dim});
    // Mamba in_proj: HF [1688,512] (out, in), GGUF may be transposed; load with actual shape
    {
        auto meta = source.require_metadata(prefix + ".mamba.in_proj.weight");
        w.ssm_in = store.load_tensor(source, prefix + ".mamba.in_proj.weight", storage_type, meta.shape);
    }
    {
        auto meta = source.require_metadata(prefix + ".mamba.conv1d.weight");
        w.ssm_conv1d = store.load_tensor(source, prefix + ".mamba.conv1d.weight", storage_type, meta.shape);
    }
    w.ssm_conv1d_b = source.require_f32_tensor(prefix + ".mamba.conv1d.bias");
    {
        auto meta = source.require_metadata(prefix + ".mamba.dt_bias");
        w.ssm_dt_b = store.load_tensor(source, prefix + ".mamba.dt_bias", storage_type, meta.shape);
    }
    {
        auto meta = source.require_metadata(prefix + ".mamba.A_log");
        w.ssm_A = store.load_tensor(source, prefix + ".mamba.A_log", storage_type, meta.shape);
    }
    {
        auto meta = source.require_metadata(prefix + ".mamba.D");
        w.ssm_D = store.load_tensor(source, prefix + ".mamba.D", storage_type, meta.shape);
    }
    {
        auto meta = source.require_metadata(prefix + ".mamba.out_proj.weight");
        w.ssm_out = store.load_tensor(source, prefix + ".mamba.out_proj.weight", storage_type, meta.shape);
    }
    {
        auto meta = source.require_metadata(prefix + ".self_attn.q_proj.weight");
        w.attn_q_proj = store.load_tensor(source, prefix + ".self_attn.q_proj.weight", storage_type, meta.shape);
    }
    {
        auto meta = source.require_metadata(prefix + ".self_attn.k_proj.weight");
        w.attn_k_proj = store.load_tensor(source, prefix + ".self_attn.k_proj.weight", storage_type, meta.shape);
    }
    {
        auto meta = source.require_metadata(prefix + ".self_attn.v_proj.weight");
        w.attn_v_proj = store.load_tensor(source, prefix + ".self_attn.v_proj.weight", storage_type, meta.shape);
    }
    {
        auto meta = source.require_metadata(prefix + ".self_attn.o_proj.weight");
        w.attn_o_proj = store.load_tensor(source, prefix + ".self_attn.o_proj.weight", storage_type, meta.shape);
    }
    w.pre_ff_layernorm = source.require_f32_tensor(prefix + ".pre_ff_layernorm.weight", {text_config.dim});
    {
        auto meta = source.require_metadata(prefix + ".feed_forward.gate_proj.weight");
        w.ffn_gate = store.load_tensor(source, prefix + ".feed_forward.gate_proj.weight", storage_type, meta.shape);
    }
    {
        auto meta = source.require_metadata(prefix + ".feed_forward.up_proj.weight");
        w.ffn_up = store.load_tensor(source, prefix + ".feed_forward.up_proj.weight", storage_type, meta.shape);
    }
    {
        auto meta = source.require_metadata(prefix + ".feed_forward.down_proj.weight");
        w.ffn_down = store.load_tensor(source, prefix + ".feed_forward.down_proj.weight", storage_type, meta.shape);
    }
    return w;
}

ArkttsARWeights load_ar_weights(
    const Audio8TtsAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t weight_context_bytes,
    assets::TensorStorageType storage_type) {
    const auto & source = *assets.model_weights;
    const auto & config = assets.config;
    ArkttsARWeights weights;
    weights.store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "audio8_tts.ar.weights",
        weight_context_bytes);
    const bool is_mamba = config.text.slow_backbone == "falcon_h1" || source.has_tensor("slow.embed_tokens.weight");
    if (is_mamba) {
        weights.text_embedding_host = source.require_tensor(
            "slow.embed_tokens.weight",
            assets::TensorStorageType::Native,
            {config.text.vocab_size, config.text.dim});
    } else {
        weights.text_embedding_host = source.require_tensor(
            "embeddings.weight",
            assets::TensorStorageType::Native,
            {config.text.vocab_size, config.text.dim});
    }
    weights.codebook_embedding_host = source.require_tensor(
        "codebook_embeddings.weight",
        assets::TensorStorageType::Native,
        {config.fast.vocab_size * config.fast.num_codebooks, config.text.dim});
    weights.fast_embedding_host = source.require_tensor(
        "fast_embeddings.weight",
        assets::TensorStorageType::Native,
        {config.fast.vocab_size, config.fast.dim});
    if (is_mamba) {
        weights.text_embedding = weights.store->load_tensor(
            source,
            "slow.embed_tokens.weight",
            storage_type,
            {config.text.vocab_size, config.text.dim});
    } else {
        weights.text_embedding = weights.store->load_tensor(
            source,
            "embeddings.weight",
            storage_type,
            {config.text.vocab_size, config.text.dim});
    }
    weights.slow_layers.reserve(static_cast<size_t>(config.text.n_layer));
    weights.falcon_layers.reserve(is_mamba ? static_cast<size_t>(config.text.n_layer) : 0);
    if (is_mamba) {
        // Falcon-H1 hybrid: weights are loadable via native ggml; graph will be via
        // ../SenseVoice/runtime/llama.cpp/build/_deps/llama-src/src/models/mamba-base.cpp:149
        // build_mamba2_layer + falcon-h1.cpp aggregation (ggml_ssm_conv/scan).
        for (int64_t i = 0; i < config.text.n_layer; ++i) {
            weights.falcon_layers.push_back(load_falcon_layer(
                *weights.store, source, "slow.layers." + std::to_string(i), config.text, storage_type));
        }
        weights.slow_norm = source.require_f32_tensor("slow.final_layernorm.weight", {config.text.dim});
        {
            auto meta = source.require_metadata("semantic_output.weight");
            weights.falcon_lm_head = weights.store->load_tensor(source, "semantic_output.weight", storage_type, meta.shape);
        }
    } else {
        for (int64_t i = 0; i < config.text.n_layer; ++i) {
            weights.slow_layers.push_back(load_layer(
                *weights.store,
                source,
                "layers." + std::to_string(i),
                config.text.dim,
                config.text.n_head,
                config.text.n_local_heads,
                config.text.head_dim,
                config.text.intermediate_size,
                config.text.attention_qk_norm,
                true,
                storage_type));
        }
        weights.slow_norm = source.require_f32_tensor("norm.weight", {config.text.dim});
    }
    weights.fast_layers.reserve(static_cast<size_t>(config.fast.n_layer));
    for (int64_t i = 0; i < config.fast.n_layer; ++i) {
        weights.fast_layers.push_back(load_layer(
            *weights.store,
            source,
            "fast_layers." + std::to_string(i),
            config.fast.dim,
            config.fast.n_head,
            config.fast.n_local_heads,
            config.fast.head_dim,
            config.fast.intermediate_size,
            config.fast.attention_qk_norm,
            false,
            storage_type));
    }
    weights.fast_norm = source.require_f32_tensor("fast_norm.weight", {config.fast.dim});
    weights.fast_output = weights.store->load_tensor(
        source,
        "fast_output.weight",
        storage_type,
        {config.fast.vocab_size, config.fast.dim});
    weights.store->upload();
    return weights;
}

struct SampleState {
    uint64_t seed = 0;
    uint64_t call_index = 0;
    std::mt19937 rng;
    std::vector<int32_t> previous_main;
};

SampleDistribution logits_to_distribution(
    const std::vector<float> & logits,
    float temperature,
    float top_p,
    int top_k) {
    if (logits.empty()) {
        throw std::runtime_error("Audio8 TTS sampling requires non-empty logits");
    }
    std::vector<int32_t> order;
    order.reserve(logits.size());
    float max_logit = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < logits.size(); ++i) {
        const float logit = logits[i];
        if (!std::isfinite(logit)) {
            continue;
        }
        order.push_back(static_cast<int32_t>(i));
        max_logit = std::max(max_logit, logit);
    }
    double denom = 0.0;
    for (const int32_t index : order) {
        denom += std::exp(logits[static_cast<size_t>(index)] - max_logit);
    }
    if (denom <= 0.0) {
        throw std::runtime_error("Audio8 TTS sampling logits produced zero probability mass");
    }
    const size_t candidate_count = std::min(order.size(), static_cast<size_t>(std::max(top_k, 1)));
    const auto by_logit_desc = [&](int32_t lhs, int32_t rhs) {
        return logits[static_cast<size_t>(lhs)] > logits[static_cast<size_t>(rhs)];
    };
    if (candidate_count < order.size()) {
        std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(candidate_count), order.end(), by_logit_desc);
        order.resize(candidate_count);
    } else {
        std::sort(order.begin(), order.end(), by_logit_desc);
    }
    double cumulative = 0.0;
    std::vector<SampleCandidate> kept;
    kept.reserve(candidate_count);
    for (size_t i = 0; i < order.size(); ++i) {
        const int32_t index = order[i];
        const float logit = logits[static_cast<size_t>(index)];
        const float prob = static_cast<float>(std::exp(logit - max_logit) / denom);
        cumulative += prob;
        const bool remove = cumulative > static_cast<double>(top_p) && i != 0;
        if (!remove) {
            kept.push_back({index, 0.0F});
        }
    }
    float filtered_max = -std::numeric_limits<float>::infinity();
    const float temperature_scale = std::max(temperature, 1.0e-5F);
    for (const auto & candidate : kept) {
        filtered_max = std::max(filtered_max, logits[static_cast<size_t>(candidate.index)] / temperature_scale);
    }
    double filtered_denom = 0.0;
    for (auto & candidate : kept) {
        candidate.probability =
            std::exp(logits[static_cast<size_t>(candidate.index)] / temperature_scale - filtered_max);
        filtered_denom += candidate.probability;
    }
    if (filtered_denom <= 0.0) {
        throw std::runtime_error("Audio8 TTS sampling filter produced zero probability mass");
    }
    for (auto & candidate : kept) {
        candidate.probability = static_cast<float>(static_cast<double>(candidate.probability) / filtered_denom);
    }
    return {logits.size(), std::move(kept)};
}

int32_t sample_from_logits(
    const std::vector<float> & logits,
    float temperature,
    float top_p,
    int top_k,
    SampleState & state,
    const sampling::TorchCudaSamplingPolicy & policy) {
    const auto distribution = logits_to_distribution(logits, temperature, top_p, top_k);
    const uint64_t call_index = state.call_index++;
    if (!policy.cuda_fast_path) {
        std::vector<double> weights;
        weights.reserve(distribution.candidates.size());
        for (const auto & candidate : distribution.candidates) {
            weights.push_back(static_cast<double>(std::max(candidate.probability, 0.0F)));
        }
        std::discrete_distribution<size_t> sampler(weights.begin(), weights.end());
        return distribution.candidates[sampler(state.rng)].index;
    }
    int32_t best = 0;
    double best_score = -std::numeric_limits<double>::infinity();
    for (const auto & candidate : distribution.candidates) {
        if (!(candidate.probability > 0.0F)) {
            continue;
        }
        const float exponential = sampling::torch_cuda_tensor_iterator_exponential_element(
            state.seed,
            static_cast<uint64_t>(distribution.source_size),
            static_cast<uint64_t>(candidate.index),
            call_index,
            policy.multiprocessor_count,
            policy.max_threads_per_multiprocessor);
        const float uniform = std::exp(-exponential);
        const float uniform_bf16 = ggml_bf16_to_fp32(ggml_fp32_to_bf16(uniform));
        const float exponential_bf16 = ggml_bf16_to_fp32(ggml_fp32_to_bf16(-std::log(uniform_bf16)));
        const double score = static_cast<double>(candidate.probability) / static_cast<double>(exponential_bf16);
        if (score > best_score) {
            best_score = score;
            best = candidate.index;
        }
    }
    return best;
}

std::vector<float> apply_semantic_bias(
    const Audio8TtsConfig & config,
    int32_t im_end_id,
    const std::vector<float> & logits) {
    std::vector<float> out(logits.size(), -std::numeric_limits<float>::infinity());
    const int64_t begin = std::max<int64_t>(0, config.semantic_start_token_id);
    const int64_t end = std::min<int64_t>(static_cast<int64_t>(logits.size()) - 1, config.semantic_end_token_id);
    for (int64_t i = begin; i <= end; ++i) {
        out[static_cast<size_t>(i)] = logits[static_cast<size_t>(i)];
    }
    if (im_end_id >= 0 && static_cast<size_t>(im_end_id) < logits.size()) {
        out[static_cast<size_t>(im_end_id)] = logits[static_cast<size_t>(im_end_id)];
    }
    return out;
}

core::TensorValue make_arktts_causal_mask(
    core::ModuleBuildContext &,
    core::ConstantTensorCache & constants,
    int64_t steps) {
    auto values = modules::qwen_causal_prefill_mask_values(1, steps);
    return constants.make_tensor(
        core::TensorShape::from_dims({1, 1, steps, steps}),
        GGML_TYPE_F16,
        values.data(),
        values.size() * sizeof(ggml_fp16_t));
}

struct ArkttsCausalDecoderOutputs {
    core::TensorValue hidden;
    core::TensorValue logits;
    modules::QwenDecoderStackState state;
};

ArkttsCausalDecoderOutputs build_arktts_causal_decoder(
    core::ModuleBuildContext & ctx,
    core::ConstantTensorCache & constants,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const modules::QwenCausalDecoderWeights & weights,
    const modules::QwenCausalDecoderConfig & config,
    bool norm_fastlayer_input) {
    auto mask = make_arktts_causal_mask(ctx, constants, input.shape.dims[1]);
    auto x = input;
    modules::QwenDecoderStackState state;
    state.layers.reserve(weights.stack.layers.size());
    const auto layer_config = modules::qwen_decoder_layer_config_from_stack(config.stack);
    const modules::QwenDecoderLayerModule layer_module(layer_config);
    for (const auto & layer : weights.stack.layers) {
        auto out = layer_module.build(ctx, x, positions, layer, std::nullopt, std::nullopt, mask);
        x = out.output;
        auto state_key = core::wrap_tensor(ggml_dup(ctx.ggml, out.key.tensor), out.key.shape, out.key.type);
        auto state_value = core::wrap_tensor(ggml_dup(ctx.ggml, out.value.tensor), out.value.shape, out.value.type);
        state.layers.push_back({state_key, state_value});
    }
    auto hidden_sequence = modules::RMSNormModule({config.stack.hidden_size, config.stack.rms_norm_eps, true, false})
                               .build(ctx, x, weights.final_norm);
    const int64_t steps = hidden_sequence.shape.dims[1];
    auto fast_hidden_source = norm_fastlayer_input ? hidden_sequence : x;
    auto hidden = modules::SliceModule({1, steps - 1, 1}).build(ctx, fast_hidden_source);
    auto logits = modules::LinearModule({config.stack.hidden_size, config.logits_size, false, config.lm_head_precision})
                      .build(ctx, modules::SliceModule({1, steps - 1, 1}).build(ctx, hidden_sequence), weights.lm_head);
    auto hidden_out = core::wrap_tensor(ggml_dup(ctx.ggml, hidden.tensor), hidden.shape, hidden.type);
    auto logits_out = core::wrap_tensor(ggml_dup(ctx.ggml, logits.tensor), logits.shape, logits.type);
    return {hidden_out, logits_out, std::move(state)};
}

struct ArkttsStaticDecoderOutputs {
    core::TensorValue hidden;
    core::TensorValue logits;
    runtime::TransformerKVCache cache;
};

ArkttsStaticDecoderOutputs build_arktts_static_decoder(
    core::ModuleBuildContext & ctx,
    ggml_cgraph * graph,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const modules::QwenCausalDecoderWeights & weights,
    const modules::QwenCausalDecoderConfig & config,
    int64_t cache_steps,
    const core::TensorValue & attention_mask,
    const core::TensorValue & cache_slot,
    std::vector<core::TensorValue> cache_keys,
    std::vector<core::TensorValue> cache_values,
    bool norm_fastlayer_input) {
    if (cache_keys.size() != weights.stack.layers.size() || cache_values.size() != weights.stack.layers.size()) {
        throw std::runtime_error("Audio8 TTS static decoder cache layer count mismatch");
    }
    const int64_t step_elems = config.stack.num_key_value_heads * config.stack.head_dim;
    auto x = input;
    const auto layer_config = modules::qwen_decoder_layer_config_from_stack(config.stack);
    const modules::QwenDecoderLayerModule layer_module(layer_config);
    for (size_t layer_index = 0; layer_index < weights.stack.layers.size(); ++layer_index) {
        auto out = layer_module.build_with_static_cache_tail(
            ctx,
            graph,
            x,
            positions,
            weights.stack.layers[layer_index],
            cache_keys[layer_index],
            cache_values[layer_index],
            cache_slot,
            attention_mask);
        x = out.output;
    }
    auto hidden = modules::RMSNormModule({config.stack.hidden_size, config.stack.rms_norm_eps, true, false})
                      .build(ctx, x, weights.final_norm);
    const auto logits = modules::LinearModule({
                            config.stack.hidden_size,
                            config.logits_size,
                            config.use_lm_head_bias,
                            config.lm_head_precision,
                        })
                            .build(ctx, hidden, weights.lm_head);
    auto fast_hidden = norm_fastlayer_input ? hidden : x;
    runtime::TransformerKVCacheOptions cache_options;
    cache_options.allow_bf16_storage = !cache_keys.empty() && cache_keys.front().type == GGML_TYPE_BF16;
    return {
        fast_hidden,
        logits,
        runtime::TransformerKVCache(
            cache_steps,
            step_elems,
            std::move(cache_keys),
            std::move(cache_values),
            cache_options),
    };
}

std::vector<float> build_falcon_embeddings(
    const Audio8TtsConfig & config,
    const ArkttsARWeights & weights,
    const int32_t * matrix,
    int64_t steps) {
    const int64_t hidden = config.text.dim;
    std::vector<float> out(static_cast<size_t>(steps * hidden), 0.0F);
    for (int64_t step = 0; step < steps; ++step) {
        const int32_t token = matrix[step];
        auto row = lookup_row(weights.text_embedding_host, token, hidden);
        for (auto & v : row) v *= config.text.embedding_multiplier;
        if (is_semantic_token(config, token)) {
            for (int64_t codebook = 0; codebook < config.fast.num_codebooks; ++codebook) {
                const int32_t code = matrix[(codebook + 1) * steps + step];
                add_row(weights.codebook_embedding_host, codebook * config.fast.vocab_size + code, hidden, row);
            }
        }
        std::copy(row.begin(), row.end(), out.begin() + static_cast<std::ptrdiff_t>(step * hidden));
    }
    return out;
}

// TODO(Falcon-H1): Replace with full Mamba2 port (ggml_ssm_conv + B/C/dt/A/D
// + ggml_ssm_scan + recurrent conv/ssm state + hybrid attention).
// See docs/FALCON_H1_0.1B_PORT_PLAN.md M2/M3 and
// ../llama.cpp/src/models/mamba-base.cpp:151 / falcon-h1.cpp:132.
// Current stub keeps weight loading native but omits the SSM core and
// hybrid attention (attn_out = 0), recomputes full sequence each step,
// and only applies ssm_out/lm_head multipliers — tracked for follow-up.
SlowForwardOutput falcon_forward_stateless(
    ggml_backend_t backend,
    int threads,
    size_t arena_bytes,
    const Audio8TtsConfig & config,
    const ArkttsARWeights & weights,
    const std::vector<float> & embeddings,
    int64_t seq_len) {
    if (seq_len <= 0) throw std::runtime_error("falcon_forward: zero seq");
    if (weights.falcon_layers.empty()) throw std::runtime_error("falcon_forward: no falcon layers");
    const int64_t dim = config.text.dim;
    const float eps = config.text.norm_eps;
    const float lm_mult = config.text.lm_head_multiplier;
    ggml_init_params params{arena_bytes, nullptr, true};
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx(ggml_init(params));
    if (!ctx) throw std::runtime_error("falcon_forward: ggml_init failed");
    ggml_tensor * cur = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, dim, seq_len);
    ggml_set_name(cur, "falcon_input");
    std::vector<ggml_tensor*> ln_ws;
    std::vector<ggml_tensor*> bias_ws;
    std::vector<ggml_tensor*> pre_ws;
    ln_ws.reserve(weights.falcon_layers.size());
    bias_ws.reserve(weights.falcon_layers.size());
    pre_ws.reserve(weights.falcon_layers.size());
    for (size_t li = 0; li < weights.falcon_layers.size(); ++li) {
        const auto & layer = weights.falcon_layers[li];
        ggml_tensor * ln_w = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, dim);
        ln_ws.push_back(ln_w);
        ggml_tensor * normed = ggml_rms_norm(ctx.get(), cur, eps);
        normed = ggml_mul(ctx.get(), normed, ln_w);
        ggml_tensor * proj = ggml_mul_mat(ctx.get(), layer.ssm_in.tensor, normed);
        ggml_tensor * gate = ggml_view_2d(ctx.get(), proj, 768, seq_len, proj->nb[1], 0);
        ggml_tensor * xBC = ggml_view_2d(ctx.get(), proj, 896, seq_len, proj->nb[1], 768 * sizeof(float));
        ggml_tensor * bias = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 896);
        bias_ws.push_back(bias);
        ggml_tensor * bias_bcast = ggml_repeat(ctx.get(), bias, xBC);
        ggml_tensor * xBC_b = ggml_add(ctx.get(), xBC, bias_bcast);
        ggml_tensor * xBC_silu = ggml_silu(ctx.get(), xBC_b);
        ggml_tensor * x = ggml_view_2d(ctx.get(), xBC_silu, 768, seq_len, xBC_silu->nb[1], 0);
        ggml_tensor * gate_silu = ggml_silu(ctx.get(), gate);
        ggml_tensor * y_gated = ggml_mul(ctx.get(), x, gate_silu);
        ggml_tensor * out_mamba = ggml_mul_mat(ctx.get(), layer.ssm_out.tensor, y_gated);
        if (std::abs(config.text.ssm_out_multiplier - 1.0f) > 1e-6) out_mamba = ggml_scale(ctx.get(), out_mamba, config.text.ssm_out_multiplier);
        ggml_tensor * attn_out = ggml_scale(ctx.get(), cur, 0.0f);
        ggml_tensor * hybrid = ggml_add(ctx.get(), out_mamba, attn_out);
        ggml_tensor * cur_res = ggml_add(ctx.get(), cur, hybrid);
        ggml_tensor * pre_w = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, dim);
        pre_ws.push_back(pre_w);
        ggml_tensor * pre_norm = ggml_rms_norm(ctx.get(), cur_res, eps);
        pre_norm = ggml_mul(ctx.get(), pre_norm, pre_w);
        ggml_tensor * gate_ff = ggml_mul_mat(ctx.get(), layer.ffn_gate.tensor, pre_norm);
        ggml_tensor * up_ff = ggml_mul_mat(ctx.get(), layer.ffn_up.tensor, pre_norm);
        ggml_tensor * gate_silu2 = ggml_silu(ctx.get(), gate_ff);
        ggml_tensor * gated = ggml_mul(ctx.get(), gate_silu2, up_ff);
        ggml_tensor * down = ggml_mul_mat(ctx.get(), layer.ffn_down.tensor, gated);
        cur = ggml_add(ctx.get(), cur_res, down);
    }
    ggml_tensor * final_w = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, dim);
    ggml_tensor * final_norm = ggml_rms_norm(ctx.get(), cur, eps);
    final_norm = ggml_mul(ctx.get(), final_norm, final_w);
    ggml_tensor * last_hidden = ggml_view_2d(ctx.get(), final_norm, dim, 1, final_norm->nb[1], (seq_len - 1) * final_norm->nb[1]);
    ggml_tensor * logits = ggml_mul_mat(ctx.get(), weights.falcon_lm_head.tensor, last_hidden);
    if (std::abs(lm_mult - 1.0f) > 1e-6) logits = ggml_scale(ctx.get(), logits, lm_mult);
    ggml_tensor * logits_out = ggml_dup(ctx.get(), logits);
    ggml_tensor * hidden_out = ggml_dup(ctx.get(), last_hidden);
    ggml_set_name(logits_out, "logits_out");
    ggml_set_name(hidden_out, "hidden_out");
    ggml_cgraph * gf = ggml_new_graph_custom(ctx.get(), 8192, false);
    ggml_build_forward_expand(gf, logits_out);
    ggml_build_forward_expand(gf, hidden_out);
    ggml_gallocr_t gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!gallocr || !ggml_gallocr_reserve(gallocr, gf) || !ggml_gallocr_alloc_graph(gallocr, gf)) throw std::runtime_error("falcon_forward: gallocr failed");
    for (size_t i = 0; i < ln_ws.size(); ++i) {
        const auto & vals = weights.falcon_layers[i].input_layernorm.values;
        if (!vals.empty()) ggml_backend_tensor_set(ln_ws[i], vals.data(), 0, vals.size() * sizeof(float));
        else { std::vector<float> ones(static_cast<size_t>(dim), 1.0f); ggml_backend_tensor_set(ln_ws[i], ones.data(), 0, ones.size() * sizeof(float)); }
    }
    for (size_t i = 0; i < bias_ws.size(); ++i) {
        const auto & vals = weights.falcon_layers[i].ssm_conv1d_b.values;
        if (!vals.empty()) ggml_backend_tensor_set(bias_ws[i], vals.data(), 0, vals.size() * sizeof(float));
        else { std::vector<float> zeros(896, 0.0f); ggml_backend_tensor_set(bias_ws[i], zeros.data(), 0, zeros.size() * sizeof(float)); }
    }
    for (size_t i = 0; i < pre_ws.size(); ++i) {
        const auto & vals = weights.falcon_layers[i].pre_ff_layernorm.values;
        if (!vals.empty()) ggml_backend_tensor_set(pre_ws[i], vals.data(), 0, vals.size() * sizeof(float));
        else { std::vector<float> ones(static_cast<size_t>(dim), 1.0f); ggml_backend_tensor_set(pre_ws[i], ones.data(), 0, ones.size() * sizeof(float)); }
    }
    if (!weights.slow_norm.values.empty()) ggml_backend_tensor_set(final_w, weights.slow_norm.values.data(), 0, weights.slow_norm.values.size() * sizeof(float));
    else { std::vector<float> ones(static_cast<size_t>(dim), 1.0f); ggml_backend_tensor_set(final_w, ones.data(), 0, ones.size() * sizeof(float)); }
    std::vector<float> cur_data(static_cast<size_t>(dim * seq_len));
    for (int64_t s = 0; s < seq_len; ++s) for (int64_t d = 0; d < dim; ++d) cur_data[static_cast<size_t>(d + s * dim)] = embeddings[static_cast<size_t>(s * dim + d)];
    ggml_backend_tensor_set(cur, cur_data.data(), 0, cur_data.size() * sizeof(float));
    core::set_backend_threads(backend, threads);
    ggml_status status = core::compute_backend_graph(backend, gf, nullptr, "falcon_forward");
    ggml_backend_synchronize(backend);
    if (status != GGML_STATUS_SUCCESS) throw std::runtime_error("falcon_forward compute failed");
    SlowForwardOutput out;
    size_t vocab = static_cast<size_t>(logits_out->ne[0]);
    if (vocab == 0) vocab = 4097;
    out.logits.resize(vocab);
    out.hidden.resize(static_cast<size_t>(dim));
    ggml_backend_tensor_get(logits_out, out.logits.data(), 0, vocab * sizeof(float));
    ggml_backend_tensor_get(hidden_out, out.hidden.data(), 0, static_cast<size_t>(dim) * sizeof(float));
    ggml_gallocr_free(gallocr);
    core::release_backend_graph_resources(backend, gf);
    return out;
}

}  // namespace

class ArkttsARWeightsRuntime {
public:
    ArkttsARWeightsRuntime(
        std::shared_ptr<const Audio8TtsAssets> assets,
        core::BackendConfig backend_config,
        int threads,
        size_t graph_arena_bytes,
        size_t weight_context_bytes,
        assets::TensorStorageType weight_storage_type)
        : assets_(std::move(assets)),
          threads_(threads),
          graph_arena_bytes_(graph_arena_bytes) {
        if (assets_ == nullptr) {
            throw std::runtime_error("Audio8 TTS AR weights runtime requires assets");
        }
        backend_config.threads = threads_;
        backend_ = core::init_backend(backend_config);
        backend_type_ = core::backend_type(backend_);
        weights_ = std::make_shared<ArkttsARWeights>(
            load_ar_weights(*assets_, backend_, backend_type_, weight_context_bytes, weight_storage_type));
        slow_step_constants_ = std::make_unique<core::ConstantTensorCache>(
            backend_,
            threads_,
            "audio8_tts.ar.step.constants",
            256ull * 1024ull * 1024ull);
        fast_constants_ = std::make_unique<core::ConstantTensorCache>(
            backend_,
            threads_,
            "audio8_tts.ar.fast.constants",
            256ull * 1024ull * 1024ull);
    }

    ~ArkttsARWeightsRuntime() {
        fast_constants_.reset();
        slow_step_constants_.reset();
        weights_.reset();
        if (backend_ != nullptr) {
            ggml_backend_free(backend_);
        }
    }

    ArkttsARWeightsRuntime(const ArkttsARWeightsRuntime &) = delete;
    ArkttsARWeightsRuntime & operator=(const ArkttsARWeightsRuntime &) = delete;

    const Audio8TtsAssets & assets() const noexcept {
        return *assets_;
    }

    const ArkttsARWeights & weights() const noexcept {
        return *weights_;
    }

    int threads() const noexcept {
        return threads_;
    }

    size_t graph_arena_bytes() const noexcept {
        return graph_arena_bytes_;
    }

    ggml_backend_t backend() const noexcept {
        return backend_;
    }

    core::BackendType backend_type() const noexcept {
        return backend_type_;
    }

    core::ConstantTensorCache & slow_step_constants() const noexcept {
        return *slow_step_constants_;
    }

    core::ConstantTensorCache & fast_constants() const noexcept {
        return *fast_constants_;
    }

private:
    std::shared_ptr<const Audio8TtsAssets> assets_;
    std::shared_ptr<const ArkttsARWeights> weights_;
    int threads_ = 1;
    size_t graph_arena_bytes_ = 0;
    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    std::unique_ptr<core::ConstantTensorCache> slow_step_constants_;
    std::unique_ptr<core::ConstantTensorCache> fast_constants_;
};

class Audio8TtsARRuntime::Impl {
public:
    Impl(
        std::shared_ptr<const Audio8TtsAssets> assets,
        core::BackendConfig backend_config,
        int threads,
        size_t graph_arena_bytes,
        size_t weight_context_bytes,
        assets::TensorStorageType weight_storage_type)
        : runtime_(std::make_shared<ArkttsARWeightsRuntime>(
              std::move(assets),
              backend_config,
              threads,
              graph_arena_bytes,
              weight_context_bytes,
              weight_storage_type)),
          sampling_policy_(sampling::resolve_torch_cuda_sampling_policy(
              runtime_->backend_type(),
              backend_config.device,
              "audio8_tts.ar.cuda_sampling_policy",
              "Audio8 TTS",
              sampling::TorchCudaSamplingPolicyFailureMode::FallbackToDefault)) {}

    ~Impl() {
        step_graph_.reset();
        prefill_graph_.reset();
        fast_graph_.reset();
        runtime_.reset();
    }

    Audio8TtsCodes generate(const Audio8TtsPrompt & prompt, const Audio8TtsGenerationOptions & options) {
        ArkttsARProfile profile;
        const auto & assets = runtime_->assets();
        const auto & weights = runtime_->weights();
        const bool is_falcon = assets.config.text.slow_backbone == "falcon_h1" ||
                               assets.model_weights->has_tensor("slow.embed_tokens.weight");
        if (is_falcon) {
            // Falcon-H1 0.1B — native ggml path (see docs/FALCON_H1_0.1B_PORT_PLAN.md).
            // Current limitation (drawback stub): falcon_forward_stateless is a
            // simplified forward that implements RMSNorm + Mamba in_proj split
            // (gate/xBC) + conv bias SiLU + gated out_proj + FFN, but stubs the
            // SSM core (no ggml_ssm_conv / B/C / dt / A / D / ggml_ssm_scan /
            // recurrent conv/ssm state, no hybrid attention). It recomputes the
            // full sequence each step O(N^2) and only applies ssm_out/lm_head
            // multipliers. This produces prompt-invariant logits and fails STT
            // without the full Mamba2 port (see mamba-base.cpp:151,
            // falcon-h1.cpp:132). The full port is tracked in the plan file and
            // reuses vendored external/ggml ssm backends (cpu/cuda/metal/vulkan)
            // — no Python dependency, no /tmp or system() calls.
            if (prompt.codebook_rows != assets.config.fast.num_codebooks + 1 ||
                static_cast<int64_t>(prompt.matrix.size()) != prompt.codebook_rows * prompt.steps) {
                throw std::runtime_error("Audio8 TTS AR prompt shape mismatch");
            }
            const int64_t max_new_tokens = std::min(options.max_new_tokens, assets.config.text.max_seq_len - prompt.steps);
            if (max_new_tokens <= 0) throw std::runtime_error("Audio8 TTS prompt leaves no room for generated tokens");
            ensure_fast_graph(profile);
            SampleState sample;
            sample.seed = options.seed;
            sample.rng.seed(options.seed);
            sample.previous_main.assign(static_cast<size_t>(kRasWindow), 0);
            std::vector<int32_t> full_matrix = prompt.matrix;
            int64_t cur_steps = prompt.steps;
            auto expand_compact = [&](const std::vector<float> & compact) {
                const int64_t vocab = assets.config.text.vocab_size;
                std::vector<float> full(static_cast<size_t>(vocab), -std::numeric_limits<float>::infinity());
                const int64_t codebook_size = assets.config.fast.vocab_size;
                for (int64_t i = 0; i < codebook_size && i < static_cast<int64_t>(compact.size()); ++i) {
                    int64_t dst = assets.config.semantic_start_token_id + i;
                    if (dst >= 0 && dst < vocab) full[static_cast<size_t>(dst)] = compact[static_cast<size_t>(i)];
                }
                if (codebook_size < static_cast<int64_t>(compact.size())) {
                    int64_t eos = assets.config.im_end_token_id;
                    if (eos >= 0 && eos < vocab) full[static_cast<size_t>(eos)] = compact[static_cast<size_t>(codebook_size)];
                }
                return full;
            };
            auto pre_emb = build_falcon_embeddings(assets.config, weights, full_matrix.data(), cur_steps);
            auto pre_out = falcon_forward_stateless(runtime_->backend(), runtime_->threads(), runtime_->graph_arena_bytes(), assets.config, weights, pre_emb, cur_steps);
            auto pre_logits_full = expand_compact(pre_out.logits);
            auto frame = sample_frame(pre_logits_full, pre_out.hidden, options, sample, false, profile);
            if (frame.front() == im_end_id()) {
                log_profile(profile);
                return Audio8TtsCodes{{}, assets.config.fast.num_codebooks, 0};
            }
            std::vector<int32_t> generated_frame_major;
            generated_frame_major.reserve(static_cast<size_t>(max_new_tokens * assets.config.fast.num_codebooks));
            for (size_t i = 1; i < frame.size(); ++i) generated_frame_major.push_back(frame[i]);
            ++profile.generated_frames;
            {
                std::vector<int32_t> new_mat(static_cast<size_t>((cur_steps + 1) * (assets.config.fast.num_codebooks + 1)), 0);
                for (int64_t r = 0; r < assets.config.fast.num_codebooks + 1; ++r) {
                    for (int64_t s = 0; s < cur_steps; ++s) new_mat[static_cast<size_t>(r * (cur_steps + 1) + s)] = full_matrix[static_cast<size_t>(r * cur_steps + s)];
                    new_mat[static_cast<size_t>(r * (cur_steps + 1) + cur_steps)] = frame[static_cast<size_t>(r)];
                }
                full_matrix.swap(new_mat);
                cur_steps += 1;
            }
            bool ended_by_im_end = false;
            for (int64_t step = 1; step < max_new_tokens; ++step) {
                auto emb = build_falcon_embeddings(assets.config, weights, full_matrix.data(), cur_steps);
                auto out = falcon_forward_stateless(runtime_->backend(), runtime_->threads(), runtime_->graph_arena_bytes(), assets.config, weights, emb, cur_steps);
                auto logits_full = expand_compact(out.logits);
                auto next_frame = sample_frame(logits_full, out.hidden, options, sample, true, profile);
                if (next_frame.front() == im_end_id()) { ended_by_im_end = true; break; }
                for (size_t i = 1; i < next_frame.size(); ++i) generated_frame_major.push_back(next_frame[i]);
                ++profile.generated_frames;
                std::vector<int32_t> new_mat(static_cast<size_t>((cur_steps + 1) * (assets.config.fast.num_codebooks + 1)), 0);
                for (int64_t r = 0; r < assets.config.fast.num_codebooks + 1; ++r) {
                    for (int64_t s = 0; s < cur_steps; ++s) new_mat[static_cast<size_t>(r * (cur_steps + 1) + s)] = full_matrix[static_cast<size_t>(r * cur_steps + s)];
                    new_mat[static_cast<size_t>(r * (cur_steps + 1) + cur_steps)] = next_frame[static_cast<size_t>(r)];
                }
                full_matrix.swap(new_mat);
                cur_steps += 1;
            }
            if (!ended_by_im_end && !generated_frame_major.empty()) {
                generated_frame_major.resize(generated_frame_major.size() - static_cast<size_t>(assets.config.fast.num_codebooks));
                --profile.generated_frames;
            }
            Audio8TtsCodes out;
            out.codebooks = assets.config.fast.num_codebooks;
            out.frames = static_cast<int64_t>(generated_frame_major.size()) / out.codebooks;
            out.codes.assign(static_cast<size_t>(out.codebooks * out.frames), 0);
            for (int64_t f = 0; f < out.frames; ++f) for (int64_t cb = 0; cb < out.codebooks; ++cb) out.codes[static_cast<size_t>(cb * out.frames + f)] = generated_frame_major[static_cast<size_t>(f * out.codebooks + cb)];
            log_profile(profile);
            return out;
        }
        if (prompt.codebook_rows != assets.config.fast.num_codebooks + 1 ||
            static_cast<int64_t>(prompt.matrix.size()) != prompt.codebook_rows * prompt.steps) {
            throw std::runtime_error("Audio8 TTS AR prompt shape mismatch");
        }
        const int64_t max_new_tokens = std::min(options.max_new_tokens, assets.config.text.max_seq_len - prompt.steps);
        if (max_new_tokens <= 0) {
            throw std::runtime_error("Audio8 TTS prompt leaves no room for generated tokens");
        }
        // Prefill writes into the reusable step KV cache; rebuild the copy graph for each request.
        prefill_graph_.reset();
        ensure_step_graph(prompt.steps + max_new_tokens, profile);
        ensure_prefill_graph(prompt.steps, profile);
        ensure_fast_graph(profile);
        SampleState sample;
        sample.seed = options.seed;
        sample.rng.seed(options.seed);
        sample.previous_main.assign(static_cast<size_t>(kRasWindow), 0);
        auto timing_start = Clock::now();
        auto embeddings = build_slow_embeddings(assets.config, weights, prompt.matrix.data(), prompt.steps);
        profile.slow_embedding_ms += engine::debug::elapsed_ms(timing_start, Clock::now());
        auto prefill = prefill_graph_->run(embeddings, profile);
        std::vector<int32_t> generated_frame_major;
        generated_frame_major.reserve(static_cast<size_t>(max_new_tokens * assets.config.fast.num_codebooks));
        auto frame = sample_frame(prefill.forward.logits, prefill.forward.hidden, options, sample, false, profile);
        if (frame.front() == im_end_id()) {
            log_profile(profile);
            return Audio8TtsCodes{{}, assets.config.fast.num_codebooks, 0};
        }
        append_frame(generated_frame_major, frame);
        ++profile.generated_frames;
        step_graph_->finish_prefill(prompt.steps);
        bool ended_by_im_end = false;
        for (int64_t step = 1; step < max_new_tokens; ++step) {
            timing_start = Clock::now();
            const auto input = build_slow_embedding_for_frame(assets.config, weights, frame);
            profile.slow_embedding_ms += engine::debug::elapsed_ms(timing_start, Clock::now());
            auto step_out = step_graph_->run(input, profile);
            frame = sample_frame(step_out.logits, step_out.hidden, options, sample, true, profile);
            if (frame.front() == im_end_id()) {
                ended_by_im_end = true;
                break;
            }
            append_frame(generated_frame_major, frame);
            ++profile.generated_frames;
        }
        if (!ended_by_im_end && !generated_frame_major.empty()) {
            generated_frame_major.resize(generated_frame_major.size() - static_cast<size_t>(assets.config.fast.num_codebooks));
            --profile.generated_frames;
        }
        Audio8TtsCodes out;
        out.codebooks = assets.config.fast.num_codebooks;
        out.frames = static_cast<int64_t>(generated_frame_major.size()) / out.codebooks;
        out.codes.assign(static_cast<size_t>(out.codebooks * out.frames), 0);
        for (int64_t frame_index = 0; frame_index < out.frames; ++frame_index) {
            for (int64_t codebook = 0; codebook < out.codebooks; ++codebook) {
                out.codes[static_cast<size_t>(codebook * out.frames + frame_index)] =
                    generated_frame_major[static_cast<size_t>(frame_index * out.codebooks + codebook)];
            }
        }
        log_profile(profile);
        return out;
    }

    void release_runtime_graphs() {
        step_graph_.reset();
        prefill_graph_.reset();
        fast_graph_.reset();
    }

private:
    class PrefillGraph {
    public:
        PrefillGraph(
            std::shared_ptr<const ArkttsARWeightsRuntime> runtime,
            int64_t steps,
            ArkttsPrefillCacheTarget target_cache)
            : runtime_(std::move(runtime)),
              steps_(steps),
              target_cache_(std::move(target_cache)) {
            const auto & assets = runtime_->assets();
            const auto & config = assets.config.text;
            if (target_cache_.keys.size() != runtime_->weights().slow_layers.size() ||
                target_cache_.values.size() != runtime_->weights().slow_layers.size()) {
                throw std::runtime_error("Audio8 TTS prefill target cache layer count mismatch");
            }
            ggml_init_params params{runtime_->graph_arena_bytes(), nullptr, true};
            ctx_.reset(ggml_init(params));
            if (ctx_ == nullptr) {
                throw std::runtime_error("failed to initialize Audio8 TTS AR prefill context");
            }
            core::ModuleBuildContext ctx{ctx_.get(), "audio8_tts.ar.prefill", runtime_->backend_type()};
            auto input = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, steps_, config.dim}));
            input_ = input.tensor;
            positions_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, steps_);
            auto positions_value = core::wrap_tensor(positions_, core::TensorShape::from_dims({steps_}), GGML_TYPE_I32);
            constants_ = std::make_unique<core::ConstantTensorCache>(
                runtime_->backend(),
                runtime_->threads(),
                "audio8_tts.ar.prefill.constants",
                256ull * 1024ull * 1024ull);
            constants_->begin_graph();
            auto decoder = build_arktts_causal_decoder(
                ctx,
                *constants_,
                input,
                positions_value,
                bind_slow_weights(*constants_, runtime_->weights(), config),
                make_slow_decoder_config(config, runtime_->backend_type()),
                assets.config.norm_fastlayer_input);
            graph_ = ggml_new_graph_custom(ctx_.get(), 65536, false);
            for (size_t layer_index = 0; layer_index < decoder.state.layers.size(); ++layer_index) {
                const auto & layer = decoder.state.layers[layer_index];
                if (!layer.key.has_value() || !layer.value.has_value()) {
                    throw std::runtime_error("Audio8 TTS prefill decoder did not produce K/V state");
                }
                auto key_dest = runtime::view_transformer_kv_cache_steps(
                    ctx,
                    target_cache_.keys[layer_index],
                    0,
                    steps_,
                    config.n_local_heads,
                    config.head_dim,
                    "Audio8 TTS prefill key cache",
                    target_cache_.keys[layer_index].type);
                auto value_dest = runtime::view_transformer_kv_cache_steps(
                    ctx,
                    target_cache_.values[layer_index],
                    0,
                    steps_,
                    config.n_local_heads,
                    config.head_dim,
                    "Audio8 TTS prefill value cache",
                    target_cache_.values[layer_index].type);
                ggml_build_forward_expand(graph_, ggml_cpy(ctx_.get(), layer.key->tensor, key_dest.tensor));
                ggml_build_forward_expand(graph_, ggml_cpy(ctx_.get(), layer.value->tensor, value_dest.tensor));
            }
            hidden_ = decoder.hidden.tensor;
            logits_ = decoder.logits.tensor;
            ggml_set_output(hidden_);
            ggml_set_output(logits_);
            ggml_build_forward_expand(graph_, logits_);
            ggml_build_forward_expand(graph_, hidden_);
            constants_->finish_graph();
            constants_->ensure_uploaded();
            gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(runtime_->backend()));
            if (gallocr_ == nullptr ||
                !ggml_gallocr_reserve(gallocr_, graph_) ||
                !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
                throw std::runtime_error("failed to allocate Audio8 TTS AR prefill graph");
            }
            auto positions = modules::qwen_position_ids(steps_);
            ggml_backend_tensor_set(positions_, positions.data(), 0, positions.size() * sizeof(int32_t));
        }

        ~PrefillGraph() {
            core::release_backend_graph_resources(runtime_->backend(), graph_);
            if (gallocr_ != nullptr) {
                ggml_gallocr_free(gallocr_);
            }
        }

        SlowPrefillOutput run(const std::vector<float> & embeddings, ArkttsARProfile & profile) {
            const auto & config = runtime_->assets().config.text;
            if (static_cast<int64_t>(embeddings.size()) != steps_ * config.dim) {
                throw std::runtime_error("Audio8 TTS prefill embedding size mismatch");
            }
            ++profile.prefill_runs;
            auto timing_start = Clock::now();
            ggml_backend_tensor_set(input_, embeddings.data(), 0, embeddings.size() * sizeof(float));
            profile.prefill_input_upload_ms += engine::debug::elapsed_ms(timing_start, Clock::now());
            core::set_backend_threads(runtime_->backend(), runtime_->threads());
            timing_start = Clock::now();
            const ggml_status status = core::compute_backend_graph(runtime_->backend(), graph_, nullptr, "audio8_tts.ar.prefill");
            ggml_backend_synchronize(runtime_->backend());
            profile.prefill_graph_ms += engine::debug::elapsed_ms(timing_start, Clock::now());
            if (status != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("Audio8 TTS AR prefill graph compute failed");
            }
            SlowPrefillOutput out;
            out.forward.logits.resize(static_cast<size_t>(config.vocab_size));
            out.forward.hidden.resize(static_cast<size_t>(config.dim));
            timing_start = Clock::now();
            ggml_backend_tensor_get(logits_, out.forward.logits.data(), 0, out.forward.logits.size() * sizeof(float));
            ggml_backend_tensor_get(hidden_, out.forward.hidden.data(), 0, out.forward.hidden.size() * sizeof(float));
            profile.prefill_output_read_ms += engine::debug::elapsed_ms(timing_start, Clock::now());
            return out;
        }

        int64_t steps() const noexcept { return steps_; }

    private:
        std::shared_ptr<const ArkttsARWeightsRuntime> runtime_;
        int64_t steps_ = 0;
        std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
        ggml_tensor * input_ = nullptr;
        ggml_tensor * positions_ = nullptr;
        ggml_tensor * hidden_ = nullptr;
        ggml_tensor * logits_ = nullptr;
        ggml_cgraph * graph_ = nullptr;
        ggml_gallocr_t gallocr_ = nullptr;
        std::unique_ptr<core::ConstantTensorCache> constants_;
        ArkttsPrefillCacheTarget target_cache_;
    };

    class StepGraph {
    public:
        StepGraph(std::shared_ptr<const ArkttsARWeightsRuntime> runtime, int64_t cache_steps)
            : runtime_(std::move(runtime)),
              cache_steps_(cache_steps) {
            ggml_init_params state_params{8ull * 1024ull * 1024ull, nullptr, true};
            state_ctx_.reset(ggml_init(state_params));
            if (state_ctx_ == nullptr) {
                throw std::runtime_error("failed to initialize Audio8 TTS AR step state context");
            }
            ggml_init_params graph_params{runtime_->graph_arena_bytes(), nullptr, true};
            graph_ctx_.reset(ggml_init(graph_params));
            if (graph_ctx_ == nullptr) {
                throw std::runtime_error("failed to initialize Audio8 TTS AR step context");
            }
            const auto & assets = runtime_->assets();
            const auto & config = assets.config.text;
            input_ = ggml_new_tensor_3d(state_ctx_.get(), GGML_TYPE_F32, config.dim, 1, 1);
            position_ = ggml_new_tensor_1d(state_ctx_.get(), GGML_TYPE_I32, 1);
            cache_slot_ = ggml_new_tensor_1d(state_ctx_.get(), GGML_TYPE_I32, 1);
            mask_ = ggml_new_tensor_4d(state_ctx_.get(), GGML_TYPE_F16, cache_steps_, 1, 1, 1);
            std::vector<core::TensorValue> cache_keys;
            std::vector<core::TensorValue> cache_values;
            cache_keys.reserve(runtime_->weights().slow_layers.size());
            cache_values.reserve(runtime_->weights().slow_layers.size());
            const ggml_type cache_type =
                runtime_->backend_type() == core::BackendType::Vulkan ? GGML_TYPE_F32 : GGML_TYPE_BF16;
            for (size_t layer = 0; layer < runtime_->weights().slow_layers.size(); ++layer) {
                cache_keys.push_back(core::wrap_tensor(
                    ggml_new_tensor_4d(
                        state_ctx_.get(),
                        cache_type,
                        config.head_dim,
                        config.n_local_heads,
                        cache_steps_,
                        1),
                    core::TensorShape::from_dims({1, cache_steps_, config.n_local_heads, config.head_dim}),
                    cache_type));
                cache_values.push_back(core::wrap_tensor(
                    ggml_new_tensor_4d(
                        state_ctx_.get(),
                        cache_type,
                        config.head_dim,
                        config.n_local_heads,
                        cache_steps_,
                        1),
                    core::TensorShape::from_dims({1, cache_steps_, config.n_local_heads, config.head_dim}),
                    cache_type));
            }
            state_buffer_ = ggml_backend_alloc_ctx_tensors(state_ctx_.get(), runtime_->backend());
            if (state_buffer_ == nullptr) {
                throw std::runtime_error("failed to allocate Audio8 TTS AR step state tensors");
            }

            core::ModuleBuildContext ctx{graph_ctx_.get(), "audio8_tts.ar.step", runtime_->backend_type()};
            auto input = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 1, config.dim}));
            input = core::wrap_tensor(ggml_cpy(ctx.ggml, input_, input.tensor), input.shape, input.type);
            auto position_value = core::wrap_tensor(position_, core::TensorShape::from_dims({1}), GGML_TYPE_I32);
            auto cache_slot_value = core::wrap_tensor(cache_slot_, core::TensorShape::from_dims({1}), GGML_TYPE_I32);
            auto mask_value = core::wrap_tensor(mask_, core::TensorShape::from_dims({1, 1, 1, cache_steps_}), GGML_TYPE_F16);
            graph_ = ggml_new_graph_custom(graph_ctx_.get(), 65536, false);
            auto & constants = runtime_->slow_step_constants();
            constants.begin_graph();
            auto decoder = build_arktts_static_decoder(
                ctx,
                graph_,
                input,
                position_value,
                bind_slow_weights(constants, runtime_->weights(), config),
                make_slow_decoder_config(config, runtime_->backend_type()),
                cache_steps_,
                mask_value,
                cache_slot_value,
                std::move(cache_keys),
                std::move(cache_values),
                assets.config.norm_fastlayer_input);
            cache_ = std::move(decoder.cache);
            hidden_ = decoder.hidden.tensor;
            logits_ = decoder.logits.tensor;
            ggml_set_output(hidden_);
            ggml_set_output(logits_);
            ggml_build_forward_expand(graph_, logits_);
            ggml_build_forward_expand(graph_, hidden_);
            constants.finish_graph();
            constants.ensure_uploaded();
            gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(runtime_->backend()));
            if (gallocr_ == nullptr ||
                !ggml_gallocr_reserve(gallocr_, graph_) ||
                !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
                throw std::runtime_error("failed to allocate Audio8 TTS AR step tensors");
            }
            mask_scratch_.assign(static_cast<size_t>(cache_steps_), ggml_fp32_to_fp16(-INFINITY));
        }

        ~StepGraph() {
            core::release_backend_graph_resources(runtime_->backend(), graph_);
            if (gallocr_ != nullptr) {
                ggml_gallocr_free(gallocr_);
            }
            if (state_buffer_ != nullptr) {
                ggml_backend_buffer_free(state_buffer_);
            }
        }

        int64_t cache_steps() const noexcept { return cache_steps_; }

        ArkttsPrefillCacheTarget prefill_target_cache() const {
            ArkttsPrefillCacheTarget out;
            out.keys.reserve(runtime_->weights().slow_layers.size());
            out.values.reserve(runtime_->weights().slow_layers.size());
            for (size_t layer = 0; layer < runtime_->weights().slow_layers.size(); ++layer) {
                out.keys.push_back(cache_.key_tensor(layer));
                out.values.push_back(cache_.value_tensor(layer));
            }
            return out;
        }

        void finish_prefill(int64_t steps) {
            cache_.retain_prefix(0);
            cache_.advance_after_direct_append(steps);
            const auto masked = ggml_fp32_to_fp16(-INFINITY);
            const auto visible = ggml_fp32_to_fp16(0.0F);
            std::fill(mask_scratch_.begin(), mask_scratch_.end(), masked);
            for (int64_t i = 0; i < cache_.valid_steps(); ++i) {
                mask_scratch_[static_cast<size_t>(i)] = visible;
            }
            ggml_backend_tensor_set(mask_, mask_scratch_.data(), 0, mask_scratch_.size() * sizeof(ggml_fp16_t));
        }

        SlowForwardOutput run(const std::vector<float> & embedding, ArkttsARProfile & profile) {
            const auto & config = runtime_->assets().config.text;
            if (static_cast<int64_t>(embedding.size()) != config.dim) {
                throw std::runtime_error("Audio8 TTS step embedding size mismatch");
            }
            if (cache_.valid_steps() >= cache_steps_) {
                throw std::runtime_error("Audio8 TTS step cache exceeds capacity");
            }
            ++profile.step_runs;
            auto timing_start = Clock::now();
            const int32_t pos = static_cast<int32_t>(cache_.current_end());
            ggml_backend_tensor_set(position_, &pos, 0, sizeof(pos));
            const int32_t cache_slot = static_cast<int32_t>(cache_.valid_steps());
            ggml_backend_tensor_set(cache_slot_, &cache_slot, 0, sizeof(cache_slot));
            const auto visible = ggml_fp32_to_fp16(0.0F);
            mask_scratch_[static_cast<size_t>(cache_.valid_steps())] = visible;
            ggml_backend_tensor_set(
                mask_,
                &visible,
                static_cast<size_t>(cache_.valid_steps()) * sizeof(ggml_fp16_t),
                sizeof(ggml_fp16_t));
            profile.step_mask_upload_ms += engine::debug::elapsed_ms(timing_start, Clock::now());
            timing_start = Clock::now();
            ggml_backend_tensor_set(input_, embedding.data(), 0, embedding.size() * sizeof(float));
            profile.step_input_upload_ms += engine::debug::elapsed_ms(timing_start, Clock::now());
            core::set_backend_threads(runtime_->backend(), runtime_->threads());
            timing_start = Clock::now();
            const ggml_status status = core::compute_backend_graph(runtime_->backend(), graph_, nullptr, "audio8_tts.ar.step");
            ggml_backend_synchronize(runtime_->backend());
            profile.step_graph_ms += engine::debug::elapsed_ms(timing_start, Clock::now());
            if (status != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("Audio8 TTS AR step graph compute failed");
            }
            cache_.advance_after_direct_append(1);
            SlowForwardOutput out;
            out.logits.resize(static_cast<size_t>(config.vocab_size));
            out.hidden.resize(static_cast<size_t>(config.dim));
            timing_start = Clock::now();
            ggml_backend_tensor_get(logits_, out.logits.data(), 0, out.logits.size() * sizeof(float));
            ggml_backend_tensor_get(hidden_, out.hidden.data(), 0, out.hidden.size() * sizeof(float));
            profile.step_output_read_ms += engine::debug::elapsed_ms(timing_start, Clock::now());
            return out;
        }

    private:
        std::shared_ptr<const ArkttsARWeightsRuntime> runtime_;
        int64_t cache_steps_ = 0;
        std::unique_ptr<ggml_context, GgmlContextDeleter> state_ctx_;
        std::unique_ptr<ggml_context, GgmlContextDeleter> graph_ctx_;
        ggml_tensor * input_ = nullptr;
        ggml_tensor * position_ = nullptr;
        ggml_tensor * cache_slot_ = nullptr;
        ggml_tensor * mask_ = nullptr;
        ggml_tensor * hidden_ = nullptr;
        ggml_tensor * logits_ = nullptr;
        runtime::TransformerKVCache cache_;
        std::vector<ggml_fp16_t> mask_scratch_;
        ggml_cgraph * graph_ = nullptr;
        ggml_gallocr_t gallocr_ = nullptr;
        ggml_backend_buffer_t state_buffer_ = nullptr;
    };

    class FastGraph {
    public:
        explicit FastGraph(std::shared_ptr<const ArkttsARWeightsRuntime> runtime)
            : runtime_(std::move(runtime)) {
            ggml_init_params state_params{8ull * 1024ull * 1024ull, nullptr, true};
            state_ctx_.reset(ggml_init(state_params));
            if (state_ctx_ == nullptr) {
                throw std::runtime_error("failed to initialize Audio8 TTS fast AR state context");
            }
            ggml_init_params graph_params{runtime_->graph_arena_bytes(), nullptr, true};
            graph_ctx_.reset(ggml_init(graph_params));
            if (graph_ctx_ == nullptr) {
                throw std::runtime_error("failed to initialize Audio8 TTS fast AR context");
            }
            const auto & config = runtime_->assets().config.fast;
            const auto & weights = runtime_->weights();
            input_ = ggml_new_tensor_3d(state_ctx_.get(), GGML_TYPE_F32, config.dim, 1, 1);
            position_ = ggml_new_tensor_1d(state_ctx_.get(), GGML_TYPE_I32, 1);
            mask_ = ggml_new_tensor_4d(state_ctx_.get(), GGML_TYPE_F16, config.num_codebooks, 1, 1, 1);
            std::vector<core::TensorValue> cache_keys;
            std::vector<core::TensorValue> cache_values;
            cache_keys.reserve(weights.fast_layers.size());
            cache_values.reserve(weights.fast_layers.size());
            const ggml_type cache_type =
                runtime_->backend_type() == core::BackendType::Vulkan ? GGML_TYPE_F32 : GGML_TYPE_BF16;
            for (size_t layer = 0; layer < weights.fast_layers.size(); ++layer) {
                cache_keys.push_back(core::wrap_tensor(
                    ggml_new_tensor_4d(
                        state_ctx_.get(),
                        cache_type,
                        config.head_dim,
                        config.n_local_heads,
                        config.num_codebooks,
                        1),
                    core::TensorShape::from_dims({1, config.num_codebooks, config.n_local_heads, config.head_dim}),
                    cache_type));
                cache_values.push_back(core::wrap_tensor(
                    ggml_new_tensor_4d(
                        state_ctx_.get(),
                        cache_type,
                        config.head_dim,
                        config.n_local_heads,
                        config.num_codebooks,
                        1),
                    core::TensorShape::from_dims({1, config.num_codebooks, config.n_local_heads, config.head_dim}),
                    cache_type));
            }
            state_buffer_ = ggml_backend_alloc_ctx_tensors(state_ctx_.get(), runtime_->backend());
            if (state_buffer_ == nullptr) {
                throw std::runtime_error("failed to allocate Audio8 TTS fast AR state tensors");
            }
            for (const auto & cache : cache_keys) {
                std::vector<uint8_t> zeros(static_cast<size_t>(ggml_nbytes(cache.tensor)), 0);
                ggml_backend_tensor_set(cache.tensor, zeros.data(), 0, zeros.size());
            }
            for (const auto & cache : cache_values) {
                std::vector<uint8_t> zeros(static_cast<size_t>(ggml_nbytes(cache.tensor)), 0);
                ggml_backend_tensor_set(cache.tensor, zeros.data(), 0, zeros.size());
            }

            core::ModuleBuildContext ctx{graph_ctx_.get(), "audio8_tts.ar.fast", runtime_->backend_type()};
            auto input = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 1, config.dim}));
            input = core::wrap_tensor(ggml_cpy(ctx.ggml, input_, input.tensor), input.shape, input.type);
            auto position_value = core::wrap_tensor(position_, core::TensorShape::from_dims({1}), GGML_TYPE_I32);
            auto mask_value = core::wrap_tensor(mask_, core::TensorShape::from_dims({1, 1, 1, config.num_codebooks}), GGML_TYPE_F16);
            graph_ = ggml_new_graph_custom(graph_ctx_.get(), 32768, false);
            auto & constants = runtime_->fast_constants();
            constants.begin_graph();
            modules::QwenCausalDecoderWeights decoder_weights;
            decoder_weights.stack.layers.reserve(weights.fast_layers.size());
            for (const auto & layer : weights.fast_layers) {
                decoder_weights.stack.layers.push_back(bind_fast_layer(constants, layer, config));
            }
            decoder_weights.final_norm = binding::norm_data(constants, weights.fast_norm);
            decoder_weights.lm_head = binding::linear_data(constants, weights.fast_output);
            auto decoder = build_arktts_static_decoder(
                ctx,
                graph_,
                input,
                position_value,
                decoder_weights,
                make_fast_decoder_config(config, runtime_->backend_type()),
                config.num_codebooks,
                mask_value,
                position_value,
                std::move(cache_keys),
                std::move(cache_values),
                true);
            logits_ = decoder.logits.tensor;
            ggml_set_output(logits_);
            ggml_build_forward_expand(graph_, logits_);
            constants.finish_graph();
            constants.ensure_uploaded();
            gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(runtime_->backend()));
            if (gallocr_ == nullptr ||
                !ggml_gallocr_reserve(gallocr_, graph_) ||
                !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
                throw std::runtime_error("failed to allocate Audio8 TTS fast AR graph");
            }
            mask_scratch_.assign(static_cast<size_t>(config.num_codebooks), ggml_fp32_to_fp16(-INFINITY));
        }

        ~FastGraph() {
            core::release_backend_graph_resources(runtime_->backend(), graph_);
            if (gallocr_ != nullptr) {
                ggml_gallocr_free(gallocr_);
            }
            if (state_buffer_ != nullptr) {
                ggml_backend_buffer_free(state_buffer_);
            }
        }

        std::vector<float> run(const std::vector<float> & input, int64_t position, ArkttsARProfile & profile) {
            const auto & config = runtime_->assets().config.fast;
            if (static_cast<int64_t>(input.size()) != config.dim) {
                throw std::runtime_error("Audio8 TTS fast AR input size mismatch");
            }
            ++profile.fast_runs;
            auto timing_start = Clock::now();
            const int32_t pos = static_cast<int32_t>(position);
            ggml_backend_tensor_set(position_, &pos, 0, sizeof(pos));
            const auto visible = ggml_fp32_to_fp16(0.0F);
            if (position == 0) {
                std::fill(mask_scratch_.begin(), mask_scratch_.end(), ggml_fp32_to_fp16(-INFINITY));
                mask_scratch_[0] = visible;
                ggml_backend_tensor_set(mask_, mask_scratch_.data(), 0, mask_scratch_.size() * sizeof(ggml_fp16_t));
            } else {
                mask_scratch_[static_cast<size_t>(position)] = visible;
                ggml_backend_tensor_set(
                    mask_,
                    &visible,
                    static_cast<size_t>(position) * sizeof(ggml_fp16_t),
                    sizeof(ggml_fp16_t));
            }
            profile.fast_mask_upload_ms += engine::debug::elapsed_ms(timing_start, Clock::now());
            timing_start = Clock::now();
            ggml_backend_tensor_set(input_, input.data(), 0, input.size() * sizeof(float));
            profile.fast_input_upload_ms += engine::debug::elapsed_ms(timing_start, Clock::now());
            core::set_backend_threads(runtime_->backend(), runtime_->threads());
            timing_start = Clock::now();
            const ggml_status status = core::compute_backend_graph(runtime_->backend(), graph_, nullptr, "audio8_tts.ar.fast");
            ggml_backend_synchronize(runtime_->backend());
            profile.fast_graph_ms += engine::debug::elapsed_ms(timing_start, Clock::now());
            if (status != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("Audio8 TTS fast AR graph compute failed");
            }
            std::vector<float> logits(static_cast<size_t>(config.vocab_size), 0.0F);
            timing_start = Clock::now();
            ggml_backend_tensor_get(logits_, logits.data(), 0, logits.size() * sizeof(float));
            profile.fast_output_read_ms += engine::debug::elapsed_ms(timing_start, Clock::now());
            return logits;
        }

    private:
        std::shared_ptr<const ArkttsARWeightsRuntime> runtime_;
        std::unique_ptr<ggml_context, GgmlContextDeleter> state_ctx_;
        std::unique_ptr<ggml_context, GgmlContextDeleter> graph_ctx_;
        ggml_tensor * input_ = nullptr;
        ggml_tensor * position_ = nullptr;
        ggml_tensor * mask_ = nullptr;
        ggml_tensor * logits_ = nullptr;
        std::vector<ggml_fp16_t> mask_scratch_;
        ggml_cgraph * graph_ = nullptr;
        ggml_gallocr_t gallocr_ = nullptr;
        ggml_backend_buffer_t state_buffer_ = nullptr;
    };

    void ensure_prefill_graph(int64_t steps, ArkttsARProfile & profile) {
        if (!prefill_graph_ || prefill_graph_->steps() != steps) {
            const auto build_start = Clock::now();
            if (!step_graph_) {
                throw std::runtime_error("Audio8 TTS AR prefill requires a step graph");
            }
            prefill_graph_ = std::make_unique<PrefillGraph>(runtime_, steps, step_graph_->prefill_target_cache());
            profile.graph_build_prefill_ms += engine::debug::elapsed_ms(build_start, Clock::now());
        }
    }

    void ensure_step_graph(int64_t cache_steps, ArkttsARProfile & profile) {
        if (!step_graph_ || step_graph_->cache_steps() < cache_steps) {
            const auto build_start = Clock::now();
            step_graph_ = std::make_unique<StepGraph>(runtime_, cache_steps);
            prefill_graph_.reset();
            profile.graph_build_step_ms += engine::debug::elapsed_ms(build_start, Clock::now());
        }
    }

    void ensure_fast_graph(ArkttsARProfile & profile) {
        if (!fast_graph_) {
            const auto build_start = Clock::now();
            fast_graph_ = std::make_unique<FastGraph>(runtime_);
            profile.graph_build_fast_ms += engine::debug::elapsed_ms(build_start, Clock::now());
        }
    }

    int32_t im_end_id() const {
        return static_cast<int32_t>(runtime_->assets().config.im_end_token_id);
    }

    void append_frame(std::vector<int32_t> & out, const std::vector<int32_t> & frame) const {
        if (static_cast<int64_t>(frame.size()) != runtime_->assets().config.fast.num_codebooks + 1) {
            throw std::runtime_error("Audio8 TTS generated frame shape mismatch");
        }
        out.insert(out.end(), frame.begin() + 1, frame.end());
    }

    std::vector<int32_t> sample_frame(
        const std::vector<float> & slow_logits,
        const std::vector<float> & slow_hidden,
        const Audio8TtsGenerationOptions & options,
        SampleState & sample,
        bool apply_ras,
        ArkttsARProfile & profile) {
        const auto & config = runtime_->assets().config;
        const auto & weights = runtime_->weights();
        auto timing_start = Clock::now();
        const auto biased = apply_semantic_bias(config, im_end_id(), slow_logits);
        profile.sample_bias_ms += engine::debug::elapsed_ms(timing_start, Clock::now());
        timing_start = Clock::now();
        int32_t main_token = sample_from_logits(
            biased,
            options.temperature,
            options.top_p,
            options.top_k,
            sample,
            sampling_policy_);
        profile.sample_main_ms += engine::debug::elapsed_ms(timing_start, Clock::now());
        timing_start = Clock::now();
        const int32_t high_token = sample_from_logits(
            biased,
            kRasHighTemperature,
            kRasHighTopP,
            options.top_k,
            sample,
            sampling_policy_);
        profile.sample_high_ms += engine::debug::elapsed_ms(timing_start, Clock::now());
        if (apply_ras && is_semantic_token(config, main_token) &&
            std::find(sample.previous_main.begin(), sample.previous_main.end(), main_token) != sample.previous_main.end()) {
            main_token = high_token;
        }
        std::rotate(sample.previous_main.begin(), sample.previous_main.begin() + 1, sample.previous_main.end());
        sample.previous_main.back() = main_token;

        std::vector<int32_t> frame(static_cast<size_t>(config.fast.num_codebooks + 1), 0);
        frame[0] = main_token;
        if (!is_semantic_token(config, main_token)) {
            return frame;
        }
        const auto fast0_logits = fast_graph_->run(slow_hidden, 0, profile);
        int32_t code = std::clamp<int32_t>(
            main_token - static_cast<int32_t>(config.semantic_start_token_id),
            0,
            static_cast<int32_t>(config.fast.vocab_size - 1));
        frame[1] = code;
        for (int64_t codebook = 1; codebook < config.fast.num_codebooks; ++codebook) {
            timing_start = Clock::now();
            const auto embedding = build_fast_embedding(config, weights, code);
            profile.fast_embedding_ms += engine::debug::elapsed_ms(timing_start, Clock::now());
            const auto logits = fast_graph_->run(embedding, codebook, profile);
            timing_start = Clock::now();
            code = sample_from_logits(
                logits,
                options.temperature,
                options.top_p,
                options.top_k,
                sample,
                sampling_policy_);
            profile.sample_fast_ms += engine::debug::elapsed_ms(timing_start, Clock::now());
            frame[static_cast<size_t>(codebook + 1)] = code;
        }
        if (engine::debug::trace_log_enabled()) {
            engine::debug::trace_log_i32(
                "audio8_tts.ar.frame", {1, static_cast<int64_t>(frame.size())}, frame);
        }
        return frame;
    }

    void log_profile(const ArkttsARProfile & profile) const {
        engine::debug::timing_log_scalar("audio8_tts.ar.profile.graph_build_prefill_ms", profile.graph_build_prefill_ms);
        engine::debug::timing_log_scalar("audio8_tts.ar.profile.graph_build_step_ms", profile.graph_build_step_ms);
        engine::debug::timing_log_scalar("audio8_tts.ar.profile.graph_build_fast_ms", profile.graph_build_fast_ms);
        engine::debug::timing_log_scalar("audio8_tts.ar.profile.slow_embedding_ms", profile.slow_embedding_ms);
        engine::debug::timing_log_scalar("audio8_tts.ar.profile.fast_embedding_ms", profile.fast_embedding_ms);
        engine::debug::timing_log_scalar("audio8_tts.ar.profile.prefill_input_upload_ms", profile.prefill_input_upload_ms);
        engine::debug::timing_log_scalar("audio8_tts.ar.profile.prefill_graph_ms", profile.prefill_graph_ms);
        engine::debug::timing_log_scalar("audio8_tts.ar.profile.prefill_output_read_ms", profile.prefill_output_read_ms);
        engine::debug::timing_log_scalar("audio8_tts.ar.profile.step_input_upload_ms", profile.step_input_upload_ms);
        engine::debug::timing_log_scalar("audio8_tts.ar.profile.step_mask_upload_ms", profile.step_mask_upload_ms);
        engine::debug::timing_log_scalar("audio8_tts.ar.profile.step_graph_ms", profile.step_graph_ms);
        engine::debug::timing_log_scalar("audio8_tts.ar.profile.step_output_read_ms", profile.step_output_read_ms);
        engine::debug::timing_log_scalar("audio8_tts.ar.profile.fast_input_upload_ms", profile.fast_input_upload_ms);
        engine::debug::timing_log_scalar("audio8_tts.ar.profile.fast_mask_upload_ms", profile.fast_mask_upload_ms);
        engine::debug::timing_log_scalar("audio8_tts.ar.profile.fast_graph_ms", profile.fast_graph_ms);
        engine::debug::timing_log_scalar("audio8_tts.ar.profile.fast_output_read_ms", profile.fast_output_read_ms);
        engine::debug::timing_log_scalar("audio8_tts.ar.profile.sample_bias_ms", profile.sample_bias_ms);
        engine::debug::timing_log_scalar("audio8_tts.ar.profile.sample_main_ms", profile.sample_main_ms);
        engine::debug::timing_log_scalar("audio8_tts.ar.profile.sample_high_ms", profile.sample_high_ms);
        engine::debug::timing_log_scalar("audio8_tts.ar.profile.sample_fast_ms", profile.sample_fast_ms);
        engine::debug::trace_log_scalar("audio8_tts.ar.profile.prefill_runs", profile.prefill_runs);
        engine::debug::trace_log_scalar("audio8_tts.ar.profile.step_runs", profile.step_runs);
        engine::debug::trace_log_scalar("audio8_tts.ar.profile.fast_runs", profile.fast_runs);
        engine::debug::trace_log_scalar("audio8_tts.ar.profile.generated_frames", profile.generated_frames);
    }

    std::shared_ptr<const ArkttsARWeightsRuntime> runtime_;
    sampling::TorchCudaSamplingPolicy sampling_policy_;
    std::unique_ptr<PrefillGraph> prefill_graph_;
    std::unique_ptr<StepGraph> step_graph_;
    std::unique_ptr<FastGraph> fast_graph_;
};

Audio8TtsARRuntime::Audio8TtsARRuntime(
    std::shared_ptr<const Audio8TtsAssets> assets,
    core::BackendConfig backend,
    int threads,
    size_t graph_arena_bytes,
    size_t weight_context_bytes,
    assets::TensorStorageType weight_storage_type)
    : impl_(std::make_unique<Impl>(
          std::move(assets),
          backend,
          threads,
          graph_arena_bytes,
          weight_context_bytes,
          weight_storage_type)) {}

Audio8TtsARRuntime::~Audio8TtsARRuntime() = default;

Audio8TtsCodes Audio8TtsARRuntime::generate(
    const Audio8TtsPrompt & prompt,
    const Audio8TtsGenerationOptions & options) {
    return impl_->generate(prompt, options);
}

void Audio8TtsARRuntime::release_runtime_graphs() {
    impl_->release_runtime_graphs();
}

}  // namespace engine::models::audio8_tts
