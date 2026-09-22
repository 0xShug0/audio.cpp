#include "engine/community_models/audio8_tts/ar.h"

#include "falcon_kv_cache.h"

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
    double falcon_step_init_ms = 0.0;
    double falcon_step_build_ms = 0.0;
    double falcon_step_gallocr_ms = 0.0;
    double falcon_step_upload_ms = 0.0;
    double falcon_step_compute_ms = 0.0;
    double falcon_step_download_ms = 0.0;
    int64_t falcon_step_runs = 0;
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
    assets::TensorDataF32 conv1d_kernel;             // host [d_conv, conv_dim] GGUF layout for ggml ssm_conv
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
        // ssm_conv Metal pipeline requires contiguous F32 conv weights.
        auto meta = source.require_metadata(prefix + ".mamba.conv1d.weight");
        w.ssm_conv1d = store.load_tensor(source, prefix + ".mamba.conv1d.weight", assets::TensorStorageType::F32, meta.shape);
        {
            // ggml ssm_conv computes y[c] = sum_k w[k,c]*window[k,c] with window[0]
            // the OLDEST frame. HF (both nn.Conv1d prefill and the cached
            // torch.sum(conv_states * w, dim=-1) decode) uses the identical
            // orientation: w[...,0] multiplies the oldest frame. The GGUF tensor
            // is the HF [conv_dim,1,d_conv] weight with reversed dims
            // [d_conv,1,conv_dim] and unchanged flat bytes, i.e.
            // raw[k + d_conv*c] == hf_w[c,k] — exactly what ssm_conv wants.
            // Feed it through UNFLIPPED (an earlier kernel flip here reversed the
            // tap order and corrupted the x/B/C split on every step).
            auto raw = source.require_f32_tensor(prefix + ".mamba.conv1d.weight");
            const int64_t d_conv = raw.shape.dims[0];
            const int64_t conv_dim = raw.shape.dims[2];
            w.conv1d_kernel.shape = core::TensorShape::from_dims({d_conv, conv_dim});
            w.conv1d_kernel.values = raw.values;
        }
    }
    w.ssm_conv1d_b = source.require_f32_tensor(prefix + ".mamba.conv1d.bias");
    // Per-head Mamba params must stay unquantized (Native): they are consumed as
    // raw F32 scalars by the SSM path (A = -exp(A_log), D, dt bias).
    {
        auto meta = source.require_metadata(prefix + ".mamba.dt_bias");
        w.ssm_dt_b = store.load_tensor(source, prefix + ".mamba.dt_bias", assets::TensorStorageType::F32, meta.shape);
    }
    {
        auto meta = source.require_metadata(prefix + ".mamba.A_log");
        w.ssm_A = store.load_tensor(source, prefix + ".mamba.A_log", assets::TensorStorageType::F32, meta.shape);
    }
    {
        auto meta = source.require_metadata(prefix + ".mamba.D");
        w.ssm_D = store.load_tensor(source, prefix + ".mamba.D", assets::TensorStorageType::F32, meta.shape);
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
        if (is_semantic_token(config, token)) {
            for (int64_t codebook = 0; codebook < config.fast.num_codebooks; ++codebook) {
                const int32_t code = matrix[(codebook + 1) * steps + step];
                add_row(weights.codebook_embedding_host, codebook * config.fast.vocab_size + code, hidden, row);
            }
        }
        for (auto & v : row) v *= config.text.embedding_multiplier;
        std::copy(row.begin(), row.end(), out.begin() + static_cast<std::ptrdiff_t>(step * hidden));
    }
    return out;
}

// ============================================================================
// Falcon-H1 (Mamba2 + hybrid GQA attention) stateful single-token forward.
// Replaces the documented stub (TODO(Falcon-H1)) with a full Mamba2 port
// mirroring transformers.models.falcon_h1 FalconH1DecoderLayer and
// llama.cpp mamba-base.cpp build_mamba2_layer. Verified shapes against
// Audio8-TTS-Preview-0.1b (dim 512, d_ssm 768, d_state 64, d_conv 4,
// mamba heads 24 x head 32, GQA 8/2 x 64, RoPE NEOX base 1e11).
// ============================================================================

// One baked Falcon-H1 step graph (zero-copy path), valid for a fixed KV
// capacity bucket. All weights and recurrent state are bound as external views
// of host memory owned by the weights/state structs, so a plan can be reused
// for every step whose sequence fits the bucket.
struct FalconStepPlan {
    int64_t cap = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx;
    ggml_cgraph * gf = nullptr;  // owned by ctx
    ggml_gallocr_t gallocr = nullptr;
    ggml_tensor * logits_out = nullptr;
    ggml_tensor * hidden_out = nullptr;
    ggml_backend_t backend = nullptr;  // borrowed; the runtime outlives any generation
    ~FalconStepPlan() {
        if (backend != nullptr && gf != nullptr) {
            core::release_backend_graph_resources(backend, gf);
        }
        if (gallocr != nullptr) {
            ggml_gallocr_free(gallocr);
        }
    }
    FalconStepPlan() = default;
    FalconStepPlan(const FalconStepPlan &) = delete;
    FalconStepPlan & operator=(const FalconStepPlan &) = delete;
};

struct FalconH1StepState {
    int64_t n_layer = 0;
    int64_t d_inner = 0;        // mamba_d_ssm (768)
    int64_t d_state = 0;        // mamba_d_state (64)
    int64_t d_conv = 0;         // mamba_d_conv (4)
    int64_t n_groups = 0;       // mamba_n_groups (1)
    int64_t n_mamba_heads = 0;  // mamba_n_heads (24)
    int64_t conv_dim = 0;       // d_inner + 2*ng*d_state (896)
    int64_t kv_dim = 0;         // n_local_heads * head_dim (128)
    std::vector<std::vector<float>> conv_states;  // [layer][(d_conv-1)*conv_dim]
    std::vector<std::vector<float>> ssm_states;   // [layer][d_state*d_inner]
    std::vector<std::vector<float>> k_cache;      // [layer][seq*kv_dim]
    std::vector<std::vector<float>> v_cache;      // [layer][seq*kv_dim]
    int64_t seq_len = 0;
    // Padded KV caches for the zero-copy path: flat [head_dim, kv_cap, n_kv]
    // (head stride = head_dim*kv_cap), grown geometrically so each step's graph
    // can write the new token's k/v in-place and flash-attend over a strided
    // prefix view — no per-step host upload, read-back, or concat copy. The
    // exact-size k_cache/v_cache vectors above are only used on the fallback
    // (non-host backend) path.
    std::vector<std::vector<float>> k_pad;
    std::vector<std::vector<float>> v_pad;
    int64_t kv_cap = 0;
    // Zero-copy constant cache (filled once per generation on host backends):
    // A = -exp(A_log) per mamba head, D expanded per channel, plus fallback
    // buffers for absent norm/bias weights and the scalar graph inputs.
    std::vector<std::vector<float>> pre_A;        // [layer][n_mamba_heads]
    std::vector<std::vector<float>> pre_D;        // [layer][d_inner]
    std::vector<float> ones_dim;                  // [dim]
    std::vector<float> zeros_conv;                // [conv_dim]
    std::vector<float> zeros_conv_kernel;         // [d_conv*conv_dim]
    int32_t ids_value = 0;
    int32_t pos_value = 0;
    bool constants_ready = false;
    // Bucketed plan reuse (zero-copy path): staging buffer for the per-step
    // embedding (the graph input tensor is baked to its address), the flash
    // mask scratch (slot seq is the last visible position), the dynamic
    // set_rows slot index, and the baked graph itself.
    std::vector<float> emb_stage;          // [dim]
    std::vector<ggml_fp16_t> kv_mask;      // [kv_cap]
    int32_t kv_slot = 0;
    std::unique_ptr<FalconStepPlan> plan;
};

FalconH1StepState init_falcon_step_state(const Audio8TtsConfig & config) {
    FalconH1StepState st;
    st.n_layer = config.text.n_layer;
    st.d_inner = config.text.mamba_d_ssm;
    st.d_state = config.text.mamba_d_state;
    st.d_conv = config.text.mamba_d_conv;
    st.n_groups = config.text.mamba_n_groups;
    st.n_mamba_heads = config.text.mamba_n_heads;
    st.conv_dim = st.d_inner + 2 * st.n_groups * st.d_state;
    st.kv_dim = config.text.n_local_heads * config.text.head_dim;
    st.conv_states.resize(static_cast<size_t>(st.n_layer));
    st.ssm_states.resize(static_cast<size_t>(st.n_layer));
    st.k_cache.resize(static_cast<size_t>(st.n_layer));
    st.v_cache.resize(static_cast<size_t>(st.n_layer));
    st.k_pad.resize(static_cast<size_t>(st.n_layer));
    st.v_pad.resize(static_cast<size_t>(st.n_layer));
    const size_t conv_sz = static_cast<size_t>((st.d_conv - 1) * st.conv_dim);
    const size_t ssm_sz = static_cast<size_t>(st.d_state * st.d_inner);
    for (int64_t i = 0; i < st.n_layer; ++i) {
        st.conv_states[static_cast<size_t>(i)].assign(conv_sz, 0.0F);
        st.ssm_states[static_cast<size_t>(i)].assign(ssm_sz, 0.0F);
    }
    st.seq_len = 0;
    return st;
}

// Grow the zero-copy padded KV caches. The head stride changes with capacity,
// so live slots are re-laid-out per head; called between steps, and per-step
// graphs re-bind their views from scratch afterwards.
void grow_falcon_kv_pad(FalconH1StepState & state, int64_t head_dim, int64_t n_kv, int64_t need) {
    if (need <= state.kv_cap) return;
    // 128-slot buckets: fine-grained enough to keep padded flash cheap, and
    // they keep nek1 below the CPU flash kernel's split-KV threshold (512) for
    // as long as possible so the masked padded reduction stays bitwise equal
    // to the exact-prefix one.
    int64_t new_cap = std::max<int64_t>(64, ((need + 127) / 128) * 128);
    const size_t live = static_cast<size_t>(head_dim * state.seq_len);  // valid floats per head
    for (auto * caches : {&state.k_pad, &state.v_pad}) {
        for (auto & cache : *caches) {
            std::vector<float> next(static_cast<size_t>(head_dim * new_cap * n_kv), 0.0F);
            if (!cache.empty() && live > 0) {
                const size_t old_stride = static_cast<size_t>(head_dim * state.kv_cap);
                const size_t new_stride = static_cast<size_t>(head_dim * new_cap);
                for (int64_t h = 0; h < n_kv; ++h) {
                    std::memcpy(next.data() + h * new_stride,
                                cache.data() + h * old_stride,
                                live * sizeof(float));
                }
            }
            cache.swap(next);
        }
    }
    state.kv_cap = new_cap;
}

// Loop-invariant SSM constants, resolved once per generation: A = -exp(A_log)
// per mamba head, D expanded per channel.
void precompute_falcon_constants(
    const Audio8TtsConfig & config,
    const ArkttsARWeights & weights,
    FalconH1StepState & state) {
    if (state.constants_ready) return;
    const int64_t n_layer = config.text.n_layer;
    const int64_t d_inner = config.text.mamba_d_ssm;
    const int64_t n_mamba_heads = config.text.mamba_n_heads;
    const int64_t mamba_head_dim = config.text.mamba_d_head;
    state.pre_A.resize(static_cast<size_t>(n_layer));
    state.pre_D.resize(static_cast<size_t>(n_layer));
    for (int64_t li = 0; li < n_layer; ++li) {
        const auto & layer = weights.falcon_layers[static_cast<size_t>(li)];
        auto & a_vals = state.pre_A[static_cast<size_t>(li)];
        a_vals.resize(static_cast<size_t>(n_mamba_heads));
        std::vector<float> a_log(static_cast<size_t>(n_mamba_heads));
        ggml_backend_tensor_get(layer.ssm_A.tensor, a_log.data(), 0, a_log.size() * sizeof(float));
        for (int64_t h = 0; h < n_mamba_heads; ++h) {
            a_vals[static_cast<size_t>(h)] = -std::exp(a_log[static_cast<size_t>(h)]);
        }
        auto & d_vals = state.pre_D[static_cast<size_t>(li)];
        d_vals.resize(static_cast<size_t>(d_inner));
        std::vector<float> d_raw(static_cast<size_t>(n_mamba_heads));
        ggml_backend_tensor_get(layer.ssm_D.tensor, d_raw.data(), 0, d_raw.size() * sizeof(float));
        for (int64_t h = 0; h < n_mamba_heads; ++h) {
            for (int64_t d = 0; d < mamba_head_dim; ++d) {
                d_vals[static_cast<size_t>(d + h * mamba_head_dim)] = d_raw[static_cast<size_t>(h)];
            }
        }
    }
    state.constants_ready = true;
}

// Builds the reusable per-bucket Falcon step graph for the zero-copy path.
// Every weight and recurrent-state tensor is bound as an external view of host
// memory owned by `weights`/`state` (gallocr skips tensors whose data is set
// externally), conv/ssm state write-backs and the KV slot writes run in-graph
// (ggml_cpy / ggml_set_rows), and flash attention reads the full padded cache
// under a mask so the graph topology does not depend on the sequence length.
std::unique_ptr<FalconStepPlan> build_falcon_step_plan(
    ggml_backend_t backend,
    size_t arena_bytes,
    const Audio8TtsConfig & config,
    const ArkttsARWeights & weights,
    FalconH1StepState & state,
    ArkttsARProfile * profile) {
    const int64_t dim = config.text.dim;
    const int64_t n_layer = config.text.n_layer;
    const int64_t d_inner = config.text.mamba_d_ssm;
    const int64_t d_state = config.text.mamba_d_state;
    const int64_t d_conv = config.text.mamba_d_conv;
    const int64_t n_groups = config.text.mamba_n_groups;
    const int64_t n_mamba_heads = config.text.mamba_n_heads;
    const int64_t mamba_head_dim = config.text.mamba_d_head;
    const int64_t conv_dim = d_inner + 2 * n_groups * d_state;
    const int64_t n_head = config.text.n_head;
    const int64_t n_kv = config.text.n_local_heads;
    const int64_t head_dim = config.text.head_dim;
    const float norm_eps = config.text.norm_eps;
    const float rope_base = config.text.rope_base;
    const int64_t cap = state.kv_cap;

    auto plan = std::make_unique<FalconStepPlan>();
    plan->cap = cap;
    plan->backend = backend;

    auto t_init = Clock::now();
    ggml_init_params params{arena_bytes, nullptr, true};
    plan->ctx.reset(ggml_init(params));
    if (!plan->ctx) throw std::runtime_error("build_falcon_step_plan: ggml_init failed");
    ggml_context * ctx = plan->ctx.get();
    auto t_build = Clock::now();

    // Mask scratch: slots [0, seq_len] visible, the rest -inf; the runner
    // reveals one more slot per step.
    state.kv_mask.assign(static_cast<size_t>(cap), ggml_fp32_to_fp16(-INFINITY));
    for (int64_t i = 0; i <= state.seq_len && i < cap; ++i) {
        state.kv_mask[static_cast<size_t>(i)] = ggml_fp32_to_fp16(0.0F);
    }
    ggml_tensor * mask_t = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, cap, 1, 1, 1);
    mask_t->data = state.kv_mask.data();

    auto bind_const = [&](ggml_tensor * t, const std::vector<float> & values,
                          std::vector<float> & fallback, float fallback_fill) {
        if (!values.empty()) {
            t->data = const_cast<float *>(values.data());
            return;
        }
        if (fallback.empty()) {
            fallback.assign(static_cast<size_t>(ggml_nelements(t)), fallback_fill);
        }
        t->data = fallback.data();
    };

    ggml_tensor * cur = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim);
    ggml_set_name(cur, "falcon_input");
    cur->data = state.emb_stage.data();

    std::vector<ggml_tensor*> writebacks;     // conv/ssm state tails (dependency-ordered)
    std::vector<ggml_tensor*> kv_writebacks;  // KV slot writes alias the flash reads: expanded first
    writebacks.reserve(static_cast<size_t>(n_layer) * 2);
    kv_writebacks.reserve(static_cast<size_t>(n_layer) * 2);

    for (int64_t li = 0; li < n_layer; ++li) {
        const auto & layer = weights.falcon_layers[static_cast<size_t>(li)];

        // input_layernorm (RMS)
        ggml_tensor * ln_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim);
        bind_const(ln_w, layer.input_layernorm.values, state.ones_dim, 1.0F);
        ggml_tensor * normed = ggml_rms_norm(ctx, cur, norm_eps);
        normed = ggml_mul(ctx, normed, ln_w);

        // ---- Mamba2 branch ----
        // zxBCdt = in_proj(normed) -> [d_inner + conv_dim + n_mamba_heads]
        ggml_tensor * zxBCdt = ggml_mul_mat(ctx, layer.ssm_in.tensor, normed);
        ggml_tensor * z = ggml_view_1d(ctx, zxBCdt, d_inner, 0);
        ggml_tensor * xBC = ggml_view_1d(ctx, zxBCdt, conv_dim, d_inner * ggml_element_size(zxBCdt));
        ggml_tensor * dt = ggml_view_1d(ctx, zxBCdt, n_mamba_heads, (d_inner + conv_dim) * ggml_element_size(zxBCdt));

        // conv: state (d_conv-1 rows) + current xBC -> [d_conv, conv_dim, 1]
        ggml_tensor * st_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, conv_dim, d_conv - 1);
        st_t->data = state.conv_states[static_cast<size_t>(li)].data();
        ggml_tensor * stT = ggml_cont(ctx, ggml_transpose(ctx, st_t));  // [d_conv-1, conv_dim]
        ggml_tensor * xBC_r = ggml_cont(ctx, ggml_transpose(ctx, ggml_reshape_2d(ctx, xBC, conv_dim, 1)));  // [1, conv_dim]
        ggml_tensor * sx = ggml_concat(ctx, stT, xBC_r, 0);             // [d_conv, conv_dim]
        // Next conv state = sx rows 1..d_conv-1, written back into the host
        // state vector in-graph (the cpy depends on sx, hence on the cont()
        // that consumed st_t, so the old state is read before it is
        // overwritten).
        ggml_tensor * tail = ggml_view_2d(ctx, sx, d_conv - 1, conv_dim,
                                          sx->nb[1], ggml_element_size(sx));
        writebacks.push_back(ggml_cpy(ctx, ggml_transpose(ctx, tail), st_t));
        ggml_tensor * sx3 = ggml_reshape_3d(ctx, sx, d_conv, conv_dim, 1);
        // ggml ssm_conv computes y[c] = sum_k w[k,c]*sx[k,c] with sx row 0 the
        // oldest frame — the same orientation as the HF conv1d/cached decode.
        ggml_tensor * conv_w2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_conv, conv_dim);
        bind_const(conv_w2, layer.conv1d_kernel.values, state.zeros_conv_kernel, 0.0F);
        ggml_tensor * xBC_conv = ggml_ssm_conv(ctx, sx3, conv_w2);  // [conv_dim, 1, 1]
        ggml_tensor * conv_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, conv_dim);
        bind_const(conv_b, layer.ssm_conv1d_b.values, state.zeros_conv, 0.0F);
        xBC_conv = ggml_add(ctx, xBC_conv, ggml_reshape_3d(ctx, conv_b, conv_dim, 1, 1));
        xBC_conv = ggml_silu(ctx, xBC_conv);

        // split x / B / C (conv output is contiguous [conv_dim,1,1])
        ggml_tensor * x = ggml_view_1d(ctx, xBC_conv, d_inner, 0);
        ggml_tensor * B = ggml_view_1d(ctx, xBC_conv, d_state * n_groups, d_inner * ggml_element_size(xBC_conv));
        ggml_tensor * C = ggml_view_1d(ctx, xBC_conv, d_state * n_groups, (d_inner + d_state * n_groups) * ggml_element_size(xBC_conv));

        // x -> [head_dim, n_mamba_heads, 1, 1]
        ggml_tensor * x4 = ggml_view_4d(ctx, x, mamba_head_dim, n_mamba_heads, 1, 1,
                                        mamba_head_dim * ggml_element_size(x),
                                        mamba_head_dim * n_mamba_heads * ggml_element_size(x),
                                        mamba_head_dim * n_mamba_heads * ggml_element_size(x), 0);
        ggml_tensor * B4 = ggml_view_4d(ctx, B, d_state, n_groups, 1, 1,
                                        d_state * ggml_element_size(B),
                                        d_state * n_groups * ggml_element_size(B),
                                        d_state * n_groups * ggml_element_size(B), 0);
        ggml_tensor * C4 = ggml_view_4d(ctx, C, d_state, n_groups, 1, 1,
                                        d_state * ggml_element_size(C),
                                        d_state * n_groups * ggml_element_size(C),
                                        d_state * n_groups * ggml_element_size(C), 0);

        // dt = dt + dt_bias -> [n_mamba_heads, 1, 1]
        ggml_tensor * dt_eff = ggml_add(ctx, dt, layer.ssm_dt_b.tensor);
        ggml_tensor * dt3 = ggml_view_3d(ctx, dt_eff, n_mamba_heads, 1, 1,
                                         n_mamba_heads * ggml_element_size(dt_eff),
                                         n_mamba_heads * ggml_element_size(dt_eff), 0);

        // A = -exp(A_log), precomputed: [1, n_mamba_heads]
        ggml_tensor * A_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, n_mamba_heads);
        A_t->data = state.pre_A[static_cast<size_t>(li)].data();

        // ssm state: [d_state, mamba_head_dim, n_mamba_heads]
        ggml_tensor * ssm_t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_state, mamba_head_dim, n_mamba_heads);
        ssm_t->data = state.ssm_states[static_cast<size_t>(li)].data();

        // ids for scan (1 sequence)
        ggml_tensor * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        ids->data = &state.ids_value;

        ggml_tensor * scan = ggml_ssm_scan(ctx, ssm_t, x4, dt3, A_t, B4, C4, ids);
        // New ssm state = scan tail (after the d_inner y values), written back
        // into the host state vector in-graph (ordered after the scan read).
        ggml_tensor * next_state = ggml_view_3d(ctx, scan,
                                                d_state, mamba_head_dim, n_mamba_heads,
                                                d_state * ggml_element_size(scan),
                                                d_state * mamba_head_dim * ggml_element_size(scan),
                                                d_inner * ggml_element_size(scan));
        writebacks.push_back(ggml_cpy(ctx, next_state, ssm_t));
        ggml_tensor * y = ggml_view_1d(ctx, scan, d_inner, 0);

        // y += x * D  (D precomputed)
        ggml_tensor * D_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d_inner);
        D_t->data = state.pre_D[static_cast<size_t>(li)].data();
        y = ggml_add(ctx, y, ggml_mul(ctx, ggml_view_1d(ctx, x4, d_inner, 0), D_t));

        // z gate: y *= silu(z)
        y = ggml_mul(ctx, y, ggml_silu(ctx, z));

        ggml_tensor * out_mamba = ggml_mul_mat(ctx, layer.ssm_out.tensor, y);  // [dim]

        // ---- GQA attention branch ----
        // ggml_flash_attn_ext layout: [head_dim, n_tokens, n_head, batch].
        ggml_tensor * q = ggml_mul_mat(ctx, layer.attn_q_proj.tensor, normed);  // [n_head*head_dim]
        ggml_tensor * k = ggml_mul_mat(ctx, layer.attn_k_proj.tensor, normed);  // [n_kv*head_dim]
        ggml_tensor * v = ggml_mul_mat(ctx, layer.attn_v_proj.tensor, normed);  // [n_kv*head_dim]

        ggml_tensor * q4 = ggml_view_4d(ctx, q, head_dim, n_head, 1, 1,
                                        head_dim * ggml_element_size(q),
                                        head_dim * n_head * ggml_element_size(q),
                                        head_dim * n_head * ggml_element_size(q), 0);
        ggml_tensor * k4 = ggml_view_4d(ctx, k, head_dim, n_kv, 1, 1,
                                        head_dim * ggml_element_size(k),
                                        head_dim * n_kv * ggml_element_size(k),
                                        head_dim * n_kv * ggml_element_size(k), 0);
        ggml_tensor * v4 = ggml_view_4d(ctx, v, head_dim, n_kv, 1, 1,
                                        head_dim * ggml_element_size(v),
                                        head_dim * n_kv * ggml_element_size(v),
                                        head_dim * n_kv * ggml_element_size(v), 0);

        // RoPE (NEOX / HF default half rotation), base 1e11; ne2 = n_tokens = 1.
        ggml_tensor * pos_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        pos_t->data = &state.pos_value;
        ggml_tensor * q_r = ggml_rope_ext(ctx, q4, pos_t, nullptr, head_dim,
                                          GGML_ROPE_TYPE_NEOX, config.text.max_seq_len,
                                          rope_base, 1.0F, 1.0F, 1.0F, 32.0F, 1.0F);
        ggml_tensor * k_r = ggml_rope_ext(ctx, k4, pos_t, nullptr, head_dim,
                                          GGML_ROPE_TYPE_NEOX, config.text.max_seq_len,
                                          rope_base, 1.0F, 1.0F, 1.0F, 32.0F, 1.0F);

        // [head_dim, n_head, 1, 1] -> permute(0,2,1,3) -> [head_dim, 1, n_head, 1]
        ggml_tensor * q_p = ggml_permute(ctx, q_r, 0, 2, 1, 3);
        ggml_tensor * k_p = ggml_permute(ctx, k_r, 0, 2, 1, 3);
        ggml_tensor * v_p = ggml_permute(ctx, v4, 0, 2, 1, 3);

        // Padded caches as external leaves [head_dim, cap, n_kv, 1]; the fresh
        // k/v land in slot `kv_slot` via in-graph set_rows, and flash attends
        // the whole padded cache under the mask (slots > seq are -inf, which
        // contributes exactly zero to the softmax).
        ggml_tensor * k_pad_t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, cap, n_kv, 1);
        ggml_tensor * v_pad_t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, cap, n_kv, 1);
        k_pad_t->data = state.k_pad[static_cast<size_t>(li)].data();
        v_pad_t->data = state.v_pad[static_cast<size_t>(li)].data();
        ggml_tensor * slot_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        slot_t->data = &state.kv_slot;
        kv_writebacks.push_back(ggml_set_rows(ctx, k_pad_t, k_p, slot_t));
        kv_writebacks.push_back(ggml_set_rows(ctx, v_pad_t, v_p, slot_t));

        // Single-token causal: all unmasked cache slots are visible.
        ggml_tensor * attn = ggml_flash_attn_ext(ctx, q_p, k_pad_t, v_pad_t, mask_t,
                                                 1.0F / std::sqrt(static_cast<float>(head_dim)),
                                                 0.0F, 0.0F);
        ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
        ggml_tensor * attn_flat = ggml_cont(ctx, ggml_reshape_1d(ctx, attn, n_head * head_dim));
        ggml_tensor * attn_out = ggml_mul_mat(ctx, layer.attn_o_proj.tensor, attn_flat);  // [dim]

        // ---- merge + residual ----
        ggml_tensor * h = ggml_add(ctx, out_mamba, attn_out);
        h = ggml_add(ctx, cur, h);

        // ---- FFN ----
        ggml_tensor * pre_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim);
        bind_const(pre_w, layer.pre_ff_layernorm.values, state.ones_dim, 1.0F);
        ggml_tensor * h2 = ggml_rms_norm(ctx, h, norm_eps);
        h2 = ggml_mul(ctx, h2, pre_w);
        ggml_tensor * gate_ff = ggml_mul_mat(ctx, layer.ffn_gate.tensor, h2);
        ggml_tensor * up_ff = ggml_mul_mat(ctx, layer.ffn_up.tensor, h2);
        ggml_tensor * gated = ggml_mul(ctx, ggml_silu(ctx, gate_ff), up_ff);
        ggml_tensor * down = ggml_mul_mat(ctx, layer.ffn_down.tensor, gated);
        cur = ggml_add(ctx, h, down);
    }

    // final norm + semantic head
    ggml_tensor * final_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim);
    bind_const(final_w, weights.slow_norm.values, state.ones_dim, 1.0F);
    ggml_tensor * final_norm = ggml_rms_norm(ctx, cur, norm_eps);
    final_norm = ggml_mul(ctx, final_norm, final_w);
    ggml_tensor * logits = ggml_mul_mat(ctx, weights.falcon_lm_head.tensor, final_norm);  // [vocab]
    // NOTE: lm_head_multiplier applies only to FalconH1ForCausalLM's full-vocab head.
    // ArkttsModel uses the compact semantic_output head and does NOT scale logits.
    plan->logits_out = ggml_dup(ctx, logits);
    plan->hidden_out = ggml_dup(ctx, final_norm);
    ggml_set_name(plan->logits_out, "logits_out");
    ggml_set_name(plan->hidden_out, "hidden_out");
    // Pin the outputs: the writeback nodes expanded after them allocate no
    // memory of their own, but the flag keeps gallocr from ever reusing the
    // output buffers while the plan is reused across steps.
    ggml_set_output(plan->logits_out);
    ggml_set_output(plan->hidden_out);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);
    // KV slot writes first: they alias the cache the attention reads, so they
    // must precede those nodes in graph order (backends execute sequentially).
    for (ggml_tensor * wb : kv_writebacks) {
        ggml_build_forward_expand(gf, wb);
    }
    ggml_build_forward_expand(gf, plan->logits_out);
    ggml_build_forward_expand(gf, plan->hidden_out);
    for (ggml_tensor * wb : writebacks) {
        ggml_build_forward_expand(gf, wb);
    }
    plan->gf = gf;
    auto t_graph = Clock::now();
    plan->gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!plan->gallocr || !ggml_gallocr_reserve(plan->gallocr, gf) || !ggml_gallocr_alloc_graph(plan->gallocr, gf)) {
        throw std::runtime_error("build_falcon_step_plan: gallocr failed");
    }
    auto t_alloc = Clock::now();
    if (profile != nullptr) {
        profile->falcon_step_init_ms += engine::debug::elapsed_ms(t_init, t_build);
        profile->falcon_step_build_ms += engine::debug::elapsed_ms(t_build, t_graph);
        profile->falcon_step_gallocr_ms += engine::debug::elapsed_ms(t_graph, t_alloc);
    }
    return plan;
}

// Zero-copy runner: feed the staged inputs, run the baked bucket graph, read
// logits/hidden. No per-step graph construction, allocation, or state I/O.
SlowForwardOutput falcon_forward_step_zero_copy(
    ggml_backend_t backend,
    int threads,
    size_t arena_bytes,
    const Audio8TtsConfig & config,
    const ArkttsARWeights & weights,
    const std::vector<float> & embedding,
    FalconH1StepState & state,
    int64_t position,
    ArkttsARProfile * profile) {
    const int64_t dim = config.text.dim;
    const int64_t head_dim = config.text.head_dim;
    const int64_t n_kv = config.text.n_local_heads;
    const int64_t vocab = config.fast.vocab_size + 1;
    const int64_t seq = state.seq_len;

    precompute_falcon_constants(config, weights, state);
    grow_falcon_kv_pad(state, head_dim, n_kv, seq + 1);
    if (state.emb_stage.empty()) {
        state.emb_stage.assign(static_cast<size_t>(dim), 0.0F);
    }
    if (!state.plan || state.plan->cap != state.kv_cap) {
        state.plan = build_falcon_step_plan(backend, arena_bytes, config, weights, state, profile);
    }
    auto t_feed = Clock::now();
    std::memcpy(state.emb_stage.data(), embedding.data(), embedding.size() * sizeof(float));
    state.pos_value = static_cast<int32_t>(position);
    state.kv_slot = static_cast<int32_t>(seq);
    state.kv_mask[static_cast<size_t>(seq)] = ggml_fp32_to_fp16(0.0F);
    auto t_compute = Clock::now();
    core::set_backend_threads(backend, threads);
    const ggml_status status = core::compute_backend_graph(backend, state.plan->gf, nullptr, "falcon_forward_step");
    ggml_backend_synchronize(backend);
    auto t_read = Clock::now();
    if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("falcon_forward_step compute failed");
    }
    SlowForwardOutput out;
    out.logits.resize(static_cast<size_t>(vocab));
    out.hidden.resize(static_cast<size_t>(dim));
    ggml_backend_tensor_get(state.plan->logits_out, out.logits.data(), 0, out.logits.size() * sizeof(float));
    ggml_backend_tensor_get(state.plan->hidden_out, out.hidden.data(), 0, out.hidden.size() * sizeof(float));
    state.seq_len = seq + 1;
    if (profile != nullptr) {
        profile->falcon_step_upload_ms += engine::debug::elapsed_ms(t_feed, t_compute);
        profile->falcon_step_compute_ms += engine::debug::elapsed_ms(t_compute, t_read);
        profile->falcon_step_download_ms += engine::debug::elapsed_ms(t_read, Clock::now());
        profile->falcon_step_runs += 1;
    }
    return out;
}

// Single-token Falcon-H1 forward. `embedding` is the pre-multiplied token
// embedding (text embedding * embedding_multiplier + codebook sum).
SlowForwardOutput falcon_forward_step(
    ggml_backend_t backend,
    int threads,
    size_t arena_bytes,
    const Audio8TtsConfig & config,
    const ArkttsARWeights & weights,
    const std::vector<float> & embedding,  // [dim]
    FalconH1StepState & state,
    int64_t position,
    ArkttsARProfile * profile = nullptr) {
    const int64_t dim = config.text.dim;
    const int64_t n_layer = config.text.n_layer;
    const int64_t d_inner = config.text.mamba_d_ssm;
    const int64_t d_state = config.text.mamba_d_state;
    const int64_t d_conv = config.text.mamba_d_conv;
    const int64_t n_groups = config.text.mamba_n_groups;
    const int64_t n_mamba_heads = config.text.mamba_n_heads;
    const int64_t mamba_head_dim = config.text.mamba_d_head;
    const int64_t conv_dim = d_inner + 2 * n_groups * d_state;
    const int64_t n_head = config.text.n_head;
    const int64_t n_kv = config.text.n_local_heads;
    const int64_t head_dim = config.text.head_dim;
    const float norm_eps = config.text.norm_eps;
    const float rope_base = config.text.rope_base;
    const int64_t vocab = config.fast.vocab_size + 1;
    const int64_t seq = state.seq_len;

    if (embedding.size() != static_cast<size_t>(dim)) {
        throw std::runtime_error("falcon_forward_step: embedding size mismatch");
    }

    // Host backends run the reusable zero-copy plan graph (one baked graph per
    // KV capacity bucket; weights/state bound as external host views, state
    // write-backs in-graph). Other backends keep the per-call explicit
    // upload/download path below.
    const bool zero_copy = core::backend_type(backend) == core::BackendType::Cpu;
    if (zero_copy) {
        return falcon_forward_step_zero_copy(backend, threads, arena_bytes, config, weights,
                                             embedding, state, position, profile);
    }

    auto t_init = Clock::now();
    ggml_init_params params{arena_bytes, nullptr, true};
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx(ggml_init(params));
    if (!ctx) throw std::runtime_error("falcon_forward_step: ggml_init failed");
    auto t_build = Clock::now();

    ggml_tensor * cur = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, dim);
    ggml_set_name(cur, "falcon_input");
    if (zero_copy) {
        cur->data = const_cast<float *>(embedding.data());
    } else {
        ggml_set_input(cur);
    }

    // Bind a loop-invariant F32 tensor to its host values with zero copies when
    // the backend reads host memory directly; otherwise mark it for the upload
    // pass below. `fallback` (filled once with `fallback_fill`) covers weights
    // that are absent from the checkpoint.
    auto bind_const = [&](ggml_tensor * t, const std::vector<float> & values,
                          std::vector<float> & fallback, float fallback_fill) {
        if (!zero_copy) {
            ggml_set_input(t);
            return;
        }
        if (!values.empty()) {
            t->data = const_cast<float *>(values.data());
            return;
        }
        if (fallback.empty()) {
            fallback.assign(static_cast<size_t>(ggml_nelements(t)), fallback_fill);
        }
        t->data = fallback.data();
    };
    auto bind_state = [&](ggml_tensor * t, const std::vector<float> & values) {
        if (!zero_copy) {
            ggml_set_input(t);
            return;
        }
        t->data = const_cast<float *>(values.data());
    };
    std::vector<ggml_tensor*> writebacks;
    writebacks.reserve(static_cast<size_t>(n_layer) * 2);
    // KV slot writes have no data dependency protecting them from the
    // attention read of the same cache, so they are expanded into the graph
    // FIRST below — backends execute nodes sequentially in graph order.
    std::vector<ggml_tensor*> kv_writebacks;
    kv_writebacks.reserve(static_cast<size_t>(n_layer) * 2);

    std::vector<ggml_tensor*> ln_w_ts;
    std::vector<ggml_tensor*> pre_w_ts;
    std::vector<ggml_tensor*> conv_st_ts;
    std::vector<ggml_tensor*> ssm_st_ts;
    std::vector<ggml_tensor*> k_cur_ts;
    std::vector<ggml_tensor*> v_cur_ts;
    std::vector<ggml_tensor*> conv_b_ts;
    std::vector<ggml_tensor*> conv_w2_ts;
    std::vector<ggml_tensor*> k_cache_ts;
    std::vector<ggml_tensor*> v_cache_ts;
    std::vector<ggml_tensor*> sx_ts;
    std::vector<ggml_tensor*> scan_ts;
    std::vector<ggml_tensor*> ids_ts;
    std::vector<ggml_tensor*> pos_ts;
    ln_w_ts.reserve(static_cast<size_t>(n_layer));
    pre_w_ts.reserve(static_cast<size_t>(n_layer));
    conv_st_ts.reserve(static_cast<size_t>(n_layer));
    ssm_st_ts.reserve(static_cast<size_t>(n_layer));
    k_cur_ts.reserve(static_cast<size_t>(n_layer));
    v_cur_ts.reserve(static_cast<size_t>(n_layer));
    conv_b_ts.reserve(static_cast<size_t>(n_layer));
    conv_w2_ts.reserve(static_cast<size_t>(n_layer));
    k_cache_ts.reserve(static_cast<size_t>(n_layer));
    v_cache_ts.reserve(static_cast<size_t>(n_layer));
    sx_ts.reserve(static_cast<size_t>(n_layer));
    scan_ts.reserve(static_cast<size_t>(n_layer));
    ids_ts.reserve(static_cast<size_t>(n_layer));
    pos_ts.reserve(static_cast<size_t>(n_layer));

    // A = -exp(A_log) per layer
    std::vector<ggml_tensor*> A_ts;
    A_ts.reserve(static_cast<size_t>(n_layer));
    // D expanded to [d_inner]: D[h] repeated mamba_head_dim times
    std::vector<ggml_tensor*> D_ts;

    for (int64_t li = 0; li < n_layer; ++li) {
        const auto & layer = weights.falcon_layers[static_cast<size_t>(li)];

        // input_layernorm (RMS)
        ggml_tensor * ln_w = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, dim);
        bind_const(ln_w, layer.input_layernorm.values, state.ones_dim, 1.0F);
        ln_w_ts.push_back(ln_w);
        ggml_tensor * normed = ggml_rms_norm(ctx.get(), cur, norm_eps);
        normed = ggml_mul(ctx.get(), normed, ln_w);

        // ---- Mamba2 branch ----
        // zxBCdt = in_proj(normed) -> [d_inner + conv_dim + n_mamba_heads]
        ggml_tensor * zxBCdt = ggml_mul_mat(ctx.get(), layer.ssm_in.tensor, normed);
        ggml_tensor * z = ggml_view_1d(ctx.get(), zxBCdt, d_inner, 0);
        ggml_tensor * xBC = ggml_view_1d(ctx.get(), zxBCdt, conv_dim, d_inner * ggml_element_size(zxBCdt));
        ggml_tensor * dt = ggml_view_1d(ctx.get(), zxBCdt, n_mamba_heads, (d_inner + conv_dim) * ggml_element_size(zxBCdt));

        // conv: state (d_conv-1 rows) + current xBC -> [d_conv, conv_dim, 1]
        ggml_tensor * st_t = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, conv_dim, d_conv - 1);
        bind_state(st_t, state.conv_states[static_cast<size_t>(li)]);
        conv_st_ts.push_back(st_t);
        ggml_tensor * stT = ggml_cont(ctx.get(), ggml_transpose(ctx.get(), st_t));  // [d_conv-1, conv_dim]
        ggml_tensor * xBC_r = ggml_cont(ctx.get(), ggml_transpose(ctx.get(), ggml_reshape_2d(ctx.get(), xBC, conv_dim, 1)));  // [1, conv_dim]
        ggml_tensor * sx = ggml_concat(ctx.get(), stT, xBC_r, 0);                   // [d_conv, conv_dim]
        sx_ts.push_back(sx);
        if (zero_copy) {
            // Next conv state = sx rows 1..d_conv-1, written back into the host
            // state vector in-graph. The cpy depends on sx (hence on the cont()
            // that consumed st_t), so the old state is read before it is
            // overwritten on every backend.
            ggml_tensor * tail = ggml_view_2d(ctx.get(), sx, d_conv - 1, conv_dim,
                                              sx->nb[1], ggml_element_size(sx));
            writebacks.push_back(ggml_cpy(ctx.get(), ggml_transpose(ctx.get(), tail), st_t));
        } else {
            ggml_set_output(sx);  // host reads the conv window back — pin the buffer
        }
        ggml_tensor * sx3 = ggml_reshape_3d(ctx.get(), sx, d_conv, conv_dim, 1);
        // ggml ssm_conv computes y[c] = sum_k w[k,c]*sx[k,c] with sx row 0 the oldest
        // frame — the same orientation as the HF conv1d/cached decode, so the GGUF
        // kernel is fed as loaded (see load_falcon_layer).
        ggml_tensor * conv_w2 = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, d_conv, conv_dim);
        bind_const(conv_w2, layer.conv1d_kernel.values, state.zeros_conv_kernel, 0.0F);
        conv_w2_ts.push_back(conv_w2);
        ggml_tensor * xBC_conv = ggml_ssm_conv(ctx.get(), sx3, conv_w2);  // [conv_dim, 1, 1]
        ggml_tensor * conv_b = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, conv_dim);
        bind_const(conv_b, layer.ssm_conv1d_b.values, state.zeros_conv, 0.0F);
        conv_b_ts.push_back(conv_b);
        xBC_conv = ggml_add(ctx.get(), xBC_conv, ggml_reshape_3d(ctx.get(), conv_b, conv_dim, 1, 1));
        xBC_conv = ggml_silu(ctx.get(), xBC_conv);

        // split x / B / C (conv output is contiguous [conv_dim,1,1])
        ggml_tensor * x = ggml_view_1d(ctx.get(), xBC_conv, d_inner, 0);
        ggml_tensor * B = ggml_view_1d(ctx.get(), xBC_conv, d_state * n_groups, d_inner * ggml_element_size(xBC_conv));
        ggml_tensor * C = ggml_view_1d(ctx.get(), xBC_conv, d_state * n_groups, (d_inner + d_state * n_groups) * ggml_element_size(xBC_conv));

        // x -> [head_dim, n_mamba_heads, 1, 1]
        ggml_tensor * x4 = ggml_view_4d(ctx.get(), x, mamba_head_dim, n_mamba_heads, 1, 1,
                                        mamba_head_dim * ggml_element_size(x),
                                        mamba_head_dim * n_mamba_heads * ggml_element_size(x),
                                        mamba_head_dim * n_mamba_heads * ggml_element_size(x), 0);
        ggml_tensor * B4 = ggml_view_4d(ctx.get(), B, d_state, n_groups, 1, 1,
                                        d_state * ggml_element_size(B),
                                        d_state * n_groups * ggml_element_size(B),
                                        d_state * n_groups * ggml_element_size(B), 0);
        ggml_tensor * C4 = ggml_view_4d(ctx.get(), C, d_state, n_groups, 1, 1,
                                        d_state * ggml_element_size(C),
                                        d_state * n_groups * ggml_element_size(C),
                                        d_state * n_groups * ggml_element_size(C), 0);

        // dt = dt + dt_bias -> [n_mamba_heads, 1, 1]
        ggml_tensor * dt_eff = ggml_add(ctx.get(), dt, layer.ssm_dt_b.tensor);
        ggml_tensor * dt3 = ggml_view_3d(ctx.get(), dt_eff, n_mamba_heads, 1, 1,
                                         n_mamba_heads * ggml_element_size(dt_eff),
                                         n_mamba_heads * ggml_element_size(dt_eff), 0);

        // A = -exp(A_log): [1, n_mamba_heads] — precomputed once per generation
        // into state.pre_A (zero-copy) or uploaded per step below.
        ggml_tensor * A_t = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 1, n_mamba_heads);
        if (zero_copy) {
            A_t->data = state.pre_A[static_cast<size_t>(li)].data();
        } else {
            ggml_set_input(A_t);
        }
        A_ts.push_back(A_t);

        // ssm state: [d_state, mamba_head_dim, n_mamba_heads]
        ggml_tensor * ssm_t = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, d_state, mamba_head_dim, n_mamba_heads);
        bind_state(ssm_t, state.ssm_states[static_cast<size_t>(li)]);
        ssm_st_ts.push_back(ssm_t);

        // ids for scan (1 sequence)
        ggml_tensor * ids = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 1);
        if (zero_copy) {
            ids->data = &state.ids_value;
        } else {
            ggml_set_input(ids);
        }
        ids_ts.push_back(ids);

        ggml_tensor * scan = ggml_ssm_scan(ctx.get(), ssm_t, x4, dt3, A_t, B4, C4, ids);
        scan_ts.push_back(scan);
        if (zero_copy) {
            // New ssm state = scan tail (after the d_inner y values), written
            // back into the host state vector in-graph. The cpy depends on the
            // scan that read ssm_t, so the old state is fully consumed first.
            ggml_tensor * next_state = ggml_view_3d(ctx.get(), scan,
                                                    d_state, mamba_head_dim, n_mamba_heads,
                                                    d_state * ggml_element_size(scan),
                                                    d_state * mamba_head_dim * ggml_element_size(scan),
                                                    d_inner * ggml_element_size(scan));
            writebacks.push_back(ggml_cpy(ctx.get(), next_state, ssm_t));
        } else {
            ggml_set_output(scan);  // keep state tail alive for host read-back
        }
        ggml_tensor * y = ggml_view_1d(ctx.get(), scan, d_inner, 0);

        // y += x * D  (D precomputed per generation / uploaded per step)
        ggml_tensor * D_t = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, d_inner);
        if (zero_copy) {
            D_t->data = state.pre_D[static_cast<size_t>(li)].data();
        } else {
            ggml_set_input(D_t);
        }
        D_ts.push_back(D_t);
        y = ggml_add(ctx.get(), y, ggml_mul(ctx.get(), ggml_view_1d(ctx.get(), x4, d_inner, 0), D_t));

        // z gate: y *= silu(z)
        y = ggml_mul(ctx.get(), y, ggml_silu(ctx.get(), z));

        ggml_tensor * out_mamba = ggml_mul_mat(ctx.get(), layer.ssm_out.tensor, y);  // [dim]

        // ---- GQA attention branch ----
        // ggml_flash_attn_ext layout: [head_dim, n_tokens, n_head, batch].
        // Project -> [head_dim, n_head, 1, 1] -> RoPE (ne2 = tokens) -> permute(0,2,1,3)
        // -> [head_dim, 1, n_head, 1]; KV cache kept in the same layout.
        ggml_tensor * q = ggml_mul_mat(ctx.get(), layer.attn_q_proj.tensor, normed);  // [n_head*head_dim]
        ggml_tensor * k = ggml_mul_mat(ctx.get(), layer.attn_k_proj.tensor, normed);  // [n_kv*head_dim]
        ggml_tensor * v = ggml_mul_mat(ctx.get(), layer.attn_v_proj.tensor, normed);  // [n_kv*head_dim]

        ggml_tensor * q4 = ggml_view_4d(ctx.get(), q, head_dim, n_head, 1, 1,
                                        head_dim * ggml_element_size(q),
                                        head_dim * n_head * ggml_element_size(q),
                                        head_dim * n_head * ggml_element_size(q), 0);
        ggml_tensor * k4 = ggml_view_4d(ctx.get(), k, head_dim, n_kv, 1, 1,
                                        head_dim * ggml_element_size(k),
                                        head_dim * n_kv * ggml_element_size(k),
                                        head_dim * n_kv * ggml_element_size(k), 0);
        ggml_tensor * v4 = ggml_view_4d(ctx.get(), v, head_dim, n_kv, 1, 1,
                                        head_dim * ggml_element_size(v),
                                        head_dim * n_kv * ggml_element_size(v),
                                        head_dim * n_kv * ggml_element_size(v), 0);

        // RoPE (NEOX / HF default half rotation), base 1e11; ne2 = n_tokens = 1.
        ggml_tensor * pos_t = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 1);
        if (zero_copy) {
            pos_t->data = &state.pos_value;
        } else {
            ggml_set_input(pos_t);
        }
        pos_ts.push_back(pos_t);
        ggml_tensor * q_r = ggml_rope_ext(ctx.get(), q4, pos_t, nullptr, head_dim,
                                          GGML_ROPE_TYPE_NEOX, config.text.max_seq_len,
                                          rope_base, 1.0F, 1.0F, 1.0F, 32.0F, 1.0F);
        ggml_tensor * k_r = ggml_rope_ext(ctx.get(), k4, pos_t, nullptr, head_dim,
                                          GGML_ROPE_TYPE_NEOX, config.text.max_seq_len,
                                          rope_base, 1.0F, 1.0F, 1.0F, 32.0F, 1.0F);
        // k_p/v_p (and their k_r/v bases) are read back on the host for the KV
        // cache on the fallback path — pin the underlying buffers there.
        if (!zero_copy) {
            ggml_set_output(k_r);
            ggml_set_output(v);
        }

        // [head_dim, n_head, 1, 1] -> permute(0,2,1,3) -> [head_dim, 1, n_head, 1]
        ggml_tensor * q_p = ggml_permute(ctx.get(), q_r, 0, 2, 1, 3);
        ggml_tensor * k_p = ggml_permute(ctx.get(), k_r, 0, 2, 1, 3);
        ggml_tensor * v_p = ggml_permute(ctx.get(), v4, 0, 2, 1, 3);
        k_cur_ts.push_back(k_p);
        v_cur_ts.push_back(v_p);

        // KV cache in flash layout: [head_dim, n_tokens, n_kv, 1]
        ggml_tensor * K_all = nullptr;
        ggml_tensor * V_all = nullptr;
        if (zero_copy) {
            // Padded caches bound as external leaves; the fresh k/v are written
            // into slot `seq` in-graph (kv_writebacks are expanded before the
            // attention nodes, ordering the writes first), and flash attends
            // over a strided prefix view of slots [0, seq+1). CPU
            // flash_attn_ext only requires contiguous rows (nb0), so the
            // padded head stride is fine.
            ggml_tensor * k_pad_t = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, head_dim, state.kv_cap, n_kv, 1);
            ggml_tensor * v_pad_t = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, head_dim, state.kv_cap, n_kv, 1);
            k_pad_t->data = state.k_pad[static_cast<size_t>(li)].data();
            v_pad_t->data = state.v_pad[static_cast<size_t>(li)].data();
            const size_t slot_off = static_cast<size_t>(seq * head_dim) * ggml_element_size(k_pad_t);
            kv_writebacks.push_back(ggml_cpy(ctx.get(), k_p,
                ggml_view_4d(ctx.get(), k_pad_t, head_dim, 1, n_kv, 1,
                             k_pad_t->nb[1], k_pad_t->nb[2], k_pad_t->nb[3], slot_off)));
            kv_writebacks.push_back(ggml_cpy(ctx.get(), v_p,
                ggml_view_4d(ctx.get(), v_pad_t, head_dim, 1, n_kv, 1,
                             v_pad_t->nb[1], v_pad_t->nb[2], v_pad_t->nb[3], slot_off)));
            K_all = ggml_view_4d(ctx.get(), k_pad_t, head_dim, seq + 1, n_kv, 1,
                                 k_pad_t->nb[1], k_pad_t->nb[2], k_pad_t->nb[3], 0);
            V_all = ggml_view_4d(ctx.get(), v_pad_t, head_dim, seq + 1, n_kv, 1,
                                 v_pad_t->nb[1], v_pad_t->nb[2], v_pad_t->nb[3], 0);
        } else {
            ggml_tensor * k_cache_t = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, head_dim, seq, n_kv, 1);
            ggml_tensor * v_cache_t = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, head_dim, seq, n_kv, 1);
            ggml_set_input(k_cache_t);
            ggml_set_input(v_cache_t);
            k_cache_ts.push_back(k_cache_t);
            v_cache_ts.push_back(v_cache_t);
            K_all = ggml_concat(ctx.get(), k_cache_t, k_p, 1);  // [head_dim, seq+1, n_kv, 1]
            V_all = ggml_concat(ctx.get(), v_cache_t, v_p, 1);
        }

        // Single-token causal: current query attends to all cached keys (all visible).
        ggml_tensor * attn = ggml_flash_attn_ext(ctx.get(), q_p, K_all, V_all, nullptr,
                                                 1.0F / std::sqrt(static_cast<float>(head_dim)),
                                                 0.0F, 0.0F);
        ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
        ggml_tensor * attn_flat = ggml_cont(ctx.get(), ggml_reshape_1d(ctx.get(), attn, n_head * head_dim));
        ggml_tensor * attn_out = ggml_mul_mat(ctx.get(), layer.attn_o_proj.tensor, attn_flat);  // [dim]

        // ---- merge + residual ----
        ggml_tensor * h = ggml_add(ctx.get(), out_mamba, attn_out);
        h = ggml_add(ctx.get(), cur, h);

        // ---- FFN ----
        ggml_tensor * pre_w = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, dim);
        bind_const(pre_w, layer.pre_ff_layernorm.values, state.ones_dim, 1.0F);
        pre_w_ts.push_back(pre_w);
        ggml_tensor * h2 = ggml_rms_norm(ctx.get(), h, norm_eps);
        h2 = ggml_mul(ctx.get(), h2, pre_w);
        ggml_tensor * gate_ff = ggml_mul_mat(ctx.get(), layer.ffn_gate.tensor, h2);
        ggml_tensor * up_ff = ggml_mul_mat(ctx.get(), layer.ffn_up.tensor, h2);
        ggml_tensor * gated = ggml_mul(ctx.get(), ggml_silu(ctx.get(), gate_ff), up_ff);
        ggml_tensor * down = ggml_mul_mat(ctx.get(), layer.ffn_down.tensor, gated);
        cur = ggml_add(ctx.get(), h, down);
    }

    // final norm + semantic head
    ggml_tensor * final_w = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, dim);
    bind_const(final_w, weights.slow_norm.values, state.ones_dim, 1.0F);
    ggml_tensor * final_norm = ggml_rms_norm(ctx.get(), cur, norm_eps);
    final_norm = ggml_mul(ctx.get(), final_norm, final_w);
    ggml_tensor * logits = ggml_mul_mat(ctx.get(), weights.falcon_lm_head.tensor, final_norm);  // [vocab]
    // NOTE: lm_head_multiplier applies only to FalconH1ForCausalLM's full-vocab head.
    // ArkttsModel uses the compact semantic_output head and does NOT scale logits.
    ggml_tensor * logits_out = ggml_dup(ctx.get(), logits);
    ggml_tensor * hidden_out = ggml_dup(ctx.get(), final_norm);
    ggml_set_name(logits_out, "logits_out");
    ggml_set_name(hidden_out, "hidden_out");

    ggml_cgraph * gf = ggml_new_graph_custom(ctx.get(), 8192, false);
    // KV slot writes first: they alias the cache views the attention reads, so
    // they must precede those nodes in graph order.
    for (ggml_tensor * wb : kv_writebacks) {
        ggml_build_forward_expand(gf, wb);
    }
    ggml_build_forward_expand(gf, logits_out);
    ggml_build_forward_expand(gf, hidden_out);
    for (ggml_tensor * wb : writebacks) {
        ggml_build_forward_expand(gf, wb);
    }
    auto t_graph = Clock::now();
    ggml_gallocr_t gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!gallocr || !ggml_gallocr_reserve(gallocr, gf) || !ggml_gallocr_alloc_graph(gallocr, gf)) {
        throw std::runtime_error("falcon_forward_step: gallocr failed");
    }
    auto t_alloc = Clock::now();

    // ---- feed host constants ----
    if (zero_copy) {
        // Everything except the rope position is bound to host memory already;
        // the graph reads the position straight from the state struct.
        state.pos_value = static_cast<int32_t>(position);
    } else {
    ggml_backend_tensor_set(cur, embedding.data(), 0, embedding.size() * sizeof(float));
    {
        const int32_t ids0 = 0;
        const int32_t posv = static_cast<int32_t>(position);
        for (int64_t li = 0; li < n_layer; ++li) {
            ggml_backend_tensor_set(ids_ts[static_cast<size_t>(li)], &ids0, 0, sizeof(int32_t));
            ggml_backend_tensor_set(pos_ts[static_cast<size_t>(li)], &posv, 0, sizeof(int32_t));
        }
    }
    for (int64_t li = 0; li < n_layer; ++li) {
        const auto & layer = weights.falcon_layers[static_cast<size_t>(li)];
        if (!layer.input_layernorm.values.empty()) {
            ggml_backend_tensor_set(ln_w_ts[static_cast<size_t>(li)], layer.input_layernorm.values.data(), 0,
                                    layer.input_layernorm.values.size() * sizeof(float));
        } else {
            std::vector<float> ones(static_cast<size_t>(dim), 1.0F);
            ggml_backend_tensor_set(ln_w_ts[static_cast<size_t>(li)], ones.data(), 0, ones.size() * sizeof(float));
        }
        if (!layer.pre_ff_layernorm.values.empty()) {
            ggml_backend_tensor_set(pre_w_ts[static_cast<size_t>(li)], layer.pre_ff_layernorm.values.data(), 0,
                                    layer.pre_ff_layernorm.values.size() * sizeof(float));
        } else {
            std::vector<float> ones(static_cast<size_t>(dim), 1.0F);
            ggml_backend_tensor_set(pre_w_ts[static_cast<size_t>(li)], ones.data(), 0, ones.size() * sizeof(float));
        }
        // conv state [d_conv-1, conv_dim] (col-major: element (c,r) at r*conv_dim + c)
        const auto & cstate = state.conv_states[static_cast<size_t>(li)];
        ggml_backend_tensor_set(conv_st_ts[static_cast<size_t>(li)], cstate.data(), 0, cstate.size() * sizeof(float));
        // ssm state [d_state, mamba_head_dim, n_mamba_heads]
        const auto & sstate = state.ssm_states[static_cast<size_t>(li)];
        ggml_backend_tensor_set(ssm_st_ts[static_cast<size_t>(li)], sstate.data(), 0, sstate.size() * sizeof(float));
        // A = -exp(A_log)
        {
            std::vector<float> a_log(static_cast<size_t>(n_mamba_heads));
            ggml_backend_tensor_get(layer.ssm_A.tensor, a_log.data(), 0, a_log.size() * sizeof(float));
            std::vector<float> av(static_cast<size_t>(n_mamba_heads));
            for (int64_t h = 0; h < n_mamba_heads; ++h) {
                av[static_cast<size_t>(h)] = -std::exp(a_log[static_cast<size_t>(h)]);
            }
            ggml_backend_tensor_set(A_ts[static_cast<size_t>(li)], av.data(), 0, av.size() * sizeof(float));
        }
        // D expanded
        {
            std::vector<float> d_raw(static_cast<size_t>(n_mamba_heads));
            ggml_backend_tensor_get(layer.ssm_D.tensor, d_raw.data(), 0, d_raw.size() * sizeof(float));
            std::vector<float> dv(static_cast<size_t>(d_inner));
            for (int64_t h = 0; h < n_mamba_heads; ++h) {
                for (int64_t d = 0; d < mamba_head_dim; ++d) {
                    dv[static_cast<size_t>(d + h * mamba_head_dim)] = d_raw[static_cast<size_t>(h)];
                }
            }
            ggml_backend_tensor_set(D_ts[static_cast<size_t>(li)], dv.data(), 0, dv.size() * sizeof(float));
        }
        // conv bias
        if (!layer.ssm_conv1d_b.values.empty()) {
            ggml_backend_tensor_set(conv_b_ts[static_cast<size_t>(li)], layer.ssm_conv1d_b.values.data(), 0,
                                    layer.ssm_conv1d_b.values.size() * sizeof(float));
        } else {
            std::vector<float> zeros(static_cast<size_t>(conv_dim), 0.0F);
            ggml_backend_tensor_set(conv_b_ts[static_cast<size_t>(li)], zeros.data(), 0, zeros.size() * sizeof(float));
        }
        // conv1d weight: host [d_conv, conv_dim] kernel (GGUF layout, no flip)
        {
            const auto & cw = layer.conv1d_kernel;
            ggml_backend_tensor_set(conv_w2_ts[static_cast<size_t>(li)], cw.values.data(), 0, cw.values.size() * sizeof(float));
        }
        // KV cache
        const auto & kc = state.k_cache[static_cast<size_t>(li)];
        const auto & vc = state.v_cache[static_cast<size_t>(li)];
        if (!kc.empty()) ggml_backend_tensor_set(k_cache_ts[static_cast<size_t>(li)], kc.data(), 0, kc.size() * sizeof(float));
        if (!vc.empty()) ggml_backend_tensor_set(v_cache_ts[static_cast<size_t>(li)], vc.data(), 0, vc.size() * sizeof(float));
    }
    if (!weights.slow_norm.values.empty()) {
        ggml_backend_tensor_set(final_w, weights.slow_norm.values.data(), 0, weights.slow_norm.values.size() * sizeof(float));
    } else {
        std::vector<float> ones(static_cast<size_t>(dim), 1.0F);
        ggml_backend_tensor_set(final_w, ones.data(), 0, ones.size() * sizeof(float));
    }
    }

    core::set_backend_threads(backend, threads);
    auto t_upload = Clock::now();
    ggml_status status = core::compute_backend_graph(backend, gf, nullptr, "falcon_forward_step");
    ggml_backend_synchronize(backend);
    auto t_compute = Clock::now();
    if (status != GGML_STATUS_SUCCESS) {
        ggml_gallocr_free(gallocr);
        throw std::runtime_error("falcon_forward_step compute failed");
    }

    // ---- read outputs + update states ----
    SlowForwardOutput out;
    out.logits.resize(static_cast<size_t>(vocab));
    out.hidden.resize(static_cast<size_t>(dim));
    ggml_backend_tensor_get(logits_out, out.logits.data(), 0, static_cast<size_t>(vocab) * sizeof(float));
    ggml_backend_tensor_get(hidden_out, out.hidden.data(), 0, static_cast<size_t>(dim) * sizeof(float));

    // conv/ssm states: updated in-graph on host backends; on GPU backends the
    // host reads the new state tails back and shifts them into the vectors.
    if (!zero_copy) {
    // conv state: last d_conv-1 kernel rows of sx (per layer). sx is col-major
    // [d_conv, conv_dim], element (k, c) at k + d_conv*c.
    {
        std::vector<float> sx_vals(static_cast<size_t>(d_conv * conv_dim));
        for (int64_t li = 0; li < n_layer; ++li) {
            ggml_backend_tensor_get(sx_ts[static_cast<size_t>(li)], sx_vals.data(), 0, sx_vals.size() * sizeof(float));
            auto & cstate = state.conv_states[static_cast<size_t>(li)];
            for (int64_t r = 0; r < d_conv - 1; ++r) {
                for (int64_t c = 0; c < conv_dim; ++c) {
                    cstate[r * conv_dim + c] = sx_vals[(r + 1) + d_conv * c];
                }
            }
        }
    }

    // ssm state: tail of scan output (d_state*d_inner per layer) — per-layer tensors!
    {
        const size_t y_sz = static_cast<size_t>(d_inner);
        const size_t s_sz = static_cast<size_t>(d_state * d_inner);
        std::vector<float> scan_vals(y_sz + s_sz);
        for (int64_t li = 0; li < n_layer; ++li) {
            ggml_backend_tensor_get(scan_ts[static_cast<size_t>(li)], scan_vals.data(), 0, scan_vals.size() * sizeof(float));
            auto & sstate = state.ssm_states[static_cast<size_t>(li)];
            for (size_t i = 0; i < s_sz; ++i) {
                sstate[i] = scan_vals[y_sz + i];
            }
        }
    }
    }

    // KV cache append (fallback path only; the zero-copy path wrote the fresh
    // k/v into the padded caches in-graph). ggml col-major layout
    // [head_dim, seq, n_kv, 1]: element (d, t, h) at
    // d + head_dim*(t + new_seq_len*h). The freshly projected/roped k/v read
    // back as [d + head_dim*h] (128 values).
    if (!zero_copy) {
        std::vector<float> kv(static_cast<size_t>(n_kv * head_dim));
        for (int64_t li = 0; li < n_layer; ++li) {
            ggml_backend_tensor_get(k_cur_ts[static_cast<size_t>(li)], kv.data(), 0, kv.size() * sizeof(float));
            append_falcon_kv_token(state.k_cache[static_cast<size_t>(li)], seq, n_kv, head_dim, kv.data());
            ggml_backend_tensor_get(v_cur_ts[static_cast<size_t>(li)], kv.data(), 0, kv.size() * sizeof(float));
            append_falcon_kv_token(state.v_cache[static_cast<size_t>(li)], seq, n_kv, head_dim, kv.data());
        }
    }
    state.seq_len = seq + 1;

    if (profile != nullptr) {
        profile->falcon_step_init_ms += engine::debug::elapsed_ms(t_init, t_build);
        profile->falcon_step_build_ms += engine::debug::elapsed_ms(t_build, t_graph);
        profile->falcon_step_gallocr_ms += engine::debug::elapsed_ms(t_graph, t_alloc);
        profile->falcon_step_upload_ms += engine::debug::elapsed_ms(t_alloc, t_upload);
        profile->falcon_step_compute_ms += engine::debug::elapsed_ms(t_upload, t_compute);
        profile->falcon_step_download_ms += engine::debug::elapsed_ms(t_compute, Clock::now());
        profile->falcon_step_runs += 1;
    }

    ggml_gallocr_free(gallocr);
    core::release_backend_graph_resources(backend, gf);
    return out;
}

// Single-token Falcon embedding: text_embedding(semantic) * embedding_multiplier
// + sum(codebook_embeddings) for semantic tokens. `matrix` is [codebook_rows][steps].
std::vector<float> build_falcon_embedding_step(
    const Audio8TtsConfig & config,
    const ArkttsARWeights & weights,
    const int32_t * matrix,
    int64_t steps,
    int64_t step) {
    const int64_t hidden = config.text.dim;
    const int64_t codebook_rows = config.fast.num_codebooks + 1;
    const int32_t token = matrix[step];
    std::vector<float> out(static_cast<size_t>(hidden), 0.0F);
    auto row = lookup_row(weights.text_embedding_host, token, hidden);
    if (token >= config.semantic_start_token_id && token <= config.semantic_end_token_id) {
        for (int64_t cb = 0; cb < config.fast.num_codebooks; ++cb) {
            const int32_t code = matrix[(cb + 1) * steps + step];
            add_row(weights.codebook_embedding_host, cb * config.fast.vocab_size + code, hidden, row);
        }
    }
    // HF _embed + _slow_backbone: (text_emb + codebook_sum) * embedding_multiplier.
    for (auto & v : row) v *= config.text.embedding_multiplier;
    std::copy(row.begin(), row.end(), out.begin());
    return out;
}


}  // namespace

// Copies a backend-resident weight tensor byte-for-byte (same ggml type and
// dimensions) so it can live on a second backend: the fast AR graph runs on a
// dedicated CPU backend while the slow path and codec stay on the GPU (the
// per-step fast AR submit+sync latency dominates on GPU backends, while CPU
// computes the same graph several times faster). q8_0/f32/f16 all copy
// losslessly — the point is backend placement, not conversion.
core::TensorValue schedule_tensor_copy(
    ggml_context * dst_ctx,
    const core::TensorValue & src) {
    ggml_tensor * dst = ggml_new_tensor(
        dst_ctx, src.tensor->type, ggml_n_dims(src.tensor), src.tensor->ne);
    return core::wrap_tensor(dst, src.shape, src.type);
}

void copy_tensor_bytes(const core::TensorValue & src, const core::TensorValue & dst) {
    const size_t bytes = static_cast<size_t>(ggml_nbytes(src.tensor));
    std::vector<uint8_t> host(bytes);
    ggml_backend_tensor_get(src.tensor, host.data(), 0, bytes);
    ggml_backend_tensor_set(dst.tensor, host.data(), 0, bytes);
}

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
        ArkttsARWeights loaded =
            load_ar_weights(*assets_, backend_, backend_type_, weight_context_bytes, weight_storage_type);
        if (backend_type_ != core::BackendType::Cpu) {
            // Fast AR is submit+sync latency bound on GPU backends (one graph
            // submission per generated codebook token); the same graph computes
            // several times faster on CPU. Give it a dedicated CPU backend and
            // move the fast-layer weights over, leaving slow path + codec on
            // the GPU backend.
            core::BackendConfig fast_backend_config;
            fast_backend_config.type = core::BackendType::Cpu;
            fast_backend_config.threads = threads_;
            fast_backend_ = core::init_backend(fast_backend_config);
            fast_backend_type_ = core::backend_type(fast_backend_);
            retarget_fast_weights(loaded);
            if (!loaded.falcon_layers.empty()) {
                // The Falcon-H1 per-token step graph is ~600 tiny nodes; on Metal it
                // is dispatch-latency bound (measured 5.9 ms/step vs 2.0 ms on CPU,
                // same weights). Run it on the same dedicated CPU backend.
                retarget_falcon_weights(loaded);
            }
        }
        weights_ = std::make_shared<const ArkttsARWeights>(std::move(loaded));
        slow_step_constants_ = std::make_unique<core::ConstantTensorCache>(
            backend_,
            threads_,
            "audio8_tts.ar.step.constants",
            256ull * 1024ull * 1024ull);
        fast_constants_ = std::make_unique<core::ConstantTensorCache>(
            fast_backend(),
            threads_,
            "audio8_tts.ar.fast.constants",
            256ull * 1024ull * 1024ull);
    }

    ~ArkttsARWeightsRuntime() {
        fast_constants_.reset();
        slow_step_constants_.reset();
        weights_.reset();
        if (falcon_weight_buffer_ != nullptr) {
            ggml_backend_buffer_free(falcon_weight_buffer_);
        }
        falcon_weight_ctx_.reset();
        if (fast_weight_buffer_ != nullptr) {
            ggml_backend_buffer_free(fast_weight_buffer_);
        }
        fast_weight_ctx_.reset();
        if (fast_backend_ != nullptr) {
            ggml_backend_free(fast_backend_);
        }
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

    // Backend hosting the fast AR graph: a dedicated CPU backend when the main
    // backend is a GPU, otherwise the main backend itself.
    ggml_backend_t fast_backend() const noexcept {
        return fast_backend_ != nullptr ? fast_backend_ : backend_;
    }

    // Falcon-H1 per-token steps run on the dedicated CPU backend when the main
    // backend is a GPU one (dispatch-latency bound there); identical tensors
    // otherwise, so this is always the right backend for falcon_forward_step.
    ggml_backend_t falcon_step_backend() const noexcept {
        return fast_backend_ != nullptr ? fast_backend_ : backend_;
    }

    core::BackendType fast_backend_type() const noexcept {
        return fast_backend_ != nullptr ? fast_backend_type_ : backend_type_;
    }

    core::ConstantTensorCache & slow_step_constants() const noexcept {
        return *slow_step_constants_;
    }

    core::ConstantTensorCache & fast_constants() const noexcept {
        return *fast_constants_;
    }

private:
    // Re-binds the fast AR layer weights (and fast_output) onto the dedicated
    // CPU fast backend. The projections are byte copies of the tensors the
    // weight store uploaded to the main backend; norms stay host TensorData
    // and upload through the (CPU-backed) fast constants cache at graph build.
    void retarget_fast_weights(ArkttsARWeights & weights) {
        ggml_init_params params{8ull * 1024ull * 1024ull, nullptr, true};
        fast_weight_ctx_.reset(ggml_init(params));
        if (fast_weight_ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize Audio8 TTS fast AR CPU weight context");
        }
        struct ScheduledCopy {
            const core::TensorValue * source;
            core::TensorValue target;
        };
        std::vector<ScheduledCopy> copies;
        copies.reserve(weights.fast_layers.size() * 5 + 1);
        auto schedule = [&](const core::TensorValue & value) {
            if (!value.valid()) {
                throw std::runtime_error("Audio8 TTS fast AR weight tensor is missing");
            }
            copies.push_back({&value, schedule_tensor_copy(fast_weight_ctx_.get(), value)});
        };
        for (const auto & layer : weights.fast_layers) {
            schedule(layer.qkv_proj);
            if (layer.qkv_bias.has_value()) {
                schedule(*layer.qkv_bias);
            }
            schedule(layer.o_proj);
            schedule(layer.gate_up_proj);
            schedule(layer.down_proj);
        }
        schedule(weights.fast_output);
        fast_weight_buffer_ = ggml_backend_alloc_ctx_tensors(fast_weight_ctx_.get(), fast_backend_);
        if (fast_weight_buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate Audio8 TTS fast AR CPU weights");
        }
        for (auto & copy : copies) {
            copy_tensor_bytes(*copy.source, copy.target);
        }
        size_t index = 0;
        auto commit = [&](core::TensorValue & value) {
            if (!value.valid()) {
                return;
            }
            value = std::move(copies[index].target);
            ++index;
        };
        for (auto & layer : weights.fast_layers) {
            commit(layer.qkv_proj);
            if (layer.qkv_bias.has_value()) {
                commit(*layer.qkv_bias);
            }
            commit(layer.o_proj);
            commit(layer.gate_up_proj);
            commit(layer.down_proj);
        }
        commit(weights.fast_output);
    }

    // Re-binds the Falcon-H1 layer weights (and the semantic head) onto the
    // dedicated CPU backend, mirroring retarget_fast_weights. ssm_A / ssm_D are
    // included so the per-step A=-exp(A_log) / D-expansion reads become plain
    // CPU memcpys instead of GPU->host syncs.
    void retarget_falcon_weights(ArkttsARWeights & weights) {
        ggml_init_params params{8ull * 1024ull * 1024ull, nullptr, true};
        falcon_weight_ctx_.reset(ggml_init(params));
        if (falcon_weight_ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize Audio8 TTS Falcon step CPU weight context");
        }
        struct ScheduledCopy {
            const core::TensorValue * source;
            core::TensorValue target;
        };
        std::vector<ScheduledCopy> copies;
        copies.reserve(weights.falcon_layers.size() * 12 + 1);
        auto schedule = [&](const core::TensorValue & value) {
            if (!value.valid()) {
                throw std::runtime_error("Audio8 TTS Falcon step weight tensor is missing");
            }
            copies.push_back({&value, schedule_tensor_copy(falcon_weight_ctx_.get(), value)});
        };
        for (const auto & layer : weights.falcon_layers) {
            schedule(layer.ssm_in);
            schedule(layer.ssm_dt_b);
            schedule(layer.ssm_A);
            schedule(layer.ssm_D);
            schedule(layer.ssm_out);
            schedule(layer.attn_q_proj);
            schedule(layer.attn_k_proj);
            schedule(layer.attn_v_proj);
            schedule(layer.attn_o_proj);
            schedule(layer.ffn_gate);
            schedule(layer.ffn_up);
            schedule(layer.ffn_down);
        }
        schedule(weights.falcon_lm_head);
        falcon_weight_buffer_ = ggml_backend_alloc_ctx_tensors(falcon_weight_ctx_.get(), fast_backend_);
        if (falcon_weight_buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate Audio8 TTS Falcon step CPU weights");
        }
        for (auto & copy : copies) {
            copy_tensor_bytes(*copy.source, copy.target);
        }
        size_t index = 0;
        auto commit = [&](core::TensorValue & value) {
            value = std::move(copies[index].target);
            ++index;
        };
        for (auto & layer : weights.falcon_layers) {
            commit(layer.ssm_in);
            commit(layer.ssm_dt_b);
            commit(layer.ssm_A);
            commit(layer.ssm_D);
            commit(layer.ssm_out);
            commit(layer.attn_q_proj);
            commit(layer.attn_k_proj);
            commit(layer.attn_v_proj);
            commit(layer.attn_o_proj);
            commit(layer.ffn_gate);
            commit(layer.ffn_up);
            commit(layer.ffn_down);
        }
        commit(weights.falcon_lm_head);
    }

    std::shared_ptr<const Audio8TtsAssets> assets_;
    std::shared_ptr<const ArkttsARWeights> weights_;
    int threads_ = 1;
    size_t graph_arena_bytes_ = 0;
    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    ggml_backend_t fast_backend_ = nullptr;
    core::BackendType fast_backend_type_ = core::BackendType::Cpu;
    std::unique_ptr<ggml_context, GgmlContextDeleter> fast_weight_ctx_;
    ggml_backend_buffer_t fast_weight_buffer_ = nullptr;
    std::unique_ptr<ggml_context, GgmlContextDeleter> falcon_weight_ctx_;
    ggml_backend_buffer_t falcon_weight_buffer_ = nullptr;
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
            // Falcon-H1 0.1B — native ggml path. Stateful Mamba2 + hybrid GQA
            // attention single-token forward (falcon_forward_step), mirroring
            // transformers.models.falcon_h1 FalconH1DecoderLayer and llama.cpp
            // mamba-base.cpp build_mamba2_layer. Prefill runs each prompt token
            // through the step graph to populate conv/ssm states and the KV
            // cache; generation continues token by token (O(N) per step instead
            // of the former O(N^2) stateless recompute).
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
            FalconH1StepState fstate = init_falcon_step_state(assets.config);
            SlowForwardOutput pre_out;
            for (int64_t p = 0; p < cur_steps; ++p) {
                auto p_emb = build_falcon_embedding_step(assets.config, weights, full_matrix.data(), cur_steps, p);
                pre_out = falcon_forward_step(runtime_->falcon_step_backend(), runtime_->threads(), runtime_->graph_arena_bytes(),
                                              assets.config, weights, p_emb, fstate, p, &profile);
            }
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
                const int64_t pos = cur_steps - 1;
                auto emb = build_falcon_embedding_step(assets.config, weights, full_matrix.data(), cur_steps, pos);
                auto out = falcon_forward_step(runtime_->falcon_step_backend(), runtime_->threads(), runtime_->graph_arena_bytes(),
                                               assets.config, weights, emb, fstate, pos, &profile);
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
                runtime_->fast_backend_type() == core::BackendType::Vulkan ? GGML_TYPE_F32 : GGML_TYPE_BF16;
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
            state_buffer_ = ggml_backend_alloc_ctx_tensors(state_ctx_.get(), runtime_->fast_backend());
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

            core::ModuleBuildContext ctx{graph_ctx_.get(), "audio8_tts.ar.fast", runtime_->fast_backend_type()};
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
                make_fast_decoder_config(config, runtime_->fast_backend_type()),
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
            gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(runtime_->fast_backend()));
            if (gallocr_ == nullptr ||
                !ggml_gallocr_reserve(gallocr_, graph_) ||
                !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
                throw std::runtime_error("failed to allocate Audio8 TTS fast AR graph");
            }
            mask_scratch_.assign(static_cast<size_t>(config.num_codebooks), ggml_fp32_to_fp16(-INFINITY));
        }

        ~FastGraph() {
            core::release_backend_graph_resources(runtime_->fast_backend(), graph_);
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
            core::set_backend_threads(runtime_->fast_backend(), runtime_->threads());
            timing_start = Clock::now();
            const ggml_status status = core::compute_backend_graph(runtime_->fast_backend(), graph_, nullptr, "audio8_tts.ar.fast");
            ggml_backend_synchronize(runtime_->fast_backend());
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
        engine::debug::timing_log_scalar("audio8_tts.ar.profile.falcon_step_init_ms", profile.falcon_step_init_ms);
        engine::debug::timing_log_scalar("audio8_tts.ar.profile.falcon_step_build_ms", profile.falcon_step_build_ms);
        engine::debug::timing_log_scalar("audio8_tts.ar.profile.falcon_step_gallocr_ms", profile.falcon_step_gallocr_ms);
        engine::debug::timing_log_scalar("audio8_tts.ar.profile.falcon_step_upload_ms", profile.falcon_step_upload_ms);
        engine::debug::timing_log_scalar("audio8_tts.ar.profile.falcon_step_compute_ms", profile.falcon_step_compute_ms);
        engine::debug::timing_log_scalar("audio8_tts.ar.profile.falcon_step_download_ms", profile.falcon_step_download_ms);
        engine::debug::timing_log_scalar("audio8_tts.ar.profile.falcon_step_runs", profile.falcon_step_runs);
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
