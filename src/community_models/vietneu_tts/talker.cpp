#include "engine/community_models/vietneu_tts/talker.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/transformers/qwen_causal_decoder.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"
#include "engine/framework/runtime/kv_cache.h"
#include "engine/framework/sampling/torch_random.h"

#include "engine/framework/core/constant_tensor_cache.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace engine::models::vietneu_tts {
namespace {

using Clock = std::chrono::steady_clock;
namespace binding = modules::binding;

constexpr int64_t kInitialGeneratedStepCacheFrames = 128;
constexpr size_t kTalkerWeightContextBytes = 64ull * 1024ull * 1024ull;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct LinearTensorWeights {
    assets::TensorData weight;
    std::optional<assets::TensorData> bias;
};

struct GraphLinearTensorWeights {
    core::TensorValue weight;
    std::optional<core::TensorValue> bias;
};

struct TalkerLayerWeights {
    assets::TensorDataF32 input_norm;
    core::TensorValue q_proj;
    core::TensorValue k_proj;
    core::TensorValue v_proj;
    core::TensorValue o_proj;
    assets::TensorDataF32 q_norm;
    assets::TensorDataF32 k_norm;
    assets::TensorDataF32 post_norm;
    core::TensorValue gate_proj;
    core::TensorValue up_proj;
    core::TensorValue down_proj;
};

struct CodePredictorWeights {
    std::vector<TalkerLayerWeights> layers;
    assets::TensorDataF32 norm;
    std::vector<core::TensorValue> lm_heads;
    std::optional<GraphLinearTensorWeights> small_to_mtp_projection;
};

struct VietneuTalkerWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    assets::TensorData codec_embedding;
    std::vector<assets::TensorData> code_predictor_embeddings;
    CodePredictorWeights code_predictor;
    assets::TensorData text_embedding;
    LinearTensorWeights text_projection_fc1;
    LinearTensorWeights text_projection_fc2;
    std::vector<TalkerLayerWeights> layers;
    assets::TensorDataF32 norm;
    core::TensorValue codec_head;

    // VieNeu extensions
    std::optional<assets::TensorData> slot_pos_emb;
    std::optional<LinearTensorWeights> xvec_proj_linear;
    std::optional<assets::TensorDataF32> xvec_proj_ln_weight;
    std::optional<assets::TensorDataF32> xvec_proj_ln_bias;
};

struct PromptEmbeddingState {
    std::vector<float> prompt;
    std::vector<float> trailing_text;
    std::vector<float> tts_pad;
};

bool speech_codes_equal(const Qwen3SpeechCodes & lhs, const Qwen3SpeechCodes & rhs) {
    return lhs.frames == rhs.frames &&
           lhs.code_groups == rhs.code_groups &&
           lhs.codes == rhs.codes;
}

bool speaker_embedding_equal(const VietneuSpeakerEmbedding & lhs, const VietneuSpeakerEmbedding & rhs) {
    return lhs.dims == rhs.dims &&
           lhs.values == rhs.values;
}

bool talker_prefill_equal(const VietneuTalkerPrefill & lhs, const VietneuTalkerPrefill & rhs) {
    if (lhs.prompt_mode != rhs.prompt_mode ||
        lhs.input_ids != rhs.input_ids ||
        lhs.instruct_ids != rhs.instruct_ids ||
        lhs.reference_ids != rhs.reference_ids ||
        lhs.speaker != rhs.speaker ||
        lhs.language != rhs.language ||
        lhs.icl_mode != rhs.icl_mode ||
        lhs.x_vector_only_mode != rhs.x_vector_only_mode ||
        lhs.reference_codes.has_value() != rhs.reference_codes.has_value() ||
        lhs.speaker_embedding.has_value() != rhs.speaker_embedding.has_value()) {
        return false;
    }
    if (lhs.reference_codes.has_value() && !speech_codes_equal(*lhs.reference_codes, *rhs.reference_codes)) {
        return false;
    }
    if (lhs.speaker_embedding.has_value() &&
        !speaker_embedding_equal(*lhs.speaker_embedding, *rhs.speaker_embedding)) {
        return false;
    }
    return true;
}

struct VietneuTalkerPrefillLogits {
    std::vector<float> values;
    int64_t vocab_size = 0;
};

struct VietneuTalkerPrefillResult {
    VietneuTalkerPrefillLogits logits;
    VietneuSpeakerEmbedding last_hidden;
};

struct VietneuTalkerCodePredictorInput {
    VietneuSpeakerEmbedding talker_hidden;
    int32_t first_code = 0;
};

struct VietneuTalkerFrameCodes {
    std::vector<int32_t> codes;
};

struct CachedStepTiming {
    double input_upload_ms = 0.0;
    double mask_upload_ms = 0.0;
    double graph_compute_ms = 0.0;
    double output_read_ms = 0.0;
    double kv_copy_ms = 0.0;
};

struct CodePredictorTiming {
    double input_upload_ms = 0.0;
    double graph_compute_ms = 0.0;
    double output_read_ms = 0.0;
};

std::string ascii_lower(std::string value) {
    for (char & ch : value) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return value;
}

template <typename Config>
int64_t attention_head_dim(const Config & config) {
    if (config.num_attention_heads <= 0 || config.num_key_value_heads <= 0 || config.head_dim <= 0) {
        throw std::runtime_error("VieNeu-TTS talker attention configuration is invalid");
    }
    return config.head_dim;
}

core::TensorValue cache_view(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & cache,
    int64_t start,
    int64_t steps,
    int64_t heads,
    int64_t head_dim) {
    if (start < 0 || steps <= 0 || start + steps > cache.shape.dims[1]) {
        throw std::runtime_error("VieNeu-TTS talker cache view range is invalid");
    }
    return core::wrap_tensor(
        ggml_view_4d(
            ctx.ggml,
            cache.tensor,
            head_dim,
            heads,
            steps,
            1,
            cache.tensor->nb[1],
            cache.tensor->nb[2],
            cache.tensor->nb[3],
            static_cast<size_t>(start) * cache.tensor->nb[2]),
        core::TensorShape::from_dims({1, steps, heads, head_dim}),
        GGML_TYPE_F32);
}

template <typename Config>
modules::QwenCausalDecoderConfig make_qwen_decoder_config(
    const Config & config,
    int64_t logits_size,
    bool is_vieneu = false,
    bool is_acoustic_decoder = false) {
    modules::QwenCausalDecoderConfig out;
    out.stack.hidden_size = config.hidden_size;
    out.stack.num_attention_heads = config.num_attention_heads;
    out.stack.num_key_value_heads = config.num_key_value_heads;
    out.stack.head_dim = attention_head_dim(config);
    out.stack.intermediate_size = config.intermediate_size;
    out.stack.layers = config.num_hidden_layers;
    out.stack.rms_norm_eps = config.rms_norm_eps;
    out.stack.rope_theta = config.rope_theta;
    if (is_vieneu && is_acoustic_decoder) {
        out.stack.position_encoding = modules::QwenDecoderPositionEncoding::None; // Bypass RoPE
        out.stack.qkv_layout = modules::QwenDecoderQKVLayout::PackedQKV;
    }
    out.stack.attention_precision = GGML_PREC_F32;
    out.stack.use_qk_norm = true;
    out.stack.runtime.static_cache.update_mode = modules::QwenDecoderStaticCacheUpdateMode::DirectSetRows;
    out.logits_size = logits_size;
    out.logits_mode = modules::QwenCausalDecoderLogitsMode::LastStep;
    return out;
}

modules::QwenDecoderLayerWeights make_qwen_decoder_layer_weights(
    core::ConstantTensorCache & constants,
    const TalkerLayerWeights & weights,
    bool is_vieneu = false) {
    modules::QwenDecoderLayerWeights out;
    out.input_norm = binding::norm_data(constants, weights.input_norm);
    if (is_vieneu) {
        out.self_attention.qkv_weight = binding::tensor_data(constants, weights.q_proj);
    } else {
        out.self_attention.q_weight = binding::tensor_data(constants, weights.q_proj);
        out.self_attention.k_weight = binding::tensor_data(constants, weights.k_proj);
        out.self_attention.v_weight = binding::tensor_data(constants, weights.v_proj);
    }
    out.self_attention.out_weight = binding::tensor_data(constants, weights.o_proj);
    out.q_norm = binding::norm_data(constants, weights.q_norm);
    out.k_norm = binding::norm_data(constants, weights.k_norm);
    out.post_norm = binding::norm_data(constants, weights.post_norm);
    out.mlp.gate_proj = binding::linear_data(constants, weights.gate_proj);
    out.mlp.up_proj = binding::linear_data(constants, weights.up_proj);
    out.mlp.down_proj = binding::linear_data(constants, weights.down_proj);
    return out;
}

modules::QwenCausalDecoderWeights make_qwen_decoder_weights(
    core::ConstantTensorCache & constants,
    const std::vector<TalkerLayerWeights> & layers,
    const assets::TensorDataF32 & norm,
    const core::TensorValue & lm_head,
    bool is_vieneu = false) {
    modules::QwenCausalDecoderWeights out;
    out.stack.layers.reserve(layers.size());
    for (const auto & layer : layers) {
        out.stack.layers.push_back(make_qwen_decoder_layer_weights(constants, layer, is_vieneu));
    }
    out.final_norm = binding::norm_data(constants, norm);
    out.lm_head = binding::linear_data(constants, lm_head);
    return out;
}

float silu(float value) {
    return value / (1.0F + std::exp(-value));
}

std::vector<float> linear_host(
    const std::vector<float> & input,
    int64_t rows,
    int64_t in_features,
    const LinearTensorWeights & weights) {
    if (weights.weight.shape.dims[0] <= 0 || weights.weight.shape.dims[1] != in_features) {
        throw std::runtime_error("VieNeu-TTS talker host linear weight shape mismatch");
    }
    const int64_t out_features = weights.weight.shape.dims[0];
    const auto weight_values = assets::tensor_data_to_f32("VieNeu-TTS talker host linear weight", weights.weight);
    const auto bias_values = weights.bias.has_value()
        ? assets::tensor_data_to_f32("VieNeu-TTS talker host linear bias", *weights.bias)
        : std::vector<float>{};
    std::vector<float> output(static_cast<size_t>(rows * out_features), 0.0F);
    for (int64_t row = 0; row < rows; ++row) {
        for (int64_t out = 0; out < out_features; ++out) {
            float sum = weights.bias.has_value() ? bias_values[static_cast<size_t>(out)] : 0.0F;
            for (int64_t in = 0; in < in_features; ++in) {
                sum += weight_values[static_cast<size_t>(out * in_features + in)] *
                    input[static_cast<size_t>(row * in_features + in)];
            }
            output[static_cast<size_t>(row * out_features + out)] = sum;
        }
    }
    return output;
}

std::vector<float> text_project_host(
    const std::vector<float> & text_hidden,
    int64_t rows,
    const VietneuTalkerWeights & weights,
    const VietneuTTSTalkerConfig & config) {
    if (weights.text_projection_fc1.weight.bytes.empty()) {
        return text_hidden;
    }
    auto hidden = linear_host(text_hidden, rows, config.text_hidden_size, weights.text_projection_fc1);
    for (float & value : hidden) {
        value = silu(value);
    }
    return linear_host(hidden, rows, config.text_hidden_size, weights.text_projection_fc2);
}

core::TensorValue project_code_predictor_input(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const VietneuTalkerWeights & weights,
    const VietneuTTSConfig & config,
    core::ConstantTensorCache & constants) {
    if (config.code_predictor.hidden_size == config.talker.hidden_size) {
        return input;
    }
    if (!weights.code_predictor.small_to_mtp_projection.has_value()) {
        throw std::runtime_error("VieNeu-TTS code predictor requires small_to_mtp_projection weights");
    }
    const auto & projection = *weights.code_predictor.small_to_mtp_projection;
    return modules::LinearModule({config.talker.hidden_size, config.code_predictor.hidden_size, projection.bias.has_value()})
        .build(ctx, input, binding::linear_data(constants, projection.weight, projection.bias));
}

std::vector<float> lookup_rows(
    const assets::TensorData & table,
    int64_t dim,
    const std::vector<int32_t> & ids) {
    if (table.shape.rank != 2 || table.shape.dims[1] != dim) {
        throw std::runtime_error("VieNeu-TTS talker embedding table shape mismatch");
    }
    const size_t row_bytes = ggml_row_size(table.type, dim);
    if (table.bytes.size() != static_cast<size_t>(table.shape.dims[0]) * row_bytes) {
        throw std::runtime_error("VieNeu-TTS talker embedding table byte size mismatch");
    }
    std::vector<float> output(static_cast<size_t>(ids.size() * static_cast<size_t>(dim)), 0.0F);
    for (size_t row = 0; row < ids.size(); ++row) {
        const int32_t id = ids[row];
        if (id == table.shape.dims[0]) {
            auto * dst = output.data() + row * static_cast<size_t>(dim);
            std::fill(dst, dst + dim, 0.0f);
            continue;
        }
        if (id < 0 || id >= table.shape.dims[0]) {
            std::fprintf(stderr, "DEBUG: lookup_rows failed. table.shape.dims[0]=%d, id=%d, row=%zu, total_ids=%zu\n",
                         static_cast<int>(table.shape.dims[0]), static_cast<int>(id), row, ids.size());
            std::fprintf(stderr, "DEBUG: ids: ");
            for (auto item : ids) {
                std::fprintf(stderr, "%d ", item);
            }
            std::fprintf(stderr, "\n");
            throw std::runtime_error("VieNeu-TTS talker embedding id out of range");
        }
        auto * dst = output.data() + row * static_cast<size_t>(dim);
        const size_t offset = static_cast<size_t>(id) * row_bytes;
        const auto * src = table.bytes.data() + static_cast<std::ptrdiff_t>(offset);
        if (table.type == GGML_TYPE_F32) {
            const auto * values = reinterpret_cast<const float *>(src);
            std::copy(values, values + dim, dst);
        } else if (table.type == GGML_TYPE_F16) {
            ggml_fp16_to_fp32_row(reinterpret_cast<const ggml_fp16_t *>(src), dst, dim);
        } else if (table.type == GGML_TYPE_BF16) {
            ggml_bf16_to_fp32_row(reinterpret_cast<const ggml_bf16_t *>(src), dst, dim);
        } else {
            throw std::runtime_error("VieNeu-TTS talker embedding table must be f32, f16, or bf16");
        }
    }
    return output;
}

void append_rows(std::vector<float> & dst, const std::vector<float> & rows) {
    dst.insert(dst.end(), rows.begin(), rows.end());
}

void append_row(std::vector<float> & dst, const std::vector<float> & row) {
    dst.insert(dst.end(), row.begin(), row.end());
}

std::vector<float> row_at(const std::vector<float> & rows, int64_t row, int64_t dim) {
    const auto * begin = rows.data() + static_cast<size_t>(row * dim);
    return std::vector<float>(begin, begin + dim);
}

std::vector<float> add_rows(const std::vector<float> & lhs, const std::vector<float> & rhs) {
    if (lhs.size() != rhs.size()) {
        throw std::runtime_error("VieNeu-TTS talker add_rows shape mismatch");
    }
    std::vector<float> out(lhs.size(), 0.0F);
    for (size_t i = 0; i < lhs.size(); ++i) {
        out[i] = lhs[i] + rhs[i];
    }
    return out;
}

std::vector<float> repeat_row(const std::vector<float> & row, int64_t repeats) {
    std::vector<float> out;
    out.reserve(row.size() * static_cast<size_t>(repeats));
    for (int64_t i = 0; i < repeats; ++i) {
        append_row(out, row);
    }
    return out;
}

std::vector<float> layernorm_host(
    const std::vector<float> & input,
    const assets::TensorDataF32 & weight,
    const assets::TensorDataF32 & bias,
    float eps = 1e-5f) {
    const size_t N = input.size();
    float sum = 0.0f;
    for (float x : input) sum += x;
    float mean = sum / N;
    float sq_sum = 0.0f;
    for (float x : input) sq_sum += (x - mean) * (x - mean);
    float var = sq_sum / N;
    std::vector<float> output(N);
    for (size_t i = 0; i < N; ++i) {
        output[i] = (input[i] - mean) / std::sqrt(var + eps) * weight.values[i] + bias.values[i];
    }
    return output;
}

PromptEmbeddingState build_prompt_state(
    const VietneuTalkerPrefill & prefill,
    const VietneuTTSConfig & root_config,
    const VietneuTalkerWeights & weights) {
    const auto & config = root_config.talker;
    if (prefill.input_ids.size() < 8) {
        throw std::runtime_error("VieNeu-TTS talker prefill input ids are too short");
    }
    const auto tts_special_hidden = lookup_rows(
        weights.text_embedding,
        config.text_hidden_size,
        {
            static_cast<int32_t>(root_config.tts_bos_token_id),
            static_cast<int32_t>(root_config.tts_eos_token_id),
            static_cast<int32_t>(root_config.tts_pad_token_id),
        });
    const auto tts_special = text_project_host(tts_special_hidden, 3, weights, config);
    const auto tts_bos = row_at(tts_special, 0, config.hidden_size);
    const auto tts_eos = row_at(tts_special, 1, config.hidden_size);
    const auto tts_pad = row_at(tts_special, 2, config.hidden_size);

    std::vector<float> custom_voice_speaker_embed;
    std::string language = ascii_lower(prefill.language);
    if (prefill.prompt_mode == VietneuTalkerPromptMode::CustomVoice) {
        const std::string speaker = ascii_lower(prefill.speaker);
        const auto speaker_it = config.speaker_id.find(speaker);
        if (speaker_it == config.speaker_id.end()) {
            throw std::runtime_error("VieNeu-TTS custom voice unsupported speaker: " + prefill.speaker);
        }
        const auto dialect_it = config.speaker_dialect.find(speaker);
        if (dialect_it == config.speaker_dialect.end()) {
            throw std::runtime_error("VieNeu-TTS custom voice missing dialect entry for speaker: " + prefill.speaker);
        }
        if ((language == "chinese" || language == "auto") && dialect_it->second.has_value()) {
            language = *dialect_it->second;
        }
        custom_voice_speaker_embed = lookup_rows(
            weights.codec_embedding,
            config.hidden_size,
            {static_cast<int32_t>(speaker_it->second)});
    }

    std::vector<int32_t> codec_prefix;
    if (language == "auto") {
        codec_prefix = {
            static_cast<int32_t>(config.codec_nothink_id),
            static_cast<int32_t>(config.codec_think_bos_id),
            static_cast<int32_t>(config.codec_think_eos_id),
        };
    } else {
        const auto language_it = config.codec_language_id.find(language);
        if (language_it == config.codec_language_id.end()) {
            throw std::runtime_error("VieNeu-TTS talker unsupported language: " + prefill.language);
        }
        codec_prefix = {
            static_cast<int32_t>(config.codec_think_id),
            static_cast<int32_t>(config.codec_think_bos_id),
            static_cast<int32_t>(language_it->second),
            static_cast<int32_t>(config.codec_think_eos_id),
        };
    }

    if (prefill.prompt_mode == VietneuTalkerPromptMode::VoiceDesign ||
        prefill.prompt_mode == VietneuTalkerPromptMode::CustomVoice) {
        PromptEmbeddingState state;
        state.tts_pad = tts_pad;
        if (!prefill.instruct_ids.empty()) {
            append_rows(
                state.prompt,
                text_project_host(
                    lookup_rows(weights.text_embedding, config.text_hidden_size, prefill.instruct_ids),
                    static_cast<int64_t>(prefill.instruct_ids.size()),
                    weights,
                    config));
        }
        const std::vector<int32_t> role_ids(prefill.input_ids.begin(), prefill.input_ids.begin() + 3);
        append_rows(
            state.prompt,
            text_project_host(lookup_rows(weights.text_embedding, config.text_hidden_size, role_ids), 3, weights, config));

        auto codec_embed = lookup_rows(weights.codec_embedding, config.hidden_size, codec_prefix);
        if (prefill.prompt_mode == VietneuTalkerPromptMode::CustomVoice) {
            append_rows(codec_embed, custom_voice_speaker_embed);
        }
        append_rows(
            codec_embed,
            lookup_rows(
                weights.codec_embedding,
                config.hidden_size,
                {
                    static_cast<int32_t>(config.codec_pad_id),
                    static_cast<int32_t>(config.codec_bos_id),
        }));
        const int64_t codec_rows = static_cast<int64_t>(codec_embed.size()) / config.hidden_size;
        append_rows(
            state.prompt,
            add_rows(
                repeat_row(tts_pad, codec_rows - 2),
                std::vector<float>(
                    codec_embed.begin(),
                    codec_embed.begin() + static_cast<std::ptrdiff_t>((codec_rows - 2) * config.hidden_size))));
        append_row(state.prompt, add_rows(tts_bos, row_at(codec_embed, codec_rows - 2, config.hidden_size)));

        const std::vector<int32_t> text_ids(prefill.input_ids.begin() + 3, prefill.input_ids.end() - 5);
        auto text_embed = text_project_host(
            lookup_rows(weights.text_embedding, config.text_hidden_size, text_ids),
            static_cast<int64_t>(text_ids.size()),
            weights,
            config);
        append_row(text_embed, tts_eos);
        append_rows(
            state.prompt,
            add_rows(
                text_embed,
                lookup_rows(
                    weights.codec_embedding,
                    config.hidden_size,
                    std::vector<int32_t>(text_ids.size() + 1, static_cast<int32_t>(config.codec_pad_id)))));
        append_row(
            state.prompt,
            add_rows(
                tts_pad,
                row_at(codec_embed, codec_rows - 1, config.hidden_size)));
        state.trailing_text = tts_pad;
        return state;
    }

    if (root_config.is_vieneu) {
        PromptEmbeddingState state;
        const int64_t hidden_size = config.hidden_size;

        // 1. Project speaker embedding to create the speaker anchor vector
        std::vector<float> speaker_anchor(hidden_size, 0.0f);
        if (prefill.speaker_embedding.has_value() && !prefill.speaker_embedding->values.empty()) {
            std::vector<float> speaker_embed = prefill.speaker_embedding->values;
            if (weights.xvec_proj_linear.has_value()) {
                auto projected = linear_host(speaker_embed, 1, root_config.speaker_embedding_dim, *weights.xvec_proj_linear);
                speaker_anchor = layernorm_host(projected, *weights.xvec_proj_ln_weight, *weights.xvec_proj_ln_bias);
            } else {
                if (static_cast<int64_t>(speaker_embed.size()) == hidden_size) {
                    speaker_anchor = speaker_embed;
                }
            }
        }
        std::fflush(stderr);

        // 2. Build text prompt tokens list: [style_token_id, tps] + phone_ids + [tpe]
        const int32_t style_token_id = 16;
        const int32_t tps = static_cast<int32_t>(root_config.text_prompt_start_token_id);
        const int32_t tpe = static_cast<int32_t>(root_config.text_prompt_end_token_id);
        
        std::vector<int32_t> text_token_ids;
        text_token_ids.push_back(style_token_id);
        text_token_ids.push_back(tps);
        text_token_ids.insert(text_token_ids.end(), prefill.input_ids.begin(), prefill.input_ids.end());
        text_token_ids.push_back(tpe);

        // 3. Embed text prompt tokens
        auto text_embed = lookup_rows(weights.text_embedding, hidden_size, text_token_ids);

        // 4. If reference codes are present, build reference rows
        std::vector<float> ref_embed;
        if (prefill.reference_codes.has_value() && !prefill.reference_ids.empty()) {
            const auto & ref_codes = *prefill.reference_codes;
            const int64_t T_ref = ref_codes.frames;
            const int32_t ref_slot = static_cast<int32_t>(root_config.audio_ref_slot_token_id);
            auto ref_slot_embed = lookup_rows(weights.text_embedding, hidden_size, {ref_slot});

            ref_embed.reserve(static_cast<size_t>(T_ref * hidden_size));
            for (int64_t frame = 0; frame < T_ref; ++frame) {
                std::vector<float> frame_emb = ref_slot_embed;
                for (int64_t ch = 0; ch < ref_codes.code_groups; ++ch) {
                    const int32_t code = ref_codes.codes[static_cast<size_t>(frame * ref_codes.code_groups + ch)];
                    const auto code_emb = ch == 0
                        ? lookup_rows(weights.codec_embedding, hidden_size, {code})
                        : lookup_rows(weights.code_predictor_embeddings.at(static_cast<size_t>(ch - 1)), hidden_size, {code});
                    for (int64_t dim = 0; dim < hidden_size; ++dim) {
                        frame_emb[static_cast<size_t>(dim)] += code_emb[static_cast<size_t>(dim)];
                    }
                }
                ref_embed.insert(ref_embed.end(), frame_emb.begin(), frame_emb.end());
            }
        }

        // 5. Combine text embeds and ref embeds into final prompt state
        state.prompt = std::move(text_embed);
        if (!ref_embed.empty()) {
            state.prompt.insert(state.prompt.end(), ref_embed.begin(), ref_embed.end());
        }

        for (size_t i = 0; i < state.prompt.size(); i += static_cast<size_t>(hidden_size)) {
            for (size_t j = 0; j < static_cast<size_t>(hidden_size); ++j) {
                state.prompt[i + j] += speaker_anchor[j];
            }
        }

        // 7. Initialize tts_pad as: text_embeddings(speech_generation_start_token_id) + speaker_anchor
        const int32_t sgs_id = static_cast<int32_t>(root_config.speech_generation_start_token_id);
        auto sgs_embed = lookup_rows(weights.text_embedding, hidden_size, {sgs_id});
        state.tts_pad = std::move(sgs_embed);
        for (size_t j = 0; j < static_cast<size_t>(hidden_size); ++j) {
            state.tts_pad[j] += speaker_anchor[j];
        }

        // 8. Trailing text is unused under VieNeu
        state.trailing_text = {};

        return state;
    }

    if (!prefill.speaker_embedding.has_value() || prefill.speaker_embedding->dims != config.hidden_size) {
        throw std::runtime_error("VieNeu-TTS talker voice clone prefill requires speaker embedding");
    }
    if (!prefill.reference_codes.has_value() || prefill.reference_ids.empty()) {
        throw std::runtime_error("VieNeu-TTS talker ICL prefill requires reference ids and codes");
    }

    PromptEmbeddingState state;
    state.tts_pad = tts_pad;
    const std::vector<int32_t> role_ids(prefill.input_ids.begin(), prefill.input_ids.begin() + 3);
    append_rows(state.prompt, text_project_host(lookup_rows(weights.text_embedding, config.text_hidden_size, role_ids), 3, weights, config));

    auto codec_embed = lookup_rows(weights.codec_embedding, config.hidden_size, codec_prefix);
    append_row(codec_embed, prefill.speaker_embedding->values);
    append_rows(codec_embed, lookup_rows(
                                 weights.codec_embedding,
                                 config.hidden_size,
                                 {
                                     static_cast<int32_t>(config.codec_pad_id),
                                     static_cast<int32_t>(config.codec_bos_id),
                                 }));
    const int64_t codec_rows = static_cast<int64_t>(codec_embed.size()) / config.hidden_size;
    append_rows(state.prompt, add_rows(repeat_row(tts_pad, codec_rows - 2), std::vector<float>(
                          codec_embed.begin(),
                          codec_embed.begin() + static_cast<std::ptrdiff_t>((codec_rows - 2) * config.hidden_size))));
    append_row(state.prompt, add_rows(tts_bos, row_at(codec_embed, codec_rows - 2, config.hidden_size)));

    const std::vector<int32_t> text_ids(prefill.input_ids.begin() + 3, prefill.input_ids.end() - 5);
    const std::vector<int32_t> ref_text_ids(prefill.reference_ids.begin() + 3, prefill.reference_ids.end() - 2);
    std::vector<int32_t> combined_text = ref_text_ids;
    combined_text.insert(combined_text.end(), text_ids.begin(), text_ids.end());
    auto text_embed = text_project_host(
        lookup_rows(weights.text_embedding, config.text_hidden_size, combined_text),
        static_cast<int64_t>(combined_text.size()),
        weights,
        config);
    append_row(text_embed, tts_eos);

    const auto & ref_codes = *prefill.reference_codes;
    std::vector<float> ref_codec;
    append_rows(ref_codec, lookup_rows(weights.codec_embedding, config.hidden_size, {static_cast<int32_t>(config.codec_bos_id)}));
    for (int64_t frame = 0; frame < ref_codes.frames; ++frame) {
        std::vector<float> summed(config.hidden_size, 0.0F);
        for (int64_t group = 0; group < ref_codes.code_groups; ++group) {
            const int32_t code = ref_codes.codes[static_cast<size_t>(frame * ref_codes.code_groups + group)];
            auto row = group == 0
                ? lookup_rows(weights.codec_embedding, config.hidden_size, {code})
                : lookup_rows(weights.code_predictor_embeddings.at(static_cast<size_t>(group - 1)), config.hidden_size, {code});
            for (int64_t dim = 0; dim < config.hidden_size; ++dim) {
                summed[static_cast<size_t>(dim)] += row[static_cast<size_t>(dim)];
            }
        }
        append_row(ref_codec, summed);
    }
    const int64_t text_rows = static_cast<int64_t>(text_embed.size()) / config.hidden_size;
    const int64_t ref_codec_rows = static_cast<int64_t>(ref_codec.size()) / config.hidden_size;
    if (text_rows > ref_codec_rows) {
        std::vector<float> text_part(text_embed.begin(), text_embed.begin() + static_cast<std::ptrdiff_t>(ref_codec.size()));
        append_rows(state.prompt, add_rows(text_part, ref_codec));
        state.trailing_text.assign(
            text_embed.begin() + static_cast<std::ptrdiff_t>(ref_codec.size()),
            text_embed.end());
    } else {
        std::vector<float> padded_text = text_embed;
        append_rows(padded_text, repeat_row(tts_pad, ref_codec_rows - text_rows));
        append_rows(state.prompt, add_rows(padded_text, ref_codec));
        state.trailing_text = tts_pad;
    }
    return state;
}

VietneuTalkerWeights load_talker_weights(
    const VietneuTTSAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t weight_context_bytes,
    engine::assets::TensorStorageType weight_storage_type) {
    const auto & source = *assets.model_weights;
    const auto & config = assets.config.talker;
    VietneuTalkerWeights weights;
    weights.store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "vietneu_tts.talker.weights",
        weight_context_bytes);

    if (assets.config.is_vieneu) {
        // VieNeu-TTS v3 Turbo loading
        weights.codec_embedding = source.require_tensor(
            "audio_embeddings.0.weight",
            assets::TensorStorageType::Native,
            {config.vocab_size, config.hidden_size});
        weights.code_predictor_embeddings.reserve(static_cast<size_t>(config.num_code_groups - 1));
        for (int64_t group = 0; group < config.num_code_groups - 1; ++group) {
            weights.code_predictor_embeddings.push_back(source.require_tensor(
                "audio_embeddings." + std::to_string(group + 1) + ".weight",
                assets::TensorStorageType::Native,
                {assets.config.code_predictor.vocab_size, config.hidden_size}));
        }
        weights.text_embedding = source.require_tensor(
            "text_embeddings.weight",
            assets::TensorStorageType::Native,
            {config.text_vocab_size, config.text_hidden_size});

        weights.layers.reserve(static_cast<size_t>(config.num_hidden_layers));
        const int64_t dim = attention_head_dim(config);
        for (int64_t layer = 0; layer < config.num_hidden_layers; ++layer) {
            const std::string prefix = "semantic_backbone.layers." + std::to_string(layer);
            TalkerLayerWeights w;
            w.input_norm = source.require_f32_tensor(prefix + ".input_layernorm.weight", {config.hidden_size});
            w.q_proj = weights.store->load_tensor(
                source,
                prefix + ".self_attn.q_proj.weight",
                weight_storage_type,
                {config.num_attention_heads * dim, config.hidden_size});
            w.k_proj = weights.store->load_tensor(
                source,
                prefix + ".self_attn.k_proj.weight",
                weight_storage_type,
                {config.num_key_value_heads * dim, config.hidden_size});
            w.v_proj = weights.store->load_tensor(
                source,
                prefix + ".self_attn.v_proj.weight",
                weight_storage_type,
                {config.num_key_value_heads * dim, config.hidden_size});
            w.o_proj = weights.store->load_tensor(
                source,
                prefix + ".self_attn.o_proj.weight",
                weight_storage_type,
                {config.hidden_size, config.num_attention_heads * dim});
            w.q_norm = source.require_f32_tensor(prefix + ".self_attn.q_norm.weight", {dim});
            w.k_norm = source.require_f32_tensor(prefix + ".self_attn.k_norm.weight", {dim});
            w.post_norm = source.require_f32_tensor(prefix + ".post_attention_layernorm.weight", {config.hidden_size});
            w.gate_proj = weights.store->load_tensor(
                source,
                prefix + ".mlp.gate_proj.weight",
                weight_storage_type,
                {config.intermediate_size, config.hidden_size});
            w.up_proj = weights.store->load_tensor(
                source,
                prefix + ".mlp.up_proj.weight",
                weight_storage_type,
                {config.intermediate_size, config.hidden_size});
            w.down_proj = weights.store->load_tensor(
                source,
                prefix + ".mlp.down_proj.weight",
                weight_storage_type,
                {config.hidden_size, config.intermediate_size});
            weights.layers.push_back(std::move(w));
        }
        weights.norm = source.require_f32_tensor("semantic_backbone.norm.weight", {config.hidden_size});
        weights.codec_head = weights.store->load_tensor(
            source,
            "text_embeddings.weight",
            weight_storage_type,
            {config.text_vocab_size, config.hidden_size});

        const auto & predictor_config = assets.config.code_predictor;
        const int64_t predictor_dim = attention_head_dim(predictor_config);
        weights.code_predictor.layers.reserve(static_cast<size_t>(predictor_config.num_hidden_layers));
        for (int64_t layer = 0; layer < predictor_config.num_hidden_layers; ++layer) {
            const std::string prefix = "acoustic_decoder.layers." + std::to_string(layer);
            TalkerLayerWeights w;
            w.input_norm = source.require_f32_tensor(prefix + ".norm1.weight", {predictor_config.hidden_size});
            // Combined QKV weight in VieNeu
            w.q_proj = weights.store->load_tensor(
                source,
                prefix + ".attn.qkv.weight",
                weight_storage_type,
                {3 * predictor_config.hidden_size, predictor_config.hidden_size});
            w.o_proj = weights.store->load_tensor(
                source,
                prefix + ".attn.o_proj.weight",
                weight_storage_type,
                {predictor_config.hidden_size, predictor_config.hidden_size});
            w.q_norm = source.require_f32_tensor(prefix + ".attn.q_norm.weight", {predictor_dim});
            w.k_norm = source.require_f32_tensor(prefix + ".attn.k_norm.weight", {predictor_dim});
            w.post_norm = source.require_f32_tensor(prefix + ".norm2.weight", {predictor_config.hidden_size});
            w.gate_proj = weights.store->load_tensor(
                source,
                prefix + ".ff_gate.weight",
                weight_storage_type,
                {predictor_config.intermediate_size, predictor_config.hidden_size});
            w.up_proj = weights.store->load_tensor(
                source,
                prefix + ".ff_up.weight",
                weight_storage_type,
                {predictor_config.intermediate_size, predictor_config.hidden_size});
            w.down_proj = weights.store->load_tensor(
                source,
                prefix + ".ff_down.weight",
                weight_storage_type,
                {predictor_config.hidden_size, predictor_config.intermediate_size});
            weights.code_predictor.layers.push_back(std::move(w));
        }
        weights.code_predictor.norm = source.require_f32_tensor(
            "acoustic_decoder.norm.weight",
            {predictor_config.hidden_size});

        weights.code_predictor.lm_heads.reserve(static_cast<size_t>(config.num_code_groups));
        for (int64_t group = 0; group < config.num_code_groups; ++group) {
            weights.code_predictor.lm_heads.push_back(weights.store->load_tensor(
                source,
                "audio_embeddings." + std::to_string(group) + ".weight",
                weight_storage_type,
                {config.vocab_size, config.hidden_size}));
        }

        weights.slot_pos_emb = source.require_tensor(
            "acoustic_decoder.slot_pos_emb.weight",
            assets::TensorStorageType::Native,
            {config.num_code_groups + 1, config.hidden_size});

        if (assets.config.use_speaker_embedding) {
            weights.xvec_proj_linear = {
                source.require_tensor(
                    "xvec_proj.0.weight",
                    weight_storage_type,
                    {config.hidden_size, assets.config.speaker_embedding_dim}),
                source.require_tensor(
                    "xvec_proj.0.bias",
                    assets::TensorStorageType::F32,
                    {config.hidden_size}),
            };
            weights.xvec_proj_ln_weight = source.require_f32_tensor("xvec_proj.1.weight", {config.hidden_size});
            weights.xvec_proj_ln_bias = source.require_f32_tensor("xvec_proj.1.bias", {config.hidden_size});
        }
    } else {
        // Standard VieNeu-TTS loading
        weights.codec_embedding = source.require_tensor(
            "talker.model.codec_embedding.weight",
            assets::TensorStorageType::Native,
            {config.vocab_size, config.hidden_size});
        weights.code_predictor_embeddings.reserve(static_cast<size_t>(config.num_code_groups - 1));
        for (int64_t group = 0; group < config.num_code_groups - 1; ++group) {
            weights.code_predictor_embeddings.push_back(source.require_tensor(
                "talker.code_predictor.model.codec_embedding." + std::to_string(group) + ".weight",
                assets::TensorStorageType::Native,
                {assets.config.code_predictor.vocab_size, config.hidden_size}));
        }
        weights.text_embedding = source.require_tensor(
            "talker.model.text_embedding.weight",
            assets::TensorStorageType::Native,
            {config.text_vocab_size, config.text_hidden_size});
        weights.text_projection_fc1 = {
            source.require_tensor(
                "talker.text_projection.linear_fc1.weight",
                weight_storage_type,
                {config.text_hidden_size, config.text_hidden_size}),
            source.require_tensor(
                "talker.text_projection.linear_fc1.bias",
                assets::TensorStorageType::F32,
                {config.text_hidden_size}),
        };
        weights.text_projection_fc2 = {
            source.require_tensor(
                "talker.text_projection.linear_fc2.weight",
                weight_storage_type,
                {config.hidden_size, config.text_hidden_size}),
            source.require_tensor(
                "talker.text_projection.linear_fc2.bias",
                assets::TensorStorageType::F32,
                {config.hidden_size}),
        };
        weights.layers.reserve(static_cast<size_t>(config.num_hidden_layers));
        const int64_t dim = attention_head_dim(config);
        for (int64_t layer = 0; layer < config.num_hidden_layers; ++layer) {
            const std::string prefix = "talker.model.layers." + std::to_string(layer);
            TalkerLayerWeights w;
            w.input_norm = source.require_f32_tensor(prefix + ".input_layernorm.weight", {config.hidden_size});
            w.q_proj = weights.store->load_tensor(
                source,
                prefix + ".self_attn.q_proj.weight",
                weight_storage_type,
                {config.num_attention_heads * dim, config.hidden_size});
            w.k_proj = weights.store->load_tensor(
                source,
                prefix + ".self_attn.k_proj.weight",
                weight_storage_type,
                {config.num_key_value_heads * dim, config.hidden_size});
            w.v_proj = weights.store->load_tensor(
                source,
                prefix + ".self_attn.v_proj.weight",
                weight_storage_type,
                {config.num_key_value_heads * dim, config.hidden_size});
            w.o_proj = weights.store->load_tensor(
                source,
                prefix + ".self_attn.o_proj.weight",
                weight_storage_type,
                {config.hidden_size, config.num_attention_heads * dim});
            w.q_norm = source.require_f32_tensor(prefix + ".self_attn.q_norm.weight", {dim});
            w.k_norm = source.require_f32_tensor(prefix + ".self_attn.k_norm.weight", {dim});
            w.post_norm = source.require_f32_tensor(prefix + ".post_attention_layernorm.weight", {config.hidden_size});
            w.gate_proj = weights.store->load_tensor(
                source,
                prefix + ".mlp.gate_proj.weight",
                weight_storage_type,
                {config.intermediate_size, config.hidden_size});
            w.up_proj = weights.store->load_tensor(
                source,
                prefix + ".mlp.up_proj.weight",
                weight_storage_type,
                {config.intermediate_size, config.hidden_size});
            w.down_proj = weights.store->load_tensor(
                source,
                prefix + ".mlp.down_proj.weight",
                weight_storage_type,
                {config.hidden_size, config.intermediate_size});
            weights.layers.push_back(std::move(w));
        }
        weights.norm = source.require_f32_tensor("talker.model.norm.weight", {config.hidden_size});
        weights.codec_head = weights.store->load_tensor(
            source,
            "talker.codec_head.weight",
            weight_storage_type,
            {config.vocab_size, config.hidden_size});

        const auto & predictor_config = assets.config.code_predictor;
        const int64_t predictor_dim = attention_head_dim(predictor_config);
        weights.code_predictor.layers.reserve(static_cast<size_t>(predictor_config.num_hidden_layers));
        for (int64_t layer = 0; layer < predictor_config.num_hidden_layers; ++layer) {
            const std::string prefix = "talker.code_predictor.model.layers." + std::to_string(layer);
            TalkerLayerWeights w;
            w.input_norm = source.require_f32_tensor(prefix + ".input_layernorm.weight", {predictor_config.hidden_size});
            w.q_proj = weights.store->load_tensor(
                source,
                prefix + ".self_attn.q_proj.weight",
                weight_storage_type,
                {predictor_config.num_attention_heads * predictor_dim, predictor_config.hidden_size});
            w.k_proj = weights.store->load_tensor(
                source,
                prefix + ".self_attn.k_proj.weight",
                weight_storage_type,
                {predictor_config.num_key_value_heads * predictor_dim, predictor_config.hidden_size});
            w.v_proj = weights.store->load_tensor(
                source,
                prefix + ".self_attn.v_proj.weight",
                weight_storage_type,
                {predictor_config.num_key_value_heads * predictor_dim, predictor_config.hidden_size});
            w.o_proj = weights.store->load_tensor(
                source,
                prefix + ".self_attn.o_proj.weight",
                weight_storage_type,
                {predictor_config.hidden_size, predictor_config.num_attention_heads * predictor_dim});
            w.q_norm = source.require_f32_tensor(prefix + ".self_attn.q_norm.weight", {predictor_dim});
            w.k_norm = source.require_f32_tensor(prefix + ".self_attn.k_norm.weight", {predictor_dim});
            w.post_norm = source.require_f32_tensor(prefix + ".post_attention_layernorm.weight", {predictor_config.hidden_size});
            w.gate_proj = weights.store->load_tensor(
                source,
                prefix + ".mlp.gate_proj.weight",
                weight_storage_type,
                {predictor_config.intermediate_size, predictor_config.hidden_size});
            w.up_proj = weights.store->load_tensor(
                source,
                prefix + ".mlp.up_proj.weight",
                weight_storage_type,
                {predictor_config.intermediate_size, predictor_config.hidden_size});
            w.down_proj = weights.store->load_tensor(
                source,
                prefix + ".mlp.down_proj.weight",
                weight_storage_type,
                {predictor_config.hidden_size, predictor_config.intermediate_size});
            weights.code_predictor.layers.push_back(std::move(w));
        }
        weights.code_predictor.norm = source.require_f32_tensor(
            "talker.code_predictor.model.norm.weight",
            {predictor_config.hidden_size});
        if (predictor_config.hidden_size != config.hidden_size) {
            weights.code_predictor.small_to_mtp_projection = GraphLinearTensorWeights{
                weights.store->load_tensor(
                    source,
                    "talker.code_predictor.small_to_mtp_projection.weight",
                    weight_storage_type,
                    {predictor_config.hidden_size, config.hidden_size}),
                weights.store->load_tensor(
                    source,
                    "talker.code_predictor.small_to_mtp_projection.bias",
                    assets::TensorStorageType::F32,
                    {predictor_config.hidden_size}),
            };
        }
        weights.code_predictor.lm_heads.reserve(static_cast<size_t>(config.num_code_groups - 1));
        for (int64_t group = 0; group < config.num_code_groups - 1; ++group) {
            weights.code_predictor.lm_heads.push_back(weights.store->load_tensor(
                source,
                "talker.code_predictor.lm_head." + std::to_string(group) + ".weight",
                weight_storage_type,
                {predictor_config.vocab_size, predictor_config.hidden_size}));
        }
    }

    weights.store->upload();
    return weights;
}

}  // namespace

class VietneuTalkerWeightsRuntime {
public:
    VietneuTalkerWeightsRuntime(
        std::shared_ptr<const VietneuTTSAssets> assets,
        core::BackendType backend_type,
        int device,
        int threads,
        size_t graph_arena_bytes,
        size_t talker_constant_context_bytes,
        size_t code_predictor_constant_context_bytes,
        engine::assets::TensorStorageType weight_storage_type)
        : assets_(std::move(assets)),
          threads_(threads),
          graph_arena_bytes_(graph_arena_bytes) {
        if (assets_ == nullptr) {
            throw std::runtime_error("VieNeu-TTS talker weights runtime requires assets");
        }
        if (threads_ <= 0) {
            throw std::runtime_error("VieNeu-TTS talker weights runtime requires positive thread count");
        }
        backend_type_ = backend_type;
        sampling_policy_ = backend_type_ == core::BackendType::Cuda
            ? engine::sampling::resolve_torch_cuda_sampling_policy(
                  backend_type_,
                  device,
                  "vietneu_tts.talker.cuda_sampling_policy",
                  "VieNeu-TTS TTS",
                  engine::sampling::TorchCudaSamplingPolicyFailureMode::StrictCuda)
            : engine::sampling::TorchCudaSamplingPolicy{};
        backend_ = core::init_backend({backend_type_, device, threads_});
        weights_ = std::make_shared<VietneuTalkerWeights>(
            load_talker_weights(*assets_, backend_, backend_type_, kTalkerWeightContextBytes, weight_storage_type));
        talker_constants_ = std::make_unique<core::ConstantTensorCache>(
            backend_,
            threads_,
            "vietneu_tts.talker.constants",
            talker_constant_context_bytes);
        code_predictor_constants_ = std::make_unique<core::ConstantTensorCache>(
            backend_,
            threads_,
            "vietneu_tts.talker.code_predictor.constants",
            code_predictor_constant_context_bytes);
    }

    ~VietneuTalkerWeightsRuntime() {
        code_predictor_constants_.reset();
        talker_constants_.reset();
        weights_.reset();
        if (backend_ != nullptr) {
            ggml_backend_free(backend_);
        }
    }

    const VietneuTTSAssets & assets() const noexcept {
        return *assets_;
    }

    const VietneuTalkerWeights & weights() const noexcept {
        return *weights_;
    }

    int threads() const noexcept {
        return threads_;
    }

    ggml_backend_t backend() const noexcept {
        return backend_;
    }

    core::ConstantTensorCache & talker_constants() const noexcept {
        return *talker_constants_;
    }

    core::ConstantTensorCache & code_predictor_constants() const noexcept {
        return *code_predictor_constants_;
    }

    size_t graph_arena_bytes() const noexcept {
        return graph_arena_bytes_;
    }

    const engine::sampling::TorchCudaSamplingPolicy & sampling_policy() const noexcept {
        return sampling_policy_;
    }

private:
    std::shared_ptr<const VietneuTTSAssets> assets_;
    std::shared_ptr<const VietneuTalkerWeights> weights_;
    int threads_ = 1;
    size_t graph_arena_bytes_ = 0;
    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    engine::sampling::TorchCudaSamplingPolicy sampling_policy_;
    std::unique_ptr<core::ConstantTensorCache> talker_constants_;
    std::unique_ptr<core::ConstantTensorCache> code_predictor_constants_;
};

class TalkerPrefillGraph {
public:
    TalkerPrefillGraph(
        std::shared_ptr<const VietneuTalkerWeightsRuntime> weights,
        int64_t prompt_capacity)
        : weights_(std::move(weights)),
          prompt_capacity_(prompt_capacity) {
        if (prompt_capacity_ <= 0) {
            throw std::runtime_error("VieNeu-TTS talker prefill graph requires positive prompt capacity");
        }
        ggml_init_params params{weights_->graph_arena_bytes(), nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize VieNeu-TTS talker prefill graph context");
        }
        const auto & config = weights_->assets().config.talker;
        const auto & tensor_weights = weights_->weights();
        core::ModuleBuildContext ctx{ctx_.get(), "vietneu_tts.talker.prefill"};
        auto x = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, prompt_capacity_, config.hidden_size}));
        input_ = x.tensor;
        positions_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, prompt_capacity_);
        auto positions_value = core::wrap_tensor(positions_, core::TensorShape::from_dims({prompt_capacity_}), GGML_TYPE_I32);
        auto & constants = weights_->talker_constants();
        constants.begin_graph();
        const auto & root_config = weights_->assets().config;
        const int64_t logits_size = root_config.is_vieneu ? config.text_vocab_size : config.vocab_size;
        auto decoder_out = modules::QwenCausalDecoderModule(make_qwen_decoder_config(config, logits_size, root_config.is_vieneu))
                               .build(
                                   ctx,
                                   x,
                                   positions_value,
                                   make_qwen_decoder_weights(constants, tensor_weights.layers, tensor_weights.norm, tensor_weights.codec_head));
        for (const auto & layer : decoder_out.state.layers) {
            if (!layer.key.has_value() || !layer.value.has_value()) {
                throw std::runtime_error("VieNeu-TTS talker prefill decoder did not return K/V state");
            }
            keys_.push_back(layer.key->tensor);
            values_.push_back(layer.value->tensor);
        }
        hidden_output_ = decoder_out.hidden.tensor;
        logits_output_ = decoder_out.logits.tensor;
        ggml_set_output(logits_output_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 65536, false);
        ggml_build_forward_expand(graph_, logits_output_);
        constants.finish_graph();
        constants.ensure_uploaded();
        buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), weights_->backend());
        if (buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate VieNeu-TTS talker prefill graph");
        }
        std::vector<int32_t> positions(static_cast<size_t>(prompt_capacity_), 0);
        for (int64_t i = 0; i < prompt_capacity_; ++i) {
            positions[static_cast<size_t>(i)] = static_cast<int32_t>(i);
        }
        ggml_backend_tensor_set(positions_, positions.data(), 0, positions.size() * sizeof(int32_t));
    }

    ~TalkerPrefillGraph() {
        engine::core::release_backend_graph_resources(weights_->backend(), graph_);
        if (buffer_ != nullptr) {
            ggml_backend_buffer_free(buffer_);
        }
    }

    bool matches(const VietneuTalkerWeightsRuntime & weights, int64_t prompt_capacity) const {
        return weights_.get() == &weights && prompt_capacity_ == prompt_capacity;
    }

    struct OutputWithCache {
        VietneuTalkerPrefillResult result;
        runtime::TransformerKVState state;
    };

    OutputWithCache run_with_state(const std::vector<float> & embeddings) {
        const auto & config = weights_->assets().config.talker;
        if (static_cast<int64_t>(embeddings.size()) != prompt_capacity_ * config.hidden_size) {
            throw std::runtime_error("VieNeu-TTS talker prefill embedding size mismatch");
        }
        ggml_backend_tensor_set(input_, embeddings.data(), 0, embeddings.size() * sizeof(float));
        core::set_backend_threads(weights_->backend(), weights_->threads());
        const ggml_status status = engine::core::compute_backend_graph(weights_->backend(), graph_);
        ggml_backend_synchronize(weights_->backend());
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("VieNeu-TTS talker prefill graph compute failed");
        }
        OutputWithCache out;
        const auto & root_config = weights_->assets().config;
        const int64_t logits_vocab_size = root_config.is_vieneu ? config.text_vocab_size : config.vocab_size;
        out.result.logits.vocab_size = logits_vocab_size;
        out.result.logits.values.resize(static_cast<size_t>(logits_vocab_size));
        ggml_backend_tensor_get(logits_output_, out.result.logits.values.data(), 0, out.result.logits.values.size() * sizeof(float));
        out.result.last_hidden.dims = config.hidden_size;
        out.result.last_hidden.values.resize(static_cast<size_t>(config.hidden_size));
        ggml_backend_tensor_get(hidden_output_, out.result.last_hidden.values.data(), 0, out.result.last_hidden.values.size() * sizeof(float));
        out.state.current_end = prompt_capacity_;
        out.state.layers.resize(keys_.size());
        const size_t layer_values = static_cast<size_t>(
            prompt_capacity_ * config.num_key_value_heads * attention_head_dim(config));
        for (size_t layer = 0; layer < keys_.size(); ++layer) {
            auto & state_layer = out.state.layers[layer];
            state_layer.valid_steps = prompt_capacity_;
            state_layer.key.resize(layer_values);
            state_layer.value.resize(layer_values);
            ggml_backend_tensor_get(keys_[layer], state_layer.key.data(), 0, state_layer.key.size() * sizeof(float));
            ggml_backend_tensor_get(values_[layer], state_layer.value.data(), 0, state_layer.value.size() * sizeof(float));
        }
        return out;
    }

private:
    std::shared_ptr<const VietneuTalkerWeightsRuntime> weights_;
    int64_t prompt_capacity_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * positions_ = nullptr;
    ggml_tensor * hidden_output_ = nullptr;
    ggml_tensor * logits_output_ = nullptr;
    std::vector<ggml_tensor *> keys_;
    std::vector<ggml_tensor *> values_;
    ggml_cgraph * graph_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
};

class TalkerCachedStepGraph {
public:
    TalkerCachedStepGraph(
        std::shared_ptr<const VietneuTalkerWeightsRuntime> weights,
        int64_t cache_steps)
        : weights_(std::move(weights)),
          cache_steps_(cache_steps) {
        if (cache_steps_ <= 0) {
            throw std::runtime_error("VieNeu-TTS talker cached step graph requires positive cache capacity");
        }
        ggml_init_params params{weights_->graph_arena_bytes(), nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize VieNeu-TTS talker cached step graph context");
        }
        const auto & config = weights_->assets().config.talker;
        const auto & tensor_weights = weights_->weights();
        core::ModuleBuildContext ctx{ctx_.get(), "vietneu_tts.talker.cached_step"};
        auto x = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 1, config.hidden_size}));
        input_ = x.tensor;
        positions_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, 1);
        auto positions_value = core::wrap_tensor(positions_, core::TensorShape::from_dims({1}), GGML_TYPE_I32);
        cache_slot_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, 1);
        auto cache_slot_value = core::wrap_tensor(cache_slot_, core::TensorShape::from_dims({1}), GGML_TYPE_I32);
        attention_mask_ = ggml_new_tensor_4d(ctx_.get(), GGML_TYPE_F16, cache_steps_, 1, 1, 1);
        auto attention_mask_value = core::wrap_tensor(
            attention_mask_,
            core::TensorShape::from_dims({1, 1, 1, cache_steps_}),
            GGML_TYPE_F16);
        graph_ = ggml_new_graph_custom(ctx_.get(), 65536, false);
        auto & constants = weights_->talker_constants();
        constants.begin_graph();
        const auto & root_config = weights_->assets().config;
        const int64_t logits_size = root_config.is_vieneu ? config.text_vocab_size : config.vocab_size;
        auto decoder_out = modules::QwenCausalDecoderModule(make_qwen_decoder_config(config, logits_size, root_config.is_vieneu))
                               .build_static_cache_tail(
                                   ctx,
                                   graph_,
                                   x,
                                   positions_value,
                                   make_qwen_decoder_weights(constants, tensor_weights.layers, tensor_weights.norm, tensor_weights.codec_head),
                                   cache_steps_,
                                   attention_mask_value,
                                   cache_slot_value);
        step_cache_ = std::move(decoder_out.cache);
        hidden_output_ = decoder_out.hidden.tensor;
        logits_output_ = decoder_out.logits.tensor;
        ggml_set_output(logits_output_);
        ggml_build_forward_expand(graph_, logits_output_);
        constants.finish_graph();
        constants.ensure_uploaded();
        buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), weights_->backend());
        if (buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate VieNeu-TTS talker cached step graph");
        }
        attention_mask_buffer_.assign(static_cast<size_t>(cache_steps_), ggml_fp32_to_fp16(-INFINITY));
    }

    ~TalkerCachedStepGraph() {
        engine::core::release_backend_graph_resources(weights_->backend(), graph_);
        if (buffer_ != nullptr) {
            ggml_backend_buffer_free(buffer_);
        }
    }

    bool can_run(const VietneuTalkerWeightsRuntime & weights, int64_t required_capacity) const {
        return weights_.get() == &weights && cache_steps_ >= required_capacity;
    }

    int64_t cache_steps() const noexcept {
        return cache_steps_;
    }

    void import_prefill_state(const runtime::TransformerKVState & state) {
        step_cache_.import_state(state);
    }

    runtime::TransformerKVState export_state() const {
        return step_cache_.export_state();
    }

    VietneuTalkerPrefillResult run_step(const std::vector<float> & embedding) {
        last_timing_ = {};
        const auto & config = weights_->assets().config.talker;
        if (static_cast<int64_t>(embedding.size()) != config.hidden_size) {
            throw std::runtime_error("VieNeu-TTS talker cached step embedding size mismatch");
        }

        if (step_cache_.valid_steps() >= cache_steps_) {
            throw std::runtime_error("VieNeu-TTS talker cached step exceeds cache capacity");
        }
        auto timing_start = Clock::now();
        ggml_backend_tensor_set(input_, embedding.data(), 0, embedding.size() * sizeof(float));
        const int32_t position = static_cast<int32_t>(step_cache_.current_end());
        ggml_backend_tensor_set(positions_, &position, 0, sizeof(int32_t));
        const int32_t cache_slot = static_cast<int32_t>(step_cache_.valid_steps());
        ggml_backend_tensor_set(cache_slot_, &cache_slot, 0, sizeof(int32_t));
        last_timing_.input_upload_ms = engine::debug::elapsed_ms(timing_start, Clock::now());
        std::fill(attention_mask_buffer_.begin(), attention_mask_buffer_.end(), ggml_fp32_to_fp16(-INFINITY));
        for (int64_t i = 0; i < step_cache_.valid_steps(); ++i) {
            attention_mask_buffer_[static_cast<size_t>(i)] = ggml_fp32_to_fp16(0.0F);
        }
        attention_mask_buffer_[static_cast<size_t>(cache_slot)] = ggml_fp32_to_fp16(0.0F);
        timing_start = Clock::now();
        ggml_backend_tensor_set(
            attention_mask_,
            attention_mask_buffer_.data(),
            0,
            attention_mask_buffer_.size() * sizeof(ggml_fp16_t));
        last_timing_.mask_upload_ms = engine::debug::elapsed_ms(timing_start, Clock::now());
        core::set_backend_threads(weights_->backend(), weights_->threads());
        timing_start = Clock::now();
        const ggml_status status = engine::core::compute_backend_graph(weights_->backend(), graph_);
        ggml_backend_synchronize(weights_->backend());
        last_timing_.graph_compute_ms = engine::debug::elapsed_ms(timing_start, Clock::now());
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("VieNeu-TTS talker cached step graph compute failed");
        }
        VietneuTalkerPrefillResult out;
        const auto & root_config = weights_->assets().config;
        const int64_t logits_vocab_size = root_config.is_vieneu ? config.text_vocab_size : config.vocab_size;
        out.logits.vocab_size = logits_vocab_size;
        out.logits.values.resize(static_cast<size_t>(logits_vocab_size));
        out.last_hidden.dims = config.hidden_size;
        out.last_hidden.values.resize(static_cast<size_t>(config.hidden_size));
        timing_start = Clock::now();
        ggml_backend_tensor_get(logits_output_, out.logits.values.data(), 0, out.logits.values.size() * sizeof(float));
        ggml_backend_tensor_get(hidden_output_, out.last_hidden.values.data(), 0, out.last_hidden.values.size() * sizeof(float));
        last_timing_.output_read_ms = engine::debug::elapsed_ms(timing_start, Clock::now());

        step_cache_.advance_after_direct_append(1);
        return out;
    }

    const CachedStepTiming & last_timing() const noexcept {
        return last_timing_;
    }

private:
    std::shared_ptr<const VietneuTalkerWeightsRuntime> weights_;
    int64_t cache_steps_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * positions_ = nullptr;
    ggml_tensor * cache_slot_ = nullptr;
    ggml_tensor * attention_mask_ = nullptr;
    ggml_tensor * hidden_output_ = nullptr;
    ggml_tensor * logits_output_ = nullptr;
    std::vector<ggml_fp16_t> attention_mask_buffer_;
    runtime::TransformerKVCache step_cache_;
    ggml_cgraph * graph_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
    CachedStepTiming last_timing_;
};

int32_t argmax_index(const std::vector<float> & values) {
    if (values.empty()) {
        throw std::runtime_error("VieNeu-TTS talker cannot select from empty logits");
    }
    size_t best = 0;
    for (size_t i = 1; i < values.size(); ++i) {
        if (values[i] > values[best]) {
            best = i;
        }
    }
    return static_cast<int32_t>(best);
}

int32_t sample_index(
    const std::vector<float> & logits,
    int top_k,
    float top_p,
    float temperature,
    std::mt19937 & rng,
    const engine::sampling::TorchCudaSamplingPolicy & sampling_policy,
    uint64_t seed,
    uint64_t call_index) {
    if (temperature <= 0.0F) {
        throw std::runtime_error("VieNeu-TTS sampler temperature must be positive");
    }
    std::vector<int32_t> indices;
    indices.reserve(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) {
        if (std::isfinite(logits[i])) {
            indices.push_back(static_cast<int32_t>(i));
        }
    }
    if (indices.empty()) {
        throw std::runtime_error("VieNeu-TTS sampler has no finite logits");
    }
    std::sort(indices.begin(), indices.end(), [&](int32_t lhs, int32_t rhs) {
        return logits[static_cast<size_t>(lhs)] > logits[static_cast<size_t>(rhs)];
    });
    if (top_k > 0 && static_cast<int>(indices.size()) > top_k) {
        indices.resize(static_cast<size_t>(top_k));
    }

    const float max_logit = logits[static_cast<size_t>(indices.front())] / temperature;
    std::vector<double> weights;
    weights.reserve(indices.size());
    double total = 0.0;
    for (const int32_t index : indices) {
        const double weight = std::exp(static_cast<double>(logits[static_cast<size_t>(index)] / temperature - max_logit));
        weights.push_back(weight);
        total += weight;
    }
    if (top_p > 0.0F && top_p < 1.0F) {
        double cumulative = 0.0;
        size_t keep = weights.size();
        for (size_t i = 0; i < weights.size(); ++i) {
            cumulative += weights[i] / total;
            if (cumulative >= top_p) {
                keep = i + 1;
                break;
            }
        }
        indices.resize(keep);
        weights.resize(keep);
    }
    if (sampling_policy.cuda_fast_path) {
        double best_rank = -std::numeric_limits<double>::infinity();
        int32_t best_token = -1;
        for (size_t i = 0; i < indices.size(); ++i) {
            const float exponential = engine::sampling::torch_cuda_tensor_iterator_exponential_element(
                seed,
                static_cast<uint64_t>(logits.size()),
                static_cast<uint64_t>(indices[i]),
                call_index,
                sampling_policy.multiprocessor_count,
                sampling_policy.max_threads_per_multiprocessor);
            const double rank = weights[i] / static_cast<double>(exponential);
            if (rank > best_rank) {
                best_rank = rank;
                best_token = indices[i];
            }
        }
        if (best_token < 0) {
            throw std::runtime_error("VieNeu-TTS CUDA sampler failed to select a token");
        }
        return best_token;
    }
    std::discrete_distribution<size_t> distribution(weights.begin(), weights.end());
    return indices[distribution(rng)];
}

void apply_main_talker_processors(
    std::vector<float> & logits,
    const VietneuTTSTalkerConfig & config,
    const std::vector<int32_t> & generated_first_codes,
    int64_t step,
    float repetition_penalty) {
    const int64_t expected_size = (static_cast<int64_t>(logits.size()) == config.text_vocab_size) ? config.text_vocab_size : config.vocab_size;
    if (static_cast<int64_t>(logits.size()) != expected_size) {
        throw std::runtime_error("VieNeu-TTS talker logits size mismatch");
    }
    const int64_t eos_token_id = expected_size == config.text_vocab_size ? 2 : config.codec_eos_token_id;
    if (step < 2 && eos_token_id >= 0 && eos_token_id < expected_size) {
        logits[static_cast<size_t>(eos_token_id)] = -std::numeric_limits<float>::infinity();
    }
    const int64_t suppress_start = expected_size == config.text_vocab_size ? expected_size : std::max<int64_t>(0, expected_size - 1024);
    for (int64_t token = suppress_start; token < expected_size; ++token) {
        if (token != eos_token_id) {
            logits[static_cast<size_t>(token)] = -std::numeric_limits<float>::infinity();
        }
    }
    if (repetition_penalty == 1.0F) {
        return;
    }
    if (repetition_penalty <= 0.0F) {
        throw std::runtime_error("VieNeu-TTS talker repetition penalty must be positive");
    }
    std::unordered_set<int32_t> seen_tokens;
    for (const int32_t token : generated_first_codes) {
        if (token < 0 || token >= expected_size) {
            continue;
        }
        if (!seen_tokens.insert(token).second) {
            continue;
        }
        float & value = logits[static_cast<size_t>(token)];
        value = value < 0.0F ? value * repetition_penalty : value / repetition_penalty;
    }
}

std::vector<float> frame_embedding(
    const VietneuTalkerFrameCodes & frame,
    const std::vector<float> & text_hidden,
    const VietneuTalkerWeights & weights,
    const VietneuTTSTalkerConfig & config) {
    if (static_cast<int64_t>(frame.codes.size()) != config.num_code_groups) {
        throw std::runtime_error("VieNeu-TTS talker frame code group count mismatch");
    }
    if (static_cast<int64_t>(text_hidden.size()) != config.hidden_size) {
        throw std::runtime_error("VieNeu-TTS talker frame text hidden size mismatch");
    }
    std::vector<float> out = text_hidden;
    for (int64_t group = 0; group < config.num_code_groups; ++group) {
        const int32_t code = frame.codes[static_cast<size_t>(group)];
        const auto row = group == 0
            ? lookup_rows(weights.codec_embedding, config.hidden_size, {code})
            : lookup_rows(weights.code_predictor_embeddings.at(static_cast<size_t>(group - 1)), config.hidden_size, {code});
        for (int64_t dim = 0; dim < config.hidden_size; ++dim) {
            out[static_cast<size_t>(dim)] += row[static_cast<size_t>(dim)];
        }
    }
    return out;
}

class CodePredictorGraph {
public:
    explicit CodePredictorGraph(std::shared_ptr<const VietneuTalkerWeightsRuntime> weights)
        : weights_(std::move(weights)),
          code_groups_(weights_->assets().config.talker.num_code_groups) {
        if (code_groups_ <= 1) {
            throw std::runtime_error("VieNeu-TTS code predictor requires multiple code groups");
        }
        ggml_init_params params{weights_->graph_arena_bytes(), nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize VieNeu-TTS code predictor graph context");
        }
        const auto & config = weights_->assets().config.code_predictor;
        const int64_t head_dim = attention_head_dim(config);
        core::ModuleBuildContext ctx{ctx_.get(), "vietneu_tts.talker.code_predictor"};
        cache_keys_.reserve(static_cast<size_t>(config.num_hidden_layers));
        cache_values_.reserve(static_cast<size_t>(config.num_hidden_layers));
        const int64_t cache_capacity = code_groups_ + 1;
        for (int64_t layer = 0; layer < config.num_hidden_layers; ++layer) {
            cache_keys_.push_back(core::make_tensor(
                ctx,
                GGML_TYPE_F32,
                core::TensorShape::from_dims({1, cache_capacity, config.num_key_value_heads, head_dim})));
            cache_values_.push_back(core::make_tensor(
                ctx,
                GGML_TYPE_F32,
                core::TensorShape::from_dims({1, cache_capacity, config.num_key_value_heads, head_dim})));
        }
        auto & constants = weights_->code_predictor_constants();
        constants.begin_graph();
        build_prefill_graph(ctx, constants);
        step_graphs_.reserve(static_cast<size_t>(code_groups_ - 1));
        for (int64_t group = 1; group < code_groups_; ++group) {
            step_graphs_.push_back(build_step_graph(ctx, constants, group));
        }
        constants.finish_graph();
        constants.ensure_uploaded();
        buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), weights_->backend());
        if (buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate VieNeu-TTS code predictor graph");
        }

        step_attention_mask_buffer_.assign(static_cast<size_t>(code_groups_ + 1), ggml_fp32_to_fp16(-INFINITY));
        int32_t prefill_positions[2] = {0, 1};
        ggml_backend_tensor_set(prefill_positions_, prefill_positions, 0, sizeof(prefill_positions));
    }

    ~CodePredictorGraph() {
        engine::core::release_backend_graph_resources(weights_->backend(), prefill_graph_);
        for (auto & step_graph : step_graphs_) {
            engine::core::release_backend_graph_resources(weights_->backend(), step_graph.graph);
        }
        if (buffer_ != nullptr) {
            ggml_backend_buffer_free(buffer_);
        }
    }

    VietneuTalkerFrameCodes generate(
        const VietneuTalkerCodePredictorInput & input,
        const VietneuTTSGenerationOptions & options,
        std::mt19937 & rng,
        uint64_t & sample_call_index) {
        timing_ = {};

        auto embeddings = make_prefill_embeddings(input);
        VietneuTalkerFrameCodes out;
        out.codes.reserve(static_cast<size_t>(code_groups_));
        auto logits = run_prefill(embeddings);

        int32_t code = options.subtalker_do_sample
            ? sample_index(
                logits.values,
                options.subtalker_top_k,
                options.subtalker_top_p,
                options.subtalker_temperature,
                rng,
                weights_->sampling_policy(),
                options.seed,
                sample_call_index++)
            : argmax_index(logits.values);
        out.codes.push_back(code);
        for (int64_t group = 1; group < code_groups_; ++group) {
            const auto & w = weights_->weights();
            const auto row = weights_->assets().config.is_vieneu
                ? (group == 1
                    ? lookup_rows(w.codec_embedding, weights_->assets().config.talker.hidden_size, {code})
                    : lookup_rows(w.code_predictor_embeddings.at(static_cast<size_t>(group - 2)), weights_->assets().config.talker.hidden_size, {code}))
                : lookup_rows(w.code_predictor_embeddings.at(static_cast<size_t>(group - 1)), weights_->assets().config.talker.hidden_size, {code});
            logits = run_step(group, row);
            code = options.subtalker_do_sample
                ? sample_index(
                    logits.values,
                    options.subtalker_top_k,
                    options.subtalker_top_p,
                    options.subtalker_temperature,
                    rng,
                    weights_->sampling_policy(),
                    options.seed,
                    sample_call_index++)
                : argmax_index(logits.values);
            out.codes.push_back(code);
        }
        return out;
    }

    const CodePredictorTiming & timing() const noexcept {
        return timing_;
    }

    VietneuTalkerPrefillLogits read_text_logits() {
        const auto & config = weights_->assets().config.talker;
        VietneuTalkerPrefillLogits out;
        out.vocab_size = config.text_vocab_size;
        out.values.resize(static_cast<size_t>(config.text_vocab_size));
        const auto timing_start = Clock::now();
        ggml_backend_tensor_get(prefill_text_logits_, out.values.data(), 0, out.values.size() * sizeof(float));
        timing_.output_read_ms += engine::debug::elapsed_ms(timing_start, Clock::now());
        return out;
    }

private:
    struct StepGraph {
        ggml_tensor * input = nullptr;
        ggml_tensor * position = nullptr;
        ggml_tensor * cache_slot = nullptr;
        ggml_tensor * attention_mask = nullptr;
        ggml_tensor * logits = nullptr;
        ggml_cgraph * graph = nullptr;
    };

    std::vector<float> make_prefill_embeddings(const VietneuTalkerCodePredictorInput & input) const {
        const auto & config = weights_->assets().config;
        if (input.talker_hidden.dims != config.talker.hidden_size ||
            static_cast<int64_t>(input.talker_hidden.values.size()) != config.talker.hidden_size) {
            throw std::runtime_error("VieNeu-TTS code predictor talker hidden shape mismatch");
        }
        if (input.first_code < 0 || input.first_code >= config.talker.text_vocab_size) {
            throw std::runtime_error("VieNeu-TTS code predictor first code out of range");
        }
        std::vector<float> embeddings(static_cast<size_t>(2 * config.talker.hidden_size), 0.0F);
        std::copy(input.talker_hidden.values.begin(), input.talker_hidden.values.end(), embeddings.begin());
        auto code_embed = lookup_rows(
            weights_->weights().text_embedding,
            config.talker.hidden_size,
            {input.first_code});
        std::copy(code_embed.begin(), code_embed.end(), embeddings.begin() + config.talker.hidden_size);
        return embeddings;
    }

    void build_prefill_graph(core::ModuleBuildContext & ctx, core::ConstantTensorCache & constants) {
        const auto & root_config = weights_->assets().config;
        const auto & config = root_config.code_predictor;
        const auto & tensor_weights = weights_->weights();
        auto input = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 2, root_config.talker.hidden_size}));
        prefill_input_ = input.tensor;
        auto x = project_code_predictor_input(ctx, input, tensor_weights, root_config, constants);
        prefill_positions_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, 2);
        auto positions_value = core::wrap_tensor(prefill_positions_, core::TensorShape::from_dims({2}), GGML_TYPE_I32);
        if (root_config.is_vieneu) {
            ggml_tensor * slot_emb = ggml_get_rows(ctx.ggml, binding::tensor_data(constants, *tensor_weights.slot_pos_emb).tensor, positions_value.tensor);
            x = core::wrap_tensor(ggml_add(ctx.ggml, x.tensor, slot_emb), x.shape, x.type);
        }
        const int64_t head_dim = attention_head_dim(config);
        prefill_graph_ = ggml_new_graph_custom(ctx_.get(), 32768, false);
        auto decoder_out = modules::QwenCausalDecoderModule(make_qwen_decoder_config(config, config.vocab_size, root_config.is_vieneu, true))
                               .build(
                                   ctx,
                                   x,
                                   positions_value,
                                   make_qwen_decoder_weights(
                                       constants,
                                       tensor_weights.code_predictor.layers,
                                       tensor_weights.code_predictor.norm,
                                       tensor_weights.code_predictor.lm_heads.front(),
                                       root_config.is_vieneu));
        for (size_t layer_index = 0; layer_index < decoder_out.state.layers.size(); ++layer_index) {
            const auto & layer = decoder_out.state.layers[layer_index];
            if (!layer.key.has_value() || !layer.value.has_value()) {
                throw std::runtime_error("VieNeu-TTS code predictor prefill decoder did not return K/V state");
            }
            auto key_dest = cache_view(ctx, cache_keys_[layer_index], 0, 2, config.num_key_value_heads, head_dim);
            auto value_dest = cache_view(ctx, cache_values_[layer_index], 0, 2, config.num_key_value_heads, head_dim);
            ggml_build_forward_expand(prefill_graph_, ggml_cpy(ctx.ggml, layer.key->tensor, key_dest.tensor));
            ggml_build_forward_expand(prefill_graph_, ggml_cpy(ctx.ggml, layer.value->tensor, value_dest.tensor));
        }
        prefill_logits_ = decoder_out.logits.tensor;
        ggml_set_output(prefill_logits_);
        ggml_build_forward_expand(prefill_graph_, prefill_logits_);

        if (root_config.is_vieneu) {
            ggml_tensor * slot0 = ggml_view_2d(
                ctx.ggml,
                decoder_out.sequence.tensor,
                root_config.talker.hidden_size,
                1,
                decoder_out.sequence.tensor->nb[1],
                0);
            auto norm_module = modules::RMSNormModule({root_config.talker.hidden_size, config.rms_norm_eps, true, false});
            auto norm_out = norm_module.build(
                ctx,
                core::wrap_tensor(slot0, core::TensorShape::from_dims({1, 1, root_config.talker.hidden_size}), GGML_TYPE_F32),
                binding::norm_data(constants, tensor_weights.code_predictor.norm));
            auto lm_head_module = modules::LinearModule(modules::LinearConfig{
                root_config.talker.hidden_size,
                root_config.talker.text_vocab_size,
                false,
                GGML_PREC_DEFAULT
            });
            auto text_logits = lm_head_module.build(
                ctx,
                norm_out,
                binding::linear_data(constants, tensor_weights.codec_head));
            prefill_text_logits_ = text_logits.tensor;
            ggml_set_output(prefill_text_logits_);
            ggml_build_forward_expand(prefill_graph_, prefill_text_logits_);
        }
    }

    StepGraph build_step_graph(
        core::ModuleBuildContext & ctx,
        core::ConstantTensorCache & constants,
        int64_t group) {
        const auto & root_config = weights_->assets().config;
        const auto & config = root_config.code_predictor;
        const auto & tensor_weights = weights_->weights();
        StepGraph step;
        auto input = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 1, root_config.talker.hidden_size}));
        step.input = input.tensor;
        auto x = project_code_predictor_input(ctx, input, tensor_weights, root_config, constants);
        step.position = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, 1);
        auto position_value = core::wrap_tensor(step.position, core::TensorShape::from_dims({1}), GGML_TYPE_I32);
        if (root_config.is_vieneu) {
            ggml_tensor * slot_emb = ggml_get_rows(ctx.ggml, binding::tensor_data(constants, *tensor_weights.slot_pos_emb).tensor, position_value.tensor);
            x = core::wrap_tensor(ggml_add(ctx.ggml, x.tensor, slot_emb), x.shape, x.type);
        }
        step.cache_slot = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, 1);
        auto cache_slot_value = core::wrap_tensor(step.cache_slot, core::TensorShape::from_dims({1}), GGML_TYPE_I32);
        step.attention_mask = ggml_new_tensor_4d(ctx_.get(), GGML_TYPE_F16, code_groups_ + 1, 1, 1, 1);
        auto attention_mask_value = core::wrap_tensor(
            step.attention_mask,
            core::TensorShape::from_dims({1, 1, 1, code_groups_ + 1}),
            GGML_TYPE_F16);
        step.graph = ggml_new_graph_custom(ctx_.get(), 32768, false);
        const auto & step_head = tensor_weights.code_predictor.lm_heads.at(static_cast<size_t>(group));
        const auto decoder_config = make_qwen_decoder_config(config, config.vocab_size, root_config.is_vieneu, true);
        const modules::QwenDecoderLayerModule layer_module(
            modules::qwen_decoder_layer_config_from_stack(decoder_config.stack));
        for (size_t layer_index = 0; layer_index < tensor_weights.code_predictor.layers.size(); ++layer_index) {
            auto layer_out = layer_module.build_with_static_cache_tail(
                ctx,
                step.graph,
                x,
                position_value,
                make_qwen_decoder_layer_weights(constants, tensor_weights.code_predictor.layers[layer_index], root_config.is_vieneu),
                cache_keys_[layer_index],
                cache_values_[layer_index],
                cache_slot_value,
                attention_mask_value);
            x = layer_out.output;
        }
        x = modules::RMSNormModule({config.hidden_size, config.rms_norm_eps, true, false})
                .build(ctx, x, binding::norm_data(constants, tensor_weights.code_predictor.norm));
        auto logits = modules::LinearModule(
                          binding::linear_config(config.hidden_size, config.vocab_size, false))
                          .build(ctx, x, binding::linear_data(constants, step_head));
        step.logits = logits.tensor;
        ggml_set_output(step.logits);
        ggml_build_forward_expand(step.graph, step.logits);
        return step;
    }

    VietneuTalkerPrefillLogits run_prefill(const std::vector<float> & embeddings) {
        const auto & config = weights_->assets().config.talker;
        if (static_cast<int64_t>(embeddings.size()) != 2 * config.hidden_size) {
            throw std::runtime_error("VieNeu-TTS code predictor prefill embedding size mismatch");
        }
        auto timing_start = Clock::now();
        ggml_backend_tensor_set(prefill_input_, embeddings.data(), 0, embeddings.size() * sizeof(float));
        timing_.input_upload_ms += engine::debug::elapsed_ms(timing_start, Clock::now());
        core::set_backend_threads(weights_->backend(), weights_->threads());
        timing_start = Clock::now();
        const ggml_status status = engine::core::compute_backend_graph(weights_->backend(), prefill_graph_);
        ggml_backend_synchronize(weights_->backend());
        timing_.graph_compute_ms += engine::debug::elapsed_ms(timing_start, Clock::now());
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("VieNeu-TTS code predictor prefill graph compute failed");
        }
        valid_steps_ = 2;
        current_end_ = 2;

        return read_logits(prefill_logits_);
    }

    VietneuTalkerPrefillLogits run_step(int64_t group, const std::vector<float> & embedding) {
        const auto & config = weights_->assets().config;
        if (group <= 0 || group >= code_groups_) {
            throw std::runtime_error("VieNeu-TTS code predictor step group out of range");
        }
        if (static_cast<int64_t>(embedding.size()) != config.talker.hidden_size) {
            throw std::runtime_error("VieNeu-TTS code predictor step embedding size mismatch");
        }
        if (valid_steps_ <= 0 || valid_steps_ >= code_groups_ + 1) {
            throw std::runtime_error("VieNeu-TTS code predictor cache state is invalid for step");
        }
        auto & step_graph = step_graphs_.at(static_cast<size_t>(group - 1));
        auto timing_start = Clock::now();
        ggml_backend_tensor_set(step_graph.input, embedding.data(), 0, embedding.size() * sizeof(float));
        const int32_t position = static_cast<int32_t>(current_end_);
        ggml_backend_tensor_set(step_graph.position, &position, 0, sizeof(position));
        const int32_t cache_slot = static_cast<int32_t>(valid_steps_);
        ggml_backend_tensor_set(step_graph.cache_slot, &cache_slot, 0, sizeof(cache_slot));
        std::fill(step_attention_mask_buffer_.begin(), step_attention_mask_buffer_.end(), ggml_fp32_to_fp16(-INFINITY));
        for (int64_t i = 0; i < valid_steps_; ++i) {
            step_attention_mask_buffer_[static_cast<size_t>(i)] = ggml_fp32_to_fp16(0.0F);
        }
        step_attention_mask_buffer_[static_cast<size_t>(cache_slot)] = ggml_fp32_to_fp16(0.0F);
        ggml_backend_tensor_set(
            step_graph.attention_mask,
            step_attention_mask_buffer_.data(),
            0,
            step_attention_mask_buffer_.size() * sizeof(ggml_fp16_t));
        timing_.input_upload_ms += engine::debug::elapsed_ms(timing_start, Clock::now());
        core::set_backend_threads(weights_->backend(), weights_->threads());
        timing_start = Clock::now();
        const ggml_status status = engine::core::compute_backend_graph(weights_->backend(), step_graph.graph);
        ggml_backend_synchronize(weights_->backend());
        timing_.graph_compute_ms += engine::debug::elapsed_ms(timing_start, Clock::now());
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("VieNeu-TTS code predictor step graph compute failed");
        }
        ++valid_steps_;
        ++current_end_;
        return read_logits(step_graph.logits);
    }

    VietneuTalkerPrefillLogits read_logits(ggml_tensor * logits_tensor) {
        const auto & config = weights_->assets().config.code_predictor;

        VietneuTalkerPrefillLogits out;
        out.vocab_size = config.vocab_size;
        out.values.resize(static_cast<size_t>(config.vocab_size));
        const auto timing_start = Clock::now();
        ggml_backend_tensor_get(logits_tensor, out.values.data(), 0, out.values.size() * sizeof(float));
        timing_.output_read_ms += engine::debug::elapsed_ms(timing_start, Clock::now());
        return out;
    }



    std::shared_ptr<const VietneuTalkerWeightsRuntime> weights_;
    int64_t code_groups_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    std::vector<core::TensorValue> cache_keys_;
    std::vector<core::TensorValue> cache_values_;
    ggml_tensor * prefill_input_ = nullptr;
    ggml_tensor * prefill_positions_ = nullptr;
    ggml_tensor * prefill_logits_ = nullptr;
    ggml_tensor * prefill_text_logits_ = nullptr;
    ggml_cgraph * prefill_graph_ = nullptr;
    std::vector<StepGraph> step_graphs_;
    std::vector<ggml_fp16_t> step_attention_mask_buffer_;
    int64_t valid_steps_ = 0;
    int64_t current_end_ = 0;
    ggml_backend_buffer_t buffer_ = nullptr;
    CodePredictorTiming timing_;
};

class VietneuTalkerStepRuntime::Impl {
public:
    Impl(
        std::shared_ptr<const VietneuTalkerWeightsRuntime> weights,
        int64_t prompt_capacity,
        int64_t generation_capacity)
        : weights_(std::move(weights)),
          prompt_capacity_(prompt_capacity),
          generation_capacity_(generation_capacity) {
        if (weights_ == nullptr) {
            throw std::runtime_error("VieNeu-TTS talker step runtime requires weights runtime");
        }
        if (prompt_capacity_ <= 0 || generation_capacity_ <= 0) {
            throw std::runtime_error("VieNeu-TTS talker step runtime requires positive capacities");
        }
    }

    VietneuTalkerCodes generate(
        const VietneuTalkerPrefill & request,
        const VietneuTTSGenerationOptions & options,
        float repetition_penalty) {
        const auto total_start = Clock::now();
        const int64_t max_new_tokens = options.max_new_tokens;
        if (max_new_tokens <= 0 || max_new_tokens > generation_capacity_) {
            throw std::runtime_error("VieNeu-TTS talker generation token count exceeds capacity");
        }
        std::mt19937 rng(options.seed);
        const auto prompt_state_start = Clock::now();
        if (!cached_prompt_state_.has_value() ||
            !cached_prompt_prefill_.has_value() ||
            !talker_prefill_equal(*cached_prompt_prefill_, request)) {
            cached_prompt_state_ = build_prompt_state(request, weights_->assets().config, weights_->weights());
            cached_prompt_prefill_ = request;
        }
        const auto & state = *cached_prompt_state_;
        const auto prompt_state_end = Clock::now();
        const auto & config = weights_->assets().config.talker;
        const int64_t prompt_steps = static_cast<int64_t>(state.prompt.size()) / config.hidden_size;
        if (prompt_steps <= 0 || prompt_steps > prompt_capacity_) {
            throw std::runtime_error("VieNeu-TTS talker prompt exceeds step runtime capacity");
        }
        const auto prefill_start = Clock::now();
        auto prefill_output = run_prefill_embeddings_with_state(state.prompt, prompt_steps);
        const auto prefill_end = Clock::now();
        auto current = std::move(prefill_output.result);
        double code_predictor_build_ms = 0.0;
        if (code_predictor_graph_ == nullptr) {
            const auto build_start = Clock::now();
            code_predictor_graph_ = std::make_unique<CodePredictorGraph>(weights_);
            code_predictor_build_ms = engine::debug::elapsed_ms(build_start, Clock::now());
        }
        double cached_step_build_ms = 0.0;
        double import_prefill_state_ms = 0.0;
        int64_t cached_step_capacity = 0;
        auto cached_state = std::move(prefill_output.state);
        bool cached_graph_has_state = false;
        auto ensure_cached_step_capacity = [&](int64_t required_capacity) {
            if (cached_step_graph_ != nullptr && cached_graph_has_state &&
                !cached_step_graph_->can_run(*weights_, required_capacity)) {
                cached_state = cached_step_graph_->export_state();
                cached_graph_has_state = false;
            }
            if (cached_step_graph_ == nullptr || !cached_step_graph_->can_run(*weights_, required_capacity)) {
                const int64_t old_generated_capacity = cached_step_graph_ == nullptr
                    ? 0
                    : std::max<int64_t>(1, cached_step_graph_->cache_steps() - prompt_steps);
                const int64_t required_generated_capacity = std::max<int64_t>(1, required_capacity - prompt_steps);
                const int64_t next_generated_capacity = std::min<int64_t>(
                    max_new_tokens,
                    std::max<int64_t>(
                        required_generated_capacity,
                        old_generated_capacity > 0 ? old_generated_capacity * 2 : kInitialGeneratedStepCacheFrames));
                const auto build_start = Clock::now();
                cached_step_graph_.reset();
                cached_step_graph_ = std::make_unique<TalkerCachedStepGraph>(
                    weights_,
                    prompt_steps + next_generated_capacity);
                cached_step_build_ms += engine::debug::elapsed_ms(build_start, Clock::now());
            }
            if (!cached_graph_has_state) {
                const auto import_start = Clock::now();
                cached_step_graph_->import_prefill_state(cached_state);
                import_prefill_state_ms += engine::debug::elapsed_ms(import_start, Clock::now());
                cached_graph_has_state = true;
            }
            cached_step_capacity = cached_step_graph_->cache_steps();
        };
        VietneuTalkerCodes out;
        out.generated_codes.code_groups = config.num_code_groups;
        out.decoder_input_codes.code_groups = config.num_code_groups;
        const int64_t trailing_rows = static_cast<int64_t>(state.trailing_text.size()) / config.hidden_size;
        std::vector<int32_t> generated_first_codes;
        generated_first_codes.reserve(static_cast<size_t>(max_new_tokens));
        uint64_t sample_call_index = 0;
        double processor_ms = 0.0;
        double code_predictor_ms = 0.0;
        double frame_embed_ms = 0.0;
        double cached_step_ms = 0.0;
        CodePredictorTiming code_predictor_timing;
        CachedStepTiming cached_step_timing;
        for (int64_t step = 0; step < max_new_tokens; ++step) {
            auto logits = current.logits.values;
            const auto processor_start = Clock::now();
            apply_main_talker_processors(logits, config, generated_first_codes, step, repetition_penalty);
            const int32_t first_code = options.do_sample
                ? sample_index(
                    logits,
                    options.top_k,
                    options.top_p,
                    options.temperature,
                    rng,
                    weights_->sampling_policy(),
                    options.seed,
                    sample_call_index++)
                : argmax_index(logits);

            processor_ms += engine::debug::elapsed_ms(processor_start, Clock::now());
            if (!weights_->assets().config.is_vieneu) {
                if (first_code == config.codec_eos_token_id) {
                    break;
                }
            }
            if (step + 1 >= max_new_tokens) {
                break;
            }
            generated_first_codes.push_back(first_code);
            VietneuTalkerCodePredictorInput predictor_input;
            predictor_input.talker_hidden = current.last_hidden;
            const int32_t sgs_id = static_cast<int32_t>(weights_->assets().config.speech_generation_start_token_id);
            predictor_input.first_code = sgs_id;
            const auto code_predictor_start = Clock::now();
            const auto frame = code_predictor_graph_->generate(predictor_input, options, rng, sample_call_index);
            code_predictor_ms += engine::debug::elapsed_ms(code_predictor_start, Clock::now());
            const auto & predictor_timing = code_predictor_graph_->timing();
            code_predictor_timing.input_upload_ms += predictor_timing.input_upload_ms;
            code_predictor_timing.graph_compute_ms += predictor_timing.graph_compute_ms;
            code_predictor_timing.output_read_ms += predictor_timing.output_read_ms;
            out.generated_codes.codes.insert(out.generated_codes.codes.end(), frame.codes.begin(), frame.codes.end());
            ++out.generated_codes.frames;

            if (weights_->assets().config.is_vieneu) {
                const auto text_logits = code_predictor_graph_->read_text_logits();
                const int32_t text_token = argmax_index(text_logits.values);

                if (text_token == config.codec_eos_token_id) {
                    break;
                }
            }

            const auto frame_embed_start = Clock::now();
            const auto text_hidden = step < trailing_rows
                ? row_at(state.trailing_text, step, config.hidden_size)
                : state.tts_pad;
            const auto embed = frame_embedding(frame, text_hidden, weights_->weights(), config);
            frame_embed_ms += engine::debug::elapsed_ms(frame_embed_start, Clock::now());
            if (step + 1 < max_new_tokens) {
                ensure_cached_step_capacity(prompt_steps + out.generated_codes.frames);
                const auto step_start = Clock::now();
                current = cached_step_graph_->run_step(embed);
                cached_step_ms += engine::debug::elapsed_ms(step_start, Clock::now());
                const auto & step_timing = cached_step_graph_->last_timing();
                cached_step_timing.input_upload_ms += step_timing.input_upload_ms;
                cached_step_timing.mask_upload_ms += step_timing.mask_upload_ms;
                cached_step_timing.graph_compute_ms += step_timing.graph_compute_ms;
                cached_step_timing.output_read_ms += step_timing.output_read_ms;
                cached_step_timing.kv_copy_ms += step_timing.kv_copy_ms;
            }
        }
        if (request.reference_codes.has_value()) {
            out.decoder_input_codes.codes = request.reference_codes->codes;
            out.decoder_input_codes.frames = request.reference_codes->frames;
        }
        out.decoder_input_codes.codes.insert(
            out.decoder_input_codes.codes.end(),
            out.generated_codes.codes.begin(),
            out.generated_codes.codes.end());
        out.decoder_input_codes.frames += out.generated_codes.frames;
        debug::timing_log_scalar("vietneu_tts.talker.prompt_state_ms", engine::debug::elapsed_ms(prompt_state_start, prompt_state_end));
        debug::timing_log_scalar("vietneu_tts.talker.prefill_ms", engine::debug::elapsed_ms(prefill_start, prefill_end));
        debug::timing_log_scalar("vietneu_tts.talker.code_predictor_build_ms", code_predictor_build_ms);
        debug::timing_log_scalar("vietneu_tts.talker.cached_step_build_ms", cached_step_build_ms);
        debug::timing_log_scalar("vietneu_tts.talker.import_prefill_state_ms", import_prefill_state_ms);
        debug::timing_log_scalar("vietneu_tts.talker.processor_ms", processor_ms);
        debug::timing_log_scalar("vietneu_tts.talker.code_predictor_ms", code_predictor_ms);
        debug::timing_log_scalar("vietneu_tts.talker.code_predictor.input_upload_ms", code_predictor_timing.input_upload_ms);
        debug::timing_log_scalar("vietneu_tts.talker.code_predictor.graph.compute_ms", code_predictor_timing.graph_compute_ms);
        debug::timing_log_scalar("vietneu_tts.talker.code_predictor.output_read_ms", code_predictor_timing.output_read_ms);
        debug::timing_log_scalar("vietneu_tts.talker.frame_embed_ms", frame_embed_ms);
        debug::timing_log_scalar("vietneu_tts.talker.cached_step_ms", cached_step_ms);
        debug::timing_log_scalar("vietneu_tts.talker.cached_step.input_upload_ms", cached_step_timing.input_upload_ms);
        debug::timing_log_scalar("vietneu_tts.talker.cached_step.mask_upload_ms", cached_step_timing.mask_upload_ms);
        debug::timing_log_scalar("vietneu_tts.talker.cached_step.graph.compute_ms", cached_step_timing.graph_compute_ms);
        debug::timing_log_scalar("vietneu_tts.talker.cached_step.output_read_ms", cached_step_timing.output_read_ms);
        debug::timing_log_scalar("vietneu_tts.talker.cached_step.kv_copy_ms", cached_step_timing.kv_copy_ms);
        debug::timing_log_scalar("vietneu_tts.talker.total_ms", engine::debug::elapsed_ms(total_start, Clock::now()));
        return out;
    }

    int64_t release_cached_step_graph() {
        const int64_t released_steps = cached_step_graph_ != nullptr ? cached_step_graph_->cache_steps() : 0;
        cached_step_graph_.reset();
        return released_steps;
    }

private:
    TalkerPrefillGraph::OutputWithCache run_prefill_embeddings_with_state(
        const std::vector<float> & embeddings,
        int64_t prompt_steps) {
        if (prompt_steps > prompt_capacity_) {
            throw std::runtime_error("VieNeu-TTS talker prompt exceeds step runtime capacity");
        }
        double graph_build_ms = 0.0;
        const bool graph_cache_hit = graph_ != nullptr && graph_->matches(*weights_, prompt_steps);
        if (!graph_cache_hit) {
            const auto build_start = Clock::now();
            graph_.reset();
            graph_ = std::make_unique<TalkerPrefillGraph>(weights_, prompt_steps);
            graph_build_ms = engine::debug::elapsed_ms(build_start, Clock::now());
        }
        debug::timing_log_scalar("vietneu_tts.talker.prefill.graph.build_ms", graph_build_ms);
        return graph_->run_with_state(embeddings);
    }

    std::shared_ptr<const VietneuTalkerWeightsRuntime> weights_;
    int64_t prompt_capacity_ = 0;
    int64_t generation_capacity_ = 0;
    std::unique_ptr<TalkerPrefillGraph> graph_;
    std::unique_ptr<TalkerCachedStepGraph> cached_step_graph_;
    std::unique_ptr<CodePredictorGraph> code_predictor_graph_;
    std::optional<VietneuTalkerPrefill> cached_prompt_prefill_;
    std::optional<PromptEmbeddingState> cached_prompt_state_;
};

VietneuTalkerStepRuntime::VietneuTalkerStepRuntime(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {
    if (impl_ == nullptr) {
        throw std::runtime_error("VieNeu-TTS talker step runtime requires implementation");
    }
}

VietneuTalkerStepRuntime::~VietneuTalkerStepRuntime() = default;

VietneuTalkerCodes VietneuTalkerStepRuntime::generate(
    const VietneuTalkerPrefill & prefill,
    const VietneuTTSGenerationOptions & options,
    float repetition_penalty) {
    return impl_->generate(prefill, options, repetition_penalty);
}

int64_t VietneuTalkerStepRuntime::release_cached_step_graph() {
    return impl_->release_cached_step_graph();
}

VietneuTalker::VietneuTalker(VietneuTTSTalkerConfig config) : config_(std::move(config)) {}

const VietneuTTSTalkerConfig & VietneuTalker::config() const noexcept {
    return config_;
}

std::shared_ptr<const VietneuTalkerWeightsRuntime> VietneuTalker::create_weights_runtime(
    std::shared_ptr<const VietneuTTSAssets> assets,
    core::BackendType backend_type,
    int device,
    int threads,
    size_t graph_arena_bytes,
    size_t talker_constant_context_bytes,
    size_t code_predictor_constant_context_bytes,
    engine::assets::TensorStorageType weight_storage_type) const {
    return std::make_shared<VietneuTalkerWeightsRuntime>(
        std::move(assets),
        backend_type,
        device,
        threads,
        graph_arena_bytes,
        talker_constant_context_bytes,
        code_predictor_constant_context_bytes,
        weight_storage_type);
}

std::shared_ptr<VietneuTalkerStepRuntime> VietneuTalker::create_step_runtime(
    std::shared_ptr<const VietneuTalkerWeightsRuntime> weights,
    int64_t prompt_capacity,
    int64_t generation_capacity) const {
    return std::make_shared<VietneuTalkerStepRuntime>(
        std::make_unique<VietneuTalkerStepRuntime::Impl>(std::move(weights), prompt_capacity, generation_capacity));
}

}  // namespace engine::models::vietneu_tts
