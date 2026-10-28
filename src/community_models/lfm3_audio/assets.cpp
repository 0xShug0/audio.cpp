#include "engine/community_models/lfm3_audio/assets.h"

#include "engine/community_models/lfm3_audio/published_gguf.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/io/json.h"
#include "engine/framework/model_spec/package.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::community_models::lfm3_audio {
namespace {

namespace json = engine::io::json;
using blocks::LayerKind;
using blocks::StackConfig;
using engine::debug::format_dims;

std::vector<int64_t> require_shape(const assets::TensorSource & source, const std::string & name,
                                   const std::vector<int64_t> & expected = {}) {
    const auto shape = source.require_metadata(name).shape;
    if (!expected.empty() && shape != expected) {
        throw std::runtime_error("LFM3-Audio tensor " + name + " has shape " + format_dims(shape) + ", expected " +
                                 format_dims(expected));
    }

    return shape;
}

// `attention` is the kind every attention layer gets when the stack does not
// distinguish full from sliding (encoder).
std::vector<LayerKind> parse_layer_kinds(const json::Value & config, const char * stack, std::optional<LayerKind> attention) {
    std::vector<LayerKind> kinds;
    for (const auto & value : config.require("layer_types").as_array()) {
        const auto & type = value.as_string();
        if (type == "conv") {
            kinds.push_back(LayerKind::Conv);
        } else if (type == "full_attention") {
            kinds.push_back(attention.value_or(LayerKind::FullAttention));
        } else if (type == "sliding_attention" && !attention) {
            kinds.push_back(LayerKind::SlidingAttention);
        } else {
            throw std::runtime_error(std::string("LFM3-Audio ") + stack + " has an unsupported layer type: " + type);
        }
    }

    if (kinds.size() != static_cast<size_t>(json::require_i64(config, "num_hidden_layers"))) {
        throw std::runtime_error(std::string("LFM3-Audio ") + stack + " layer_types does not match num_hidden_layers");
    }

    return kinds;
}

// rope_parameters is either flat (LFM2: {rope_theta}) or keyed by layer type
// (LFM3: {full_attention: {...}, sliding_attention: {...}}).
const json::Value & rope_parameters(const json::Value & config, const char * layer_type) {
    const auto & rope = config.require("rope_parameters");
    if (const auto * keyed = rope.find(layer_type)) return *keyed;
    if (const auto * full = rope.find("full_attention")) return *full;
    return rope;
}

// Rotated dims as HF counts them, or 0 when every dim rotates.
int64_t full_rotary_dims(const json::Value & full_rope, int64_t head_dim) {
    const auto factor = json::optional_f32(full_rope, "partial_rotary_factor", 1.0f);
    const auto rotary = 2 * (static_cast<int64_t>(factor * static_cast<float>(head_dim)) / 2);
    if (rotary <= 0 || rotary > head_dim) {
        throw std::runtime_error("LFM3-Audio partial_rotary_factor is out of range");
    }

    return rotary == head_dim ? 0 : rotary;
}

bool has_attention(const StackConfig & stack) {
    for (const auto kind : stack.layers) {
        if (kind != LayerKind::Conv) return true;
    }

    return false;
}

// Shape checks, so a wrong or mixed-up checkpoint fails at load.
void check_stack(const assets::TensorSource & source, const std::string & prefix, const StackConfig & stack) {
    const auto h = stack.hidden;
    require_shape(source, prefix + "embedding_norm.weight", {h});
    for (size_t i = 0; i < stack.layers.size(); ++i) {
        const auto p = prefix + "layers." + std::to_string(i) + ".";
        require_shape(source, p + "feed_forward.w1.weight", {stack.ffn, h});
        require_shape(source, p + "feed_forward.w2.weight", {h, stack.ffn});
        if (stack.sandwich_norm) {
            require_shape(source, p + "post_operator_norm.weight", {h});
        }

        if (stack.layers[i] == LayerKind::Conv) {
            require_shape(source, p + "conv.conv.weight", {h, 1, stack.conv_kernel});
            require_shape(source, p + "conv.in_proj.weight", {3 * h, h});
            continue;
        }

        const bool lfm2 = stack.names == blocks::Names::Lfm2;
        require_shape(source, p + "self_attn.q_proj.weight", {stack.heads * stack.head_dim, h});
        require_shape(source, p + "self_attn.k_proj.weight", {stack.kv_heads * stack.head_dim, h});
        require_shape(source, p + (lfm2 ? "self_attn.q_layernorm.weight" : "self_attn.q_norm.weight"), {stack.head_dim});
        if (stack.output_gate) {
            require_shape(source, p + "self_attn.gate_proj.weight", {stack.heads, h});
        }
    }
}

int64_t ffn_width(const assets::TensorSource & source, const std::string & prefix) {
    return require_shape(source, prefix + "layers.0.feed_forward.w1.weight").at(0);
}

StackConfig lfm3_stack(const json::Value & config, const assets::TensorSource & source, const std::string & prefix,
                       const char * stack_name) {
    StackConfig stack;
    stack.hidden = json::require_i64(config, "hidden_size");
    stack.heads = json::require_i64(config, "num_attention_heads");
    stack.kv_heads = json::require_i64(config, "num_key_value_heads");
    stack.head_dim = json::require_i64(config, "head_dim");
    stack.conv_kernel = json::require_i64(config, "conv_L_cache");
    stack.sliding_window = json::optional_nullable_i64(config, "sliding_window", 0);
    stack.eps = json::require_f32(config, "norm_eps");
    stack.layers = parse_layer_kinds(config, stack_name, std::nullopt);
    stack.sandwich_norm = json::optional_bool(config, "use_sandwich_norm", false);
    stack.output_gate = true;
    stack.causal = json::optional_bool(config, "is_causal", true);
    const auto & full = rope_parameters(config, "full_attention");
    stack.full_theta = json::require_f32(full, "rope_theta");
    stack.sliding_theta = json::require_f32(rope_parameters(config, "sliding_attention"), "rope_theta");
    stack.full_rotary_dims = full_rotary_dims(full, stack.head_dim);
    stack.names = blocks::Names::Lfm3;
    stack.ffn = ffn_width(source, prefix);
    check_stack(source, prefix, stack);
    return stack;
}

Lfm3BackboneConfig load_backbone(const json::Value & config, const assets::TensorSource & model) {
    Lfm3BackboneConfig out;
    const auto & lfm = config.require("lfm");
    out.stack = lfm3_stack(lfm, model, "lfm.model.", "backbone");
    out.vocab_size = json::require_i64(lfm, "vocab_size");
    if (!json::optional_bool(lfm, "tie_word_embeddings", false) || model.has_tensor("lfm.lm_head.weight")) {
        throw std::runtime_error("LFM3-Audio backbone must tie its LM head to embed_tokens");
    }

    require_shape(model, "lfm.model.embed_tokens.weight", {out.vocab_size, out.stack.hidden});
    return out;
}

Lfm3EncoderConfig load_encoder(const json::Value & config, const assets::TensorSource & model, int64_t backbone_hidden) {
    Lfm3EncoderConfig out;
    const auto & encoder = config.require("encoder");
    const auto & lfm = encoder.require("lfm");
    out.n_mels = json::require_i64(encoder, "dim_in");
    out.conv_channels = json::require_i64(encoder, "dim_conv");
    // Three stride-2, pad-1, kernel-3 convs: ceil(n / 2) each.
    out.mel_after_downsample = (((out.n_mels + 1) / 2 + 1) / 2 + 1) / 2;

    auto & stack = out.stack;
    stack.hidden = json::require_i64(lfm, "hidden_size");
    stack.heads = json::require_i64(lfm, "num_attention_heads");
    stack.kv_heads = json::require_i64(lfm, "num_key_value_heads");
    if (stack.hidden != json::require_i64(encoder, "dim_out") || stack.heads <= 0 || stack.hidden % stack.heads != 0) {
        throw std::runtime_error("LFM3-Audio encoder hidden size, dim_out and heads are inconsistent");
    }

    stack.head_dim = stack.hidden / stack.heads;
    stack.conv_kernel = json::require_i64(lfm, "conv_L_cache");
    // Every encoder attention layer gets a causal sliding-window mask.
    stack.sliding_window = json::require_i64(encoder, "sliding_window");
    stack.eps = json::require_f32(lfm, "norm_eps");
    stack.layers = parse_layer_kinds(lfm, "encoder", LayerKind::SlidingAttention);
    stack.full_theta = stack.sliding_theta = json::require_f32(rope_parameters(lfm, "full_attention"), "rope_theta");
    stack.names = blocks::Names::Lfm2;
    // config.json intermediate_size is auto-adjusted upstream; the weights hold the real width.
    stack.ffn = ffn_width(model, "encoder.lfm.lfm.");
    check_stack(model, "encoder.lfm.lfm.", stack);

    require_shape(model, "encoder.downsample.conv_in.weight", {out.conv_channels, 1, 3, 3});
    // A Conv1d [hidden, conv * mel, 1] upstream; Liquid's GGUFs drop the unit axis, which a quantized matmul weight cannot keep.
    auto lin_out = require_shape(model, "encoder.downsample.lin_out.weight");
    if (lin_out.size() == 3 && lin_out[2] == 1) {
        lin_out.pop_back();
    }

    if (lin_out != std::vector<int64_t>{stack.hidden, out.conv_channels * out.mel_after_downsample}) {
        throw std::runtime_error("LFM3-Audio tensor encoder.downsample.lin_out.weight has shape " + format_dims(lin_out) + ", expected " +
                                 format_dims({stack.hidden, out.conv_channels * out.mel_after_downsample, 1}));
    }

    require_shape(model, "mlp.model.1.weight", {backbone_hidden, stack.hidden});
    out.output_size = require_shape(model, "mlp.model.3.weight", {backbone_hidden, backbone_hidden}).at(0);
    return out;
}

Lfm3DepthConfig load_depth(const json::Value & config, const assets::TensorSource & model, int64_t backbone_hidden) {
    Lfm3DepthConfig out;
    out.stack = lfm3_stack(config.require("depthformer"), model, "depthformer.model.", "depth decoder");
    out.input_size = backbone_hidden;
    out.codebooks = json::require_i64(config, "codebooks");
    out.audio_vocab_size = json::require_i64(config, "audio_vocab_size");
    const auto steps = json::require_i64(config, "maskgit_steps");
    if (out.codebooks != 12) {
        throw std::runtime_error("LFM3-Audio supports 12 codebooks, the checkpoint has " + std::to_string(out.codebooks));
    }

    if (steps <= 0 || out.codebooks % steps != 0) {
        throw std::runtime_error("LFM3-Audio maskgit_steps must divide the codebook count");
    }

    out.maskgit_schedule.assign(static_cast<size_t>(steps), out.codebooks / steps);
    const auto depth_hidden = out.stack.hidden;
    require_shape(model, "depth_linear.weight", {out.codebooks * depth_hidden, backbone_hidden});
    require_shape(model, "mask_embedding", {depth_hidden});
    require_shape(model, "codebook_embedding.weight", {out.codebooks * out.audio_vocab_size, backbone_hidden});
    for (int64_t i = 0; i < out.codebooks; ++i) {
        const auto p = "depth_embeddings." + std::to_string(i) + ".";
        require_shape(model, p + "embedding.weight", {out.audio_vocab_size, depth_hidden});
        require_shape(model, p + "embedding_norm.weight", {depth_hidden});
    }

    return out;
}

// The codec decoder's dims are hard-coded in the upstream python, so they
// come from tensor shapes; the window and FSQ levels from its config.json.
Lfm3CodecConfig load_codec(const json::Value & config, const assets::TensorSource & codec, const Lfm3DepthConfig & depth) {
    if (json::require_string(config, "model_type") != "liquid_codec") {
        throw std::runtime_error("LFM3-Audio audio_tokenizer/config.json is not a liquid_codec config");
    }

    Lfm3CodecConfig out;
    const std::string prefix = "decoder.lfm.";
    auto & stack = out.stack;
    stack.hidden = require_shape(codec, "from_quantizer.weight").at(0);
    stack.ffn = ffn_width(codec, prefix);
    for (int64_t i = 0; codec.has_tensor(prefix + "layers." + std::to_string(i) + ".operator_norm.weight"); ++i) {
        const auto p = prefix + "layers." + std::to_string(i) + ".";
        if (codec.has_tensor(p + "conv.conv.weight")) {
            stack.layers.push_back(LayerKind::Conv);
            stack.conv_kernel = require_shape(codec, p + "conv.conv.weight").at(2);
        } else if (codec.has_tensor(p + "self_attn.q_proj.weight")) {
            // Upstream LiquidCodecConfig makes every attention layer sliding.
            stack.layers.push_back(LayerKind::SlidingAttention);
            stack.head_dim = require_shape(codec, p + "self_attn.q_norm.weight").at(0);
            if (stack.head_dim <= 0) {
                throw std::runtime_error("LFM3-Audio codec layer " + std::to_string(i) + " has an empty q_norm");
            }
            stack.heads = require_shape(codec, p + "self_attn.q_proj.weight").at(0) / stack.head_dim;
            stack.kv_heads = require_shape(codec, p + "self_attn.k_proj.weight").at(0) / stack.head_dim;
        } else {
            throw std::runtime_error("LFM3-Audio codec layer " + std::to_string(i) + " is neither conv nor attention");
        }
    }

    if (stack.layers.empty() || !has_attention(stack)) {
        throw std::runtime_error("LFM3-Audio codec decoder has no attention layers");
    }

    stack.sliding_window = json::require_i64(config, "decoder_window");
    stack.eps = 1e-5f;
    stack.sandwich_norm = codec.has_tensor(prefix + "layers.0.post_operator_norm.weight");
    stack.output_gate = true;
    stack.full_theta = stack.sliding_theta = 1e6f;
    stack.names = blocks::Names::Lfm3;
    check_stack(codec, prefix, stack);

    out.fsq_levels = json::require_i64_array(config, "fsq_levels");
    out.codebooks = json::require_i64(config, "n_q");
    out.quantizer_dim = json::require_i64(config, "quantizer_dim");
    out.upsample = json::require_i64(config, "stft_frame_stack");
    out.n_fft = json::require_i64(config, "stft_n_fft");
    out.hop = json::require_i64(config, "stft_hop_length");
    out.sample_rate = json::require_i32(config, "sampling_rate");
    int64_t codes = 1;
    for (const auto level : out.fsq_levels) {
        codes *= level;
    }

    if (out.codebooks != depth.codebooks || out.fsq_levels.empty() || codes + 1 != depth.audio_vocab_size) {
        throw std::runtime_error("LFM3-Audio codec codebooks or FSQ levels do not match the model's audio vocabulary");
    }

    require_shape(codec, "quantizer.project_out.weight",
                  {out.quantizer_dim, out.codebooks * static_cast<int64_t>(out.fsq_levels.size())});
    require_shape(codec, "from_quantizer.weight", {stack.hidden, out.quantizer_dim});
    require_shape(codec, "decoder.lin.weight", {out.n_fft + 2, stack.hidden});
    require_shape(codec, "decoder.istft.window", {out.n_fft});
    return out;
}

Lfm3GenerationConfig load_generation(const json::Value & config, const json::Value & generation) {
    Lfm3GenerationConfig out;
    out.interleaved_text = json::require_i64(config, "interleaved_n_text");
    out.interleaved_audio = json::require_i64(config, "interleaved_n_audio");
    for (const auto id : json::require_i64_array_or_scalar(generation, "eos_token_id")) {
        out.specials.eos_ids.push_back(static_cast<int32_t>(id));
    }

    if (out.interleaved_text <= 0 || out.interleaved_audio <= 0 || out.specials.eos_ids.empty()) {
        throw std::runtime_error("LFM3-Audio interleaving or end-of-turn tokens are missing");
    }

    auto & speech = out.speech;
    speech.temperature = json::optional_f32(generation, "audio_temperature", speech.temperature);
    speech.top_k = json::optional_i64(generation, "audio_top_k", speech.top_k);
    speech.repetition_penalty = json::optional_f32(generation, "audio_repetition_penalty", speech.repetition_penalty);
    speech.repetition_window = json::optional_i64(generation, "audio_repetition_window", speech.repetition_window);
    if (speech.temperature < 0.0f || speech.top_k < 0 || speech.repetition_penalty <= 0.0f || speech.repetition_window < 0) {
        throw std::runtime_error("LFM3-Audio generation_config.json audio sampling is out of range");
    }

    return out;
}

// The parts every layout gives; the loader fills in where they come from.
void load_parts(Lfm3AudioAssets & out, const json::Value & config, const json::Value & codec_config, const json::Value & generation) {
    if (json::require_string(config, "model_type") != "lfm3_audio") {
        throw std::runtime_error("LFM3-Audio requires an lfm3_audio config.json");
    }

    out.backbone = load_backbone(config, *out.model);
    out.encoder = load_encoder(config, *out.model, out.backbone.stack.hidden);
    out.depth = load_depth(config, *out.model, out.backbone.stack.hidden);
    out.codec = load_codec(codec_config, *out.codec_weights, out.depth);
    out.generation = load_generation(config, generation);
}

}  // namespace

std::shared_ptr<const Lfm3AudioAssets> load_lfm3_audio_assets(const std::filesystem::path & model_path) {
    auto out = std::make_shared<Lfm3AudioAssets>();
    if (is_lfm3_audio_published_gguf(model_path)) {
        auto published = open_lfm3_audio_published_gguf(model_path);
        out->model = std::move(published.model);
        out->codec_weights = std::move(published.codec);
        out->vocabulary = std::move(published.vocabulary);
        out->published_backbone = std::move(published.backbone_path);
        load_parts(*out, published.config, published.codec_config, published.generation_config);
        return out;
    }

    out->resources = engine::model_spec::load_resource_bundle_for_family(model_path, "lfm3_audio");
    out->model = out->resources.open_tensor_source("model");
    out->codec_weights = out->resources.open_tensor_source("codec");
    out->tokenizer_json = out->resources.require_file("tokenizer_json");
    load_parts(*out, out->resources.parse_json("config"), out->resources.parse_json("codec_config"),
               out->resources.parse_json("generation_config"));
    return out;
}

}  // namespace engine::community_models::lfm3_audio
