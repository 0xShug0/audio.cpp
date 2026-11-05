#include "engine/community_models/lfm3_audio/assets.h"

#include "engine/framework/assets/gguf_metadata.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/modules/weight_binding.h"
#include "engine/framework/io/filesystem.h"
#include "engine/framework/io/json.h"
#include "engine/framework/model_spec/package.h"

#include <algorithm>
#include <cctype>
#include <cmath>
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
using core::TensorShape;
using core::TensorValue;
using LayerKind = modules::LfmDecoderLayerKind;
using StackConfig = modules::LfmDecoderConfig;
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
            kinds.push_back(LayerKind::ShortConv);
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

    return rotary == head_dim ? -1 : rotary;
}

bool has_attention(const StackConfig & stack) {
    for (const auto kind : stack.layers) {
        if (kind != LayerKind::ShortConv) return true;
    }

    return false;
}

// Shape checks, so a wrong or mixed-up checkpoint fails at load.
void check_stack(const assets::TensorSource & source, const std::string & prefix, const StackConfig & stack) {
    const auto h = stack.hidden_size;
    require_shape(source, prefix + "embedding_norm.weight", {h});
    for (size_t i = 0; i < stack.layers.size(); ++i) {
        const auto p = prefix + "layers." + std::to_string(i) + ".";
        require_shape(source, p + "feed_forward.w1.weight", {stack.intermediate_size, h});
        require_shape(source, p + "feed_forward.w2.weight", {h, stack.intermediate_size});
        if (stack.sandwich_norm) {
            require_shape(source, p + "post_operator_norm.weight", {h});
        }

        if (stack.layers[i] == LayerKind::ShortConv) {
            require_shape(source, p + "conv.conv.weight", {h, 1, stack.conv_kernel_size});
            require_shape(source, p + "conv.in_proj.weight", {3 * h, h});
            continue;
        }

        const bool lfm2 = prefix == "encoder.lfm.lfm.";
        require_shape(source, p + "self_attn.q_proj.weight", {stack.num_attention_heads * stack.head_dim, h});
        require_shape(source, p + "self_attn.k_proj.weight", {stack.num_key_value_heads * stack.head_dim, h});
        require_shape(source, p + (lfm2 ? "self_attn.q_layernorm.weight" : "self_attn.q_norm.weight"), {stack.head_dim});
        if (stack.attention_gate) {
            require_shape(source, p + "self_attn.gate_proj.weight", {stack.num_attention_heads, h});
        }
    }
}

int64_t ffn_width(const assets::TensorSource & source, const std::string & prefix) {
    return require_shape(source, prefix + "layers.0.feed_forward.w1.weight").at(0);
}

StackConfig lfm3_stack(const json::Value & config, const assets::TensorSource & source, const std::string & prefix,
                       const char * stack_name) {
    StackConfig stack;
    stack.hidden_size = json::require_i64(config, "hidden_size");
    stack.num_attention_heads = json::require_i64(config, "num_attention_heads");
    stack.num_key_value_heads = json::require_i64(config, "num_key_value_heads");
    stack.head_dim = json::require_i64(config, "head_dim");
    stack.conv_kernel_size = json::require_i64(config, "conv_L_cache");
    stack.norm_eps = json::require_f32(config, "norm_eps");
    stack.layers = parse_layer_kinds(config, stack_name, std::nullopt);
    stack.sandwich_norm = json::optional_bool(config, "use_sandwich_norm", false);
    stack.attention_gate = true;
    const auto & full = rope_parameters(config, "full_attention");
    stack.full_rope.theta = json::require_f32(full, "rope_theta");
    stack.sliding_rope.theta = json::require_f32(rope_parameters(config, "sliding_attention"), "rope_theta");
    stack.full_rope.rotary_dims = full_rotary_dims(full, stack.head_dim);
    stack.intermediate_size = ffn_width(source, prefix);
    check_stack(source, prefix, stack);
    return stack;
}

Lfm3BackboneConfig load_backbone(const json::Value & config, const assets::TensorSource & model) {
    Lfm3BackboneConfig out;
    const auto & lfm = config.require("lfm");
    out.stack = lfm3_stack(lfm, model, "lfm.model.", "backbone");
    out.sliding_window = json::optional_nullable_i64(lfm, "sliding_window", 0);
    if (!json::optional_bool(lfm, "is_causal", true)) {
        throw std::runtime_error("LFM3-Audio backbone must use causal attention");
    }
    out.vocab_size = json::require_i64(lfm, "vocab_size");
    if (!json::optional_bool(lfm, "tie_word_embeddings", false) || model.has_tensor("lfm.lm_head.weight")) {
        throw std::runtime_error("LFM3-Audio backbone must tie its LM head to embed_tokens");
    }

    require_shape(model, "lfm.model.embed_tokens.weight", {out.vocab_size, out.stack.hidden_size});
    return out;
}

Lfm3EncoderConfig load_encoder(const json::Value & config, const assets::TensorSource & model, int64_t backbone_hidden) {
    Lfm3EncoderConfig out;
    const auto & encoder = config.require("encoder");
    const auto & lfm = encoder.require("lfm");
    out.mel_bins = json::require_i64(encoder, "dim_in");
    if (out.mel_bins <= 0 || out.mel_bins % 8 != 0) {
        throw std::runtime_error("LFM3-Audio encoder mel bins must be positive and divisible by 8");
    }
    out.conv_channels = json::require_i64(encoder, "dim_conv");
    // Three stride-2, pad-1, kernel-3 convs: ceil(n / 2) each.

    auto & stack = out.decoder;
    stack.sandwich_norm = false;
    stack.attention_gate = false;
    stack.hidden_size = json::require_i64(lfm, "hidden_size");
    stack.num_attention_heads = json::require_i64(lfm, "num_attention_heads");
    stack.num_key_value_heads = json::require_i64(lfm, "num_key_value_heads");
    if (stack.hidden_size != json::require_i64(encoder, "dim_out") || stack.num_attention_heads <= 0 || stack.hidden_size % stack.num_attention_heads != 0) {
        throw std::runtime_error("LFM3-Audio encoder hidden size, dim_out and heads are inconsistent");
    }

    stack.head_dim = stack.hidden_size / stack.num_attention_heads;
    stack.conv_kernel_size = json::require_i64(lfm, "conv_L_cache");
    // Every encoder attention layer gets a causal sliding-window mask.
    out.sliding_window = json::require_i64(encoder, "sliding_window");
    stack.norm_eps = json::require_f32(lfm, "norm_eps");
    stack.layers = parse_layer_kinds(lfm, "encoder", LayerKind::SlidingAttention);
    stack.full_rope.theta = stack.sliding_rope.theta = json::require_f32(rope_parameters(lfm, "full_attention"), "rope_theta");
    // config.json intermediate_size is auto-adjusted upstream; the weights hold the real width.
    stack.intermediate_size = ffn_width(model, "encoder.lfm.lfm.");
    check_stack(model, "encoder.lfm.lfm.", stack);

    require_shape(model, "encoder.downsample.conv_in.weight", {out.conv_channels, 1, 3, 3});
    // A Conv1d [hidden, conv * mel, 1] upstream; Liquid's GGUFs drop the unit axis, which a quantized matmul weight cannot keep.
    auto lin_out = require_shape(model, "encoder.downsample.lin_out.weight");
    if (lin_out.size() == 3 && lin_out[2] == 1) {
        lin_out.pop_back();
    }

    if (lin_out != std::vector<int64_t>{stack.hidden_size, out.conv_channels * (out.mel_bins / 8)}) {
        throw std::runtime_error("LFM3-Audio tensor encoder.downsample.lin_out.weight has shape " + format_dims(lin_out) + ", expected " +
                                 format_dims({stack.hidden_size, out.conv_channels * (out.mel_bins / 8), 1}));
    }

    require_shape(model, "mlp.model.1.weight", {backbone_hidden, stack.hidden_size});
    out.output_size = require_shape(model, "mlp.model.3.weight", {backbone_hidden, backbone_hidden}).at(0);
    return out;
}

Lfm3DepthConfig load_depth(const json::Value & config, const assets::TensorSource & model, int64_t backbone_hidden) {
    Lfm3DepthConfig out;
    out.stack = lfm3_stack(config.require("depthformer"), model, "depthformer.model.", "depth decoder");
    out.input_size = backbone_hidden;
    if (json::optional_bool(config.require("depthformer"), "is_causal", true)) {
        throw std::runtime_error("LFM3-Audio depth decoder must use bidirectional attention");
    }
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
    const auto depth_hidden = out.stack.hidden_size;
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
    stack.hidden_size = require_shape(codec, "from_quantizer.weight").at(0);
    stack.intermediate_size = ffn_width(codec, prefix);
    for (int64_t i = 0; codec.has_tensor(prefix + "layers." + std::to_string(i) + ".operator_norm.weight"); ++i) {
        const auto p = prefix + "layers." + std::to_string(i) + ".";
        if (codec.has_tensor(p + "conv.conv.weight")) {
            stack.layers.push_back(LayerKind::ShortConv);
            stack.conv_kernel_size = require_shape(codec, p + "conv.conv.weight").at(2);
        } else if (codec.has_tensor(p + "self_attn.q_proj.weight")) {
            // Upstream LiquidCodecConfig makes every attention layer sliding.
            stack.layers.push_back(LayerKind::SlidingAttention);
            stack.head_dim = require_shape(codec, p + "self_attn.q_norm.weight").at(0);
            if (stack.head_dim <= 0) {
                throw std::runtime_error("LFM3-Audio codec layer " + std::to_string(i) + " has an empty q_norm");
            }
            stack.num_attention_heads = require_shape(codec, p + "self_attn.q_proj.weight").at(0) / stack.head_dim;
            stack.num_key_value_heads = require_shape(codec, p + "self_attn.k_proj.weight").at(0) / stack.head_dim;
        } else {
            throw std::runtime_error("LFM3-Audio codec layer " + std::to_string(i) + " is neither conv nor attention");
        }
    }

    if (stack.layers.empty() || !has_attention(stack)) {
        throw std::runtime_error("LFM3-Audio codec decoder has no attention layers");
    }

    out.sliding_window = json::require_i64(config, "decoder_window");
    stack.norm_eps = 1e-5f;
    stack.sandwich_norm = codec.has_tensor(prefix + "layers.0.post_operator_norm.weight");
    stack.attention_gate = true;
    stack.full_rope.theta = stack.sliding_rope.theta = 1e6f;
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
    require_shape(codec, "from_quantizer.weight", {stack.hidden_size, out.quantizer_dim});
    require_shape(codec, "decoder.lin.weight", {out.n_fft + 2, stack.hidden_size});
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

// HF configuration and tensor validation.
void load_hf_configs(Lfm3AudioAssets & out, const json::Value & config, const json::Value & codec_config, const json::Value & generation) {
    if (json::require_string(config, "model_type") != "lfm3_audio") {
        throw std::runtime_error("LFM3-Audio requires an lfm3_audio config.json");
    }

    out.backbone = load_backbone(config, *out.model);
    out.encoder = load_encoder(config, *out.model, out.backbone.stack.hidden_size);
    out.depth = load_depth(config, *out.model, out.backbone.stack.hidden_size);
    out.codec = load_codec(codec_config, *out.codec_weights, out.depth);
    out.generation = load_generation(config, generation);
}

ggml_type stored_type(const assets::TensorSource & source, const std::string & name) {
    return assets::ggml_type_for_tensor_dtype(source.require_metadata(name).dtype);
}

TensorValue lookup_of(Lfm3Weights & stores, ggml_backend_t backend, const assets::TensorSource & source, const std::string & name,
                      const TensorValue & table) {
    if (lfm3_backend_gathers(backend, table.tensor->type)) {
        return table;
    }
    return stores.plain().load_tensor(source, name, assets::TensorStorageType::F16, {table.shape.dims[0], table.shape.dims[1]});
}


using assets::GgufMetadata;

constexpr const char * kMmprojPrefix = "mmproj-";

bool has_gguf_extension(const std::filesystem::path & path) {
    auto extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) { return std::tolower(c); });
    return extension == ".gguf";
}

bool is_mmproj_name(const std::filesystem::path & path) {
    return path.filename().string().rfind(kMmprojPrefix, 0) == 0;
}

bool is_backbone(const std::filesystem::path & path) {
    return has_gguf_extension(path) && !is_mmproj_name(path) && GgufMetadata(path).find_string("general.architecture") == "lfm3";
}

bool is_audio_mmproj(const std::filesystem::path & path) {
    return has_gguf_extension(path) && is_mmproj_name(path) && GgufMetadata(path).find_string("clip.audio.projector_type") == "lfm3a";
}

std::vector<std::filesystem::path> list(const std::filesystem::path & root, bool (*keep)(const std::filesystem::path &)) {
    auto out = assets::directory_gguf_files(root);
    out.erase(std::remove_if(out.begin(), out.end(), [&](const auto & path) { return !keep(path); }), out.end());
    return out;
}

std::string names(const std::vector<std::filesystem::path> & paths) {
    std::string out;
    for (const auto & path : paths) {
        out += (out.empty() ? "" : ", ") + path.filename().string();
    }

    return out;
}

std::filesystem::path pick_mmproj(const std::filesystem::path & backbone) {
    const auto root = backbone.parent_path().empty() ? std::filesystem::path(".") : backbone.parent_path();
    const auto mmprojs = list(root, is_audio_mmproj);
    const auto named = [&](const std::string & name) {
        return std::find_if(mmprojs.begin(), mmprojs.end(), [&](const auto & path) { return path.filename() == name; });
    };

    if (const auto paired = named(kMmprojPrefix + backbone.filename().string()); paired != mmprojs.end()) {
        return *paired;
    }

    // LFM3-Audio-2B-Q4_K_M.gguf goes with mmproj-LFM3-Audio-2B-Q8_0.gguf in Liquid's own instructions.
    const auto stem = backbone.stem().string();
    const auto quant = stem.rfind('-');
    if (quant != std::string::npos) {
        if (const auto q8 = named(kMmprojPrefix + stem.substr(0, quant) + "-Q8_0.gguf"); q8 != mmprojs.end()) {
            return *q8;
        }
    }

    if (mmprojs.size() == 1) {
        return mmprojs.front();
    }

    throw std::runtime_error("LFM3-Audio cannot pick the mmproj GGUF for " + backbone.filename().string() +
                             (mmprojs.empty() ? std::string(": none found in ") + root.string() : " among " + names(mmprojs)));
}

std::filesystem::path pick_backbone(const std::filesystem::path & model_path) {
    if (!io::is_existing_directory(model_path)) {
        return model_path;
    }

    const auto backbones = list(model_path, is_backbone);
    if (backbones.size() != 1) {
        throw std::runtime_error("LFM3-Audio model directory " + model_path.string() +
                                 (backbones.empty() ? std::string(" has no LFM3 backbone GGUF")
                                                    : " has several LFM3 backbone GGUFs (" + names(backbones) + "); pass one as the model"));
    }

    return backbones.front();
}


void load_gguf_configs(Lfm3AudioAssets & out, const GgufMetadata & model, const GgufMetadata & mmproj) {
    const auto & backbone = *out.model;
    const auto & tower = *out.audio_weights;
    auto & b = out.backbone.stack;
    b.hidden_size = model.require_int("lfm3.embedding_length");
    b.intermediate_size = backbone.require_metadata("blk.0.ffn_gate.weight").shape.at(0);
    b.num_attention_heads = model.require_int("lfm3.attention.head_count");
    b.head_dim = model.require_int("lfm3.attention.key_length");
    b.conv_kernel_size = model.require_int("lfm3.shortconv.l_cache");
    out.backbone.sliding_window = model.require_int("lfm3.attention.sliding_window");
    b.norm_eps = model.require_f32("lfm3.attention.layer_norm_rms_epsilon");
    b.full_rope.theta = model.require_f32("lfm3.rope.freq_base");
    b.sliding_rope.theta = model.require_f32("lfm3.rope.freq_base_swa");
    b.sandwich_norm = backbone.has_tensor("blk.0.post_attention_norm.weight");
    b.attention_gate = true;
    const auto kv = model.require_int_array("lfm3.attention.head_count_kv");
    const auto sliding = model.require_int_array("lfm3.attention.sliding_window_pattern", true);
    if (kv.size() != sliding.size() || kv.size() != static_cast<size_t>(model.require_int("lfm3.block_count", 1, model.tensor_count()))) {
        throw std::runtime_error("LFM3-Audio backbone GGUF layer metadata does not match its block count");
    }
    for (size_t i = 0; i < kv.size(); ++i) {
        b.layers.push_back(kv[i] == 0 ? LayerKind::ShortConv : sliding[i] != 0 ? LayerKind::SlidingAttention : LayerKind::FullAttention);
        b.num_key_value_heads = std::max(b.num_key_value_heads, kv[i]);
    }
    if (const auto factors = backbone.optional_f32("rope_freqs.weight", {b.head_dim / 2})) {
        const int64_t rotated = 2 * std::count(factors->begin(), factors->end(), 1.0f);
        if (rotated <= 0 || rotated > b.head_dim) {
            throw std::runtime_error("LFM3-Audio GGUF partial rotary dimensions are out of range");
        }
        b.full_rope.rotary_dims = rotated == b.head_dim ? -1 : rotated;
    }
    out.backbone.vocab_size = backbone.require_metadata("token_embd.weight").shape.at(0);
    if (backbone.has_tensor("output.weight")) {
        throw std::runtime_error("LFM3-Audio backbone must tie its LM head to embed_tokens");
    }

    auto & encoder = out.encoder;
    auto & e = encoder.decoder;
    encoder.mel_bins = mmproj.require_int("clip.audio.num_mel_bins");
    if (encoder.mel_bins <= 0 || encoder.mel_bins % 8 != 0) {
        throw std::runtime_error("LFM3-Audio encoder mel bins must be positive and divisible by 8");
    }
    encoder.conv_channels = tower.require_metadata("a.conv1d.0.weight").shape.at(0);
    encoder.output_size = b.hidden_size;
    e.hidden_size = mmproj.require_int("clip.audio.embedding_length");
    e.intermediate_size = tower.require_metadata("a.blk.0.ffn_gate.weight").shape.at(0);
    e.num_attention_heads = mmproj.require_int("clip.audio.attention.head_count", 1, e.hidden_size);
    e.num_key_value_heads = mmproj.require_int("clip.audio.attention.head_count_kv");
    if (e.hidden_size % e.num_attention_heads != 0) {
        throw std::runtime_error("LFM3-Audio GGUF encoder hidden size is not divisible by its head count");
    }
    e.head_dim = e.hidden_size / e.num_attention_heads;
    e.conv_kernel_size = tower.require_metadata("a.blk.0.conv_dw.weight").shape.at(1);
    encoder.sliding_window = mmproj.require_int("clip.audio.window_size");
    e.norm_eps = mmproj.require_f32("clip.audio.attention.layer_norm_epsilon");
    e.sandwich_norm = false;
    e.attention_gate = false;
    const auto encoder_layers = mmproj.require_int("clip.audio.block_count", 1, mmproj.tensor_count());
    for (int64_t i = 0; i < encoder_layers; ++i) {
        e.layers.push_back(tower.has_tensor("a.blk." + std::to_string(i) + ".attn_q.weight") ? LayerKind::SlidingAttention : LayerKind::ShortConv);
    }

    auto & depth = out.depth;
    auto & d = depth.stack;
    const auto & embedding = tower.require_metadata("a.gen.code.embd.weight").shape;
    depth.codebooks = embedding.at(0);
    depth.audio_vocab_size = embedding.at(1);
    depth.input_size = b.hidden_size;
    if (depth.codebooks != 12 || depth.audio_vocab_size != 2017) {
        throw std::runtime_error("LFM3-Audio GGUF requires 12 codebooks with 2017 audio tokens");
    }
    depth.maskgit_schedule = {4, 4, 4};
    d.hidden_size = mmproj.require_int("clip.gen.audio.embedding_length");
    d.intermediate_size = tower.require_metadata("a.gen.code.blk.0.ffn_gate.weight").shape.at(0);
    d.num_attention_heads = mmproj.require_int("clip.gen.audio.attention.head_count", 1, d.hidden_size);
    d.num_key_value_heads = mmproj.require_int("clip.gen.audio.attention.head_count_kv");
    if (d.hidden_size % d.num_attention_heads != 0) {
        throw std::runtime_error("LFM3-Audio GGUF depth hidden size is not divisible by its head count");
    }
    d.head_dim = d.hidden_size / d.num_attention_heads;
    d.conv_kernel_size = b.conv_kernel_size;
    d.norm_eps = mmproj.require_f32("clip.gen.audio.attention.layer_norm_epsilon");
    d.attention_gate = true;
    d.sandwich_norm = false;
    d.layers.assign(static_cast<size_t>(mmproj.require_int("clip.gen.audio.block_count", 1, mmproj.tensor_count())), LayerKind::FullAttention);

    // The published LiquidCodec layout fixes these values; the converter rejects other configurations.
    auto & codec = out.codec;
    codec.fsq_levels = {8, 7, 6, 6};
    codec.codebooks = 12;
    codec.quantizer_dim = 256;
    codec.upsample = 6;
    codec.n_fft = 1280;
    codec.hop = 320;
    codec.sample_rate = 24000;
    auto & c = codec.stack;
    c.hidden_size = tower.require_metadata("a.gen.wav.tfm.in_proj.weight").shape.at(0);
    c.intermediate_size = tower.require_metadata("a.gen.wav.tfm.blk.0.ffn_gate.weight").shape.at(0);
    for (int64_t i = 0; tower.has_tensor("a.gen.wav.tfm.blk." + std::to_string(i) + ".ffn_gate.weight"); ++i) {
        const auto prefix = "a.gen.wav.tfm.blk." + std::to_string(i) + ".";
        if (tower.has_tensor(prefix + "conv_dw.weight")) {
            c.layers.push_back(LayerKind::ShortConv);
            c.conv_kernel_size = tower.require_metadata(prefix + "conv_dw.weight").shape.at(1);
        } else if (tower.has_tensor(prefix + "attn_q.weight")) {
            c.layers.push_back(LayerKind::SlidingAttention);
            c.head_dim = tower.require_metadata(prefix + "attn_q_norm.weight").shape.at(0);
            if (c.head_dim <= 0) {
                throw std::runtime_error("LFM3-Audio GGUF codec q_norm is empty");
            }
            c.num_attention_heads = tower.require_metadata(prefix + "attn_q.weight").shape.at(0) / c.head_dim;
            c.num_key_value_heads = tower.require_metadata(prefix + "attn_k.weight").shape.at(0) / c.head_dim;
        } else {
            throw std::runtime_error("LFM3-Audio GGUF codec layer is neither conv nor attention");
        }
    }
    if (std::find(c.layers.begin(), c.layers.end(), LayerKind::SlidingAttention) == c.layers.end()) {
        throw std::runtime_error("LFM3-Audio codec decoder has no attention layers");
    }
    codec.sliding_window = 30;
    c.sandwich_norm = tower.has_tensor("a.gen.wav.tfm.blk.0.attn_post_norm.weight");
    c.attention_gate = true;

    const auto text_end = std::find(out.vocabulary.tokens.begin(), out.vocabulary.tokens.end(), "<|text_end|>");
    if (text_end == out.vocabulary.tokens.end()) {
        throw std::runtime_error("LFM3-Audio backbone GGUF vocabulary has no <|text_end|>");
    }
    auto & generation = out.generation;
    generation.interleaved_text = 6;
    generation.interleaved_audio = 12;
    generation.specials.eos_ids = {static_cast<int32_t>(model.require_int("tokenizer.ggml.eos_token_id")),
                                  static_cast<int32_t>(text_end - out.vocabulary.tokens.begin())};
    auto & speech = generation.speech;
    if (const auto value = mmproj.find_number("clip.gen.audio.sampling.temp")) speech.temperature = static_cast<float>(*value);
    if (const auto value = mmproj.find_number("clip.gen.audio.sampling.top_k")) {
        if (*value < 0 || *value >= static_cast<double>(INT64_MAX) || std::floor(*value) != *value) {
            throw std::runtime_error("LFM3-Audio GGUF audio_top_k must be a nonnegative integer");
        }
        speech.top_k = static_cast<int64_t>(*value);
    }
    if (const auto value = mmproj.find_number("clip.gen.audio.sampling.penalty_repeat")) speech.repetition_penalty = static_cast<float>(*value);
    if (const auto value = mmproj.find_number("clip.gen.audio.sampling.penalty_last_n")) {
        if (*value < 0 || *value >= static_cast<double>(INT64_MAX) || std::floor(*value) != *value) {
            throw std::runtime_error("LFM3-Audio GGUF audio_repetition_window must be a nonnegative integer");
        }
        speech.repetition_window = static_cast<int64_t>(*value);
    }
    if (speech.temperature < 0 || !std::isfinite(speech.temperature) || speech.repetition_penalty <= 0 || !std::isfinite(speech.repetition_penalty)) {
        throw std::runtime_error("LFM3-Audio GGUF audio sampling is out of range");
    }
}

}  // namespace


void Lfm3AudioAssets::release_storage() const {
    std::vector<const assets::TensorSource *> released;
    for (const auto & source : {model, audio_weights, codec_weights}) {
        if (source && std::find(released.begin(), released.end(), source.get()) == released.end()) {
            source->release_storage();
            released.push_back(source.get());
        }
    }
}

bool is_lfm3_audio_gguf(const std::filesystem::path & model_path) noexcept {
    try {
        const bool file = io::is_existing_file(model_path);
        if (!file && !io::is_existing_directory(model_path)) {
            return false;
        }

        const auto root = file ? (model_path.parent_path().empty() ? std::filesystem::path(".") : model_path.parent_path()) : model_path;
        const bool backbone = file ? is_backbone(model_path) : !list(root, is_backbone).empty();
        return backbone && !list(root, is_audio_mmproj).empty();
    } catch (const std::exception &) {
        return false;
    }
}

std::shared_ptr<const Lfm3AudioAssets> load_lfm3_audio_gguf_assets(const std::filesystem::path & model_path) {
    auto result = std::make_shared<Lfm3AudioAssets>();
    auto & out = *result;
    out.published_backbone = pick_backbone(model_path);
    if (!is_backbone(out.published_backbone)) {
        throw std::runtime_error("LFM3-Audio model must be a backbone GGUF with general.architecture lfm3: " + out.published_backbone.string());
    }

    const auto mmproj_path = pick_mmproj(out.published_backbone);
    const GgufMetadata model_meta(out.published_backbone);
    const GgufMetadata mmproj_meta(mmproj_path);
    if (mmproj_meta.find_bool("clip.has_vision_encoder").value_or(false)) {
        throw std::runtime_error("LFM3-Audio does not run LFM3-Omni; " + mmproj_path.filename().string() + " has a vision encoder");
    }

    if (!mmproj_meta.find_bool("clip.has_gen_audio_encoder").value_or(false)) {
        throw std::runtime_error("LFM3-Audio mmproj GGUF has no audio output (clip.has_gen_audio_encoder): " + mmproj_path.string());
    }

    if (model_meta.find_string("tokenizer.ggml.model") != "gpt2" || model_meta.find_string("tokenizer.ggml.pre") != "lfm2") {
        throw std::runtime_error("LFM3-Audio expects the byte-level BPE tokenizer (gpt2, pre lfm2) in " + out.published_backbone.string());
    }

    out.vocabulary.tokens = model_meta.require_string_array("tokenizer.ggml.tokens");
    out.vocabulary.merges = model_meta.require_string_array("tokenizer.ggml.merges");
    const auto types = model_meta.require_int_array("tokenizer.ggml.token_type");
    out.vocabulary.token_types.assign(types.begin(), types.end());

    out.model = assets::open_tensor_source(out.published_backbone);
    out.audio_weights = assets::open_tensor_source(mmproj_path);
    out.codec_weights = out.audio_weights;
    load_gguf_configs(out, model_meta, mmproj_meta);
    return result;
}

std::shared_ptr<const Lfm3AudioAssets> load_lfm3_audio_assets(const std::filesystem::path & model_path) {
    if (is_lfm3_audio_gguf(model_path)) {
        return load_lfm3_audio_gguf_assets(model_path);
    }

    auto out = std::make_shared<Lfm3AudioAssets>();
    out->resources = engine::model_spec::load_resource_bundle_for_family(model_path, "lfm3_audio");
    out->model = out->resources.open_tensor_source("model");
    out->codec_weights = out->resources.open_tensor_source("codec");
    out->tokenizer_json = out->resources.require_file("tokenizer_json");
    load_hf_configs(*out, out->resources.parse_json("config"), out->resources.parse_json("codec_config"),
               out->resources.parse_json("generation_config"));
    return out;
}

modules::LfmDecoderWeights load_lfm3_stack_weights(Lfm3Weights & stores, const assets::TensorSource & source, const std::string & prefix,
    const modules::LfmDecoderConfig & config, Lfm3StackLayout layout) {
    const int64_t d = config.hidden_size;
    const int64_t hd = config.head_dim;
    struct TensorNames {
        const char * layers;
        const char * final_norm;
        const char * operator_norm;
        const char * ffn_norm;
        const char * post_operator_norm;
        const char * post_ffn_norm;
        const char * w1;
        const char * w2;
        const char * w3;
        const char * conv_in;
        const char * conv_out;
        const char * conv_kernel;
        const char * q;
        const char * k;
        const char * v;
        const char * o;
        const char * q_norm;
        const char * k_norm;
        const char * gate;
    };
    static constexpr TensorNames hf = {
        "layers.", "embedding_norm", "operator_norm", "ffn_norm", "post_operator_norm", "post_ffn_norm",
        "feed_forward.w1", "feed_forward.w2", "feed_forward.w3", "conv.in_proj", "conv.out_proj", "conv.conv",
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "self_attn.q_norm", "self_attn.k_norm", "self_attn.gate_proj"};
    static constexpr TensorNames gguf_backbone = {
        "blk.", "output_norm", "attn_norm", "ffn_norm", "post_attention_norm", "post_ffw_norm",
        "ffn_gate", "ffn_down", "ffn_up", "shortconv.in_proj", "shortconv.out_proj", "shortconv.conv",
        "attn_q", "attn_k", "attn_v", "attn_output", "attn_q_norm", "attn_k_norm", "attn_gate"};
    static constexpr TensorNames gguf_audio = {
        "blk.", "output_norm", "ln1", "ln2", "attn_post_norm", "ffn_post_norm",
        "ffn_gate", "ffn_down", "ffn_up", "conv_pw1", "conv_pw2", "conv_dw",
        "attn_q", "attn_k", "attn_v", "attn_out", "attn_q_norm", "attn_k_norm", "attn_gate"};
    auto names = (layout == Lfm3StackLayout::Hf || layout == Lfm3StackLayout::HfEncoder) ? hf : layout == Lfm3StackLayout::GgufBackbone ? gguf_backbone : gguf_audio;
    if (layout == Lfm3StackLayout::HfEncoder) {
        names.o = "self_attn.out_proj";
        names.q_norm = "self_attn.q_layernorm";
        names.k_norm = "self_attn.k_layernorm";
    }
    if (layout == Lfm3StackLayout::GgufEncoder) names.final_norm = "post_ln";

    modules::LfmDecoderWeights out;
    for (size_t layer = 0; layer < config.layers.size(); ++layer) {
        const std::string p = prefix + names.layers + std::to_string(layer) + ".";
        modules::LfmDecoderLayerWeights w;
        w.operator_norm = modules::binding::norm_weight_from_source(stores.plain(), source, p + names.operator_norm, d);
        w.ffn_norm = modules::binding::norm_weight_from_source(stores.plain(), source, p + names.ffn_norm, d);
        if (config.sandwich_norm) {
            w.post_operator_norm = modules::binding::norm_weight_from_source(stores.plain(), source, p + names.post_operator_norm, d);
            w.post_ffn_norm = modules::binding::norm_weight_from_source(stores.plain(), source, p + names.post_ffn_norm, d);
        }
        w.feed_forward.gate_proj = modules::binding::linear_from_source(stores, source, p + names.w1, assets::TensorStorageType::Native, config.intermediate_size, d, false);
        w.feed_forward.up_proj = modules::binding::linear_from_source(stores, source, p + names.w3, assets::TensorStorageType::Native, config.intermediate_size, d, false);
        w.feed_forward.down_proj = modules::binding::linear_from_source(stores, source, p + names.w2, assets::TensorStorageType::Native, d, config.intermediate_size, false);

        if (config.layers[layer] == LayerKind::ShortConv) {
            const int64_t k = config.conv_kernel_size;
            w.conv.in_proj = modules::binding::linear_from_source(stores, source, p + names.conv_in, assets::TensorStorageType::Native, 3 * d, d, false);
            w.conv.out_proj = modules::binding::linear_from_source(stores, source, p + names.conv_out, assets::TensorStorageType::Native, d, d, false);
            const auto kernel_shape = (layout == Lfm3StackLayout::Hf || layout == Lfm3StackLayout::HfEncoder) ? std::vector<int64_t>{d, 1, k} : std::vector<int64_t>{d, k};
            const auto kernel = source.require_f32(p + names.conv_kernel + ".weight", kernel_shape);
            w.conv.kernel = stores.plain().make_f32(TensorShape::from_dims({d, 1, k}), kernel);
        } else {
            const int64_t q_rows = config.num_attention_heads * hd;
            const int64_t kv_rows = config.num_key_value_heads * hd;
            w.attention.q_proj = modules::binding::linear_from_source(stores, source, p + names.q, assets::TensorStorageType::Native, q_rows, d, false);
            w.attention.k_proj = modules::binding::linear_from_source(stores, source, p + names.k, assets::TensorStorageType::Native, kv_rows, d, false);
            w.attention.v_proj = modules::binding::linear_from_source(stores, source, p + names.v, assets::TensorStorageType::Native, kv_rows, d, false);
            w.attention.out_proj = modules::binding::linear_from_source(stores, source, p + names.o, assets::TensorStorageType::Native, d, q_rows, false);
            w.attention.q_norm = modules::binding::norm_weight_from_source(stores.plain(), source, p + names.q_norm, hd);
            w.attention.k_norm = modules::binding::norm_weight_from_source(stores.plain(), source, p + names.k_norm, hd);
            if (config.attention_gate) {
                w.attention.gate_proj = modules::binding::linear_from_source(stores, source, p + names.gate, assets::TensorStorageType::Native, config.num_attention_heads, d, false);
            }
        }
        out.layers.push_back(std::move(w));
    }

    out.final_norm = modules::binding::norm_weight_from_source(stores.plain(), source, prefix + names.final_norm, d);
    const auto rotated = config.full_rope.rotary_dims;
    if (rotated > 0 && rotated < hd) {
        std::vector<float> factors(static_cast<size_t>(hd / 2), 1e30f);
        std::fill_n(factors.begin(), rotated / 2, 1.0f);
        out.full_freq_factors = stores.plain().make_f32(TensorShape::from_dims({hd / 2}), std::move(factors));
    }
    return out;
}

Lfm3BackboneWeights load_lfm3_backbone_weights(const Lfm3AudioAssets & assets, core::ExecutionContext & execution,
    const Lfm3WeightOptions & options) {
    const bool gguf = !assets.published_backbone.empty();
    const auto & config = assets.backbone;
    const auto & source = *assets.model;
    const auto & frame_source = gguf ? *assets.audio_weights : source;
    const std::string token_name = gguf ? "token_embd.weight" : "lfm.model.embed_tokens.weight";
    const std::string frame_name = gguf ? "a.gen.code.out_embd.weight" : "codebook_embedding.weight";
    const int64_t d = config.stack.hidden_size;
    Lfm3BackboneWeights out;
    auto * backend = execution.backend();
    out.stores = std::make_unique<Lfm3Weights>(execution, "lfm3_audio.backbone.weights", 16ull * 1024ull * 1024ull, options);
    auto & plain = out.stores->plain();
    out.stack = load_lfm3_stack_weights(*out.stores, source, gguf ? "" : "lfm.model.", config.stack,
        gguf ? Lfm3StackLayout::GgufBackbone : Lfm3StackLayout::Hf);

    const auto storage = lfm3_storage_type(backend, stored_type(source, token_name), options);
    const auto table = plain.load_tensor(source, token_name, storage, {config.vocab_size, d});
    out.token_lookup = lookup_of(*out.stores, backend, source, token_name, table);
    // The tied head gets its own repacked copy where a CPU extra buffer supports it.
    const bool repack = storage == assets::TensorStorageType::Native &&
                        out.stores->matmul_buffer_type(table.tensor->type, config.vocab_size, d) != nullptr;
    out.text_head = repack ? out.stores->load_tensor(source, token_name, assets::TensorStorageType::Native, {config.vocab_size, d}) : table;

    const auto frames = plain.load_tensor(frame_source, frame_name, lfm3_storage_type(backend, stored_type(frame_source, frame_name), options),
        {assets.depth.codebooks * assets.depth.audio_vocab_size, d});
    out.frame_lookup = lookup_of(*out.stores, backend, frame_source, frame_name, frames);
    out.stores->upload();
    return out;
}

Lfm3EncoderWeights load_lfm3_encoder_weights(const Lfm3AudioAssets & assets, core::ExecutionContext & execution,
    const Lfm3WeightOptions & options) {
    const bool gguf = !assets.published_backbone.empty();
    const auto & config = assets.encoder;
    const auto & source = gguf ? *assets.audio_weights : *assets.model;
    static constexpr const char * hf_convs[] = {"encoder.downsample.conv_in", "encoder.downsample.conv_mid",
        "encoder.downsample.lin_mid", "encoder.downsample.conv_end", "encoder.downsample.lin_end"};
    static constexpr const char * gguf_convs[] = {"a.conv1d.0", "a.conv1d.2", "a.conv1d.3", "a.conv1d.5", "a.conv1d.6"};
    const auto & conv_names = gguf ? gguf_convs : hf_convs;
    const std::string projector = gguf ? "mm.a.mlp." : "mlp.model.";
    Lfm3EncoderWeights out;
    out.stores = std::make_unique<Lfm3Weights>(execution, "lfm3_audio.encoder.weights", 4ull * 1024ull * 1024ull, options);
    auto & store = out.stores->plain();
    const int64_t c = config.conv_channels;
    const int64_t d = config.decoder.hidden_size;
    const int64_t o = config.output_size;

    // Conv weights stay F32: Metal's depthwise convolution requires it.
    const auto conv = [&](size_t index, int64_t in, int64_t k) {
        const std::string p = conv_names[index];
        if (!gguf) return modules::binding::conv2d_from_source(store, source, p, assets::TensorStorageType::F32, c, in, k, k, true);
        auto weight = store.load_f32_tensor(source, p + ".weight", {c, in, k, k});
        auto bias = store.make_f32(TensorShape::from_dims({c}), source.require_f32(p + ".bias", {c, 1, 1}));
        return modules::Conv2dWeights{weight, bias};
    };
    const auto linear = [&](const std::string & name, int64_t in, int64_t rows) {
        return modules::binding::linear_from_source(*out.stores, source, name, assets::TensorStorageType::Native, rows, in, true);
    };
    out.tensors.conv_in = conv(0, 1, 3);
    out.tensors.conv_mid = conv(1, 1, 3);
    out.tensors.lin_mid = conv(2, c, 1);
    out.tensors.conv_end = conv(3, 1, 3);
    out.tensors.lin_end = conv(4, c, 1);
    out.tensors.lin_out = linear(gguf ? "a.pre_encode.out" : "encoder.downsample.lin_out", c * (config.mel_bins / 8), d);
    out.tensors.decoder = load_lfm3_stack_weights(*out.stores, source, gguf ? "a." : "encoder.lfm.lfm.", config.decoder,
        gguf ? Lfm3StackLayout::GgufEncoder : Lfm3StackLayout::HfEncoder);
    out.tensors.projector_norm = modules::binding::norm_from_source(store, source, projector + "0", d);
    out.tensors.projector_1 = linear(projector + "1", d, o);
    out.tensors.projector_2 = linear(projector + "3", o, o);
    out.stores->upload();
    return out;
}

Lfm3DepthWeights load_lfm3_depth_weights(const Lfm3AudioAssets & assets, core::ExecutionContext & execution,
    const Lfm3WeightOptions & options) {
    const bool gguf = !assets.published_backbone.empty();
    const auto & config = assets.depth;
    const auto & source = gguf ? *assets.audio_weights : *assets.model;
    const std::string projection = gguf ? "a.gen.code.proj_in" : "depth_linear";
    const std::string table_name = gguf ? "a.gen.code.embd.weight" : "depth_embeddings.0.embedding.weight";
    const int64_t d = config.stack.hidden_size;
    const int64_t vocab = config.audio_vocab_size;
    const int64_t books = config.codebooks;
    auto * backend = execution.backend();
    Lfm3DepthWeights out;
    out.stores = std::make_unique<Lfm3Weights>(execution, "lfm3_audio.depth.weights", 1024 * ggml_tensor_overhead(), options);
    auto & store = out.stores->plain();
    out.input_weight = out.stores->load_tensor(source, projection + ".weight", assets::TensorStorageType::Native, {books * d, config.input_size});
    out.input_bias = store.load_f32_tensor(source, projection + ".bias", {books * d});

    const auto mask = source.require_f32(gguf ? "a.gen.code.mask_embd" : "mask_embedding", {d});
    std::vector<float> mask_rows;
    for (int64_t i = 0; i < books; ++i) mask_rows.insert(mask_rows.end(), mask.begin(), mask.end());
    out.mask_rows = store.make_f32(TensorShape::from_dims({books, d}), std::move(mask_rows));

    // These tables serve both embedding lookup and tied logits heads.
    const auto table_type = stored_type(source, table_name);
    auto table_storage = lfm3_storage_type(backend, table_type, options);
    if (table_storage == assets::TensorStorageType::Native && !lfm3_backend_gathers(backend, table_type)) {
        table_storage = assets::TensorStorageType::F16;
    }
    if (gguf) {
        if (table_storage == assets::TensorStorageType::Native && ggml_is_quantized(table_type)) {
            // Flatten without dequantizing and requantizing stored blocks.
            const auto tables = source.require_tensor(table_name, table_storage, {books, vocab, d});
            out.embeddings = store.make_tensor(TensorShape::from_dims({books * vocab, d}), tables.type, tables.bytes.data(), tables.bytes.size());
        } else {
            out.embeddings = store.load_tensor_as_shape(source, table_name, table_storage, {books, vocab, d}, TensorShape::from_dims({books * vocab, d}));
        }
        out.norms = store.load_f32_tensor(source, "a.gen.code.head_norm.weight", {books, d});
    } else {
        std::vector<core::BackendWeightStore::TensorPart> tables;
        std::vector<float> norms;
        for (int64_t i = 0; i < books; ++i) {
            const auto p = "depth_embeddings." + std::to_string(i) + ".";
            tables.push_back({p + "embedding.weight", {vocab, d}});
            const auto norm = source.require_f32(p + "embedding_norm.weight", {d});
            norms.insert(norms.end(), norm.begin(), norm.end());
        }
        out.embeddings = store.load_tensor(source, tables, table_storage);
        out.norms = store.make_f32(TensorShape::from_dims({books, d}), std::move(norms));
    }
    out.stack = load_lfm3_stack_weights(*out.stores, source, gguf ? "a.gen.code." : "depthformer.model.", config.stack,
        gguf ? Lfm3StackLayout::GgufAudio : Lfm3StackLayout::Hf);
    out.stores->upload();
    return out;
}

Lfm3CodecWeights load_lfm3_codec_weights(const Lfm3AudioAssets & assets, core::ExecutionContext & execution,
    const Lfm3WeightOptions & options) {
    const bool gguf = !assets.published_backbone.empty();
    const auto & config = assets.codec;
    const auto & source = *assets.codec_weights;
    const std::string project_out = gguf ? "a.gen.wav.quant.first.out_proj" : "quantizer.project_out";
    const std::string from_quantizer = gguf ? "a.gen.wav.tfm.in_proj" : "from_quantizer";
    const std::string output = gguf ? "a.gen.wav.tfm.out_proj" : "decoder.lin";
    const int64_t d = config.stack.hidden_size;
    const int64_t latent = config.codebooks * static_cast<int64_t>(config.fsq_levels.size());
    const int64_t q = config.quantizer_dim;
    Lfm3CodecWeights out;
    out.stores = std::make_unique<Lfm3Weights>(execution, "lfm3_audio.codec.weights", 1024 * ggml_tensor_overhead(), options);
    auto & plain = out.stores->plain();
    out.project_out = plain.load_f32_tensor(source, project_out + ".weight", {q, latent});
    out.project_out_bias = plain.load_f32_tensor(source, project_out + ".bias", {q});
    out.from_quantizer = out.stores->load_tensor(source, from_quantizer + ".weight", assets::TensorStorageType::Native, {d, q});
    out.from_quantizer_bias = plain.load_f32_tensor(source, from_quantizer + ".bias", {d});
    out.stack = load_lfm3_stack_weights(*out.stores, source, gguf ? "a.gen.wav.tfm." : "decoder.lfm.", config.stack,
        gguf ? Lfm3StackLayout::GgufAudio : Lfm3StackLayout::Hf);
    out.lin = out.stores->load_tensor(source, output + ".weight", assets::TensorStorageType::Native, {config.n_fft + 2, d});
    out.lin_bias = plain.load_f32_tensor(source, output + ".bias", {config.n_fft + 2});
    if (gguf) {
        const double pi = std::acos(-1.0);
        out.window.resize(static_cast<size_t>(config.n_fft));
        for (int64_t i = 0; i < config.n_fft; ++i) {
            out.window[static_cast<size_t>(i)] = static_cast<float>(0.5 - 0.5 * std::cos(2.0 * pi * static_cast<double>(i) / static_cast<double>(config.n_fft)));
        }
    } else {
        out.window = source.require_f32_tensor("decoder.istft.window", {config.n_fft}).values;
    }
    out.stores->upload();
    return out;
}

}  // namespace engine::community_models::lfm3_audio
