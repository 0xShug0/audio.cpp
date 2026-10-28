// Assets, weight storage and debug dumps.
//
// - Weight storage policy on the CPU: lfm3_audio.weight_type f32/f16 force
//   the type, and native keeps the stored one except BF16 on arm64, which
//   becomes F16.
// - load_lfm3_audio_assets on the synthetic package: every config value,
//   including the codec dims derived from tensor shapes, and clear errors for
//   a wrong directory.
// - The same package written in the layout of Liquid's published GGUFs: the
//   same configs, tensors and tokenizer, the speech sampling metadata, which
//   backbone and mmproj a path picks, and the loader claiming the directory.
// - npy dumps read back with lfm3_read_npy_*, and MaskGIT order injection.
#include "engine/community_models/lfm3_audio/assets.h"
#include "engine/community_models/lfm3_audio/debug.h"
#include "engine/community_models/lfm3_audio/published_gguf.h"
#include "engine/community_models/lfm3_audio/session.h"
#include "engine/community_models/lfm3_audio/tokenizer.h"
#include "engine/community_models/lfm3_audio/weights.h"
#include "engine/framework/core/backend.h"
#include "lfm3_audio_test_package.h"
#include "test_assert.h"

#include <ggml-cpp.h>
#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

namespace lfm3 = engine::community_models::lfm3_audio;
using engine::assets::TensorStorageType;
using engine::test::require;
using engine::test::require_eq;
using lfm3::blocks::LayerKind;

using lfm3_audio_test::set_env;

std::string kinds(const std::vector<LayerKind> & layers) {
    std::string out;
    for (const auto kind : layers) out += kind == LayerKind::Conv ? 'C' : (kind == LayerKind::FullAttention ? 'F' : 'S');
    return out;
}

std::string describe(const lfm3::blocks::StackConfig & s) {
    std::ostringstream out;
    out << "hidden=" << s.hidden << " ffn=" << s.ffn << " heads=" << s.heads << "/" << s.kv_heads << " hd=" << s.head_dim
        << " conv_k=" << s.conv_kernel << " window=" << s.sliding_window << " eps=" << s.eps << " layers=" << kinds(s.layers)
        << " sandwich=" << s.sandwich_norm << " gate=" << s.output_gate << " causal=" << s.causal << " theta=" << s.full_theta
        << "/" << s.sliding_theta << " rotary=" << s.full_rotary_dims << " names=" << (s.names == lfm3::blocks::Names::Lfm2 ? "lfm2" : "lfm3");
    return out.str();
}

// Every config value, one stack per line; equal strings = identical configs.
std::string describe(const lfm3::Lfm3AudioAssets & a) {
    std::ostringstream out;
    out << "backbone: " << describe(a.backbone.stack) << " vocab=" << a.backbone.vocab_size << '\n';
    out << "encoder:  " << describe(a.encoder.stack) << " mels=" << a.encoder.n_mels << " conv=" << a.encoder.conv_channels
        << " mel_ds=" << a.encoder.mel_after_downsample << " out=" << a.encoder.output_size << '\n';
    out << "depth:    " << describe(a.depth.stack) << " in=" << a.depth.input_size << " codebooks=" << a.depth.codebooks
        << " audio_vocab=" << a.depth.audio_vocab_size << " schedule=";
    for (const auto n : a.depth.maskgit_schedule) out << n << ',';
    out << '\n' << "codec:    " << describe(a.codec.stack) << " fsq=";
    for (const auto n : a.codec.fsq_levels) out << n << ',';
    out << " codebooks=" << a.codec.codebooks << " qdim=" << a.codec.quantizer_dim << " upsample=" << a.codec.upsample
        << " n_fft=" << a.codec.n_fft << " hop=" << a.codec.hop << " sr=" << a.codec.sample_rate << '\n';
    out << "generation: text=" << a.generation.interleaved_text << " audio=" << a.generation.interleaved_audio << " eos=";
    for (const auto id : a.generation.specials.eos_ids) out << id << ',';
    out << '\n';
    return out.str();
}

void test_storage_types() {
    const ggml_backend_ptr cpu(engine::core::init_backend({engine::core::BackendType::Cpu, 0, 1}));
    const lfm3::Lfm3WeightOptions native;
    const lfm3::Lfm3WeightOptions f16{lfm3::Lfm3WeightType::F16, true};
    const lfm3::Lfm3WeightOptions f32{lfm3::Lfm3WeightType::F32, true};
    for (const ggml_type stored : {GGML_TYPE_BF16, GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_Q8_0}) {
        require(lfm3::lfm3_storage_type(cpu.get(), stored, f32) == TensorStorageType::F32, "f32 forces F32");
        require(lfm3::lfm3_storage_type(cpu.get(), stored, f16) == TensorStorageType::F16, "f16 forces F16");
    }
    for (const ggml_type stored : {GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_Q8_0, GGML_TYPE_Q4_K}) {
        require(lfm3::lfm3_storage_type(cpu.get(), stored, native) == TensorStorageType::Native, "native keeps non-BF16 types");
    }

    // ggml's CPU BF16 dot product is scalar on arm64.
#if defined(__aarch64__) || defined(_M_ARM64)
    const auto cpu_bf16 = TensorStorageType::F16;
#else
    const auto cpu_bf16 = TensorStorageType::Native;
#endif
    require(lfm3::lfm3_storage_type(cpu.get(), GGML_TYPE_BF16, native) == cpu_bf16, "native BF16 on the CPU");
    require(lfm3::lfm3_reference_precision(f32), "f32 is reference precision");
    require(!lfm3::lfm3_reference_precision(native) && !lfm3::lfm3_reference_precision(f16), "only f32 is reference precision");
}

void test_tiny_package() {
    const auto package = lfm3_audio_test::write_package("lfm3_audio_assets_test");
    const auto assets = lfm3::load_lfm3_audio_assets(package.root);

    const auto & s = package.shape;
    const auto & b = assets->backbone.stack;
    require_eq(kinds(b.layers), std::string("CFCS"), "backbone layer kinds");
    require(b.hidden == 128 && b.ffn == 192 && b.heads == 2 && b.kv_heads == 1 && b.head_dim == 64 && b.conv_kernel == 7,
            "backbone dims");
    require(b.sliding_window == 4 && b.sandwich_norm && b.output_gate && b.causal && b.names == lfm3::blocks::Names::Lfm3,
            "backbone flags");
    require(b.full_theta == 1e6f && b.sliding_theta == 1e4f && b.full_rotary_dims == 32, "backbone rope");
    require_eq(assets->backbone.vocab_size, s.vocab_size, "vocab size");

    const auto & e = assets->encoder;
    require_eq(kinds(e.stack.layers), std::string("CSS"), "encoder attention layers are sliding");
    require(e.stack.hidden == 128 && e.stack.heads == 2 && e.stack.kv_heads == 1 && e.stack.head_dim == 64 &&
                e.stack.conv_kernel == 3 && e.stack.sliding_window == 4,
            "encoder dims");
    require_eq(e.stack.ffn, int64_t{96}, "encoder ffn comes from the weights, not config.json");
    require(!e.stack.sandwich_norm && !e.stack.output_gate && e.stack.causal && e.stack.names == lfm3::blocks::Names::Lfm2 &&
                e.stack.full_theta == 1e6f && e.stack.sliding_theta == 1e6f && e.stack.full_rotary_dims == 0,
            "encoder flags");
    require(e.n_mels == 128 && e.conv_channels == 8 && e.mel_after_downsample == 16 && e.output_size == 128, "encoder io");

    const auto & d = assets->depth;
    require_eq(kinds(d.stack.layers), std::string("F"), "depth layers");
    require(d.stack.hidden == 64 && d.stack.heads == 2 && d.stack.kv_heads == 1 && d.stack.head_dim == 32 && d.stack.ffn == 96,
            "depth dims");
    require(!d.stack.causal && !d.stack.sandwich_norm && d.stack.output_gate && d.stack.full_rotary_dims == 0 &&
                d.stack.full_theta == 1e6f,
            "depth flags");
    require(d.input_size == 128 && d.codebooks == 12 && d.audio_vocab_size == 49 && d.end_of_audio() == 48, "depth vocab");
    require(d.maskgit_schedule == std::vector<int64_t>{4, 4, 4}, "maskgit schedule");

    const auto & c = assets->codec;
    require_eq(kinds(c.stack.layers), std::string("CSC"), "codec layer kinds from tensors");
    require(c.stack.hidden == 128 && c.stack.ffn == 160 && c.stack.heads == 2 && c.stack.kv_heads == 1 && c.stack.head_dim == 64 &&
                c.stack.conv_kernel == 7 && c.stack.sliding_window == 4,
            "codec dims from shapes");
    require(c.stack.sandwich_norm && c.stack.output_gate && c.stack.causal && c.stack.full_theta == 1e6f, "codec flags");
    require(c.fsq_levels == s.fsq_levels && c.codebooks == 12 && c.quantizer_dim == 32 && c.upsample == 6 && c.n_fft == 64 &&
                c.hop == 16 && c.sample_rate == 24000,
            "codec config");

    const auto & g = assets->generation;
    require(g.interleaved_text == 6 && g.interleaved_audio == 12, "interleaving");
    require(g.specials.eos_ids == std::vector<int32_t>{package.ids.at("<|im_end|>"), package.ids.at("<|text_end|>")},
            "eos ids from generation_config.json");

    require(assets->model->has_tensor("lfm.model.layers.1.self_attn.gate_proj.weight"), "model tensors use HF names");
    require(assets->codec_weights->has_tensor("decoder.istft.window"), "codec tensors use HF names");
    require(assets->codec_weights->require_metadata("decoder.istft.window").dtype == "F32", "the window stays F32");
    require_eq(assets->tokenizer_json.filename().string(), std::string("tokenizer.json"), "tokenizer path");
    require(std::filesystem::equivalent(assets->resources.model_root(), package.root), "resources root is the package");
    std::filesystem::remove_all(package.root);
}

// HF layer tensor -> llama.cpp name, for one stack ("" when `name` is not one of its layer tensors).
std::string llama_layer(const std::string & name, const std::string & hf_prefix, const std::string & prefix, bool backbone) {
    static const std::map<std::string, std::string> backbone_keys = {
        {"operator_norm", "attn_norm"}, {"ffn_norm", "ffn_norm"}, {"post_operator_norm", "post_attention_norm"},
        {"post_ffn_norm", "post_ffw_norm"}, {"feed_forward.w1", "ffn_gate"}, {"feed_forward.w3", "ffn_up"}, {"feed_forward.w2", "ffn_down"},
        {"conv.conv", "shortconv.conv"}, {"conv.in_proj", "shortconv.in_proj"}, {"conv.out_proj", "shortconv.out_proj"},
        {"self_attn.q_proj", "attn_q"}, {"self_attn.k_proj", "attn_k"}, {"self_attn.v_proj", "attn_v"}, {"self_attn.o_proj", "attn_output"},
        {"self_attn.q_norm", "attn_q_norm"}, {"self_attn.k_norm", "attn_k_norm"}, {"self_attn.gate_proj", "attn_gate"}};
    static const std::map<std::string, std::string> tower_keys = {
        {"operator_norm", "ln1"}, {"ffn_norm", "ln2"}, {"post_operator_norm", "attn_post_norm"}, {"post_ffn_norm", "ffn_post_norm"},
        {"feed_forward.w1", "ffn_gate"}, {"feed_forward.w3", "ffn_up"}, {"feed_forward.w2", "ffn_down"}, {"conv.conv", "conv_dw"},
        {"conv.in_proj", "conv_pw1"}, {"conv.out_proj", "conv_pw2"}, {"self_attn.q_proj", "attn_q"}, {"self_attn.k_proj", "attn_k"},
        {"self_attn.v_proj", "attn_v"}, {"self_attn.o_proj", "attn_out"}, {"self_attn.out_proj", "attn_out"},
        {"self_attn.q_norm", "attn_q_norm"}, {"self_attn.k_norm", "attn_k_norm"}, {"self_attn.q_layernorm", "attn_q_norm"},
        {"self_attn.k_layernorm", "attn_k_norm"}, {"self_attn.gate_proj", "attn_gate"}};
    const auto layers = hf_prefix + "layers.";
    if (name.rfind(layers, 0) != 0) return "";
    const auto rest = name.substr(layers.size());
    const auto dot = rest.find('.');
    const auto suffix = rest.rfind('.');
    const auto & keys = backbone ? backbone_keys : tower_keys;
    return prefix + rest.substr(0, dot) + "." + keys.at(rest.substr(dot + 1, suffix - dot - 1)) + rest.substr(suffix);
}

// The conv kernels lose their unit axis: [h, 1, k] -> [h, k].
std::vector<int64_t> llama_shape(const std::string & name, std::vector<int64_t> shape) {
    if (name.size() > 17 && name.compare(name.size() - 17, 17, ".conv.conv.weight") == 0) shape.erase(shape.begin() + 1);
    return shape;
}

// The synthetic package in the layout of Liquid's published GGUFs, as
// llama.cpp's conversion/lfm3.py writes it: llama.cpp names, ggml axis order,
// conv kernels without their unit axis, downsampler biases with two unit axes,
// the per-codebook depth tables stacked, and the configs as metadata.
void write_published(const lfm3_audio_test::Package & package, const std::filesystem::path & backbone_path,
                     const std::filesystem::path & mmproj_path, const std::map<std::string, uint32_t> & mmproj_overrides = {}) {
    const auto & s = package.shape;
    lfm2_audio_test::GgufWriter backbone;
    for (const auto & [name, t] : package.model) {
        if (name == "lfm.model.embed_tokens.weight") {
            backbone.add("token_embd.weight", t);
        } else if (name == "lfm.model.embedding_norm.weight") {
            backbone.add("output_norm.weight", t);
        } else if (const auto llama = llama_layer(name, "lfm.model.", "blk.", true); !llama.empty()) {
            backbone.add(llama, {llama_shape(name, t.shape), t.values});
        }
    }

    const auto rotated = static_cast<size_t>(s.partial_rotary_factor * static_cast<float>(s.backbone.head_dim)) / 2;
    std::vector<float> rope(static_cast<size_t>(s.backbone.head_dim / 2), 1e30f);
    std::fill(rope.begin(), rope.begin() + static_cast<std::ptrdiff_t>(rotated), 1.0f);
    backbone.add("rope_freqs.weight", {{s.backbone.head_dim / 2}, rope});

    std::vector<int32_t> kv;
    std::vector<int8_t> sliding;
    for (const auto & type : s.backbone.layer_types) {
        kv.push_back(type == "conv" ? 0 : static_cast<int32_t>(s.backbone.kv_heads));
        sliding.push_back(type == "sliding_attention" ? 1 : 0);
    }

    backbone.set("general.architecture", "lfm3");
    backbone.set_u32("lfm3.block_count", static_cast<uint32_t>(kv.size()));
    backbone.set_u32("lfm3.embedding_length", static_cast<uint32_t>(s.backbone.hidden));
    backbone.set_u32("lfm3.attention.head_count", static_cast<uint32_t>(s.backbone.heads));
    backbone.set_i32_array("lfm3.attention.head_count_kv", kv);
    backbone.set_u32("lfm3.attention.key_length", static_cast<uint32_t>(s.backbone.head_dim));
    backbone.set_bool_array("lfm3.attention.sliding_window_pattern", sliding);
    backbone.set_u32("lfm3.attention.sliding_window", static_cast<uint32_t>(s.sliding_window));
    backbone.set_u32("lfm3.shortconv.l_cache", static_cast<uint32_t>(s.backbone.conv_kernel));
    backbone.set_f32("lfm3.attention.layer_norm_rms_epsilon", 1e-5f);
    backbone.set_f32("lfm3.rope.freq_base", s.full_theta);
    backbone.set_f32("lfm3.rope.freq_base_swa", s.sliding_theta);

    // The vocabulary as llama.cpp writes it: control for special added tokens, user-defined for the others,
    // and unused [PADn] entries up to the head's rows.
    std::vector<std::string> tokens(static_cast<size_t>(s.vocab_size));
    std::vector<int32_t> types(tokens.size(), 5);
    for (size_t id = package.ids.size(); id < tokens.size(); ++id) tokens[id] = "[PAD" + std::to_string(id) + "]";
    for (const auto & [content, id] : package.ids) {
        tokens.at(static_cast<size_t>(id)) = content;
        types.at(static_cast<size_t>(id)) = 1;
    }
    for (const auto & [content, special] : lfm3_audio_test::special_tokens()) types.at(static_cast<size_t>(package.ids.at(content))) = special ? 3 : 4;
    // llama.cpp's converter marks the tool-call tags control too, though tokenizer.json has them not special.
    for (const char * tag : {"<|tool_call_start|>", "<|tool_call_end|>"}) types.at(static_cast<size_t>(package.ids.at(tag))) = 3;
    backbone.set("tokenizer.ggml.model", "gpt2");
    backbone.set("tokenizer.ggml.pre", "lfm2");
    backbone.set_str_array("tokenizer.ggml.tokens", tokens);
    backbone.set_str_array("tokenizer.ggml.merges", {"u s", "e r", "us er", "Ġ t", "h e", "Ġt he"});
    backbone.set_i32_array("tokenizer.ggml.token_type", types);
    backbone.set_u32("tokenizer.ggml.eos_token_id", static_cast<uint32_t>(package.ids.at("<|im_end|>")));
    backbone.write(backbone_path);

    lfm2_audio_test::GgufWriter mmproj;
    const std::map<std::string, std::string> downsample = {{"conv_in", "0"}, {"conv_mid", "2"}, {"lin_mid", "3"}, {"conv_end", "5"}, {"lin_end", "6"}};
    const std::map<std::string, std::string> fixed = {
        {"encoder.lfm.lfm.embedding_norm.weight", "a.post_ln.weight"}, {"depthformer.model.embedding_norm.weight", "a.gen.code.output_norm.weight"},
        {"depth_linear.weight", "a.gen.code.proj_in.weight"}, {"depth_linear.bias", "a.gen.code.proj_in.bias"},
        {"mask_embedding", "a.gen.code.mask_embd"}, {"codebook_embedding.weight", "a.gen.code.out_embd.weight"}};
    std::vector<float> embd, head_norm;
    for (int64_t i = 0; i < s.codebooks; ++i) {
        const auto p = "depth_embeddings." + std::to_string(i) + ".";
        const auto & e = package.model.at(p + "embedding.weight").values;
        const auto & n = package.model.at(p + "embedding_norm.weight").values;
        embd.insert(embd.end(), e.begin(), e.end());
        head_norm.insert(head_norm.end(), n.begin(), n.end());
    }

    mmproj.add("a.gen.code.embd.weight", {{s.codebooks, s.audio_vocab_size(), s.depth.hidden}, embd});
    mmproj.add("a.gen.code.head_norm.weight", {{s.codebooks, s.depth.hidden}, head_norm});
    for (const auto & [name, t] : package.model) {
        const std::string ds = "encoder.downsample.";
        if (name.rfind(ds, 0) == 0) {
            const auto rest = name.substr(ds.size());
            const auto dot = rest.find('.');
            const auto layer = rest.substr(0, dot);
            const auto suffix = rest.substr(dot + 1);
            if (layer == "lin_out") {
                mmproj.add("a.pre_encode.out." + suffix, {std::vector<int64_t>(t.shape.begin(), t.shape.begin() + (suffix == "weight" ? 2 : 1)), t.values});
            } else {
                mmproj.add("a.conv1d." + downsample.at(layer) + "." + suffix, {suffix == "bias" ? std::vector<int64_t>{t.shape[0], 1, 1} : t.shape, t.values});
            }
        } else if (const auto it = fixed.find(name); it != fixed.end()) {
            mmproj.add(it->second, t);
        } else if (name.rfind("mlp.model.", 0) == 0) {
            mmproj.add("mm.a.mlp." + name.substr(10), t);
        } else if (auto llama = llama_layer(name, "encoder.lfm.lfm.", "a.blk.", false); !llama.empty()) {
            mmproj.add(llama, {llama_shape(name, t.shape), t.values});
        } else if (auto depth = llama_layer(name, "depthformer.model.", "a.gen.code.blk.", false); !depth.empty()) {
            mmproj.add(depth, {llama_shape(name, t.shape), t.values});
        }
    }

    const std::map<std::string, std::string> codec_fixed = {
        {"quantizer.project_out.weight", "a.gen.wav.quant.first.out_proj.weight"}, {"quantizer.project_out.bias", "a.gen.wav.quant.first.out_proj.bias"},
        {"from_quantizer.weight", "a.gen.wav.tfm.in_proj.weight"}, {"from_quantizer.bias", "a.gen.wav.tfm.in_proj.bias"},
        {"decoder.lin.weight", "a.gen.wav.tfm.out_proj.weight"}, {"decoder.lin.bias", "a.gen.wav.tfm.out_proj.bias"},
        {"decoder.lfm.embedding_norm.weight", "a.gen.wav.tfm.output_norm.weight"}};
    for (const auto & [name, t] : package.codec) {
        if (const auto it = codec_fixed.find(name); it != codec_fixed.end()) {
            mmproj.add(it->second, t);
        } else if (auto llama = llama_layer(name, "decoder.lfm.", "a.gen.wav.tfm.blk.", false); !llama.empty()) {
            mmproj.add(llama, {llama_shape(name, t.shape), t.values});
        }
    }

    mmproj.set("general.architecture", "clip");
    mmproj.set("clip.audio.projector_type", "lfm3a");
    mmproj.set_bool("clip.has_audio_encoder", true);
    mmproj.set_u32("clip.audio.embedding_length", static_cast<uint32_t>(s.encoder.hidden));
    mmproj.set_u32("clip.audio.block_count", static_cast<uint32_t>(s.encoder.layer_types.size()));
    mmproj.set_u32("clip.audio.attention.head_count", static_cast<uint32_t>(s.encoder.heads));
    mmproj.set_u32("clip.audio.attention.head_count_kv", static_cast<uint32_t>(s.encoder.kv_heads));
    mmproj.set_f32("clip.audio.attention.layer_norm_epsilon", 1e-5f);
    mmproj.set_u32("clip.audio.num_mel_bins", static_cast<uint32_t>(s.n_mels));
    mmproj.set_u32("clip.audio.window_size", static_cast<uint32_t>(s.encoder_window));
    mmproj.set_bool("clip.has_gen_audio_encoder", true);
    mmproj.set_u32("clip.gen.audio.embedding_length", static_cast<uint32_t>(s.depth.hidden));
    mmproj.set_u32("clip.gen.audio.block_count", static_cast<uint32_t>(s.depth.layer_types.size()));
    mmproj.set_u32("clip.gen.audio.attention.head_count", static_cast<uint32_t>(s.depth.heads));
    mmproj.set_u32("clip.gen.audio.attention.head_count_kv", static_cast<uint32_t>(s.depth.kv_heads));
    mmproj.set_f32("clip.gen.audio.attention.layer_norm_epsilon", 1e-5f);
    mmproj.set_f32("clip.gen.audio.sampling.temp", 0.9f);
    mmproj.set_u32("clip.gen.audio.sampling.top_k", 7);
    mmproj.set_f32("clip.gen.audio.sampling.penalty_repeat", 1.25f);
    mmproj.set_u32("clip.gen.audio.sampling.penalty_last_n", 99);
    for (const auto & [key, value] : mmproj_overrides) mmproj.set_u32(key.c_str(), value);
    mmproj.write(mmproj_path);
}

// Shapes may differ by trailing unit axes (lin_out drops its Conv1d one).
void require_same_values(const engine::assets::TensorSource & source, const lfm3_audio_test::TensorMap & expected, float tolerance) {
    const auto squeezed = [](std::vector<int64_t> shape) {
        while (shape.size() > 1 && shape.back() == 1) shape.pop_back();
        return shape;
    };
    for (const auto & [name, tensor] : expected) {
        require(squeezed(source.require_metadata(name).shape) == squeezed(tensor.shape), name + " has the checkpoint's shape");
        const auto values = source.require_f32(name);
        float worst = 0.0f;
        for (size_t i = 0; i < values.size(); ++i) worst = std::max(worst, std::abs(values[i] - tensor.values[i]));
        require(worst <= tolerance, name + " from the published layout differs by " + std::to_string(worst));
    }
}

void test_published_package() {
    // The codec config is constant in llama.cpp's layout, so the codec takes the real one.
    lfm3_audio_test::PackageShape shape;
    shape.fsq_levels = {8, 7, 6, 6};
    shape.quantizer_dim = 256;
    shape.n_fft = 1280;
    shape.hop = 320;
    shape.codec_window = 30;
    shape.sliding_window = 256;  // the depth config repeats the backbone's, as the release's does
    const auto package = lfm3_audio_test::write_package("lfm3_audio_published_source", shape);
    const auto dir = lfm3_audio_test::fresh_directory("lfm3_audio_published");
    const auto backbone = dir / "LFM3-Audio-2B-F32.gguf";
    write_published(package, backbone, dir / "mmproj-LFM3-Audio-2B-F32.gguf");
    require(lfm3::is_lfm3_audio_published_gguf(dir) && lfm3::is_lfm3_audio_published_gguf(backbone) &&
                !lfm3::is_lfm3_audio_published_gguf(dir / "mmproj-LFM3-Audio-2B-F32.gguf") && !lfm3::is_lfm3_audio_published_gguf(package.root),
            "the published layout is recognized, and only it");
    engine::runtime::ModelLoadRequest load;
    load.model_path = dir;
    const auto loader = lfm3::make_lfm3_audio_loader();
    require(loader->can_load(load) && loader->inspect(load).metadata.family == "lfm3_audio", "the loader claims the published directory");

    const auto reference = lfm3::load_lfm3_audio_assets(package.root);
    const auto published = lfm3::load_lfm3_audio_assets(dir);
    require_eq(describe(*published), describe(*reference), "the published layout gives the checkpoint's configs");
    require_same_values(*published->model, package.model, 0.0f);
    require_same_values(*published->codec_weights, package.codec, 1e-6f);  // the ISTFT window is recomputed
    const auto & speech = published->generation.speech;
    require(speech.temperature == 0.9f && speech.top_k == 7 && speech.repetition_penalty == 1.25f && speech.repetition_window == 99,
            "speech sampling comes from clip.gen.audio.sampling.*");
    const auto & card = reference->generation.speech;
    require(card.temperature == 1.0f && card.top_k == 128 && card.repetition_penalty == 1.1f && card.repetition_window == 4096,
            "without audio_* in generation_config.json the model card's settings apply");

    const lfm3::Lfm3Tokenizer from_json(reference->tokenizer_json);
    const lfm3::Lfm3Tokenizer from_gguf(published->vocabulary);
    for (const std::string text : {"<|im_start|>user\nthe user<|audio|><|im_end|>", "<think>the</think> us er",
                                   "<|tool_call_start|>the<|tool_call_end|>", "\xc3\xa9t\xc3\xa9 ok"}) {
        const auto ids = from_json.encode(text);
        require(from_gguf.encode(text) == ids, "the GGUF vocabulary encodes " + text + " as tokenizer.json");
        require(from_gguf.decode(ids) == from_json.decode(ids) && from_gguf.decode(ids, false) == from_json.decode(ids, false),
                "the GGUF vocabulary decodes " + text + " as tokenizer.json");
    }

    // Two backbones: the directory is ambiguous; a 4-bit backbone takes the Q8_0 mmproj, as Liquid pairs them.
    std::filesystem::copy_file(backbone, dir / "LFM3-Audio-2B-Q4_0.gguf");
    std::filesystem::copy_file(dir / "mmproj-LFM3-Audio-2B-F32.gguf", dir / "mmproj-LFM3-Audio-2B-Q8_0.gguf");
    lfm3_audio_test::require_throws_with([&] { (void) lfm3::load_lfm3_audio_assets(dir); }, "several LFM3 backbone GGUFs", "two backbones");
    const auto q4 = lfm3::load_lfm3_audio_assets(dir / "LFM3-Audio-2B-Q4_0.gguf");
    require_eq(q4->codec_weights->source_path().filename().string(), std::string("mmproj-LFM3-Audio-2B-Q8_0.gguf"), "the 4-bit pairing");
    require_eq(lfm3::load_lfm3_audio_assets(backbone)->codec_weights->source_path().filename().string(),
               std::string("mmproj-LFM3-Audio-2B-F32.gguf"), "the same-name pairing");

    // Counts that size loops and allocations, or divide, fail with their key instead of hanging or crashing.
    for (const auto & [key, value] : std::map<std::string, uint32_t>{
             {"clip.audio.block_count", 1u << 30}, {"clip.gen.audio.block_count", 0}, {"clip.gen.audio.attention.head_count", 0}}) {
        write_published(package, backbone, dir / "mmproj-LFM3-Audio-2B-F32.gguf", {{key, value}});
        lfm3_audio_test::require_throws_with([&] { (void) lfm3::load_lfm3_audio_assets(backbone); }, key, key + " = " + std::to_string(value));
    }
    std::cout << "published layout: configs, " << package.model.size() + package.codec.size() << " tensors and the tokenizer match\n";
    std::filesystem::remove_all(dir);
    std::filesystem::remove_all(package.root);
}

void require_load_error(const lfm3_audio_test::Package & package, const std::string & file, const std::string & from,
                        const std::string & to, const std::string & needle, const std::function<void()> & while_edited = {}) {
    const auto path = package.root / file;
    std::ifstream in(path, std::ios::binary);
    const std::string original((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    in.close();
    const auto at = original.find(from);
    require(at != std::string::npos, file + " has no " + from);
    lfm3_audio_test::write_text(path, original.substr(0, at) + to + original.substr(at + from.size()));
    lfm3_audio_test::require_throws_with([&] { (void) lfm3::load_lfm3_audio_assets(package.root); }, needle, file + ": " + to);
    if (while_edited) while_edited();
    lfm3_audio_test::write_text(path, original);
}

void test_load_errors() {
    const auto package = lfm3_audio_test::write_package("lfm3_audio_assets_errors");
    engine::runtime::ModelLoadRequest load;
    load.model_path = package.root;
    const auto loader = lfm3::make_lfm3_audio_loader();
    require(loader->can_load(load), "the loader claims an LFM3-Audio directory");
    require_load_error(package, "config.json", "\"model_type\": \"lfm3_audio\"", "\"model_type\": \"lfm2_audio\"",
                       "lfm3_audio config.json", [&] { require(!loader->can_load(load), "the loader leaves another family's directory"); });
    require_load_error(package, "audio_tokenizer/config.json", "\"liquid_codec\"", "\"other_codec\"", "liquid_codec");
    require_load_error(package, "audio_tokenizer/config.json", "\"fsq_levels\": [4, 3, 2, 2]", "\"fsq_levels\": [4, 3, 2, 3]",
                       "FSQ levels");
    require_load_error(package, "config.json", "\"layer_types\": [\"conv\", \"full_attention\", \"conv\", \"sliding_attention\"]",
                       "\"layer_types\": [\"conv\", \"conv\", \"full_attention\", \"sliding_attention\"]",
                       "missing tensor: lfm.model.layers.1.conv.conv.weight");
    require_load_error(package, "config.json", "\"head_dim\": 64", "\"head_dim\": 32", "has shape");
    require_load_error(package, "config.json", "\"maskgit_steps\": 3", "\"maskgit_steps\": 5", "maskgit_steps");
    require_load_error(package, "config.json", "\"layer_types\": [\"full_attention\"]", "\"layer_types\": [\"mamba\"]",
                       "unsupported layer type");

    std::filesystem::remove(package.root / "generation_config.json");
    lfm3_audio_test::require_throws([&] { (void) lfm3::load_lfm3_audio_assets(package.root); }, "missing generation_config.json");
    std::filesystem::remove_all(package.root);
}

void test_npy_dumps() {
    const auto dir = lfm3_audio_test::fresh_directory("lfm3_audio_npy_test");
    set_env("LFM3_AUDIO_DUMP_DIR", "");
    require(!lfm3::lfm3_dump_enabled(), "dumps are off without LFM3_AUDIO_DUMP_DIR");
    const std::vector<float> f = {1.5f, -2.0f, 3.25f, 0.0f, 1e-8f, -7.0f};
    lfm3::lfm3_dump("off", f.data(), {6});
    require(!std::filesystem::exists(dir / "off.npy"), "nothing is written when dumps are off");

    set_env("LFM3_AUDIO_DUMP_DIR", dir.string());
    require(lfm3::lfm3_dump_enabled(), "dumps are on");
    lfm3::lfm3_dump("matrix", f.data(), {2, 3});
    std::vector<int64_t> shape;
    require(lfm3::lfm3_read_npy_f32(dir / "matrix.npy", &shape) == f && shape == std::vector<int64_t>{2, 3}, "f32 [2,3] round trip");
    lfm3_audio_test::require_throws_with([&] { (void) lfm3::lfm3_read_npy_i32(dir / "matrix.npy", nullptr); }, "<i4",
                                         "i32 read of an f32 file");

    set_env("LFM3_AUDIO_DUMP_FRAMES", "");
    require_eq(lfm3::lfm3_dump_frames(), int64_t{4}, "default dump frames");
    set_env("LFM3_AUDIO_DUMP_FRAMES", "9");
    require_eq(lfm3::lfm3_dump_frames(), int64_t{9}, "dump frames from the environment");
    set_env("LFM3_AUDIO_DUMP_FRAMES", "x");
    lfm3_audio_test::require_throws([] { (void) lfm3::lfm3_dump_frames(); }, "bad LFM3_AUDIO_DUMP_FRAMES");
    set_env("LFM3_AUDIO_DUMP_FRAMES", "");

    set_env("LFM3_AUDIO_MASKGIT_ORDERS", "");
    require(lfm3::lfm3_injected_orders().empty(), "no injected orders by default");
    std::vector<int32_t> orders = {11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 3, 1, 4, 0, 5, 9, 2, 6, 8, 7, 11, 10};
    lfm3::lfm3_dump("orders", orders.data(), {2, 12});
    set_env("LFM3_AUDIO_MASKGIT_ORDERS", (dir / "orders.npy").string());
    const auto injected = lfm3::lfm3_injected_orders();
    require(injected.size() == 2 && injected[0][0] == 11 && injected[1][0] == 3 && injected[1][11] == 10, "injected orders");
    orders[5] = 7;
    lfm3::lfm3_dump("orders", orders.data(), {2, 12});
    lfm3_audio_test::require_throws_with([] { (void) lfm3::lfm3_injected_orders(); }, "permutation", "orders with a repeat");
    lfm3::lfm3_dump("orders", orders.data(), {24});
    lfm3_audio_test::require_throws_with([] { (void) lfm3::lfm3_injected_orders(); }, "[frames, 12]", "orders of the wrong shape");

    set_env("LFM3_AUDIO_MASKGIT_ORDERS", "");
    set_env("LFM3_AUDIO_DUMP_DIR", "");
    std::cout << "npy dumps kept in " << dir.string() << '\n';  // for a numpy cross-check
}

}  // namespace

int main() {
    try {
        test_storage_types();
        test_tiny_package();
        test_published_package();
        test_load_errors();
        test_npy_dumps();
    } catch (const std::exception & e) {
        std::cerr << "lfm3_audio_assets_test failed: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
