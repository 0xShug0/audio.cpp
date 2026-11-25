#pragma once

// A synthetic LFM3-Audio package for the unit tests: the upstream directory
// layout (model.safetensors, audio_tokenizer/model.safetensors and the four
// sidecars) with HF tensor names, tiny shapes and generated weights. It loads
// through the spec's safetensors source exactly as the real checkpoint does.
// Head dims are 64 (backbone, encoder, codec) and 32 (depth), as upstream, so
// GPU runs take the same attention paths.

#include "../lfm2_audio/lfm2_audio_test_package.h"
#include "engine/framework/io/json.h"
#include "engine/framework/io/safetensors.h"

#include <ggml.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace lfm3_audio_test {

using lfm2_audio_test::Fill;
using lfm2_audio_test::fresh_directory;
using lfm2_audio_test::make_tensor;
using lfm2_audio_test::random_fill;
using lfm2_audio_test::require_throws;
using lfm2_audio_test::require_throws_with;
using lfm2_audio_test::Tensor;
using lfm2_audio_test::TensorMap;

struct StackShape {
    int64_t hidden = 0, ffn = 0, heads = 0, kv_heads = 0, head_dim = 0, conv_kernel = 0;
    std::vector<std::string> layer_types;  // "conv", "full_attention", "sliding_attention"
};

struct PackageShape {
    // Backbone: partial rotary (32 of 64 dims) on the full-attention layer.
    StackShape backbone{128, 192, 2, 1, 64, 7, {"conv", "full_attention", "conv", "sliding_attention"}};
    int64_t vocab_size = 320;  // > the tokenizer's size, like upstream (128000 > 125033)
    int64_t sliding_window = 4;
    float full_theta = 1e6f, sliding_theta = 1e4f, partial_rotary_factor = 0.5f;

    // Encoder (LFM2 names): head_dim = hidden / heads. config.json intermediate_size 144 auto-adjusts to
    // int(2 * 144 / 3) = 96, a multiple of 32, which is what the weights hold.
    StackShape encoder{128, 96, 2, 1, 64, 3, {"conv", "full_attention", "full_attention"}};
    int64_t encoder_config_ffn = 144, n_mels = 128, conv_channels = 8, encoder_window = 4;

    // Depth decoder: bidirectional, attention only, no sandwich norm.
    StackShape depth{64, 96, 2, 1, 32, 7, {"full_attention"}};
    int64_t codebooks = 12, maskgit_steps = 3;

    // Codec decoder: sandwich norm, every attention layer sliding.
    StackShape codec{128, 160, 2, 1, 64, 7, {"conv", "sliding_attention", "conv"}};
    std::vector<int64_t> fsq_levels = {4, 3, 2, 2};  // includes an odd level, like upstream [8, 7, 6, 6]
    int64_t quantizer_dim = 32, codec_window = 4, n_fft = 64, hop = 16, frame_stack = 6;

    [[nodiscard]] int64_t audio_vocab_size() const {
        int64_t codes = 1;
        for (const auto level : fsq_levels) codes *= level;
        return codes + 1;
    }
};

// lfm3 = false: the LFM2 layout (LFM2 names, no gate_proj), as the encoder uses.
inline void add_stack(TensorMap & out, const std::string & prefix, const StackShape & s, bool lfm3, bool sandwich,
                      const Fill & fill) {
    const auto add = [&](const std::string & name, std::vector<int64_t> dims) {
        out[prefix + name] = make_tensor(prefix + name, std::move(dims), fill);
    };
    const int64_t h = s.hidden, q = s.heads * s.head_dim, kv = s.kv_heads * s.head_dim;
    add("embedding_norm.weight", {h});
    for (size_t i = 0; i < s.layer_types.size(); ++i) {
        const auto p = "layers." + std::to_string(i) + ".";
        add(p + "operator_norm.weight", {h});
        add(p + "ffn_norm.weight", {h});
        if (sandwich) {
            add(p + "post_operator_norm.weight", {h});
            add(p + "post_ffn_norm.weight", {h});
        }

        add(p + "feed_forward.w1.weight", {s.ffn, h});
        add(p + "feed_forward.w3.weight", {s.ffn, h});
        add(p + "feed_forward.w2.weight", {h, s.ffn});
        if (s.layer_types[i] == "conv") {
            add(p + "conv.conv.weight", {h, 1, s.conv_kernel});
            add(p + "conv.in_proj.weight", {3 * h, h});
            add(p + "conv.out_proj.weight", {h, h});
            continue;
        }

        add(p + "self_attn.q_proj.weight", {q, h});
        add(p + "self_attn.k_proj.weight", {kv, h});
        add(p + "self_attn.v_proj.weight", {kv, h});
        add(p + (lfm3 ? "self_attn.o_proj.weight" : "self_attn.out_proj.weight"), {h, q});
        add(p + (lfm3 ? "self_attn.q_norm.weight" : "self_attn.q_layernorm.weight"), {s.head_dim});
        add(p + (lfm3 ? "self_attn.k_norm.weight" : "self_attn.k_layernorm.weight"), {s.head_dim});
        if (lfm3) add(p + "self_attn.gate_proj.weight", {s.heads, h});
    }
}

// model.safetensors: backbone, encoder + projector, frame embedding, depth decoder.
inline TensorMap model_tensors(const PackageShape & s, const Fill & fill) {
    TensorMap out;
    const auto add = [&](const std::string & name, std::vector<int64_t> dims) { out[name] = make_tensor(name, std::move(dims), fill); };
    const int64_t h = s.backbone.hidden, eh = s.encoder.hidden, c = s.conv_channels, dh = s.depth.hidden;
    const int64_t vocab = s.audio_vocab_size();
    add("lfm.model.embed_tokens.weight", {s.vocab_size, h});
    add_stack(out, "lfm.model.", s.backbone, true, true, fill);

    const int64_t mel = (((s.n_mels + 1) / 2 + 1) / 2 + 1) / 2;
    add("encoder.downsample.conv_in.weight", {c, 1, 3, 3});
    add("encoder.downsample.conv_in.bias", {c});
    for (const char * stage : {"mid", "end"}) {
        add(std::string("encoder.downsample.conv_") + stage + ".weight", {c, 1, 3, 3});
        add(std::string("encoder.downsample.conv_") + stage + ".bias", {c});
        add(std::string("encoder.downsample.lin_") + stage + ".weight", {c, c, 1, 1});
        add(std::string("encoder.downsample.lin_") + stage + ".bias", {c});
    }

    add("encoder.downsample.lin_out.weight", {eh, c * mel, 1});
    add("encoder.downsample.lin_out.bias", {eh});
    add_stack(out, "encoder.lfm.lfm.", s.encoder, false, false, fill);
    add("mlp.model.0.weight", {eh});
    add("mlp.model.0.bias", {eh});
    add("mlp.model.1.weight", {h, eh});
    add("mlp.model.1.bias", {h});
    add("mlp.model.3.weight", {h, h});
    add("mlp.model.3.bias", {h});

    add("codebook_embedding.weight", {s.codebooks * vocab, h});
    add("depth_linear.weight", {s.codebooks * dh, h});
    add("depth_linear.bias", {s.codebooks * dh});
    add("mask_embedding", {dh});
    add_stack(out, "depthformer.model.", s.depth, true, false, fill);
    for (int64_t i = 0; i < s.codebooks; ++i) {
        const auto p = "depth_embeddings." + std::to_string(i) + ".";
        add(p + "embedding.weight", {vocab, dh});
        add(p + "embedding_norm.weight", {dh});
        out[p + "to_logits.weight"] = out[p + "embedding.weight"];  // upstream ships an unused copy
    }

    return out;
}

// audio_tokenizer/model.safetensors: the decode half of LiquidCodec.
inline TensorMap codec_tensors(const PackageShape & s, const Fill & fill) {
    TensorMap out;
    const auto add = [&](const std::string & name, std::vector<int64_t> dims) { out[name] = make_tensor(name, std::move(dims), fill); };
    const int64_t h = s.codec.hidden;
    add("quantizer.project_out.weight", {s.quantizer_dim, s.codebooks * static_cast<int64_t>(s.fsq_levels.size())});
    add("quantizer.project_out.bias", {s.quantizer_dim});
    add("from_quantizer.weight", {h, s.quantizer_dim});
    add("from_quantizer.bias", {h});
    add_stack(out, "decoder.lfm.", s.codec, true, true, fill);
    add("decoder.lin.weight", {s.n_fft + 2, h});
    add("decoder.lin.bias", {s.n_fft + 2});
    Tensor window{{s.n_fft}, std::vector<float>(static_cast<size_t>(s.n_fft))};
    for (int64_t i = 0; i < s.n_fft; ++i) {
        window.values[static_cast<size_t>(i)] =
            static_cast<float>(0.5 - 0.5 * std::cos(2.0 * 3.14159265358979323846 * static_cast<double>(i) / static_cast<double>(s.n_fft)));
    }

    out["decoder.istft.window"] = window;
    return out;
}

// Rounds the BF16 tensors in place so `tensors` holds exactly what the file stores.
inline void write_safetensors(const std::filesystem::path & path, TensorMap & tensors) {
    std::vector<engine::io::SafeTensorWriteEntry> entries;
    for (auto & [name, tensor] : tensors) {
        const bool as_bf16 = name != "decoder.istft.window";
        engine::io::SafeTensorWriteEntry entry{name, as_bf16 ? "BF16" : "F32", tensor.shape, {}};
        entry.data.resize(tensor.values.size() * (as_bf16 ? 2 : 4));
        for (size_t i = 0; i < tensor.values.size(); ++i) {
            if (as_bf16) {
                const auto value = ggml_fp32_to_bf16(tensor.values[i]);
                tensor.values[i] = ggml_bf16_to_fp32(value);
                std::memcpy(entry.data.data() + 2 * i, &value, 2);
            } else {
                std::memcpy(entry.data.data() + 4 * i, &tensor.values[i], 4);
            }
        }

        entries.push_back(std::move(entry));
    }

    engine::io::write_safetensors_file(path, entries);
}

inline std::string quoted(const std::string & text) {
    return engine::io::json::stringify_string(text);
}

inline std::string string_list(const std::vector<std::string> & values) {
    std::string out = "[";
    for (size_t i = 0; i < values.size(); ++i) out += (i == 0 ? "" : ", ") + quoted(values[i]);
    return out + "]";
}

// Ids unrelated to the real ones (the code resolves every id by content);
// <think>, </think> and the tool-call tags are added but not special, as upstream.
inline const std::vector<std::pair<std::string, bool>> & special_tokens() {
    static const std::vector<std::pair<std::string, bool>> tokens = {
        {"<|audio_out|>", true}, {"<|audio|>", true},       {"<|mixed_end|>", true},   {"<|mixed_start|>", true},
        {"<|text_end|>", true},  {"<|text_start|>", true},  {"<|audio_start|>", true}, {"</think>", false},
        {"<think>", false},      {"<|im_end|>", true},      {"<|im_start|>", true},    {"<|endoftext|>", true},
        {"<|startoftext|>", true}, {"<|pad|>", true},
        {"<|tool_call_start|>", false}, {"<|tool_call_end|>", false},
    };
    return tokens;
}

struct TextTokenizer {
    std::string json;
    std::map<std::string, int32_t> ids;  // every vocab entry and special token by content
};

// A byte-level BPE tokenizer.json shaped like upstream's (same pre-tokenizer
// regex, ByteLevel decoder, ignore_merges): the 256 byte tokens, a few merges,
// then the special tokens as added tokens.
inline TextTokenizer byte_level_tokenizer() {
    TextTokenizer out;
    std::vector<std::string> vocab = lfm2_audio_test::byte_tokens();
    const std::vector<std::pair<std::string, std::string>> merges = {
        {"u", "s"}, {"e", "r"}, {"us", "er"}, {"Ġ", "t"}, {"h", "e"}, {"Ġt", "he"}};
    for (const auto & [left, right] : merges) vocab.push_back(left + right);
    std::ostringstream vocab_json, merges_json, added_json;
    for (size_t i = 0; i < vocab.size(); ++i) {
        out.ids[vocab[i]] = static_cast<int32_t>(i);
        vocab_json << (i == 0 ? "" : ", ") << quoted(vocab[i]) << ": " << i;
    }

    for (size_t i = 0; i < merges.size(); ++i) {
        merges_json << (i == 0 ? "" : ", ") << string_list({merges[i].first, merges[i].second});
    }

    int32_t next = static_cast<int32_t>(vocab.size());
    for (const auto & [content, special] : special_tokens()) {
        out.ids[content] = next;
        added_json << (next == static_cast<int32_t>(vocab.size()) ? "" : ", ") << "{\"id\": " << next
                   << ", \"content\": " << quoted(content)
                   << ", \"single_word\": false, \"lstrip\": false, \"rstrip\": false, \"normalized\": false, \"special\": "
                   << (special ? "true" : "false") << "}";
        ++next;
    }

    const std::string split =
        "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    out.json = "{\"version\": \"1.0\", \"truncation\": null, \"padding\": null, \"added_tokens\": [" + added_json.str() +
               "], \"normalizer\": null, \"pre_tokenizer\": {\"type\": \"Sequence\", \"pretokenizers\": ["
               "{\"type\": \"Split\", \"pattern\": {\"Regex\": " + quoted(split) +
               "}, \"behavior\": \"Isolated\", \"invert\": false}, "
               "{\"type\": \"ByteLevel\", \"add_prefix_space\": false, \"trim_offsets\": true, \"use_regex\": false}]}, "
               "\"post_processor\": {\"type\": \"ByteLevel\", \"add_prefix_space\": true, \"trim_offsets\": false, \"use_regex\": true}, "
               "\"decoder\": {\"type\": \"ByteLevel\", \"add_prefix_space\": true, \"trim_offsets\": true, \"use_regex\": true}, "
               "\"model\": {\"type\": \"BPE\", \"dropout\": null, \"unk_token\": null, \"continuing_subword_prefix\": null, "
               "\"end_of_word_suffix\": null, \"fuse_unk\": false, \"byte_fallback\": false, \"ignore_merges\": true, "
               "\"vocab\": {" + vocab_json.str() + "}, \"merges\": [" + merges_json.str() + "]}}";
    return out;
}

inline std::string stack_config_json(const StackShape & s) {
    std::ostringstream out;
    out << "\"hidden_size\": " << s.hidden << ", \"num_hidden_layers\": " << s.layer_types.size()
        << ", \"num_attention_heads\": " << s.heads << ", \"num_key_value_heads\": " << s.kv_heads
        << ", \"norm_eps\": 1e-05, \"conv_bias\": false, \"conv_L_cache\": " << s.conv_kernel
        << ", \"layer_types\": " << string_list(s.layer_types) << ", ";
    return out.str();
}

// config.json with upstream's keys (unused upstream keys left out).
inline std::string model_config_json(const PackageShape & s, const TextTokenizer & tokenizer) {
    std::ostringstream lfm, depth, encoder, out;
    lfm << "{" << stack_config_json(s.backbone) << "\"model_type\": \"lfm3\", \"vocab_size\": " << s.vocab_size
        << ", \"intermediate_size\": " << s.backbone.ffn << ", \"head_dim\": " << s.backbone.head_dim
        << ", \"sliding_window\": " << s.sliding_window << ", \"use_sandwich_norm\": true, \"tie_word_embeddings\": true"
        << ", \"pad_token_id\": " << tokenizer.ids.at("<|pad|>") << ", \"bos_token_id\": " << tokenizer.ids.at("<|startoftext|>")
        << ", \"eos_token_id\": " << tokenizer.ids.at("<|im_end|>")
        << ", \"rope_parameters\": {\"full_attention\": {\"partial_rotary_factor\": " << s.partial_rotary_factor
        << ", \"rope_theta\": " << s.full_theta << ", \"rope_type\": \"default\"}, \"sliding_attention\": {\"rope_theta\": "
        << s.sliding_theta << ", \"rope_type\": \"default\"}}}";
    depth << "{" << stack_config_json(s.depth) << "\"model_type\": \"lfm3\", \"vocab_size\": 1, \"intermediate_size\": "
          << s.depth.ffn << ", \"head_dim\": " << s.depth.head_dim
          << ", \"sliding_window\": 256, \"use_sandwich_norm\": false, \"is_causal\": false, \"tie_word_embeddings\": true"
          << ", \"rope_parameters\": {\"full_attention\": {\"rope_type\": \"default\", \"rope_theta\": " << s.full_theta
          << ", \"partial_rotary_factor\": 1.0}}}";
    encoder << "{\"dim_in\": " << s.n_mels << ", \"dim_conv\": " << s.conv_channels << ", \"dim_out\": " << s.encoder.hidden
            << ", \"sliding_window\": " << s.encoder_window << ", \"lfm\": {" << stack_config_json(s.encoder)
            << "\"model_type\": \"lfm2\", \"vocab_size\": 2, \"intermediate_size\": " << s.encoder_config_ffn
            << ", \"block_multiple_of\": 32, \"block_ffn_dim_multiplier\": 1.0, \"block_auto_adjust_ff_dim\": true"
            << ", \"full_attn_idxs\": null, \"tie_word_embeddings\": true"
            << ", \"rope_parameters\": {\"rope_theta\": " << s.full_theta << ", \"rope_type\": \"default\"}}}";
    out << "{\"architectures\": [\"Lfm3AudioForConditionalGeneration\"], \"model_type\": \"lfm3_audio\", \"format_version\": 1"
        << ", \"codebooks\": " << s.codebooks << ", \"audio_vocab_size\": " << s.audio_vocab_size()
        << ", \"maskgit_steps\": " << s.maskgit_steps << ", \"interleaved_n_text\": 6, \"interleaved_n_audio\": 12"
        << ", \"encoder\": " << encoder.str() << ", \"depthformer\": " << depth.str() << ", \"lfm\": " << lfm.str() << "}";
    return out.str();
}

inline std::string codec_config_json(const PackageShape & s) {
    std::ostringstream levels, out;
    for (size_t i = 0; i < s.fsq_levels.size(); ++i) levels << (i == 0 ? "" : ", ") << s.fsq_levels[i];
    out << "{\"architectures\": [\"LiquidCodecModel\"], \"model_type\": \"liquid_codec\", \"encoder_dim\": " << s.codec.hidden
        << ", \"stft_n_fft\": " << s.n_fft << ", \"stft_hop_length\": " << s.hop << ", \"stft_win_length\": " << s.n_fft
        << ", \"stft_frame_stack\": " << s.frame_stack << ", \"quantizer_dim\": " << s.quantizer_dim
        << ", \"fsq_levels\": [" << levels.str() << "], \"n_q\": " << s.codebooks << ", \"decoder_window\": " << s.codec_window
        << ", \"sampling_rate\": 24000, \"sample_rate\": 24000, \"frame_samples\": " << s.hop * s.frame_stack << "}";
    return out.str();
}

// An empty value unsets the variable.
inline void set_env(const char * name, const std::string & value) {
#ifdef _WIN32
    _putenv_s(name, value.c_str());
#else
    if (value.empty()) {
        unsetenv(name);
    } else {
        setenv(name, value.c_str(), 1);
    }
#endif
}

inline void write_text(const std::filesystem::path & path, const std::string & text) {
    std::ofstream out(path, std::ios::binary);
    out << text;
    engine::test::require(static_cast<bool>(out), "write " + path.string());
}

struct Package {
    std::filesystem::path root;
    PackageShape shape;
    TensorMap model, codec;              // exactly the stored values (BF16-rounded)
    std::map<std::string, int32_t> ids;  // tokenizer ids by content
};

// `edit` may change the tensors before they are written.
inline Package write_package(const std::string & name, const PackageShape & shape = {}, const std::function<void(Package &)> & edit = {}) {
    const auto tokenizer = byte_level_tokenizer();
    Package out{fresh_directory(name), shape, model_tensors(shape, random_fill(1)), codec_tensors(shape, random_fill(2)), tokenizer.ids};
    if (edit) edit(out);
    std::filesystem::create_directories(out.root / "audio_tokenizer");
    write_safetensors(out.root / "model.safetensors", out.model);
    write_safetensors(out.root / "audio_tokenizer" / "model.safetensors", out.codec);
    write_text(out.root / "tokenizer.json", tokenizer.json);
    write_text(out.root / "config.json", model_config_json(shape, tokenizer));
    write_text(out.root / "audio_tokenizer" / "config.json", codec_config_json(shape));
    write_text(out.root / "generation_config.json", "{\"eos_token_id\": [" + std::to_string(out.ids.at("<|im_end|>")) + ", " +
                                                        std::to_string(out.ids.at("<|text_end|>")) + "]}");
    return out;
}

}  // namespace lfm3_audio_test
