#pragma once

// Synthetic LFM2-Audio packages for the unit tests: GGUFs with the tensor
// names and layouts of Liquid's llama.cpp conversions, tiny shapes and
// generated weights.

#include "test_assert.h"

#include <ggml.h>
#include <gguf.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace lfm2_audio_test {

// Shapes are in PyTorch order, outermost first, as TensorSource reports them;
// values are row-major in that order.
struct Tensor {
    std::vector<int64_t> shape;
    std::vector<float> values;

    [[nodiscard]] float at(int64_t row, int64_t col) const { return values[static_cast<size_t>(row * shape.back() + col)]; }
};

using TensorMap = std::map<std::string, Tensor>;

// Fills a tensor from its name and element count.
using Fill = std::function<std::vector<float>(const std::string & name, size_t count)>;

// Portable pseudo-random values in [-scale, scale): the same on every
// platform, unlike the std distributions.
class Random {
public:
    explicit Random(uint64_t seed) : state_(seed) {}

    float uniform(float scale) {
        state_ += 0x9e3779b97f4a7c15ull;
        uint64_t z = state_;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        z ^= z >> 31;
        return scale * (static_cast<float>(z >> 40) / static_cast<float>(1ull << 23) - 1.0f);
    }

    std::vector<float> uniform(size_t count, float scale) {
        std::vector<float> out(count);
        for (auto & value : out) {
            value = uniform(scale);
        }

        return out;
    }

private:
    uint64_t state_;
};

class GgufWriter {
public:
    GgufWriter() : ctx_(gguf_init_empty()) {}

    void set(const char * key, const std::string & value) { gguf_set_val_str(ctx_.get(), key, value.c_str()); }
    void set_u32(const char * key, uint32_t value) { gguf_set_val_u32(ctx_.get(), key, value); }
    void set_f32(const char * key, float value) { gguf_set_val_f32(ctx_.get(), key, value); }

    void set_i32_array(const char * key, const std::vector<int32_t> & values) {
        gguf_set_arr_data(ctx_.get(), key, GGUF_TYPE_INT32, values.data(), values.size());
    }

    void set_str_array(const char * key, const std::vector<std::string> & values) {
        std::vector<const char *> data;
        for (const auto & value : values) {
            data.push_back(value.c_str());
        }

        gguf_set_arr_str(ctx_.get(), key, data.data(), data.size());
    }

    void add(const std::string & name, const Tensor & tensor) { tensors_.emplace_back(name, tensor); }

    void add(const TensorMap & tensors) {
        for (const auto & [name, tensor] : tensors) {
            add(name, tensor);
        }
    }

    void write(const std::filesystem::path & path) {
        size_t bytes = 1024 * 1024;
        for (const auto & entry : tensors_) {
            bytes += entry.second.values.size() * sizeof(float) + ggml_tensor_overhead();
        }

        std::unique_ptr<ggml_context, void (*)(ggml_context *)> data_ctx(ggml_init({bytes, nullptr, false}), ggml_free);
        for (const auto & [name, tensor] : tensors_) {
            // ggml lists dimensions innermost first.
            const std::vector<int64_t> ne(tensor.shape.rbegin(), tensor.shape.rend());
            auto * t = ggml_new_tensor(data_ctx.get(), GGML_TYPE_F32, static_cast<int>(ne.size()), ne.data());
            engine::test::require(
                static_cast<size_t>(ggml_nelements(t)) == tensor.values.size(), "tensor " + name + " has the wrong size");
            ggml_set_name(t, name.c_str());
            std::copy(tensor.values.begin(), tensor.values.end(), static_cast<float *>(t->data));
            gguf_add_tensor(ctx_.get(), t);
        }

        engine::test::require(gguf_write_to_file(ctx_.get(), path.string().c_str(), false), "write " + path.string());
    }

private:
    struct Deleter {
        void operator()(gguf_context * ctx) const noexcept { gguf_free(ctx); }
    };

    std::unique_ptr<gguf_context, Deleter> ctx_;
    std::vector<std::pair<std::string, Tensor>> tensors_;
};

inline Tensor make_tensor(const std::string & name, std::vector<int64_t> shape, const Fill & fill) {
    size_t count = 1;
    for (const int64_t dim : shape) {
        count *= static_cast<size_t>(dim);
    }

    Tensor out{std::move(shape), fill(name, count)};
    engine::test::require(out.values.size() == count, "fill returned the wrong size for " + name);
    return out;
}

// Norm weights near one and everything else small and random.
inline Fill random_fill(uint64_t seed, float scale = 0.3f) {
    auto random = std::make_shared<Random>(seed);
    return [random, scale](const std::string & name, size_t count) {
        const bool is_norm_weight = name.find("norm") != std::string::npos && name.find(".weight") != std::string::npos;
        auto values = random->uniform(count, is_norm_weight ? 0.1f : scale);
        if (is_norm_weight) {
            for (auto & value : values) {
                value += 1.0f;
            }
        }

        return values;
    };
}

struct TextVocab {
    std::vector<std::string> tokens;
    std::vector<int32_t> types;
    std::vector<std::string> merges;
};

struct BackboneShape {
    int64_t hidden = 16;
    int64_t intermediate = 24;
    int64_t heads = 4;
    int64_t kernel = 3;
    int64_t context = 64;
    std::vector<int32_t> kv_heads = {0, 2, 0, 2};
    float rms_eps = 1e-5f;
    float rope_theta = 10000.0f;

    [[nodiscard]] int64_t head_dim() const { return hidden / heads; }
};

inline TensorMap backbone_tensors(const BackboneShape & shape, int64_t vocab_size, const Fill & fill) {
    const int64_t d = shape.hidden;
    const int64_t ff = shape.intermediate;
    const int64_t hd = shape.head_dim();
    TensorMap out;
    const auto add = [&](const std::string & name, std::vector<int64_t> dims) { out[name] = make_tensor(name, std::move(dims), fill); };
    add("token_embd.weight", {vocab_size, d});
    add("token_embd_norm.weight", {d});
    for (size_t layer = 0; layer < shape.kv_heads.size(); ++layer) {
        const std::string p = "blk." + std::to_string(layer) + ".";
        add(p + "attn_norm.weight", {d});
        add(p + "ffn_norm.weight", {d});
        add(p + "ffn_gate.weight", {ff, d});
        add(p + "ffn_up.weight", {ff, d});
        add(p + "ffn_down.weight", {d, ff});
        if (shape.kv_heads[layer] > 0) {
            const int64_t kv = shape.kv_heads[layer] * hd;
            add(p + "attn_q.weight", {shape.heads * hd, d});
            add(p + "attn_k.weight", {kv, d});
            add(p + "attn_v.weight", {kv, d});
            add(p + "attn_output.weight", {d, shape.heads * hd});
            add(p + "attn_q_norm.weight", {hd});
            add(p + "attn_k_norm.weight", {hd});
        } else {
            add(p + "shortconv.in_proj.weight", {3 * d, d});
            add(p + "shortconv.out_proj.weight", {d, d});
            add(p + "shortconv.conv.weight", {d, shape.kernel});
        }
    }

    return out;
}

inline void write_backbone(
    const std::filesystem::path & path,
    const BackboneShape & shape,
    const TextVocab & vocab,
    const TensorMap & tensors,
    const std::vector<std::string> & languages = {"en"}) {
    GgufWriter gguf;
    gguf.set("general.architecture", "lfm2");
    if (!languages.empty()) {
        gguf.set_str_array("general.languages", languages);
    }

    gguf.set_u32("lfm2.block_count", static_cast<uint32_t>(shape.kv_heads.size()));
    gguf.set_u32("lfm2.embedding_length", static_cast<uint32_t>(shape.hidden));
    gguf.set_u32("lfm2.feed_forward_length", static_cast<uint32_t>(shape.intermediate));
    gguf.set_u32("lfm2.attention.head_count", static_cast<uint32_t>(shape.heads));
    gguf.set_u32("lfm2.context_length", static_cast<uint32_t>(shape.context));
    gguf.set_u32("lfm2.shortconv.l_cache", static_cast<uint32_t>(shape.kernel));
    gguf.set_f32("lfm2.attention.layer_norm_rms_epsilon", shape.rms_eps);
    gguf.set_f32("lfm2.rope.freq_base", shape.rope_theta);
    gguf.set_i32_array("lfm2.attention.head_count_kv", shape.kv_heads);

    gguf.set("tokenizer.ggml.model", "gpt2");
    gguf.set("tokenizer.ggml.pre", "lfm2");
    gguf.set_str_array("tokenizer.ggml.tokens", vocab.tokens);
    gguf.set_str_array("tokenizer.ggml.merges", vocab.merges);
    gguf.set_i32_array("tokenizer.ggml.token_type", vocab.types);

    gguf.add(tensors);
    gguf.write(path);
}

struct EncoderShape {
    int64_t n_mels = 128;
    int64_t hidden = 16;
    int64_t layers = 1;
    int64_t heads = 2;
    int64_t intermediate = 32;
    int64_t kernel = 9;
    int64_t channels = 4;
    int64_t adapter_hidden = 24;
    int64_t output = 16;
};

inline TensorMap encoder_tensors(const EncoderShape & shape, const Fill & fill) {
    const int64_t d = shape.hidden;
    const int64_t c = shape.channels;
    TensorMap out;
    const auto add = [&](const std::string & name, std::vector<int64_t> dims) { out[name] = make_tensor(name, std::move(dims), fill); };
    const auto linear = [&](const std::string & name, int64_t out_features, int64_t in_features) {
        add(name + ".weight", {out_features, in_features});
        add(name + ".bias", {out_features});
    };
    const auto norm = [&](const std::string & name, int64_t features) {
        add(name + ".weight", {features});
        add(name + ".bias", {features});
    };
    // The converter keeps the conv biases as [C, 1, 1].
    add("a.conv1d.0.weight", {c, 1, 3, 3});
    add("a.conv1d.0.bias", {c, 1, 1});
    for (const char * depthwise : {"a.conv1d.2", "a.conv1d.5"}) {
        add(std::string(depthwise) + ".weight", {c, 1, 3, 3});
        add(std::string(depthwise) + ".bias", {c, 1, 1});
    }

    for (const char * pointwise : {"a.conv1d.3", "a.conv1d.6"}) {
        add(std::string(pointwise) + ".weight", {c, c, 1, 1});
        add(std::string(pointwise) + ".bias", {c, 1, 1});
    }

    linear("a.pre_encode.out", d, c * (shape.n_mels / 8));
    for (int64_t layer = 0; layer < shape.layers; ++layer) {
        const std::string p = "a.blk." + std::to_string(layer);
        norm(p + ".ffn_norm", d);
        linear(p + ".ffn_up", shape.intermediate, d);
        linear(p + ".ffn_down", d, shape.intermediate);
        norm(p + ".ln1", d);
        for (const char * projection : {".attn_q", ".attn_k", ".attn_v", ".attn_out"}) {
            linear(p + projection, d, d);
        }

        add(p + ".linear_pos.weight", {d, d});
        add(p + ".pos_bias_u", {shape.heads, d / shape.heads});
        add(p + ".pos_bias_v", {shape.heads, d / shape.heads});
        norm(p + ".norm_conv", d);
        linear(p + ".conv_pw1", 2 * d, d);
        linear(p + ".conv_pw2", d, d);
        add(p + ".conv_dw.weight", {d, shape.kernel});
        add(p + ".conv_dw.bias", {d});
        norm(p + ".conv_norm", d);
        norm(p + ".ffn_norm_1", d);
        linear(p + ".ffn_up_1", shape.intermediate, d);
        linear(p + ".ffn_down_1", d, shape.intermediate);
        norm(p + ".ln2", d);
    }

    norm("mm.a.mlp.0", d);
    linear("mm.a.mlp.1", shape.adapter_hidden, d);
    linear("mm.a.mlp.3", shape.output, shape.adapter_hidden);
    return out;
}

inline void write_mmproj(
    const std::filesystem::path & path,
    const EncoderShape & shape,
    const TensorMap & tensors,
    const std::string & projector = "lfm2a") {
    GgufWriter gguf;
    gguf.set("general.architecture", "clip");
    gguf.set("clip.projector_type", projector);
    gguf.set_u32("clip.audio.num_mel_bins", static_cast<uint32_t>(shape.n_mels));
    gguf.set_u32("clip.audio.embedding_length", static_cast<uint32_t>(shape.hidden));
    gguf.set_u32("clip.audio.block_count", static_cast<uint32_t>(shape.layers));
    gguf.set_u32("clip.audio.attention.head_count", static_cast<uint32_t>(shape.heads));
    gguf.set_f32("clip.audio.attention.layer_norm_epsilon", 1e-5f);

    gguf.add(tensors);
    gguf.write(path);
}

inline std::filesystem::path fresh_directory(const std::string & name) {
    const auto root = std::filesystem::temp_directory_path() / name;
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
    return root;
}

template <typename Fn>
void require_throws(Fn && fn, const std::string & label) {
    bool threw = false;
    try {
        fn();
    } catch (const std::exception &) {
        threw = true;
    }

    engine::test::require(threw, label + " must throw");
}

// Like require_throws, and the message must mention `needle`.
template <typename Fn>
void require_throws_with(Fn && fn, const std::string & needle, const std::string & label) {
    try {
        fn();
    } catch (const std::exception & error) {
        const std::string message = error.what();
        engine::test::require(
            message.find(needle) != std::string::npos, label + " threw \"" + message + "\", expected \"" + needle + "\"");
        return;
    }

    engine::test::require(false, label + " must throw");
}

}  // namespace lfm2_audio_test
