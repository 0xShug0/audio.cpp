#include "engine/community_models/lfm3_audio/published_gguf.h"

#include "engine/framework/io/filesystem.h"

#include <gguf.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace engine::community_models::lfm3_audio {
namespace {

namespace json = engine::io::json;

constexpr const char * kMmprojPrefix = "mmproj-";

class GgufMetadata {
public:
    explicit GgufMetadata(const std::filesystem::path & path) : path_(path) {
        gguf_init_params params{};
        params.no_alloc = true;
        params.ctx = nullptr;
        ctx_.reset(gguf_init_from_file(path.string().c_str(), params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("LFM3-Audio failed to read GGUF metadata: " + path.string());
        }
    }

    [[nodiscard]] std::optional<std::string> find_str(const char * key) const {
        const int64_t id = gguf_find_key(ctx_.get(), key);
        if (id < 0 || gguf_get_kv_type(ctx_.get(), id) != GGUF_TYPE_STRING) {
            return std::nullopt;
        }

        return std::string(gguf_get_val_str(ctx_.get(), id));
    }

    [[nodiscard]] bool find_bool(const char * key) const {
        const int64_t id = gguf_find_key(ctx_.get(), key);
        return id >= 0 && gguf_get_kv_type(ctx_.get(), id) == GGUF_TYPE_BOOL && gguf_get_val_bool(ctx_.get(), id);
    }

    [[nodiscard]] int64_t require_int(const char * key) const {
        const int64_t id = require_key(key);
        switch (gguf_get_kv_type(ctx_.get(), id)) {
            case GGUF_TYPE_UINT32: return static_cast<int64_t>(gguf_get_val_u32(ctx_.get(), id));
            case GGUF_TYPE_INT32: return static_cast<int64_t>(gguf_get_val_i32(ctx_.get(), id));
            case GGUF_TYPE_UINT64: return static_cast<int64_t>(gguf_get_val_u64(ctx_.get(), id));
            case GGUF_TYPE_INT64: return gguf_get_val_i64(ctx_.get(), id);
            default: throw std::runtime_error("LFM3-Audio GGUF key " + std::string(key) + " is not an integer in " + path_.string());
        }
    }

    // For values the reader divides by or sizes loops and allocations with.
    [[nodiscard]] int64_t require_count(const char * key, int64_t max) const {
        const int64_t value = require_int(key);
        if (value <= 0 || value > max) {
            throw std::runtime_error("LFM3-Audio GGUF key " + std::string(key) + " is " + std::to_string(value) + ", not in [1, " +
                                     std::to_string(max) + "], in " + path_.string());
        }

        return value;
    }

    [[nodiscard]] int64_t tensor_count() const { return gguf_get_n_tensors(ctx_.get()); }

    [[nodiscard]] float require_f32(const char * key) const {
        const int64_t id = require_key(key);
        if (gguf_get_kv_type(ctx_.get(), id) != GGUF_TYPE_FLOAT32) {
            throw std::runtime_error("LFM3-Audio GGUF key " + std::string(key) + " is not a float in " + path_.string());
        }

        return gguf_get_val_f32(ctx_.get(), id);
    }

    [[nodiscard]] std::optional<double> find_number(const char * key) const {
        if (gguf_find_key(ctx_.get(), key) < 0) {
            return std::nullopt;
        }

        const int64_t id = require_key(key);
        return gguf_get_kv_type(ctx_.get(), id) == GGUF_TYPE_FLOAT32 ? static_cast<double>(require_f32(key))
                                                                      : static_cast<double>(require_int(key));
    }

    [[nodiscard]] std::vector<int64_t> require_int_array(const char * key) const {
        const int64_t id = require_key(key);
        if (gguf_get_kv_type(ctx_.get(), id) != GGUF_TYPE_ARRAY) {
            throw std::runtime_error("LFM3-Audio GGUF key " + std::string(key) + " is not an array in " + path_.string());
        }

        const size_t count = gguf_get_arr_n(ctx_.get(), id);
        const void * data = gguf_get_arr_data(ctx_.get(), id);
        std::vector<int64_t> values(count);
        switch (gguf_get_arr_type(ctx_.get(), id)) {
            case GGUF_TYPE_INT32: std::copy_n(static_cast<const int32_t *>(data), count, values.begin()); break;
            case GGUF_TYPE_UINT32: std::copy_n(static_cast<const uint32_t *>(data), count, values.begin()); break;
            case GGUF_TYPE_BOOL: std::copy_n(static_cast<const int8_t *>(data), count, values.begin()); break;
            default: throw std::runtime_error("LFM3-Audio GGUF key " + std::string(key) + " is not an integer array in " + path_.string());
        }

        return values;
    }

    [[nodiscard]] std::vector<std::string> require_str_array(const char * key) const {
        const int64_t id = require_key(key);
        if (gguf_get_kv_type(ctx_.get(), id) != GGUF_TYPE_ARRAY || gguf_get_arr_type(ctx_.get(), id) != GGUF_TYPE_STRING) {
            throw std::runtime_error("LFM3-Audio GGUF key " + std::string(key) + " is not a string array in " + path_.string());
        }

        const size_t count = gguf_get_arr_n(ctx_.get(), id);
        std::vector<std::string> values;
        values.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            values.emplace_back(gguf_get_arr_str(ctx_.get(), id, i));
        }

        return values;
    }

private:
    int64_t require_key(const char * key) const {
        const int64_t id = gguf_find_key(ctx_.get(), key);
        if (id < 0) {
            throw std::runtime_error("LFM3-Audio GGUF is missing " + std::string(key) + ": " + path_.string());
        }

        return id;
    }

    struct Deleter {
        void operator()(gguf_context * ctx) const noexcept { gguf_free(ctx); }
    };

    std::filesystem::path path_;
    std::unique_ptr<gguf_context, Deleter> ctx_;
};

bool has_gguf_extension(const std::filesystem::path & path) {
    auto extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) { return std::tolower(c); });
    return extension == ".gguf";
}

bool is_mmproj_name(const std::filesystem::path & path) {
    return path.filename().string().rfind(kMmprojPrefix, 0) == 0;
}

bool is_backbone(const std::filesystem::path & path) {
    return has_gguf_extension(path) && !is_mmproj_name(path) && GgufMetadata(path).find_str("general.architecture") == "lfm3";
}

bool is_audio_mmproj(const std::filesystem::path & path) {
    return has_gguf_extension(path) && is_mmproj_name(path) && GgufMetadata(path).find_str("clip.audio.projector_type") == "lfm3a";
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

// One tensor of the HF checkpoint: a GGUF tensor (or one row of a stacked
// one) with the HF shape, which has the same row-major layout; or values the
// converter dropped because llama.cpp computes them.
struct Entry {
    std::shared_ptr<const assets::TensorSource> source;
    std::string gguf_name;
    assets::TensorMetadata metadata;
    int64_t slice = -1;
    std::vector<std::byte> values;
};

using Entries = std::map<std::string, Entry, std::less<>>;

class PublishedTensorSource final : public assets::TensorSource {
public:
    PublishedTensorSource(std::filesystem::path path, Entries entries) : path_(std::move(path)), entries_(std::move(entries)) {}

    const std::filesystem::path & source_path() const noexcept override { return path_; }
    bool has_tensor(std::string_view name) const noexcept override { return entries_.find(name) != entries_.end(); }
    assets::TensorMetadata require_metadata(std::string_view name) const override { return require(name).metadata; }

    std::vector<assets::TensorMetadata> tensors() const override {
        std::vector<assets::TensorMetadata> out;
        out.reserve(entries_.size());
        for (const auto & [_, entry] : entries_) {
            out.push_back(entry.metadata);
        }

        return out;
    }

    void release_storage() const override {
        std::set<const assets::TensorSource *> released;
        for (const auto & [_, entry] : entries_) {
            if (entry.source != nullptr && released.insert(entry.source.get()).second) {
                entry.source->release_storage();
            }
        }
    }

    assets::RawTensorData require_tensor_data(std::string_view name) const override {
        const auto & entry = require(name);
        if (entry.source == nullptr) {
            return {entry.metadata, entry.values};
        }

        auto raw = entry.source->require_tensor_data(entry.gguf_name);
        if (entry.slice >= 0) {
            const auto rows = raw.metadata.shape.front();
            if (rows <= entry.slice || raw.bytes.size() % static_cast<size_t>(rows) != 0) {
                throw std::runtime_error("LFM3-Audio GGUF tensor " + entry.gguf_name + " cannot be split into " + std::string(name));
            }

            const auto count = raw.bytes.size() / static_cast<size_t>(rows);
            const auto begin = raw.bytes.begin() + static_cast<std::ptrdiff_t>(count * static_cast<size_t>(entry.slice));
            raw.bytes = std::vector<std::byte>(begin, begin + static_cast<std::ptrdiff_t>(count));
        }

        raw.metadata = entry.metadata;
        return raw;
    }

    std::vector<float> require_f32(std::string_view name, const std::optional<std::vector<int64_t>> & expected_shape) const override {
        const auto & metadata = require(name).metadata;
        if (expected_shape && *expected_shape != metadata.shape) {
            throw std::runtime_error("tensor shape mismatch for " + std::string(name));
        }

        return assets::tensor_data_to_f32(name, require_tensor(name, assets::TensorStorageType::Native, metadata.shape));
    }

    std::optional<std::vector<float>> optional_f32(std::string_view name,
                                                   const std::optional<std::vector<int64_t>> & expected_shape) const override {
        if (!has_tensor(name)) {
            return std::nullopt;
        }

        return require_f32(name, expected_shape);
    }

    int64_t require_i64_scalar(std::string_view name) const override {
        throw std::runtime_error("LFM3-Audio GGUF has no integer scalar " + std::string(name));
    }

private:
    const Entry & require(std::string_view name) const {
        const auto it = entries_.find(name);
        if (it == entries_.end()) {
            throw std::runtime_error("missing tensor: " + std::string(name));
        }

        return it->second;
    }

    std::filesystem::path path_;
    Entries entries_;
};

// llama.cpp's per-layer names (gguf-py tensor_mapping) -> the HF ones.
struct LayerNames {
    std::map<std::string, std::string, std::less<>> names;
};

const LayerNames kBackboneLayer{{
    {"attn_norm", "operator_norm"}, {"ffn_norm", "ffn_norm"}, {"post_attention_norm", "post_operator_norm"},
    {"post_ffw_norm", "post_ffn_norm"}, {"ffn_gate", "feed_forward.w1"}, {"ffn_up", "feed_forward.w3"},
    {"ffn_down", "feed_forward.w2"}, {"shortconv.conv", "conv.conv"}, {"shortconv.in_proj", "conv.in_proj"},
    {"shortconv.out_proj", "conv.out_proj"}, {"attn_q", "self_attn.q_proj"}, {"attn_k", "self_attn.k_proj"},
    {"attn_v", "self_attn.v_proj"}, {"attn_output", "self_attn.o_proj"}, {"attn_q_norm", "self_attn.q_norm"},
    {"attn_k_norm", "self_attn.k_norm"}, {"attn_gate", "self_attn.gate_proj"},
}};

// The encoder keeps LFM2's names (out_proj, q_layernorm), the depth decoder
// and the codec LFM3's (o_proj, q_norm).
LayerNames tower_layer(bool lfm2) {
    return {{
        {"ln1", "operator_norm"}, {"ln2", "ffn_norm"}, {"attn_post_norm", "post_operator_norm"}, {"ffn_post_norm", "post_ffn_norm"},
        {"ffn_gate", "feed_forward.w1"}, {"ffn_up", "feed_forward.w3"}, {"ffn_down", "feed_forward.w2"}, {"conv_dw", "conv.conv"},
        {"conv_pw1", "conv.in_proj"}, {"conv_pw2", "conv.out_proj"}, {"attn_q", "self_attn.q_proj"}, {"attn_k", "self_attn.k_proj"},
        {"attn_v", "self_attn.v_proj"}, {"attn_out", lfm2 ? "self_attn.out_proj" : "self_attn.o_proj"},
        {"attn_q_norm", lfm2 ? "self_attn.q_layernorm" : "self_attn.q_norm"},
        {"attn_k_norm", lfm2 ? "self_attn.k_layernorm" : "self_attn.k_norm"}, {"attn_gate", "self_attn.gate_proj"},
    }};
}

bool starts_with(std::string_view text, std::string_view prefix) {
    return text.substr(0, prefix.size()) == prefix;
}

// "<prefix><n>.<key>.<suffix>" -> "<hf_prefix>layers.<n>.<hf key>.<suffix>", or "" when `name` is no such layer tensor.
std::string layer_name(std::string_view name, std::string_view prefix, const LayerNames & table, const std::string & hf_prefix) {
    if (!starts_with(name, prefix)) {
        return "";
    }

    const auto rest = name.substr(prefix.size());
    const auto dot = rest.find('.');
    const auto suffix = rest.rfind('.');
    if (dot == std::string_view::npos || suffix <= dot) {
        return "";
    }

    const auto key = table.names.find(rest.substr(dot + 1, suffix - dot - 1));
    if (key == table.names.end()) {
        return "";
    }

    return hf_prefix + "layers." + std::string(rest.substr(0, dot)) + "." + key->second + std::string(rest.substr(suffix));
}

class EntryBuilder {
public:
    explicit EntryBuilder(std::shared_ptr<const assets::TensorSource> source) : source_(std::move(source)) {}

    void add(Entries & out, const std::string & hf_name, const std::string & gguf_name, std::vector<int64_t> shape, int64_t slice = -1) const {
        auto metadata = source_->require_metadata(gguf_name);
        metadata.name = hf_name;
        metadata.shape = std::move(shape);
        out[hf_name] = Entry{source_, gguf_name, std::move(metadata), slice, {}};
    }

    // The HF shape of a GGUF tensor: the same, or with the conv kernels' channel axis back.
    void add_same(Entries & out, const std::string & hf_name, const std::string & gguf_name) const {
        auto shape = source_->require_metadata(gguf_name).shape;
        if (hf_name.size() >= 17 && hf_name.compare(hf_name.size() - 17, 17, ".conv.conv.weight") == 0 && shape.size() == 2) {
            shape.insert(shape.begin() + 1, 1);
        }

        add(out, hf_name, gguf_name, std::move(shape));
    }

    [[nodiscard]] const assets::TensorSource & source() const { return *source_; }

private:
    std::shared_ptr<const assets::TensorSource> source_;
};

Entries backbone_entries(const EntryBuilder & gguf) {
    Entries out;
    for (const auto & tensor : gguf.source().tensors()) {
        const auto & name = tensor.name;
        if (name == "token_embd.weight") {
            gguf.add_same(out, "lfm.model.embed_tokens.weight", name);
        } else if (name == "output_norm.weight") {
            gguf.add_same(out, "lfm.model.embedding_norm.weight", name);
        } else if (const auto hf = layer_name(name, "blk.", kBackboneLayer, "lfm.model."); !hf.empty()) {
            gguf.add_same(out, hf, name);
        }
    }

    return out;
}

Entries mmproj_entries(const EntryBuilder & gguf, int64_t codebooks) {
    // The downsampler's GGUF indices: the LFM2-Audio pre-encoder layout.
    const std::map<std::string, std::string, std::less<>> downsample = {
        {"0", "conv_in"}, {"2", "conv_mid"}, {"3", "lin_mid"}, {"5", "conv_end"}, {"6", "lin_end"}};
    const auto encoder = tower_layer(true);
    const auto depth = tower_layer(false);
    Entries out;
    for (const auto & tensor : gguf.source().tensors()) {
        const auto & name = tensor.name;
        const auto & shape = tensor.shape;
        if (starts_with(name, "a.conv1d.")) {
            const auto rest = name.substr(9);
            const auto dot = rest.find('.');
            const auto layer = downsample.find(rest.substr(0, dot));
            if (layer == downsample.end()) {
                continue;
            }

            const auto suffix = rest.substr(dot + 1);
            const auto hf = "encoder.downsample." + layer->second + "." + suffix;
            // The converter gave the biases two trailing unit axes for ggml's add.
            gguf.add(out, hf, name, suffix == "bias" ? std::vector<int64_t>{shape.front()} : shape);
        } else if (starts_with(name, "a.pre_encode.out.")) {
            gguf.add_same(out, "encoder.downsample.lin_out." + name.substr(17), name);
        } else if (name == "a.post_ln.weight") {
            gguf.add_same(out, "encoder.lfm.lfm.embedding_norm.weight", name);
        } else if (starts_with(name, "mm.a.mlp.")) {
            gguf.add_same(out, "mlp.model." + name.substr(9), name);
        } else if (name == "a.gen.code.output_norm.weight") {
            gguf.add_same(out, "depthformer.model.embedding_norm.weight", name);
        } else if (starts_with(name, "a.gen.code.proj_in.")) {
            gguf.add_same(out, "depth_linear." + name.substr(19), name);
        } else if (name == "a.gen.code.mask_embd") {
            gguf.add_same(out, "mask_embedding", name);
        } else if (name == "a.gen.code.out_embd.weight") {
            gguf.add_same(out, "codebook_embedding.weight", name);
        } else if (name == "a.gen.code.embd.weight" || name == "a.gen.code.head_norm.weight") {
            // One table per codebook, stacked; the embeddings are also the tied logits heads.
            if (shape.empty() || shape.front() != codebooks) {
                throw std::runtime_error("LFM3-Audio GGUF " + name + " does not stack " + std::to_string(codebooks) + " codebooks");
            }

            const std::vector<int64_t> row(shape.begin() + 1, shape.end());
            for (int64_t i = 0; i < codebooks; ++i) {
                const auto p = "depth_embeddings." + std::to_string(i) + ".";
                if (name == "a.gen.code.embd.weight") {
                    gguf.add(out, p + "embedding.weight", name, row, i);
                    gguf.add(out, p + "to_logits.weight", name, row, i);
                } else {
                    gguf.add(out, p + "embedding_norm.weight", name, row, i);
                }
            }
        } else if (auto hf = layer_name(name, "a.gen.code.blk.", depth, "depthformer.model."); !hf.empty()) {
            gguf.add_same(out, hf, name);
        } else if (auto hf_encoder = layer_name(name, "a.blk.", encoder, "encoder.lfm.lfm."); !hf_encoder.empty()) {
            gguf.add_same(out, hf_encoder, name);
        }
    }

    return out;
}

Entries codec_entries(const EntryBuilder & gguf, int64_t n_fft) {
    const std::map<std::string, std::string, std::less<>> fixed = {
        {"a.gen.wav.quant.first.out_proj.weight", "quantizer.project_out.weight"},
        {"a.gen.wav.quant.first.out_proj.bias", "quantizer.project_out.bias"},
        {"a.gen.wav.tfm.in_proj.weight", "from_quantizer.weight"},
        {"a.gen.wav.tfm.in_proj.bias", "from_quantizer.bias"},
        {"a.gen.wav.tfm.out_proj.weight", "decoder.lin.weight"},
        {"a.gen.wav.tfm.out_proj.bias", "decoder.lin.bias"},
        {"a.gen.wav.tfm.output_norm.weight", "decoder.lfm.embedding_norm.weight"},
    };
    const auto layer = tower_layer(false);
    Entries out;
    for (const auto & tensor : gguf.source().tensors()) {
        if (const auto it = fixed.find(tensor.name); it != fixed.end()) {
            gguf.add_same(out, it->second, tensor.name);
        } else if (const auto hf = layer_name(tensor.name, "a.gen.wav.tfm.blk.", layer, "decoder.lfm."); !hf.empty()) {
            gguf.add_same(out, hf, tensor.name);
        }
    }

    // The ISTFT window (torch.hann_window, periodic), which llama.cpp computes.
    const double pi = std::acos(-1.0);
    std::vector<std::byte> window(static_cast<size_t>(n_fft) * sizeof(float));
    for (int64_t i = 0; i < n_fft; ++i) {
        const auto value = static_cast<float>(0.5 - 0.5 * std::cos(2.0 * pi * static_cast<double>(i) / static_cast<double>(n_fft)));
        std::memcpy(window.data() + static_cast<size_t>(i) * sizeof(float), &value, sizeof(float));
    }

    out["decoder.istft.window"] = Entry{nullptr, "", {"decoder.istft.window", "F32", {n_fft}}, -1, std::move(window)};
    return out;
}

std::string number(double value) {
    return json::stringify_number(value);
}

std::string string_array(const std::vector<std::string> & values) {
    std::string out = "[";
    for (const auto & value : values) {
        out += (out.size() > 1 ? "," : "") + json::stringify_string(value);
    }

    return out + "]";
}

// The config.json of the checkpoint, from the metadata llama.cpp writes and
// the constants its LFM3-Audio graphs hard-code (the converter refuses a
// checkpoint that differs from them).
json::Value make_config(const GgufMetadata & model, const GgufMetadata & mmproj, const assets::TensorSource & backbone,
                        const assets::TensorSource & tower, int64_t vocab_size) {
    const auto kv_heads = model.require_int_array("lfm3.attention.head_count_kv");
    const auto sliding = model.require_int_array("lfm3.attention.sliding_window_pattern");
    if (kv_heads.size() != sliding.size() || kv_heads.size() != static_cast<size_t>(model.require_int("lfm3.block_count"))) {
        throw std::runtime_error("LFM3-Audio backbone GGUF layer metadata does not match its block count");
    }

    std::vector<std::string> layer_types;
    int64_t kv = 0;
    for (size_t i = 0; i < kv_heads.size(); ++i) {
        layer_types.push_back(kv_heads[i] == 0 ? "conv" : sliding[i] != 0 ? "sliding_attention" : "full_attention");
        kv = std::max(kv, kv_heads[i]);
    }

    // Partial RoPE as llama.cpp stores it: frequency factors 1 for the rotated pairs, 1e30 for the rest.
    const auto head_dim = model.require_int("lfm3.attention.key_length");
    const auto factors = backbone.optional_f32("rope_freqs.weight", {head_dim / 2});
    const auto rotated = factors ? std::count(factors->begin(), factors->end(), 1.0f) : head_dim / 2;
    const bool sandwich = backbone.has_tensor("blk.0.post_attention_norm.weight");

    // A layer has tensors, so no block count exceeds the file's tensors.
    const auto encoder_layers = mmproj.require_count("clip.audio.block_count", mmproj.tensor_count());
    std::vector<std::string> encoder_types;
    for (int64_t i = 0; i < encoder_layers; ++i) {
        encoder_types.push_back(tower.has_tensor("a.blk." + std::to_string(i) + ".attn_q.weight") ? "full_attention" : "conv");
    }

    std::vector<std::string> depth_types(static_cast<size_t>(mmproj.require_count("clip.gen.audio.block_count", mmproj.tensor_count())),
                                         "full_attention");
    const auto encoder_kernel = tower.require_metadata("a.blk.0.conv_dw.weight").shape.at(1);
    const auto & embd = tower.require_metadata("a.gen.code.embd.weight").shape;
    const auto depth_hidden = mmproj.require_int("clip.gen.audio.embedding_length");
    const auto depth_heads = mmproj.require_count("clip.gen.audio.attention.head_count", depth_hidden);

    return json::parse(
        "{\"model_type\":\"lfm3_audio\","
        "\"codebooks\":" + number(static_cast<double>(embd.at(0))) +
        ",\"audio_vocab_size\":" + number(static_cast<double>(embd.at(1))) +
        ",\"maskgit_steps\":3,\"interleaved_n_text\":6,\"interleaved_n_audio\":12"
        ",\"lfm\":{\"hidden_size\":" + number(static_cast<double>(model.require_int("lfm3.embedding_length"))) +
        ",\"num_attention_heads\":" + number(static_cast<double>(model.require_int("lfm3.attention.head_count"))) +
        ",\"num_key_value_heads\":" + number(static_cast<double>(kv)) +
        ",\"head_dim\":" + number(static_cast<double>(head_dim)) +
        ",\"conv_L_cache\":" + number(static_cast<double>(model.require_int("lfm3.shortconv.l_cache"))) +
        ",\"sliding_window\":" + number(static_cast<double>(model.require_int("lfm3.attention.sliding_window"))) +
        ",\"norm_eps\":" + number(model.require_f32("lfm3.attention.layer_norm_rms_epsilon")) +
        ",\"layer_types\":" + string_array(layer_types) +
        ",\"num_hidden_layers\":" + number(static_cast<double>(layer_types.size())) +
        ",\"use_sandwich_norm\":" + (sandwich ? "true" : "false") +
        ",\"is_causal\":true,\"vocab_size\":" + number(static_cast<double>(vocab_size)) +
        ",\"tie_word_embeddings\":" + (backbone.has_tensor("output.weight") ? "false" : "true") +
        ",\"rope_parameters\":{\"full_attention\":{\"rope_theta\":" + number(model.require_f32("lfm3.rope.freq_base")) +
        ",\"partial_rotary_factor\":" + number(2.0 * static_cast<double>(rotated) / static_cast<double>(head_dim)) +
        "},\"sliding_attention\":{\"rope_theta\":" + number(model.require_f32("lfm3.rope.freq_base_swa")) + "}}}"
        ",\"encoder\":{\"dim_in\":" + number(static_cast<double>(mmproj.require_int("clip.audio.num_mel_bins"))) +
        ",\"dim_conv\":" + number(static_cast<double>(tower.require_metadata("a.conv1d.0.weight").shape.at(0))) +
        ",\"dim_out\":" + number(static_cast<double>(mmproj.require_int("clip.audio.embedding_length"))) +
        ",\"sliding_window\":" + number(static_cast<double>(mmproj.require_int("clip.audio.window_size"))) +
        ",\"lfm\":{\"hidden_size\":" + number(static_cast<double>(mmproj.require_int("clip.audio.embedding_length"))) +
        ",\"num_attention_heads\":" + number(static_cast<double>(mmproj.require_int("clip.audio.attention.head_count"))) +
        ",\"num_key_value_heads\":" + number(static_cast<double>(mmproj.require_int("clip.audio.attention.head_count_kv"))) +
        ",\"conv_L_cache\":" + number(static_cast<double>(encoder_kernel)) +
        ",\"norm_eps\":" + number(mmproj.require_f32("clip.audio.attention.layer_norm_epsilon")) +
        ",\"layer_types\":" + string_array(encoder_types) +
        ",\"num_hidden_layers\":" + number(static_cast<double>(encoder_types.size())) +
        ",\"rope_parameters\":{\"rope_theta\":1000000}}}"
        ",\"depthformer\":{\"hidden_size\":" + number(static_cast<double>(depth_hidden)) +
        ",\"num_attention_heads\":" + number(static_cast<double>(depth_heads)) +
        ",\"num_key_value_heads\":" + number(static_cast<double>(mmproj.require_int("clip.gen.audio.attention.head_count_kv"))) +
        ",\"head_dim\":" + number(static_cast<double>(depth_hidden / depth_heads)) +
        // The attention-only depth decoder uses neither; the release's config repeats the backbone's LFM3 defaults.
        ",\"conv_L_cache\":" + number(static_cast<double>(model.require_int("lfm3.shortconv.l_cache"))) +
        ",\"sliding_window\":" + number(static_cast<double>(model.require_int("lfm3.attention.sliding_window"))) +
        ",\"is_causal\":false,\"use_sandwich_norm\":false"
        ",\"norm_eps\":" + number(mmproj.require_f32("clip.gen.audio.attention.layer_norm_epsilon")) +
        ",\"layer_types\":" + string_array(depth_types) +
        ",\"num_hidden_layers\":" + number(static_cast<double>(depth_types.size())) +
        ",\"rope_parameters\":{\"full_attention\":{\"rope_theta\":1000000}}}}");
}

json::Value make_generation_config(const GgufMetadata & model, const GgufMetadata & mmproj, const tokenizers::LlamaBpeGgufVocabulary & vocabulary) {
    // generation_config.json ends a turn on <|im_end|> (the GGUF's eos) and <|text_end|>.
    const auto text_end = std::find(vocabulary.tokens.begin(), vocabulary.tokens.end(), "<|text_end|>");
    if (text_end == vocabulary.tokens.end()) {
        throw std::runtime_error("LFM3-Audio backbone GGUF vocabulary has no <|text_end|>");
    }

    std::string sampling;
    const std::pair<const char *, const char *> keys[] = {
        {"audio_temperature", "clip.gen.audio.sampling.temp"},
        {"audio_top_k", "clip.gen.audio.sampling.top_k"},
        {"audio_repetition_penalty", "clip.gen.audio.sampling.penalty_repeat"},
        {"audio_repetition_window", "clip.gen.audio.sampling.penalty_last_n"},
    };
    for (const auto & [field, key] : keys) {
        if (const auto value = mmproj.find_number(key)) {
            sampling += ",\"" + std::string(field) + "\":" + number(*value);
        }
    }

    return json::parse("{\"eos_token_id\":[" + number(static_cast<double>(model.require_int("tokenizer.ggml.eos_token_id"))) + "," +
                       number(static_cast<double>(text_end - vocabulary.tokens.begin())) + "]" + sampling + "}");
}

// The LiquidCodec config llama.cpp's codec graph hard-codes.
json::Value make_codec_config() {
    return json::parse(
        "{\"model_type\":\"liquid_codec\",\"fsq_levels\":[8,7,6,6],\"n_q\":12,\"quantizer_dim\":256,\"stft_n_fft\":1280,"
        "\"stft_hop_length\":320,\"stft_frame_stack\":6,\"decoder_window\":30,\"sampling_rate\":24000}");
}

}  // namespace

bool is_lfm3_audio_published_gguf(const std::filesystem::path & model_path) noexcept {
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

Lfm3PublishedGguf open_lfm3_audio_published_gguf(const std::filesystem::path & model_path) {
    Lfm3PublishedGguf out;
    out.backbone_path = pick_backbone(model_path);
    if (!is_backbone(out.backbone_path)) {
        throw std::runtime_error("LFM3-Audio model must be a backbone GGUF with general.architecture lfm3: " + out.backbone_path.string());
    }

    out.mmproj_path = pick_mmproj(out.backbone_path);
    const GgufMetadata model_meta(out.backbone_path);
    const GgufMetadata mmproj_meta(out.mmproj_path);
    if (mmproj_meta.find_bool("clip.has_vision_encoder")) {
        throw std::runtime_error("LFM3-Audio does not run LFM3-Omni; " + out.mmproj_path.filename().string() + " has a vision encoder");
    }

    if (!mmproj_meta.find_bool("clip.has_gen_audio_encoder")) {
        throw std::runtime_error("LFM3-Audio mmproj GGUF has no audio output (clip.has_gen_audio_encoder): " + out.mmproj_path.string());
    }

    if (model_meta.find_str("tokenizer.ggml.model") != "gpt2" || model_meta.find_str("tokenizer.ggml.pre") != "lfm2") {
        throw std::runtime_error("LFM3-Audio expects the byte-level BPE tokenizer (gpt2, pre lfm2) in " + out.backbone_path.string());
    }

    out.vocabulary.tokens = model_meta.require_str_array("tokenizer.ggml.tokens");
    out.vocabulary.merges = model_meta.require_str_array("tokenizer.ggml.merges");
    const auto types = model_meta.require_int_array("tokenizer.ggml.token_type");
    out.vocabulary.token_types.assign(types.begin(), types.end());

    const auto backbone = assets::open_tensor_source(out.backbone_path);
    const auto tower = assets::open_tensor_source(out.mmproj_path);
    // The head's rows: llama.cpp pads the vocabulary with unused [PADn] entries up to them.
    out.config = make_config(model_meta, mmproj_meta, *backbone, *tower, backbone->require_metadata("token_embd.weight").shape.at(0));
    out.generation_config = make_generation_config(model_meta, mmproj_meta, out.vocabulary);
    out.codec_config = make_codec_config();

    auto model = backbone_entries(EntryBuilder(backbone));
    model.merge(mmproj_entries(EntryBuilder(tower), json::require_i64(out.config, "codebooks")));
    out.model = std::make_shared<PublishedTensorSource>(out.backbone_path, std::move(model));
    out.codec = std::make_shared<PublishedTensorSource>(out.mmproj_path,
                                                        codec_entries(EntryBuilder(tower), json::require_i64(out.codec_config, "stft_n_fft")));
    return out;
}

}  // namespace engine::community_models::lfm3_audio
