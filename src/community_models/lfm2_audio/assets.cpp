#include "engine/community_models/lfm2_audio/assets.h"

#include "engine/framework/io/filesystem.h"

#include <gguf.h>

#include <algorithm>
#include <cctype>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace engine::community_models::lfm2_audio {
namespace {

constexpr const char * kMmprojPrefix = "mmproj-";

class GgufMetadata {
public:
    explicit GgufMetadata(const std::filesystem::path & path) : path_(path) {
        gguf_init_params params{};
        params.no_alloc = true;
        params.ctx = nullptr;

        ctx_.reset(gguf_init_from_file(path.string().c_str(), params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("LFM2-Audio failed to read GGUF metadata: " + path.string());
        }
    }

    [[nodiscard]] bool has(const char * key) const { return gguf_find_key(ctx_.get(), key) >= 0; }

    [[nodiscard]] std::optional<std::string> find_str(const char * key) const {
        const int64_t id = gguf_find_key(ctx_.get(), key);
        if (id < 0 || gguf_get_kv_type(ctx_.get(), id) != GGUF_TYPE_STRING) {
            return std::nullopt;
        }

        return std::string(gguf_get_val_str(ctx_.get(), id));
    }

    [[nodiscard]] int64_t require_int(const char * key) const {
        const int64_t id = require_key(key);
        switch (gguf_get_kv_type(ctx_.get(), id)) {
            case GGUF_TYPE_UINT32: return static_cast<int64_t>(gguf_get_val_u32(ctx_.get(), id));
            case GGUF_TYPE_INT32: return static_cast<int64_t>(gguf_get_val_i32(ctx_.get(), id));
            case GGUF_TYPE_UINT64: return static_cast<int64_t>(gguf_get_val_u64(ctx_.get(), id));
            case GGUF_TYPE_INT64: return gguf_get_val_i64(ctx_.get(), id);
            default: throw std::runtime_error("LFM2-Audio GGUF key " + std::string(key) + " is not an integer in " + path_.string());
        }
    }

    [[nodiscard]] float require_f32(const char * key) const {
        const int64_t id = require_key(key);
        if (gguf_get_kv_type(ctx_.get(), id) != GGUF_TYPE_FLOAT32) {
            throw std::runtime_error("LFM2-Audio GGUF key " + std::string(key) + " is not a float in " + path_.string());
        }

        return gguf_get_val_f32(ctx_.get(), id);
    }

    [[nodiscard]] std::vector<int64_t> require_int_array(const char * key) const {
        const int64_t id = require_key(key);
        if (gguf_get_kv_type(ctx_.get(), id) != GGUF_TYPE_ARRAY) {
            throw std::runtime_error("LFM2-Audio GGUF key " + std::string(key) + " is not an array in " + path_.string());
        }

        const size_t count = gguf_get_arr_n(ctx_.get(), id);
        const void * data = gguf_get_arr_data(ctx_.get(), id);
        std::vector<int64_t> values(count);
        switch (gguf_get_arr_type(ctx_.get(), id)) {
            case GGUF_TYPE_INT32:
                std::transform(static_cast<const int32_t *>(data), static_cast<const int32_t *>(data) + count, values.begin(),
                    [](int32_t v) { return static_cast<int64_t>(v); });
                break;
            case GGUF_TYPE_UINT32:
                std::transform(static_cast<const uint32_t *>(data), static_cast<const uint32_t *>(data) + count, values.begin(),
                    [](uint32_t v) { return static_cast<int64_t>(v); });
                break;
            default:
                throw std::runtime_error("LFM2-Audio GGUF key " + std::string(key) + " is not an integer array in " + path_.string());
        }

        return values;
    }

    [[nodiscard]] std::vector<std::string> str_array(const char * key) const {
        const int64_t id = gguf_find_key(ctx_.get(), key);
        if (id < 0) {
            return {};
        }

        if (gguf_get_kv_type(ctx_.get(), id) != GGUF_TYPE_ARRAY || gguf_get_arr_type(ctx_.get(), id) != GGUF_TYPE_STRING) {
            throw std::runtime_error("LFM2-Audio GGUF key " + std::string(key) + " is not a string array in " + path_.string());
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
            throw std::runtime_error("LFM2-Audio GGUF is missing " + std::string(key) + ": " + path_.string());
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

bool is_backbone_gguf(const std::filesystem::path & path) {
    return GgufMetadata(path).find_str("general.architecture") == "lfm2";
}

bool is_encoder_gguf(const std::filesystem::path & path) {
    return GgufMetadata(path).find_str("clip.projector_type") == "lfm2a";
}

// Backbone candidates are the GGUFs without a component prefix. Liquid's
// packages also carry vocoder- and tokenizer- files for audio output.
std::vector<std::string> list_gguf_files(const std::filesystem::path & root, bool mmproj) {
    std::vector<std::string> names;
    std::error_code error;
    for (const auto & entry : std::filesystem::directory_iterator(root, error)) {
        if (!entry.is_regular_file() || !has_gguf_extension(entry.path())) {
            continue;
        }

        const auto name = entry.path().filename().string();
        const bool is_mmproj = name.rfind(kMmprojPrefix, 0) == 0;
        const bool is_other_component = name.rfind("vocoder-", 0) == 0 || name.rfind("tokenizer-", 0) == 0;
        if (mmproj ? is_mmproj : (!is_mmproj && !is_other_component)) {
            names.push_back(name);
        }
    }

    std::sort(names.begin(), names.end());
    return names;
}

std::string join(const std::vector<std::string> & values) {
    std::string out;
    for (const auto & value : values) {
        out += out.empty() ? value : ", " + value;
    }

    return out;
}

std::filesystem::path resolve_component(
    const std::filesystem::path & root, const char * option_name, const std::string & value) {
    const auto relative = std::filesystem::path(value).lexically_normal();
    if (value.empty() || relative.is_absolute() || (!relative.empty() && *relative.begin() == "..")) {
        throw std::runtime_error(std::string(option_name) + " must be a nonempty path inside the model directory");
    }

    const auto path = root / relative;
    if (!io::is_existing_file(path) || !has_gguf_extension(path)) {
        throw std::runtime_error(std::string(option_name) + " must name an existing GGUF file: " + path.string());
    }

    return path;
}

std::string default_model_gguf(const Lfm2AudioAssets & assets) {
    if (!assets.default_model_gguf.empty()) {
        return assets.default_model_gguf;
    }

    const auto candidates = list_gguf_files(assets.model_root, false);
    if (candidates.size() == 1) {
        return candidates.front();
    }

    if (candidates.empty()) {
        throw std::runtime_error("LFM2-Audio model directory has no backbone GGUF: " + assets.model_root.string());
    }

    throw std::runtime_error("LFM2-Audio model directory has several backbone GGUFs (" + join(candidates) +
                             "); choose one with lfm2_audio.model_gguf");
}

std::string default_mmproj_gguf(const Lfm2AudioAssets & assets, const std::filesystem::path & model_path) {
    const auto paired = std::string(kMmprojPrefix) + model_path.filename().string();
    if (io::is_existing_file(assets.model_root / paired)) {
        return paired;
    }

    const auto candidates = list_gguf_files(assets.model_root, true);
    if (candidates.size() == 1) {
        return candidates.front();
    }

    throw std::runtime_error("LFM2-Audio cannot pick the mmproj GGUF for " + model_path.filename().string() +
                             (candidates.empty() ? std::string(": none found") : " among " + join(candidates)) +
                             "; choose one with lfm2_audio.mmproj_gguf");
}

int64_t require_dim(const assets::TensorSource & source, const std::string & name, size_t axis) {
    const auto metadata = source.require_metadata(name);
    if (axis >= metadata.shape.size()) {
        throw std::runtime_error("LFM2-Audio tensor " + name + " has too few dimensions");
    }

    return metadata.shape[axis];
}

Lfm2BackboneConfig read_backbone_config(const GgufMetadata & meta, const assets::TensorSource & source) {
    Lfm2BackboneConfig config;
    const auto layers = meta.require_int("lfm2.block_count");
    config.hidden_size = meta.require_int("lfm2.embedding_length");
    config.intermediate_size = meta.require_int("lfm2.feed_forward_length");
    config.num_attention_heads = meta.require_int("lfm2.attention.head_count");
    config.context_length = meta.require_int("lfm2.context_length");
    config.conv_kernel_size = meta.require_int("lfm2.shortconv.l_cache");
    config.rms_norm_eps = meta.require_f32("lfm2.attention.layer_norm_rms_epsilon");
    config.rope_theta = meta.require_f32("lfm2.rope.freq_base");

    config.kv_heads = meta.require_int_array("lfm2.attention.head_count_kv");
    if (static_cast<int64_t>(config.kv_heads.size()) != layers) {
        throw std::runtime_error("LFM2-Audio head_count_kv must list one value per layer");
    }

    if (std::none_of(config.kv_heads.begin(), config.kv_heads.end(), [](int64_t heads) { return heads > 0; })) {
        throw std::runtime_error("LFM2-Audio backbone has no attention layer");
    }

    if (config.num_attention_heads <= 0 || config.hidden_size % config.num_attention_heads != 0) {
        throw std::runtime_error("LFM2-Audio attention head count does not divide the hidden size");
    }

    config.head_dim = config.hidden_size / config.num_attention_heads;
    config.vocab_size = require_dim(source, "token_embd.weight", 0);
    if (require_dim(source, "token_embd.weight", 1) != config.hidden_size) {
        throw std::runtime_error("LFM2-Audio token embedding width does not match the hidden size");
    }

    return config;
}

Lfm2FastConformerEncoderConfig read_encoder_config(const GgufMetadata & meta, const assets::TensorSource & source) {
    Lfm2FastConformerEncoderConfig config;
    config.n_mels = meta.require_int("clip.audio.num_mel_bins");
    config.hidden_size = meta.require_int("clip.audio.embedding_length");
    config.num_layers = meta.require_int("clip.audio.block_count");
    config.num_heads = meta.require_int("clip.audio.attention.head_count");
    config.layer_norm_eps = meta.require_f32("clip.audio.attention.layer_norm_epsilon");

    // clip.audio.feed_forward_length holds d_model, not the FFN width (llama.cpp's
    // LFM2AudioModel.set_gguf_parameters), so these sizes come from the tensors.
    config.intermediate_size = require_dim(source, "a.blk.0.ffn_up.weight", 0);
    config.conv_kernel_size = require_dim(source, "a.blk.0.conv_dw.weight", 1);
    config.subsampling_channels = require_dim(source, "a.conv1d.0.weight", 0);
    config.adapter_hidden_size = require_dim(source, "mm.a.mlp.1.weight", 0);
    config.output_size = require_dim(source, "mm.a.mlp.3.weight", 0);

    if (config.num_heads <= 0 || config.hidden_size % config.num_heads != 0) {
        throw std::runtime_error("LFM2-Audio encoder head count does not divide the encoder width");
    }

    return config;
}

}  // namespace

std::shared_ptr<const Lfm2AudioAssets> load_lfm2_audio_assets(const std::filesystem::path & model_path) {
    auto assets = std::make_shared<Lfm2AudioAssets>();
    if (io::is_existing_directory(model_path)) {
        assets->model_root = model_path;
    } else if (io::is_existing_file(model_path) && has_gguf_extension(model_path)) {
        assets->model_root = model_path.parent_path().empty() ? std::filesystem::path(".") : model_path.parent_path();
        assets->default_model_gguf = model_path.filename().string();
        if (!is_backbone_gguf(model_path)) {
            throw std::runtime_error("LFM2-Audio --model must be the package directory or its LFM2 backbone GGUF");
        }
    } else {
        throw std::runtime_error("LFM2-Audio --model must be the package directory: " + model_path.string());
    }

    // The loader is also asked about unrelated models, so require the shape of
    // an LFM2-Audio package: an LFM2 backbone and an lfm2a encoder.
    const auto backbones = list_gguf_files(assets->model_root, false);
    const auto encoders = list_gguf_files(assets->model_root, true);

    const bool has_backbone = std::any_of(backbones.begin(), backbones.end(), [&](const std::string & name) {
        return is_backbone_gguf(assets->model_root / name);
    });
    const bool has_encoder = std::any_of(encoders.begin(), encoders.end(), [&](const std::string & name) {
        return is_encoder_gguf(assets->model_root / name);
    });

    if (!has_backbone || !has_encoder) {
        throw std::runtime_error(std::string("LFM2-Audio model directory has no ") +
                                 (has_backbone ? "mmproj GGUF with clip.projector_type = lfm2a"
                                               : "backbone GGUF with general.architecture = lfm2") +
                                 ": " + assets->model_root.string());
    }

    return assets;
}

bool has_lfm2_audio_component(const std::filesystem::path & model_path) {
    const auto is_component = [](const std::filesystem::path & path) {
        try {
            return has_gguf_extension(path) && (is_backbone_gguf(path) || is_encoder_gguf(path));
        } catch (const std::exception &) {
            return false;  // an unreadable file is not ours to report
        }
    };

    if (io::is_existing_file(model_path)) {
        return is_component(model_path);
    }

    std::error_code error;
    for (const auto & entry : std::filesystem::directory_iterator(model_path, error)) {
        if (entry.is_regular_file() && is_component(entry.path())) {
            return true;
        }
    }

    return false;
}

std::shared_ptr<const Lfm2AudioComponents> load_lfm2_audio_components(
    const Lfm2AudioAssets & assets,
    const std::string & model_gguf,
    const std::string & mmproj_gguf) {
    auto out = std::make_shared<Lfm2AudioComponents>();
    out->model_path = resolve_component(
        assets.model_root, "lfm2_audio.model_gguf", model_gguf.empty() ? default_model_gguf(assets) : model_gguf);
    out->mmproj_path = resolve_component(
        assets.model_root,
        "lfm2_audio.mmproj_gguf",
        mmproj_gguf.empty() ? default_mmproj_gguf(assets, out->model_path) : mmproj_gguf);

    const GgufMetadata model_meta(out->model_path);
    if (model_meta.find_str("general.architecture") != "lfm2") {
        throw std::runtime_error("LFM2-Audio backbone GGUF must have general.architecture = lfm2: " + out->model_path.string());
    }

    const GgufMetadata mmproj_meta(out->mmproj_path);
    if (mmproj_meta.find_str("clip.projector_type") != "lfm2a") {
        throw std::runtime_error("LFM2-Audio mmproj GGUF must have clip.projector_type = lfm2a: " + out->mmproj_path.string());
    }

    out->model = assets::open_tensor_source(out->model_path);
    out->mmproj = assets::open_tensor_source(out->mmproj_path);

    out->backbone = read_backbone_config(model_meta, *out->model);
    out->encoder = read_encoder_config(mmproj_meta, *out->mmproj);
    if (out->encoder.output_size != out->backbone.hidden_size) {
        throw std::runtime_error("LFM2-Audio adapter output width does not match the backbone hidden size");
    }

    out->vocabulary.tokens = model_meta.str_array("tokenizer.ggml.tokens");
    out->vocabulary.merges = model_meta.str_array("tokenizer.ggml.merges");
    const auto types = model_meta.require_int_array("tokenizer.ggml.token_type");
    out->vocabulary.token_types.assign(types.begin(), types.end());
    out->vocabulary.pre_tokenizer = model_meta.find_str("tokenizer.ggml.pre").value_or("");

    if (model_meta.find_str("tokenizer.ggml.model") != "gpt2") {
        throw std::runtime_error("LFM2-Audio expects a byte-level BPE (gpt2) tokenizer in the backbone GGUF");
    }

    // Without merges the BPE would fall back to single characters.
    if (out->vocabulary.tokens.empty() || out->vocabulary.merges.empty() ||
        out->vocabulary.token_types.size() != out->vocabulary.tokens.size()) {
        throw std::runtime_error("LFM2-Audio backbone GGUF has an incomplete tokenizer vocabulary");
    }

    out->languages = model_meta.str_array("general.languages");
    return out;
}

}  // namespace engine::community_models::lfm2_audio
