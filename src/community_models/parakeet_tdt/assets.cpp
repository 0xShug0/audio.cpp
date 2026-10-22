#include "engine/community_models/parakeet_tdt/assets.h"

#include "engine/framework/model_spec/package.h"
#include "engine/framework/io/json.h"

#include <gguf.h>

#include <algorithm>
#include <cmath>
#include <optional>
#include <regex>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace engine::community_models::parakeet_tdt {
namespace json = engine::io::json;
namespace {

void validate_config(const ParakeetConfig & config) {
    if (config.model_type != "parakeet_tdt") {
        throw std::runtime_error("Parakeet TDT expects model_type=parakeet_tdt");
    }
    if (config.vocab_size <= 0 || config.blank_token_id < 0 || config.blank_token_id >= config.vocab_size) {
        throw std::runtime_error("Parakeet TDT invalid vocab metadata");
    }
    if (config.decoder_hidden_size <= 0 || config.decoder_layers != 2) {
        throw std::runtime_error("Parakeet TDT expects 2-layer LSTM decoder");
    }
    if (config.encoder.hidden_size <= 0 || config.encoder.layers <= 0 || config.encoder.heads <= 0 ||
        config.encoder.hidden_size % config.encoder.heads != 0) {
        throw std::runtime_error("Parakeet TDT invalid encoder metadata");
    }
    if (config.encoder.subsampling_factor != 8 || config.encoder.subsampling_kernel != 3 ||
        config.encoder.subsampling_stride != 2) {
        throw std::runtime_error("Parakeet TDT expects factor-8 causal subsampling config");
    }
    if (config.frontend.sample_rate != 16000 || config.frontend.feature_size <= 0 ||
        config.frontend.n_fft <= 0 || config.frontend.win_length <= 0 || config.frontend.hop_length <= 0) {
        throw std::runtime_error("Parakeet TDT invalid frontend metadata");
    }
}

std::vector<uint8_t> parse_special_token_ids(const std::filesystem::path & tokenizer_json, int64_t vocab_size) {
    std::vector<uint8_t> special(static_cast<size_t>(vocab_size), 0);
    const auto root = json::parse_file(tokenizer_json);
    if (const auto * added = root.find("added_tokens"); added != nullptr && added->is_array()) {
        for (const auto & item : added->as_array()) {
            if (!json::optional_bool(item, "special", false)) {
                continue;
            }
            const int64_t id = json::require_i64(item, "id");
            if (id >= 0 && id < vocab_size) {
                special[static_cast<size_t>(id)] = 1;
            }
        }
    }
    return special;
}

// transcribe.cpp writes Parakeet TDT checkpoints as a single GGUF in NeMo tensor
// naming, with the model config under stt.* keys and the SentencePiece vocabulary
// under tokenizer.ggml.*. These helpers read that layout directly so existing
// transcribe.cpp files load without a config.json / tokenizer.json package.
class TranscribeGgufMetadata {
public:
    explicit TranscribeGgufMetadata(const std::filesystem::path & path) {
        gguf_init_params params{};
        params.no_alloc = true;
        params.ctx = nullptr;
        ctx_.reset(gguf_init_from_file(path.string().c_str(), params));
        if (!ctx_) throw std::runtime_error("failed to read GGUF metadata: " + path.string());
    }

    bool has(const char * key) const { return gguf_find_key(ctx_.get(), key) >= 0; }

    int64_t require_int(const char * key) const {
        const int64_t id = require_key(key);
        switch (gguf_get_kv_type(ctx_.get(), id)) {
            case GGUF_TYPE_UINT8: return gguf_get_val_u8(ctx_.get(), id);
            case GGUF_TYPE_INT8: return gguf_get_val_i8(ctx_.get(), id);
            case GGUF_TYPE_UINT16: return gguf_get_val_u16(ctx_.get(), id);
            case GGUF_TYPE_INT16: return gguf_get_val_i16(ctx_.get(), id);
            case GGUF_TYPE_UINT32: return gguf_get_val_u32(ctx_.get(), id);
            case GGUF_TYPE_INT32: return gguf_get_val_i32(ctx_.get(), id);
            case GGUF_TYPE_UINT64: return static_cast<int64_t>(gguf_get_val_u64(ctx_.get(), id));
            case GGUF_TYPE_INT64: return gguf_get_val_i64(ctx_.get(), id);
            default: throw std::runtime_error(std::string("GGUF key is not an integer: ") + key);
        }
    }

    float require_float(const char * key) const {
        const int64_t id = require_key(key);
        switch (gguf_get_kv_type(ctx_.get(), id)) {
            case GGUF_TYPE_FLOAT32: return gguf_get_val_f32(ctx_.get(), id);
            case GGUF_TYPE_FLOAT64: return static_cast<float>(gguf_get_val_f64(ctx_.get(), id));
            default: throw std::runtime_error(std::string("GGUF key is not a float: ") + key);
        }
    }

    std::string require_string(const char * key) const {
        const int64_t id = require_key(key);
        if (gguf_get_kv_type(ctx_.get(), id) != GGUF_TYPE_STRING) {
            throw std::runtime_error(std::string("GGUF key is not a string: ") + key);
        }
        return gguf_get_val_str(ctx_.get(), id);
    }

    std::vector<int32_t> require_int_array(const char * key) const {
        const int64_t id = require_key(key);
        if (gguf_get_kv_type(ctx_.get(), id) != GGUF_TYPE_ARRAY) {
            throw std::runtime_error(std::string("GGUF key is not an array: ") + key);
        }
        const size_t n = gguf_get_arr_n(ctx_.get(), id);
        const void * data = gguf_get_arr_data(ctx_.get(), id);
        std::vector<int32_t> out(n);
        switch (gguf_get_arr_type(ctx_.get(), id)) {
            case GGUF_TYPE_INT32:
                for (size_t i = 0; i < n; ++i) out[i] = static_cast<const int32_t *>(data)[i];
                break;
            case GGUF_TYPE_UINT32:
                for (size_t i = 0; i < n; ++i) out[i] = static_cast<int32_t>(static_cast<const uint32_t *>(data)[i]);
                break;
            case GGUF_TYPE_INT64:
                for (size_t i = 0; i < n; ++i) out[i] = static_cast<int32_t>(static_cast<const int64_t *>(data)[i]);
                break;
            default:
                throw std::runtime_error(std::string("GGUF array is not integer: ") + key);
        }
        return out;
    }

    std::vector<std::string> require_string_array(const char * key) const {
        const int64_t id = require_key(key);
        if (gguf_get_kv_type(ctx_.get(), id) != GGUF_TYPE_ARRAY ||
            gguf_get_arr_type(ctx_.get(), id) != GGUF_TYPE_STRING) {
            throw std::runtime_error(std::string("GGUF key is not a string array: ") + key);
        }
        const size_t n = gguf_get_arr_n(ctx_.get(), id);
        std::vector<std::string> out;
        out.reserve(n);
        for (size_t i = 0; i < n; ++i) out.emplace_back(gguf_get_arr_str(ctx_.get(), id, i));
        return out;
    }

private:
    int64_t require_key(const char * key) const {
        const int64_t id = gguf_find_key(ctx_.get(), key);
        if (id < 0) throw std::runtime_error(std::string("transcribe.cpp GGUF is missing key: ") + key);
        return id;
    }

    struct Deleter {
        void operator()(gguf_context * ctx) const noexcept { gguf_free(ctx); }
    };
    std::unique_ptr<gguf_context, Deleter> ctx_;
};

bool is_transcribe_cpp_gguf(const std::filesystem::path & path) {
    if (!std::filesystem::is_regular_file(path) || path.extension() != ".gguf") return false;
    const TranscribeGgufMetadata metadata(path);
    return metadata.has("stt.parakeet.head_kind");
}

// NeMo -> Hugging Face tensor names, the inverse of transcribe.cpp's
// convert-parakeet.py tables. The LSTM keeps PyTorch's (i, f, g, o) gate order;
// transcribe.cpp stores bias_ih + bias_hh as one bias, exposed here as bias_ih.
std::optional<std::string> hf_name_for_transcribe_tensor(const std::string & name) {
    static const std::vector<std::pair<std::regex, std::string>> rules = [] {
        const std::vector<std::pair<const char *, const char *>> table = {
            {R"(^enc\.pre_encode\.conv\.(\d+)\.(weight|bias)$)", "encoder.subsampling.layers.$1.$2"},
            {R"(^enc\.pre_encode\.out\.(weight|bias)$)", "encoder.subsampling.linear.$1"},
            {R"(^enc\.blocks\.(\d+)\.norm_ff([12])\.(weight|bias)$)", "encoder.layers.$1.norm_feed_forward$2.$3"},
            {R"(^enc\.blocks\.(\d+)\.ff([12])\.(linear[12])\.(weight|bias)$)", "encoder.layers.$1.feed_forward$2.$3.$4"},
            {R"(^enc\.blocks\.(\d+)\.norm_attn\.(weight|bias)$)", "encoder.layers.$1.norm_self_att.$2"},
            {R"(^enc\.blocks\.(\d+)\.attn\.linear_q\.(weight|bias)$)", "encoder.layers.$1.self_attn.q_proj.$2"},
            {R"(^enc\.blocks\.(\d+)\.attn\.linear_k\.(weight|bias)$)", "encoder.layers.$1.self_attn.k_proj.$2"},
            {R"(^enc\.blocks\.(\d+)\.attn\.linear_v\.(weight|bias)$)", "encoder.layers.$1.self_attn.v_proj.$2"},
            {R"(^enc\.blocks\.(\d+)\.attn\.linear_out\.(weight|bias)$)", "encoder.layers.$1.self_attn.o_proj.$2"},
            {R"(^enc\.blocks\.(\d+)\.attn\.linear_pos\.weight$)", "encoder.layers.$1.self_attn.relative_k_proj.weight"},
            {R"(^enc\.blocks\.(\d+)\.attn\.pos_bias_([uv])$)", "encoder.layers.$1.self_attn.bias_$2"},
            {R"(^enc\.blocks\.(\d+)\.norm_conv\.(weight|bias)$)", "encoder.layers.$1.norm_conv.$2"},
            {R"(^enc\.blocks\.(\d+)\.conv\.pointwise([12])\.(weight|bias)$)", "encoder.layers.$1.conv.pointwise_conv$2.$3"},
            {R"(^enc\.blocks\.(\d+)\.conv\.depthwise\.(weight|bias)$)", "encoder.layers.$1.conv.depthwise_conv.$2"},
            {R"(^enc\.blocks\.(\d+)\.conv\.bn\.(weight|bias|running_mean|running_var)$)", "encoder.layers.$1.conv.norm.$2"},
            {R"(^enc\.blocks\.(\d+)\.norm_out\.(weight|bias)$)", "encoder.layers.$1.norm_out.$2"},
            {R"(^pred\.embed\.weight$)", "decoder.embedding.weight"},
            {R"(^pred\.lstm\.(\d+)\.Wx$)", "decoder.lstm.weight_ih_l$1"},
            {R"(^pred\.lstm\.(\d+)\.Wh$)", "decoder.lstm.weight_hh_l$1"},
            {R"(^pred\.lstm\.(\d+)\.bias$)", "decoder.lstm.bias_ih_l$1"},
            {R"(^joint\.pred\.(weight|bias)$)", "decoder.decoder_projector.$1"},
            {R"(^joint\.enc\.(weight|bias)$)", "encoder_projector.$1"},
            {R"(^joint\.out\.(weight|bias)$)", "joint.head.$1"},
        };
        std::vector<std::pair<std::regex, std::string>> compiled;
        compiled.reserve(table.size());
        for (const auto & [pattern, replacement] : table) compiled.emplace_back(std::regex(pattern), replacement);
        return compiled;
    }();
    for (const auto & [pattern, replacement] : rules) {
        if (std::regex_match(name, pattern)) return std::regex_replace(name, pattern, replacement);
    }
    return std::nullopt;
}

class TranscribeGgufTensorView final : public assets::TensorSource {
public:
    explicit TranscribeGgufTensorView(std::shared_ptr<const assets::TensorSource> source)
        : source_(std::move(source)) {
        for (const auto & tensor : source_->tensors()) {
            const auto hf_name = hf_name_for_transcribe_tensor(tensor.name);
            if (!hf_name) {
                throw std::runtime_error("unsupported tensor in transcribe.cpp Parakeet GGUF: " + tensor.name);
            }
            routes_.emplace(*hf_name, tensor.name);
        }
    }

    const std::filesystem::path & source_path() const noexcept override { return source_->source_path(); }
    bool has_tensor(std::string_view name) const noexcept override {
        return routes_.find(std::string(name)) != routes_.end();
    }
    assets::TensorMetadata require_metadata(std::string_view name) const override {
        auto metadata = source_->require_metadata(require_name(name));
        metadata.name = std::string(name);
        return metadata;
    }
    std::vector<assets::TensorMetadata> tensors() const override {
        std::vector<assets::TensorMetadata> result;
        result.reserve(routes_.size());
        for (const auto & [name, _] : routes_) result.push_back(require_metadata(name));
        std::sort(result.begin(), result.end(), [](const auto & lhs, const auto & rhs) { return lhs.name < rhs.name; });
        return result;
    }
    void release_storage() const override { source_->release_storage(); }
    assets::RawTensorData require_tensor_data(std::string_view name) const override {
        auto data = source_->require_tensor_data(require_name(name));
        data.metadata.name = std::string(name);
        return data;
    }
    std::vector<float> require_f32(
        std::string_view name,
        const std::optional<std::vector<int64_t>> & expected_shape) const override {
        return source_->require_f32(require_name(name), expected_shape);
    }
    std::optional<std::vector<float>> optional_f32(
        std::string_view name,
        const std::optional<std::vector<int64_t>> & expected_shape) const override {
        if (!has_tensor(name)) return std::nullopt;
        return require_f32(name, expected_shape);
    }
    void set_backend_tensor(
        ggml_tensor * tensor,
        std::string_view name,
        assets::TensorStorageType storage_type,
        const std::vector<int64_t> & expected_shape) const override {
        source_->set_backend_tensor(tensor, require_name(name), storage_type, expected_shape);
    }
    void set_backend_f32_tensor(
        ggml_tensor * tensor,
        std::string_view name,
        const std::vector<int64_t> & expected_shape) const override {
        source_->set_backend_f32_tensor(tensor, require_name(name), expected_shape);
    }
    int64_t require_i64_scalar(std::string_view name) const override {
        return source_->require_i64_scalar(require_name(name));
    }

private:
    const std::string & require_name(std::string_view name) const {
        const auto it = routes_.find(std::string(name));
        if (it == routes_.end()) {
            throw std::runtime_error("missing tensor in transcribe.cpp Parakeet GGUF: " + std::string(name));
        }
        return it->second;
    }

    std::shared_ptr<const assets::TensorSource> source_;
    std::unordered_map<std::string, std::string> routes_;
};

std::shared_ptr<const ParakeetTDTAssets> load_transcribe_cpp_assets(const std::filesystem::path & path) {
    const TranscribeGgufMetadata metadata(path);
    if (metadata.require_string("stt.parakeet.head_kind") != "tdt") {
        throw std::runtime_error("transcribe.cpp Parakeet GGUF is not a TDT model: " + path.string());
    }
    auto assets = std::make_shared<ParakeetTDTAssets>();
    auto & config = assets->config;
    config.model_type = "parakeet_tdt";
    config.vocab_size = metadata.require_int("stt.parakeet.predictor.vocab");
    config.blank_token_id = metadata.require_int("tokenizer.ggml.blank_token_id");
    config.decoder_hidden_size = metadata.require_int("stt.parakeet.predictor.hidden");
    config.decoder_layers = metadata.require_int("stt.parakeet.predictor.n_layers");
    config.max_symbols_per_step = metadata.require_int("stt.parakeet.tdt.max_symbols");
    config.durations = metadata.require_int_array("stt.parakeet.tdt.durations");

    config.encoder.hidden_size = metadata.require_int("stt.parakeet.encoder.d_model");
    config.encoder.intermediate_size = metadata.require_int("stt.parakeet.encoder.d_ff");
    config.encoder.layers = metadata.require_int("stt.parakeet.encoder.n_layers");
    config.encoder.heads = metadata.require_int("stt.parakeet.encoder.n_heads");
    config.encoder.conv_kernel = metadata.require_int("stt.parakeet.encoder.conv_kernel");
    config.encoder.subsampling_factor = metadata.require_int("stt.parakeet.encoder.subsampling_factor");
    config.encoder.subsampling_channels = metadata.require_int("stt.parakeet.encoder.subsampling_channels");
    config.encoder.max_position_embeddings = metadata.require_int("stt.parakeet.encoder.pos_emb_max_len");

    config.frontend.sample_rate = metadata.require_int("stt.frontend.sample_rate");
    config.frontend.feature_size = metadata.require_int("stt.frontend.num_mels");
    config.frontend.n_fft = metadata.require_int("stt.frontend.n_fft");
    config.frontend.win_length = metadata.require_int("stt.frontend.win_length");
    config.frontend.hop_length = metadata.require_int("stt.frontend.hop_length");
    config.frontend.preemphasis = metadata.require_float("stt.frontend.pre_emphasis");

    auto tokens = metadata.require_string_array("tokenizer.ggml.tokens");
    if (static_cast<int64_t>(tokens.size()) != config.vocab_size) {
        throw std::runtime_error("transcribe.cpp Parakeet GGUF vocabulary does not match predictor.vocab");
    }
    const auto pad = std::find(tokens.begin(), tokens.end(), "<pad>");
    if (pad == tokens.end()) throw std::runtime_error("transcribe.cpp Parakeet GGUF vocabulary has no <pad> token");
    config.pad_token_id = static_cast<int64_t>(pad - tokens.begin());

    assets->special_token_ids.assign(static_cast<size_t>(config.vocab_size), 0);
    for (const int64_t id : {metadata.require_int("tokenizer.ggml.unknown_token_id"), config.pad_token_id, config.blank_token_id}) {
        if (id < 0 || id >= config.vocab_size) throw std::runtime_error("transcribe.cpp Parakeet GGUF special token id out of range");
        assets->special_token_ids[static_cast<size_t>(id)] = 1;
    }
    assets->tokenizer = std::make_shared<engine::tokenizers::HuggingFaceTokenizerJson>(std::move(tokens), "\xE2\x96\x81", true);

    assets->source = std::make_shared<TranscribeGgufTensorView>(assets::open_tensor_source(path));
    const auto conv_in = assets->source->require_metadata("encoder.subsampling.layers.0.weight");
    if (conv_in.shape.size() != 4) throw std::runtime_error("transcribe.cpp Parakeet GGUF has an unexpected subsampling conv");
    config.encoder.subsampling_kernel = conv_in.shape[3];
    config.encoder.subsampling_stride = 2;
    validate_config(config);
    return assets;
}

ParakeetConfig parse_config(const assets::ResourceBundle & resources) {
    const auto config_root = resources.parse_json("config");

    ParakeetConfig config;
    config.model_type = json::require_string(config_root, "model_type");
    if (config_root.find("variant")) {
        config.variant = json::require_string(config_root, "variant");
    }
    config.vocab_size = json::require_i64(config_root, "vocab_size");
    config.blank_token_id = json::require_i64(config_root, "blank_token_id");
    config.pad_token_id = json::require_i64(config_root, "pad_token_id");
    config.decoder_hidden_size = json::require_i64(config_root, "decoder_hidden_size");
    config.decoder_layers = json::require_i64(config_root, "num_decoder_layers");
    config.max_symbols_per_step = json::require_i64(config_root, "max_symbols_per_step");

    // Old development packages used unscaled projections and special runtime
    // precision controls. Reject them instead of silently applying the normal
    // loader's sqrt(hidden_size) fold to incompatible weights.
    const auto & enc_config = config_root.require("encoder_config");
    if (config_root.find("audiocpp_matmul_precision") ||
        config_root.find("audiocpp_cpu_matmul_weight_type") ||
        enc_config.find("subsampling_input_scale") || enc_config.find("batch_norm_variance_floor")) {
        throw std::runtime_error("Phonon development package uses removed precision/scaling controls; reconvert with the current convert_phonon2.py");
    }
    if (config_root.find("word_timestamp_mode")) {
        config.word_timestamp_mode = json::require_string(config_root, "word_timestamp_mode");
        if (config.word_timestamp_mode != "default" && config.word_timestamp_mode != "token_duration")
            throw std::runtime_error("Parakeet TDT word_timestamp_mode must be default or token_duration");
    }
    if (const auto * ids = config_root.find("audiocpp_punctuation_token_ids")) {
        for (const auto & value : ids->as_array()) {
            const auto number = value.as_number();
            if (!std::isfinite(number) || number < 0 || number >= config.vocab_size || std::floor(number) != number)
                throw std::runtime_error("Parakeet TDT punctuation token id out of range");
            config.punctuation_token_ids.push_back(static_cast<int32_t>(number));
        }
        std::sort(config.punctuation_token_ids.begin(), config.punctuation_token_ids.end());
    } else if (config.word_timestamp_mode == "token_duration") {
        throw std::runtime_error("Parakeet TDT token_duration timestamps require punctuation token metadata");
    }

    if (const auto * dur = config_root.find("durations"); dur != nullptr && dur->is_array()) {
        config.durations.clear();
        for (const auto & d : dur->as_array()) {
            if (!d.is_number()) continue;
            config.durations.push_back(static_cast<int32_t>(d.as_i64()));
        }
    }

    const auto & encoder = config_root.require("encoder_config");
    config.encoder.hidden_size = json::require_i64(encoder, "hidden_size");
    config.encoder.intermediate_size = json::require_i64(encoder, "intermediate_size");
    config.encoder.layers = json::require_i64(encoder, "num_hidden_layers");
    config.encoder.heads = json::require_i64(encoder, "num_attention_heads");
    config.encoder.conv_kernel = json::require_i64(encoder, "conv_kernel_size");
    config.encoder.subsampling_factor = json::require_i64(encoder, "subsampling_factor");
    config.encoder.subsampling_channels = json::require_i64(encoder, "subsampling_conv_channels");
    config.encoder.subsampling_kernel = json::require_i64(encoder, "subsampling_conv_kernel_size");
    config.encoder.subsampling_stride = json::require_i64(encoder, "subsampling_conv_stride");
    config.encoder.max_position_embeddings = json::require_i64(encoder, "max_position_embeddings");

    // processor_config.json is genuinely optional (not every model directory
    // layout provides it), so fall back to the ParakeetFrontendConfig
    // defaults when it's absent. But if the file IS present, any parse or
    // schema failure is a real problem worth failing loudly on: silently
    // keeping the defaults could load a variant model's weights against the
    // wrong frontend parameters (sample rate, FFT size, hop length, ...)
    // without any indication anything went wrong.
    if (resources.has_file("processor_config")) {
        const auto processor_root = resources.parse_json("processor_config");
        const auto & feature = processor_root.require("feature_extractor");
        config.frontend.sample_rate = json::require_i64(feature, "sampling_rate");
        config.frontend.feature_size = json::require_i64(feature, "feature_size");
        config.frontend.n_fft = json::require_i64(feature, "n_fft");
        config.frontend.win_length = json::require_i64(feature, "win_length");
        config.frontend.hop_length = json::require_i64(feature, "hop_length");
        if (const auto * p = feature.find("preemphasis"); p != nullptr) {
            config.frontend.preemphasis = json::require_f32(feature, "preemphasis");
        }
    }

    validate_config(config);
    return config;
}

}  // namespace

std::shared_ptr<const ParakeetTDTAssets> load_parakeet_assets(const std::filesystem::path & model_path) {
    if (is_transcribe_cpp_gguf(model_path)) return load_transcribe_cpp_assets(model_path);
    auto resources = engine::model_spec::load_resource_bundle(
        model_path,
        engine::model_spec::default_spec_path("parakeet_tdt"));
    auto assets = std::make_shared<ParakeetTDTAssets>();
    assets->resources = std::move(resources);
    assets->source = assets->resources.open_tensor_source("weights");
    assets->config = parse_config(assets->resources);
    const auto & tokenizer_json = assets->resources.require_file("tokenizer_json");
    assets->tokenizer = engine::tokenizers::load_huggingface_tokenizer_json(tokenizer_json);
    assets->special_token_ids =
        parse_special_token_ids(tokenizer_json, assets->config.vocab_size);
    return assets;
}

}  // namespace engine::community_models::parakeet_tdt
