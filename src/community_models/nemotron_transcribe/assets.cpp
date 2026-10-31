#include "engine/community_models/nemotron_transcribe/assets.h"

#include "engine/framework/model_spec/package.h"

#include <gguf.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace engine::models::nemotron_transcribe {
namespace {
namespace json = engine::io::json;

// Read only the native GGUF header. Tensor uploads remain framework-owned.
class Metadata {
public:
    explicit Metadata(const std::filesystem::path & path)
        : data_(gguf_init_from_file(path.string().c_str(), gguf_init_params{true, nullptr}), gguf_free) {
        if (!data_) throw std::runtime_error("Nemotron Transcribe cannot read GGUF: " + path.string());
    }
    std::string text(const std::string & name) const {
        const auto key = require(name, GGUF_TYPE_STRING);
        return gguf_get_val_str(data_.get(), key);
    }
    int count(const std::string & name) const {
        const auto key = require(name, GGUF_TYPE_UINT32);
        const auto value = gguf_get_val_u32(data_.get(), key);
        if (value > static_cast<uint32_t>(std::numeric_limits<int>::max()))
            throw std::runtime_error("Nemotron Transcribe GGUF count is too large: " + name);
        return static_cast<int>(value);
    }
    int optional_count(const std::string & name, int fallback) const {
        return gguf_find_key(data_.get(), name.c_str()) < 0 ? fallback : count(name);
    }
    bool boolean(const std::string & name) const {
        return gguf_get_val_bool(data_.get(), require(name, GGUF_TYPE_BOOL));
    }
    float real(const std::string & name) const {
        const float value = gguf_get_val_f32(data_.get(), require(name, GGUF_TYPE_FLOAT32));
        if (!std::isfinite(value)) throw std::runtime_error("non-finite GGUF field: " + name);
        return value;
    }
private:
    int64_t require(const std::string & name, gguf_type type) const {
        const auto key = gguf_find_key(data_.get(), name.c_str());
        if (key < 0 || gguf_get_kv_type(data_.get(), key) != type)
            throw std::runtime_error("Nemotron Transcribe missing or invalid GGUF field: " + name);
        return key;
    }
    std::unique_ptr<gguf_context, decltype(&gguf_free)> data_;
};

RopeExpertConfig branch(const Metadata & metadata, const std::string & prefix) {
    RopeExpertConfig config;
    config.feature_size = metadata.count(prefix + ".feat_in");
    config.hidden_size = metadata.count(prefix + ".d_model");
    config.heads = metadata.count(prefix + ".n_heads");
    config.layers = metadata.count(prefix + ".n_layers");
    config.intermediate_size = metadata.count(prefix + ".ff_hidden");
    config.subsampling_factor = metadata.count(prefix + ".subsampling_factor");
    config.rotary_dimensions = metadata.count(prefix + ".rotary_dims");
    config.qk_norm = metadata.boolean(prefix + ".qk_norm");
    config.qkv_bias = metadata.boolean(prefix + ".qkv_bias");
    config.pre_block_norm = metadata.boolean(prefix + ".pre_block_norm");
    return config;
}

std::string decode_tokenizer(const std::string & encoded) {
    if (encoded.empty() || encoded.size() % 4 != 0)
        throw std::runtime_error("invalid Nemotron Transcribe CTC tokenizer encoding");
    const auto digit = [](char c) -> int {
        if (c >= 'A' && c <= 'Z') return c - 'A';
        if (c >= 'a' && c <= 'z') return c - 'a' + 26;
        if (c >= '0' && c <= '9') return c - '0' + 52;
        if (c == '+') return 62;
        if (c == '/') return 63;
        return -1;
    };
    std::string result;
    result.reserve(encoded.size() / 4 * 3);
    for (size_t i = 0; i < encoded.size(); i += 4) {
        const int a = digit(encoded[i]), b = digit(encoded[i + 1]);
        const bool pad2 = encoded[i + 2] == '=', pad3 = encoded[i + 3] == '=';
        const int c = pad2 ? 0 : digit(encoded[i + 2]), d = pad3 ? 0 : digit(encoded[i + 3]);
        if (a < 0 || b < 0 || c < 0 || d < 0 || (pad2 && !pad3) ||
            ((pad2 || pad3) && i + 4 != encoded.size()) || (pad2 && (b & 15)) || (pad3 && !pad2 && (c & 3)))
            throw std::runtime_error("invalid Nemotron Transcribe CTC tokenizer base64");
        const uint32_t word = (static_cast<uint32_t>(a) << 18) | (static_cast<uint32_t>(b) << 12) |
            (static_cast<uint32_t>(c) << 6) | static_cast<uint32_t>(d);
        result.push_back(static_cast<char>(word >> 16));
        if (!pad2) result.push_back(static_cast<char>((word >> 8) & 255));
        if (!pad3) result.push_back(static_cast<char>(word & 255));
    }
    return result;
}

void parse_decoder(const json::Value & root, TranscribeAssets & assets) {
    if (root.require("model_type").as_string() != "nemotron_h" ||
        root.require("mlp_hidden_act").as_string() != "relu2" ||
        root.require("mamba_hidden_act").as_string() != "silu" ||
        !root.require("norm_topk_prob").as_bool() || root.require("n_group").as_i64() != 1 ||
        !root.require("moe_latent_size").is_null() ||
        root.require("attention_bias").as_bool() || root.require("mamba_proj_bias").as_bool()) {
        throw std::runtime_error("unsupported Nemotron Transcribe backbone configuration");
    }
    auto & c = assets.decoder;
    c.hidden_size = root.require("hidden_size").as_i64();
    c.vocab_size = root.require("vocab_size").as_i64();
    c.attention_heads = root.require("num_attention_heads").as_i64();
    c.attention_kv_heads = root.require("num_key_value_heads").as_i64();
    c.attention_head_dim = root.require("head_dim").as_i64();
    c.mamba_heads = root.require("mamba_num_heads").as_i64();
    c.mamba_head_dim = root.require("mamba_head_dim").as_i64();
    c.mamba_state_size = root.require("ssm_state_size").as_i64();
    c.mamba_groups = root.require("n_groups").as_i64();
    c.conv_kernel = root.require("conv_kernel").as_i64();
    c.experts = root.require("n_routed_experts").as_i64();
    c.experts_per_token = root.require("num_experts_per_tok").as_i64();
    c.expert_width = root.require("moe_intermediate_size").as_i64();
    c.shared_expert_width = root.require("moe_shared_expert_intermediate_size").as_i64();
    c.routed_scaling_factor = root.require("routed_scaling_factor").as_f32();
    c.rms_norm_eps = root.require("norm_eps").as_f32();
    c.block_pattern.clear();
    for (const auto & item : root.require("layers_block_type").as_array()) {
        const auto & name = item.as_string();
        if (name == "mamba") c.block_pattern += 'M';
        else if (name == "moe") c.block_pattern += 'E';
        else if (name == "attention") c.block_pattern += 'A';
        else throw std::runtime_error("unsupported Nemotron Transcribe block: " + name);
    }
    assets.maximum_context = root.require("max_position_embeddings").as_i64();
    // Match the native SALM serving context rather than allocating the whole
    // backbone's 262k context for every session. Audio and output share this cap.
    c.max_context = std::min<int64_t>(73728, assets.maximum_context);
}
}  // namespace

assets::TensorStorageType transcribe_weight_type(std::string_view value) {
    if (value != "native" && value != "f32" && value != "f16" && value != "bf16" && value != "q8_0")
        throw std::runtime_error("Nemotron Transcribe weight_type must be native, f32, f16, bf16 or q8_0");
    return assets::parse_tensor_storage_type(value);
}

size_t estimate_transcribe_weight_bytes(const std::vector<assets::TensorMetadata> & tensors,
    assets::TensorStorageType storage) {
    const auto multiply = [](size_t a, size_t b) {
        if (b != 0 && a > std::numeric_limits<size_t>::max() / b)
            throw std::runtime_error("Nemotron Transcribe weight estimate overflows");
        return a * b;
    };
    const auto bytes = [&](const assets::TensorMetadata & tensor, ggml_type type) {
        if (tensor.shape.empty() || tensor.shape.size() > 4 ||
            std::any_of(tensor.shape.begin(), tensor.shape.end(), [](int64_t x) {
                return x <= 0 || static_cast<uint64_t>(x) > std::numeric_limits<size_t>::max();
            }))
            throw std::runtime_error("Nemotron Transcribe weight estimate has an invalid shape");
        const auto row = static_cast<size_t>(tensor.shape.back());
        const auto block = static_cast<size_t>(ggml_blck_size(type));
        if (row % block != 0)
            throw std::runtime_error("Nemotron Transcribe weight estimate has an invalid native row");
        size_t size = multiply(row / block, ggml_type_size(type));
        for (size_t i = 0; i + 1 < tensor.shape.size(); ++i)
            size = multiply(size, static_cast<size_t>(tensor.shape[i]));
        return size;
    };
    size_t total = 0;
    for (const auto & tensor : tensors) {
        const auto native = assets::ggml_type_for_tensor_storage(assets::tensor_storage_type_for_dtype(tensor.dtype));
        auto type = storage == assets::TensorStorageType::Native
            ? native : assets::ggml_type_for_tensor_storage(storage);
        const auto & name = tensor.name;
        const bool fixed_f32 = tensor.shape.size() == 1 || native == GGML_TYPE_F32 ||
            name.find("norm") != std::string::npos || name.rfind("ctc.", 0) == 0 ||
            name.find("ssm_conv1d") != std::string::npos || name.find(".ssm_a") != std::string::npos ||
            name.find(".ssm_d") != std::string::npos || name.find("ffn_gate_inp") != std::string::npos;
        if (fixed_f32 || (ggml_is_quantized(type) &&
            (tensor.shape.size() < 2 || tensor.shape.back() % ggml_blck_size(type) != 0))) type = GGML_TYPE_F32;
        const size_t cost = std::max(bytes(tensor, native), bytes(tensor, type));
        if (cost > std::numeric_limits<size_t>::max() - total)
            throw std::runtime_error("Nemotron Transcribe weight estimate overflows");
        total += cost;
    }
    return total;
}

std::shared_ptr<const TranscribeAssets> load_transcribe_assets(const std::filesystem::path & path) {
    auto result = std::make_shared<TranscribeAssets>();
    result->resources = model_spec::load_resource_bundle_for_family(path, "nemotron_transcribe");
    result->decoder_weights = result->resources.open_tensor_source("weights", assets::TensorSourceOptions{true});
    const Metadata decoder(result->decoder_weights->source_path());
    if (decoder.text("general.architecture") != "nemotron_h_moe")
        throw std::runtime_error("Nemotron Transcribe requires a Nemotron-H MoE decoder GGUF");
    const auto identity = decoder.text("nemo_speech.checkpoint_id");
    const auto sibling_name = std::filesystem::u8path(decoder.text("nemo_speech.speech_model"));
    if (identity.empty() || sibling_name.empty() || sibling_name.is_absolute() ||
        sibling_name.filename() != sibling_name || sibling_name == "." || sibling_name == "..")
        throw std::runtime_error("invalid Nemotron Transcribe encoder pairing metadata");
    auto sibling = result->resources.model_directory() / sibling_name;
    if (!std::filesystem::exists(sibling))
        sibling = result->resources.model_directory() / std::filesystem::u8path("mmproj-" + sibling_name.u8string());
    const Metadata encoder(sibling);
    if (encoder.text("nemo_speech.checkpoint_id") != identity)
        throw std::runtime_error("Nemotron Transcribe encoder and decoder belong to different checkpoints");
    result->encoder_weights = assets::open_tensor_source(sibling, assets::TensorSourceOptions{true});
    auto & e = result->encoder;
    e.asr = branch(encoder, "pee.asr");
    e.diar = branch(encoder, "pee.diar");
    e.projection_dim = encoder.count("pee.fusion.proj_dim");
    e.speaker_count = encoder.count("pee.diar.num_speakers");
    e.speaker_hidden_size = encoder.count("pee.diar.tf_d_model");
    e.upsample_factor = encoder.count("pee.diar.upsample_factor");
    e.speaker_activity_threshold = encoder.real("pee.fusion.speaker_activity_threshold");
    e.online_inference_length = encoder.count("pee.online.inference_length");
    e.chunk_left_context = encoder.count("pee.online.chunk_left_context");
    e.chunk_right_context = encoder.count("pee.online.chunk_right_context");
    if (encoder.text("pee.asr.normalize_type") != "per_feature" ||
        encoder.text("pee.diar.normalize_type") != "NA" ||
        encoder.text("pee.fusion.speaker_feature_mode") != "thresholded")
        throw std::runtime_error("unsupported Nemotron Transcribe feature normalization/fusion");
    const auto processor = result->resources.parse_json("processor_config");
    const auto & fe = processor.require("feature_extractor");
    auto & f = result->frontend;
    f.sample_rate = encoder.count("pee.preprocessor.sample_rate");
    f.n_fft = encoder.count("pee.preprocessor.n_fft");
    f.mel_bins = encoder.count("pee.preprocessor.features");
    f.window_length = static_cast<int>(std::lround(encoder.real("pee.preprocessor.window_size") * f.sample_rate));
    f.hop_length = static_cast<int>(std::lround(encoder.real("pee.preprocessor.window_stride") * f.sample_rate));
    f.preemphasis = fe.require("preemphasis").as_f32();
    if (f.sample_rate != fe.require("sampling_rate").as_i64() ||
        f.n_fft != fe.require("n_fft").as_i64() || f.mel_bins != fe.require("feature_size").as_i64() ||
        f.window_length != fe.require("win_length").as_i64() || f.hop_length != fe.require("hop_length").as_i64())
        throw std::runtime_error("Nemotron Transcribe processor and encoder frontend disagree");
    result->mel_filterbank = result->encoder_weights->require_f32("preprocessor.fb", {f.mel_bins, f.n_fft / 2 + 1});
    result->frontend_window = json::number_array_as<float>(result->resources.parse_json("frontend_window"));
    if (result->frontend_window.size() != static_cast<size_t>(f.window_length) ||
        std::any_of(result->frontend_window.begin(), result->frontend_window.end(), [](float x) { return !std::isfinite(x); }))
        throw std::runtime_error("invalid Nemotron Transcribe trained STFT window");
    auto & c = result->ctc;
    c.dimension = encoder.count("ctc.d_model");
    c.heads = encoder.count("ctc.n_heads");
    c.layers = encoder.count("ctc.n_layers");
    c.ff_hidden = encoder.count("ctc.ff_hidden");
    c.rotary_dims = encoder.count("ctc.rotary_dims");
    c.vocabulary = encoder.count("ctc.vocab_size");
    c.blank = c.vocabulary - 1;
    c.upsampling = encoder.count("ctc.upsampling_factor");
    c.qk_norm = encoder.boolean("ctc.qk_norm");
    c.qkv_bias = encoder.boolean("ctc.qkv_bias");
    c.residual_scale = encoder.real("ctc.residual_scale");
    c.model_name = encoder.text("ctc.model_name");
    result->alignment_tokenizer = decode_tokenizer(encoder.text("ctc.tokenizer.spm_model"));
    parse_decoder(result->resources.parse_json("backbone_config"), *result);
    result->decoder.expert_row_alignment = decoder.optional_count("nemotron_transcribe.expert_row_alignment", 1);
    if (result->decoder.expert_row_alignment != 1 && result->decoder.expert_row_alignment != 256)
        throw std::runtime_error("unsupported Nemotron Transcribe expert row alignment");
    const auto generation = result->resources.parse_json("generation_config");
    if (json::require_i64_array_or_scalar(generation, "eos_token_id") != std::vector<int64_t>{11})
        throw std::runtime_error("Nemotron Transcribe requires the trained end-of-turn token 11");
    result->max_new_tokens = json::optional_i64(generation, "max_new_tokens", result->max_new_tokens);
    if (e.projection_dim != result->decoder.hidden_size || c.dimension != e.asr.hidden_size ||
        f.mel_bins != e.asr.feature_size || f.mel_bins != e.diar.feature_size ||
        result->max_new_tokens < 0 || result->maximum_context <= 0)
        throw std::runtime_error("inconsistent Nemotron Transcribe encoder/decoder dimensions");
    validate_encoder_config(e);
    return result;
}
}  // namespace engine::models::nemotron_transcribe
