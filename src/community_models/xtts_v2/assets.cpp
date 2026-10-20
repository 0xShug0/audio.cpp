#include "engine/community_models/xtts_v2/assets.h"
#include "engine/community_models/xtts_v2/tokenizer_text.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/io/binary.h"
#include "engine/framework/model_spec/package.h"

#include <gguf.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>

namespace engine::models::xtts_v2 {
namespace {
using GgufMetadata = std::unique_ptr<gguf_context, decltype(&gguf_free)>;

uint32_t require_u32(const gguf_context * metadata, const char * key) {
    const int64_t index = gguf_find_key(metadata, key);
    if (index >= 0) {
        const auto type = gguf_get_kv_type(metadata, index);
        if (type == GGUF_TYPE_UINT32) return gguf_get_val_u32(metadata, index);
        if (type == GGUF_TYPE_INT32 && gguf_get_val_i32(metadata, index) >= 0) {
            return static_cast<uint32_t>(gguf_get_val_i32(metadata, index));
        }
    }
    throw std::runtime_error(std::string("XTTS v2 requires nonnegative integer metadata: ") + key);
}
}

std::optional<std::filesystem::path> find_xtts_v2_gpt_file(const std::filesystem::path & path) {
    const auto candidates = std::filesystem::is_directory(path)
        ? engine::assets::directory_gguf_files(path) : std::vector<std::filesystem::path>{path};
    std::optional<std::filesystem::path> selected;
    for (const auto & candidate : candidates) {
        if (!std::filesystem::is_regular_file(candidate) || candidate.extension() != ".gguf") continue;
        const auto source = engine::assets::open_tensor_source(candidate);
        if (source->has_tensor("gpt.text_embedding.weight") && source->has_tensor("gpt.perc.latents") &&
            (!selected || std::filesystem::file_size(candidate) > std::filesystem::file_size(*selected))) {
            selected = candidate;
        }
    }
    return selected;
}

std::shared_ptr<const XttSv2Assets> load_xtts_v2_assets(const std::filesystem::path & root) {
    auto result = std::make_shared<XttSv2Assets>();
    auto & assets = *result;
    const auto primary = find_xtts_v2_gpt_file(root);
    if (!primary) throw std::runtime_error("XTTS v2 requires its published GPT GGUF");
    assets.resources = model_spec::load_resource_bundle_for_family(*primary, "xtts_v2");
    assets.gpt_source = assets.resources.open_tensor_source("gpt_weights");
    assets.vocoder_source = assets.resources.open_tensor_source("vocoder_weights");
    assets.gpt_gguf_path = assets.gpt_source->source_path();
    assets.vocoder_gguf_path = assets.vocoder_source->source_path();
    GgufMetadata metadata(gguf_init_from_file(assets.gpt_gguf_path.string().c_str(), {true, nullptr}), gguf_free);
    if (!metadata) {
        throw std::runtime_error("XTTS v2 cannot read GPT metadata");
    }
    auto & hp = assets.hyper_params;
    hp.n_layers = require_u32(metadata.get(), "xtts.gpt.layers");
    hp.d_model = require_u32(metadata.get(), "xtts.gpt.model_dim");
    hp.n_heads = require_u32(metadata.get(), "xtts.gpt.heads");
    if (hp.n_heads == 0 || hp.d_model == 0 || hp.d_model % hp.n_heads != 0) {
        throw std::runtime_error("XTTS v2 invalid GPT head dimensions");
    }
    hp.head_dim = hp.d_model / hp.n_heads;
    hp.cond_n_heads = hp.n_heads;
    hp.ff_dim = assets.gpt_source->require_metadata("gpt.gpt.h.0.mlp.c_fc.weight").shape.at(0);
    hp.text_vocab_size = require_u32(metadata.get(), "xtts.gpt.number_text_tokens");
    hp.mel_vocab_size = require_u32(metadata.get(), "xtts.gpt.number_mel_codes");
    hp.start_text_token = require_u32(metadata.get(), "xtts.gpt.start_text_token");
    hp.stop_text_token = require_u32(metadata.get(), "xtts.gpt.stop_text_token");
    hp.start_mel_token = require_u32(metadata.get(), "xtts.gpt.start_mel_token");
    hp.stop_mel_token = require_u32(metadata.get(), "xtts.gpt.stop_mel_token");
    hp.max_mel_tokens = require_u32(metadata.get(), "xtts.gpt.max_mel_tokens");
    hp.code_stride = require_u32(metadata.get(), "xtts.gpt.code_stride_len");
    hp.perceiver_n_latents = require_u32(metadata.get(), "xtts.gpt.condition_num_latent");
    hp.cond_n_layers = require_u32(metadata.get(), "xtts.gpt.cond_enc_blocks");
    hp.cond_d_model = require_u32(metadata.get(), "xtts.gpt.cond_enc_dim");
    hp.perceiver_n_layers = require_u32(metadata.get(), "xtts.gpt.perc_layers");
    hp.text_pos_size = assets.gpt_source->require_metadata("gpt.text_pos.weight").shape.at(0);
    hp.mel_pos_size = assets.gpt_source->require_metadata("gpt.mel_pos.weight").shape.at(0);
    if (hp.d_model != 1024 || hp.cond_d_model != hp.d_model || hp.code_stride != 1024 ||
        hp.n_layers == 0 || hp.ff_dim == 0 || hp.perceiver_n_latents == 0) {
        throw std::runtime_error("XTTS v2 GPT dimensions do not match the supported decoder architecture");
    }
    const int64_t dim = hp.d_model;
    engine::assets::require_tensor_shape(*assets.gpt_source, "gpt.text_embedding.weight", {hp.text_vocab_size, dim});
    engine::assets::require_tensor_shape(*assets.gpt_source, "gpt.mel_embedding.weight", {hp.mel_vocab_size, dim});
    engine::assets::require_tensor_shape(*assets.gpt_source, "gpt.text_pos.weight", {hp.text_pos_size, dim});
    engine::assets::require_tensor_shape(*assets.gpt_source, "gpt.mel_pos.weight", {hp.mel_pos_size, dim});
    engine::assets::require_tensor_shape(*assets.gpt_source, "gpt.mel_head.weight", {hp.mel_vocab_size, dim});
    engine::assets::require_tensor_shape(*assets.gpt_source, "gpt.cond_enc.init.weight", {dim, 80, 1});
    engine::assets::require_tensor_shape(*assets.gpt_source, "gpt.perc.latents", {hp.perceiver_n_latents, dim});
    engine::assets::require_tensor_shape(*assets.vocoder_source, "hifigan_decoder.dec.conv_pre.weight", {512, dim, 7});
    if (!load_xtts_v2_tokenizer(&assets.vocab, assets.gpt_gguf_path)) {
        throw std::runtime_error("XTTS v2 tokenizer metadata is incomplete");
    }
    for (const auto & path : {assets.gpt_gguf_path, assets.vocoder_gguf_path}) {
        GgufMetadata info(gguf_init_from_file(path.string().c_str(), {true, nullptr}), gguf_free);
        if (!info) throw std::runtime_error("XTTS v2 cannot read metadata: " + path.string());
        const int64_t index = gguf_find_key(info.get(), "xtts.mel_stats");
        if (index < 0) continue;
        if (gguf_get_kv_type(info.get(), index) != GGUF_TYPE_ARRAY ||
            gguf_get_arr_type(info.get(), index) != GGUF_TYPE_FLOAT32 || gguf_get_arr_n(info.get(), index) != 80) {
            throw std::runtime_error("XTTS v2 mel normalization metadata must contain 80 floats");
        }
        const auto * values = static_cast<const float *>(gguf_get_arr_data(info.get(), index));
        assets.mel_stats.assign(values, values + 80);
        break;
    }
    if (assets.mel_stats.empty()) {
        for (const auto & source : {assets.gpt_source, assets.vocoder_source}) {
            if (source->has_tensor("mel_stats")) {
                assets.mel_stats = source->require_f32("mel_stats", {80});
                engine::debug::trace_log_scalar("xtts_v2.assets.mel_norms", source->source_path().string() + ":mel_stats");
                break;
            }
        }
    }
    if (assets.mel_stats.empty()) {
        const auto sidecar = assets.resources.model_directory() / "mel_norms.bin";
        const auto bytes = engine::io::read_binary_file(sidecar);
        if (bytes.size() != 80 * sizeof(float)) {
            throw std::runtime_error("XTTS v2 mel_norms.bin must contain 80 float32 values");
        }
        assets.mel_stats.resize(80);
        std::memcpy(assets.mel_stats.data(), bytes.data(), bytes.size());
        engine::debug::trace_log_scalar("xtts_v2.assets.mel_norms", sidecar.string());
    }
    if (std::any_of(assets.mel_stats.begin(), assets.mel_stats.end(), [](float value) {
            return !std::isfinite(value) || value == 0;
        })) {
        throw std::runtime_error("XTTS v2 mel normalization values must be finite and nonzero");
    }
    return result;
}
}  // namespace engine::models::xtts_v2
