#include "engine/community_models/zonos3/assets.h"

#include "engine/framework/assets/gguf_metadata.h"
#include "engine/framework/model_spec/package.h"

#include <gguf.h>

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::zonos3 {
namespace {

[[noreturn]] void fail(const std::string & key, const std::string & message) {
    throw std::runtime_error("zonos3: backbone GGUF: " + key + " " + message);
}

// GgufMetadata has no float-array getter; the condition edges are F32 arrays.
class GgufFloatArrays {
public:
    explicit GgufFloatArrays(const std::filesystem::path & path) {
        gguf_init_params params{};
        params.no_alloc = true;
        params.ctx = nullptr;
        ctx_.reset(gguf_init_from_file(path.string().c_str(), params));
        if (ctx_ == nullptr) throw std::runtime_error("zonos3: cannot read GGUF metadata: " + path.string());
    }

    std::vector<float> require(const std::string & key) const {
        const int64_t id = gguf_find_key(ctx_.get(), key.c_str());
        if (id < 0 || gguf_get_kv_type(ctx_.get(), id) != GGUF_TYPE_ARRAY ||
            gguf_get_arr_type(ctx_.get(), id) != GGUF_TYPE_FLOAT32) {
            fail(key, "is missing or not a float32 array");
        }
        const auto * data = static_cast<const float *>(gguf_get_arr_data(ctx_.get(), id));
        return std::vector<float>(data, data + gguf_get_arr_n(ctx_.get(), id));
    }

private:
    struct Deleter {
        void operator()(gguf_context * ctx) const noexcept { gguf_free(ctx); }
    };
    std::unique_ptr<gguf_context, Deleter> ctx_;
};

int require_u32(const assets::GgufMetadata & meta, const char * key) {
    return static_cast<int>(meta.require_int(key, 0, 0xFFFFFFFFll));
}

void require_equal(const assets::GgufMetadata & meta, const char * key, const char * value) {
    if (meta.require_string(key) != value) fail(key, std::string("must be '") + value + "'");
}

void require_value(const char * key, int64_t actual, int64_t expected) {
    if (actual != expected) fail(key, "must be " + std::to_string(expected) + " (got " + std::to_string(actual) + ")");
}

}  // namespace

std::shared_ptr<const Zonos3Assets> load_zonos3_assets(const std::filesystem::path & model_path) {
    auto out = std::make_shared<Zonos3Assets>();
    out->resources = model_spec::load_resource_bundle_for_family(model_path, "zonos3");
    out->backbone = out->resources.open_tensor_source("backbone");
    out->decoder = out->resources.open_tensor_source("decoder");
    out->speaker_encoder = out->resources.open_tensor_source("speaker_encoder");

    const auto & path = out->backbone->source_path();
    const assets::GgufMetadata meta(path);
    auto & hp = out->hparams;
    require_equal(meta, "general.architecture", "zonos3");
    require_equal(meta, "zonos3.attention.pattern", "gdn3_global1");
    require_equal(meta, "zonos3.residual.type", "full_attnres");
    require_equal(meta, "zonos3.codec", "mossdual");
    require_equal(meta, "zonos3.moe.balancing", "quantile");
    hp.n_layer = require_u32(meta, "zonos3.block_count");
    hp.n_embd = require_u32(meta, "zonos3.embedding_length");
    hp.head_dim = require_u32(meta, "zonos3.head_dim");
    hp.n_head = require_u32(meta, "zonos3.attention.head_count");
    hp.n_head_kv = require_u32(meta, "zonos3.attention.head_count_kv");
    hp.n_ff = require_u32(meta, "zonos3.feed_forward_length");
    hp.n_expert = require_u32(meta, "zonos3.expert_count");
    hp.n_ctx_train = require_u32(meta, "zonos3.context_length");
    hp.rope_freq_base = meta.require_f32("zonos3.rope.freq_base");
    hp.rms_eps = meta.require_f32("zonos3.attention.layer_norm_rms_epsilon");
    hp.qk_norm_eps = meta.require_f32("zonos3.attention.qk_norm_epsilon");
    hp.logit_softcap = meta.require_f32("zonos3.logit_softcap");
    hp.n_codebooks = require_u32(meta, "zonos3.n_codebooks");
    hp.codebook_size = require_u32(meta, "zonos3.codebook_size");
    hp.audio_vocab = require_u32(meta, "zonos3.audio_vocab");
    hp.eoa_id = require_u32(meta, "zonos3.eoa_id");
    hp.audio_pad_id = require_u32(meta, "zonos3.audio_pad_id");
    hp.text_vocab = require_u32(meta, "zonos3.text_vocab");
    hp.gdn_heads = require_u32(meta, "zonos3.gdn.head_count");
    hp.gdn_head_dim = require_u32(meta, "zonos3.gdn.head_dim");
    hp.gdn_conv_kernel = require_u32(meta, "zonos3.gdn.conv_kernel");
    hp.speaker_dim = require_u32(meta, "zonos3.speaker.embedding_length");
    hp.n_cond = require_u32(meta, "zonos3.condition.count");
    require_value("zonos3.tokenizer.offset", require_u32(meta, "zonos3.tokenizer.offset"), 4);
    require_value("zonos3.tokenizer.bos", require_u32(meta, "zonos3.tokenizer.bos"), 2);
    require_value("zonos3.tokenizer.eos", require_u32(meta, "zonos3.tokenizer.eos"), 3);
    if (meta.has("zonos3.span_end_id")) {
        fail("zonos3.span_end_id", "is set: speech-edit checkpoints are not supported");
    }
    hp.file_type = meta.has("general.file_type") ? require_u32(meta, "general.file_type") : 0;

    if (meta.has("zonos3.condition.bucket_counts")) {
        for (const auto count : meta.require_int_array("zonos3.condition.bucket_counts")) {
            hp.cond_bucket_counts.push_back(static_cast<int32_t>(count < 0 ? 0 : count));
        }
        if (static_cast<int>(hp.cond_bucket_counts.size()) != hp.n_cond) {
            fail("zonos3.condition.bucket_counts", "must have zonos3.condition.count entries");
        }
    } else {
        // Legacy checkpoints carry one count shared by every condition table.
        hp.cond_bucket_counts.assign(static_cast<size_t>(hp.n_cond), require_u32(meta, "zonos3.condition.bucket_count"));
    }
    hp.cond_names = meta.require_string_array("zonos3.condition.names");
    hp.cond_units = meta.require_string_array("zonos3.condition.units");
    if (hp.cond_names != std::vector<std::string>{"loudness", "fmax", "speaking_rate"}) {
        fail("zonos3.condition.names", "must be [loudness, fmax, speaking_rate]");
    }
    if (hp.n_cond != static_cast<int>(hp.cond_names.size()) || hp.cond_units.size() != hp.cond_names.size()) {
        fail("zonos3.condition.count", "does not match the condition names and units");
    }
    for (const auto count : hp.cond_bucket_counts) {
        if (count < 2 || count > 1024) fail("zonos3.condition.bucket_counts", "entries must be 2..1024");
    }
    if (hp.speaker_dim <= 0 || hp.speaker_dim > 65536) fail("zonos3.speaker.embedding_length", "must be 1..65536");
    const GgufFloatArrays arrays(path);
    for (size_t c = 0; c < hp.cond_names.size(); ++c) {
        const std::string key = "zonos3.condition.edges." + hp.cond_names[c];
        auto edges = arrays.require(key);
        bool valid = edges.size() == static_cast<size_t>(hp.cond_bucket_counts[c] - 1);
        for (size_t i = 0; valid && i < edges.size(); ++i) {
            valid = std::isfinite(edges[i]) && (i == 0 || edges[i] > edges[i - 1]);
        }
        if (!valid) fail(key, "must hold bucket_count - 1 finite, strictly increasing edges");
        hp.cond_edges.push_back(std::move(edges));
    }
    for (const auto value : meta.require_int_array("zonos3.silence_frame")) {
        hp.silence_frame.push_back(static_cast<int32_t>(value));
    }
    require_value("zonos3.n_codebooks", hp.n_codebooks, kZonos3Codebooks);
    require_value("zonos3.codebook_size", hp.codebook_size, kZonos3CodebookSize);
    require_value("zonos3.audio_vocab", hp.audio_vocab, kZonos3AudioVocab);
    require_value("zonos3.eoa_id", hp.eoa_id, kZonos3Eoa);
    require_value("zonos3.audio_pad_id", hp.audio_pad_id, kZonos3AudioPad);
    require_value("zonos3.text_vocab", hp.text_vocab, kZonos3TextPad);
    require_value("zonos3.gdn.head_dim", hp.gdn_head_dim, 128);
    require_value("zonos3.gdn.conv_kernel", hp.gdn_conv_kernel, 4);
    if (hp.silence_frame.size() != static_cast<size_t>(hp.n_codebooks)) {
        fail("zonos3.silence_frame", "must have n_codebooks entries");
    }
    for (const auto code : hp.silence_frame) {
        if (code < 0 || code >= hp.codebook_size) fail("zonos3.silence_frame", "entries must be 0..4095");
    }
    if (hp.n_layer <= 0 || hp.n_layer > 128) fail("zonos3.block_count", "must be 1..128");
    if (hp.n_embd <= 0 || hp.n_ff <= 0) fail("zonos3.embedding_length", "and feed_forward_length must be positive");
    if (hp.n_ctx_train <= 0 || hp.n_ctx_train > 32768) fail("zonos3.context_length", "must be 1..32768");
    if (hp.n_head <= 0 || hp.n_head_kv <= 0 || hp.n_head % hp.n_head_kv != 0) {
        fail("zonos3.attention.head_count", "must be a positive multiple of head_count_kv");
    }
    if (hp.head_dim <= 0 || hp.head_dim % 2 != 0) fail("zonos3.head_dim", "must be positive and even");
    if (hp.gdn_heads <= 0 || hp.gdn_heads > 64) fail("zonos3.gdn.head_count", "must be 1..64");
    if (!std::isfinite(hp.rms_eps) || hp.rms_eps <= 0 || !std::isfinite(hp.qk_norm_eps) || hp.qk_norm_eps <= 0 ||
        !std::isfinite(hp.rope_freq_base) || hp.rope_freq_base <= 0 || !std::isfinite(hp.logit_softcap)) {
        fail("zonos3.attention", "epsilons, rope base and softcap must be finite and positive");
    }
    hp.expert_used.clear();
    for (const auto used : meta.require_int_array("zonos3.expert_used_count_per_layer")) {
        hp.expert_used.push_back(static_cast<int32_t>(used));
    }
    if (hp.expert_used.size() != static_cast<size_t>(hp.n_layer)) {
        fail("zonos3.expert_used_count_per_layer", "must have block_count entries");
    }
    for (int layer = 0; layer < hp.n_layer; ++layer) {
        const int used = hp.expert_used[static_cast<size_t>(layer)];
        if ((layer == 0 && used != 0) || (layer > 0 && (used <= 0 || used > 2 || used > hp.n_expert))) {
            fail("zonos3.expert_used_count_per_layer", "is invalid for layer " + std::to_string(layer));
        }
    }
    const auto text_rows = out->backbone->require_metadata("text_embd.weight").shape;
    if (text_rows.size() != 2 || text_rows[0] < hp.text_vocab + 1) {
        fail("text_embd.weight", "must have at least text_vocab + 1 rows");
    }
    hp.text_extra_tokens = static_cast<int>(text_rows[0] - hp.text_vocab - 1);

    out->tokenizer_json = meta.require_string("zonos3.tokenizer.json");
    (void) meta.require_string("zonos3.text_tokenizer_spec");
    return out;
}

}  // namespace engine::models::zonos3
