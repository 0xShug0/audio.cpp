#include "engine/community_models/zonos2/assets.h"

#include "engine/framework/assets/gguf_metadata.h"
#include "engine/framework/model_spec/package.h"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>

namespace engine::models::zonos2 {
namespace {

[[noreturn]] void fail(const std::string & file, const std::string & key, const std::string & message) {
    throw std::runtime_error("zonos2: " + file + " GGUF: " + key + " " + message);
}

int require_u32(const assets::GgufMetadata & meta, const char * key) {
    return static_cast<int>(meta.require_int(key, 0, 0xFFFFFFFFll));
}

void require_architecture(const assets::GgufMetadata & meta, const std::string & file, const char * value) {
    if (meta.require_string("general.architecture") != value) {
        fail(file, "general.architecture", std::string("must be '") + value + "'");
    }
}

void require_value(const std::string & file, const char * key, int64_t actual, int64_t expected) {
    if (actual != expected) {
        fail(file, key, "must be " + std::to_string(expected) + " (got " + std::to_string(actual) + ")");
    }
}

}  // namespace

std::shared_ptr<const Zonos2Assets> load_zonos2_assets(const std::filesystem::path & model_path) {
    auto out = std::make_shared<Zonos2Assets>();
    out->resources = model_spec::load_resource_bundle_for_family(model_path, "zonos2");
    out->backbone = out->resources.open_tensor_source("backbone");
    out->decoder = out->resources.open_tensor_source("decoder");
    out->speaker_encoder = out->resources.open_tensor_source("speaker_encoder");

    const std::string backbone = "backbone";
    const assets::GgufMetadata meta(out->backbone->source_path());
    auto & hp = out->hparams;
    require_architecture(meta, backbone, "zonos2");
    hp.n_layer = require_u32(meta, "zonos2.block_count");
    hp.n_embd = require_u32(meta, "zonos2.embedding_length");
    hp.head_dim = require_u32(meta, "zonos2.head_dim");
    hp.n_head = require_u32(meta, "zonos2.attention.head_count");
    hp.n_head_kv = require_u32(meta, "zonos2.attention.head_count_kv");
    hp.n_ff = require_u32(meta, "zonos2.feed_forward_length");
    hp.n_expert = require_u32(meta, "zonos2.expert_count");
    hp.n_ctx_train = require_u32(meta, "zonos2.context_length");
    hp.rope_freq_base = meta.require_f32("zonos2.rope.freq_base");
    hp.rms_eps = meta.require_f32("zonos2.attention.layer_norm_rms_epsilon");
    hp.qk_norm_eps = meta.require_f32("zonos2.attention.qk_norm_epsilon");
    hp.logit_softcap = meta.require_f32("zonos2.logit_softcap");
    hp.n_codebooks = require_u32(meta, "zonos2.n_codebooks");
    hp.codebook_size = require_u32(meta, "zonos2.codebook_size");
    hp.audio_vocab = require_u32(meta, "zonos2.audio_vocab");
    hp.eoa_id = require_u32(meta, "zonos2.eoa_id");
    hp.audio_pad_id = require_u32(meta, "zonos2.audio_pad_id");
    hp.text_vocab = require_u32(meta, "zonos2.text_vocab");
    hp.spk_dim = require_u32(meta, "zonos2.speaker.embedding_dim");
    hp.spk_lda_dim = require_u32(meta, "zonos2.speaker.lda_dim");

    require_value(backbone, "zonos2.n_codebooks", hp.n_codebooks, 9);
    require_value(backbone, "zonos2.audio_vocab", hp.audio_vocab, static_cast<int64_t>(hp.codebook_size) + 2);
    require_value(backbone, "zonos2.eoa_id", hp.eoa_id, hp.codebook_size);
    require_value(backbone, "zonos2.audio_pad_id", hp.audio_pad_id, static_cast<int64_t>(hp.codebook_size) + 1);
    // The hardcoded conditioning layout (8 rate + 60 quality + 2 background + 1 accurate ids) assumes this vocab.
    require_value(backbone, "zonos2.text_vocab", hp.text_vocab, 519);
    if (hp.n_layer <= 0 || hp.n_layer > 128) fail(backbone, "zonos2.block_count", "must be 1..128");
    if (hp.n_embd <= 0 || hp.n_ff <= 0) fail(backbone, "zonos2.embedding_length", "and feed_forward_length must be positive");
    if (hp.n_ctx_train <= 0 || hp.n_ctx_train > 32768) fail(backbone, "zonos2.context_length", "must be 1..32768");
    if (hp.n_head <= 0 || hp.n_head_kv <= 0 || hp.n_head % hp.n_head_kv != 0) {
        fail(backbone, "zonos2.attention.head_count", "must be a positive multiple of head_count_kv");
    }
    if (hp.head_dim <= 0 || hp.head_dim % 2 != 0) fail(backbone, "zonos2.head_dim", "must be positive and even");
    if (hp.n_expert <= 0 || hp.n_expert > 256) fail(backbone, "zonos2.expert_count", "must be 1..256");
    if (hp.spk_dim <= 0 || hp.spk_lda_dim <= 0) fail(backbone, "zonos2.speaker", "dimensions must be positive");
    if (!std::isfinite(hp.rms_eps) || hp.rms_eps <= 0 || !std::isfinite(hp.qk_norm_eps) || hp.qk_norm_eps <= 0 ||
        !std::isfinite(hp.rope_freq_base) || hp.rope_freq_base <= 0 || !std::isfinite(hp.logit_softcap)) {
        fail(backbone, "zonos2.attention", "epsilons, rope base and softcap must be finite and positive");
    }
    for (const auto used : meta.require_int_array("zonos2.expert_used_count_per_layer")) {
        if (used < 0 || used > hp.n_expert) fail(backbone, "zonos2.expert_used_count_per_layer", "entries must be 0..expert_count");
        hp.expert_used.push_back(static_cast<int32_t>(used));
    }
    if (hp.expert_used.size() != static_cast<size_t>(hp.n_layer)) {
        fail(backbone, "zonos2.expert_used_count_per_layer", "must have block_count entries");
    }

    const std::string decoder = "decoder";
    const assets::GgufMetadata dac(out->decoder->source_path());
    require_architecture(dac, decoder, "dac");
    require_value(decoder, "dac.n_codebooks", require_u32(dac, "dac.n_codebooks"), hp.n_codebooks);
    require_value(decoder, "dac.codebook_size", require_u32(dac, "dac.codebook_size"), hp.codebook_size);
    require_value(decoder, "dac.sample_rate", require_u32(dac, "dac.sample_rate"), 44100);
    require_value(decoder, "dac.hop_length", require_u32(dac, "dac.hop_length"), 512);

    const std::string speaker = "speaker encoder";
    const assets::GgufMetadata spk(out->speaker_encoder->source_path());
    require_architecture(spk, speaker, "ecapa-tdnn");
    require_value(speaker, "ecapa-tdnn.enc_dim", require_u32(spk, "ecapa-tdnn.enc_dim"), hp.spk_dim);
    require_value(speaker, "ecapa-tdnn.sample_rate", require_u32(spk, "ecapa-tdnn.sample_rate"), 24000);
    require_value(speaker, "ecapa-tdnn.mel_dim", require_u32(spk, "ecapa-tdnn.mel_dim"), 128);
    return out;
}

}  // namespace engine::models::zonos2
