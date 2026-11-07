#include "engine/models/moss_soundeffect/assets.h"

#include "engine/framework/io/config.h"
#include "engine/framework/model_spec/package.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/weight_binding.h"

#include <stdexcept>

namespace engine::models::moss_soundeffect {

std::shared_ptr<const Assets> load_assets(const std::filesystem::path & path) {
    auto a = std::make_shared<Assets>();
    a->resources = model_spec::load_resource_bundle_for_family(path, "moss_soundeffect");
    const auto config = a->resources.parse_json("config");
    a->sample_rate = io::json::require_i32(config, "sample_rate");
    a->max_duration_sec = io::json::require_i32(config, "max_duration_sec");
    const auto & dit = config.require("transformer");
    a->dit.dim = io::json::require_i64(dit, "dim");
    a->dit.ffn_dim = io::json::require_i64(dit, "ffn_dim");
    a->dit.heads = io::json::require_i64(dit, "num_heads");
    a->dit.layers = io::json::require_i64(dit, "num_layers");
    a->dit.latent_dim = io::json::require_i64(dit, "in_dim");
    a->dit.text_dim = io::json::require_i64(dit, "text_dim");
    a->dit.freq_dim = io::json::require_i64(dit, "freq_dim");
    a->dit.eps = io::json::require_f32(dit, "eps");
    const auto & patch = dit.require("patch_size").as_array();
    if (patch.size() != 1 || io::json::require_bool(dit, "has_image_input") ||
        io::json::require_string(dit, "vae_type") != "dac" ||
        io::json::require_i64(dit, "out_dim") != a->dit.latent_dim) {
        throw std::runtime_error("MOSS-SoundEffect v2 requires a one-dimensional DAC audio DiT");
    }
    a->dit.patch_size = patch.front().as_i64();
    const auto & text = config.require("text_encoder");
    auto & t = a->text;
    t.hidden_size = io::json::require_i64(text, "hidden_size");
    t.intermediate_size = io::json::require_i64(text, "intermediate_size");
    t.layers = io::json::require_i64(text, "num_hidden_layers");
    t.num_attention_heads = io::json::require_i64(text, "num_attention_heads");
    t.num_key_value_heads = io::json::require_i64(text, "num_key_value_heads");
    t.head_dim = io::json::require_i64(text, "head_dim");
    t.rms_norm_eps = io::json::require_f32(text, "rms_norm_eps");
    t.rope_theta = io::json::require_f32(text, "rope_theta");
    a->vocab_size = io::json::require_i64(text, "vocab_size");
    const auto & vae = config.require("vae");
    a->vae.latent_dim = io::json::require_i64(vae, "latent_dim");
    a->vae.decoder_dim = io::json::require_i64(vae, "decoder_dim");
    a->vae.hop_length = 1;
    for (const auto & rate : vae.require("decoder_rates").as_array()) {
        io::require_positive(rate.as_i64(), "DAC decoder rate");
        a->vae.rates.push_back(rate.as_i64());
        a->vae.hop_length *= rate.as_i64();
    }
    for (const int64_t value : {int64_t(a->sample_rate), int64_t(a->max_duration_sec),
             a->dit.dim, a->dit.ffn_dim, a->dit.heads, a->dit.layers, a->dit.latent_dim,
             a->dit.text_dim, a->dit.freq_dim, a->dit.patch_size, t.hidden_size,
             t.intermediate_size, t.layers, t.num_attention_heads, t.num_key_value_heads,
             t.head_dim, a->vocab_size, a->vae.latent_dim, a->vae.decoder_dim}) {
        io::require_positive(value, "MOSS-SoundEffect v2 configuration dimension");
    }
    int64_t decoder_channels = a->vae.decoder_dim;
    for (size_t stage = 0; stage < a->vae.rates.size(); ++stage) {
        if (decoder_channels % 2 || decoder_channels < 2)
            throw std::runtime_error("DAC decoder channels must halve at each stage");
        decoder_channels /= 2;
    }
    if (a->vae.rates.empty() || a->dit.dim % a->dit.heads || a->dit.freq_dim % 2 ||
        a->dit.text_dim != t.hidden_size || a->dit.latent_dim != a->vae.latent_dim ||
        t.num_attention_heads % t.num_key_value_heads ||
        int64_t(a->sample_rate) * a->max_duration_sec % (a->vae.hop_length * a->dit.patch_size)) {
        throw std::runtime_error("MOSS-SoundEffect v2 component dimensions disagree");
    }
    a->text_weights = a->resources.open_tensor_source("text_encoder");
    a->dit_weights = a->resources.open_tensor_source("dit");
    a->vae_weights = a->resources.open_tensor_source("vae");
    return a;
}

modules::CausalDecoderRuntimeWeights load_text_weights(
    const Assets & a, core::BackendWeightStore & store, assets::TensorStorageType storage) {
    namespace binding = modules::binding;
    const auto & source = *a.text_weights;
    const auto & c = a.text;
    modules::CausalDecoderRuntimeWeights out;
    out.token_embedding = store.load_tensor(source, "model.embed_tokens.weight", storage, {a.vocab_size, c.hidden_size});
    out.final_norm = binding::norm_weight_from_source(store, source, "model.norm", c.hidden_size);
    for (int64_t layer = 0; layer < c.layers; ++layer) {
        const auto prefix = "model.layers." + std::to_string(layer);
        modules::DecoderLayerWeights w;
        w.input_norm = binding::norm_weight_from_source(store, source, prefix + ".input_layernorm", c.hidden_size);
        w.post_norm = binding::norm_weight_from_source(store, source, prefix + ".post_attention_layernorm", c.hidden_size);
        w.q_norm = binding::norm_weight_from_source(store, source, prefix + ".self_attn.q_norm", c.head_dim);
        w.k_norm = binding::norm_weight_from_source(store, source, prefix + ".self_attn.k_norm", c.head_dim);
        w.self_attention.q_weight = store.load_tensor(source, prefix + ".self_attn.q_proj.weight", storage,
            {c.num_attention_heads * c.head_dim, c.hidden_size});
        w.self_attention.k_weight = store.load_tensor(source, prefix + ".self_attn.k_proj.weight", storage,
            {c.num_key_value_heads * c.head_dim, c.hidden_size});
        w.self_attention.v_weight = store.load_tensor(source, prefix + ".self_attn.v_proj.weight", storage,
            {c.num_key_value_heads * c.head_dim, c.hidden_size});
        w.self_attention.out_weight = store.load_tensor(source, prefix + ".self_attn.o_proj.weight", storage,
            {c.hidden_size, c.num_attention_heads * c.head_dim});
        w.mlp.gate_proj = binding::linear_from_source(store, source, prefix + ".mlp.gate_proj", storage,
            c.intermediate_size, c.hidden_size, false);
        w.mlp.up_proj = binding::linear_from_source(store, source, prefix + ".mlp.up_proj", storage,
            c.intermediate_size, c.hidden_size, false);
        w.mlp.down_proj = binding::linear_from_source(store, source, prefix + ".mlp.down_proj", storage,
            c.hidden_size, c.intermediate_size, false);
        out.stack.layers.push_back(std::move(w));
    }
    return out;
}

}  // namespace engine::models::moss_soundeffect
