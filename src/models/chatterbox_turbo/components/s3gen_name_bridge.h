#pragma once

// Chatterbox Turbo's S3Gen weights ship in a third-party GGUF (cstr/chatterbox-turbo-GGUF)
// whose tensor names are a consistent, decodable abbreviation of the names the existing
// chatterbox S3Gen loader code (src/models/chatterbox/s3gen_flow.cpp,
// src/framework/modules/vocoders/hift_vocoder.cpp) already expects and knows how to build
// ggml graphs for. Rather than reimplement the flow encoder/decoder/vocoder, this header
// name-translates so the *existing* loaders can run unmodified against the turbo GGUF.
//
// Mapping verified against the actual GGUF header of chatterbox-turbo-s3gen-q8_0.gguf
// (parsed offline with a throwaway GGUF reader), matched segment-by-segment against the
// literal prefixes s3gen_flow.cpp/hift_vocoder.cpp construct:
//
//   flow.encoder.encoders.N            -> fe.enc.N
//   flow.encoder.up_encoders.N         -> fe.ue.N
//   flow.encoder.up_layer               -> fe.ul
//   flow.encoder.up_embed               -> fe.uemb
//   flow.encoder.pre_lookahead_layer    -> fe.pla
//   flow.encoder.embed                  -> fe.embed
//   flow.encoder.after_norm             -> fe.an
//   flow.encoder_proj / spk_embed_affine_layer / input_embedding  (unchanged, top-level "flow.")
//   (within a conformer layer) norm_mha -> nmha, self_attn.linear_{q,k,v,out,pos} -> sa.l{q,k,v,o,p},
//     self_attn.pos_bias_{u,v} -> sa.pb{u,v}, norm_ff -> nff, feed_forward.w_1/2 -> ff.w_1/2
//   flow.decoder.estimator.down_blocks  -> fd.db
//   flow.decoder.estimator.mid_blocks   -> fd.mb
//   flow.decoder.estimator.up_blocks    -> fd.ub
//   flow.decoder.estimator.final_block  -> fd.fb
//   flow.decoder.estimator.final_proj   -> fd.fp
//   flow.decoder.estimator.time_mlp     -> fd.tm  (unchanged linear_1/linear_2 suffix)
//   flow.decoder.estimator.time_embed_mixer -> fd.tmx  (turbo-only meanflow tensor, no base equivalent)
//   (within a resnet block) blockN.block.0/2 -> bN.0/2, res_conv -> rc, mlp.1 unchanged
//   (within a transformer block) norm1/norm3 unchanged, attn1.to_q/k/v -> attn1.q/k/v,
//     attn1.to_out.0 -> attn1.o, ff.net.0.proj -> ff.up, ff.net.2 -> ff.down
//   (vocoder, HiftVocoderConfig.tensor_prefix="v.") conv_pre -> cpre, conv_post -> cpost,
//     ups.N unchanged, resblocks.N -> rb.N, source_downs.N -> sd.N, source_resblocks.N -> srb.N,
//     f0_predictor.condnet.N -> f0.cn.N, f0_predictor.classifier -> f0.cls, m_source.l_linear -> ms.ll
//     (within a resblock) convs1/2 -> c1/2, activations1/2 -> a1/2
//
// This is deliberately additive/self-contained to chatterbox_turbo: it does not modify the
// base chatterbox family's own tensor naming or loading code.

#include "engine/framework/assets/tensor_source.h"

#include <memory>
#include <string>

namespace engine::models::chatterbox_turbo {

// Wraps an already s3-prefix-stripped TensorSource (see open_tensor_source(path, "s3")) and
// rewrites incoming chatterbox-style names to the turbo GGUF's abbreviated names before
// delegating every lookup.
std::shared_ptr<const engine::assets::TensorSource> make_turbo_s3gen_flow_bridge(
    std::shared_ptr<const engine::assets::TensorSource> turbo_s3_source);

// Same idea for the HiFT vocoder; construct the shared HiftVocoderComponent with
// tensor_prefix = "v." against this bridge and weight_layout = Plain (the turbo GGUF already
// folds torch weight-norm parametrization into a plain "weight" tensor at conversion time).
std::shared_ptr<const engine::assets::TensorSource> make_turbo_vocoder_bridge(
    std::shared_ptr<const engine::assets::TensorSource> turbo_s3_source);

}  // namespace engine::models::chatterbox_turbo
