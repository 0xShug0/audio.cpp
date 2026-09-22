#pragma once

#include "engine/community_models/zipvoice/weights.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

#include <cstdint>
#include <vector>
#include <utility>

namespace engine::models::zipvoice {

// Raw-ggml graph builders for the two ZipVoice networks.
//
// Physical layouts (ne0 = fastest):
//   activations [C, T, B]; attention weights [src, tgt, H, B]
// All softmaxes run over ne0 (= keys).
//
// Memory model: leaves + constants live in a dedicated tensor context with
// their own backend buffer (ggml_backend_alloc_ctx_tensors) so the graph
// arena never aliases them; per-call values are uploaded into that buffer.

struct ZipVoiceGraphResources {
    ggml_context * ctx = nullptr;
    ggml_context * tensor_ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_backend_t backend = nullptr;
    void * gallocr = nullptr;
    ZipVoiceGraphResources() = default;
    ~ZipVoiceGraphResources();
    ZipVoiceGraphResources(const ZipVoiceGraphResources &) = delete;
    ZipVoiceGraphResources & operator=(const ZipVoiceGraphResources &) = delete;
    ZipVoiceGraphResources(ZipVoiceGraphResources && other) noexcept
        : ctx(std::exchange(other.ctx, nullptr)), tensor_ctx(std::exchange(other.tensor_ctx, nullptr)),
          buffer(std::exchange(other.buffer, nullptr)), graph(std::exchange(other.graph, nullptr)),
          backend(std::exchange(other.backend, nullptr)), gallocr(std::exchange(other.gallocr, nullptr)) {}
};

struct FmDecoderGraph : ZipVoiceGraphResources {
    ggml_tensor * x_cat = nullptr;      // leaf [3F, T, B]
    ggml_tensor * time_emb = nullptr;   // leaf [time_embed_dim]
    ggml_tensor * guidance_emb = nullptr;  // leaf [time_embed_dim] (distill)
    ggml_tensor * pad_bias[8] = {};     // leaves [T_s, 1, 1, 1] (0/-1000)
    ggml_tensor * conv_gate[8] = {};    // leaves [1, T_s, 1] (0/1)
    ggml_tensor * output = nullptr;     // [F, T, B]
    int64_t T = 0;
    int64_t B = 1;
    bool cuda = false;
};

struct TextEncoderGraph : ZipVoiceGraphResources {
    ggml_tensor * token_ids = nullptr;  // leaf [S] i32
    ggml_tensor * pad_bias[1] = {};     // [S, 1, 1, 1]
    ggml_tensor * conv_gate[1] = {};    // [1, S, 1]
    ggml_tensor * output = nullptr;     // [feat_dim, S]
    std::vector<ggml_tensor *> layer_taps;  // per-layer outputs [C, S]
    std::vector<ggml_tensor *> stage_taps;  // first-layer submodule taps
    int64_t S = 0;
    bool cuda = false;
};

// Builds the flow-matching decoder graph. `with_guidance` wires the distill
// guidance-scale embedding input (ignored for the base model).
FmDecoderGraph build_fm_decoder_graph(
    const ZipVoiceWeights & weights,
    const ZipVoiceConfig & config,
    int64_t T,
    int64_t B,
    bool with_guidance,
    bool cuda_backend,
    ggml_backend_t backend);

// Builds the text encoder graph (single stack, no time embedding).
TextEncoderGraph build_text_encoder_graph(
    const ZipVoiceWeights & weights,
    const ZipVoiceConfig & config,
    int64_t S,
    bool cuda_backend,
    ggml_backend_t backend);

}  // namespace engine::models::zipvoice
