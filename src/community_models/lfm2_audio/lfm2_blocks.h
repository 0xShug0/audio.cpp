#pragma once

// LFM2 hybrid layers shared by the backbone and the audio detokenizer, which
// is a small LFM2 model with the same GGUF layout (blk.N.*).
//
// Reference: Lfm2DecoderLayer / Lfm2ShortConv in transformers 4.56
// models/lfm2/modeling_lfm2.py.

#include "engine/community_models/lfm2_audio/assets.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/transformers/decoder.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

#include <cstdint>
#include <optional>
#include <vector>

namespace engine::community_models::lfm2_audio::lfm2_blocks {

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept { ggml_free(ctx); }
};

struct GgmlGallocrDeleter {
    void operator()(ggml_gallocr_t alloc) const noexcept { ggml_gallocr_free(alloc); }
};

struct GgmlBufferDeleter {
    void operator()(ggml_backend_buffer_t buffer) const noexcept { ggml_backend_buffer_free(buffer); }
};

struct ShortConvWeights {
    core::TensorValue in_proj;
    core::TensorValue out_proj;
    core::TensorValue kernel;  // [hidden, kernel_size]
};

// Every layer is x += op(norm(x)); x += ffn(norm(x)). Attention layers map
// onto the framework's decoder layer (QK-norm, NEOX RoPE, SwiGLU); the
// short-conv layers are built here.
struct LayerWeights {
    bool attention = false;
    modules::DecoderLayerWeights decoder;
    ShortConvWeights conv;
};

// Whether the backend has a get_rows kernel for this tensor type. ggml's CUDA
// get_rows has none for K-quants, which Liquid's Q4_0 packages use for token
// embeddings.
bool backend_gathers(ggml_backend_t backend, ggml_type type);

std::vector<LayerWeights> load_layers(core::BackendWeightStore & store, const assets::TensorSource & source, const Lfm2BackboneConfig & config);

modules::DecoderLayerConfig attention_layer_config(const Lfm2BackboneConfig & config, int64_t layer);

core::TensorValue rms_norm(
    core::ModuleBuildContext & ctx, const core::TensorValue & x, const modules::NormWeights & weights, const Lfm2BackboneConfig & config);

core::TensorValue contiguous(core::ModuleBuildContext & ctx, const core::TensorValue & x);

struct ShortConvInput {
    core::TensorValue gate;     // C, [1, steps, hidden]
    core::TensorValue conv_in;  // B * x, transposed to [1, hidden, steps]
};

// in_proj splits into B, C and x; the conv runs over B * x and its output is
// gated by C (transformers Lfm2ShortConv).
ShortConvInput short_conv_input(core::ModuleBuildContext & ctx, const core::TensorValue & normed, const ShortConvWeights & weights, int64_t d);

core::TensorValue short_conv_output(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & residual,
    const core::TensorValue & conv,
    const core::TensorValue & gate,
    const ShortConvWeights & weights,
    int64_t d);

core::TensorValue feed_forward(
    core::ModuleBuildContext & ctx, const core::TensorValue & x, const LayerWeights & weights, const Lfm2BackboneConfig & config);

core::TensorValue conv_kernel(core::ModuleBuildContext & ctx, const ShortConvWeights & weights, const Lfm2BackboneConfig & config);

// State a sequence pass leaves for continuing step by step: each attention
// layer's keys and values and each short-conv layer's last kernel - 1 inputs.
struct SequenceTaps {
    std::vector<ggml_tensor *> keys;
    std::vector<ggml_tensor *> values;
    std::vector<ggml_tensor *> conv_tails;
};

// All layers over x [1, steps, hidden], starting from empty state. Attention
// is causal, or follows `mask` ([steps, steps], 0 or -inf) when given.
core::TensorValue build_sequence(
    core::ModuleBuildContext & ctx,
    core::TensorValue x,
    const core::TensorValue & positions,
    const std::vector<LayerWeights> & layers,
    const Lfm2BackboneConfig & config,
    const std::optional<core::TensorValue> & mask = std::nullopt,
    SequenceTaps * taps = nullptr);

}  // namespace engine::community_models::lfm2_audio::lfm2_blocks
