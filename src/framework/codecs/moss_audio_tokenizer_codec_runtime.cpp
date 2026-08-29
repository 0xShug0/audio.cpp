#include "engine/framework/codecs/moss_audio_tokenizer_codec_runtime.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/module.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace engine::codecs::codec_detail {

namespace modules = engine::modules;
namespace binding = engine::modules::binding;

inline constexpr float kMaskedAttentionBias = std::numeric_limits<float>::lowest();
inline constexpr int64_t kCodeDim = 768;
inline constexpr int64_t kSamplesPerFrame = 3840;  // downsample_rate (per interleaved stream frame)
inline constexpr float kRopeTheta = 10000.0F;
inline constexpr float kLayerNormEps = 1.0e-5F;

// One ProjectedTransformer stage. `patch` is the reshape factor applied to the
// stage (after the transformer for the decoder, before it for the encoder);
// `context` is the local-attention window in tokens at that stage's frame rate.
struct TransformerSpec {
    int64_t input_dim;
    int64_t output_dim;
    int64_t d_model;
    int64_t num_heads;
    int64_t num_layers;
    int64_t intermediate_size;
    int64_t context;
    int64_t patch;
};

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct GgmlGallocrDeleter {
    void operator()(ggml_gallocr_t alloc) const noexcept {
        if (alloc != nullptr) {
            ggml_gallocr_free(alloc);
        }
    }
};

struct LayerWeights {
    core::TensorValue norm1_w;
    core::TensorValue norm1_b;
    core::TensorValue in_proj;   // fused qkv [3 * d_model, d_model]
    core::TensorValue out_proj;  // [d_model, d_model]
    core::TensorValue norm2_w;
    core::TensorValue norm2_b;
    core::TensorValue fc1;  // [intermediate_size, d_model]
    core::TensorValue fc2;  // [d_model, intermediate_size]
    core::TensorValue layer_scale1;  // [d_model]
    core::TensorValue layer_scale2;  // [d_model]
};

struct TransformerWeights {
    TransformerSpec spec;
    core::TensorValue input_proj;   // [d_model, input_dim]
    core::TensorValue output_proj;  // [output_dim, d_model]
    std::vector<LayerWeights> layers;
};

struct AttentionWindow {
    int64_t query_start;
    int64_t query_steps;
    int64_t key_start;
    int64_t key_steps;
    core::TensorValue mask;
};

class CodecWeights {
public:
    explicit CodecWeights(const assets::TensorSource & source) : source_(source) {}

    const assets::TensorSource & source_for(const std::string & name) const {
        if (source_.has_tensor(name)) {
            return source_;
        }
        throw std::runtime_error("MOSS codec tensor not found: " + name);
    }

    bool has(const std::string & name) const noexcept { return source_.has_tensor(name); }

private:
    const assets::TensorSource & source_;
};

// Loads one ProjectedTransformer's weights. `stack_prefix` is "decoder" or
// "encoder"; `module_index` is the module's position in that ModuleList.
inline TransformerWeights load_transformer(
    core::BackendWeightStore & store,
    const CodecWeights & codec_weights,
    const TransformerSpec & spec,
    const std::string & stack_prefix,
    int64_t module_index) {
    const std::string prefix = stack_prefix + "." + std::to_string(module_index);
    const auto load = [&](const std::string & name, std::initializer_list<int64_t> shape) {
        return store.load_tensor(codec_weights.source_for(name), name, assets::TensorStorageType::F32, shape);
    };
    const auto load_f32 = [&](const std::string & name, std::initializer_list<int64_t> shape) {
        return store.load_f32_tensor(codec_weights.source_for(name), name, shape);
    };

    TransformerWeights weights;
    weights.spec = spec;
    weights.input_proj = load(prefix + ".input_proj.weight", {spec.d_model, spec.input_dim});
    // Upstream's ProjectedTransformer only creates an output projection when the stage
    // changes width. v2 ships one on every module; v1 leaves it out wherever
    // output_dimension already equals d_model, so treat it as optional and fall through to
    // the identity in that case.
    const std::string output_proj_name = prefix + ".output_proj.weight";
    if (codec_weights.has(output_proj_name)) {
        weights.output_proj = load(output_proj_name, {spec.output_dim, spec.d_model});
    } else if (spec.output_dim != spec.d_model) {
        throw std::runtime_error(
            "MOSS codec stage " + prefix + " changes width but carries no output projection");
    }
    weights.layers.reserve(static_cast<size_t>(spec.num_layers));
    for (int64_t layer = 0; layer < spec.num_layers; ++layer) {
        const std::string lp = prefix + ".transformer.layers." + std::to_string(layer);
        LayerWeights w;
        w.norm1_w = load_f32(lp + ".norm1.weight", {spec.d_model});
        w.norm1_b = load_f32(lp + ".norm1.bias", {spec.d_model});
        // v2 stores one attention projection per layer; v1 keeps them in an indexed
        // ModuleList (`in_projs.0`). Same tensor either way.
        // v1 names the feed-forward layers linear1/linear2 where v2 uses an nn.Sequential.
        const auto ffn_name = [&](const std::string & sequential, const std::string & named) {
            return codec_weights.has(lp + sequential) ? lp + sequential : lp + named;
        };
        const auto attention_name = [&](const std::string & single, const std::string & indexed) {
            return codec_weights.has(lp + single) ? lp + single : lp + indexed;
        };
        w.in_proj = load(
            attention_name(".self_attn.in_proj.weight", ".self_attn.in_projs.0.weight"),
            {3 * spec.d_model, spec.d_model});
        w.out_proj = load(
            attention_name(".self_attn.out_proj.weight", ".self_attn.out_projs.0.weight"),
            {spec.d_model, spec.d_model});
        w.norm2_w = load_f32(lp + ".norm2.weight", {spec.d_model});
        w.norm2_b = load_f32(lp + ".norm2.bias", {spec.d_model});
        w.fc1 = load(ffn_name(".ffn.0.weight", ".linear1.weight"), {spec.intermediate_size, spec.d_model});
        w.fc2 = load(ffn_name(".ffn.2.weight", ".linear2.weight"), {spec.d_model, spec.intermediate_size});
        w.layer_scale1 = load_f32(lp + ".layer_scale_1.scale", {spec.d_model});
        w.layer_scale2 = load_f32(lp + ".layer_scale_2.scale", {spec.d_model});
        weights.layers.push_back(std::move(w));
    }
    return weights;
}

inline core::TensorValue attention(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & q_heads,
    const core::TensorValue & k_heads,
    const core::TensorValue & v_heads,
    int64_t dim,
    const core::TensorValue & mask) {
    const modules::MatMulModule matmul;
    auto scores = matmul.build(
        ctx,
        q_heads,
        modules::TransposeModule({{0, 1, 3, 2}, k_heads.shape.rank}).build(ctx, k_heads));
    scores = core::ensure_backend_addressable_layout(ctx, scores);
    auto attn = core::wrap_tensor(
        ggml_soft_max_ext(
            ctx.ggml,
            scores.tensor,
            mask.tensor,
            1.0F / std::sqrt(static_cast<float>(dim)),
            0.0F),
        scores.shape,
        GGML_TYPE_F32);
    return matmul.build(ctx, attn, v_heads);
}

inline core::TensorValue windowed_attention(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & q_heads,
    const core::TensorValue & k_heads,
    const core::TensorValue & v_heads,
    int64_t dim,
    const std::vector<AttentionWindow> & windows) {
    if (windows.empty()) {
        throw std::runtime_error("MOSS codec windowed attention requires at least one window");
    }
    core::TensorValue merged;
    for (const auto & window : windows) {
        auto q_slice = modules::SliceModule({2, window.query_start, window.query_steps}).build(ctx, q_heads);
        auto k_slice = modules::SliceModule({2, window.key_start, window.key_steps}).build(ctx, k_heads);
        auto v_slice = modules::SliceModule({2, window.key_start, window.key_steps}).build(ctx, v_heads);
        auto part = attention(ctx, q_slice, k_slice, v_slice, dim, window.mask);
        merged = merged.valid() ? modules::ConcatModule({2}).build(ctx, merged, part) : part;
    }
    return merged;
}

inline core::TensorValue transformer_layer(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const LayerWeights & weights,
    const TransformerSpec & spec,
    const core::TensorValue & positions,
    const core::TensorValue & mask,
    const std::vector<AttentionWindow> * windows,
    int64_t steps) {
    const int64_t dim = spec.d_model / spec.num_heads;
    const modules::LayerNormModule norm({spec.d_model, kLayerNormEps, true, true});

    auto normed = norm.build(ctx, input, binding::norm_data(ctx, weights.norm1_w, weights.norm1_b));
    auto qkv = modules::LinearModule(binding::linear_config(spec.d_model, 3 * spec.d_model, false))
                   .build(ctx, normed, binding::linear_data(ctx, weights.in_proj));

    auto q = core::ensure_backend_addressable_layout(
        ctx, modules::SliceModule({2, 0, spec.d_model}).build(ctx, qkv));
    auto k = core::ensure_backend_addressable_layout(
        ctx, modules::SliceModule({2, spec.d_model, spec.d_model}).build(ctx, qkv));
    auto v = core::ensure_backend_addressable_layout(
        ctx, modules::SliceModule({2, 2 * spec.d_model, spec.d_model}).build(ctx, qkv));

    q = modules::ReshapeModule({
        core::TensorShape::from_dims({q.shape.dims[0], q.shape.dims[1], spec.num_heads, dim}),
    }).build(ctx, q);
    k = modules::ReshapeModule({
        core::TensorShape::from_dims({k.shape.dims[0], k.shape.dims[1], spec.num_heads, dim}),
    }).build(ctx, k);
    v = modules::ReshapeModule({
        core::TensorShape::from_dims({v.shape.dims[0], v.shape.dims[1], spec.num_heads, dim}),
    }).build(ctx, v);
    q = modules::RoPEModule({dim, GGML_ROPE_TYPE_NORMAL, kRopeTheta}).build(ctx, q, positions);
    k = modules::RoPEModule({dim, GGML_ROPE_TYPE_NORMAL, kRopeTheta}).build(ctx, k, positions);

    auto q_heads = modules::TransposeModule({{0, 2, 1, 3}, q.shape.rank}).build(ctx, q);
    auto k_heads = modules::TransposeModule({{0, 2, 1, 3}, k.shape.rank}).build(ctx, k);
    auto v_heads = modules::TransposeModule({{0, 2, 1, 3}, v.shape.rank}).build(ctx, v);
    auto context = windows == nullptr ? attention(ctx, q_heads, k_heads, v_heads, dim, mask)
                                      : windowed_attention(ctx, q_heads, k_heads, v_heads, dim, *windows);
    context = modules::TransposeModule({{0, 2, 1, 3}, context.shape.rank}).build(ctx, context);
    context = core::ensure_backend_addressable_layout(ctx, context);
    context = modules::ReshapeModule({
        core::TensorShape::from_dims({1, steps, spec.d_model}),
    }).build(ctx, context);
    auto attn_out = modules::LinearModule(binding::linear_config(spec.d_model, spec.d_model, false))
                        .build(ctx, context, binding::linear_data(ctx, weights.out_proj));
    auto layer_scale1 = modules::ReshapeModule({
        core::TensorShape::from_dims({1, 1, spec.d_model}),
    }).build(ctx, weights.layer_scale1);
    layer_scale1 = modules::RepeatModule({attn_out.shape}).build(ctx, layer_scale1);
    attn_out = modules::MulModule{}.build(ctx, attn_out, layer_scale1);
    auto x = modules::AddModule{}.build(ctx, input, attn_out);

    auto ff_in = norm.build(ctx, x, binding::norm_data(ctx, weights.norm2_w, weights.norm2_b));
    auto ff = modules::LinearModule(binding::linear_config(spec.d_model, spec.intermediate_size, false))
                  .build(ctx, ff_in, binding::linear_data(ctx, weights.fc1));
    ff = modules::GeluModule({modules::GeluApproximation::ExactErf}).build(ctx, ff);
    ff = modules::LinearModule(binding::linear_config(spec.intermediate_size, spec.d_model, false))
             .build(ctx, ff, binding::linear_data(ctx, weights.fc2));
    auto layer_scale2 = modules::ReshapeModule({
        core::TensorShape::from_dims({1, 1, spec.d_model}),
    }).build(ctx, weights.layer_scale2);
    layer_scale2 = modules::RepeatModule({ff.shape}).build(ctx, layer_scale2);
    ff = modules::MulModule{}.build(ctx, ff, layer_scale2);
    return modules::AddModule{}.build(ctx, x, ff);
}

// ProjectedTransformer: input projection -> transformer stack -> output
// projection. Input/output are [1, steps, channels] (feature-last).
inline core::TensorValue run_transformer(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const TransformerWeights & weights,
    const core::TensorValue & positions,
    const core::TensorValue & mask,
    int64_t steps,
    const std::vector<AttentionWindow> * windows = nullptr) {
    const auto & spec = weights.spec;
    auto x = modules::LinearModule(binding::linear_config(spec.input_dim, spec.d_model, false))
                 .build(ctx, input, binding::linear_data(ctx, weights.input_proj));
    for (const auto & layer : weights.layers) {
        x = transformer_layer(ctx, x, layer, spec, positions, mask, windows, steps);
    }
    if (!weights.output_proj.valid()) {
        return x;
    }
    return modules::LinearModule(binding::linear_config(spec.d_model, spec.output_dim, false))
        .build(ctx, x, binding::linear_data(ctx, weights.output_proj));
}

inline std::vector<float> causal_context_mask(int64_t steps, int64_t context) {
    std::vector<float> mask(static_cast<size_t>(steps * steps), kMaskedAttentionBias);
#ifdef _OPENMP
#pragma omp parallel for if(steps * steps >= 4096)
#endif
    for (int64_t query = 0; query < steps; ++query) {
        for (int64_t key = 0; key <= query; ++key) {
            if (query - key < context) {
                mask[static_cast<size_t>(query * steps + key)] = 0.0F;
            }
        }
    }
    return mask;
}

inline std::vector<float> causal_context_mask_window(
    int64_t query_start,
    int64_t query_steps,
    int64_t key_start,
    int64_t key_steps,
    int64_t context) {
    std::vector<float> mask(static_cast<size_t>(query_steps * key_steps), kMaskedAttentionBias);
#ifdef _OPENMP
#pragma omp parallel for if(query_steps * key_steps >= 4096)
#endif
    for (int64_t q = 0; q < query_steps; ++q) {
        const int64_t query = query_start + q;
        for (int64_t k = 0; k < key_steps; ++k) {
            const int64_t key = key_start + k;
            if (key <= query && query - key < context) {
                mask[static_cast<size_t>(q * key_steps + k)] = 0.0F;
            }
        }
    }
    return mask;
}

}  // namespace engine::codecs::codec_detail

namespace engine::codecs {

MossTokenRowBuilder::MossTokenRowBuilder(int64_t num_codebooks, int32_t audio_pad_token_id)
    : num_codebooks_(num_codebooks),
      audio_pad_token_id_(audio_pad_token_id) {
    if (num_codebooks_ <= 0) {
        throw std::runtime_error("MOSS token row builder requires a positive codebook count");
    }
}

void MossTokenRowBuilder::push_text_token(int32_t token_id) {
    rows_.text_tokens.push_back(token_id);
    rows_.audio_codes.insert(rows_.audio_codes.end(), static_cast<size_t>(num_codebooks_), audio_pad_token_id_);
}

void MossTokenRowBuilder::push_text_tokens(const std::vector<int32_t> & token_ids) {
    for (const int32_t token_id : token_ids) {
        push_text_token(token_id);
    }
}

void MossTokenRowBuilder::push_audio_row(int32_t text_slot_token_id, const int32_t * codes, int64_t num_codebooks) {
    if (num_codebooks != num_codebooks_) {
        throw std::runtime_error("MOSS audio row codebook count mismatch");
    }
    if (codes == nullptr) {
        throw std::runtime_error("MOSS audio row codes are missing");
    }
    rows_.text_tokens.push_back(text_slot_token_id);
    rows_.audio_codes.insert(rows_.audio_codes.end(), codes, codes + num_codebooks);
}

void MossTokenRowBuilder::push_audio_row(
    int32_t text_slot_token_id,
    const std::vector<std::vector<int32_t>> & codes,
    int64_t frame) {
    if (static_cast<int64_t>(codes.size()) != num_codebooks_) {
        throw std::runtime_error("MOSS audio row codebook count mismatch");
    }
    rows_.text_tokens.push_back(text_slot_token_id);
    for (int64_t codebook = 0; codebook < num_codebooks_; ++codebook) {
        const auto & channel = codes[static_cast<size_t>(codebook)];
        if (frame < 0 || static_cast<size_t>(frame) >= channel.size()) {
            throw std::runtime_error("MOSS audio row frame index is out of range");
        }
        rows_.audio_codes.push_back(channel[static_cast<size_t>(frame)]);
    }
}

MossTokenRows MossTokenRowBuilder::finish() {
    if (rows_.text_tokens.empty()) {
        throw std::runtime_error("MOSS token rows must not be empty");
    }
    if (static_cast<int64_t>(rows_.audio_codes.size()) !=
        static_cast<int64_t>(rows_.text_tokens.size()) * num_codebooks_) {
        throw std::runtime_error("MOSS token rows audio code shape mismatch");
    }
    return std::move(rows_);
}

// Dequantizes MOSS-Audio-Tokenizer-v2 codes (RLFQ) into the codec's continuous
// latent, i.e. the input to the codec decoder stack. Codes are the
// [num_quantizers, steps] matrix produced by generation; the returned latent is
// [code_dim, steps] row-major (channel-major), matching the Python
// quantizer.decode_codes output [1, code_dim, steps]. This is the plain-linear
// dequant path (per-codebook embedding lookup -> weight-normalized 1x1 conv ->
// residual sum -> output projection); the transformer decoder is a later phase.
class MossAudioTokenizerQuantizer {
public:
    MossAudioTokenizerQuantizer(
        const assets::TensorSource & source,
        int64_t num_quantizers,
        MossAudioTokenizerQuantizerConfig config = moss_audio_tokenizer_v2_config().quantizer);

    int64_t code_dim() const noexcept { return code_dim_; }
    int64_t num_quantizers() const noexcept { return num_quantizers_; }

    std::vector<float> decode(const std::vector<std::vector<int32_t>> & codes) const;

    // Quantizes the encoder latent into codes: the inverse of decode(). `hidden`
    // is [frames, code_dim] feature-last (row-major: frame * code_dim + channel),
    // matching the codec encoder's output. Mirrors the RLFQ forward pass
    // (input_proj -> per-quantizer in_proj -> L2-normalized nearest code ->
    // residual subtraction) and returns the [num_quantizers][frames] code matrix.
    std::vector<std::vector<int32_t>> encode(const std::vector<float> & hidden, int64_t frames) const;

private:
    struct Codebook {
        std::vector<float> table;             // [codebook_size, codebook_dim] row-major
        std::vector<float> table_normalized;  // [codebook_size, codebook_dim], L2-normalized rows (encode)
        std::vector<float> out_weight;        // [rvq_dim, codebook_dim] row-major
        std::vector<float> out_bias;          // [rvq_dim]
        std::vector<float> latent_table;      // [codebook_size, code_dim] row-major (decode)
        std::vector<float> in_weight;         // [codebook_dim, rvq_dim] row-major (encode)
        std::vector<float> in_bias;           // [codebook_dim] (encode)
    };

    int64_t codebook_size_ = 0;
    int64_t codebook_dim_ = 0;
    int64_t rvq_dim_ = 0;
    int64_t code_dim_ = 0;
    int64_t num_quantizers_ = 0;
    std::vector<Codebook> codebooks_;
    std::vector<float> output_weight_;  // [code_dim, rvq_dim] row-major
    std::vector<float> output_bias_;    // [code_dim]
    std::vector<float> input_weight_;   // [rvq_dim, code_dim] row-major (encode)
    std::vector<float> input_bias_;     // [rvq_dim] (encode)
};

// MOSS-Audio-Tokenizer encoder: turns a reference waveform into RLFQ codes
// for zero-shot voice cloning. It is the structural mirror of MossAudioTokenizerDecoder --
// stereo is interleaved into one stream, patched down and run through a stack of
// causal Transformer blocks (interleaved RoPE, LayerScale, GELU MLP), then the
// RLFQ quantizer selects the nearest codes. Produces the same [num_quantizers,
// frames] code matrix the generator consumes.
class MossAudioTokenizerEncoder {
public:
    MossAudioTokenizerEncoder(
        const assets::TensorSource & source,
        std::shared_ptr<const MossAudioTokenizerQuantizer> quantizer,
        core::ExecutionContext & execution_context,
        size_t weight_context_bytes,
        size_t graph_arena_bytes,
        MossAudioTokenizerConfig config = moss_audio_tokenizer_v2_config());
    ~MossAudioTokenizerEncoder();

    MossAudioTokenizerEncoder(const MossAudioTokenizerEncoder &) = delete;
    MossAudioTokenizerEncoder & operator=(const MossAudioTokenizerEncoder &) = delete;

    // Encodes a waveform given as {left, right} channels (each with the same
    // per-channel sample count, 48 kHz) into [num_quantizers][frames] codes.
    MossAudioTokenizerCodes encode(const MossAudioTokenizerAudio & audio);
    void release_runtime_graphs();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// MOSS-Audio-Tokenizer-v2 decoder: turns generated RVQ codes into a 48 kHz
// stereo waveform. The codec is "CNN-free" -- the decoder is a stack of causal
// Transformer blocks (interleaved RoPE, LayerScale, GELU MLP) separated by
// reshape-based patch upsamples, ending in a channel de-interleave that splits
// the jointly-processed stream back into left/right. The RLFQ dequantizer
// (codes -> latent) is provided by MossAudioTokenizerQuantizer.
class MossAudioTokenizerDecoder {
public:
    MossAudioTokenizerDecoder(
        const assets::TensorSource & source,
        std::shared_ptr<const MossAudioTokenizerQuantizer> dequantizer,
        core::ExecutionContext & execution_context,
        size_t weight_context_bytes,
        size_t graph_arena_bytes,
        MossAudioTokenizerConfig config = moss_audio_tokenizer_v2_config());
    ~MossAudioTokenizerDecoder();

    MossAudioTokenizerDecoder(const MossAudioTokenizerDecoder &) = delete;
    MossAudioTokenizerDecoder & operator=(const MossAudioTokenizerDecoder &) = delete;

    int64_t sampling_rate() const noexcept;

    // Decodes [num_quantizers][steps] codes into a stereo waveform returned as
    // {left, right}, each with steps * 3840 samples at 48 kHz.
    MossAudioTokenizerAudio decode(const MossAudioTokenizerCodes & codes);
    void release_runtime_graphs();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

namespace {

// Rebuilds a weight-normalized 1x1 conv weight from its parametrization
// (original0 = magnitude g per output channel, original1 = direction v), the
// PyTorch weight_norm(dim=0) reconstruction weight = g * v / ||v||.
std::vector<float> reconstruct_weight_norm(
    const std::vector<float> & g,
    const std::vector<float> & v,
    int64_t out_channels,
    int64_t in_channels) {
    std::vector<float> weight(static_cast<size_t>(out_channels * in_channels));
#ifdef _OPENMP
#pragma omp parallel for if(out_channels * in_channels >= 4096)
#endif
    for (int64_t o = 0; o < out_channels; ++o) {
        double norm = 0.0;
        for (int64_t k = 0; k < in_channels; ++k) {
            const double value = v[static_cast<size_t>(o * in_channels + k)];
            norm += value * value;
        }
        const float scale = static_cast<float>(g[static_cast<size_t>(o)] / std::sqrt(norm));
        for (int64_t k = 0; k < in_channels; ++k) {
            weight[static_cast<size_t>(o * in_channels + k)] =
                v[static_cast<size_t>(o * in_channels + k)] * scale;
        }
    }
    return weight;
}

std::vector<float> load_wn_conv_weight(
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t out_channels,
    int64_t in_channels) {
    const auto g = source.require_f32(prefix + ".parametrizations.weight.original0");
    const auto v = source.require_f32(prefix + ".parametrizations.weight.original1");
    return reconstruct_weight_norm(g, v, out_channels, in_channels);
}

}  // namespace

MossAudioTokenizerQuantizer::MossAudioTokenizerQuantizer(
    const assets::TensorSource & source,
    int64_t num_quantizers,
    MossAudioTokenizerQuantizerConfig config)
    : codebook_size_(config.codebook_size),
      codebook_dim_(config.codebook_dim),
      rvq_dim_(config.rvq_dim),
      code_dim_(config.code_dim),
      num_quantizers_(num_quantizers) {
    if (num_quantizers_ <= 0) {
        throw std::runtime_error("MOSS codec dequantizer requires a positive quantizer count");
    }

    output_weight_ = load_wn_conv_weight(source, "quantizer.output_proj", code_dim_, rvq_dim_);
    output_bias_ = source.require_f32("quantizer.output_proj.bias");

    codebooks_.reserve(static_cast<size_t>(num_quantizers_));
    for (int64_t index = 0; index < num_quantizers_; ++index) {
        const std::string prefix = "quantizer.quantizers." + std::to_string(index);
        Codebook codebook;
        codebook.table = source.require_f32(prefix + ".codebook.weight");
        codebook.out_weight = load_wn_conv_weight(source, prefix + ".out_proj", rvq_dim_, codebook_dim_);
        codebook.out_bias = source.require_f32(prefix + ".out_proj.bias");
        std::vector<float> combined_bias(static_cast<size_t>(code_dim_));
        std::vector<float> combined_weight(static_cast<size_t>(code_dim_ * codebook_dim_));
#ifdef _OPENMP
#pragma omp parallel for if(code_dim_ >= 256)
#endif
        for (int64_t out = 0; out < code_dim_; ++out) {
            const float * output_row = &output_weight_[static_cast<size_t>(out * rvq_dim_)];
            float bias_sum = 0.0F;
            for (int64_t rvq = 0; rvq < rvq_dim_; ++rvq) {
                bias_sum += output_row[rvq] * codebook.out_bias[static_cast<size_t>(rvq)];
            }
            combined_bias[static_cast<size_t>(out)] = bias_sum;
            for (int64_t k = 0; k < codebook_dim_; ++k) {
                float sum = 0.0F;
                for (int64_t rvq = 0; rvq < rvq_dim_; ++rvq) {
                    sum += output_row[rvq] * codebook.out_weight[static_cast<size_t>(rvq * codebook_dim_ + k)];
                }
                combined_weight[static_cast<size_t>(out * codebook_dim_ + k)] = sum;
            }
        }
        codebook.latent_table.resize(static_cast<size_t>(codebook_size_ * code_dim_));
#ifdef _OPENMP
#pragma omp parallel for collapse(2) if(codebook_size_ * code_dim_ >= 4096)
#endif
        for (int64_t code = 0; code < codebook_size_; ++code) {
            for (int64_t out = 0; out < code_dim_; ++out) {
                const float * embedding = &codebook.table[static_cast<size_t>(code * codebook_dim_)];
                float sum = combined_bias[static_cast<size_t>(out)];
                const float * row = &combined_weight[static_cast<size_t>(out * codebook_dim_)];
                for (int64_t k = 0; k < codebook_dim_; ++k) {
                    sum += row[k] * embedding[k];
                }
                codebook.latent_table[static_cast<size_t>(code * code_dim_ + out)] = sum;
            }
        }
        codebook.in_weight = load_wn_conv_weight(source, prefix + ".in_proj", codebook_dim_, rvq_dim_);
        codebook.in_bias = source.require_f32(prefix + ".in_proj.bias");
        // Pre-normalize the codebook rows once (encode does L2-normalized nearest
        // search, matching the training LFQ; F.normalize uses eps=1e-12).
        codebook.table_normalized = codebook.table;
#ifdef _OPENMP
#pragma omp parallel for if(codebook_size_ * codebook_dim_ >= 4096)
#endif
        for (int64_t code = 0; code < codebook_size_; ++code) {
            float * row = &codebook.table_normalized[static_cast<size_t>(code * codebook_dim_)];
            double norm = 0.0;
            for (int64_t k = 0; k < codebook_dim_; ++k) {
                norm += static_cast<double>(row[k]) * static_cast<double>(row[k]);
            }
            const double scale = 1.0 / std::max(std::sqrt(norm), 1.0e-12);
            for (int64_t k = 0; k < codebook_dim_; ++k) {
                row[k] = static_cast<float>(row[k] * scale);
            }
        }
        codebooks_.push_back(std::move(codebook));
    }

    input_weight_ = load_wn_conv_weight(source, "quantizer.input_proj", rvq_dim_, code_dim_);
    input_bias_ = source.require_f32("quantizer.input_proj.bias");
}

std::vector<float> MossAudioTokenizerQuantizer::decode(const std::vector<std::vector<int32_t>> & codes) const {
    if (static_cast<int64_t>(codes.size()) != num_quantizers_) {
        throw std::runtime_error("MOSS codec dequantizer got the wrong number of codebooks");
    }
    const int64_t steps = codes.empty() ? 0 : static_cast<int64_t>(codes.front().size());
    if (steps <= 0) {
        throw std::runtime_error("MOSS codec dequantizer requires a non-empty code sequence");
    }
    for (int64_t step = 0; step < steps; ++step) {
        for (int64_t index = 0; index < num_quantizers_; ++index) {
            const int64_t code = codes[static_cast<size_t>(index)][static_cast<size_t>(step)];
            if (code < 0 || code >= codebook_size_) {
                throw std::runtime_error("MOSS codec code index out of range");
            }
        }
    }

    std::vector<float> latent(static_cast<size_t>(code_dim_ * steps));
#ifdef _OPENMP
#pragma omp parallel for if(steps * code_dim_ >= 4096)
#endif
    for (int64_t step = 0; step < steps; ++step) {
        for (int64_t out = 0; out < code_dim_; ++out) {
            float value = output_bias_[static_cast<size_t>(out)];
            for (int64_t index = 0; index < num_quantizers_; ++index) {
                const auto & codebook = codebooks_[static_cast<size_t>(index)];
                const int64_t code = codes[static_cast<size_t>(index)][static_cast<size_t>(step)];
                const float * decoded = &codebook.latent_table[static_cast<size_t>(code * code_dim_)];
                value += decoded[static_cast<size_t>(out)];
            }
            latent[static_cast<size_t>(out * steps + step)] = value;
        }
    }
    return latent;
}

std::vector<std::vector<int32_t>> MossAudioTokenizerQuantizer::encode(
    const std::vector<float> & hidden, int64_t frames) const {
    if (frames <= 0) {
        throw std::runtime_error("MOSS codec quantizer requires a non-empty encoder latent");
    }
    if (static_cast<int64_t>(hidden.size()) != frames * code_dim_) {
        throw std::runtime_error("MOSS codec quantizer got a mis-shaped encoder latent");
    }

    std::vector<std::vector<int32_t>> codes(
        static_cast<size_t>(num_quantizers_), std::vector<int32_t>(static_cast<size_t>(frames)));

    const auto encode_frame = [&](int64_t step, std::vector<double> & residual, std::vector<double> & encoding) {
        std::fill(residual.begin(), residual.end(), 0.0);
        std::fill(encoding.begin(), encoding.end(), 0.0);

        // input_proj: encoder latent [code_dim] -> rvq_dim (WNConv1d 1x1).
        const float * frame_hidden = &hidden[static_cast<size_t>(step * code_dim_)];
        for (int64_t out = 0; out < rvq_dim_; ++out) {
            double sum = input_bias_[static_cast<size_t>(out)];
            const float * row = &input_weight_[static_cast<size_t>(out * code_dim_)];
            for (int64_t k = 0; k < code_dim_; ++k) {
                sum += static_cast<double>(row[k]) * static_cast<double>(frame_hidden[k]);
            }
            residual[static_cast<size_t>(out)] = sum;
        }

        for (int64_t index = 0; index < num_quantizers_; ++index) {
            const auto & codebook = codebooks_[static_cast<size_t>(index)];

            // in_proj: residual [rvq_dim] -> codebook_dim, then L2-normalize.
            double enc_norm = 0.0;
            for (int64_t c = 0; c < codebook_dim_; ++c) {
                double sum = codebook.in_bias[static_cast<size_t>(c)];
                const float * row = &codebook.in_weight[static_cast<size_t>(c * rvq_dim_)];
                for (int64_t k = 0; k < rvq_dim_; ++k) {
                    sum += static_cast<double>(row[k]) * residual[static_cast<size_t>(k)];
                }
                encoding[static_cast<size_t>(c)] = sum;
                enc_norm += sum * sum;
            }
            const double enc_scale = 1.0 / std::max(std::sqrt(enc_norm), 1.0e-12);
            for (int64_t c = 0; c < codebook_dim_; ++c) {
                encoding[static_cast<size_t>(c)] *= enc_scale;
            }

            // Nearest code by cosine similarity (both sides L2-normalized), i.e.
            // argmax dot == argmin squared distance on the unit sphere.
            int32_t best_code = 0;
            double best_dot = -std::numeric_limits<double>::infinity();
            for (int64_t code = 0; code < codebook_size_; ++code) {
                const float * row = &codebook.table_normalized[static_cast<size_t>(code * codebook_dim_)];
                double dot = 0.0;
                for (int64_t c = 0; c < codebook_dim_; ++c) {
                    dot += static_cast<double>(row[c]) * encoding[static_cast<size_t>(c)];
                }
                if (dot > best_dot) {
                    best_dot = dot;
                    best_code = static_cast<int32_t>(code);
                }
            }
            codes[static_cast<size_t>(index)][static_cast<size_t>(step)] = best_code;

            // Subtract the residual contribution: out_proj(raw codebook row).
            const float * embedding = &codebook.table[static_cast<size_t>(best_code * codebook_dim_)];
            for (int64_t out = 0; out < rvq_dim_; ++out) {
                double sum = codebook.out_bias[static_cast<size_t>(out)];
                const float * row = &codebook.out_weight[static_cast<size_t>(out * codebook_dim_)];
                for (int64_t k = 0; k < codebook_dim_; ++k) {
                    sum += static_cast<double>(row[k]) * static_cast<double>(embedding[k]);
                }
                residual[static_cast<size_t>(out)] -= sum;
            }
        }
    };

#ifdef _OPENMP
    if (frames >= 8) {
#pragma omp parallel
        {
            std::vector<double> residual(static_cast<size_t>(rvq_dim_));
            std::vector<double> encoding(static_cast<size_t>(codebook_dim_));
#pragma omp for
            for (int64_t step = 0; step < frames; ++step) {
                encode_frame(step, residual, encoding);
            }
        }
    } else
#endif
    {
        std::vector<double> residual(static_cast<size_t>(rvq_dim_));
        std::vector<double> encoding(static_cast<size_t>(codebook_dim_));
        for (int64_t step = 0; step < frames; ++step) {
            encode_frame(step, residual, encoding);
        }
    }
    return codes;
}

namespace {

namespace cd = codec_detail;

cd::TransformerSpec to_encoder_transformer_spec(const MossAudioTokenizerTransformerStage & stage) {
    return {
        stage.input_dimension,
        stage.output_dimension,
        stage.model_dimension,
        stage.num_heads,
        stage.num_layers,
        stage.feedforward_dimension,
        stage.context_window,
        stage.patch_size,
    };
}

// PatchedPretransform (encode/downsample): [1, l, d] -> [1, l/patch, d*patch].
// Packs `patch` consecutive frames into the feature dim, matching
// x.reshape(b, d, -1, h).permute(0, 1, 3, 2).reshape(b, d * h, -1) (conv layout),
// i.e. output feature (d_idx*patch + h_idx) at time lt = input feature d_idx at
// time lt*patch + h_idx. This is the exact inverse of the decoder's upsample.
core::TensorValue patch_downsample(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t patch) {
    auto contiguous = core::ensure_backend_addressable_layout(ctx, input);
    const int64_t total_length = contiguous.shape.dims[1];
    const int64_t channels = contiguous.shape.dims[2];
    const int64_t length = total_length / patch;
    auto reshaped = engine::modules::ReshapeModule({
        core::TensorShape::from_dims({1, length, patch, channels}),
    }).build(ctx, contiguous);
    auto transposed = engine::modules::TransposeModule({{0, 1, 3, 2}, reshaped.shape.rank}).build(ctx, reshaped);
    return engine::modules::ReshapeModule({
        core::TensorShape::from_dims({1, length, channels * patch}),
    }).build(ctx, core::ensure_backend_addressable_layout(ctx, transposed));
}

}  // namespace

struct MossAudioTokenizerEncoder::Impl {
    struct StageInput {
        ggml_tensor * positions = nullptr;
        std::vector<int32_t> position_host;
        ggml_tensor * mask = nullptr;
        std::vector<float> mask_host;
    };

    struct GraphCache {
        int64_t interleaved = 0;
        int64_t output_steps = 0;
        std::unique_ptr<ggml_context, cd::GgmlContextDeleter> graph_ctx;
        ggml_cgraph * graph = nullptr;
        ggml_tensor * input = nullptr;
        ggml_tensor * output = nullptr;
        std::vector<StageInput> stage_inputs;
        std::unique_ptr<std::remove_pointer_t<ggml_gallocr_t>, cd::GgmlGallocrDeleter> gallocr;
    };

    ggml_backend_t backend = nullptr;
    core::BackendType backend_type = core::BackendType::Cpu;
    int threads = 1;
    int64_t samples_per_frame = 3840;
    int64_t channels = 2;
    size_t graph_arena_bytes = 0;
    MossAudioTokenizerConfig config;
    std::shared_ptr<const MossAudioTokenizerQuantizer> quantizer;
    std::unique_ptr<core::BackendWeightStore> store;
    std::vector<cd::TransformerWeights> transformers;
    std::unique_ptr<GraphCache> graph_cache;

    GraphCache & prepare_graph(int64_t interleaved) {
        if (graph_cache != nullptr && graph_cache->interleaved == interleaved) {
            return *graph_cache;
        }

        auto cache = std::make_unique<GraphCache>();
        cache->interleaved = interleaved;

        ggml_init_params params{graph_arena_bytes, nullptr, true};
        cache->graph_ctx.reset(ggml_init(params));
        if (cache->graph_ctx == nullptr) {
            throw std::runtime_error("failed to initialize MOSS codec encoder graph context");
        }
        core::ModuleBuildContext ctx{cache->graph_ctx.get(), "moss.audio_tokenizer.encode", backend_type};

        auto input_tensor =
            core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, interleaved, 1}));
        ggml_set_input(input_tensor.tensor);
        cache->input = input_tensor.tensor;

        auto hidden = input_tensor;
        int64_t steps = interleaved;
        cache->stage_inputs.reserve(transformers.size());
        for (const auto & transformer : transformers) {
            hidden = patch_downsample(ctx, hidden, transformer.spec.patch);
            steps /= transformer.spec.patch;

            auto positions = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({steps}));
            ggml_set_input(positions.tensor);
            auto mask =
                core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 1, steps, steps}));
            ggml_set_input(mask.tensor);

            StageInput stage;
            stage.positions = positions.tensor;
            stage.position_host.resize(static_cast<size_t>(steps));
            for (int64_t i = 0; i < steps; ++i) {
                stage.position_host[static_cast<size_t>(i)] = static_cast<int32_t>(i);
            }
            stage.mask = mask.tensor;
            stage.mask_host = cd::causal_context_mask(steps, transformer.spec.context);
            cache->stage_inputs.push_back(std::move(stage));

            hidden = cd::run_transformer(ctx, hidden, transformer, positions, mask, steps);
        }
        if (config.encoder_final_patch > 1) {
            hidden = patch_downsample(ctx, hidden, config.encoder_final_patch);
            steps /= config.encoder_final_patch;
        }

        hidden = core::ensure_backend_addressable_layout(ctx, hidden);
        ggml_set_output(hidden.tensor);
        cache->output = hidden.tensor;
        cache->output_steps = steps;

        cache->graph = ggml_new_graph_custom(cache->graph_ctx.get(), 131072, false);
        ggml_build_forward_expand(cache->graph, hidden.tensor);

        cache->gallocr.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)));
        if (cache->gallocr == nullptr ||
            !ggml_gallocr_reserve(cache->gallocr.get(), cache->graph) ||
            !ggml_gallocr_alloc_graph(cache->gallocr.get(), cache->graph)) {
            throw std::runtime_error("failed to allocate MOSS codec encoder forward graph");
        }

        graph_cache = std::move(cache);
        return *graph_cache;
    }

    void release_runtime_graphs() {
        graph_cache.reset();
    }
};

MossAudioTokenizerEncoder::MossAudioTokenizerEncoder(
    const assets::TensorSource & source,
    std::shared_ptr<const MossAudioTokenizerQuantizer> quantizer,
    core::ExecutionContext & execution_context,
    size_t weight_context_bytes,
    size_t graph_arena_bytes,
    MossAudioTokenizerConfig config)
    : impl_(std::make_unique<Impl>()) {
    impl_->backend = execution_context.backend();
    if (impl_->backend == nullptr) {
        throw std::runtime_error("MOSS codec encoder backend is not initialized");
    }
    impl_->backend_type = execution_context.backend_type();
    impl_->threads = execution_context.config().threads;
    impl_->config = config;
    impl_->samples_per_frame = config.samples_per_frame;
    impl_->channels = config.channels;
    impl_->graph_arena_bytes = graph_arena_bytes;
    impl_->quantizer = std::move(quantizer);
    if (impl_->quantizer == nullptr) {
        throw std::runtime_error("MOSS codec encoder requires a quantizer");
    }

    cd::CodecWeights weights(source);
    impl_->store = std::make_unique<core::BackendWeightStore>(
        impl_->backend, impl_->backend_type, "moss.audio_tokenizer.encoder", weight_context_bytes);
    impl_->transformers.reserve(config.encoder_stages.size());
    for (size_t index = 0; index < config.encoder_stages.size(); ++index) {
        const int64_t module_index =
            config.encoder_module_start + static_cast<int64_t>(index) * config.encoder_module_stride;
        impl_->transformers.push_back(cd::load_transformer(
            *impl_->store, weights, to_encoder_transformer_spec(config.encoder_stages[index]), "encoder", module_index));
    }
    impl_->store->upload();
}

MossAudioTokenizerEncoder::~MossAudioTokenizerEncoder() = default;

MossAudioTokenizerCodes MossAudioTokenizerEncoder::encode(const MossAudioTokenizerAudio & audio) {
    const auto & channels = audio.channels;
    if (audio.sampling_rate != 0 && audio.sampling_rate != impl_->config.sampling_rate) {
        throw std::runtime_error("MOSS codec encoder input sample rate does not match codec config");
    }
    if (static_cast<int64_t>(channels.size()) != impl_->channels) {
        throw std::runtime_error("MOSS codec encoder input channel count does not match codec config");
    }
    for (int64_t channel = 1; channel < impl_->channels; ++channel) {
        if (channels[static_cast<size_t>(channel)].size() != channels.front().size()) {
            throw std::runtime_error("MOSS codec encoder channels must have equal length");
        }
    }
    const int64_t raw_per_channel = static_cast<int64_t>(channels.front().size());
    if (raw_per_channel <= 0) {
        throw std::runtime_error("MOSS codec encoder requires a non-empty waveform");
    }

    // Pad each channel up to a multiple of the downsample rate for the encoder graph,
    // but keep the official valid code length as floor(valid_samples / samples_per_frame).
    // MossAudioTokenizerPatchedPretransform pads the tensor and propagates input_lengths
    // with integer division, then slices audio_codes to audio_codes_lengths.
    const int64_t frames = (raw_per_channel + impl_->samples_per_frame - 1) / impl_->samples_per_frame;
    const int64_t valid_frames = raw_per_channel / impl_->samples_per_frame;
    const int64_t per_channel = frames * impl_->samples_per_frame;
    const int64_t interleaved = per_channel * impl_->channels;
    std::vector<float> waveform(static_cast<size_t>(interleaved), 0.0F);
#ifdef _OPENMP
#pragma omp parallel for if(raw_per_channel >= 4096)
#endif
    for (int64_t i = 0; i < raw_per_channel; ++i) {
        for (int64_t channel = 0; channel < impl_->channels; ++channel) {
            waveform[static_cast<size_t>(impl_->channels * i + channel)] =
                channels[static_cast<size_t>(channel)][static_cast<size_t>(i)];
        }
    }

    auto & graph = impl_->prepare_graph(interleaved);
    ggml_backend_tensor_set(graph.input, waveform.data(), 0, waveform.size() * sizeof(float));
    for (const auto & stage : graph.stage_inputs) {
        ggml_backend_tensor_set(
            stage.positions, stage.position_host.data(), 0, stage.position_host.size() * sizeof(int32_t));
        ggml_backend_tensor_set(
            stage.mask, stage.mask_host.data(), 0, stage.mask_host.size() * sizeof(float));
    }

    core::set_backend_threads(impl_->backend, impl_->threads);
    const ggml_status status = ggml_backend_graph_compute(impl_->backend, graph.graph);
    ggml_backend_synchronize(impl_->backend);
    if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("MOSS codec encoder forward graph compute failed");
    }

    // hidden is [1, frames, code_dim] feature-last; ggml memory order is
    // channel-fastest, i.e. flat[frame * code_dim + channel] -- exactly the
    // layout MossAudioTokenizerQuantizer::encode expects.
    std::vector<float> latent(static_cast<size_t>(graph.output_steps * cd::kCodeDim));
    ggml_backend_tensor_get(graph.output, latent.data(), 0, latent.size() * sizeof(float));

    if (valid_frames <= 0) {
        throw std::runtime_error("MOSS codec encoder input is shorter than one codec frame");
    }
    latent.resize(static_cast<size_t>(valid_frames * cd::kCodeDim));
    return MossAudioTokenizerCodes{
        valid_frames,
        impl_->quantizer->encode(latent, valid_frames),
    };
}

void MossAudioTokenizerEncoder::release_runtime_graphs() {
    impl_->release_runtime_graphs();
}

namespace {

namespace cd = codec_detail;

constexpr int64_t kAttentionQueryChunk = 1500;

cd::TransformerSpec to_decoder_transformer_spec(const MossAudioTokenizerTransformerStage & stage) {
    return {
        stage.input_dimension,
        stage.output_dimension,
        stage.model_dimension,
        stage.num_heads,
        stage.num_layers,
        stage.feedforward_dimension,
        stage.context_window,
        stage.patch_size,
    };
}

// PatchedPretransform (decode/upsample): [1, l, d*patch] -> [1, l*patch, d].
// Each frame is unpacked into `patch` consecutive frames along time, matching
// x.reshape(b, d, h, l).permute(0, 1, 3, 2).reshape(b, d, l * h).
core::TensorValue patch_upsample(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t patch) {
    auto contiguous = core::ensure_backend_addressable_layout(ctx, input);
    const int64_t length = contiguous.shape.dims[1];
    const int64_t packed = contiguous.shape.dims[2];
    const int64_t channels = packed / patch;
    auto reshaped = engine::modules::ReshapeModule({
        core::TensorShape::from_dims({1, length, channels, patch}),
    }).build(ctx, contiguous);
    auto transposed = engine::modules::TransposeModule({{0, 1, 3, 2}, reshaped.shape.rank}).build(ctx, reshaped);
    return engine::modules::ReshapeModule({
        core::TensorShape::from_dims({1, length * patch, channels}),
    }).build(ctx, core::ensure_backend_addressable_layout(ctx, transposed));
}

}  // namespace

struct MossAudioTokenizerDecoder::Impl {
    struct StageInput {
        struct MaskInput {
            ggml_tensor * tensor = nullptr;
            std::vector<float> host;
        };

        ggml_tensor * positions = nullptr;
        std::vector<int32_t> position_host;
        std::vector<MaskInput> masks;
    };

    struct GraphCache {
        int64_t frames = 0;
        int64_t interleaved = 0;
        std::unique_ptr<ggml_context, cd::GgmlContextDeleter> graph_ctx;
        ggml_cgraph * graph = nullptr;
        ggml_tensor * input = nullptr;
        ggml_tensor * output = nullptr;
        std::vector<StageInput> stage_inputs;
        std::unique_ptr<std::remove_pointer_t<ggml_gallocr_t>, cd::GgmlGallocrDeleter> gallocr;
    };

    ggml_backend_t backend = nullptr;
    core::BackendType backend_type = core::BackendType::Cpu;
    int threads = 1;
    int64_t sampling_rate = 48000;
    size_t graph_arena_bytes = 0;
    MossAudioTokenizerConfig config;
    std::shared_ptr<const MossAudioTokenizerQuantizer> dequantizer;
    std::unique_ptr<core::BackendWeightStore> store;
    std::vector<cd::TransformerWeights> transformers;
    std::unique_ptr<GraphCache> graph_cache;

    GraphCache & prepare_graph(int64_t frames) {
        if (graph_cache != nullptr && graph_cache->frames == frames) {
            return *graph_cache;
        }

        auto cache = std::make_unique<GraphCache>();
        cache->frames = frames;

        ggml_init_params params{graph_arena_bytes, nullptr, true};
        cache->graph_ctx.reset(ggml_init(params));
        if (cache->graph_ctx == nullptr) {
            throw std::runtime_error("failed to initialize MOSS codec decoder graph context");
        }
        core::ModuleBuildContext ctx{cache->graph_ctx.get(), "moss.audio_tokenizer.decode", backend_type};

        auto latent_tensor =
            core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, frames, cd::kCodeDim}));
        ggml_set_input(latent_tensor.tensor);
        cache->input = latent_tensor.tensor;

        auto hidden = latent_tensor;
        int64_t steps = frames;
        if (config.decoder_initial_patch > 1) {
            hidden = patch_upsample(ctx, hidden, config.decoder_initial_patch);
            steps *= config.decoder_initial_patch;
        }
        cache->stage_inputs.reserve(transformers.size());
        for (const auto & transformer : transformers) {
            auto positions = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({steps}));
            ggml_set_input(positions.tensor);
            const bool use_windowed_attention = steps > kAttentionQueryChunk;
            core::TensorValue mask;
            if (!use_windowed_attention) {
                mask = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 1, steps, steps}));
                ggml_set_input(mask.tensor);
            }

            StageInput stage;
            stage.positions = positions.tensor;
            stage.position_host.resize(static_cast<size_t>(steps));
            for (int64_t i = 0; i < steps; ++i) {
                stage.position_host[static_cast<size_t>(i)] = static_cast<int32_t>(i);
            }
            std::vector<cd::AttentionWindow> windows;
            if (use_windowed_attention) {
                for (int64_t query_start = 0; query_start < steps; query_start += kAttentionQueryChunk) {
                    const int64_t query_steps = std::min<int64_t>(kAttentionQueryChunk, steps - query_start);
                    const int64_t key_start = std::max<int64_t>(0, query_start - transformer.spec.context + 1);
                    const int64_t key_steps = query_start + query_steps - key_start;
                    auto window_mask = core::make_tensor(
                        ctx,
                        GGML_TYPE_F32,
                        core::TensorShape::from_dims({1, 1, query_steps, key_steps}));
                    ggml_set_input(window_mask.tensor);
                    stage.masks.push_back(StageInput::MaskInput{
                        window_mask.tensor,
                        cd::causal_context_mask_window(
                            query_start,
                            query_steps,
                            key_start,
                            key_steps,
                            transformer.spec.context),
                    });
                    windows.push_back(cd::AttentionWindow{
                        query_start,
                        query_steps,
                        key_start,
                        key_steps,
                        window_mask,
                    });
                }
            } else {
                stage.masks.push_back(StageInput::MaskInput{
                    mask.tensor,
                    cd::causal_context_mask(steps, transformer.spec.context),
                });
            }
            cache->stage_inputs.push_back(std::move(stage));

            hidden = windows.empty()
                ? cd::run_transformer(ctx, hidden, transformer, positions, mask, steps)
                : cd::run_transformer(ctx, hidden, transformer, positions, windows.front().mask, steps, &windows);
            hidden = patch_upsample(ctx, hidden, transformer.spec.patch);
            steps *= transformer.spec.patch;
        }

        hidden = core::ensure_backend_addressable_layout(ctx, hidden);
        ggml_set_output(hidden.tensor);
        cache->output = hidden.tensor;
        cache->interleaved = steps;

        cache->graph = ggml_new_graph_custom(cache->graph_ctx.get(), 131072, false);
        ggml_build_forward_expand(cache->graph, hidden.tensor);

        cache->gallocr.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)));
        if (cache->gallocr == nullptr ||
            !ggml_gallocr_reserve(cache->gallocr.get(), cache->graph) ||
            !ggml_gallocr_alloc_graph(cache->gallocr.get(), cache->graph)) {
            throw std::runtime_error("failed to allocate MOSS codec decoder forward graph");
        }

        graph_cache = std::move(cache);
        return *graph_cache;
    }

    void release_runtime_graphs() {
        graph_cache.reset();
    }
};

MossAudioTokenizerDecoder::MossAudioTokenizerDecoder(
    const assets::TensorSource & source,
    std::shared_ptr<const MossAudioTokenizerQuantizer> dequantizer,
    core::ExecutionContext & execution_context,
    size_t weight_context_bytes,
    size_t graph_arena_bytes,
    MossAudioTokenizerConfig config)
    : impl_(std::make_unique<Impl>()) {
    impl_->backend = execution_context.backend();
    if (impl_->backend == nullptr) {
        throw std::runtime_error("MOSS codec decoder backend is not initialized");
    }
    impl_->backend_type = execution_context.backend_type();
    impl_->threads = execution_context.config().threads;
    impl_->config = config;
    impl_->sampling_rate = config.sampling_rate;
    impl_->graph_arena_bytes = graph_arena_bytes;
    impl_->dequantizer = std::move(dequantizer);
    if (impl_->dequantizer == nullptr) {
        throw std::runtime_error("MOSS codec decoder requires a quantizer");
    }

    cd::CodecWeights weights(source);
    impl_->store = std::make_unique<core::BackendWeightStore>(
        impl_->backend, impl_->backend_type, "moss.audio_tokenizer.decoder", weight_context_bytes);
    impl_->transformers.reserve(config.decoder_stages.size());
    for (size_t index = 0; index < config.decoder_stages.size(); ++index) {
        const int64_t module_index =
            config.decoder_module_start + static_cast<int64_t>(index) * config.decoder_module_stride;
        impl_->transformers.push_back(cd::load_transformer(
            *impl_->store, weights, to_decoder_transformer_spec(config.decoder_stages[index]), "decoder", module_index));
    }
    impl_->store->upload();
}

MossAudioTokenizerDecoder::~MossAudioTokenizerDecoder() = default;

int64_t MossAudioTokenizerDecoder::sampling_rate() const noexcept {
    return impl_->sampling_rate;
}

MossAudioTokenizerAudio MossAudioTokenizerDecoder::decode(const MossAudioTokenizerCodes & codes) {
    const int64_t frames = codes.frames;
    if (frames <= 0) {
        throw std::runtime_error("MOSS codec decoder requires a non-empty code sequence");
    }
    if (codes.codebooks.empty()) {
        throw std::runtime_error("MOSS codec decoder requires codebooks");
    }
    for (const auto & codebook : codes.codebooks) {
        if (static_cast<int64_t>(codebook.size()) != frames) {
            throw std::runtime_error("MOSS codec decoder codebooks do not match frame count");
        }
    }

    // Codes -> continuous latent [code_dim, frames] (channel-major), transposed
    // into the feature-last [1, frames, code_dim] layout the decoder expects.
    double dequant_ms = 0.0;
    double latent_pack_ms = 0.0;
    double graph_build_ms = 0.0;
    double input_upload_ms = 0.0;
    double graph_compute_ms = 0.0;
    double output_read_ms = 0.0;
    double deinterleave_ms = 0.0;
    const bool collect_timing = engine::debug::timing_log_enabled();
    std::vector<float> latent;
    if (collect_timing) {
        dequant_ms = engine::debug::measure_ms([&]() {
            latent = impl_->dequantizer->decode(codes.codebooks);
        });
    } else {
        latent = impl_->dequantizer->decode(codes.codebooks);
    }
    std::vector<float> latent_input;
    if (collect_timing) {
        latent_pack_ms = engine::debug::measure_ms([&]() {
            latent_input.resize(static_cast<size_t>(frames * cd::kCodeDim));
#ifdef _OPENMP
#pragma omp parallel for collapse(2) if(frames * cd::kCodeDim >= 4096)
#endif
            for (int64_t channel = 0; channel < cd::kCodeDim; ++channel) {
                for (int64_t step = 0; step < frames; ++step) {
                    latent_input[static_cast<size_t>(step * cd::kCodeDim + channel)] =
                        latent[static_cast<size_t>(channel * frames + step)];
                }
            }
        });
    } else {
        latent_input.resize(static_cast<size_t>(frames * cd::kCodeDim));
#ifdef _OPENMP
#pragma omp parallel for collapse(2) if(frames * cd::kCodeDim >= 4096)
#endif
        for (int64_t channel = 0; channel < cd::kCodeDim; ++channel) {
            for (int64_t step = 0; step < frames; ++step) {
                latent_input[static_cast<size_t>(step * cd::kCodeDim + channel)] =
                    latent[static_cast<size_t>(channel * frames + step)];
            }
        }
    }

    const auto graph_build_start = std::chrono::steady_clock::now();
    auto & graph = impl_->prepare_graph(frames);
    if (collect_timing) {
        graph_build_ms = engine::debug::elapsed_ms(graph_build_start);
    }

    const auto upload_start = std::chrono::steady_clock::now();
    ggml_backend_tensor_set(
        graph.input, latent_input.data(), 0, latent_input.size() * sizeof(float));
    for (const auto & stage : graph.stage_inputs) {
        ggml_backend_tensor_set(
            stage.positions, stage.position_host.data(), 0, stage.position_host.size() * sizeof(int32_t));
        for (const auto & mask : stage.masks) {
            ggml_backend_tensor_set(mask.tensor, mask.host.data(), 0, mask.host.size() * sizeof(float));
        }
    }
    if (collect_timing) {
        input_upload_ms = engine::debug::elapsed_ms(upload_start);
    }

    const auto compute_start = std::chrono::steady_clock::now();
    core::set_backend_threads(impl_->backend, impl_->threads);
    const ggml_status status = ggml_backend_graph_compute(impl_->backend, graph.graph);
    ggml_backend_synchronize(impl_->backend);
    if (collect_timing) {
        graph_compute_ms = engine::debug::elapsed_ms(compute_start);
    }
    if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("MOSS codec decoder forward graph compute failed");
    }

    const int64_t interleaved = graph.interleaved;  // frames * samples_per_frame * channels
    std::vector<float> flat(static_cast<size_t>(interleaved));
    const auto read_start = std::chrono::steady_clock::now();
    ggml_backend_tensor_get(graph.output, flat.data(), 0, flat.size() * sizeof(float));
    if (collect_timing) {
        output_read_ms = engine::debug::elapsed_ms(read_start);
    }

    // De-interleave the jointly-processed stream back into left/right channels
    // (channel 0 = even samples, channel 1 = odd samples).
    const int64_t per_channel = frames * impl_->config.samples_per_frame;
    if (impl_->config.channels == 1) {
        return MossAudioTokenizerAudio{impl_->sampling_rate, {std::move(flat)}};
    }
    std::vector<std::vector<float>> channels(
        static_cast<size_t>(impl_->config.channels),
        std::vector<float>(static_cast<size_t>(per_channel)));
    if (collect_timing) {
        deinterleave_ms = engine::debug::measure_ms([&]() {
#ifdef _OPENMP
#pragma omp parallel for if(per_channel >= 4096)
#endif
            for (int64_t i = 0; i < per_channel; ++i) {
                for (int64_t channel = 0; channel < impl_->config.channels; ++channel) {
                    channels[static_cast<size_t>(channel)][static_cast<size_t>(i)] =
                        flat[static_cast<size_t>(impl_->config.channels * i + channel)];
                }
            }
        });
        engine::debug::timing_log_scalar("moss.audio_tokenizer.decode.dequant_ms", dequant_ms);
        engine::debug::timing_log_scalar("moss.audio_tokenizer.decode.latent_pack_ms", latent_pack_ms);
        engine::debug::timing_log_scalar("moss.audio_tokenizer.decode.graph_build_ms", graph_build_ms);
        engine::debug::timing_log_scalar("moss.audio_tokenizer.decode.input_upload_ms", input_upload_ms);
        engine::debug::timing_log_scalar("moss.audio_tokenizer.decode.graph_compute_ms", graph_compute_ms);
        engine::debug::timing_log_scalar("moss.audio_tokenizer.decode.output_read_ms", output_read_ms);
        engine::debug::timing_log_scalar("moss.audio_tokenizer.decode.deinterleave_ms", deinterleave_ms);
    } else {
#ifdef _OPENMP
#pragma omp parallel for if(per_channel >= 4096)
#endif
        for (int64_t i = 0; i < per_channel; ++i) {
            for (int64_t channel = 0; channel < impl_->config.channels; ++channel) {
                channels[static_cast<size_t>(channel)][static_cast<size_t>(i)] =
                    flat[static_cast<size_t>(impl_->config.channels * i + channel)];
            }
        }
    }
    return MossAudioTokenizerAudio{impl_->sampling_rate, std::move(channels)};
}

void MossAudioTokenizerDecoder::release_runtime_graphs() {
    impl_->release_runtime_graphs();
}

// MOSS-Audio-Tokenizer v1 - 24 kHz mono, hop 1920, four transformer stages per side.
// Same layer math as v2 with two fewer stages, and one 10 s attention window per stage
// rather than v2's per-stage durations, so the context lengths are just the stage frame
// rate times ten.
MossAudioTokenizerConfig moss_audio_tokenizer_v1_config() {
    MossAudioTokenizerConfig config;
    config.sampling_rate = 24000;
    config.samples_per_frame = 1920;
    config.channels = 1;
    config.quantizer = MossAudioTokenizerQuantizerConfig{
        1024,
        8,
        512,
        768,
        32,
    };
    config.encoder_stages = {
        {240, 384, 768, 12, 12, 3072, 1000, 240},
        {768, 384, 768, 12, 12, 3072, 500, 2},
        {768, 640, 768, 12, 12, 3072, 250, 2},
        {1280, 768, 1280, 20, 32, 5120, 125, 2},
    };
    config.decoder_stages = {
        {768, 1280, 1280, 20, 32, 5120, 125, 2},
        {640, 768, 768, 12, 12, 3072, 250, 2},
        {384, 768, 768, 12, 12, 3072, 500, 2},
        {384, 240, 768, 12, 12, 3072, 1000, 240},
    };
    config.encoder_module_start = 1;
    config.encoder_module_stride = 2;
    config.decoder_module_start = 0;
    config.decoder_module_stride = 2;
    return config;
}

MossAudioTokenizerConfig moss_audio_tokenizer_v2_config() {
    MossAudioTokenizerConfig config;
    config.sampling_rate = 48000;
    config.samples_per_frame = 3840;
    config.quantizer = MossAudioTokenizerQuantizerConfig{
        1024,
        8,
        512,
        768,
        12,
    };
    config.encoder_stages = {
        {240, 384, 768, 12, 12, 3072, 400, 240},
        {768, 384, 768, 12, 12, 3072, 400, 2},
        {768, 384, 768, 12, 12, 3072, 400, 2},
        {768, 384, 768, 12, 12, 3072, 400, 2},
        {768, 640, 768, 12, 12, 3072, 250, 2},
        {1280, 768, 1280, 20, 32, 5120, 125, 2},
    };
    config.decoder_stages = {
        {768, 1280, 1280, 20, 32, 5120, 125, 2},
        {640, 768, 768, 12, 12, 3072, 250, 2},
        {384, 768, 768, 12, 12, 3072, 400, 2},
        {384, 768, 768, 12, 12, 3072, 400, 2},
        {384, 768, 768, 12, 12, 3072, 400, 2},
        {384, 240, 768, 12, 12, 3072, 400, 240},
    };
    config.encoder_module_start = 1;
    config.encoder_module_stride = 2;
    config.decoder_module_start = 0;
    config.decoder_module_stride = 2;
    return config;
}

MossAudioTokenizerConfig moss_audio_tokenizer_nano_config() {
    MossAudioTokenizerConfig config;
    config.sampling_rate = 48000;
    config.samples_per_frame = 3840;
    config.quantizer = MossAudioTokenizerQuantizerConfig{
        1024,
        8,
        512,
        768,
        16,
    };
    config.encoder_stages = {
        {240, 384, 256, 4, 4, 1024, 1600, 240},
        {768, 384, 256, 4, 2, 1024, 1200, 2},
        {768, 384, 256, 4, 2, 1024, 800, 2},
        {768, 192, 256, 4, 4, 1024, 500, 2},
    };
    config.decoder_stages = {
        {192, 768, 256, 4, 4, 1024, 1000, 2},
        {384, 768, 256, 4, 2, 1024, 1600, 2},
        {384, 768, 256, 4, 2, 1024, 2400, 2},
        {384, 240, 256, 4, 4, 1024, 3200, 240},
    };
    config.encoder_final_patch = 4;
    config.decoder_initial_patch = 4;
    config.encoder_module_start = 1;
    config.encoder_module_stride = 2;
    config.decoder_module_start = 1;
    config.decoder_module_stride = 2;
    return config;
}

struct MossAudioTokenizerCodecRuntime::Impl {
    Impl(
        std::shared_ptr<const assets::TensorSource> source_in,
        core::ExecutionContext & decode_context_in,
        int64_t num_quantizers_in,
        MossAudioTokenizerCodecRuntimeOptions options_in,
        MossAudioTokenizerConfig config_in)
        : source(std::move(source_in)),
          decode_context(decode_context_in),
          num_quantizers(num_quantizers_in),
          options(options_in),
          config(std::move(config_in)) {}

    std::shared_ptr<const assets::TensorSource> source;
    core::ExecutionContext & decode_context;
    int64_t num_quantizers = 0;
    MossAudioTokenizerCodecRuntimeOptions options;
    MossAudioTokenizerConfig config;
    std::shared_ptr<const MossAudioTokenizerQuantizer> quantizer;
    std::unique_ptr<core::ExecutionContext> encode_context;
    std::unique_ptr<MossAudioTokenizerEncoder> encoder;
    std::unique_ptr<MossAudioTokenizerDecoder> decoder;

    core::ExecutionContext & encoder_execution_context() {
        if (!options.separate_encoder_context) {
            return decode_context;
        }
        if (encode_context == nullptr) {
            encode_context = std::make_unique<core::ExecutionContext>(decode_context.config());
        }
        return *encode_context;
    }

    MossAudioTokenizerEncoder & require_encoder() {
        if (encoder == nullptr) {
            encoder = std::make_unique<MossAudioTokenizerEncoder>(
                *source,
                quantizer,
                encoder_execution_context(),
                options.weight_context_bytes,
                options.encoder_graph_arena_bytes,
                config);
        }
        return *encoder;
    }

    MossAudioTokenizerDecoder & require_decoder() {
        if (decoder == nullptr) {
            decoder = std::make_unique<MossAudioTokenizerDecoder>(
                *source,
                quantizer,
                decode_context,
                options.weight_context_bytes,
                options.decoder_graph_arena_bytes,
                config);
        }
        return *decoder;
    }
};

MossAudioTokenizerCodecRuntime::MossAudioTokenizerCodecRuntime(
    std::shared_ptr<const assets::TensorSource> source,
    core::ExecutionContext & execution_context,
    int64_t num_quantizers,
    MossAudioTokenizerCodecRuntimeOptions options,
    MossAudioTokenizerConfig config)
    : impl_(std::make_unique<Impl>(
          std::move(source),
          execution_context,
          num_quantizers,
          options,
          std::move(config))) {
    if (impl_->source == nullptr) {
        throw std::runtime_error("MOSS audio tokenizer codec requires weights");
    }
    if (impl_->num_quantizers <= 0) {
        throw std::runtime_error("MOSS audio tokenizer codec requires a positive quantizer count");
    }
    impl_->quantizer = std::make_shared<MossAudioTokenizerQuantizer>(
        *impl_->source,
        impl_->num_quantizers,
        impl_->config.quantizer);
}

MossAudioTokenizerCodecRuntime::~MossAudioTokenizerCodecRuntime() = default;

int64_t MossAudioTokenizerCodecRuntime::sampling_rate() const noexcept {
    return impl_->config.sampling_rate;
}

void MossAudioTokenizerCodecRuntime::prepare_encoder() {
    (void) impl_->require_encoder();
}

void MossAudioTokenizerCodecRuntime::prepare_decoder() {
    (void) impl_->require_decoder();
}

MossAudioTokenizerCodes MossAudioTokenizerCodecRuntime::encode(const MossAudioTokenizerAudio & audio) {
    return impl_->require_encoder().encode(audio);
}

MossAudioTokenizerAudio MossAudioTokenizerCodecRuntime::decode(const MossAudioTokenizerCodes & codes) {
    return impl_->require_decoder().decode(codes);
}

void MossAudioTokenizerCodecRuntime::release_runtime_graphs() {
    if (impl_->encoder != nullptr) {
        impl_->encoder->release_runtime_graphs();
    }
    if (impl_->decoder != nullptr) {
        impl_->decoder->release_runtime_graphs();
    }
}

}  // namespace engine::codecs
