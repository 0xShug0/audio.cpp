#include "engine/community_models/lfm2_audio/backbone.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/feed_forward_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/transformers/causal_decoder.h"
#include "engine/framework/modules/transformers/decoder.h"
#include "engine/framework/runtime/errors.h"
#include "engine/framework/runtime/kv_cache.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace engine::community_models::lfm2_audio {
namespace {

namespace modules = engine::modules;
using core::TensorShape;
using core::TensorValue;

constexpr size_t kWeightContextBytes = 16ull * 1024ull * 1024ull;
constexpr size_t kPrefillArenaBytes = 64ull * 1024ull * 1024ull;
constexpr size_t kDecodeArenaBytes = 32ull * 1024ull * 1024ull;
constexpr size_t kGraphNodes = 32768;
constexpr int64_t kMaxRetainedPrefillSteps = 1024;

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
    TensorValue in_proj;
    TensorValue out_proj;
    TensorValue kernel;  // [hidden, kernel_size]
};

// Every layer is x += op(norm(x)); x += ffn(norm(x)). Attention layers map
// onto the framework's decoder layer (QK-norm, NEOX RoPE, SwiGLU); the
// short-conv layers are built here.
struct LayerWeights {
    bool attention = false;
    modules::DecoderLayerWeights decoder;
    ShortConvWeights conv;
};

struct BackboneWeights {
    std::unique_ptr<core::BackendWeightStore> store;
    TensorValue token_embedding;  // [vocab, hidden], also the output head
    TensorValue token_lookup;     // token_embedding, or an F16 copy (see load_weights)
    modules::NormWeights final_norm;
    std::vector<LayerWeights> layers;
};

// Whether the backend has a get_rows kernel for this tensor type.
bool backend_gathers(ggml_backend_t backend, ggml_type type) {
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx(ggml_init({4 * ggml_tensor_overhead(), nullptr, true}));
    if (ctx == nullptr) {
        throw std::runtime_error("failed to initialize the LFM2-Audio op probe context");
    }

    auto * table = ggml_new_tensor_2d(ctx.get(), type, ggml_blck_size(type), 1);
    auto * ids = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 1);
    return ggml_backend_supports_op(backend, ggml_get_rows(ctx.get(), table, ids));
}

BackboneWeights load_weights(
    const assets::TensorSource & source, const Lfm2BackboneConfig & config, core::ExecutionContext & execution) {
    BackboneWeights out;
    out.store = std::make_unique<core::BackendWeightStore>(
        execution.backend(), execution.backend_type(), "lfm2_audio.backbone.weights", kWeightContextBytes);
    auto & store = *out.store;
    const auto native = assets::TensorStorageType::Native;
    const int64_t d = config.hidden_size;
    const int64_t ff = config.intermediate_size;
    const int64_t hd = config.head_dim;

    out.token_embedding = store.load_tensor(source, "token_embd.weight", native, {config.vocab_size, d});
    // ggml's CUDA get_rows has no K-quant kernels, and Liquid's Q4_0 packages
    // store token_embd as Q6_K, so there the lookup reads an F16 copy. (llama.cpp
    // keeps its input embedding on the CPU instead.)
    out.token_lookup = backend_gathers(execution.backend(), out.token_embedding.tensor->type)
        ? out.token_embedding
        : store.load_tensor(source, "token_embd.weight", assets::TensorStorageType::F16, {config.vocab_size, d});
    out.final_norm = {store.load_f32_tensor(source, "token_embd_norm.weight", {d}), std::nullopt};

    for (int64_t layer = 0; layer < config.num_layers(); ++layer) {
        const std::string p = "blk." + std::to_string(layer) + ".";
        LayerWeights w;
        w.attention = config.is_attention_layer(layer);

        w.decoder.input_norm = {store.load_f32_tensor(source, p + "attn_norm.weight", {d}), std::nullopt};
        w.decoder.post_norm = {store.load_f32_tensor(source, p + "ffn_norm.weight", {d}), std::nullopt};
        w.decoder.mlp.gate_proj = {store.load_tensor(source, p + "ffn_gate.weight", native, {ff, d}), std::nullopt};
        w.decoder.mlp.up_proj = {store.load_tensor(source, p + "ffn_up.weight", native, {ff, d}), std::nullopt};
        w.decoder.mlp.down_proj = {store.load_tensor(source, p + "ffn_down.weight", native, {d, ff}), std::nullopt};

        if (w.attention) {
            const int64_t kv = config.kv_heads[static_cast<size_t>(layer)] * hd;
            w.decoder.self_attention.q_weight = store.load_tensor(source, p + "attn_q.weight", native, {config.num_attention_heads * hd, d});
            w.decoder.self_attention.k_weight = store.load_tensor(source, p + "attn_k.weight", native, {kv, d});
            w.decoder.self_attention.v_weight = store.load_tensor(source, p + "attn_v.weight", native, {kv, d});
            w.decoder.self_attention.out_weight = store.load_tensor(source, p + "attn_output.weight", native, {d, config.num_attention_heads * hd});
            w.decoder.q_norm = {store.load_f32_tensor(source, p + "attn_q_norm.weight", {hd}), std::nullopt};
            w.decoder.k_norm = {store.load_f32_tensor(source, p + "attn_k_norm.weight", {hd}), std::nullopt};
        } else {
            w.conv.in_proj = store.load_tensor(source, p + "shortconv.in_proj.weight", native, {3 * d, d});
            w.conv.out_proj = store.load_tensor(source, p + "shortconv.out_proj.weight", native, {d, d});
            w.conv.kernel = store.load_f32_tensor(source, p + "shortconv.conv.weight", {d, config.conv_kernel_size});
        }

        out.layers.push_back(std::move(w));
    }

    store.upload();
    return out;
}

modules::DecoderLayerConfig attention_layer_config(const Lfm2BackboneConfig & config, int64_t layer) {
    modules::DecoderLayerConfig out;
    out.hidden_size = config.hidden_size;
    out.num_attention_heads = config.num_attention_heads;
    out.num_key_value_heads = config.kv_heads[static_cast<size_t>(layer)];
    out.head_dim = config.head_dim;
    out.intermediate_size = config.intermediate_size;
    out.rms_norm_eps = config.rms_norm_eps;
    out.rope_theta = config.rope_theta;
    out.rope_type = GGML_ROPE_TYPE_NEOX;
    out.use_qk_norm = true;
    out.runtime.static_cache.update_mode = modules::DecoderStaticCacheUpdateMode::DirectSetRows;
    return out;
}

TensorValue rms_norm(core::ModuleBuildContext & ctx, const TensorValue & x, const modules::NormWeights & weights, const Lfm2BackboneConfig & config) {
    return modules::RMSNormModule({config.hidden_size, config.rms_norm_eps, true, false}).build(ctx, x, weights);
}

TensorValue contiguous(core::ModuleBuildContext & ctx, const TensorValue & x) {
    return core::wrap_tensor(ggml_cont(ctx.ggml, x.tensor), x.shape, x.type);
}

struct ShortConvInput {
    TensorValue gate;       // C, [1, steps, hidden]
    TensorValue conv_in;    // B * x, transposed to [1, hidden, steps]
};

// in_proj splits into B, C and x; the conv runs over B * x and its output is
// gated by C (transformers Lfm2ShortConv).
ShortConvInput short_conv_input(core::ModuleBuildContext & ctx, const TensorValue & normed, const ShortConvWeights & weights, int64_t d) {
    auto bcx = modules::LinearModule({d, 3 * d, false}).build(ctx, normed, {weights.in_proj, std::nullopt});
    auto b = contiguous(ctx, modules::SliceModule({2, 0, d}).build(ctx, bcx));
    auto c = contiguous(ctx, modules::SliceModule({2, d, d}).build(ctx, bcx));
    auto x = contiguous(ctx, modules::SliceModule({2, 2 * d, d}).build(ctx, bcx));
    auto bx = modules::MulModule{}.build(ctx, b, x);
    bx = modules::TransposeModule({{0, 2, 1}, 3}).build(ctx, bx);
    return {c, contiguous(ctx, bx)};
}

TensorValue short_conv_output(
    core::ModuleBuildContext & ctx,
    const TensorValue & residual,
    const TensorValue & conv,
    const TensorValue & gate,
    const ShortConvWeights & weights,
    int64_t d) {
    auto y = modules::MulModule{}.build(ctx, gate, conv);
    y = modules::LinearModule({d, d, false}).build(ctx, y, {weights.out_proj, std::nullopt});
    return modules::AddModule{}.build(ctx, residual, y);
}

TensorValue feed_forward(core::ModuleBuildContext & ctx, const TensorValue & x, const LayerWeights & weights, const Lfm2BackboneConfig & config) {
    auto h = rms_norm(ctx, x, weights.decoder.post_norm, config);

    modules::GatedFeedForwardConfig ff_config;
    ff_config.hidden_size = config.hidden_size;
    ff_config.intermediate_size = config.intermediate_size;
    ff_config.activation = modules::GatedFeedForwardActivation::Silu;
    h = modules::GatedFeedForwardModule(ff_config).build(
        ctx, h, {weights.decoder.mlp.gate_proj, weights.decoder.mlp.up_proj, weights.decoder.mlp.down_proj});
    return modules::AddModule{}.build(ctx, x, h);
}

TensorValue conv_kernel(core::ModuleBuildContext & ctx, const ShortConvWeights & weights, const Lfm2BackboneConfig & config) {
    return core::reshape_tensor(ctx, weights.kernel, TensorShape::from_dims({config.hidden_size, config.conv_kernel_size}));
}

TensorValue logits_from_last_step(
    core::ModuleBuildContext & ctx, const TensorValue & x, const BackboneWeights & weights, const Lfm2BackboneConfig & config) {
    const int64_t steps = x.shape.dims[1];
    auto last = steps == 1 ? x : contiguous(ctx, modules::SliceModule({1, steps - 1, 1}).build(ctx, x));
    last = rms_norm(ctx, last, weights.final_norm, config);
    return modules::LinearModule({config.hidden_size, config.vocab_size, false})
        .build(ctx, last, {weights.token_embedding, std::nullopt});
}

struct PrefillState {
    std::vector<float> logits;
    runtime::TransformerKVState kv;
    std::vector<std::vector<float>> conv_tails;  // per short-conv layer, [hidden][kernel - 1]
};

class PrefillGraph {
public:
    PrefillGraph(const BackboneWeights & weights, const Lfm2BackboneConfig & config, core::ExecutionContext & execution,
                 int64_t steps, int64_t audio_tokens)
        : config_(config), execution_(execution), steps_(steps), audio_tokens_(audio_tokens) {
        ctx_.reset(ggml_init({kPrefillArenaBytes, nullptr, true}));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize the LFM2-Audio prefill graph context");
        }

        auto * g = ctx_.get();
        core::ModuleBuildContext ctx{g, "lfm2_audio.prefill", execution.backend_type()};
        const int64_t d = config.hidden_size;
        const int64_t k = config.conv_kernel_size;

        token_ids_ = ggml_new_tensor_1d(g, GGML_TYPE_I32, steps);
        ggml_set_input(token_ids_);
        auto x = modules::EmbeddingModule({config.vocab_size, d})
                     .build(ctx, core::wrap_tensor(token_ids_, TensorShape::from_dims({steps}), GGML_TYPE_I32), weights.token_lookup);

        if (audio_tokens > 0) {
            audio_embeddings_ = ggml_new_tensor_2d(g, GGML_TYPE_F32, d, audio_tokens);
            audio_positions_ = ggml_new_tensor_1d(g, GGML_TYPE_I64, audio_tokens);
            ggml_set_input(audio_embeddings_);
            ggml_set_input(audio_positions_);
            x = core::wrap_tensor(ggml_set_rows(g, x.tensor, audio_embeddings_, audio_positions_), x.shape, GGML_TYPE_F32);
        }

        x = core::reshape_tensor(ctx, x, TensorShape::from_dims({1, steps, d}));

        positions_ = ggml_new_tensor_1d(g, GGML_TYPE_I32, steps);
        ggml_set_input(positions_);
        const auto positions = core::wrap_tensor(positions_, TensorShape::from_dims({steps}), GGML_TYPE_I32);

        for (int64_t layer = 0; layer < config.num_layers(); ++layer) {
            const auto & w = weights.layers[static_cast<size_t>(layer)];
            if (w.attention) {
                // No mask argument: without one the layer applies ggml_diag_mask_inf,
                // which is the causal mask.
                auto out = modules::DecoderLayerModule(attention_layer_config(config, layer)).build(ctx, x, positions, w.decoder);
                x = out.output;
                keys_.push_back(pin_output(out.key.tensor));
                values_.push_back(pin_output(out.value.tensor));
                continue;
            }

            auto in = short_conv_input(ctx, rms_norm(ctx, x, w.decoder.input_norm, config), w.conv, d);
            // Left-pad with the kernel's history: prefill starts from zeros.
            auto padded = core::wrap_tensor(
                ggml_pad_ext(g, in.conv_in.tensor, static_cast<int>(k - 1), 0, 0, 0, 0, 0, 0, 0),
                TensorShape::from_dims({1, d, steps + k - 1}),
                GGML_TYPE_F32);
            auto conv = core::wrap_tensor(
                ggml_ssm_conv(g, padded.tensor, conv_kernel(ctx, w.conv, config).tensor),
                TensorShape::from_dims({1, steps, d}),
                GGML_TYPE_F32);

            x = short_conv_output(ctx, x, conv, in.gate, w.conv, d);
            x = feed_forward(ctx, x, w, config);

            auto tail = contiguous(ctx, modules::SliceModule({2, steps, k - 1}).build(ctx, padded));
            conv_tails_.push_back(pin_output(tail.tensor));
        }

        logits_ = logits_from_last_step(ctx, x, weights, config).tensor;
        ggml_set_output(logits_);

        graph_ = ggml_new_graph_custom(g, kGraphNodes, false);
        ggml_build_forward_expand(graph_, logits_);
        for (auto * t : keys_) ggml_build_forward_expand(graph_, t);
        for (auto * t : values_) ggml_build_forward_expand(graph_, t);
        for (auto * t : conv_tails_) ggml_build_forward_expand(graph_, t);
        core::validate_backend_graph_supported(execution.backend(), graph_, "LFM2-Audio prefill graph");

        allocator_.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend())));
        if (allocator_ == nullptr || !ggml_gallocr_alloc_graph(allocator_.get(), graph_)) {
            throw runtime::CapacityError(
                "LFM2-Audio prefill graph does not fit in device memory at " + std::to_string(steps) + " prompt steps");
        }
    }

    ~PrefillGraph() { core::release_backend_graph_resources(execution_.backend(), graph_, true); }

    [[nodiscard]] bool matches(int64_t steps, int64_t audio_tokens) const {
        return steps_ == steps && audio_tokens_ == audio_tokens;
    }

    PrefillState run(const Lfm2Prompt & prompt, const Lfm2AudioEmbeddings & audio) {
        const auto positions = modules::decoder_position_ids(steps_);
        ggml_backend_tensor_set(positions_, positions.data(), 0, positions.size() * sizeof(int32_t));
        ggml_backend_tensor_set(token_ids_, prompt.input_ids.data(), 0, prompt.input_ids.size() * sizeof(int32_t));

        if (audio_tokens_ > 0) {
            const std::vector<int64_t> rows(prompt.audio_positions.begin(), prompt.audio_positions.end());
            ggml_backend_tensor_set(audio_embeddings_, audio.values.data(), 0, audio.values.size() * sizeof(float));
            ggml_backend_tensor_set(audio_positions_, rows.data(), 0, rows.size() * sizeof(int64_t));
        }

        core::set_backend_threads(execution_.backend(), std::max(1, execution_.config().threads));
        const ggml_status status = core::compute_backend_graph(execution_.backend(), graph_);
        ggml_backend_synchronize(execution_.backend());
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("LFM2-Audio prefill graph compute failed");
        }

        PrefillState out;
        out.logits.resize(static_cast<size_t>(config_.vocab_size));
        ggml_backend_tensor_get(logits_, out.logits.data(), 0, out.logits.size() * sizeof(float));

        out.kv.current_end = steps_;
        for (size_t i = 0; i < keys_.size(); ++i) {
            runtime::KVLayerState layer;
            layer.valid_steps = steps_;
            layer.key.resize(ggml_nelements(keys_[i]));
            layer.value.resize(ggml_nelements(values_[i]));
            ggml_backend_tensor_get(keys_[i], layer.key.data(), 0, layer.key.size() * sizeof(float));
            ggml_backend_tensor_get(values_[i], layer.value.data(), 0, layer.value.size() * sizeof(float));
            out.kv.layers.push_back(std::move(layer));
        }

        for (auto * tail : conv_tails_) {
            std::vector<float> values(static_cast<size_t>(ggml_nelements(tail)));
            ggml_backend_tensor_get(tail, values.data(), 0, values.size() * sizeof(float));
            out.conv_tails.push_back(std::move(values));
        }

        return out;
    }

private:
    // Copy out of the allocator's scratch so run() can read it back.
    ggml_tensor * pin_output(ggml_tensor * t) {
        auto * copy = ggml_cpy(ctx_.get(), t, ggml_dup_tensor(ctx_.get(), t));
        ggml_set_output(copy);
        return copy;
    }

    const Lfm2BackboneConfig & config_;
    core::ExecutionContext & execution_;
    int64_t steps_ = 0;
    int64_t audio_tokens_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * token_ids_ = nullptr;
    ggml_tensor * audio_embeddings_ = nullptr;
    ggml_tensor * audio_positions_ = nullptr;
    ggml_tensor * positions_ = nullptr;
    ggml_tensor * logits_ = nullptr;
    std::vector<ggml_tensor *> keys_;
    std::vector<ggml_tensor *> values_;
    std::vector<ggml_tensor *> conv_tails_;
    ggml_cgraph * graph_ = nullptr;
    std::unique_ptr<std::remove_pointer_t<ggml_gallocr_t>, GgmlGallocrDeleter> allocator_;
};

class DecodeGraph {
public:
    DecodeGraph(const BackboneWeights & weights, const Lfm2BackboneConfig & config, core::ExecutionContext & execution, int64_t cache_steps)
        : config_(config), execution_(execution), cache_steps_(cache_steps) {
        ctx_.reset(ggml_init({kDecodeArenaBytes, nullptr, true}));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize the LFM2-Audio decode graph context");
        }

        auto * g = ctx_.get();
        core::ModuleBuildContext ctx{g, "lfm2_audio.decode", execution.backend_type()};
        const int64_t d = config.hidden_size;
        const int64_t k = config.conv_kernel_size;

        token_ = ggml_new_tensor_1d(g, GGML_TYPE_I32, 1);
        positions_ = ggml_new_tensor_1d(g, GGML_TYPE_I32, 1);
        cache_slot_ = ggml_new_tensor_1d(g, GGML_TYPE_I32, 1);
        mask_ = ggml_new_tensor_4d(g, GGML_TYPE_F16, cache_steps, 1, 1, 1);

        auto x = modules::EmbeddingModule({config.vocab_size, d})
                     .build(ctx, core::wrap_tensor(token_, TensorShape::from_dims({1}), GGML_TYPE_I32), weights.token_lookup);
        x = core::reshape_tensor(ctx, x, TensorShape::from_dims({1, 1, d}));

        const auto positions = core::wrap_tensor(positions_, TensorShape::from_dims({1}), GGML_TYPE_I32);
        const auto slot = core::wrap_tensor(cache_slot_, TensorShape::from_dims({1}), GGML_TYPE_I32);
        const auto mask = core::wrap_tensor(mask_, TensorShape::from_dims({1, 1, 1, cache_steps}), GGML_TYPE_F16);

        graph_ = ggml_new_graph_custom(g, kGraphNodes, false);

        std::vector<TensorValue> keys;
        std::vector<TensorValue> values;
        int64_t step_elems = 0;
        for (int64_t layer = 0; layer < config.num_layers(); ++layer) {
            const auto & w = weights.layers[static_cast<size_t>(layer)];
            if (w.attention) {
                const int64_t kv_heads = config.kv_heads[static_cast<size_t>(layer)];
                if (step_elems != 0 && step_elems != kv_heads * config.head_dim) {
                    throw std::runtime_error("LFM2-Audio attention layers must share one KV head count");
                }

                step_elems = kv_heads * config.head_dim;
                keys.push_back(core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({1, cache_steps, kv_heads, config.head_dim})));
                values.push_back(core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({1, cache_steps, kv_heads, config.head_dim})));

                x = modules::DecoderLayerModule(attention_layer_config(config, layer))
                        .build_with_static_cache_tail(ctx, graph_, x, positions, w.decoder, keys.back(), values.back(), slot, mask)
                        .output;
                continue;
            }

            auto tail = core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({1, d, k - 1}));
            conv_tails_.push_back(tail.tensor);

            auto in = short_conv_input(ctx, rms_norm(ctx, x, w.decoder.input_norm, config), w.conv, d);
            auto window = modules::ConcatModule({2}).build(ctx, tail, in.conv_in);
            auto conv = core::wrap_tensor(
                ggml_ssm_conv(g, window.tensor, conv_kernel(ctx, w.conv, config).tensor),
                TensorShape::from_dims({1, 1, d}),
                GGML_TYPE_F32);

            x = short_conv_output(ctx, x, conv, in.gate, w.conv, d);
            x = feed_forward(ctx, x, w, config);

            auto next_tail = contiguous(ctx, modules::SliceModule({2, 1, k - 1}).build(ctx, window));
            ggml_build_forward_expand(graph_, ggml_cpy(g, next_tail.tensor, tail.tensor));
        }

        logits_ = logits_from_last_step(ctx, x, weights, config).tensor;
        ggml_set_output(logits_);
        ggml_build_forward_expand(graph_, logits_);
        core::validate_backend_graph_supported(execution.backend(), graph_, "LFM2-Audio decode graph");

        buffer_.reset(ggml_backend_alloc_ctx_tensors(g, execution.backend()));
        if (buffer_ == nullptr) {
            throw runtime::CapacityError(
                "LFM2-Audio decode graph does not fit in device memory at " + std::to_string(cache_steps) + " cache steps");
        }

        cache_ = runtime::TransformerKVCache(cache_steps, step_elems, keys, values);
        mask_scratch_.assign(static_cast<size_t>(cache_steps), ggml_fp32_to_fp16(-INFINITY));
    }

    ~DecodeGraph() { core::release_backend_graph_resources(execution_.backend(), graph_, true); }

    // Every step attends over the whole cache, so a cache sized for an
    // unusually long request is replaced rather than reused.
    [[nodiscard]] bool fits(int64_t required_steps) const {
        return cache_steps_ >= required_steps && cache_steps_ <= 2 * required_steps;
    }

    void import_state(const PrefillState & state) {
        cache_.import_state(state.kv);
        if (state.conv_tails.size() != conv_tails_.size()) {
            throw std::runtime_error("LFM2-Audio conv state does not match the decode graph");
        }

        for (size_t i = 0; i < conv_tails_.size(); ++i) {
            ggml_backend_tensor_set(conv_tails_[i], state.conv_tails[i].data(), 0, state.conv_tails[i].size() * sizeof(float));
        }
    }

    std::vector<float> run_step(int32_t token) {
        if (cache_.valid_steps() >= cache_steps_) {
            throw std::runtime_error("LFM2-Audio decode cache exhausted");
        }

        const auto position = static_cast<int32_t>(cache_.current_end());
        const auto slot = static_cast<int32_t>(cache_.valid_steps());
        ggml_backend_tensor_set(token_, &token, 0, sizeof(int32_t));
        ggml_backend_tensor_set(positions_, &position, 0, sizeof(int32_t));
        ggml_backend_tensor_set(cache_slot_, &slot, 0, sizeof(int32_t));
        modules::write_decoder_cached_step_mask(mask_, mask_scratch_, cache_steps_, cache_.valid_steps(), cache_.valid_steps());

        core::set_backend_threads(execution_.backend(), std::max(1, execution_.config().threads));
        const ggml_status status = core::compute_backend_graph(execution_.backend(), graph_);
        ggml_backend_synchronize(execution_.backend());
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("LFM2-Audio decode graph compute failed");
        }

        std::vector<float> logits(static_cast<size_t>(config_.vocab_size));
        ggml_backend_tensor_get(logits_, logits.data(), 0, logits.size() * sizeof(float));
        cache_.advance_after_direct_append(1);
        return logits;
    }

private:
    const Lfm2BackboneConfig & config_;
    core::ExecutionContext & execution_;
    int64_t cache_steps_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * token_ = nullptr;
    ggml_tensor * positions_ = nullptr;
    ggml_tensor * cache_slot_ = nullptr;
    ggml_tensor * mask_ = nullptr;
    ggml_tensor * logits_ = nullptr;
    std::vector<ggml_tensor *> conv_tails_;
    std::vector<ggml_fp16_t> mask_scratch_;
    runtime::TransformerKVCache cache_;
    ggml_cgraph * graph_ = nullptr;
    std::unique_ptr<std::remove_pointer_t<ggml_backend_buffer_t>, GgmlBufferDeleter> buffer_;
};

// A backend that overflows or computes garbage shows up as NaN logits, and
// max_element over them returns token 0, which decodes to nothing.
int32_t greedy_token(const std::vector<float> & logits) {
    if (!std::all_of(logits.begin(), logits.end(), [](float value) { return std::isfinite(value); })) {
        throw std::runtime_error("LFM2-Audio backbone produced non-finite logits");
    }

    return static_cast<int32_t>(std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
}

void validate_prompt(const Lfm2Prompt & prompt, const Lfm2BackboneConfig & config) {
    for (const int32_t id : prompt.input_ids) {
        if (id < 0 || id >= config.vocab_size) {
            throw std::runtime_error("LFM2-Audio prompt token id " + std::to_string(id) + " is outside the vocabulary");
        }
    }

    const auto steps = static_cast<int32_t>(prompt.input_ids.size());
    for (size_t i = 0; i < prompt.audio_positions.size(); ++i) {
        const int32_t position = prompt.audio_positions[i];
        if (position < 0 || position >= steps || (i > 0 && position <= prompt.audio_positions[i - 1])) {
            throw std::runtime_error("LFM2-Audio audio positions must increase and stay inside the prompt");
        }
    }
}

}  // namespace

struct Lfm2BackboneRuntime::Impl {
    Impl(std::shared_ptr<const assets::TensorSource> source_in, const Lfm2BackboneConfig & config_in, core::ExecutionContext & execution_in)
        : source(std::move(source_in)),
          config(config_in),
          execution(execution_in),
          weights(load_weights(*source, config, execution_in)) {}

    std::shared_ptr<const assets::TensorSource> source;
    Lfm2BackboneConfig config;
    core::ExecutionContext & execution;
    BackboneWeights weights;
    std::unique_ptr<PrefillGraph> prefill;
    std::unique_ptr<DecodeGraph> decode;
};

Lfm2BackboneRuntime::Lfm2BackboneRuntime(
    std::shared_ptr<const assets::TensorSource> source,
    const Lfm2BackboneConfig & config,
    core::ExecutionContext & execution)
    : impl_(std::make_unique<Impl>(std::move(source), config, execution)) {}

Lfm2BackboneRuntime::~Lfm2BackboneRuntime() = default;

Lfm2GenerationResult Lfm2BackboneRuntime::generate(
    const Lfm2Prompt & prompt,
    const Lfm2AudioEmbeddings & audio,
    const Lfm2GenerationOptions & options) {
    const auto & config = impl_->config;
    const auto steps = static_cast<int64_t>(prompt.input_ids.size());
    const auto audio_tokens = static_cast<int64_t>(prompt.audio_positions.size());
    if (steps == 0 || options.max_new_tokens <= 0) {
        throw std::runtime_error("LFM2-Audio generation needs a prompt and a positive token budget");
    }

    if (audio_tokens != audio.tokens || audio.values.size() != static_cast<size_t>(audio.tokens * config.hidden_size)) {
        throw std::runtime_error("LFM2-Audio audio embeddings do not match the prompt's audio positions");
    }

    // Checked here rather than left to the logits: ggml's CPU RMSNorm
    // (ggml_compute_forward_rms_norm_f32) asserts on NaN input in debug builds.
    if (!std::all_of(audio.values.begin(), audio.values.end(), [](float value) { return std::isfinite(value); })) {
        throw std::runtime_error("LFM2-Audio encoder produced non-finite audio embeddings");
    }

    if (options.max_new_tokens > config.context_length - steps) {
        throw runtime::CapacityError(
            "LFM2-Audio request needs " + std::to_string(steps) + " prompt steps plus max_tokens, more than the " +
            std::to_string(config.context_length) + "-token context");
    }

    validate_prompt(prompt, config);

    const auto prefill_start = std::chrono::steady_clock::now();
    if (impl_->prefill == nullptr || !impl_->prefill->matches(steps, audio_tokens)) {
        impl_->prefill.reset();
        impl_->prefill = std::make_unique<PrefillGraph>(impl_->weights, config, impl_->execution, steps, audio_tokens);
    }

    auto state = impl_->prefill->run(prompt, audio);
    // The graph holds steps^2 attention scores per head. Only graphs the size
    // of a default 30 s chunk are worth keeping for the next request.
    if (steps > kMaxRetainedPrefillSteps) {
        impl_->prefill.reset();
    }

    debug::timing_log_scalar("lfm2_audio.prefill.ms", engine::debug::elapsed_ms(prefill_start));

    // The last generated token is never fed back, hence the - 1.
    const int64_t required = steps + options.max_new_tokens - 1;
    if (impl_->decode == nullptr || !impl_->decode->fits(required)) {
        impl_->decode.reset();
        impl_->decode = std::make_unique<DecodeGraph>(impl_->weights, config, impl_->execution, std::max<int64_t>(required, steps + 1));
    }

    impl_->decode->import_state(state);

    Lfm2GenerationResult out;
    out.prefill_logits = state.logits;
    std::vector<float> logits = std::move(state.logits);

    const auto decode_start = std::chrono::steady_clock::now();
    for (int64_t step = 0; step < options.max_new_tokens; ++step) {
        const int32_t token = greedy_token(logits);
        if (std::find(options.stop_token_ids.begin(), options.stop_token_ids.end(), token) != options.stop_token_ids.end()) {
            out.stopped = true;
            break;
        }

        out.tokens.push_back(token);
        if (step + 1 == options.max_new_tokens) {
            break;
        }

        logits = impl_->decode->run_step(token);
    }

    debug::timing_log_scalar("lfm2_audio.decode.ms", engine::debug::elapsed_ms(decode_start));
    return out;
}

}  // namespace engine::community_models::lfm2_audio
