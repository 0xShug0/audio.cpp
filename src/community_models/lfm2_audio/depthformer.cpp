#include "engine/community_models/lfm2_audio/depthformer.h"

#include "lfm2_blocks.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/transformers/causal_decoder.h"
#include "engine/framework/modules/transformers/decoder.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <memory>
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
using lfm2_blocks::GgmlBufferDeleter;
using lfm2_blocks::GgmlContextDeleter;

constexpr size_t kWeightContextBytes = 4ull * 1024ull * 1024ull;
constexpr size_t kGraphArenaBytes = 16ull * 1024ull * 1024ull;
constexpr size_t kGraphNodes = 4096;

// One embedding table per codebook: its input embedding for the next step,
// and the norm and head that give its logits (SharedEmbedding,
// model/transformer.py; not tied in these checkpoints).
struct CodebookWeights {
    TensorValue embedding;  // [audio_vocab_size, hidden]
    modules::NormWeights norm;
    TensorValue to_logits;  // [audio_vocab_size, hidden]
};

struct DepthformerWeights {
    std::unique_ptr<WeightStores> stores;
    TensorValue input_weight;  // depth_linear, [codebooks * hidden, input_size]
    TensorValue input_bias;    // [codebooks * hidden]
    std::vector<modules::DecoderLayerWeights> layers;
    std::vector<CodebookWeights> codebooks;
};

DepthformerWeights load_weights(
    const assets::TensorSource & source, const Lfm2DepthformerConfig & config, core::ExecutionContext & execution, bool cpu_repack) {
    DepthformerWeights out;
    out.stores = std::make_unique<WeightStores>(execution, "lfm2_audio.depthformer.weights", kWeightContextBytes, cpu_repack);
    auto & store = out.stores->plain();
    const auto native = assets::TensorStorageType::Native;
    const int64_t d = config.hidden_size;
    const int64_t ff = config.intermediate_size;
    const int64_t hd = config.head_dim;

    // The matrices below go through load_matmul, being only ever src0 of
    // LinearModule's ggml_mul_mat; depth_linear, whose rows each step views,
    // and the embedding tables, gathered, are not.
    out.input_weight = store.load_tensor(source, "depth_linear.weight", native, {config.codebooks * d, config.input_size});
    out.input_bias = store.load_f32_tensor(source, "depth_linear.bias", {config.codebooks * d});

    for (int64_t layer = 0; layer < config.num_layers; ++layer) {
        const std::string p = "depthformer.layers." + std::to_string(layer) + ".";
        modules::DecoderLayerWeights w;
        w.input_norm = {store.load_f32_tensor(source, p + "operator_norm.weight", {d}), std::nullopt};
        w.self_attention.qkv_weight =
            out.stores->load_matmul(source, p + "operator.qkv_proj.weight", {d + 2 * config.num_kv_heads * hd, d});
        w.self_attention.out_weight = out.stores->load_matmul(source, p + "operator.out_proj.weight", {d, d});
        w.q_norm = {store.load_f32_tensor(source, p + "operator.attention.q_layernorm.weight", {hd}), std::nullopt};
        w.k_norm = {store.load_f32_tensor(source, p + "operator.attention.k_layernorm.weight", {hd}), std::nullopt};
        w.post_norm = {store.load_f32_tensor(source, p + "ffn_norm.weight", {d}), std::nullopt};
        // GLU: w2(silu(w1(x)) * w3(x)).
        w.mlp.gate_proj = {out.stores->load_matmul(source, p + "feed_forward.w1.weight", {ff, d}), std::nullopt};
        w.mlp.up_proj = {out.stores->load_matmul(source, p + "feed_forward.w3.weight", {ff, d}), std::nullopt};
        w.mlp.down_proj = {out.stores->load_matmul(source, p + "feed_forward.w2.weight", {d, ff}), std::nullopt};
        out.layers.push_back(std::move(w));
    }

    for (int64_t codebook = 0; codebook < config.codebooks; ++codebook) {
        const std::string p = "depth_embeddings." + std::to_string(codebook) + ".";
        CodebookWeights w;
        w.embedding = store.load_tensor(source, p + "embedding.weight", native, {config.audio_vocab_size, d});
        w.norm = {store.load_f32_tensor(source, p + "embedding_norm.weight", {d}), std::nullopt};
        w.to_logits = out.stores->load_matmul(source, p + "to_logits.weight", {config.audio_vocab_size, d});
        out.codebooks.push_back(std::move(w));
    }

    out.stores->upload();
    return out;
}

modules::DecoderLayerConfig layer_config(const Lfm2DepthformerConfig & config) {
    modules::DecoderLayerConfig out;
    out.hidden_size = config.hidden_size;
    out.num_attention_heads = config.num_heads;
    out.num_key_value_heads = config.num_kv_heads;
    out.head_dim = config.head_dim;
    out.intermediate_size = config.intermediate_size;
    out.rms_norm_eps = config.rms_norm_eps;
    out.rope_theta = config.rope_theta;
    // apply_rotary_emb rotates adjacent pairs, GGML's "normal" RoPE, unlike
    // the backbone's NEOX halves.
    out.rope_type = GGML_ROPE_TYPE_NORMAL;
    out.qkv_layout = modules::DecoderQKVLayout::PackedQKV;
    out.use_qk_norm = true;
    out.runtime.static_cache.update_mode = modules::DecoderStaticCacheUpdateMode::DirectSetRows;
    // One query over at most `codebooks` cached steps gains nothing from
    // flash attention, and ggml's CUDA kernel has no 32-wide heads.
    out.runtime.attention.allow_flash_attention = false;
    return out;
}

// One graph per codebook step. They share the input hidden state and the KV
// cache (one slot per step), so a frame runs them in order.
class FrameGraphs {
public:
    FrameGraphs(const DepthformerWeights & weights, const Lfm2DepthformerConfig & config, core::ExecutionContext & execution)
        : config_(config), execution_(execution) {
        ctx_.reset(ggml_init({kGraphArenaBytes, nullptr, true}));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize the LFM2-Audio depthformer graph context");
        }

        auto * g = ctx_.get();
        core::ModuleBuildContext ctx{g, "lfm2_audio.depthformer", execution.backend_type()};
        const int64_t d = config.hidden_size;
        const int64_t steps = config.codebooks;

        hidden_ = ggml_new_tensor_1d(g, GGML_TYPE_F32, config.input_size);
        ggml_set_input(hidden_);
        const auto hidden = core::wrap_tensor(hidden_, TensorShape::from_dims({1, config.input_size}), GGML_TYPE_F32);

        std::vector<TensorValue> keys;
        std::vector<TensorValue> values;
        for (int64_t layer = 0; layer < config.num_layers; ++layer) {
            keys.push_back(core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({1, steps, config.num_kv_heads, config.head_dim})));
            values.push_back(core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({1, steps, config.num_kv_heads, config.head_dim})));
            caches_.push_back(keys.back().tensor);
            caches_.push_back(values.back().tensor);
        }

        const modules::DecoderLayerModule layer_module(layer_config(config));
        for (int64_t step = 0; step < steps; ++step) {
            Step s;
            s.position = ggml_new_tensor_1d(g, GGML_TYPE_I32, 1);
            s.slot = ggml_new_tensor_1d(g, GGML_TYPE_I32, 1);
            s.mask = ggml_new_tensor_4d(g, GGML_TYPE_F16, steps, 1, 1, 1);
            s.graph = ggml_new_graph_custom(g, kGraphNodes, false);

            // depthformer_in[step]: rows [step * d, (step + 1) * d) of depth_linear.
            auto * w = weights.input_weight.tensor;
            auto * b = weights.input_bias.tensor;
            const auto slice = core::wrap_tensor(
                ggml_view_2d(g, w, w->ne[0], d, w->nb[1], static_cast<size_t>(step * d) * w->nb[1]),
                TensorShape::from_dims({d, config.input_size}),
                w->type);
            const auto bias = core::wrap_tensor(
                ggml_view_1d(g, b, d, static_cast<size_t>(step * d) * b->nb[0]), TensorShape::from_dims({d}), GGML_TYPE_F32);
            auto x = modules::LinearModule({config.input_size, d, true}).build(ctx, hidden, {slice, bias});

            if (step > 0) {
                s.previous_code = ggml_new_tensor_1d(g, GGML_TYPE_I32, 1);
                auto * previous = ggml_get_rows(g, weights.codebooks[static_cast<size_t>(step - 1)].embedding.tensor, s.previous_code);
                x = core::wrap_tensor(ggml_add(g, x.tensor, previous), x.shape, GGML_TYPE_F32);
            }

            x = core::reshape_tensor(ctx, x, TensorShape::from_dims({1, 1, d}));
            const auto position = core::wrap_tensor(s.position, TensorShape::from_dims({1}), GGML_TYPE_I32);
            const auto slot = core::wrap_tensor(s.slot, TensorShape::from_dims({1}), GGML_TYPE_I32);
            const auto mask = core::wrap_tensor(s.mask, TensorShape::from_dims({1, 1, 1, steps}), GGML_TYPE_F16);
            for (int64_t layer = 0; layer < config.num_layers; ++layer) {
                x = layer_module
                        .build_with_static_cache_tail(
                            ctx, s.graph, x, position, weights.layers[static_cast<size_t>(layer)], keys[static_cast<size_t>(layer)],
                            values[static_cast<size_t>(layer)], slot, mask)
                        .output;
            }

            const auto & head = weights.codebooks[static_cast<size_t>(step)];
            x = modules::RMSNormModule({d, config.rms_norm_eps, true, false}).build(ctx, x, head.norm);
            s.logits = modules::LinearModule({d, config.audio_vocab_size, false}).build(ctx, x, {head.to_logits, std::nullopt}).tensor;
            ggml_set_output(s.logits);
            ggml_build_forward_expand(s.graph, s.logits);
            core::validate_backend_graph_with_cpu_extra_buffers(execution.backend(), s.graph, "LFM2-Audio depthformer graph");
            steps_.push_back(s);
        }

        buffer_.reset(ggml_backend_alloc_ctx_tensors(g, execution.backend()));
        if (buffer_ == nullptr) {
            throw std::runtime_error("LFM2-Audio depthformer graphs do not fit in device memory");
        }

        ggml_backend_buffer_clear(buffer_.get(), 0);

        std::vector<ggml_fp16_t> scratch;
        for (int64_t step = 0; step < steps; ++step) {
            auto & s = steps_[static_cast<size_t>(step)];
            const auto index = static_cast<int32_t>(step);
            ggml_backend_tensor_set(s.position, &index, 0, sizeof(int32_t));
            ggml_backend_tensor_set(s.slot, &index, 0, sizeof(int32_t));
            modules::write_decoder_cached_step_mask(s.mask, scratch, steps, step, step);
        }
    }

    ~FrameGraphs() {
        for (const auto & s : steps_) {
            core::release_backend_graph_resources(execution_.backend(), s.graph, true);
        }
    }

    std::vector<int32_t> run(const std::vector<float> & hidden, const Lfm2DepthformerRuntime::PickCode & pick) {
        if (static_cast<int64_t>(hidden.size()) != config_.input_size) {
            throw std::runtime_error("LFM2-Audio depthformer input does not match the backbone hidden size");
        }

        ggml_backend_tensor_set(hidden_, hidden.data(), 0, hidden.size() * sizeof(float));
        // Each frame starts from an empty cache, as _sample_audio_frame's
        // does, so no frame or earlier request reaches it through the masked
        // slots.
        for (auto * cache : caches_) {
            ggml_backend_tensor_memset(cache, 0, 0, ggml_nbytes(cache));
        }
        core::set_backend_threads(execution_.backend(), std::max(1, execution_.config().threads));

        std::vector<int32_t> codes;
        std::vector<float> logits(static_cast<size_t>(config_.audio_vocab_size));
        for (size_t step = 0; step < steps_.size(); ++step) {
            const auto & s = steps_[step];
            if (step > 0) {
                ggml_backend_tensor_set(s.previous_code, &codes.back(), 0, sizeof(int32_t));
            }

            const ggml_status status = core::compute_backend_graph(execution_.backend(), s.graph);
            ggml_backend_synchronize(execution_.backend());
            if (status != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("LFM2-Audio depthformer graph compute failed");
            }

            ggml_backend_tensor_get(s.logits, logits.data(), 0, logits.size() * sizeof(float));
            if (!std::all_of(logits.begin(), logits.end(), [](float value) { return std::isfinite(value); })) {
                throw std::runtime_error("LFM2-Audio depthformer produced non-finite logits");
            }

            const int32_t code = pick(static_cast<int64_t>(step), logits);
            if (code < 0 || code >= config_.audio_vocab_size) {
                throw std::runtime_error("LFM2-Audio picked audio code " + std::to_string(code) + " outside the codebook");
            }

            codes.push_back(code);
        }

        return codes;
    }

private:
    struct Step {
        ggml_tensor * previous_code = nullptr;  // none for the first step
        ggml_tensor * position = nullptr;
        ggml_tensor * slot = nullptr;
        ggml_tensor * mask = nullptr;
        ggml_tensor * logits = nullptr;
        ggml_cgraph * graph = nullptr;
    };

    const Lfm2DepthformerConfig & config_;
    core::ExecutionContext & execution_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * hidden_ = nullptr;
    std::vector<ggml_tensor *> caches_;
    std::vector<Step> steps_;
    std::unique_ptr<std::remove_pointer_t<ggml_backend_buffer_t>, GgmlBufferDeleter> buffer_;
};

}  // namespace

struct Lfm2DepthformerRuntime::Impl {
    Impl(std::shared_ptr<const assets::TensorSource> source_in,
         const Lfm2DepthformerConfig & config_in,
         core::ExecutionContext & execution,
         bool cpu_repack)
        : source(std::move(source_in)),
          config(config_in),
          weights(load_weights(*source, config, execution, cpu_repack)),
          graphs(weights, config, execution) {}

    std::shared_ptr<const assets::TensorSource> source;
    Lfm2DepthformerConfig config;
    DepthformerWeights weights;
    FrameGraphs graphs;
};

Lfm2DepthformerRuntime::Lfm2DepthformerRuntime(
    std::shared_ptr<const assets::TensorSource> vocoder,
    const Lfm2DepthformerConfig & config,
    core::ExecutionContext & execution,
    bool cpu_repack)
    : impl_(std::make_unique<Impl>(std::move(vocoder), config, execution, cpu_repack)) {}

Lfm2DepthformerRuntime::~Lfm2DepthformerRuntime() = default;

std::vector<int32_t> Lfm2DepthformerRuntime::frame(const std::vector<float> & hidden, const PickCode & pick) {
    return impl_->graphs.run(hidden, pick);
}

}  // namespace engine::community_models::lfm2_audio
