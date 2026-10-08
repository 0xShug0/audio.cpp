#include "engine/community_models/whistle_asr/decoder.h"

#include "engine/community_models/whistle_asr/graph.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/modules/optimizations/fast_kv_modules.h"
#include "engine/framework/modules/transformers/causal_decoder.h"
#include "engine/framework/runtime/kv_cache.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

namespace engine::community_models::whistle_asr {
namespace {

using Clock = std::chrono::steady_clock;
using core::TensorShape;
using core::TensorValue;

constexpr size_t kGraphNodes = 16384;
// BOS, the language token and up to the text-token limit.
constexpr int64_t kCacheSteps = 2 + static_cast<int64_t>(kWhistleMaximumTextTokens);
// The projection history keeps two leading zero slots, so the step at position p
// writes slot p + 2 and reads the two previous steps from slots p + 1 and p.
constexpr int64_t kHistorySlots = kCacheSteps + 2;
constexpr int64_t kQueryWidth = kWhistleHeads * kWhistleQkDim;
constexpr int64_t kKeyWidth = kWhistleKvHeads * kWhistleQkDim;
constexpr int64_t kValueWidth = kWhistleKvHeads * kWhistleVDim;
// One raw self-attention projection row: [query | key | value] before tap mixing.
constexpr int64_t kRawWidth = kQueryWidth + kKeyWidth + kValueWidth;
constexpr int kProjectionTaps = 3;
constexpr int kEngramSites = 2;
constexpr int64_t kEngramTaps = 4;
constexpr int64_t kEngramSources = 4;
constexpr int64_t kEngramBuckets = 18432;
constexpr int64_t kEngramWidth = 128;
constexpr const char * kBlock = "stack/layers/block/";

float sigmoid(float value) {
    return value >= 0
        ? 1.0f / (1.0f + std::exp(-value))
        : std::exp(value) / (1.0f + std::exp(value));
}

struct DecoderLayerWeights {
    WhistleMhcWeights mhc;
    modules::NormWeights attn_norm;
    // Query, key and value projections stacked into one [608, 512] matrix.
    modules::LinearWeights qkv_proj;
    // Tap weights [3, 608]: row 0 scales the current projection, rows 1 and 2 the
    // projections of the previous two steps.
    TensorValue taps;
    modules::NormWeights q_norm;
    modules::NormWeights k_norm;
    modules::LinearWeights self_gate_proj;
    modules::LinearWeights self_out_proj;
    modules::NormWeights post_attn_norm;
    float attn_gate = 0.0f;
    modules::NormWeights cross_norm;
    modules::LinearWeights cross_q_proj;
    modules::NormWeights cross_q_norm;
    modules::LinearWeights cross_gate_proj;
    modules::LinearWeights cross_out_proj;
    modules::NormWeights post_cross_norm;
    float cross_gate = 0.0f;
    modules::NormWeights pre_hada_norm;
    WhistleHadamardWeights hadamard;
};

struct EngramWeights {
    modules::LinearWeights key_proj;
    modules::LinearWeights value_proj;
    TensorValue taps;
};

struct DecoderWeights {
    std::array<DecoderLayerWeights, kWhistleLayers> layers;
    // Tied input embedding and output projection, ggml [512, 8199].
    TensorValue embedding;
    modules::NormWeights final_norm;
    std::array<EngramWeights, kEngramSites> engrams;
    WhistleGraphConstants constants;
};

modules::LinearWeights stacked_projection(WhistleWeightLoader & loader, int layer) {
    const std::string prefix = std::string(kBlock) + "self_attn/";
    const std::array<std::pair<const char *, int64_t>, 3> parts = {{
        {"q_proj", kQueryWidth}, {"k_proj", kKeyWidth}, {"v_proj", kValueWidth}}};
    std::vector<float> weight;
    weight.reserve(static_cast<size_t>(kRawWidth * kWhistleDimension));
    for (const auto & [name, out] : parts) {
        const auto source = loader.values(prefix + name + "/kernel", {kWhistleDimension, out}, layer);
        for (int64_t col = 0; col < out; ++col) {
            for (int64_t row = 0; row < kWhistleDimension; ++row) {
                weight.push_back(source[static_cast<size_t>(row * out + col)]);
            }
        }
    }
    return {loader.store().make_f32(TensorShape::from_dims({kRawWidth, kWhistleDimension}), std::move(weight)),
        std::nullopt};
}

TensorValue stacked_taps(WhistleWeightLoader & loader, int layer) {
    const std::string prefix = std::string(kBlock) + "self_attn/";
    const std::array<std::tuple<const char *, int64_t, int64_t>, 3> parts = {{
        {"q_taps", kQueryWidth, 0}, {"k_taps", kKeyWidth, kQueryWidth}, {"v_taps", kValueWidth, kQueryWidth + kKeyWidth}}};
    std::vector<float> taps(static_cast<size_t>(kProjectionTaps * kRawWidth));
    for (const auto & [name, width, offset] : parts) {
        const auto source = loader.values(prefix + name, {kProjectionTaps, width}, layer);
        for (int64_t tap = 0; tap < kProjectionTaps; ++tap) {
            std::copy_n(source.begin() + static_cast<std::ptrdiff_t>(tap * width), width,
                taps.begin() + static_cast<std::ptrdiff_t>(tap * kRawWidth + offset));
        }
    }
    return loader.store().make_f32(TensorShape::from_dims({kProjectionTaps, kRawWidth}), std::move(taps));
}

DecoderWeights load_weights(core::BackendWeightStore & store, const WhistleAssets & assets) {
    WhistleWeightLoader loader(store, *assets.weights);
    const std::string block = kBlock;
    DecoderWeights weights;
    for (int layer = 0; layer < kWhistleLayers; ++layer) {
        auto & target = weights.layers[static_cast<size_t>(layer)];
        target.mhc = loader.mhc("stack/mhc_", layer);
        target.attn_norm = loader.norm(block + "ZCRMSNorm_0", kWhistleDimension, layer);
        target.qkv_proj = stacked_projection(loader, layer);
        target.taps = stacked_taps(loader, layer);
        target.q_norm = loader.norm(block + "self_attn/q_norm", kWhistleQkDim, layer);
        target.k_norm = loader.norm(block + "self_attn/k_norm", kWhistleQkDim, layer);
        target.self_gate_proj = loader.linear(block + "self_attn/gate_proj", kWhistleDimension, kWhistleDimension, layer);
        target.self_out_proj = loader.linear(block + "self_attn/out_proj", kWhistleDimension, kWhistleDimension, layer);
        target.post_attn_norm = loader.norm(block + "post_attn_norm", kWhistleDimension, layer);
        target.attn_gate = sigmoid(loader.scalar(block + "attn_gate", layer));
        target.cross_norm = loader.norm(block + "cross_norm", kWhistleDimension, layer);
        target.cross_q_proj = loader.linear(block + "cross_attn/q_proj", kWhistleDimension, kQueryWidth, layer);
        target.cross_q_norm = loader.norm(block + "cross_attn/q_norm", kWhistleQkDim, layer);
        target.cross_gate_proj = loader.linear(block + "cross_attn/gate_proj", kWhistleDimension, kWhistleDimension, layer);
        target.cross_out_proj = loader.linear(block + "cross_attn/out_proj", kWhistleDimension, kWhistleDimension, layer);
        target.post_cross_norm = loader.norm(block + "post_cross_norm", kWhistleDimension, layer);
        target.cross_gate = sigmoid(loader.scalar(block + "cross_gate", layer));
        target.pre_hada_norm = loader.norm(block + "pre_hada_norm", kWhistleDimension, layer);
        target.hadamard = loader.hadamard(block + "hadamard_mlp/", layer);
    }
    // The [8199, 512] checkpoint layout is already ggml [512, 8199], so it streams
    // straight from the model file without a staged host copy.
    weights.embedding = store.load_tensor(*assets.weights, "embedding/embedding",
        assets::TensorStorageType::F32, {kWhistleVocabulary, kWhistleDimension});
    weights.final_norm = loader.norm("stack/final_norm", kWhistleDimension);
    for (int site = 0; site < kEngramSites; ++site) {
        const std::string prefix = "engrams_" + std::to_string(site) + "/";
        auto & target = weights.engrams[static_cast<size_t>(site)];
        target.key_proj = loader.linear(prefix + "key_proj", kWhistleDimension, kWhistleDimension);
        target.value_proj = loader.linear(prefix + "value_proj", kWhistleDimension, kWhistleDimension);
        target.taps = store.make_f32(
            TensorShape::from_dims({kEngramTaps, kWhistleDimension}),
            loader.values(prefix + "taps", {kEngramTaps, kWhistleDimension}));
    }
    weights.constants = loader.constants(assets, 1);
    store.upload();
    return weights;
}

// Gathers the hashed n-gram rows for the newest token: tap t looks at the token
// 3 * t steps back, and each of the four sources hashes a bigram or trigram ending
// there into its own bucket table. Missing history leaves the slot at zero.
void engram_features(const std::vector<int32_t> & tokens, const std::vector<std::byte> & table, std::vector<float> & feature) {
    feature.assign(static_cast<size_t>(kEngramTaps * kWhistleDimension), 0.0f);
    const int64_t current = static_cast<int64_t>(tokens.size()) - 1;
    for (int64_t tap = 0; tap < kEngramTaps; ++tap) {
        const int64_t position = current - 3 * tap;
        if (position < 0) {
            continue;
        }
        for (int64_t source = 0; source < kEngramSources; ++source) {
            const int order = source < 2 ? 2 : 3;
            if (position < order - 1) {
                continue;
            }
            uint32_t hash = 0x9E3779B9u * static_cast<uint32_t>(source + 1);
            for (int history = 0; history < order; ++history) {
                const auto at = position - history;
                hash = (hash ^ static_cast<uint32_t>(at >= 0 ? tokens[static_cast<size_t>(at)] : 0)) * 0x01000193u;
            }
            hash ^= hash >> 15;
            const size_t row = (static_cast<size_t>(source) * kEngramBuckets + hash % kEngramBuckets) * kEngramWidth;
            std::memcpy(feature.data() + tap * kWhistleDimension + source * kEngramWidth,
                table.data() + row * sizeof(float), kEngramWidth * sizeof(float));
        }
    }
}

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept { ggml_free(ctx); }
};

struct GallocrDeleter {
    void operator()(ggml_gallocr * galloc) const noexcept { ggml_gallocr_free(galloc); }
};

ggml_tensor * column(ggml_context * ctx, ggml_tensor * matrix, int64_t index) {
    return ggml_view_2d(ctx, matrix, matrix->ne[0], 1, matrix->nb[1], static_cast<size_t>(index) * matrix->nb[1]);
}

}  // namespace

class WhistleDecoderRuntime::Graph {
public:
    Graph(std::shared_ptr<const WhistleAssets> assets, core::ExecutionContext & execution_context)
        : assets_(std::move(assets)),
          execution_context_(&execution_context),
          backend_(execution_context.backend()),
          weight_store_(
              execution_context.backend(), execution_context.backend_type(),
              "Whistle decoder weights", 8ull * 1024ull * 1024ull) {
        if (!assets_ || !assets_->weights) {
            throw std::invalid_argument("Whistle decoder needs verified model assets");
        }
        weights_ = load_weights(weight_store_, *assets_);
        // The runtime only accepts FP32 checkpoints, so the raw bytes are the f32
        // table; keeping them avoids a second full-size converted copy.
        for (int site = 0; site < kEngramSites; ++site) {
            auto table = assets_->weights->require_tensor_data("engrams_" + std::to_string(site) + "/embedding");
            if (table.metadata.shape != std::vector<int64_t>{kEngramSources, kEngramBuckets, kEngramWidth} ||
                table.bytes.size() != static_cast<size_t>(kEngramSources * kEngramBuckets * kEngramWidth) * sizeof(float)) {
                throw std::runtime_error("Whistle engram table has an unexpected shape");
            }
            engram_tables_[static_cast<size_t>(site)] = std::move(table.bytes);
        }
        allocate_state();
        build_graph();
    }

    ~Graph() {
        core::release_backend_graph_resources(backend_, graph_);
        plan_.reset();
        galloc_.reset();
        if (state_buffer_ != nullptr) {
            ggml_backend_buffer_free(state_buffer_);
        }
    }

    Graph(const Graph &) = delete;
    Graph & operator=(const Graph &) = delete;

    void start(const WhistleEncoderOutput & encoder) {
        const auto started = Clock::now();
        const auto frames = static_cast<int64_t>(encoder.frames);
        if (frames <= 0 || frames > kWhistleMaximumFrames) {
            throw std::invalid_argument("Whistle decoder needs between 1 and 375 encoder frames");
        }
        // Cross-attention keys arrive as [frame][head][48] and values as
        // [frame][head][64]. The step graph reads keys head-major, [head][frame][48],
        // and values with frames innermost, [head][64][frame], so no per-step copies
        // are needed. Frames past the end stay zero and are masked out.
        for (int layer = 0; layer < kWhistleLayers; ++layer) {
            const auto & keys = encoder.cross_k[static_cast<size_t>(layer)];
            const auto & values = encoder.cross_v[static_cast<size_t>(layer)];
            if (keys.size() != static_cast<size_t>(frames * kQueryWidth) ||
                values.size() != static_cast<size_t>(frames * kWhistleHeads * kWhistleVDim)) {
                throw std::invalid_argument("Whistle encoder cross-attention output has the wrong shape");
            }
            std::fill(cross_scratch_.begin(), cross_scratch_.end(), 0.0f);
            for (int64_t frame = 0; frame < frames; ++frame) {
                for (int64_t head = 0; head < kWhistleHeads; ++head) {
                    std::copy_n(keys.begin() + static_cast<std::ptrdiff_t>((frame * kWhistleHeads + head) * kWhistleQkDim),
                        kWhistleQkDim,
                        cross_scratch_.begin() +
                            static_cast<std::ptrdiff_t>((head * kWhistleMaximumFrames + frame) * kWhistleQkDim));
                }
            }
            ggml_backend_tensor_set(cross_keys_[static_cast<size_t>(layer)], cross_scratch_.data(), 0,
                ggml_nbytes(cross_keys_[static_cast<size_t>(layer)]));
            std::fill(cross_scratch_.begin(), cross_scratch_.end(), 0.0f);
            for (int64_t frame = 0; frame < frames; ++frame) {
                for (int64_t head = 0; head < kWhistleHeads; ++head) {
                    for (int64_t col = 0; col < kWhistleVDim; ++col) {
                        cross_scratch_[static_cast<size_t>((head * kWhistleVDim + col) * kWhistleMaximumFrames + frame)] =
                            values[static_cast<size_t>((frame * kWhistleHeads + head) * kWhistleVDim + col)];
                    }
                }
            }
            ggml_backend_tensor_set(cross_values_[static_cast<size_t>(layer)], cross_scratch_.data(), 0,
                ggml_nbytes(cross_values_[static_cast<size_t>(layer)]));
        }
        std::vector<ggml_fp16_t> cross_mask(static_cast<size_t>(kWhistleMaximumFrames),
            ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity()));
        std::fill_n(cross_mask.begin(), frames, ggml_fp32_to_fp16(0.0f));
        ggml_backend_tensor_set(cross_mask_, cross_mask.data(), 0, ggml_nbytes(cross_mask_));
        // Stale rows would be masked out anyway; clearing keeps every request starting
        // from the same backend state.
        kv_cache_.clear_on_backend();
        upload_ms_ = 0.0;
        compute_ms_ = 0.0;
        read_ms_ = 0.0;
        steps_ = 0;
        start_ms_ = debug::elapsed_ms(started);
    }

    const std::vector<float> & step(const std::vector<int32_t> & tokens) {
        if (tokens.empty()) {
            throw std::invalid_argument("Whistle decoder needs at least one input token");
        }
        const int64_t position = static_cast<int64_t>(tokens.size()) - 1;
        if (position != kv_cache_.current_end() || position >= kCacheSteps) {
            throw std::runtime_error("Whistle decoder positions must be consecutive and within the token limit");
        }
        const int32_t token = tokens.back();
        if (token < 0 || token >= kWhistleVocabulary) {
            throw std::invalid_argument("Whistle decoder token is outside the vocabulary");
        }
        const auto upload_started = Clock::now();
        const auto slot = static_cast<int32_t>(position);
        const std::array<int32_t, 3> history = {slot + 2, slot + 1, slot};
        ggml_backend_tensor_set(token_, &token, 0, sizeof(token));
        ggml_backend_tensor_set(position_, &slot, 0, sizeof(slot));
        ggml_backend_tensor_set(history_index_, history.data(), 0, sizeof(history));
        modules::write_decoder_cached_step_mask(self_mask_, mask_scratch_, kCacheSteps, position, position);
        for (int site = 0; site < kEngramSites; ++site) {
            engram_features(tokens, engram_tables_[static_cast<size_t>(site)], feature_scratch_);
            ggml_backend_tensor_set(engram_features_[static_cast<size_t>(site)], feature_scratch_.data(), 0,
                feature_scratch_.size() * sizeof(float));
        }
        upload_ms_ += debug::elapsed_ms(upload_started);

        const auto compute_started = Clock::now();
        if (core::compute_graph(*execution_context_, graph_, plan_, "whistle_asr.decoder.step") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to compute the Whistle decoder step graph");
        }
        ggml_backend_synchronize(backend_);
        compute_ms_ += debug::elapsed_ms(compute_started);
        kv_cache_.advance_after_direct_append(1);

        const auto read_started = Clock::now();
        core::read_tensor_f32_into(logits_, logits_values_);
        read_ms_ += debug::elapsed_ms(read_started);
        ++steps_;
        return logits_values_;
    }

    void log_timings() const {
        debug::timing_log_scalar("whistle_asr.decode_graph_nodes", static_cast<int64_t>(graph_nodes_));
        debug::timing_log_scalar("whistle_asr.decode_start_ms", start_ms_);
        debug::timing_log_scalar("whistle_asr.decode_step_count", steps_);
        debug::timing_log_scalar("whistle_asr.decode_step_upload_ms", upload_ms_);
        debug::timing_log_scalar("whistle_asr.decode_step_compute_ms", compute_ms_);
        debug::timing_log_scalar("whistle_asr.decode_step_read_ms", read_ms_);
    }

private:
    void allocate_state() {
        ggml_init_params params{};
        params.mem_size = ggml_tensor_overhead() * 128;
        params.no_alloc = true;
        state_ctx_.reset(ggml_init(params));
        if (!state_ctx_) {
            throw std::runtime_error("Failed to initialize the Whistle decoder state context");
        }
        ggml_context * ctx = state_ctx_.get();
        token_ = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        position_ = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        history_index_ = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, kProjectionTaps);
        self_mask_ = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, kCacheSteps);
        cross_mask_ = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, kWhistleMaximumFrames);
        for (auto & feature : engram_features_) {
            feature = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kWhistleDimension, kEngramTaps);
        }
        std::vector<TensorValue> keys;
        std::vector<TensorValue> values;
        for (int layer = 0; layer < kWhistleLayers; ++layer) {
            keys.push_back(core::wrap_tensor(
                ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kWhistleQkDim, kWhistleKvHeads, kCacheSteps, 1),
                TensorShape::from_dims({1, kCacheSteps, kWhistleKvHeads, kWhistleQkDim}), GGML_TYPE_F32));
            values.push_back(core::wrap_tensor(
                ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kWhistleVDim, kWhistleKvHeads, kCacheSteps, 1),
                TensorShape::from_dims({1, kCacheSteps, kWhistleKvHeads, kWhistleVDim}), GGML_TYPE_F32));
            history_[static_cast<size_t>(layer)] =
                ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kRawWidth, 1, kHistorySlots, 1);
            cross_keys_[static_cast<size_t>(layer)] =
                ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kWhistleQkDim, kWhistleMaximumFrames, kWhistleHeads);
            cross_values_[static_cast<size_t>(layer)] =
                ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kWhistleMaximumFrames, kWhistleVDim, kWhistleHeads);
        }
        state_buffer_ = ggml_backend_alloc_ctx_tensors(ctx, backend_);
        if (state_buffer_ == nullptr) {
            throw std::runtime_error("Failed to allocate the Whistle decoder state tensors");
        }
        // Masked attention entries get probability zero, which only stays zero when
        // the rows behind them are finite; the two leading history slots must be zero.
        ggml_backend_buffer_clear(state_buffer_, 0);
        // Key rows are 96 wide and value rows 128 wide, so the cache is only cleared
        // and advanced here; its host import and export paths assume one row width.
        runtime::TransformerKVCacheOptions options;
        options.lazy_import_scratch = true;
        kv_cache_ = runtime::TransformerKVCache(kCacheSteps, kKeyWidth, std::move(keys), std::move(values), options);
        cross_scratch_.resize(static_cast<size_t>(kWhistleMaximumFrames * kWhistleHeads * kWhistleVDim));
        mask_scratch_.resize(static_cast<size_t>(kCacheSteps));
    }

    void build_graph() {
        ggml_init_params params{};
        params.mem_size = ggml_tensor_overhead() * kGraphNodes + ggml_graph_overhead_custom(kGraphNodes, false);
        params.no_alloc = true;
        graph_ctx_.reset(ggml_init(params));
        if (!graph_ctx_) {
            throw std::runtime_error("Failed to initialize the Whistle decoder graph context");
        }
        core::ModuleBuildContext ctx{graph_ctx_.get(), "whistle_asr_decoder", execution_context_->backend_type()};
        WhistleGraphOps ops(ctx, weights_.constants);
        ggml_context * g = ctx.ggml;

        std::array<Engram, kEngramSites> engrams{};
        for (int site = 0; site < kEngramSites; ++site) {
            engrams[static_cast<size_t>(site)] = engram(ops, site);
        }
        ggml_tensor * x = ggml_scale(g, ggml_get_rows(g, weights_.embedding.tensor, token_), std::sqrt(512.0f));
        ggml_tensor * state = ggml_reshape_2d(g,
            ggml_repeat_4d(g, ggml_reshape_3d(g, x, kWhistleDimension, 1, 1), kWhistleDimension, kWhistleLanes, 1, 1),
            kWhistleLanes * kWhistleDimension, 1);
        for (int layer = 0; layer < kWhistleLayers; ++layer) {
            const auto & w = weights_.layers[static_cast<size_t>(layer)];
            state = ops.mhc(state, w.mhc, [&](ggml_tensor * mixed) {
                return block(ops, mixed, layer, w, engrams);
            });
        }
        ggml_tensor * h = ops.gemma_norm(ops.lane_mean(state), weights_.final_norm, kWhistleDimension);
        logits_ = ggml_mul_mat(g, weights_.embedding.tensor, h);
        ggml_set_output(logits_);

        graph_ = ggml_new_graph_custom(g, kGraphNodes, false);
        ggml_build_forward_expand(graph_, logits_);
        core::validate_backend_graph_supported(backend_, graph_, "Whistle decoder");
        galloc_.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_)));
        if (!galloc_ || !ggml_gallocr_reserve(galloc_.get(), graph_) || !ggml_gallocr_alloc_graph(galloc_.get(), graph_)) {
            throw std::runtime_error("Failed to allocate the Whistle decoder step graph");
        }
        core::prepare_host_graph_plan(*execution_context_, graph_, plan_);
        graph_nodes_ = ggml_graph_n_nodes(graph_);
        logits_values_.reserve(static_cast<size_t>(kWhistleVocabulary));
    }

    struct Engram {
        ggml_tensor * key = nullptr;
        ggml_tensor * value = nullptr;
    };

    // key = key_proj(feature of tap 0); value = sum over taps of taps[t] * value_proj(feature[t]).
    Engram engram(WhistleGraphOps & ops, int site) {
        ggml_context * g = ops.context().ggml;
        const auto & w = weights_.engrams[static_cast<size_t>(site)];
        ggml_tensor * feature = engram_features_[static_cast<size_t>(site)];
        Engram result;
        result.key = ops.linear(column(g, feature, 0), w.key_proj, kWhistleDimension, kWhistleDimension);
        ggml_tensor * weighted = ggml_mul(g,
            ops.linear(feature, w.value_proj, kWhistleDimension, kWhistleDimension), w.taps.tensor);
        ggml_tensor * value = column(g, weighted, 0);
        for (int64_t tap = 1; tap < kEngramTaps; ++tap) {
            value = ggml_add(g, value, column(g, weighted, tap));
        }
        result.value = value;
        return result;
    }

    // h += sigmoid(<rms(h), rms(key)> / sqrt(512)) * value
    static ggml_tensor * inject(WhistleGraphOps & ops, ggml_tensor * h, const Engram & engram) {
        ggml_context * g = ops.context().ggml;
        ggml_tensor * dot = ggml_mul_mat(g,
            ops.rms_norm(h, kWhistleDimension), ops.rms_norm(engram.key, kWhistleDimension));
        ggml_tensor * gate = ops.sigmoid(ggml_scale(g, dot, 1.0f / std::sqrt(512.0f)));
        return ggml_add(g, h, ggml_mul(g, engram.value, gate));
    }

    // Causal self-attention over the cached steps. The query, key and value of this
    // step mix the raw projections of this step and the two before it.
    ggml_tensor * self_attention(WhistleGraphOps & ops, ggml_tensor * input, int layer, const DecoderLayerWeights & w) {
        core::ModuleBuildContext & ctx = ops.context();
        ggml_context * g = ctx.ggml;
        const modules::FastKVSetRowsModule set_rows;

        ggml_tensor * current = ops.linear(input, w.qkv_proj, kWhistleDimension, kRawWidth);
        const auto history = set_rows.build(ctx,
            core::wrap_tensor(history_[static_cast<size_t>(layer)],
                TensorShape::from_dims({1, kHistorySlots, 1, kRawWidth}), GGML_TYPE_F32),
            core::wrap_tensor(current, TensorShape::from_dims({1, 1, 1, kRawWidth}), GGML_TYPE_F32),
            core::wrap_tensor(ggml_view_1d(g, history_index_, 1, 0), TensorShape::from_dims({1}), GGML_TYPE_I32));
        // [608, 3]: this step, the previous step, the step before that.
        ggml_tensor * window = ggml_get_rows(g,
            ggml_reshape_2d(g, history.tensor, kRawWidth, kHistorySlots), history_index_);
        ggml_tensor * weighted = ggml_mul(g, window, w.taps.tensor);
        ggml_tensor * mixed = ggml_add(g, ggml_add(g, column(g, weighted, 0), column(g, weighted, 1)), column(g, weighted, 2));

        ggml_tensor * query = ggml_view_1d(g, mixed, kQueryWidth, 0);
        ggml_tensor * key = ggml_view_1d(g, mixed, kKeyWidth, kQueryWidth * sizeof(float));
        ggml_tensor * value = ggml_view_1d(g, mixed, kValueWidth, (kQueryWidth + kKeyWidth) * sizeof(float));
        const auto q = ops.rope(ops.head_norm(query, kWhistleHeads, w.q_norm), kWhistleHeads, position_);
        const auto k = ops.rope(ops.head_norm(key, kWhistleKvHeads, w.k_norm), kWhistleKvHeads, position_);
        const auto slot = core::wrap_tensor(position_, TensorShape::from_dims({1}), GGML_TYPE_I32);
        const auto keys = set_rows.build(ctx, kv_cache_.key_tensor(static_cast<size_t>(layer)), k, slot);
        const auto values = set_rows.build(ctx, kv_cache_.value_tensor(static_cast<size_t>(layer)),
            core::wrap_tensor(ggml_reshape_4d(g, value, kWhistleVDim, kWhistleKvHeads, 1, 1),
                TensorShape::from_dims({1, 1, kWhistleKvHeads, kWhistleVDim}), GGML_TYPE_F32),
            slot);

        // Eight query heads over two key/value heads: ggml_mul_mat broadcasts key
        // head h / 4 to query head h, matching the grouping of the checkpoint.
        ggml_tensor * key_heads = ggml_permute(g,
            ggml_reshape_3d(g, keys.tensor, kWhistleQkDim, kWhistleKvHeads, kCacheSteps), 0, 2, 1, 3);
        ggml_tensor * scores = ggml_mul_mat(g, key_heads, ggml_reshape_3d(g, q.tensor, kWhistleQkDim, 1, kWhistleHeads));
        ggml_tensor * probabilities = ggml_soft_max_ext(
            g, scores, self_mask_, 1.0f / std::sqrt(static_cast<float>(kWhistleQkDim)), 0.0f);
        ggml_tensor * value_heads = ggml_cont(g, ggml_permute(g,
            ggml_reshape_3d(g, values.tensor, kWhistleVDim, kWhistleKvHeads, kCacheSteps), 1, 2, 0, 3));
        ggml_tensor * context = ggml_reshape_2d(g, ggml_mul_mat(g, value_heads, probabilities), kWhistleDimension, 1);
        ggml_tensor * gate = ops.sigmoid(ops.linear(input, w.self_gate_proj, kWhistleDimension, kWhistleDimension));
        return ops.linear(ggml_mul(g, context, gate), w.self_out_proj, kWhistleDimension, kWhistleDimension);
    }

    // Attention over the encoder frames of this request, eight heads, no grouping.
    ggml_tensor * cross_attention(WhistleGraphOps & ops, ggml_tensor * input, int layer, const DecoderLayerWeights & w) {
        ggml_context * g = ops.context().ggml;
        ggml_tensor * query = ops.head_norm(
            ops.linear(input, w.cross_q_proj, kWhistleDimension, kQueryWidth), kWhistleHeads, w.cross_q_norm);
        ggml_tensor * scores = ggml_mul_mat(g, cross_keys_[static_cast<size_t>(layer)],
            ggml_reshape_3d(g, query, kWhistleQkDim, 1, kWhistleHeads));
        ggml_tensor * probabilities = ggml_soft_max_ext(
            g, scores, cross_mask_, 1.0f / std::sqrt(static_cast<float>(kWhistleQkDim)), 0.0f);
        ggml_tensor * context = ggml_reshape_2d(g,
            ggml_mul_mat(g, cross_values_[static_cast<size_t>(layer)], probabilities), kWhistleDimension, 1);
        ggml_tensor * gate = ops.sigmoid(ops.linear(input, w.cross_gate_proj, kWhistleDimension, kWhistleDimension));
        return ops.linear(ggml_mul(g, context, gate), w.cross_out_proj, kWhistleDimension, kWhistleDimension);
    }

    ggml_tensor * block(WhistleGraphOps & ops, ggml_tensor * input, int layer, const DecoderLayerWeights & w,
                        const std::array<Engram, kEngramSites> & engrams) {
        ggml_context * g = ops.context().ggml;
        ggml_tensor * h = input;
        if (layer == 3 || layer == 7) {
            h = inject(ops, h, engrams[layer == 3 ? 0 : 1]);
        }
        ggml_tensor * self = self_attention(ops, ops.gemma_norm(h, w.attn_norm, kWhistleDimension), layer, w);
        self = ops.gemma_norm(self, w.post_attn_norm, kWhistleDimension);
        h = ggml_add(g, h, ggml_scale(g, self, w.attn_gate));
        ggml_tensor * cross = cross_attention(ops, ops.gemma_norm(h, w.cross_norm, kWhistleDimension), layer, w);
        cross = ops.gemma_norm(cross, w.post_cross_norm, kWhistleDimension);
        h = ggml_add(g, h, ggml_scale(g, cross, w.cross_gate));
        // The decoder adds its single Hadamard MLP at full weight.
        return ggml_add(g, h, ops.hadamard(ops.gemma_norm(h, w.pre_hada_norm, kWhistleDimension), w.hadamard));
    }

    std::shared_ptr<const WhistleAssets> assets_;
    core::ExecutionContext * execution_context_ = nullptr;
    ggml_backend_t backend_ = nullptr;
    core::BackendWeightStore weight_store_;
    DecoderWeights weights_;
    // Raw little-endian f32 rows of the two [4, 18432, 128] engram tables.
    std::array<std::vector<std::byte>, kEngramSites> engram_tables_;

    std::unique_ptr<ggml_context, GgmlContextDeleter> state_ctx_;
    ggml_backend_buffer_t state_buffer_ = nullptr;
    ggml_tensor * token_ = nullptr;
    ggml_tensor * position_ = nullptr;
    ggml_tensor * history_index_ = nullptr;
    ggml_tensor * self_mask_ = nullptr;
    ggml_tensor * cross_mask_ = nullptr;
    std::array<ggml_tensor *, kEngramSites> engram_features_{};
    std::array<ggml_tensor *, kWhistleLayers> history_{};
    std::array<ggml_tensor *, kWhistleLayers> cross_keys_{};
    std::array<ggml_tensor *, kWhistleLayers> cross_values_{};
    runtime::TransformerKVCache kv_cache_;

    std::unique_ptr<ggml_context, GgmlContextDeleter> graph_ctx_;
    std::unique_ptr<ggml_gallocr, GallocrDeleter> galloc_;
    core::HostGraphPlan plan_;
    ggml_cgraph * graph_ = nullptr;
    ggml_tensor * logits_ = nullptr;

    std::vector<float> logits_values_;
    std::vector<float> cross_scratch_;
    std::vector<float> feature_scratch_;
    std::vector<ggml_fp16_t> mask_scratch_;
    double start_ms_ = 0.0;
    double upload_ms_ = 0.0;
    double compute_ms_ = 0.0;
    double read_ms_ = 0.0;
    int64_t steps_ = 0;
    int graph_nodes_ = 0;
};

WhistleDecoderRuntime::WhistleDecoderRuntime(
    std::shared_ptr<const WhistleAssets> assets, core::ExecutionContext & execution_context)
    : graph_(std::make_unique<Graph>(std::move(assets), execution_context)) {}

WhistleDecoderRuntime::~WhistleDecoderRuntime() = default;

void WhistleDecoderRuntime::start(const WhistleEncoderOutput & encoder) {
    graph_->start(encoder);
}

const std::vector<float> & WhistleDecoderRuntime::step(const std::vector<int32_t> & tokens) {
    return graph_->step(tokens);
}

void WhistleDecoderRuntime::log_timings() const {
    graph_->log_timings();
}

}  // namespace engine::community_models::whistle_asr
