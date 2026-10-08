#include "engine/community_models/lfm2_audio/backbone.h"

#include "lfm2_blocks.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
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
using lfm2_blocks::GgmlBufferDeleter;
using lfm2_blocks::GgmlContextDeleter;
using lfm2_blocks::GgmlGallocrDeleter;
using lfm2_blocks::LayerWeights;
using lfm2_blocks::attention_layer_config;
using lfm2_blocks::backend_gathers;
using lfm2_blocks::contiguous;
using lfm2_blocks::conv_kernel;
using lfm2_blocks::feed_forward;
using lfm2_blocks::rms_norm;
using lfm2_blocks::short_conv_input;
using lfm2_blocks::short_conv_output;

constexpr size_t kWeightContextBytes = 16ull * 1024ull * 1024ull;
constexpr size_t kPrefillArenaBytes = 64ull * 1024ull * 1024ull;
constexpr size_t kDecodeArenaBytes = 32ull * 1024ull * 1024ull;
constexpr size_t kGraphNodes = 32768;
constexpr int64_t kMaxRetainedPrefillSteps = 1024;
// Lfm2DecodeCache::Speech rounds the decode cache up to this.
constexpr int64_t kCacheStepGranule = 256;

struct BackboneWeights {
    std::unique_ptr<WeightStores> stores;
    TensorValue token_embedding;  // [vocab, hidden]
    TensorValue token_lookup;     // token_embedding, or an F16 copy (see load_weights)
    TensorValue text_head;        // token_embedding, or a repacked copy (see load_weights)
    modules::NormWeights final_norm;
    std::vector<LayerWeights> layers;
    // Audio frames fed back: [codebooks * audio_vocab_size, hidden], empty
    // for text-only use.
    std::optional<TensorValue> audio_embedding;
    int64_t codebooks = 0;
    int64_t audio_vocab_size = 0;
};

struct AudioEmbeddingSource {
    std::shared_ptr<const assets::TensorSource> source;
    int64_t codebooks = 0;
    int64_t vocab_size = 0;
};

BackboneWeights load_weights(
    const assets::TensorSource & source,
    const Lfm2BackboneConfig & config,
    core::ExecutionContext & execution,
    const AudioEmbeddingSource & audio,
    bool cpu_repack) {
    BackboneWeights out;
    out.stores = std::make_unique<WeightStores>(execution, "lfm2_audio.backbone.weights", kWeightContextBytes, cpu_repack);
    auto & store = out.stores->plain();
    const auto native = assets::TensorStorageType::Native;
    const int64_t d = config.hidden_size;

    out.token_embedding = store.load_tensor(source, "token_embd.weight", native, {config.vocab_size, d});
    // ggml's CUDA get_rows has no K-quant kernels, and Liquid's Q4_0 packages
    // store token_embd as Q6_K, so there the lookup reads an F16 copy. (llama.cpp
    // keeps its input embedding on the CPU instead.)
    out.token_lookup = backend_gathers(execution.backend(), out.token_embedding.tensor->type)
        ? out.token_embedding
        : store.load_tensor(source, "token_embd.weight", assets::TensorStorageType::F16, {config.vocab_size, d});
    // The text head is tied to the token embedding, which the lookup gathers
    // from, so where a CPU extra buffer takes its type (Q6_K or Q8_0 on Arm,
    // also F16 with AMX) the head gets a copy of its own there, as llama.cpp's
    // output head does.
    out.text_head = out.stores->matmul_buffer_type(out.token_embedding.tensor->type, config.vocab_size, d) != nullptr
        ? out.stores->load_matmul(source, "token_embd.weight", {config.vocab_size, d})
        : out.token_embedding;
    out.final_norm = {store.load_f32_tensor(source, "token_embd_norm.weight", {d}), std::nullopt};

    out.layers = lfm2_blocks::load_layers(*out.stores, source, config);

    if (audio.source != nullptr) {
        // F32 in every published mmproj (the vocoder's copy is quantized with
        // the package), and ggml gathers F32 rows on every backend.
        out.audio_embedding = store.load_f32_tensor(*audio.source, "a.position_embd.weight", {audio.codebooks * audio.vocab_size, d});
        out.codebooks = audio.codebooks;
        out.audio_vocab_size = audio.vocab_size;
    }

    out.stores->upload();
    return out;
}

// The final norm's output is Lfm2Model's last_hidden_state; the text head is
// tied to the token embedding.
TensorValue hidden_of_last_step(
    core::ModuleBuildContext & ctx, const TensorValue & x, const BackboneWeights & weights, const Lfm2BackboneConfig & config) {
    const int64_t steps = x.shape.dims[1];
    auto last = steps == 1 ? x : contiguous(ctx, modules::SliceModule({1, steps - 1, 1}).build(ctx, x));
    return rms_norm(ctx, last, weights.final_norm, config);
}

TensorValue text_logits(core::ModuleBuildContext & ctx, const TensorValue & hidden, const BackboneWeights & weights, const Lfm2BackboneConfig & config) {
    return modules::LinearModule({config.hidden_size, config.vocab_size, false})
        .build(ctx, hidden, {weights.text_head, std::nullopt});
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

        lfm2_blocks::SequenceTaps taps;
        x = lfm2_blocks::build_sequence(ctx, x, positions, weights.layers, config, std::nullopt, &taps);
        for (auto * t : taps.keys) keys_.push_back(pin_output(t));
        for (auto * t : taps.values) values_.push_back(pin_output(t));
        for (auto * t : taps.conv_tails) conv_tails_.push_back(pin_output(t));

        logits_ = text_logits(ctx, hidden_of_last_step(ctx, x, weights, config), weights, config).tensor;
        ggml_set_output(logits_);

        graph_ = ggml_new_graph_custom(g, kGraphNodes, false);
        ggml_build_forward_expand(graph_, logits_);
        for (auto * t : keys_) ggml_build_forward_expand(graph_, t);
        for (auto * t : values_) ggml_build_forward_expand(graph_, t);
        for (auto * t : conv_tails_) ggml_build_forward_expand(graph_, t);
        core::validate_backend_graph_with_cpu_extra_buffers(execution.backend(), graph_, "LFM2-Audio prefill graph");

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

// A step's input: a text token, or the rows of an audio frame's codes in the
// stacked audio embedding.
struct StepInput {
    int32_t token = 0;
    std::vector<int32_t> audio_rows;
};

class DecodeGraph {
public:
    DecodeGraph(const BackboneWeights & weights, const Lfm2BackboneConfig & config, core::ExecutionContext & execution, int64_t cache_steps)
        : config_(config), execution_(execution), cache_steps_(cache_steps), codebooks_(weights.codebooks) {
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
        if (weights.audio_embedding.has_value()) {
            // Both inputs are always built; the step picks one by weighting
            // the other with zero. A frame goes in as the sum of its codes'
            // embeddings (LFM2AudioModel.generate_sequential).
            audio_rows_ = ggml_new_tensor_1d(g, GGML_TYPE_I32, codebooks_);
            text_weight_ = ggml_new_tensor_1d(g, GGML_TYPE_F32, 1);
            audio_weight_ = ggml_new_tensor_1d(g, GGML_TYPE_F32, 1);
            auto * rows = ggml_get_rows(g, weights.audio_embedding->tensor, audio_rows_);
            auto * frame = ggml_sum_rows(g, ggml_cont(g, ggml_transpose(g, rows)));
            frame = ggml_reshape_2d(g, frame, d, 1);
            auto * mixed = ggml_add(g, ggml_mul(g, x.tensor, text_weight_), ggml_mul(g, frame, audio_weight_));
            x = core::wrap_tensor(mixed, x.shape, GGML_TYPE_F32);
        }

        x = core::reshape_tensor(ctx, x, TensorShape::from_dims({1, 1, d}));

        const auto positions = core::wrap_tensor(positions_, TensorShape::from_dims({1}), GGML_TYPE_I32);
        const auto slot = core::wrap_tensor(cache_slot_, TensorShape::from_dims({1}), GGML_TYPE_I32);
        const auto mask = core::wrap_tensor(mask_, TensorShape::from_dims({1, 1, 1, cache_steps}), GGML_TYPE_F16);

        hidden_graph_ = ggml_new_graph_custom(g, kGraphNodes, false);

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
                        .build_with_static_cache_tail(ctx, hidden_graph_, x, positions, w.decoder, keys.back(), values.back(), slot, mask)
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
            ggml_build_forward_expand(hidden_graph_, ggml_cpy(g, next_tail.tensor, tail.tensor));
        }

        const auto hidden = hidden_of_last_step(ctx, x, weights, config);
        hidden_ = hidden.tensor;
        logits_ = text_logits(ctx, hidden, weights, config).tensor;
        ggml_set_output(hidden_);
        ggml_set_output(logits_);

        // Two graphs over the same nodes and cache writes: one stops at the
        // hidden state, the other goes on through the text head.
        ggml_build_forward_expand(hidden_graph_, hidden_);
        logits_graph_ = ggml_new_graph_custom(g, kGraphNodes, false);
        ggml_graph_cpy(hidden_graph_, logits_graph_);
        ggml_build_forward_expand(logits_graph_, logits_);
        core::validate_backend_graph_with_cpu_extra_buffers(execution.backend(), logits_graph_, "LFM2-Audio decode graph");

        buffer_.reset(ggml_backend_alloc_ctx_tensors(g, execution.backend()));
        if (buffer_ == nullptr) {
            throw runtime::CapacityError(
                "LFM2-Audio decode graph does not fit in device memory at " + std::to_string(cache_steps) + " cache steps");
        }

        cache_ = runtime::TransformerKVCache(cache_steps, step_elems, keys, values);
        mask_scratch_.assign(static_cast<size_t>(cache_steps), ggml_fp32_to_fp16(-INFINITY));
    }

    ~DecodeGraph() {
        core::release_backend_graph_resources(execution_.backend(), logits_graph_, true);
        core::release_backend_graph_resources(execution_.backend(), hidden_graph_, true);
    }

    [[nodiscard]] int64_t cache_steps() const noexcept { return cache_steps_; }

    // Every step attends over the whole cache, so a cache sized for an
    // unusually long request is replaced rather than reused.
    [[nodiscard]] bool fits(int64_t required_steps) const {
        return cache_steps_ >= required_steps && cache_steps_ <= 2 * required_steps;
    }

    // Starts from zeros, so nothing an earlier request left in the masked
    // slots can reach this one.
    void import_state(const PrefillState & state) {
        ggml_backend_buffer_clear(buffer_.get(), 0);
        cache_.import_state(state.kv);
        if (state.conv_tails.size() != conv_tails_.size()) {
            throw std::runtime_error("LFM2-Audio conv state does not match the decode graph");
        }

        for (size_t i = 0; i < conv_tails_.size(); ++i) {
            ggml_backend_tensor_set(conv_tails_[i], state.conv_tails[i].data(), 0, state.conv_tails[i].size() * sizeof(float));
        }
    }

    std::vector<float> run_step(const StepInput & input, Lfm2StepOutput output) {
        if (cache_.valid_steps() >= cache_steps_) {
            throw std::runtime_error("LFM2-Audio decode cache exhausted");
        }

        const bool audio = !input.audio_rows.empty();
        if (audio && audio_rows_ == nullptr) {
            throw std::runtime_error("LFM2-Audio backbone was loaded without the audio embedding");
        }

        const auto position = static_cast<int32_t>(cache_.current_end());
        const auto slot = static_cast<int32_t>(cache_.valid_steps());
        ggml_backend_tensor_set(token_, &input.token, 0, sizeof(int32_t));
        ggml_backend_tensor_set(positions_, &position, 0, sizeof(int32_t));
        ggml_backend_tensor_set(cache_slot_, &slot, 0, sizeof(int32_t));
        if (audio_rows_ != nullptr) {
            // The unused input still needs valid rows: 0 * NaN would be NaN.
            const std::vector<int32_t> rows = audio ? input.audio_rows : std::vector<int32_t>(static_cast<size_t>(codebooks_), 0);
            const float text_weight = audio ? 0.0f : 1.0f;
            const float audio_weight = audio ? 1.0f : 0.0f;
            ggml_backend_tensor_set(audio_rows_, rows.data(), 0, rows.size() * sizeof(int32_t));
            ggml_backend_tensor_set(text_weight_, &text_weight, 0, sizeof(float));
            ggml_backend_tensor_set(audio_weight_, &audio_weight, 0, sizeof(float));
        }

        modules::write_decoder_cached_step_mask(mask_, mask_scratch_, cache_steps_, cache_.valid_steps(), cache_.valid_steps());

        auto * graph = output == Lfm2StepOutput::Logits ? logits_graph_ : hidden_graph_;
        core::set_backend_threads(execution_.backend(), std::max(1, execution_.config().threads));
        const ggml_status status = core::compute_backend_graph(execution_.backend(), graph);
        ggml_backend_synchronize(execution_.backend());
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("LFM2-Audio decode graph compute failed");
        }

        auto * result = output == Lfm2StepOutput::Logits ? logits_ : hidden_;
        std::vector<float> values(static_cast<size_t>(ggml_nelements(result)));
        ggml_backend_tensor_get(result, values.data(), 0, values.size() * sizeof(float));
        cache_.advance_after_direct_append(1);
        return values;
    }

private:
    const Lfm2BackboneConfig & config_;
    core::ExecutionContext & execution_;
    int64_t cache_steps_ = 0;
    int64_t codebooks_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * token_ = nullptr;
    ggml_tensor * audio_rows_ = nullptr;
    ggml_tensor * text_weight_ = nullptr;
    ggml_tensor * audio_weight_ = nullptr;
    ggml_tensor * positions_ = nullptr;
    ggml_tensor * cache_slot_ = nullptr;
    ggml_tensor * mask_ = nullptr;
    ggml_tensor * hidden_ = nullptr;
    ggml_tensor * logits_ = nullptr;
    std::vector<ggml_tensor *> conv_tails_;
    std::vector<ggml_fp16_t> mask_scratch_;
    runtime::TransformerKVCache cache_;
    ggml_cgraph * hidden_graph_ = nullptr;
    ggml_cgraph * logits_graph_ = nullptr;
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
    Impl(std::shared_ptr<const assets::TensorSource> source_in,
         const Lfm2BackboneConfig & config_in,
         core::ExecutionContext & execution_in,
         const AudioEmbeddingSource & audio,
         bool cpu_repack)
        : source(std::move(source_in)),
          config(config_in),
          execution(execution_in),
          weights(load_weights(*source, config, execution_in, audio, cpu_repack)) {}

    DecodeGraph & require_started() {
        if (decode == nullptr || !started) {
            throw std::runtime_error("LFM2-Audio backbone step before start()");
        }

        return *decode;
    }

    std::shared_ptr<const assets::TensorSource> source;
    Lfm2BackboneConfig config;
    core::ExecutionContext & execution;
    BackboneWeights weights;
    std::unique_ptr<PrefillGraph> prefill;
    std::unique_ptr<DecodeGraph> decode;
    Lfm2DecodeCache decode_policy = Lfm2DecodeCache::Transcript;  // the policy `decode` was sized by
    bool started = false;
};

Lfm2BackboneRuntime::Lfm2BackboneRuntime(
    std::shared_ptr<const assets::TensorSource> source,
    const Lfm2BackboneConfig & config,
    core::ExecutionContext & execution,
    std::shared_ptr<const assets::TensorSource> audio_embedding,
    int64_t codebooks,
    int64_t audio_vocab_size,
    bool cpu_repack) {
    if (audio_embedding != nullptr && (codebooks <= 0 || audio_vocab_size <= 0)) {
        throw std::runtime_error("LFM2-Audio audio embedding needs its codebook count and size");
    }

    impl_ = std::make_unique<Impl>(
        std::move(source), config, execution, AudioEmbeddingSource{std::move(audio_embedding), codebooks, audio_vocab_size}, cpu_repack);
}

Lfm2BackboneRuntime::~Lfm2BackboneRuntime() = default;

std::vector<float> Lfm2BackboneRuntime::start(
    const Lfm2Prompt & prompt, const Lfm2AudioEmbeddings & audio, int64_t max_steps, Lfm2DecodeCache cache) {
    const auto & config = impl_->config;
    const auto steps = static_cast<int64_t>(prompt.input_ids.size());
    const auto audio_tokens = static_cast<int64_t>(prompt.audio_positions.size());
    impl_->started = false;
    if (steps == 0 || max_steps < 0) {
        throw std::runtime_error("LFM2-Audio generation needs a prompt and a nonnegative step budget");
    }

    if (audio_tokens != audio.tokens || audio.values.size() != static_cast<size_t>(audio.tokens * config.hidden_size)) {
        throw std::runtime_error("LFM2-Audio audio embeddings do not match the prompt's audio positions");
    }

    // Checked here rather than left to the logits: ggml's CPU RMSNorm
    // (ggml_compute_forward_rms_norm_f32) asserts on NaN input in debug builds.
    if (!std::all_of(audio.values.begin(), audio.values.end(), [](float value) { return std::isfinite(value); })) {
        throw std::runtime_error("LFM2-Audio encoder produced non-finite audio embeddings");
    }

    if (max_steps > config.context_length - steps) {
        throw runtime::CapacityError(
            "LFM2-Audio request needs " + std::to_string(steps) + " prompt steps plus " + std::to_string(max_steps) +
            " more, more than the " + std::to_string(config.context_length) + "-token context");
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

    const int64_t required = steps + max_steps;
    const int64_t needed = std::max<int64_t>(required, steps + 1);
    const bool speech = cache == Lfm2DecodeCache::Speech;
    const int64_t cache_steps = speech ? (needed + kCacheStepGranule - 1) / kCacheStepGranule * kCacheStepGranule : needed;
    const bool keep = impl_->decode != nullptr && impl_->decode_policy == cache &&
                      (speech ? impl_->decode->cache_steps() == cache_steps : impl_->decode->fits(required));
    if (!keep) {
        impl_->decode.reset();
        impl_->decode = std::make_unique<DecodeGraph>(impl_->weights, config, impl_->execution, cache_steps);
        impl_->decode_policy = cache;
    }

    impl_->decode->import_state(state);
    impl_->started = true;
    return std::move(state.logits);
}

std::vector<float> Lfm2BackboneRuntime::step_text(int32_t token, Lfm2StepOutput output) {
    if (token < 0 || token >= impl_->config.vocab_size) {
        throw std::runtime_error("LFM2-Audio token id " + std::to_string(token) + " is outside the vocabulary");
    }

    return impl_->require_started().run_step({token, {}}, output);
}

std::vector<float> Lfm2BackboneRuntime::step_audio(const std::vector<int32_t> & codes, Lfm2StepOutput output) {
    const auto & weights = impl_->weights;
    if (!weights.audio_embedding.has_value()) {
        throw std::runtime_error("LFM2-Audio backbone was loaded without the audio embedding");
    }

    if (static_cast<int64_t>(codes.size()) != weights.codebooks) {
        throw std::runtime_error("LFM2-Audio audio frame needs one code per codebook");
    }

    StepInput input;
    for (size_t codebook = 0; codebook < codes.size(); ++codebook) {
        if (codes[codebook] < 0 || codes[codebook] >= weights.audio_vocab_size) {
            throw std::runtime_error("LFM2-Audio audio code " + std::to_string(codes[codebook]) + " is outside the codebook");
        }

        input.audio_rows.push_back(static_cast<int32_t>(static_cast<int64_t>(codebook) * weights.audio_vocab_size + codes[codebook]));
    }

    return impl_->require_started().run_step(input, output);
}

int64_t Lfm2BackboneRuntime::decode_cache_steps() const noexcept {
    return impl_->decode == nullptr ? 0 : impl_->decode->cache_steps();
}

std::vector<std::pair<std::string, size_t>> Lfm2BackboneRuntime::extra_weight_buffers() const {
    return impl_->weights.stores->extra_buffers();
}

Lfm2GenerationResult Lfm2BackboneRuntime::generate(
    const Lfm2Prompt & prompt,
    const Lfm2AudioEmbeddings & audio,
    const Lfm2GenerationOptions & options) {
    const auto steps = static_cast<int64_t>(prompt.input_ids.size());
    if (steps == 0 || options.max_new_tokens <= 0) {
        throw std::runtime_error("LFM2-Audio generation needs a prompt and a positive token budget");
    }

    if (options.max_new_tokens > impl_->config.context_length - steps) {
        throw runtime::CapacityError(
            "LFM2-Audio request needs " + std::to_string(steps) + " prompt steps plus max_tokens, more than the " +
            std::to_string(impl_->config.context_length) + "-token context");
    }

    // The last generated token is never fed back, hence the - 1.
    Lfm2GenerationResult out;
    out.prefill_logits = start(prompt, audio, options.max_new_tokens - 1, Lfm2DecodeCache::Transcript);
    std::vector<float> logits = out.prefill_logits;

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

        logits = step_text(token, Lfm2StepOutput::Logits);
    }

    debug::timing_log_scalar("lfm2_audio.decode.ms", engine::debug::elapsed_ms(decode_start));
    return out;
}

}  // namespace engine::community_models::lfm2_audio
