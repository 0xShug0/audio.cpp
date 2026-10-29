#include "engine/community_models/lfm3_audio/backbone.h"

#include "../lfm2_audio/weight_stores.h"

#include "engine/community_models/lfm3_audio/blocks.h"
#include "engine/community_models/lfm3_audio/chat.h"
#include "engine/community_models/lfm3_audio/debug.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/runtime/errors.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpp.h>
#include <ggml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::community_models::lfm3_audio {
namespace {

using blocks::LayerKind;
using core::TensorShape;
using core::TensorValue;

constexpr size_t kWeightContextBytes = 16ull * 1024ull * 1024ull;
constexpr size_t kGraphNodes = 16384;
const char * const kPrefix = "lfm.model.";
const char * const kEmbedTokens = "lfm.model.embed_tokens.weight";
const char * const kCodebookEmbedding = "codebook_embedding.weight";

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error("LFM3-Audio backbone: " + message);
}

ggml_tensor * input(ggml_context * g, ggml_type type, int64_t ne0, int64_t ne1 = 1) {
    auto * t = ggml_new_tensor_2d(g, type, ne0, ne1);
    ggml_set_input(t);
    return t;
}

ggml_type stored_type(const assets::TensorSource & source, const std::string & name) {
    return assets::ggml_type_for_tensor_dtype(source.require_metadata(name).dtype);
}

bool has_kind(const blocks::StackConfig & config, LayerKind kind) {
    return std::find(config.layers.begin(), config.layers.end(), kind) != config.layers.end();
}

struct Weights {
    std::unique_ptr<WeightStores> stores;
    blocks::StackWeights stack;
    TensorValue token_lookup;  // embed_tokens, or an F16 copy where the backend cannot gather its type
    TensorValue text_head;     // embed_tokens (tied), or a CPU-repacked copy of it
    TensorValue frame_lookup;  // codebook_embedding [codebooks * audio_vocab, hidden]
};

TensorValue lookup_of(WeightStores & stores, ggml_backend_t backend, const assets::TensorSource & source, const std::string & name,
                      const TensorValue & table) {
    if (lfm3_backend_gathers(backend, table.tensor->type)) {
        return table;
    }
    return stores.plain().load_tensor(source, name, assets::TensorStorageType::F16, {table.shape.dims[0], table.shape.dims[1]});
}

Weights load_weights(const assets::TensorSource & source, const Lfm3BackboneConfig & config, int64_t codebooks, int64_t audio_vocab,
                     core::ExecutionContext & execution, const Lfm3WeightOptions & options) {
    Weights out;
    auto * backend = execution.backend();
    const int64_t d = config.stack.hidden;
    out.stores = std::make_unique<WeightStores>(execution, "lfm3_audio.backbone.weights", kWeightContextBytes, options.cpu_repack);
    auto & plain = out.stores->plain();
    out.stack = blocks::load_stack(*out.stores, backend, source, kPrefix, config.stack, options);

    const auto storage = lfm3_storage_type(backend, stored_type(source, kEmbedTokens), options);
    const auto table = plain.load_tensor(source, kEmbedTokens, storage, {config.vocab_size, d});
    out.token_lookup = lookup_of(*out.stores, backend, source, kEmbedTokens, table);
    // Where a CPU extra buffer takes the tied head's type, the head gets a repacked copy of its own, as in lfm2_audio.
    const bool repack = storage == assets::TensorStorageType::Native &&
                        out.stores->matmul_buffer_type(table.tensor->type, config.vocab_size, d) != nullptr;
    out.text_head = repack ? out.stores->load_matmul(source, kEmbedTokens, {config.vocab_size, d}) : table;

    const auto frames = plain.load_tensor(source, kCodebookEmbedding, lfm3_storage_type(backend, stored_type(source, kCodebookEmbedding), options),
                                          {codebooks * audio_vocab, d});
    out.frame_lookup = lookup_of(*out.stores, backend, source, kCodebookEmbedding, frames);
    out.stores->upload();
    return out;
}

// [hidden, frames]: each frame the sum of its codes' rows (codebook k reads row k * audio_vocab + code_k). Prompts,
// embed_frames and decode steps all sum here, so a replayed frame sums exactly as a fed-back one.
ggml_tensor * frame_embeddings(ggml_context * g, ggml_tensor * table, ggml_tensor * rows, int64_t codebooks) {
    const int64_t d = table->ne[0];
    const int64_t frames = ggml_nelements(rows) / codebooks;
    auto * gathered = ggml_reshape_3d(g, ggml_get_rows(g, table, rows), d, codebooks, frames);
    auto * sums = ggml_sum_rows(g, ggml_cont(g, ggml_transpose(g, gathered)));  // [1, hidden, frames]
    return ggml_reshape_2d(g, sums, d, frames);
}

TensorValue text_logits(core::ModuleBuildContext & ctx, const TensorValue & hidden, const Weights & weights,
                        const Lfm3BackboneConfig & config) {
    return modules::LinearModule({config.stack.hidden, config.vocab_size, false, blocks::kMatmulPrecision})
        .build(ctx, hidden, {weights.text_head, std::nullopt});
}

// One decode step over the whole cache (kv_rows = capacity): a text token or a frame (both inputs always built,
// the step weights the unused one with zero, as lfm2_audio's decode graph does), up to the hidden state or on
// through the text head. Each output gets a graph of its own nodes: ggml-cuda keys its captured graphs by the
// first node, so a hidden graph and a logits graph sharing nodes would re-capture at every S2S text/audio switch.
class DecodeGraph {
public:
    DecodeGraph(const Weights & weights, const Lfm3BackboneConfig & config, int64_t codebooks, core::ExecutionContext & execution,
                blocks::StackState & state, const Lfm3WeightOptions & options, Lfm3Output output)
        : execution_(execution), capacity_(state.capacity()), codebooks_(codebooks), window_(config.stack.sliding_window) {
        ctx_ = blocks::graph_context(kGraphNodes, "backbone decode");
        auto * g = ctx_.get();
        core::ModuleBuildContext ctx{g, "lfm3_audio.backbone.decode", execution.backend_type()};
        const int64_t d = config.stack.hidden;

        token_ = input(g, GGML_TYPE_I32, 1);
        frame_rows_ = input(g, GGML_TYPE_I32, codebooks);
        text_weight_ = input(g, GGML_TYPE_F32, 1);
        frame_weight_ = input(g, GGML_TYPE_F32, 1);
        auto * text = ggml_get_rows(g, weights.token_lookup.tensor, token_);
        auto * frame = frame_embeddings(g, weights.frame_lookup.tensor, frame_rows_, codebooks);
        auto * x = ggml_add(g, ggml_mul(g, text, text_weight_), ggml_mul(g, frame, frame_weight_));

        blocks::StepInputs in;
        in.x = core::wrap_tensor(ggml_reshape_3d(g, x, d, 1, 1), TensorShape::from_dims({1, 1, d}));
        positions_ = input(g, GGML_TYPE_I32, 1);
        slots_ = input(g, GGML_TYPE_I64, 1);
        in.positions = core::wrap_tensor(positions_, TensorShape::from_dims({1}), GGML_TYPE_I32);
        in.slots = core::wrap_tensor(slots_, TensorShape::from_dims({1}), GGML_TYPE_I64);
        in.kv_rows = capacity_;
        if (has_kind(config.stack, LayerKind::FullAttention)) {
            full_mask_ = input(g, GGML_TYPE_F16, capacity_);
            in.full_mask = core::wrap_tensor(full_mask_, TensorShape::from_dims({1, capacity_}), GGML_TYPE_F16);
        }
        if (has_kind(config.stack, LayerKind::SlidingAttention)) {
            sliding_mask_ = input(g, GGML_TYPE_F16, capacity_);
            in.sliding_mask = core::wrap_tensor(sliding_mask_, TensorShape::from_dims({1, capacity_}), GGML_TYPE_F16);
        }

        graph_ = ggml_new_graph_custom(g, kGraphNodes, false);
        const auto hidden = blocks::build_stack(ctx, graph_, config.stack, weights.stack, in, &state.layers, options);
        out_ = output == Lfm3Output::Logits ? text_logits(ctx, hidden, weights, config).tensor : hidden.tensor;
        ggml_set_output(out_);
        ggml_build_forward_expand(graph_, out_);
        core::validate_backend_graph_with_cpu_extra_buffers(execution.backend(), graph_, "LFM3-Audio backbone decode graph");

        // Every node keeps its own memory: a step is tiny next to its weights.
        buffer_.reset(ggml_backend_alloc_ctx_tensors(g, execution.backend()));
        if (buffer_ == nullptr) {
            throw runtime::CapacityError("LFM3-Audio backbone decode graph does not fit in device memory at " + std::to_string(capacity_) +
                                         " cache rows");
        }
    }

    ~DecodeGraph() { core::release_backend_graph_resources(execution_.backend(), graph_, true); }

    DecodeGraph(const DecodeGraph &) = delete;
    DecodeGraph & operator=(const DecodeGraph &) = delete;

    std::vector<float> run(int64_t position, int32_t token, const std::vector<int32_t> & frame_rows) {
        const bool frame = !frame_rows.empty();
        // The unused input still needs valid rows: 0 * NaN would be NaN.
        const std::vector<int32_t> rows = frame ? frame_rows : std::vector<int32_t>(static_cast<size_t>(codebooks_), 0);
        const int32_t id = frame ? 0 : token;
        const float text_weight = frame ? 0.0f : 1.0f;
        const float frame_weight = frame ? 1.0f : 0.0f;
        const auto pos = static_cast<int32_t>(position);
        const int64_t slot = position;
        ggml_backend_tensor_set(token_, &id, 0, sizeof(id));
        ggml_backend_tensor_set(frame_rows_, rows.data(), 0, rows.size() * sizeof(int32_t));
        ggml_backend_tensor_set(text_weight_, &text_weight, 0, sizeof(float));
        ggml_backend_tensor_set(frame_weight_, &frame_weight, 0, sizeof(float));
        ggml_backend_tensor_set(positions_, &pos, 0, sizeof(pos));
        ggml_backend_tensor_set(slots_, &slot, 0, sizeof(slot));
        if (full_mask_ != nullptr) {
            const auto mask = blocks::window_mask({pos}, capacity_, 0);
            ggml_backend_tensor_set(full_mask_, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
        }
        if (sliding_mask_ != nullptr) {
            const auto mask = blocks::window_mask({pos}, capacity_, window_);
            ggml_backend_tensor_set(sliding_mask_, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
        }

        blocks::compute(execution_, graph_, "backbone decode");
        return core::read_tensor_f32(out_);
    }

private:
    core::ExecutionContext & execution_;
    int64_t capacity_ = 0;
    int64_t codebooks_ = 0;
    int64_t window_ = 0;
    ggml_context_ptr ctx_;
    ggml_tensor * token_ = nullptr;
    ggml_tensor * frame_rows_ = nullptr;
    ggml_tensor * text_weight_ = nullptr;
    ggml_tensor * frame_weight_ = nullptr;
    ggml_tensor * positions_ = nullptr;
    ggml_tensor * slots_ = nullptr;
    ggml_tensor * full_mask_ = nullptr;
    ggml_tensor * sliding_mask_ = nullptr;
    ggml_tensor * out_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_backend_buffer_ptr buffer_;
};

// The tied text head on one hidden row. Its shape never changes, so the graph and its buffer live as long as the runtime.
class HeadGraph {
public:
    HeadGraph(const Weights & weights, const Lfm3BackboneConfig & config, core::ExecutionContext & execution) : execution_(execution) {
        ctx_ = blocks::graph_context(kGraphNodes, "backbone head");
        auto * g = ctx_.get();
        core::ModuleBuildContext build{g, "lfm3_audio.backbone.head", execution.backend_type()};
        const int64_t d = config.stack.hidden;
        hidden_ = ggml_new_tensor_3d(g, GGML_TYPE_F32, d, 1, 1);
        ggml_set_input(hidden_);
        out_ = text_logits(build, core::wrap_tensor(hidden_, TensorShape::from_dims({1, 1, d})), weights, config).tensor;
        ggml_set_output(out_);
        graph_ = ggml_new_graph_custom(g, kGraphNodes, false);
        ggml_build_forward_expand(graph_, out_);
        core::validate_backend_graph_with_cpu_extra_buffers(execution.backend(), graph_, "LFM3-Audio backbone head graph");
        buffer_.reset(ggml_backend_alloc_ctx_tensors(g, execution.backend()));
        if (buffer_ == nullptr) {
            throw runtime::CapacityError("LFM3-Audio backbone head graph does not fit in device memory");
        }
    }

    ~HeadGraph() { core::release_backend_graph_resources(execution_.backend(), graph_, true); }

    HeadGraph(const HeadGraph &) = delete;
    HeadGraph & operator=(const HeadGraph &) = delete;

    std::vector<float> run(const std::vector<float> & hidden_row) {
        ggml_backend_tensor_set(hidden_, hidden_row.data(), 0, hidden_row.size() * sizeof(float));
        blocks::compute(execution_, graph_, "backbone head");
        return core::read_tensor_f32(out_);
    }

private:
    core::ExecutionContext & execution_;
    ggml_context_ptr ctx_;
    ggml_tensor * hidden_ = nullptr;
    ggml_tensor * out_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_backend_buffer_ptr buffer_;
};

// One prefill block. The runtime places it in its block allocator, whose buffer the kept blocks share, so every run sets all inputs.
class BlockGraph {
public:
    BlockGraph(const Weights & weights, const Lfm3BackboneConfig & config, core::ExecutionContext & execution, blocks::StackState & state,
               const Lfm3WeightOptions & options, int64_t begin, int64_t steps)
        : execution_(execution),
          begin_(begin),
          steps_(steps),
          kv_rows_(GGML_PAD(begin + steps, Lfm3BackboneRuntime::kPrefillBlock)),
          window_(config.stack.sliding_window) {
        ctx_ = blocks::graph_context(kGraphNodes, "backbone prefill block");
        auto * g = ctx_.get();
        core::ModuleBuildContext build{g, "lfm3_audio.backbone.prefill", execution.backend_type()};
        const int64_t d = config.stack.hidden;

        blocks::StepInputs in;
        x_ = ggml_new_tensor_3d(g, GGML_TYPE_F32, d, steps, 1);
        positions_ = input(g, GGML_TYPE_I32, steps);
        slots_ = input(g, GGML_TYPE_I64, steps);
        ggml_set_input(x_);
        in.x = core::wrap_tensor(x_, TensorShape::from_dims({1, steps, d}));
        in.positions = core::wrap_tensor(positions_, TensorShape::from_dims({steps}), GGML_TYPE_I32);
        in.slots = core::wrap_tensor(slots_, TensorShape::from_dims({steps}), GGML_TYPE_I64);
        in.kv_rows = kv_rows_;
        if (has_kind(config.stack, LayerKind::FullAttention)) {
            full_mask_ = input(g, GGML_TYPE_F16, kv_rows_, steps);
            in.full_mask = core::wrap_tensor(full_mask_, TensorShape::from_dims({steps, kv_rows_}), GGML_TYPE_F16);
        }
        if (has_kind(config.stack, LayerKind::SlidingAttention)) {
            sliding_mask_ = input(g, GGML_TYPE_F16, kv_rows_, steps);
            in.sliding_mask = core::wrap_tensor(sliding_mask_, TensorShape::from_dims({steps, kv_rows_}), GGML_TYPE_F16);
        }

        graph_ = ggml_new_graph_custom(g, kGraphNodes, false);
        out_ = blocks::build_stack(build, graph_, config.stack, weights.stack, in, &state.layers, options).tensor;
        ggml_set_output(out_);
        ggml_build_forward_expand(graph_, out_);
        core::validate_backend_graph_with_cpu_extra_buffers(execution.backend(), graph_, "LFM3-Audio backbone prefill graph");
        if (!ggml_is_contiguous(out_)) {
            fail("prefill hidden must be contiguous");
        }
    }

    ~BlockGraph() { core::release_backend_graph_resources(execution_.backend(), graph_, true); }

    BlockGraph(const BlockGraph &) = delete;
    BlockGraph & operator=(const BlockGraph &) = delete;

    [[nodiscard]] int64_t steps() const noexcept { return steps_; }
    [[nodiscard]] ggml_cgraph * graph() const noexcept { return graph_; }

    std::vector<float> run(const float * x, const std::vector<int64_t> & wanted) {
        const int64_t d = x_->ne[0];
        std::vector<int32_t> positions(static_cast<size_t>(steps_));
        std::vector<int64_t> slots(static_cast<size_t>(steps_));
        for (int64_t i = 0; i < steps_; ++i) {
            positions[static_cast<size_t>(i)] = static_cast<int32_t>(begin_ + i);
            slots[static_cast<size_t>(i)] = begin_ + i;
        }
        ggml_backend_tensor_set(x_, x, 0, static_cast<size_t>(steps_ * d) * sizeof(float));
        ggml_backend_tensor_set(positions_, positions.data(), 0, positions.size() * sizeof(int32_t));
        ggml_backend_tensor_set(slots_, slots.data(), 0, slots.size() * sizeof(int64_t));
        // Masks over absolute cache rows: a sliding window reaches back into earlier blocks.
        if (full_mask_ != nullptr) {
            const auto mask = blocks::window_mask(positions, kv_rows_, 0);
            ggml_backend_tensor_set(full_mask_, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
        }
        if (sliding_mask_ != nullptr) {
            const auto mask = blocks::window_mask(positions, kv_rows_, window_);
            ggml_backend_tensor_set(sliding_mask_, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
        }
        blocks::compute(execution_, graph_, "backbone prefill block");

        std::vector<float> out(wanted.size() * static_cast<size_t>(d));
        for (size_t i = 0; i < wanted.size(); ++i) {
            ggml_backend_tensor_get(out_, out.data() + i * static_cast<size_t>(d), static_cast<size_t>(wanted[i] - begin_) * out_->nb[1],
                                    static_cast<size_t>(d) * sizeof(float));
        }
        // Eager every run, as before: ggml-cuda would capture a kept block on its 2nd run (+1.5-2.6 ms) to save 1.2-1.7 ms per later run.
        core::release_backend_graph_resources(execution_.backend(), graph_, true);
        return out;
    }

private:
    core::ExecutionContext & execution_;
    int64_t begin_ = 0;
    int64_t steps_ = 0;
    int64_t kv_rows_ = 0;
    int64_t window_ = 0;
    ggml_context_ptr ctx_;
    ggml_tensor * x_ = nullptr;
    ggml_tensor * positions_ = nullptr;
    ggml_tensor * slots_ = nullptr;
    ggml_tensor * full_mask_ = nullptr;
    ggml_tensor * sliding_mask_ = nullptr;
    ggml_tensor * out_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
};

}  // namespace

struct Lfm3BackboneRuntime::Impl {
    Impl(std::shared_ptr<const assets::TensorSource> source_in, const Lfm3BackboneConfig & config_in, int64_t codebooks_in,
         int64_t audio_vocab_in, core::ExecutionContext & execution_in, const Lfm3WeightOptions & options_in)
        : source(std::move(source_in)),
          config(config_in),
          codebooks(codebooks_in),
          audio_vocab(audio_vocab_in),
          execution(execution_in),
          options(options_in),
          weights(load_weights(*source, config, codebooks, audio_vocab, execution, options)) {}

    [[nodiscard]] int64_t hidden() const noexcept { return config.stack.hidden; }

    std::vector<int32_t> frame_rows(const std::vector<std::vector<int32_t>> & frames) const {
        std::vector<int32_t> rows;
        rows.reserve(frames.size() * static_cast<size_t>(codebooks));
        for (const auto & codes : frames) {
            if (static_cast<int64_t>(codes.size()) != codebooks) {
                fail("an audio frame needs " + std::to_string(codebooks) + " codes, got " + std::to_string(codes.size()));
            }
            for (size_t k = 0; k < codes.size(); ++k) {
                if (codes[k] < 0 || codes[k] >= audio_vocab) {
                    fail("audio code " + std::to_string(codes[k]) + " is outside the codebook");
                }
                rows.push_back(static_cast<int32_t>(static_cast<int64_t>(k) * audio_vocab + codes[k]));
            }
        }
        return rows;
    }

    // Text rows of `ids` and frame sums of `rows` in one graph, read back row-major [ids][hidden], [frames][hidden].
    std::pair<std::vector<float>, std::vector<float>> embed(const std::vector<int32_t> & ids, const std::vector<int32_t> & rows) {
        auto ctx = blocks::graph_context(kGraphNodes, "backbone embedding");
        auto * g = ctx.get();
        ggml_tensor * id_in = nullptr;
        ggml_tensor * row_in = nullptr;
        ggml_tensor * text = nullptr;
        ggml_tensor * frames = nullptr;
        auto * graph = ggml_new_graph_custom(g, kGraphNodes, false);
        if (!ids.empty()) {
            id_in = input(g, GGML_TYPE_I32, static_cast<int64_t>(ids.size()));
            text = ggml_get_rows(g, weights.token_lookup.tensor, id_in);
            ggml_set_output(text);
            ggml_build_forward_expand(graph, text);
        }
        if (!rows.empty()) {
            row_in = input(g, GGML_TYPE_I32, static_cast<int64_t>(rows.size()));
            frames = frame_embeddings(g, weights.frame_lookup.tensor, row_in, codebooks);
            ggml_set_output(frames);
            ggml_build_forward_expand(graph, frames);
        }
        std::pair<std::vector<float>, std::vector<float>> out;
        if (ids.empty() && rows.empty()) {
            return out;
        }
        ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(g, execution.backend()));
        if (buffer == nullptr) {
            throw runtime::CapacityError("LFM3-Audio backbone embedding graph does not fit in device memory");
        }
        if (id_in != nullptr) {
            ggml_backend_tensor_set(id_in, ids.data(), 0, ids.size() * sizeof(int32_t));
        }
        if (row_in != nullptr) {
            ggml_backend_tensor_set(row_in, rows.data(), 0, rows.size() * sizeof(int32_t));
        }
        blocks::compute(execution, graph, "backbone embedding");
        for (auto [t, dst] : {std::pair{text, &out.first}, std::pair{frames, &out.second}}) {
            if (t != nullptr) {
                *dst = core::read_tensor_f32(t);
            }
        }
        core::release_backend_graph_resources(execution.backend(), graph, true);
        return out;
    }

    void ensure_capacity(int64_t capacity) {
        if (state != nullptr && state->capacity() == capacity) {
            return;
        }
        block_graphs.clear();
        last_steps.clear();
        decode_hidden.reset();
        decode_logits.reset();
        state.reset();
        const auto cache_type = lfm3_reference_precision(options) ? GGML_TYPE_F32 : GGML_TYPE_F16;
        state = std::make_unique<blocks::StackState>(execution, config.stack, capacity, cache_type, "backbone");
    }

    // One prefill block: x rows [steps][hidden] at positions begin.., attending over cache rows [0, kv_rows).
    // Returns the final-normed hidden rows `wanted` (absolute positions inside the block), row-major.
    std::vector<float> run_block(const float * x, int64_t begin, int64_t steps, const std::vector<int64_t> & wanted) {
        const auto index = static_cast<size_t>(begin / kPrefillBlock);
        if (block_graphs.size() <= index) {
            block_graphs.resize(index + 1);
            last_steps.resize(index + 1);
        }
        // Kept only when this block had these rows the last time it ran: on Linux a kept graph built 0.1-0.2 ms slower than a dropped one.
        const bool repeated = std::exchange(last_steps[index], steps) == steps;
        auto & kept = block_graphs[index];
        if (kept != nullptr && kept->steps() == steps) {
            return kept->run(x, wanted);
        }
        auto built = std::make_unique<BlockGraph>(weights, config, execution, *state, options, begin, steps);
        const size_t held = ggml_gallocr_get_buffer_size(block_allocator.get(), 0);
        const bool placed = ggml_gallocr_reserve(block_allocator.get(), built->graph()) &&
                            ggml_gallocr_alloc_graph(block_allocator.get(), built->graph());
        // Outgrowing the buffer, or failing to, frees the one the kept graphs' tensors point into.
        // shortcut: the size shows this only while a plan fits one buffer chunk (blocks plan 3-27 MiB); compare chunks if that changes.
        if (ggml_gallocr_get_buffer_size(block_allocator.get(), 0) != held) {
            std::fill(block_graphs.begin(), block_graphs.end(), nullptr);
        }
        if (!placed) {
            throw runtime::CapacityError("LFM3-Audio backbone prefill block does not fit in device memory at " +
                                         std::to_string(GGML_PAD(begin + steps, kPrefillBlock)) + " cache rows");
        }
        ++built_blocks;
        if (!repeated) {
            return built->run(x, wanted);
        }
        kept = std::move(built);
        return kept->run(x, wanted);
    }

    DecodeGraph & require_started(Lfm3Output output) {
        if (!started) {
            fail("step before prefill()");
        }
        if (position >= state->capacity()) {
            throw runtime::CapacityError("LFM3-Audio backbone cache is full at " + std::to_string(state->capacity()) + " positions");
        }
        // Built on first use: ASR steps only need logits, TTS steps only the hidden state.
        auto & decode = output == Lfm3Output::Logits ? decode_logits : decode_hidden;
        if (decode == nullptr) {
            decode = std::make_unique<DecodeGraph>(weights, config, codebooks, execution, *state, options, output);
        }
        return *decode;
    }

    std::vector<float> output(std::vector<float> hidden_row, Lfm3Output kind) {
        return kind == Lfm3Output::Hidden ? std::move(hidden_row) : head(hidden_row);
    }

    std::vector<float> head(const std::vector<float> & hidden_row) {
        const int64_t d = hidden();
        if (static_cast<int64_t>(hidden_row.size()) != d) {
            fail("logits() needs a hidden state of " + std::to_string(d) + " values");
        }
        if (head_graph == nullptr) {
            head_graph = std::make_unique<HeadGraph>(weights, config, execution);
        }
        return head_graph->run(hidden_row);
    }

    std::shared_ptr<const assets::TensorSource> source;
    Lfm3BackboneConfig config;
    int64_t codebooks = 0;
    int64_t audio_vocab = 0;
    core::ExecutionContext & execution;
    Lfm3WeightOptions options;
    Weights weights;
    std::unique_ptr<blocks::StackState> state;  // K/V row p holds position p
    std::unique_ptr<DecodeGraph> decode_hidden;
    std::unique_ptr<DecodeGraph> decode_logits;
    ggml_gallocr_ptr block_allocator;
    std::vector<std::unique_ptr<BlockGraph>> block_graphs;  // by block index: they read `state` and live in block_allocator's buffer
    std::vector<int64_t> last_steps;                        // by block index: its rows the last time it ran at this capacity
    std::unique_ptr<HeadGraph> head_graph;                  // built on first use: TTS never needs it
    int64_t built_blocks = 0;                               // block graphs this prefill built, for the timing log
    int64_t position = 0;
    bool started = false;
};

Lfm3BackboneRuntime::Lfm3BackboneRuntime(std::shared_ptr<const assets::TensorSource> model, const Lfm3BackboneConfig & config,
                                         int64_t codebooks, int64_t audio_vocab_size, core::ExecutionContext & execution,
                                         const Lfm3WeightOptions & options) {
    if (model == nullptr || codebooks <= 0 || audio_vocab_size <= 0 || config.vocab_size <= 0) {
        fail("needs the model tensors, a vocabulary and the codebook sizes");
    }
    impl_ = std::make_unique<Impl>(std::move(model), config, codebooks, audio_vocab_size, execution, options);
    impl_->block_allocator.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend())));
    if (impl_->block_allocator == nullptr) {
        fail("failed to create the prefill allocator");
    }
}

Lfm3BackboneRuntime::~Lfm3BackboneRuntime() = default;

std::vector<float> Lfm3BackboneRuntime::prefill(const Lfm3Prompt & prompt, int32_t audio_id, int32_t audio_out_id,
                                                const Lfm3AudioEmbeddings & audio, int64_t max_new_steps, Lfm3Output output,
                                                Taps * taps) {
    auto & m = *impl_;
    m.started = false;
    m.position = 0;
    m.built_blocks = 0;
    const int64_t d = m.hidden();
    const auto steps = static_cast<int64_t>(prompt.ids.size());
    if (steps == 0 || max_new_steps < 0) {
        fail("prefill needs a prompt and a nonnegative step budget");
    }
    const int64_t limit = GGML_PAD(kLfm3MaxConversationSteps, kPrefillBlock);
    if (steps > limit || max_new_steps > limit - steps) {
        throw runtime::CapacityError("LFM3-Audio request needs " + std::to_string(steps) + " prompt steps plus " +
                                     std::to_string(max_new_steps) + " more, more than " + std::to_string(limit));
    }

    int64_t audio_slots = 0;
    int64_t frame_slots = 0;
    for (const int32_t id : prompt.ids) {
        if (id < 0 || id >= m.config.vocab_size) {
            fail("prompt token id " + std::to_string(id) + " is outside the vocabulary");
        }
        audio_slots += id == audio_id ? 1 : 0;
        frame_slots += id == audio_out_id ? 1 : 0;
    }
    if (audio_slots != audio.tokens || audio.values.size() != static_cast<size_t>(audio.tokens * d) || (audio.tokens > 0 && audio.hidden != d)) {
        fail("the prompt has " + std::to_string(audio_slots) + " <|audio|> slots but " + std::to_string(audio.tokens) +
             " audio embeddings of the backbone width");
    }
    if (!std::all_of(audio.values.begin(), audio.values.end(), [](float v) { return std::isfinite(v); })) {
        fail("audio embeddings are not finite");
    }
    if (frame_slots != static_cast<int64_t>(prompt.frames.size())) {
        fail("the prompt has " + std::to_string(frame_slots) + " <|audio_out|> slots but " + std::to_string(prompt.frames.size()) + " frames");
    }
    if (taps != nullptr) {
        for (const int64_t p : taps->positions) {
            if (p < 0 || p >= steps) {
                fail("tap position " + std::to_string(p) + " is outside the prompt");
            }
        }
    }

    const auto prefill_start = std::chrono::steady_clock::now();
    m.ensure_capacity(GGML_PAD(steps + max_new_steps, kPrefillBlock));
    // A reused cache keeps the last request's rows, and the mask does not stop a non-finite one from poisoning attention.
    m.state->clear();

    auto [x, frame_values] = m.embed(prompt.ids, m.frame_rows(prompt.frames));
    size_t next_audio = 0;
    size_t next_frame = 0;
    for (int64_t i = 0; i < steps; ++i) {
        const int32_t id = prompt.ids[static_cast<size_t>(i)];
        const float * row = id == audio_id       ? audio.values.data() + (next_audio++) * static_cast<size_t>(d)
                            : id == audio_out_id ? frame_values.data() + (next_frame++) * static_cast<size_t>(d)
                                                 : nullptr;
        if (row != nullptr) {
            std::copy(row, row + d, x.begin() + i * d);
        }
    }

    std::vector<float> last;
    if (taps != nullptr) {
        taps->rows.assign(taps->positions.size() * static_cast<size_t>(d), 0.0f);
    }
    for (int64_t begin = 0; begin < steps; begin += kPrefillBlock) {
        const int64_t length = std::min(kPrefillBlock, steps - begin);
        const int64_t end = begin + length;
        std::vector<int64_t> wanted;
        std::vector<size_t> tap_index;
        if (taps != nullptr) {
            for (size_t i = 0; i < taps->positions.size(); ++i) {
                if (taps->positions[i] >= begin && taps->positions[i] < end) {
                    wanted.push_back(taps->positions[i]);
                    tap_index.push_back(i);
                }
            }
        }
        if (end == steps) {
            wanted.push_back(steps - 1);
        }
        const auto rows = m.run_block(x.data() + begin * d, begin, length, wanted);
        for (size_t i = 0; i < tap_index.size(); ++i) {
            std::copy_n(rows.begin() + static_cast<std::ptrdiff_t>(i) * d, d, taps->rows.begin() + static_cast<std::ptrdiff_t>(tap_index[i]) * d);
        }
        if (end == steps) {
            last.assign(rows.end() - d, rows.end());
        }
    }
    m.position = steps;
    m.started = true;
    auto out = m.output(last, output);
    debug::timing_log_scalar("lfm3_audio.backbone.prefill.ms", engine::debug::elapsed_ms(prefill_start));
    debug::timing_log_scalar("lfm3_audio.backbone.prefill.built_blocks", m.built_blocks);
    if (lfm3_dump_enabled()) {
        lfm3_dump("prefill_hidden", last.data(), {d});
        if (output == Lfm3Output::Logits) {
            lfm3_dump("prefill_logits", out.data(), {static_cast<int64_t>(out.size())});
        }
    }
    return out;
}

std::vector<float> Lfm3BackboneRuntime::step_text(int32_t token, Lfm3Output output) {
    auto & m = *impl_;
    if (token < 0 || token >= m.config.vocab_size) {
        fail("token id " + std::to_string(token) + " is outside the vocabulary");
    }
    auto out = m.require_started(output).run(m.position, token, {});
    ++m.position;
    return out;
}

std::vector<float> Lfm3BackboneRuntime::step_frame(const std::vector<int32_t> & codes, Lfm3Output output) {
    auto & m = *impl_;
    const auto rows = m.frame_rows({codes});
    auto out = m.require_started(output).run(m.position, 0, rows);
    ++m.position;
    return out;
}

std::vector<float> Lfm3BackboneRuntime::logits(const std::vector<float> & hidden) {
    return impl_->head(hidden);
}

std::vector<float> Lfm3BackboneRuntime::embed_frames(const std::vector<std::vector<int32_t>> & frames) {
    return impl_->embed({}, impl_->frame_rows(frames)).second;
}

int64_t Lfm3BackboneRuntime::position() const noexcept {
    return impl_->position;
}

int64_t Lfm3BackboneRuntime::capacity() const noexcept {
    return impl_->state == nullptr ? 0 : impl_->state->capacity();
}

}  // namespace engine::community_models::lfm3_audio
