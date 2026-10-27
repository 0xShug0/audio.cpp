#include "engine/community_models/lfm3_audio/depth.h"

#include "../lfm2_audio/weight_stores.h"

#include "engine/community_models/lfm3_audio/blocks.h"
#include "engine/community_models/lfm3_audio/sampling.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/linear_module.h"

#include <ggml-backend.h>
#include <ggml-cpp.h>
#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::community_models::lfm3_audio {
namespace {

using core::TensorShape;
using core::TensorValue;

constexpr int64_t kCodebooks = 12;
constexpr size_t kGraphNodes = 2048;

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error("LFM3-Audio depth decoder: " + message);
}

struct DepthWeights {
    std::unique_ptr<WeightStores> stores;
    TensorValue input_weight;  // depth_linear [codebooks * hidden, input_size]
    TensorValue input_bias;    // [codebooks * hidden]
    TensorValue mask_rows;     // mask_embedding repeated per codebook, F32 [codebooks, hidden]
    TensorValue embeddings;    // depth_embeddings.{i}.embedding packed, [codebooks * vocab, hidden]
    TensorValue norms;         // depth_embeddings.{i}.embedding_norm packed, F32 [codebooks, hidden]
    blocks::StackWeights stack;
};

DepthWeights load_weights(const assets::TensorSource & source, const Lfm3DepthConfig & config, core::ExecutionContext & execution,
                          const Lfm3WeightOptions & options) {
    const int64_t d = config.stack.hidden;
    const int64_t vocab = config.audio_vocab_size;
    DepthWeights out;
    out.stores = std::make_unique<WeightStores>(execution, "lfm3_audio.depth.weights", 1024 * ggml_tensor_overhead(), options.cpu_repack);
    auto & store = out.stores->plain();

    out.input_weight =
        blocks::load_matmul(*out.stores, execution.backend(), source, "depth_linear.weight", {kCodebooks * d, config.input_size}, options);
    out.input_bias = store.load_f32_tensor(source, "depth_linear.bias", {kCodebooks * d});

    const auto mask = source.require_f32("mask_embedding", {d});
    std::vector<float> mask_rows;
    for (int64_t i = 0; i < kCodebooks; ++i) {
        mask_rows.insert(mask_rows.end(), mask.begin(), mask.end());
    }
    out.mask_rows = store.make_f32(TensorShape::from_dims({kCodebooks, d}), std::move(mask_rows));

    // The tables are gathered (inputs) and batch-multiplied (tied heads), so they stay in the plain store.
    std::vector<core::BackendWeightStore::TensorPart> tables;
    std::vector<float> norms;
    for (int64_t i = 0; i < kCodebooks; ++i) {
        const std::string p = "depth_embeddings." + std::to_string(i) + ".";
        tables.push_back({p + "embedding.weight", {vocab, d}});
        const auto norm = source.require_f32(p + "embedding_norm.weight", {d});
        norms.insert(norms.end(), norm.begin(), norm.end());
    }
    const ggml_type table_type = assets::ggml_type_for_tensor_dtype(source.require_metadata(tables.front().name).dtype);
    auto table_storage = lfm3_storage_type(execution.backend(), table_type, options);
    if (table_storage == assets::TensorStorageType::Native && !lfm3_backend_gathers(execution.backend(), table_type)) {
        table_storage = assets::TensorStorageType::F16;
    }
    out.embeddings = store.load_tensor(source, tables, table_storage);
    out.norms = store.make_f32(TensorShape::from_dims({kCodebooks, d}), std::move(norms));

    out.stack = blocks::load_stack(*out.stores, execution.backend(), source, "depthformer.model.", config.stack, options);
    out.stores->upload();
    return out;
}

// One 12-token pass, built once. The trunk ends with its own final norm, before the per-codebook RMSNorm.
class PassGraph {
public:
    PassGraph(const DepthWeights & weights, const Lfm3DepthConfig & config, core::ExecutionContext & execution,
              const Lfm3WeightOptions & options)
        : config_(config), execution_(execution), ctx_(blocks::graph_context(kGraphNodes, "depth decoder")) {
        auto * g = ctx_.get();
        core::ModuleBuildContext ctx{g, "lfm3_audio.depth", execution.backend_type()};
        graph_ = ggml_new_graph_custom(g, kGraphNodes, false);
        const int64_t d = config.stack.hidden;
        const int64_t vocab = config.audio_vocab_size;

        hidden_ = ggml_new_tensor_1d(g, GGML_TYPE_F32, config.input_size);
        ids_ = ggml_new_tensor_1d(g, GGML_TYPE_I32, kCodebooks);
        keep_ = ggml_new_tensor_2d(g, GGML_TYPE_F32, 1, kCodebooks);
        drop_ = ggml_new_tensor_2d(g, GGML_TYPE_F32, 1, kCodebooks);
        positions_ = ggml_new_tensor_1d(g, GGML_TYPE_I32, kCodebooks);
        for (auto * t : {hidden_, ids_, keep_, drop_, positions_}) {
            ggml_set_input(t);
        }

        const auto hidden = core::wrap_tensor(hidden_, TensorShape::from_dims({1, config.input_size}));
        auto lin = modules::LinearModule({config.input_size, kCodebooks * d, true, blocks::kMatmulPrecision})
                       .build(ctx, hidden, {weights.input_weight, weights.input_bias});
        // keep/drop are exactly 1/0, so an unmasked slot adds E_i[code_i] and a masked one mask_embedding, bit for bit.
        auto * codes = ggml_mul(g, ggml_get_rows(g, weights.embeddings.tensor, ids_), keep_);
        auto * mask = ggml_mul(g, weights.mask_rows.tensor, drop_);
        auto * x = ggml_add(g, ggml_reshape_2d(g, lin.tensor, d, kCodebooks), ggml_add(g, codes, mask));

        blocks::StepInputs in;
        in.x = core::wrap_tensor(ggml_reshape_3d(g, x, d, kCodebooks, 1), TensorShape::from_dims({1, kCodebooks, d}));
        in.positions = core::wrap_tensor(positions_, TensorShape::from_dims({kCodebooks}), GGML_TYPE_I32);
        auto o = blocks::build_stack(ctx, graph_, config.stack, weights.stack, in, nullptr, options);

        // Lfm3AudioRMSNorm per codebook, then logits_i = E_i . o_i as one batched matmul over the codebooks.
        auto * normed = ggml_mul(g, ggml_rms_norm(g, ggml_reshape_2d(g, o.tensor, d, kCodebooks), config.stack.eps), weights.norms.tensor);
        auto * heads = ggml_reshape_3d(g, weights.embeddings.tensor, d, vocab, kCodebooks);
        logits_ = ggml_mul_mat(g, heads, ggml_reshape_3d(g, normed, d, 1, kCodebooks));  // [vocab, 1, codebooks]
        // CUDA's batched cuBLAS path otherwise writes F16/BF16 heads' logits in that type, so greedy picks between
        // rounded ties (top-2 gap 0 for bf16).
        ggml_mul_mat_set_prec(logits_, GGML_PREC_F32);
        ggml_set_output(logits_);
        ggml_build_forward_expand(graph_, logits_);
        core::validate_backend_graph_with_cpu_extra_buffers(execution.backend(), graph_, "LFM3-Audio depth graph");

        buffer_.reset(ggml_backend_alloc_ctx_tensors(g, execution.backend()));
        if (buffer_ == nullptr) {
            fail("the graph does not fit in device memory");
        }
        std::vector<int32_t> positions(kCodebooks);
        std::iota(positions.begin(), positions.end(), 0);
        ggml_backend_tensor_set(positions_, positions.data(), 0, positions.size() * sizeof(int32_t));
    }

    ~PassGraph() { core::release_backend_graph_resources(execution_.backend(), graph_, true); }

    PassGraph(const PassGraph &) = delete;
    PassGraph & operator=(const PassGraph &) = delete;

    std::vector<float> run(const std::vector<float> & hidden, const Lfm3Codes & codes, const std::array<bool, 12> & masked) {
        if (static_cast<int64_t>(hidden.size()) != config_.input_size) {
            fail("input has " + std::to_string(hidden.size()) + " values, the backbone hidden size is " +
                 std::to_string(config_.input_size));
        }
        const int64_t vocab = config_.audio_vocab_size;
        std::vector<int32_t> ids(kCodebooks);
        std::vector<float> keep(kCodebooks);
        std::vector<float> drop(kCodebooks);
        for (int64_t i = 0; i < kCodebooks; ++i) {
            const auto k = static_cast<size_t>(i);
            if (!masked[k] && (codes[k] < 0 || codes[k] >= vocab)) {
                fail("code " + std::to_string(codes[k]) + " of codebook " + std::to_string(i) + " is outside [0, " + std::to_string(vocab) +
                     ")");
            }
            // A masked slot's code is irrelevant; row 0 of its table keeps the gather in range.
            ids[k] = static_cast<int32_t>(i * vocab + (masked[k] ? 0 : codes[k]));
            keep[k] = masked[k] ? 0.0f : 1.0f;
            drop[k] = 1.0f - keep[k];
        }
        ggml_backend_tensor_set(hidden_, hidden.data(), 0, hidden.size() * sizeof(float));
        ggml_backend_tensor_set(ids_, ids.data(), 0, ids.size() * sizeof(int32_t));
        ggml_backend_tensor_set(keep_, keep.data(), 0, keep.size() * sizeof(float));
        ggml_backend_tensor_set(drop_, drop.data(), 0, drop.size() * sizeof(float));
        blocks::compute(execution_, graph_, "depth decoder");

        auto logits = core::read_tensor_f32(logits_);
        if (!std::all_of(logits.begin(), logits.end(), [](float value) { return std::isfinite(value); })) {
            fail("non-finite logits");
        }
        return logits;
    }

private:
    const Lfm3DepthConfig & config_;
    core::ExecutionContext & execution_;
    ggml_context_ptr ctx_;
    ggml_cgraph * graph_ = nullptr;
    ggml_tensor * hidden_ = nullptr;
    ggml_tensor * ids_ = nullptr;
    ggml_tensor * keep_ = nullptr;
    ggml_tensor * drop_ = nullptr;
    ggml_tensor * positions_ = nullptr;
    ggml_tensor * logits_ = nullptr;
    ggml_backend_buffer_ptr buffer_;
};

void validate_config(const Lfm3DepthConfig & config) {
    if (config.codebooks != kCodebooks || config.audio_vocab_size < 2 || config.stack.causal) {
        fail("expects 12 codebooks, at least 2 codes and a bidirectional stack");
    }
}

}  // namespace

struct Lfm3DepthDecoderRuntime::Impl {
    Impl(std::shared_ptr<const assets::TensorSource> source_in, const Lfm3DepthConfig & config_in, core::ExecutionContext & execution,
         const Lfm3WeightOptions & options)
        : source(std::move(source_in)),
          config((validate_config(config_in), config_in)),
          weights(load_weights(*source, config, execution, options)),
          graph(weights, config, execution, options) {}

    std::shared_ptr<const assets::TensorSource> source;
    Lfm3DepthConfig config;
    DepthWeights weights;
    PassGraph graph;
};

Lfm3DepthDecoderRuntime::Lfm3DepthDecoderRuntime(std::shared_ptr<const assets::TensorSource> model, const Lfm3DepthConfig & config,
                                                 core::ExecutionContext & execution, const Lfm3WeightOptions & options)
    : impl_(std::make_unique<Impl>(std::move(model), config, execution, options)) {}

Lfm3DepthDecoderRuntime::~Lfm3DepthDecoderRuntime() = default;

std::vector<float> Lfm3DepthDecoderRuntime::logits(const std::vector<float> & hidden, const Lfm3Codes & codes,
                                                   const std::array<bool, 12> & masked) {
    return impl_->graph.run(hidden, codes, masked);
}

Lfm3Codes lfm3_maskgit_frame(Lfm3DepthDecoderRuntime & depth, const std::vector<float> & hidden, const Lfm3Codes & order,
                             const std::vector<int64_t> & schedule, Lfm3AudioSampler & sampler, int32_t end_of_audio,
                             Lfm3FrameTaps * taps) {
    std::array<bool, 12> seen{};
    for (const auto codebook : order) {
        if (codebook < 0 || codebook >= kCodebooks || seen[static_cast<size_t>(codebook)]) {
            fail("MaskGIT order must be a permutation of the 12 codebooks");
        }
        seen[static_cast<size_t>(codebook)] = true;
    }
    if (std::any_of(schedule.begin(), schedule.end(), [](int64_t n) { return n <= 0; }) ||
        std::accumulate(schedule.begin(), schedule.end(), int64_t{0}) != kCodebooks) {
        fail("MaskGIT schedule must be positive pass sizes that sum to 12");
    }

    Lfm3Codes codes{};
    std::array<bool, 12> masked;
    masked.fill(true);
    size_t next = 0;
    std::vector<float> row;
    for (const auto picks : schedule) {
        const auto logits = depth.logits(hidden, codes, masked);
        if (taps != nullptr) {
            taps->pass_logits.push_back(logits);
        }
        const size_t vocab = logits.size() / kCodebooks;
        for (int64_t i = 0; i < picks; ++i, ++next) {
            const auto codebook = static_cast<size_t>(order[next]);
            row.assign(logits.begin() + static_cast<std::ptrdiff_t>(codebook * vocab),
                       logits.begin() + static_cast<std::ptrdiff_t>((codebook + 1) * vocab));
            codes[codebook] = sampler.pick(static_cast<int64_t>(codebook), row);
            masked[codebook] = false;
            // Any end-of-audio code makes the whole frame end-of-audio, whatever later picks and passes give,
            // so stopping here changes only how many draws the frame takes.
            if (codes[codebook] == end_of_audio) {
                codes.fill(end_of_audio);
                return codes;
            }
        }
    }
    return codes;
}

}  // namespace engine::community_models::lfm3_audio
