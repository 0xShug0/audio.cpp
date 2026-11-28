#include "engine/community_models/lfm3_audio/depth.h"

#include "engine/community_models/lfm3_audio/generator.h"
#include "engine/framework/core/backend.h"
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

// One 12-token pass, built once. The trunk ends with its own final norm, before the per-codebook RMSNorm.
class PassGraph {
public:
    PassGraph(const Lfm3DepthWeights & weights, const Lfm3DepthConfig & config, core::ExecutionContext & execution,
              const Lfm3WeightOptions & options)
        : config_(config), execution_(execution), ctx_(ggml_init({kGraphNodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(kGraphNodes, false), nullptr, true})) {
        if (!ctx_) fail("graph context allocation failed");
        auto * g = ctx_.get();
        core::ModuleBuildContext ctx{g, "lfm3_audio.depth", execution.backend_type()};
        graph_ = ggml_new_graph_custom(g, kGraphNodes, false);
        const int64_t d = config.stack.hidden_size;
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
        auto lin = modules::LinearModule({config.input_size, kCodebooks * d, true, kLfm3MatmulPrecision})
                       .build(ctx, hidden, {weights.input_weight, weights.input_bias});
        // keep/drop are exactly 1/0, so an unmasked slot adds E_i[code_i] and a masked one mask_embedding, bit for bit.
        auto * codes = ggml_mul(g, ggml_get_rows(g, weights.embeddings.tensor, ids_), keep_);
        auto * mask = ggml_mul(g, weights.mask_rows.tensor, drop_);
        auto * x = ggml_add(g, ggml_reshape_2d(g, lin.tensor, d, kCodebooks), ggml_add(g, codes, mask));

        modules::LfmDecoderInputs in;
        in.hidden = core::wrap_tensor(ggml_reshape_3d(g, x, d, kCodebooks, 1), TensorShape::from_dims({1, kCodebooks, d}));
        in.positions = core::wrap_tensor(positions_, TensorShape::from_dims({kCodebooks}), GGML_TYPE_I32);
        auto o = modules::build_lfm_decoder(ctx, in, weights.stack, lfm3_decoder_config(config.stack, options)).hidden;

        // Lfm3AudioRMSNorm per codebook, then logits_i = E_i . o_i as one batched matmul over the codebooks.
        auto * normed = ggml_mul(g, ggml_rms_norm(g, ggml_reshape_2d(g, o.tensor, d, kCodebooks), config.stack.norm_eps), weights.norms.tensor);
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
        core::set_backend_threads(execution_.backend(), std::max(1, execution_.config().threads));
        if (core::compute_backend_graph(execution_.backend(), graph_) != GGML_STATUS_SUCCESS) fail("graph compute failed");
        ggml_backend_synchronize(execution_.backend());

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
    if (config.codebooks != kCodebooks || config.audio_vocab_size < 2) {
        fail("expects 12 codebooks, at least 2 codes and a bidirectional stack");
    }
}

}  // namespace

struct Lfm3DepthDecoderRuntime::Impl {
    Impl(Lfm3DepthWeights weights_in, const Lfm3DepthConfig & config_in, core::ExecutionContext & execution,
         const Lfm3WeightOptions & options)
        : config((validate_config(config_in), config_in)),
          weights(std::move(weights_in)),
          graph(weights, config, execution, options) {}

    Lfm3DepthConfig config;
    Lfm3DepthWeights weights;
    PassGraph graph;
};

Lfm3DepthDecoderRuntime::Lfm3DepthDecoderRuntime(Lfm3DepthWeights weights, const Lfm3DepthConfig & config,
                                                 core::ExecutionContext & execution, const Lfm3WeightOptions & options)
    : impl_(std::make_unique<Impl>(std::move(weights), config, execution, options)) {}

Lfm3DepthDecoderRuntime::~Lfm3DepthDecoderRuntime() = default;

std::vector<float> Lfm3DepthDecoderRuntime::logits(const std::vector<float> & hidden, const Lfm3Codes & codes,
                                                   const std::array<bool, 12> & masked) {
    return impl_->graph.run(hidden, codes, masked);
}

Lfm3Codes lfm3_maskgit_frame(Lfm3DepthDecoderRuntime & depth, const std::vector<float> & hidden, const Lfm3Codes & order,
                             const std::vector<int64_t> & schedule, Lfm3AudioSampler & sampler, int32_t end_of_audio) {
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
