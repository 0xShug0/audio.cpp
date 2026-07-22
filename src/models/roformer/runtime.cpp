#include "engine/models/roformer/runtime.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/audio/dsp.h"
#include "engine/framework/audio/fft.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"
#include "engine/framework/core/constant_tensor_cache.h"
#include <ggml-backend.h>
#include <ggml.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::roformer {
namespace {

namespace assets_ns = engine::assets;
namespace modules = engine::modules;
namespace binding = engine::modules::binding;

using Clock = std::chrono::steady_clock;
using LinearWeights = modules::LinearWeights;

struct FeedForwardWeights {
    core::TensorValue norm;
    LinearWeights fc1;
    LinearWeights fc2;
};

struct AttentionWeights {
    core::TensorValue norm;
    LinearWeights q;
    LinearWeights k;
    LinearWeights v;
    LinearWeights gates;
    LinearWeights out;
};

struct TransformerLayerWeights {
    AttentionWeights attention;
    FeedForwardWeights feed_forward;
};

struct TransformerBranchWeights {
    std::vector<TransformerLayerWeights> layers;
    core::TensorValue norm;
};

struct AxialTransformerWeights {
    TransformerBranchWeights time_branch;
    TransformerBranchWeights freq_branch;
};

struct BandSplitWeights {
    core::TensorValue norm;
    LinearWeights proj;
};

struct MaskBandWeights {
    LinearWeights fc0;
    LinearWeights fc1;
    LinearWeights fc2;
};

struct MelBandWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    std::vector<BandSplitWeights> band_split;
    std::vector<AxialTransformerWeights> layers;
    std::vector<MaskBandWeights> mask_bands;
};

MelBandWeights load_mel_band_weights(
    const RoformerAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    assets_ns::TensorStorageType storage_type) {
    validate_roformer_weight_storage_type(storage_type);
    const auto & config = assets.config;
    if (config.num_stems != 1) {
        throw std::runtime_error("mel_band_roformer native runtime currently supports only single-stem checkpoints");
    }
    const auto & source = *assets.tensor_source;

    MelBandWeights weights;
    weights.store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "mel_band_roformer.weights",
        768ull * 1024ull * 1024ull);

    weights.band_split.reserve(static_cast<size_t>(config.band_input_dims.size()));
    for (size_t band = 0; band < config.band_input_dims.size(); ++band) {
        const std::string prefix = "band_split.to_features." + std::to_string(band);
        BandSplitWeights band_weights;
        band_weights.norm = weights.store->load_f32_tensor(source, prefix + ".0.gamma", {config.band_input_dims[band]});
        band_weights.proj = binding::linear_from_source(
            *weights.store,
            source,
            prefix + ".1",
            storage_type,
            config.dim,
            config.band_input_dims[band],
            true);
        weights.band_split.push_back(std::move(band_weights));
    }

    weights.layers.reserve(static_cast<size_t>(config.depth));
    for (int layer = 0; layer < config.depth; ++layer) {
        AxialTransformerWeights axial;
        for (int branch = 0; branch < 2; ++branch) {
            auto & branch_weights = branch == 0 ? axial.time_branch : axial.freq_branch;
            const int depth = branch == 0 ? config.time_transformer_depth : config.freq_transformer_depth;
            const int64_t branch_hidden_dim = config.dim * config.mlp_expansion_factor;
            branch_weights.layers.reserve(static_cast<size_t>(depth));
            for (int inner = 0; inner < depth; ++inner) {
                const std::string attn_prefix =
                    "layers." + std::to_string(layer) + "." + std::to_string(branch) + ".layers." + std::to_string(inner) + ".0";
                const std::string ff_prefix =
                    "layers." + std::to_string(layer) + "." + std::to_string(branch) + ".layers." + std::to_string(inner) + ".1.net";

                TransformerLayerWeights block;
                block.attention.norm = weights.store->load_f32_tensor(source, attn_prefix + ".norm.gamma", {config.dim});
                block.attention.q = binding::linear_from_source(*weights.store, source, attn_prefix + ".to_q", storage_type, config.heads * config.dim_head, config.dim, false);
                block.attention.k = binding::linear_from_source(*weights.store, source, attn_prefix + ".to_k", storage_type, config.heads * config.dim_head, config.dim, false);
                block.attention.v = binding::linear_from_source(*weights.store, source, attn_prefix + ".to_v", storage_type, config.heads * config.dim_head, config.dim, false);
                block.attention.gates = binding::linear_from_source(*weights.store, source, attn_prefix + ".to_gates", storage_type, config.heads, config.dim, true);
                block.attention.out = binding::linear_from_source(*weights.store, source, attn_prefix + ".to_out.0", storage_type, config.dim, config.heads * config.dim_head, false);

                block.feed_forward.norm = weights.store->load_f32_tensor(source, ff_prefix + ".0.gamma", {config.dim});
                block.feed_forward.fc1 = binding::linear_from_source(
                    *weights.store,
                    source,
                    ff_prefix + ".1",
                    storage_type,
                    branch_hidden_dim,
                    config.dim,
                    true);
                block.feed_forward.fc2 = binding::linear_from_source(
                    *weights.store,
                    source,
                    ff_prefix + ".4",
                    storage_type,
                    config.dim,
                    branch_hidden_dim,
                    true);
                branch_weights.layers.push_back(std::move(block));
            }
            branch_weights.norm = weights.store->load_f32_tensor(
                source,
                "layers." + std::to_string(layer) + "." + std::to_string(branch) + ".norm.gamma",
                {config.dim});
        }
        weights.layers.push_back(std::move(axial));
    }

    weights.mask_bands.reserve(static_cast<size_t>(config.band_input_dims.size()));
    const int hidden_dim = config.dim * config.mlp_expansion_factor;
    for (size_t band = 0; band < config.band_input_dims.size(); ++band) {
        const std::string prefix = "mask_estimators.0.to_freqs." + std::to_string(band) + ".0";
        MaskBandWeights band_weights;
        band_weights.fc0 = binding::linear_from_source(*weights.store, source, prefix + ".0", storage_type, hidden_dim, config.dim, true);
        band_weights.fc1 = binding::linear_from_source(*weights.store, source, prefix + ".2", storage_type, hidden_dim, hidden_dim, true);
        band_weights.fc2 = binding::linear_from_source(*weights.store, source, prefix + ".4", storage_type, config.band_input_dims[band] * 2, hidden_dim, true);
        weights.mask_bands.push_back(std::move(band_weights));
    }

    weights.store->upload();
    return weights;
}

core::TensorValue ensure_contiguous(core::ModuleBuildContext & ctx, const core::TensorValue & value) {
    return core::ensure_backend_addressable_layout(ctx, value);
}

core::TensorValue matmul_f32(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & lhs,
    const core::TensorValue & rhs) {
    core::validate_rank_between(lhs, 2, core::kMaxTensorRank, "lhs");
    core::validate_rank_between(rhs, static_cast<int64_t>(lhs.shape.rank), static_cast<int64_t>(lhs.shape.rank), "rhs");
    const size_t rank = lhs.shape.rank;
    for (size_t i = 0; i + 2 < rank; ++i) {
        if (lhs.shape.dims[i] != rhs.shape.dims[i]) {
            throw std::runtime_error("RoFormer matmul batch dimensions must match");
        }
    }
    if (lhs.shape.dims[rank - 1] != rhs.shape.dims[rank - 2]) {
        throw std::runtime_error("RoFormer matmul inner dimensions must match");
    }

    auto rhs_transposed = modules::TransposeModule({{0, 1, 3, 2}, rhs.shape.rank}).build(ctx, rhs);
    rhs_transposed = ensure_contiguous(ctx, rhs_transposed);
    core::TensorShape output_shape = lhs.shape;
    output_shape.dims[rank - 1] = rhs.shape.dims[rank - 1];
    ggml_tensor * output = ggml_mul_mat(ctx.ggml, rhs_transposed.tensor, lhs.tensor);
    ggml_mul_mat_set_prec(output, GGML_PREC_F32);
    return core::wrap_tensor(output, output_shape, GGML_TYPE_F32);
}

core::TensorValue build_reference_rms_norm(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t hidden_size,
    const core::TensorValue & weight) {
    return modules::RMSNormModule({hidden_size, 1e-12f, true, false}).build(
        ctx,
        input,
        binding::norm_data(ctx, weight));
}

core::TensorValue reshape_heads(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t heads,
    int64_t dim) {
    return core::reshape_tensor(
        ctx,
        ensure_contiguous(ctx, input),
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], heads, dim}));
}

core::TensorValue attention_from_heads(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & q_heads,
    const core::TensorValue & k_heads,
    const core::TensorValue & v_heads,
    int64_t dim) {
    auto scores = matmul_f32(
        ctx,
        q_heads,
        modules::TransposeModule({{0, 1, 3, 2}, k_heads.shape.rank}).build(ctx, k_heads));
    scores = core::wrap_tensor(
        ggml_scale(ctx.ggml, ensure_contiguous(ctx, scores).tensor, 1.0f / std::sqrt(static_cast<float>(dim))),
        scores.shape,
        GGML_TYPE_F32);
    scores = ensure_contiguous(ctx, scores);
    auto attn = modules::SoftmaxModule{}.build(ctx, scores);
    return matmul_f32(ctx, attn, v_heads);
}

core::TensorValue build_attention(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const AttentionWeights & weights,
    const RoformerArchitectureConfig & config) {
    const modules::LinearModule q_proj(binding::linear_config(config.dim, config.heads * config.dim_head, false));
    const modules::LinearModule k_proj(binding::linear_config(config.dim, config.heads * config.dim_head, false));
    const modules::LinearModule v_proj(binding::linear_config(config.dim, config.heads * config.dim_head, false));
    const modules::LinearModule gate_proj(binding::linear_config(config.dim, config.heads, true));
    const modules::LinearModule out_proj(binding::linear_config(config.heads * config.dim_head, config.dim, false));

    auto x = build_reference_rms_norm(ctx, input, config.dim, weights.norm);
    auto q = q_proj.build(ctx, x, binding::linear_data(ctx, weights.q.weight, weights.q.bias));
    auto k = k_proj.build(ctx, x, binding::linear_data(ctx, weights.k.weight, weights.k.bias));
    auto v = v_proj.build(ctx, x, binding::linear_data(ctx, weights.v.weight, weights.v.bias));
    auto gates = gate_proj.build(ctx, x, binding::linear_data(ctx, weights.gates.weight, weights.gates.bias));

    q = reshape_heads(ctx, q, config.heads, config.dim_head);
    k = reshape_heads(ctx, k, config.heads, config.dim_head);
    v = reshape_heads(ctx, v, config.heads, config.dim_head);
    q = modules::RoPEModule({config.dim_head, GGML_ROPE_TYPE_NORMAL, config.rope_theta}).build(ctx, q, positions);
    k = modules::RoPEModule({config.dim_head, GGML_ROPE_TYPE_NORMAL, config.rope_theta}).build(ctx, k, positions);

    auto q_heads = modules::TransposeModule({{0, 2, 1, 3}, q.shape.rank}).build(ctx, q);
    auto k_heads = modules::TransposeModule({{0, 2, 1, 3}, k.shape.rank}).build(ctx, k);
    auto v_heads = modules::TransposeModule({{0, 2, 1, 3}, v.shape.rank}).build(ctx, v);
    auto attn = attention_from_heads(ctx, q_heads, k_heads, v_heads, config.dim_head);
    attn = modules::TransposeModule({{0, 2, 1, 3}, attn.shape.rank}).build(ctx, attn);
    attn = ensure_contiguous(ctx, attn);

    auto gates_sigmoid = modules::SigmoidModule{}.build(ctx, gates);
    gates_sigmoid = core::reshape_tensor(
        ctx,
        ensure_contiguous(ctx, gates_sigmoid),
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], config.heads, 1}));
    gates_sigmoid = ensure_contiguous(ctx, gates_sigmoid);
    gates_sigmoid = modules::RepeatModule({attn.shape}).build(ctx, gates_sigmoid);
    attn = modules::MulModule{}.build(ctx, attn, gates_sigmoid);
    attn = core::reshape_tensor(
        ctx,
        ensure_contiguous(ctx, attn),
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], config.heads * config.dim_head}));
    return out_proj.build(ctx, attn, binding::linear_data(ctx, weights.out.weight, weights.out.bias));
}

core::TensorValue build_feed_forward(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const FeedForwardWeights & weights,
    const RoformerArchitectureConfig & config) {
    const int64_t hidden_dim = config.dim * config.mlp_expansion_factor;
    const modules::LinearModule fc1(binding::linear_config(config.dim, hidden_dim, true));
    const modules::LinearModule fc2(binding::linear_config(hidden_dim, config.dim, true));

    auto x = build_reference_rms_norm(ctx, input, config.dim, weights.norm);
    x = fc1.build(ctx, x, binding::linear_data(ctx, weights.fc1.weight, weights.fc1.bias));
    x = modules::GeluModule({modules::GeluApproximation::ExactErf}).build(ctx, x);
    x = fc2.build(ctx, x, binding::linear_data(ctx, weights.fc2.weight, weights.fc2.bias));
    return x;
}

core::TensorValue build_transformer_branch(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const TransformerBranchWeights & weights,
    const RoformerArchitectureConfig & config) {
    auto x = input;
    for (const auto & layer : weights.layers) {
        x = modules::AddModule{}.build(ctx, x, build_attention(ctx, x, positions, layer.attention, config));
        x = modules::AddModule{}.build(ctx, x, build_feed_forward(ctx, x, layer.feed_forward, config));
    }
    return build_reference_rms_norm(ctx, x, config.dim, weights.norm);
}

core::TensorValue build_band_split(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const MelBandWeights & weights,
    const RoformerArchitectureConfig & config) {
    std::vector<core::TensorValue> outputs;
    outputs.reserve(weights.band_split.size());
    int64_t offset = 0;
    for (size_t band = 0; band < weights.band_split.size(); ++band) {
        const int64_t dim_in = config.band_input_dims[band];
        auto band_input = modules::SliceModule({2, offset, dim_in}).build(ctx, input);
        band_input = build_reference_rms_norm(ctx, band_input, dim_in, weights.band_split[band].norm);
        band_input = modules::LinearModule(binding::linear_config(dim_in, config.dim, true))
                         .build(ctx, band_input, binding::linear_data(ctx, weights.band_split[band].proj.weight, weights.band_split[band].proj.bias));
        band_input = core::reshape_tensor(
            ctx,
            ensure_contiguous(ctx, band_input),
            core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], 1, config.dim}));
        outputs.push_back(band_input);
        offset += dim_in;
    }

    while (outputs.size() > 1) {
        std::vector<core::TensorValue> next;
        next.reserve((outputs.size() + 1) / 2);
        for (size_t i = 0; i < outputs.size(); i += 2) {
            if (i + 1 < outputs.size()) {
                next.push_back(modules::ConcatModule({2}).build(ctx, outputs[i], outputs[i + 1]));
            } else {
                next.push_back(outputs[i]);
            }
        }
        outputs = std::move(next);
    }
    return outputs.front();
}

core::TensorValue build_mask_output(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const MelBandWeights & weights,
    const RoformerArchitectureConfig & config) {
    std::vector<core::TensorValue> outputs;
    outputs.reserve(weights.mask_bands.size());
    for (size_t band = 0; band < weights.mask_bands.size(); ++band) {
        auto band_input = modules::SliceModule({2, static_cast<int64_t>(band), 1}).build(ctx, input);
        band_input = core::reshape_tensor(
            ctx,
            ensure_contiguous(ctx, band_input),
            core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], config.dim}));
        const auto & band_weights = weights.mask_bands[band];
        auto hidden = modules::LinearModule(binding::linear_config(config.dim, config.dim * config.mlp_expansion_factor, true))
                          .build(ctx, band_input, binding::linear_data(ctx, band_weights.fc0.weight, band_weights.fc0.bias));
        hidden = modules::TanhModule{}.build(ctx, hidden);
        hidden = modules::LinearModule(binding::linear_config(
                                           config.dim * config.mlp_expansion_factor,
                                           config.dim * config.mlp_expansion_factor,
                                           true))
                     .build(ctx, hidden, binding::linear_data(ctx, band_weights.fc1.weight, band_weights.fc1.bias));
        hidden = modules::TanhModule{}.build(ctx, hidden);
        hidden = modules::LinearModule(binding::linear_config(
                                           config.dim * config.mlp_expansion_factor,
                                           config.band_input_dims[band] * 2,
                                           true))
                     .build(ctx, hidden, binding::linear_data(ctx, band_weights.fc2.weight, band_weights.fc2.bias));
        hidden = modules::GLUModule{}.build(ctx, hidden);
        outputs.push_back(hidden);
    }

    while (outputs.size() > 1) {
        std::vector<core::TensorValue> next;
        next.reserve((outputs.size() + 1) / 2);
        for (size_t i = 0; i < outputs.size(); i += 2) {
            if (i + 1 < outputs.size()) {
                next.push_back(modules::ConcatModule({2}).build(ctx, outputs[i], outputs[i + 1]));
            } else {
                next.push_back(outputs[i]);
            }
        }
        outputs = std::move(next);
    }
    return outputs.front();
}

int64_t reflect_index(int64_t index, int64_t length) {
    while (index < 0 || index >= length) {
        if (index < 0) {
            index = -index;
        } else {
            index = 2 * length - index - 2;
        }
    }
    return index;
}

std::vector<int32_t> make_positions(int64_t steps) {
    std::vector<int32_t> positions(static_cast<size_t>(steps), 0);
    for (int64_t i = 0; i < steps; ++i) {
        positions[static_cast<size_t>(i)] = static_cast<int32_t>(i);
    }
    return positions;
}

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

class FixedShapeGraph {
public:
    FixedShapeGraph(ggml_backend_t backend, int compute_threads)
        : backend_(backend),
          compute_threads_(compute_threads) {
    }

    virtual ~FixedShapeGraph() {
        if (graph_ != nullptr) {
            engine::core::release_backend_graph_resources(backend_, graph_);
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
    }

    const std::vector<float> & run(const std::vector<float> & input_values) {
        if (static_cast<int64_t>(input_values.size()) != input_shape_.num_elements()) {
            throw std::runtime_error("roformer graph input size mismatch");
        }
        const auto upload_start = Clock::now();
        ggml_backend_tensor_set(input_, input_values.data(), 0, input_values.size() * sizeof(float));
        engine::debug::timing_log_scalar("mel_band_roformer.graph.input_upload_ms", engine::debug::elapsed_ms(upload_start));
        core::set_backend_threads(backend_, compute_threads_);
        const auto compute_start = Clock::now();
        const ggml_status status = engine::core::compute_backend_graph(backend_, graph_);
        ggml_backend_synchronize(backend_);
        engine::debug::timing_log_scalar("mel_band_roformer.graph.compute_ms", engine::debug::elapsed_ms(compute_start));
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("roformer graph compute failed");
        }
        const auto read_start = Clock::now();
        output_host_.resize(static_cast<size_t>(output_shape_.num_elements()));
        ggml_backend_tensor_get(output_, output_host_.data(), 0, output_host_.size() * sizeof(float));
        engine::debug::timing_log_scalar("mel_band_roformer.graph.output_read_ms", engine::debug::elapsed_ms(read_start));
        return output_host_;
    }

protected:
    void init_context(size_t arena_bytes) {
        ggml_init_params params{arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize roformer ggml context");
        }
    }

    core::ModuleBuildContext make_build_context(core::ExecutionContext & execution_context, const char * name) {
        return core::ModuleBuildContext{
            ctx_.get(),
            name,
            execution_context.backend_type(),
        };
    }

    void finalize_graph(size_t max_nodes) {
        graph_ = ggml_new_graph_custom(ctx_.get(), max_nodes, false);
        ggml_build_forward_expand(graph_, output_);
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        if (gallocr_ == nullptr || !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("failed to allocate roformer graph buffer");
        }
    }

    ggml_backend_t backend_ = nullptr;
    int compute_threads_ = 1;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    core::TensorShape input_shape_;
    core::TensorShape output_shape_;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    std::vector<float> output_host_;
};

class MelBandGraph final : public FixedShapeGraph {
public:
    MelBandGraph(
        std::shared_ptr<const RoformerAssets> assets,
        core::ExecutionContext & execution_context,
        assets_ns::TensorStorageType weight_storage_type)
        : FixedShapeGraph(execution_context.backend(), std::max(1, execution_context.config().threads)) {
        if (assets == nullptr) {
            throw std::runtime_error("mel_band_roformer graph requires assets");
        }
        const auto build_start = Clock::now();
        const auto & config = assets->config;
        weights_ = load_mel_band_weights(*assets, execution_context.backend(), execution_context.backend_type(), weight_storage_type);

        init_context(256ull * 1024ull * 1024ull);
        constants_ = std::make_unique<core::ConstantTensorCache>(
            backend_,
            compute_threads_,
            "mel_band_roformer.constants",
            4ull * 1024ull * 1024ull);
        auto build_ctx = make_build_context(execution_context, "mel_band_roformer");
        constants_->begin_graph();
        input_shape_ = core::TensorShape::from_dims({1, config.chunk_frames, config.total_band_input_dim});
        output_shape_ = core::TensorShape::from_dims({1, config.chunk_frames, config.total_band_input_dim});

        auto input = core::make_tensor(build_ctx, GGML_TYPE_F32, input_shape_);
        input_ = input.tensor;

        const auto time_positions_host = make_positions(config.chunk_frames);
        const auto time_positions = constants_->make_tensor(
            core::TensorShape::from_dims({config.chunk_frames}),
            GGML_TYPE_I32,
            time_positions_host.data(),
            time_positions_host.size() * sizeof(int32_t));

        const auto freq_positions_host = make_positions(config.num_bands);
        const auto freq_positions = constants_->make_tensor(
            core::TensorShape::from_dims({config.num_bands}),
            GGML_TYPE_I32,
            freq_positions_host.data(),
            freq_positions_host.size() * sizeof(int32_t));

        core::TensorValue x = build_band_split(build_ctx, input, weights_, config);
        for (const auto & layer : weights_.layers) {
            x = modules::TransposeModule({{0, 2, 1, 3}, x.shape.rank}).build(build_ctx, x);
            x = core::reshape_tensor(
                build_ctx,
                ensure_contiguous(build_ctx, x),
                core::TensorShape::from_dims({config.num_bands, config.chunk_frames, config.dim}));
            x = build_transformer_branch(build_ctx, x, time_positions, layer.time_branch, config);
            x = core::reshape_tensor(
                build_ctx,
                ensure_contiguous(build_ctx, x),
                core::TensorShape::from_dims({1, config.num_bands, config.chunk_frames, config.dim}));
            x = modules::TransposeModule({{0, 2, 1, 3}, x.shape.rank}).build(build_ctx, x);
            x = ensure_contiguous(build_ctx, x);

            x = core::reshape_tensor(
                build_ctx,
                x,
                core::TensorShape::from_dims({config.chunk_frames, config.num_bands, config.dim}));
            x = build_transformer_branch(build_ctx, x, freq_positions, layer.freq_branch, config);
            x = core::reshape_tensor(
                build_ctx,
                ensure_contiguous(build_ctx, x),
                core::TensorShape::from_dims({1, config.chunk_frames, config.num_bands, config.dim}));
            x = ensure_contiguous(build_ctx, x);
        }

        auto output = build_mask_output(build_ctx, x, weights_, config);
        output = ensure_contiguous(build_ctx, output);
        output_ = output.tensor;
        ggml_set_output(output_);
        constants_->finish_graph();
        constants_->ensure_uploaded();
        finalize_graph(131072);
        engine::debug::timing_log_scalar("mel_band_roformer.graph.build_ms", engine::debug::elapsed_ms(build_start));
        engine::debug::timing_log_scalar("mel_band_roformer.graph.rebuilt", true);
    }

private:
    MelBandWeights weights_;
    std::unique_ptr<core::ConstantTensorCache> constants_;
};

std::vector<float> build_band_features(
    const engine::audio::AudioTensor & stft,
    const RoformerArchitectureConfig & config) {
    if (stft.shape.size() != 4) {
        throw std::runtime_error("RoFormer STFT tensor must be rank-4");
    }
    const int64_t channels = stft.shape[0];
    const int64_t freq_bins = stft.shape[1];
    const int64_t frames = stft.shape[2];
    if (channels != config.channels || freq_bins != config.stft_freq_bins || frames != config.chunk_frames || stft.shape[3] != 2) {
        throw std::runtime_error("RoFormer STFT tensor shape mismatch");
    }
    std::vector<float> features(static_cast<size_t>(frames * config.total_band_input_dim), 0.0f);
    const int64_t merged = static_cast<int64_t>(config.merged_freq_indices.size());
#ifdef _OPENMP
    #pragma omp parallel for collapse(2) if(frames * merged >= 4096)
#endif
    for (int64_t t = 0; t < frames; ++t) {
        for (int64_t m = 0; m < merged; ++m) {
            const int64_t merged_index = config.merged_freq_indices[static_cast<size_t>(m)];
            const int64_t channel = merged_index % config.channels;
            const int64_t freq = merged_index / config.channels;
            const size_t stft_base = static_cast<size_t>((((channel * freq_bins) + freq) * frames + t) * 2);
            const size_t dst = static_cast<size_t>(t * config.total_band_input_dim + m * 2);
            features[dst] = stft.values[stft_base];
            features[dst + 1] = stft.values[stft_base + 1];
        }
    }
    return features;
}

std::vector<float> apply_masks_to_stft(
    const std::vector<float> & raw_masks,
    const engine::audio::AudioTensor & stft,
    const RoformerArchitectureConfig & config) {
    const int64_t channels = config.channels;
    const int64_t freq_bins = config.stft_freq_bins;
    const int64_t frames = config.chunk_frames;
    std::vector<float> averaged_masks(static_cast<size_t>(channels * freq_bins * frames * 2), 0.0f);
    const int64_t merged = static_cast<int64_t>(config.merged_freq_indices.size());

#ifdef _OPENMP
    #pragma omp parallel for if(merged * frames >= 4096)
#endif
    for (int64_t m = 0; m < merged; ++m) {
        const int64_t merged_index = config.merged_freq_indices[static_cast<size_t>(m)];
        for (int64_t t = 0; t < frames; ++t) {
            const size_t src = static_cast<size_t>(t * config.total_band_input_dim + m * 2);
            const size_t dst = static_cast<size_t>(((merged_index * frames) + t) * 2);
            averaged_masks[dst] += raw_masks[src];
            averaged_masks[dst + 1] += raw_masks[src + 1];
        }
    }

#ifdef _OPENMP
    #pragma omp parallel for if(channels * freq_bins >= 512)
#endif
    for (int64_t merged_index = 0; merged_index < channels * freq_bins; ++merged_index) {
        const float denom = static_cast<float>(std::max<int64_t>(1, config.merged_band_counts[static_cast<size_t>(merged_index)]));
        for (int64_t t = 0; t < frames; ++t) {
            const size_t base = static_cast<size_t>(((merged_index * frames) + t) * 2);
            averaged_masks[base] /= denom;
            averaged_masks[base + 1] /= denom;
        }
    }

    std::vector<float> modulated = stft.values;
#ifdef _OPENMP
    #pragma omp parallel for collapse(2) if(channels * freq_bins >= 512)
#endif
    for (int ch = 0; ch < channels; ++ch) {
        for (int64_t freq = 0; freq < freq_bins; ++freq) {
            const int64_t merged_index = freq * channels + ch;
            for (int64_t t = 0; t < frames; ++t) {
                const size_t base = static_cast<size_t>((((ch * freq_bins) + freq) * frames + t) * 2);
                const size_t mask_base = static_cast<size_t>(((merged_index * frames) + t) * 2);
                const float a = stft.values[base];
                const float b = stft.values[base + 1];
                const float c = averaged_masks[mask_base];
                const float d = averaged_masks[mask_base + 1];
                modulated[base] = a * c - b * d;
                modulated[base + 1] = a * d + b * c;
            }
        }
    }
    return modulated;
}

engine::audio::AudioTensor compute_roformer_stft(
    const std::vector<float> & waveform,
    const std::vector<float> & window,
    int64_t channels,
    int64_t samples,
    const engine::audio::STFTConfig & config,
    size_t threads) {
    if (static_cast<int64_t>(waveform.size()) != channels * samples ||
        static_cast<int64_t>(window.size()) != config.win_length) {
        throw std::runtime_error("RoFormer local STFT input size mismatch");
    }
    const int64_t pad = config.center ? config.n_fft / 2 : 0;
    const int64_t frames = 1 + (samples + 2 * pad - config.n_fft) / config.hop_length;
    const int64_t freq_bins = (config.n_fft / 2) + 1;
    const int64_t window_offset = (config.n_fft - config.win_length) / 2;

    std::vector<float> framed(static_cast<size_t>(channels * config.n_fft * frames), 0.0f);
#ifdef _OPENMP
    #pragma omp parallel for if(channels * frames >= 8)
#endif
    for (int64_t ch = 0; ch < channels; ++ch) {
        const float * signal = waveform.data() + static_cast<size_t>(ch * samples);
        for (int64_t frame_index = 0; frame_index < frames; ++frame_index) {
            const int64_t start = frame_index * config.hop_length - pad;
            for (int64_t i = 0; i < config.win_length; ++i) {
                const int64_t sample_index = start + window_offset + i;
                float sample = 0.0f;
                if (sample_index >= 0 && sample_index < samples) {
                    sample = signal[sample_index];
                } else if (config.pad_mode == engine::audio::STFTPadMode::Reflect) {
                    sample = signal[reflect_index(sample_index, samples)];
                }
                framed[static_cast<size_t>(((ch * config.n_fft) + (window_offset + i)) * frames + frame_index)] =
                    sample * window[static_cast<size_t>(i)];
            }
        }
    }

    engine::audio::TensorShape input_shape{
        static_cast<size_t>(channels),
        static_cast<size_t>(config.n_fft),
        static_cast<size_t>(frames),
    };
    engine::audio::TensorStrideBytes input_strides{
        static_cast<std::ptrdiff_t>(config.n_fft * frames * static_cast<int64_t>(sizeof(float))),
        static_cast<std::ptrdiff_t>(frames * static_cast<int64_t>(sizeof(float))),
        static_cast<std::ptrdiff_t>(sizeof(float)),
    };
    engine::audio::TensorStrideBytes output_strides{
        static_cast<std::ptrdiff_t>(freq_bins * frames * static_cast<int64_t>(sizeof(std::complex<float>))),
        static_cast<std::ptrdiff_t>(frames * static_cast<int64_t>(sizeof(std::complex<float>))),
        static_cast<std::ptrdiff_t>(sizeof(std::complex<float>)),
    };

    std::vector<std::complex<float>> spectrum(
        static_cast<size_t>(channels * freq_bins * frames),
        std::complex<float>(0.0f, 0.0f));
    engine::audio::real_fft_forward(
        input_shape,
        input_strides,
        output_strides,
        1,
        framed.data(),
        spectrum.data(),
        1.0f,
        threads);

    engine::audio::AudioTensor result;
    result.shape = {channels, freq_bins, frames, 2};
    result.values.assign(static_cast<size_t>(channels * freq_bins * frames * 2), 0.0f);
#ifdef _OPENMP
    #pragma omp parallel for collapse(2) if(channels * freq_bins >= 512)
#endif
    for (int64_t ch = 0; ch < channels; ++ch) {
        for (int64_t freq = 0; freq < freq_bins; ++freq) {
            for (int64_t t = 0; t < frames; ++t) {
                const auto value = spectrum[static_cast<size_t>(((ch * freq_bins) + freq) * frames + t)];
                const size_t base = static_cast<size_t>((((ch * freq_bins) + freq) * frames + t) * 2);
                result.values[base] = value.real();
                result.values[base + 1] = value.imag();
            }
        }
    }
    return result;
}

engine::audio::AudioTensor compute_roformer_istft(
    const std::vector<float> & complex_spec,
    const std::vector<float> & window,
    int64_t channels,
    int64_t freq_bins,
    int64_t frames,
    int64_t samples,
    const engine::audio::STFTConfig & config,
    size_t threads) {
    if (static_cast<int64_t>(complex_spec.size()) != channels * freq_bins * frames * 2 ||
        static_cast<int64_t>(window.size()) != config.win_length) {
        throw std::runtime_error("RoFormer local ISTFT input size mismatch");
    }

    std::vector<std::complex<float>> spectrum(
        static_cast<size_t>(channels * freq_bins * frames),
        std::complex<float>(0.0f, 0.0f));
#ifdef _OPENMP
    #pragma omp parallel for collapse(2) if(channels * freq_bins >= 512)
#endif
    for (int64_t ch = 0; ch < channels; ++ch) {
        for (int64_t freq = 0; freq < freq_bins; ++freq) {
            for (int64_t frame_index = 0; frame_index < frames; ++frame_index) {
                const size_t src = static_cast<size_t>((((ch * freq_bins) + freq) * frames + frame_index) * 2);
                spectrum[static_cast<size_t>(((ch * freq_bins) + freq) * frames + frame_index)] = {
                    complex_spec[src],
                    complex_spec[src + 1],
                };
            }
        }
    }

    engine::audio::TensorShape output_shape{
        static_cast<size_t>(channels),
        static_cast<size_t>(config.n_fft),
        static_cast<size_t>(frames),
    };
    engine::audio::TensorStrideBytes input_strides{
        static_cast<std::ptrdiff_t>(freq_bins * frames * static_cast<int64_t>(sizeof(std::complex<float>))),
        static_cast<std::ptrdiff_t>(frames * static_cast<int64_t>(sizeof(std::complex<float>))),
        static_cast<std::ptrdiff_t>(sizeof(std::complex<float>)),
    };
    engine::audio::TensorStrideBytes output_strides{
        static_cast<std::ptrdiff_t>(config.n_fft * frames * static_cast<int64_t>(sizeof(float))),
        static_cast<std::ptrdiff_t>(frames * static_cast<int64_t>(sizeof(float))),
        static_cast<std::ptrdiff_t>(sizeof(float)),
    };

    std::vector<float> framed(static_cast<size_t>(channels * config.n_fft * frames), 0.0f);
    engine::audio::real_fft_inverse(
        output_shape,
        input_strides,
        output_strides,
        1,
        spectrum.data(),
        framed.data(),
        1.0f / static_cast<float>(config.n_fft),
        threads);

    const int64_t pad = config.center ? config.n_fft / 2 : 0;
    const int64_t padded_samples = samples + 2 * pad;
    const int64_t usable_window = std::min<int64_t>(config.win_length, config.n_fft);
    const int64_t window_offset = (config.n_fft - config.win_length) / 2;
    std::vector<float> window_sq(static_cast<size_t>(usable_window), 0.0f);
#ifdef _OPENMP
    #pragma omp parallel for if(usable_window >= 256)
#endif
    for (int64_t i = 0; i < usable_window; ++i) {
        const float w = window[static_cast<size_t>(i)];
        window_sq[static_cast<size_t>(i)] = w * w;
    }

    std::vector<float> accum(static_cast<size_t>(channels * padded_samples), 0.0f);
    std::vector<float> window_sums(static_cast<size_t>(channels * padded_samples), 0.0f);
#ifdef _OPENMP
    #pragma omp parallel for if(channels >= 2)
#endif
    for (int64_t ch = 0; ch < channels; ++ch) {
        for (int64_t frame_index = 0; frame_index < frames; ++frame_index) {
            const int64_t start = frame_index * config.hop_length;
            const float * frame = framed.data() + static_cast<size_t>((ch * config.n_fft) * frames + frame_index);
            float * accum_row = accum.data() + static_cast<size_t>(ch * padded_samples + start);
            float * window_row = window_sums.data() + static_cast<size_t>(ch * padded_samples + start);
            for (int64_t n = 0; n < usable_window; ++n) {
                const float sample = frame[static_cast<size_t>((window_offset + n) * frames)];
                accum_row[n] += sample * window[static_cast<size_t>(n)];
                window_row[n] += window_sq[static_cast<size_t>(n)];
            }
        }
    }

    engine::audio::AudioTensor result;
    result.shape = {channels, samples};
    result.values.assign(static_cast<size_t>(channels * samples), 0.0f);
#ifdef _OPENMP
    #pragma omp parallel for if(channels >= 2)
#endif
    for (int64_t ch = 0; ch < channels; ++ch) {
        for (int64_t i = 0; i < samples; ++i) {
            const size_t padded_index = static_cast<size_t>(ch * padded_samples + i + pad);
            const float denom = window_sums[padded_index] > 1.0e-8f ? window_sums[padded_index] : 1.0f;
            result.values[static_cast<size_t>(ch * samples + i)] = accum[padded_index] / denom;
        }
    }
    return result;
}

void separate_runtime_chunk(
    MelBandGraph & graph,
    const std::vector<float> & chunk_planar,
    const RoformerArchitectureConfig & config,
    size_t fft_threads,
    std::vector<float> & output_planar) {
    const engine::audio::STFTConfig stft_config{
        config.n_fft,
        config.hop_length,
        config.win_length,
        true,
        engine::audio::STFTPadMode::Reflect,
        engine::audio::STFTFamily::Kokoro,
    };
    const auto & window = engine::audio::get_cached_stft_window(stft_config);
    const auto stft_start = Clock::now();
    auto stft = compute_roformer_stft(
        chunk_planar,
        window,
        config.channels,
        config.chunk_size,
        stft_config,
        fft_threads);
    engine::debug::timing_log_scalar("mel_band_roformer.stft_ms", engine::debug::elapsed_ms(stft_start));
    const auto feature_start = Clock::now();
    auto features = build_band_features(stft, config);
    engine::debug::timing_log_scalar("mel_band_roformer.feature_build_ms", engine::debug::elapsed_ms(feature_start));
    const auto graph_start = Clock::now();
    const auto & raw_masks = graph.run(features);
    engine::debug::timing_log_scalar("mel_band_roformer.graph.total_ms", engine::debug::elapsed_ms(graph_start));
    const auto mask_start = Clock::now();
    auto masked = apply_masks_to_stft(raw_masks, stft, config);
    engine::debug::timing_log_scalar("mel_band_roformer.mask_apply_ms", engine::debug::elapsed_ms(mask_start));
    const auto istft_start = Clock::now();
    auto vocals = compute_roformer_istft(
        masked,
        window,
        config.channels,
        config.stft_freq_bins,
        config.chunk_frames,
        config.chunk_size,
        stft_config,
        fft_threads);
    engine::debug::timing_log_scalar("mel_band_roformer.istft_ms", engine::debug::elapsed_ms(istft_start));
    if (vocals.shape.size() != 2 || vocals.shape[0] != config.channels || vocals.shape[1] != config.chunk_size) {
        throw std::runtime_error("RoFormer ISTFT returned an unexpected waveform shape");
    }
    output_planar = std::move(vocals.values);
}

}  // namespace
class RoformerRuntime::Impl {
public:
    std::unique_ptr<MelBandGraph> mel_graph;
    std::vector<float> chunk_output_planar;
};

RoformerRuntime::RoformerRuntime(
    std::shared_ptr<const RoformerAssets> assets,
    core::ExecutionContext & execution_context,
    assets::TensorStorageType weight_storage_type)
    : assets_(std::move(assets)),
      impl_(std::make_unique<Impl>()) {
    if (assets_ == nullptr) {
        throw std::runtime_error("RoFormer runtime requires assets");
    }
    validate_roformer_weight_storage_type(weight_storage_type);
    fft_threads_ = std::max<size_t>(1, static_cast<size_t>(execution_context.config().threads));
    impl_->mel_graph = std::make_unique<MelBandGraph>(assets_, execution_context, weight_storage_type);
}

RoformerRuntime::~RoformerRuntime() = default;

const RoformerArchitectureConfig & RoformerRuntime::config() const noexcept {
    return assets_->config;
}

const std::vector<float> & RoformerRuntime::separate_chunk(const std::vector<float> & chunk_planar) {
    if (impl_->mel_graph == nullptr) {
        throw std::runtime_error("mel_band_roformer graph is not initialized");
    }
    separate_runtime_chunk(
        *impl_->mel_graph,
        chunk_planar,
        assets_->config,
        fft_threads_,
        impl_->chunk_output_planar);
    return impl_->chunk_output_planar;
}

}  // namespace engine::models::roformer
