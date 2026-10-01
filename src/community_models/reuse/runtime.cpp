#include "engine/community_models/reuse/runtime.h"
#include "engine/community_models/reuse/frontend.h"

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/io/json.h"
#include "engine/framework/model_spec/package.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/runtime/graph_optimizer.h"

#include <ggml-alloc.h>
#include <algorithm>
#include <cmath>
#include <future>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace engine::models::reuse {
namespace {

using core::TensorShape;
using core::TensorValue;
using TensorMap = std::unordered_map<std::string, TensorValue>;

class ReuseGraph {
public:
    ReuseGraph(core::ExecutionContext & execution, const TensorMap & weights,
               int64_t batch, int64_t frames, int64_t bins)
        : backend_(execution.backend()), weights_(weights) {
        const auto started = std::chrono::steady_clock::now();
        constexpr size_t nodes = 32768;
        context_.reset(ggml_init({nodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(nodes, false), nullptr, true}));
        if (!context_) {
            throw std::runtime_error("RE-USE graph context allocation failed");
        }
        core::ModuleBuildContext ctx{context_.get(), "reuse", execution.backend_type()};
        ctx_ = &ctx;
        input_ = core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({batch, 2, frames, bins})).tensor;
        ggml_set_input(input_);
        auto * x = modules::Pad2dModule({0, 2, 0, 2}).build(ctx,
            core::wrap_tensor(input_, TensorShape::from_dims({batch, 2, frames, bins}))).tensor;
        x = conv_norm(x, "dense_encoder.dense_conv_1");
        x = dense(x, "dense_encoder.dense_block");
        x = conv_norm(x, "dense_encoder.dense_conv_2", 4, 2);
        const int64_t time = x->ne[1];
        const int64_t frequency = x->ne[0];
        const int64_t time_lanes = frequency * batch;
        const int64_t frequency_lanes = time * batch;
        auto * time_reverse = reverse_indices(time, time_lanes);
        auto * freq_reverse = reverse_indices(frequency, frequency_lanes);
        auto * time_ids = sequence_ids(time_lanes);
        auto * freq_ids = sequence_ids(frequency_lanes);
        auto * time_state = ggml_fill(ctx.ggml,
            core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({time_lanes, 256, 1, 16})).tensor, 0);
        auto * freq_state = ggml_fill(ctx.ggml,
            core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({frequency_lanes, 256, 1, 16})).tensor, 0);
        auto encoded = core::ensure_backend_addressable_layout(ctx,
            modules::TransposeModule({{0, 3, 2, 1}, 4}).build(ctx,
                core::wrap_tensor(x, TensorShape::from_dims({batch, 64, time, frequency}))));
        for (int i = 0; i < 30; ++i) {
            const auto prefix = "TSMamba." + std::to_string(i);
            const auto time_shape = TensorShape::from_dims({time_lanes, time, 64});
            encoded = core::reshape_tensor(ctx, encoded, time_shape);
            encoded = modules::AddModule().build(ctx,
                encoded,
                core::wrap_tensor(
                    bidirectional(encoded.tensor, prefix + ".time_mamba", time_state, time_ids, time_reverse),
                    time_shape));
            encoded = core::reshape_tensor(ctx, encoded, TensorShape::from_dims({batch, frequency, time, 64}));
            encoded = core::ensure_backend_addressable_layout(ctx,
                modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, encoded));
            const auto freq_shape = TensorShape::from_dims({frequency_lanes, frequency, 64});
            encoded = core::reshape_tensor(ctx, encoded, freq_shape);
            encoded = modules::AddModule().build(ctx,
                encoded,
                core::wrap_tensor(
                    bidirectional(encoded.tensor, prefix + ".freq_mamba", freq_state, freq_ids, freq_reverse),
                    freq_shape));
            encoded = core::reshape_tensor(ctx, encoded, TensorShape::from_dims({batch, time, frequency, 64}));
            if (i != 29) {
                encoded = core::ensure_backend_addressable_layout(ctx,
                    modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, encoded));
            }
        }
        x = core::ensure_backend_addressable_layout(ctx,
            modules::TransposeModule({{0, 3, 1, 2}, 4}).build(ctx, encoded)).tensor;
        auto * magnitude = decode(x, "mask_decoder");
        magnitude = conv(magnitude, "mask_decoder.final_conv");
        auto * phase = decode(x, "phase_decoder");
        auto * real = conv(phase, "phase_decoder.phase_conv_r");
        auto * imaginary = conv(phase, "phase_decoder.phase_conv_i");
        outputs_ = {magnitude, real, imaginary};
        graph_ = ggml_new_graph_custom(ctx.ggml, nodes, false);
        for (auto * output : outputs_) {
            ggml_set_output(output);
            ggml_build_forward_expand(graph_, output);
        }
        auto optimization = runtime::graph_optimization_options_for_backend(execution.backend_type() == core::BackendType::Cpu
            ? runtime::GraphOptimizationBackend::Cpu : runtime::GraphOptimizationBackend::Gpu);
        // The allocator must visit view/reshape nodes, even when CPU execution can skip them.
        optimization.elide_noop_nodes = false;
        optimization.elide_metadata_only_ops = false;
        runtime::optimize_graph(*graph_, optimization);
        for (int i = 0; i < ggml_graph_n_nodes(graph_); ++i) {
            auto * node = ggml_graph_node(graph_, i);
            if (node->op == GGML_OP_MUL_MAT) {
                ggml_mul_mat_set_prec(node, GGML_PREC_F32);
            } else if (execution.backend_type() == core::BackendType::Cuda &&
                       node->op == GGML_OP_IM2COL && node->type == GGML_TYPE_F32 &&
                       node->src[0]->ne[0] == 3 && node->src[0]->ne[1] == 3 &&
                       node->src[0]->ne[2] >= 32 && node->ne[1] >= 32) {
                ggml_im2col_2d_set_lowering(node, GGML_IM2COL_2D_LOWERING_CUDA_F32_K3_TILED);
            }
        }
        core::validate_backend_graph_supported(backend_, graph_, "RE-USE");
        allocator_.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_)));
        if (!allocator_ || !ggml_gallocr_alloc_graph(allocator_.get(), graph_)) {
            throw std::runtime_error("RE-USE graph allocation failed");
        }
        for (const auto & item : indices_) {
            ggml_backend_tensor_set(item.first, item.second.data(), 0, item.second.size() * sizeof(int32_t));
        }
        indices_.clear();
        ctx_ = nullptr;
        debug::timing_log_scalar("reuse.graph.build_ms", debug::elapsed_ms(started));
        debug::timing_log_scalar("reuse.graph.workspace_bytes", ggml_gallocr_get_buffer_size(allocator_.get(), 0));
    }

    ~ReuseGraph() {
        core::release_backend_graph_resources(backend_, graph_, true);
    }

    std::vector<std::vector<float>> run(const std::vector<float> & features) {
        if (features.size() * sizeof(float) != ggml_nbytes(input_)) {
            throw std::runtime_error("RE-USE feature shape mismatch");
        }
        const auto started = std::chrono::steady_clock::now();
        ggml_backend_tensor_set(input_, features.data(), 0, features.size() * sizeof(float));
        if (core::compute_backend_graph(backend_, graph_) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("RE-USE graph execution failed");
        }
        std::vector<std::vector<float>> result;
        for (auto * output : outputs_) {
            result.emplace_back(ggml_nelements(output));
            ggml_backend_tensor_get(output, result.back().data(), 0, ggml_nbytes(output));
        }
        debug::timing_log_scalar("reuse.network.ms", debug::elapsed_ms(started));
        return result;
    }

private:
    ggml_tensor * sequence_ids(int64_t count) {
        auto * ids = core::make_tensor(*ctx_, GGML_TYPE_I32, TensorShape::from_dims({count})).tensor;
        ggml_set_input(ids);
        ggml_set_output(ids);
        std::vector<int32_t> values(count);
        std::iota(values.begin(), values.end(), 0);
        indices_.emplace_back(ids, std::move(values));
        return ids;
    }

    ggml_tensor * reverse_indices(int64_t length, int64_t batch) {
        auto * ids = core::make_tensor(*ctx_, GGML_TYPE_I32, TensorShape::from_dims({batch, length})).tensor;
        ggml_set_input(ids);
        ggml_set_output(ids);
        std::vector<int32_t> values(length * batch);
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t i = 0; i < length; ++i) {
                values[b * length + i] = static_cast<int32_t>(length - 1 - i);
            }
        }
        indices_.emplace_back(ids, std::move(values));
        return ids;
    }

    ggml_tensor * conv(ggml_tensor * input, const std::string & prefix,
                       int stride_time = 1, int stride_frequency = 1, int dilation = 1, bool same = false) {
        const auto & weight = weights_.at(prefix + ".weight");
        const auto & shape = weight.shape;
        auto & ctx = *ctx_;
        const int pad_time = same ? dilation * static_cast<int>(shape.dims[2] - 1) / 2 : 0;
        const int pad_frequency = same ? static_cast<int>(shape.dims[3] - 1) / 2 : 0;
        const int64_t receptive = dilation * (shape.dims[2] - 1) + 1;
        const int64_t height = (input->ne[1] + 2 * pad_time - receptive) / stride_time + 1;
        const int64_t width = (input->ne[0] + 2 * pad_frequency - shape.dims[3]) / stride_frequency + 1;
        const int64_t batch = input->ne[3];
        // Spatial tiling bounds im2col memory without changing global normalization or Mamba context.
        const int64_t bytes_per_row = width * shape.dims[1] * shape.dims[2] * shape.dims[3] * sizeof(float);
        const bool cuda_f32 = ctx.backend_type == core::BackendType::Cuda && weight.type == GGML_TYPE_F32;
        const int64_t tile_bytes = (cuda_f32 ? 512LL : 128LL) * 1024 * 1024;
        const int64_t tile = std::max<int64_t>(1, std::min<int64_t>(height, tile_bytes / bytes_per_row));
        if (pad_time) {
            input = modules::Pad2dModule({0, 0, pad_time, pad_time}).build(ctx,
                core::wrap_tensor(input, TensorShape::from_dims({batch, input->ne[2], input->ne[1], input->ne[0]}))).tensor;
        }
        std::vector<ggml_tensor *> tiles;
        for (int64_t start = 0; start < height; start += tile) {
            const int64_t rows = std::min(tile, height - start);
            const int64_t input_rows = (rows - 1) * stride_time + receptive;
            const auto value = modules::SliceModule({2, start * stride_time, input_rows}).build(ctx,
                core::wrap_tensor(input, TensorShape::from_dims({batch, input->ne[2], input->ne[1], input->ne[0]})));
            tiles.push_back(modules::Conv2dModule({shape.dims[1], shape.dims[0], shape.dims[2], shape.dims[3],
                stride_time, stride_frequency, 0, pad_frequency, dilation, 1, false})
                .build(ctx, value, {weight, std::nullopt}).tensor);
        }
        while (tiles.size() > 1) {
            std::vector<ggml_tensor *> joined;
            for (size_t i = 0; i < tiles.size(); i += 2) {
                if (i + 1 == tiles.size()) {
                    joined.push_back(tiles[i]);
                    continue;
                }
                joined.push_back(modules::ConcatModule({2}).build(ctx,
                    core::wrap_tensor(tiles[i], TensorShape::from_dims(
                        {batch, tiles[i]->ne[2], tiles[i]->ne[1], tiles[i]->ne[0]})),
                    core::wrap_tensor(tiles[i + 1], TensorShape::from_dims(
                        {batch, tiles[i + 1]->ne[2], tiles[i + 1]->ne[1], tiles[i + 1]->ne[0]}))).tensor);
            }
            tiles = std::move(joined);
        }
        const auto output_shape = TensorShape::from_dims({batch, shape.dims[0], height, width});
        const auto bias = core::reshape_tensor(ctx, weights_.at(prefix + ".bias"),
            TensorShape::from_dims({1, shape.dims[0], 1, 1}));
        return modules::AddModule().build(ctx, core::wrap_tensor(tiles.front(), output_shape),
            modules::RepeatModule({output_shape}).build(ctx, bias)).tensor;
    }

    ggml_tensor * norm_prelu(ggml_tensor * input, const std::string & prefix) {
        auto & ctx = *ctx_;
        const auto value = core::wrap_tensor(
            input, TensorShape::from_dims({input->ne[3], 64, input->ne[1], input->ne[0]}));
        const auto x = modules::GroupNormModule({64, 64, 1e-5f, true, true}).build(ctx, value,
            {weights_.at(prefix + ".1.weight"), weights_.at(prefix + ".1.bias")});
        const auto alpha = core::reshape_tensor(ctx, weights_.at(prefix + ".2.weight"),
            TensorShape::from_dims({1, 64, 1, 1}));
        const auto positive = modules::ReluModule().build(ctx, x);
        const auto negative = modules::ReluModule().build(ctx,
            core::wrap_tensor(ggml_neg(ctx.ggml, x.tensor), x.shape));
        const auto scaled = modules::MulModule().build(ctx, negative,
            modules::RepeatModule({x.shape}).build(ctx, alpha));
        // The framework has no channel-wise PReLU module.
        return ggml_sub(ctx.ggml, positive.tensor, scaled.tensor);
    }

    ggml_tensor * conv_norm(ggml_tensor * input, const std::string & prefix,
                            int stride_time = 1, int stride_frequency = 1, int dilation = 1, bool same = false) {
        return norm_prelu(conv(input, prefix + ".0", stride_time, stride_frequency, dilation, same), prefix);
    }

    ggml_tensor * dense(ggml_tensor * input, const std::string & prefix) {
        auto * skip = input;
        auto * x = input;
        for (int i = 0; i < 4; ++i) {
            x = conv_norm(skip, prefix + ".dense_block." + std::to_string(i), 1, 1, 1 << i, true);
            if (i != 3) {
                skip = modules::ConcatModule({1}).build(*ctx_,
                    core::wrap_tensor(x, TensorShape::from_dims({x->ne[3], x->ne[2], x->ne[1], x->ne[0]})),
                    core::wrap_tensor(skip, TensorShape::from_dims(
                        {skip->ne[3], skip->ne[2], skip->ne[1], skip->ne[0]}))).tensor;
            }
        }
        return x;
    }

    ggml_tensor * decode(ggml_tensor * input, const std::string & prefix) {
        auto & ctx = *ctx_;
        const int64_t batch = input->ne[3];
        auto * x = dense(input, prefix + ".dense_block");
        auto decoded = core::wrap_tensor(x, TensorShape::from_dims({batch, x->ne[2], x->ne[1], x->ne[0]}));
        for (int i = 1; i <= 2; ++i) {
            if (i == 2) {
                decoded = core::ensure_backend_addressable_layout(ctx,
                    modules::TransposeModule({{0, 1, 3, 2}, 4}).build(ctx, decoded));
            }
            const auto name = prefix + ".up_conv" + std::to_string(i);
            const int64_t r = i == 1 ? 2 : 4;
            decoded = modules::Pad2dModule({1, 1, 0, 0}).build(ctx, decoded);
            x = conv(decoded.tensor, name + ".0.conv");
            const int64_t width = x->ne[0], height = x->ne[1];
            decoded = core::reshape_tensor(ctx,
                core::wrap_tensor(x, TensorShape::from_dims({batch, x->ne[2], height, width})),
                TensorShape::from_dims({batch, r, 64, width * height}));
            decoded = core::ensure_backend_addressable_layout(ctx,
                modules::TransposeModule({{0, 2, 3, 1}, 4}).build(ctx, decoded));
            decoded = core::reshape_tensor(ctx, decoded, TensorShape::from_dims({batch, 64, height, width * r}));
            decoded = core::wrap_tensor(norm_prelu(decoded.tensor, name), decoded.shape);
            if (i == 2) {
                decoded = core::ensure_backend_addressable_layout(ctx,
                    modules::TransposeModule({{0, 1, 3, 2}, 4}).build(ctx, decoded));
            }
        }
        return decoded.tensor;
    }

    ggml_tensor * mamba(ggml_tensor * input, const std::string & prefix,
                        ggml_tensor * state, ggml_tensor * ids) {
        auto & ctx = *ctx_;
        const int64_t length = input->ne[1], batch = input->ne[2];
        auto x = core::wrap_tensor(input, TensorShape::from_dims({batch, length, 64}));
        const bool precise_vulkan_projection = ctx.backend_type == core::BackendType::Vulkan;
        auto * xz = modules::LinearModule({64, 512, false, GGML_PREC_F32, false, precise_vulkan_projection}).build(ctx, x,
            {weights_.at(prefix + ".in_proj.weight"), std::nullopt}).tensor;
        const auto packed = core::wrap_tensor(xz, TensorShape::from_dims({batch, length, 512}));
        auto * gate = modules::SliceModule({2, 256, 256}).build(ctx, packed).tensor;
        if (ctx.backend_type == core::BackendType::Vulkan) {
            gate = core::ensure_backend_addressable_layout(
                ctx, core::wrap_tensor(gate, TensorShape::from_dims({batch, length, 256}))).tensor;
        }
        const auto projection = modules::SliceModule({2, 0, 256}).build(ctx, packed);
        auto * projected = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, projection).tensor;
        // Preserve the strided causal-padding/SSM-convolution fusion layout.
        projected = ggml_pad_ext(ctx.ggml, projected, 3, 0, 0, 0, 0, 0, 0, 0);
        projected = ggml_ssm_conv(ctx.ggml, projected, weights_.at(prefix + ".conv1d.weight").tensor);
        if (ctx.backend_type == core::BackendType::Cuda) {
            ggml_ssm_conv_set_fusion(projected, GGML_SSM_CONV_FUSION_CAUSAL_PAD);
        }
        const auto sequence_shape = TensorShape::from_dims({batch, length, 256});
        const auto bias = core::wrap_tensor(weights_.at(prefix + ".conv1d.bias").tensor,
            TensorShape::from_dims({1, 1, 256}));
        const auto value = modules::SiluModule().build(ctx, modules::AddModule().build(ctx,
            core::wrap_tensor(projected, sequence_shape),
            modules::RepeatModule({sequence_shape}).build(ctx, bias)));
        projected = value.tensor;
        auto * parameters = modules::LinearModule({256, 36, false, GGML_PREC_F32, false, precise_vulkan_projection}).build(ctx, value,
            {weights_.at(prefix + ".x_proj.weight"), std::nullopt}).tensor;
        const auto parameter_value = core::wrap_tensor(parameters, TensorShape::from_dims({batch, length, 36}));
        auto * dt = core::ensure_backend_addressable_layout(ctx,
            modules::SliceModule({2, 0, 4}).build(ctx, parameter_value)).tensor;
        if (ctx.backend_type == core::BackendType::Cuda &&
            weights_.at(prefix + ".dt_proj.weight").tensor->type == GGML_TYPE_F32) {
            const auto tail = modules::RepeatModule({TensorShape::from_dims({batch, length, 4})}).build(
                ctx, weights_.at("reuse.affine_tail"));
            const auto affine_input = modules::ConcatModule({2}).build(ctx,
                core::wrap_tensor(dt, TensorShape::from_dims({batch, length, 4})), tail);
            dt = modules::LinearModule({8, 256, false, GGML_PREC_F32, false, false}).build(ctx,
                affine_input, {weights_.at(prefix + ".dt_proj.weight.affine"), std::nullopt}).tensor;
        } else {
            dt = modules::LinearModule({4, 256, true, GGML_PREC_F32, false, precise_vulkan_projection}).build(ctx,
                core::wrap_tensor(dt, TensorShape::from_dims({batch, length, 4})),
                {weights_.at(prefix + ".dt_proj.weight"), weights_.at(prefix + ".dt_proj.bias")}).tensor;
        }
        auto * b = ggml_view_4d(ctx.ggml, parameters, 16, 1, length, batch,
            16 * sizeof(float), parameters->nb[1], parameters->nb[2], 4 * sizeof(float));
        auto * c = ggml_view_4d(ctx.ggml, parameters, 16, 1, length, batch,
            16 * sizeof(float), parameters->nb[1], parameters->nb[2], 20 * sizeof(float));
        auto * scan = ggml_ssm_scan(ctx.ggml, state,
            core::reshape_tensor(ctx, value, TensorShape::from_dims({batch, length, 256, 1})).tensor, dt,
            weights_.at(prefix + ".A_log").tensor, b, c, ids);
        if (ctx.backend_type == core::BackendType::Cuda) {
            ggml_ssm_scan_set_fusion(scan, GGML_SSM_SCAN_FUSION_GATE);
        }
        auto * y = ggml_view_3d(ctx.ggml, scan, 256, length, batch,
            256 * sizeof(float), 256 * length * sizeof(float), 0);
        auto * scan_output = y;
        const auto scale = core::wrap_tensor(weights_.at(prefix + ".D").tensor, TensorShape::from_dims({1, 1, 256}));
        const auto skip = modules::MulModule().build(ctx, value,
            modules::RepeatModule({sequence_shape}).build(ctx, scale));
        y = modules::AddModule().build(ctx, core::wrap_tensor(y, sequence_shape), skip).tensor;
        y = ggml_swiglu_split(ctx.ggml, gate, y);
        if (ctx.backend_type == core::BackendType::Cuda) {
            // Keep the scan buffer as destination; a fused scan must not overwrite live B/C storage.
            y = ggml_cpy(ctx.ggml, y, scan_output);
        }
        return modules::LinearModule({256, 64, false, GGML_PREC_F32, false, precise_vulkan_projection}).build(ctx,
            core::wrap_tensor(y, TensorShape::from_dims({batch, length, 256})),
            {weights_.at(prefix + ".out_proj.weight"), std::nullopt}).tensor;
    }

    ggml_tensor * bidirectional(ggml_tensor * input, const std::string & prefix,
                                ggml_tensor * state, ggml_tensor * ids, ggml_tensor * reverse) {
        auto & ctx = *ctx_;
        const auto input_shape = TensorShape::from_dims({input->ne[2], input->ne[1], input->ne[0]});
        auto * forward = modules::AddModule().build(ctx,
            core::wrap_tensor(mamba(input, prefix + ".forward_blocks", state, ids), input_shape),
            core::wrap_tensor(input, input_shape)).tensor;
        auto * reversed = ggml_get_rows(ctx.ggml, input, reverse);
        auto * backward = modules::AddModule().build(ctx,
            core::wrap_tensor(mamba(reversed, prefix + ".backward_blocks", state, ids), input_shape),
            core::wrap_tensor(reversed, input_shape)).tensor;
        backward = ggml_get_rows(ctx.ggml, backward, reverse);
        auto * joined = modules::ConcatModule({2}).build(ctx,
            core::wrap_tensor(forward, input_shape), core::wrap_tensor(backward, input_shape)).tensor;
        const auto shape = TensorShape::from_dims({input->ne[2], input->ne[1], 128});
        const bool precise_vulkan_projection = ctx.backend_type == core::BackendType::Vulkan;
        const auto projected = modules::LinearModule({128, 64, true, GGML_PREC_F32, false, precise_vulkan_projection}).build(ctx,
            core::wrap_tensor(joined, shape),
            {weights_.at(prefix + ".output_proj.weight"), weights_.at(prefix + ".output_proj.bias")});
        return modules::LayerNormModule({64, 1e-5f, true, true}).build(ctx, projected,
            {weights_.at(prefix + ".norm.weight"), weights_.at(prefix + ".norm.bias")}).tensor;
    }

    ggml_backend_t backend_;
    const TensorMap & weights_;
    core::ModuleBuildContext * ctx_ = nullptr;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context_{nullptr, ggml_free};
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator_{nullptr, ggml_gallocr_free};
    ggml_cgraph * graph_ = nullptr;
    ggml_tensor * input_ = nullptr;
    std::vector<ggml_tensor *> outputs_;
    std::vector<std::pair<ggml_tensor *, std::vector<int32_t>>> indices_;
};

}  // namespace

std::shared_ptr<const ReuseAssets> load_reuse_assets(const std::filesystem::path & path) {
    auto result = std::make_shared<ReuseAssets>();
    result->resources = model_spec::load_resource_bundle(path, model_spec::default_spec_path("reuse"));
    result->tensors = result->resources.open_tensor_source("weights");
    const auto json = result->resources.parse_json("config");
    const auto & model = json.require("model_cfg");
    const auto & stft = json.require("stft_cfg");
    if (io::json::require_i32(model, "hid_feature") != 64 || io::json::require_i32(model, "d_state") != 16 ||
        io::json::require_i32(model, "d_conv") != 4 || io::json::require_i32(model, "expand") != 4 ||
        io::json::require_i32(model, "num_tfmamba") != 30 || io::json::require_string(model, "compress_factor") != "relu_log1p" ||
        io::json::require_i32(stft, "n_fft") != 320 || io::json::require_i32(stft, "hop_size") != 40 ||
        io::json::require_i32(stft, "win_size") != 320 || io::json::require_i32(stft, "sampling_rate") != 8000) {
        throw std::runtime_error("RE-USE requires the official SEMamba configuration");
    }
    return result;
}

class ReuseRuntime::Impl {
public:
    Impl(std::shared_ptr<const ReuseAssets> assets, core::ExecutionContext & execution, assets::TensorStorageType storage)
        : assets_(std::move(assets)), execution_(execution),
          store_(execution.backend(), execution.backend_type(), "reuse.weights", 4 * 1024 * 1024) {
        const auto & source = *assets_->tensors;
        for (const auto & metadata : source.tensors()) {
            const auto & name = metadata.name;
            const auto & shape = metadata.shape;
            if (name.size() >= 6 && name.compare(name.size() - 6, 6, ".A_log") == 0) {
                auto values = source.require_f32(name, shape);
                for (auto & value : values) {
                    value = -std::exp(value);
                }
                weights_.emplace(name, store_.make_f32(TensorShape::from_dims({256, 16}), std::move(values)));
            } else if (shape.size() == 3) {
                weights_.emplace(name, store_.load_tensor_as_shape(source, name, assets::TensorStorageType::F32,
                    shape, TensorShape::from_dims({256, 4})));
            } else {
                weights_.emplace(name, store_.load_tensor(source, name,
                    shape.size() == 1 ? assets::TensorStorageType::F32 : storage, shape));
            }
        }
        if (execution.backend_type() == core::BackendType::Cuda) {
            weights_.emplace("reuse.affine_tail", store_.make_f32(
                TensorShape::from_dims({1, 1, 4}), {1.0f, 0.0f, 0.0f, 0.0f}));
            for (const auto & metadata : source.tensors()) {
                const auto & name = metadata.name;
                if (name.size() < 15 || name.compare(name.size() - 15, 15, ".dt_proj.weight") != 0 ||
                    weights_.at(name).tensor->type != GGML_TYPE_F32) {
                    continue;
                }
                const auto matrix = source.require_f32(name, metadata.shape);
                const auto bias = source.require_f32(name.substr(0, name.size() - 6) + "bias", {256});
                std::vector<float> affine(256 * 8, 0.0f);
                for (size_t row = 0; row < 256; ++row) {
                    std::copy_n(matrix.data() + row * 4, 4, affine.data() + row * 8);
                    affine[row * 8 + 4] = bias[row];
                }
                weights_.emplace(name + ".affine", store_.make_f32(
                    TensorShape::from_dims({256, 8}), std::move(affine)));
            }
        }
        store_.upload();
        assets_->tensors->release_storage();
        core::set_backend_threads(execution.backend(), execution.config().threads);
    }

    std::vector<std::vector<float>> restore_batch(
        const std::vector<std::vector<float>> & waveforms,
        int sample_rate) {
        if (waveforms.empty()) {
            throw std::runtime_error("RE-USE batch must not be empty");
        }
        const auto started = std::chrono::steady_clock::now();
        std::vector<FrontendSpectrum> frontends;
        frontends.reserve(waveforms.size());
        for (const auto & waveform : waveforms) {
            frontends.push_back(analyze_spectrum(waveform, sample_rate, execution_.config().threads));
        }
        debug::timing_log_scalar("reuse.frontend.ms", debug::elapsed_ms(started));
        const int64_t batch = static_cast<int64_t>(frontends.size());
        const int64_t frames = frontends.front().frames;
        const int64_t bins = frontends.front().bins;
        std::vector<float> features;
        if (batch == 1) {
            features = std::move(frontends.front().features);
        } else {
            features.reserve(static_cast<size_t>(batch * frames * bins * 2));
            for (const auto & frontend : frontends) {
                if (frontend.frames != frames || frontend.bins != bins) {
                    throw std::runtime_error("RE-USE batch inputs must have the same spectrum shape");
                }
                features.insert(features.end(), frontend.features.begin(), frontend.features.end());
            }
        }
        const auto heads = run_graph(features, batch, frames, bins);
        const auto decode_started = std::chrono::steady_clock::now();
        std::vector<std::vector<float>> outputs;
        outputs.reserve(frontends.size());
        std::vector<std::vector<float>> item_heads(heads.size());
        for (int64_t item = 0; item < batch; ++item) {
            if (batch > 1) {
                for (size_t head = 0; head < heads.size(); ++head) {
                    if (heads[head].size() % static_cast<size_t>(batch) != 0) {
                        throw std::runtime_error("RE-USE batched output shape mismatch");
                    }
                    const size_t stride = heads[head].size() / static_cast<size_t>(batch);
                    const auto begin = heads[head].begin() + item * stride;
                    item_heads[head].assign(begin, begin + stride);
                }
            }
            outputs.push_back(synthesize_spectrum(
                frontends[static_cast<size_t>(item)], batch == 1 ? heads : item_heads, execution_.config().threads));
        }
        debug::timing_log_scalar("reuse.synthesis.ms", debug::elapsed_ms(decode_started));
        return outputs;
    }

    void process_chunks(size_t count, int sample_rate,
        const std::function<void(size_t, std::vector<float> &)> & source,
        const std::function<void(size_t, const std::vector<float> &)> & sink) {
        if (execution_.backend_type() != core::BackendType::Cuda || count < 2) {
            std::vector<float> chunk;
            for (size_t i = 0; i < count; ++i) {
                source(i, chunk);
                sink(i, restore_batch({chunk}, sample_rate).front());
            }
            return;
        }
        const auto threads = execution_.config().threads;
        // As in RoFormer, only host stages overlap. One graph and ordered
        // overlap-add stay on the caller; futures own and join their buffers.
        const auto launch = [&](size_t index) {
            return std::async(std::launch::async, [&, index] {
#ifdef _OPENMP
                if (threads > 0) omp_set_num_threads(static_cast<int>(threads));
#endif
                std::vector<float> chunk;
                source(index, chunk);
                const auto started = std::chrono::steady_clock::now();
                auto prepared = analyze_spectrum(chunk, sample_rate, threads);
                debug::timing_log_scalar("reuse.frontend.ms", debug::elapsed_ms(started));
                return prepared;
            });
        };
        auto pending = launch(0);
        std::future<std::vector<float>> reconstructed;
        for (size_t i = 0; i < count; ++i) {
            auto prepared = pending.get();
            if (i + 1 < count) {
                pending = launch(i + 1);
            }
            auto heads = run_graph(prepared.features, 1, prepared.frames, prepared.bins);
            if (reconstructed.valid()) {
                sink(i - 1, reconstructed.get());
            }
            reconstructed = std::async(std::launch::async,
                [threads, prepared = std::move(prepared), heads = std::move(heads)]() mutable {
#ifdef _OPENMP
                    if (threads > 0) omp_set_num_threads(static_cast<int>(threads));
#endif
                    const auto started = std::chrono::steady_clock::now();
                    auto output = synthesize_spectrum(prepared, heads, threads);
                    debug::timing_log_scalar("reuse.synthesis.ms", debug::elapsed_ms(started));
                    return output;
                });
        }
        sink(count - 1, reconstructed.get());
    }

private:
    std::vector<std::vector<float>> run_graph(const std::vector<float> & features,
                                             int64_t batch, int64_t frames, int64_t bins) {
        if (!graph_ || batch != batch_ || frames != frames_ || bins != bins_) {
            graph_.reset();
            graph_ = std::make_unique<ReuseGraph>(execution_, weights_, batch, frames, bins);
            batch_ = batch;
            frames_ = frames;
            bins_ = bins;
        }
        return graph_->run(features);
    }

    std::shared_ptr<const ReuseAssets> assets_;
    core::ExecutionContext & execution_;
    core::BackendWeightStore store_;
    TensorMap weights_;
    std::unique_ptr<ReuseGraph> graph_;
    int64_t batch_ = 0, frames_ = 0, bins_ = 0;
};

ReuseRuntime::ReuseRuntime(std::shared_ptr<const ReuseAssets> assets, core::ExecutionContext & execution,
                           assets::TensorStorageType storage)
    : impl_(std::make_unique<Impl>(std::move(assets), execution, storage)) {}
ReuseRuntime::~ReuseRuntime() = default;
std::vector<float> ReuseRuntime::restore(const std::vector<float> & waveform, int sample_rate) {
    auto outputs = impl_->restore_batch({waveform}, sample_rate);
    return std::move(outputs.front());
}
void ReuseRuntime::process_chunks(size_t count, int sample_rate,
    const std::function<void(size_t, std::vector<float> &)> & source,
    const std::function<void(size_t, const std::vector<float> &)> & sink) {
    impl_->process_chunks(count, sample_rate, source, sink);
}
std::vector<std::vector<float>> ReuseRuntime::restore_batch(
    const std::vector<std::vector<float>> & waveforms,
    int sample_rate) {
    return impl_->restore_batch(waveforms, sample_rate);
}

}  // namespace engine::models::reuse
