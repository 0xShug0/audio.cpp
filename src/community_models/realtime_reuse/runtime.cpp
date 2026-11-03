#include "engine/community_models/realtime_reuse/runtime.h"

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <ggml-alloc.h>
#include <cmath>
#include <map>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <unordered_map>

namespace engine::models::realtime_reuse {
namespace {
using core::TensorShape;
using core::TensorValue;
using TensorMap = std::unordered_map<std::string, TensorValue>;

class Graph {
public:
    Graph(core::ExecutionContext & execution, const TensorMap & weights,
        ggml_context * state_context, TensorMap & states,
        std::vector<ggml_backend_buffer_t> & state_buffers, Config config,
        int64_t frames, int64_t bins, bool streaming, bool first)
        : execution_(execution), weights_(weights), state_context_(state_context), states_(states), state_buffers_(state_buffers), state_prefix_(std::to_string(bins) + "/"),
          config_(config), streaming_(streaming), first_(first),
          context_(ggml_init({32768 * ggml_tensor_overhead() + ggml_graph_overhead_custom(32768, false), nullptr, true}), ggml_free),
          ctx_{context_.get(), "realtime_reuse", execution.backend_type()} {
        if (!context_) throw std::runtime_error("Real-time RE-USE graph allocation failed");
        const auto started = std::chrono::steady_clock::now();
        input_ = core::make_tensor(ctx_, GGML_TYPE_F32, TensorShape::from_dims({1, 2, frames, bins})).tensor;
        ggml_set_input(input_);
        auto padded = modules::Pad2dModule({0, 2, 0, 0}).build(ctx_,
            core::wrap_tensor(input_, TensorShape::from_dims({1, 2, frames, bins})));
        auto * x = conv(padded.tensor, "dense_encoder.dense_conv_1_" + std::to_string(config.lookahead_frames + 1) + ".0", 1, 1, true);
        x = norm_prelu(x, "dense_encoder.dense_conv_1_" + std::to_string(config.lookahead_frames + 1));
        x = dense(x, "dense_encoder.dense_block");
        x = norm_prelu(conv(x, "dense_encoder.dense_conv_2.0", 1, 2, false, false), "dense_encoder.dense_conv_2");
        const int64_t time = x->ne[1], frequency = x->ne[0];
        for (int layer = 0; layer < config.exit_layer; ++layer) {
            const auto prefix = "TSMamba." + std::to_string(layer);
            auto value = core::wrap_tensor(x, TensorShape::from_dims({1, 64, time, frequency}));
            value = core::ensure_backend_addressable_layout(ctx_, modules::TransposeModule({{0, 3, 2, 1}, 4}).build(ctx_, value));
            value = core::reshape_tensor(ctx_, value, TensorShape::from_dims({frequency, time, 64}));
            const auto causal = core::wrap_tensor(mamba(value.tensor, prefix + ".time_mamba.forward_blocks", true), value.shape);
            auto time_out = modules::AddModule().build(ctx_, causal, value);
            time_out = modules::LinearModule({64, 64, true, GGML_PREC_DEFAULT}).build(ctx_, time_out,
                {weights_.at(prefix + ".time_mamba.output_proj.weight"), weights_.at(prefix + ".time_mamba.output_proj.bias")});
            time_out = modules::LayerNormModule({64}).build(ctx_, time_out,
                {weights_.at(prefix + ".time_mamba.norm.weight"), weights_.at(prefix + ".time_mamba.norm.bias")});
            time_out = modules::AddModule().build(ctx_, time_out, value);
            time_out = core::reshape_tensor(ctx_, time_out, TensorShape::from_dims({1, frequency, time, 64}));
            time_out = core::ensure_backend_addressable_layout(ctx_, modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx_, time_out));
            time_out = core::reshape_tensor(ctx_, time_out, TensorShape::from_dims({time, frequency, 64}));
            auto freq_out = modules::AddModule().build(ctx_,
                core::wrap_tensor(bidirectional(time_out.tensor, prefix + ".freq_mamba"), time_out.shape), time_out);
            freq_out = core::reshape_tensor(ctx_, freq_out, TensorShape::from_dims({1, time, frequency, 64}));
            freq_out = core::ensure_backend_addressable_layout(ctx_, modules::TransposeModule({{0, 3, 1, 2}, 4}).build(ctx_, freq_out));
            x = freq_out.tensor;
        }
        const auto suffix = std::to_string(config.exit_layer - 1);
        auto * magnitude = conv(decode(x, "mask_decoder_list." + suffix), "mask_decoder_list." + suffix + ".final_conv");
        auto * phase = decode(x, "phase_decoder_list." + suffix);
        outputs_ = {magnitude, conv(phase, "phase_decoder_list." + suffix + ".phase_conv_r"),
                    conv(phase, "phase_decoder_list." + suffix + ".phase_conv_i")};
        graph_ = ggml_new_graph_custom(ctx_.ggml, 32768, false);
        for (auto * output : outputs_) {
            ggml_set_output(output);
            ggml_build_forward_expand(graph_, output);
        }
        // Commit state only after every reader of the previous frame has completed.
        for (auto * update : updates_) ggml_build_forward_expand(graph_, update);
        for (int i = 0; i < ggml_graph_n_nodes(graph_); ++i) {
            auto * node = ggml_graph_node(graph_, i);
            if (ctx_.backend_type == core::BackendType::Cuda && node->op == GGML_OP_IM2COL &&
                node->type == GGML_TYPE_F32 && node->src[0]->ne[0] == 3 && node->src[0]->ne[1] == 3 &&
                node->src[0]->ne[2] >= 32 && node->ne[1] >= 32)
                ggml_im2col_2d_set_lowering(node, GGML_IM2COL_2D_LOWERING_CUDA_F32_K3_TILED);
        }
        core::validate_backend_graph_supported(execution.backend(), graph_, "Real-time RE-USE");
        allocator_.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend())));
        if (!ggml_gallocr_alloc_graph(allocator_.get(), graph_)) throw std::runtime_error("Real-time RE-USE graph allocation failed");
        for (const auto & entry : indices_) ggml_backend_tensor_set(entry.first, entry.second.data(), 0, entry.second.size() * sizeof(int32_t));
        indices_.clear();
        debug::timing_log_scalar("realtime_reuse.graph.build_ms", debug::elapsed_ms(started));
        debug::timing_log_scalar("realtime_reuse.graph.buffer_mb", ggml_gallocr_get_buffer_size(allocator_.get(), 0) / 1048576.0);
    }
    ~Graph() { core::release_backend_graph_resources(execution_.backend(), graph_, true); }

    std::array<std::vector<float>, 3> run(const std::vector<float> & input) {
        if (input.size() * sizeof(float) != ggml_nbytes(input_)) throw std::runtime_error("Real-time RE-USE feature shape mismatch");
        ggml_backend_tensor_set(input_, input.data(), 0, ggml_nbytes(input_));
        if (core::compute_backend_graph(execution_.backend(), graph_) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("Real-time RE-USE graph execution failed");
        std::array<std::vector<float>, 3> result;
        for (size_t i = 0; i < outputs_.size(); ++i) {
            result[i].resize(ggml_nelements(outputs_[i]));
            ggml_backend_tensor_get(outputs_[i], result[i].data(), 0, ggml_nbytes(outputs_[i]));
        }
        return result;
    }

private:
    ggml_tensor * state(const std::string & name, const TensorShape & shape) {
        const auto key = state_prefix_ + name;
        auto found = states_.find(key);
        if (found == states_.end()) {
            const size_t count = shape.num_elements();
            core::ModuleBuildContext state_ctx{state_context_, "realtime_reuse.state", execution_.backend_type()};
            auto value = core::make_tensor(state_ctx, GGML_TYPE_F32, shape);
            const auto buft = ggml_backend_get_default_buffer_type(execution_.backend());
            auto * buffer = ggml_backend_alloc_buffer(execution_.backend(), ggml_backend_buft_get_alloc_size(buft, value.tensor));
            if (!buffer) throw std::runtime_error("Real-time RE-USE recurrent state allocation failed");
            state_buffers_.push_back(buffer);
            if (ggml_backend_tensor_alloc(buffer, value.tensor, ggml_backend_buffer_get_base(buffer)) != GGML_STATUS_SUCCESS)
                throw std::runtime_error("Real-time RE-USE recurrent state binding failed");
            ggml_backend_tensor_memset(value.tensor, 0, 0, count * sizeof(float));
            ggml_set_input(value.tensor);
            found = states_.emplace(key, value).first;
        }
        core::validate_shape(found->second, shape, name.c_str());
        return found->second.tensor;
    }

    ggml_tensor * conv(ggml_tensor * x, const std::string & prefix, int dilation = 1, int stride = 1, bool initial = false, bool symmetric_frequency = true) {
        const auto & weight = weights_.at(prefix + ".weight");
        const auto & dims = weight.shape.dims;
        const int64_t history = (dims[2] - 1) * dilation;
        auto value = core::wrap_tensor(x, TensorShape::from_dims({1, x->ne[2], x->ne[1], x->ne[0]}));
        if (history) {
            if (streaming_) {
                const auto shape = TensorShape::from_dims({1, x->ne[2], history, x->ne[0]});
                auto * cache = state(prefix + ".history", shape);
                auto previous = core::wrap_tensor(cache, shape);
                if (initial && first_ && config_.lookahead_frames && config_.lookahead_frames < history) {
                    previous = modules::SliceModule({2, config_.lookahead_frames, history - config_.lookahead_frames}).build(ctx_, previous);
                }
                if (!initial || !first_ || config_.lookahead_frames < history)
                    value = modules::ConcatModule({2}).build(ctx_, previous, value);
                const auto tail = modules::SliceModule({2, value.shape.dims[2] - history, history}).build(ctx_, value);
                updates_.push_back(ggml_cpy(ctx_.ggml, tail.tensor, cache));
            } else {
                const int ahead = initial ? config_.lookahead_frames : 0;
                value = modules::Pad2dModule({0, 0, history - ahead, ahead}).build(ctx_, value);
            }
        }
        modules::Conv2dConfig config{dims[1], dims[0], dims[2], dims[3], 1, stride, 0,
            symmetric_frequency ? static_cast<int>(dims[3] / 2) : 0, dilation, 1, true};
        // Metal: avoid im2col; the module checks kernel eligibility.
        config.implicit_gemm = ctx_.backend_type == core::BackendType::Metal;
        return modules::Conv2dModule(config)
            .build(ctx_, value, {weight, weights_.at(prefix + ".bias")}).tensor;
    }

    ggml_tensor * norm_prelu(ggml_tensor * x, const std::string & prefix, int norm_index = 1) {
        auto value = core::wrap_tensor(x, TensorShape::from_dims({1, 64, x->ne[1], x->ne[0]}));
        value = core::ensure_backend_addressable_layout(ctx_, modules::TransposeModule({{0, 2, 3, 1}, 4}).build(ctx_, value));
        const auto norm = prefix + "." + std::to_string(norm_index) + ".norm";
        value = modules::LayerNormModule({64}).build(ctx_, value, {weights_.at(norm + ".weight"), weights_.at(norm + ".bias")});
        value = core::ensure_backend_addressable_layout(ctx_, modules::TransposeModule({{0, 3, 1, 2}, 4}).build(ctx_, value));
        return modules::PReluModule({1}).build(ctx_, value, weights_.at(prefix + "." + std::to_string(norm_index + 1) + ".weight")).tensor;
    }

    ggml_tensor * dense(ggml_tensor * input, const std::string & prefix) {
        auto * skip = input;
        auto * x = input;
        for (int i = 0; i < 4; ++i) {
            const auto name = prefix + ".dense_block." + std::to_string(i);
            x = norm_prelu(conv(skip, name + ".1", 1 << i), name, 2);
            if (i != 3) skip = modules::ConcatModule({1}).build(ctx_,
                core::wrap_tensor(x, TensorShape::from_dims({1, 64, x->ne[1], x->ne[0]})),
                core::wrap_tensor(skip, TensorShape::from_dims({1, skip->ne[2], skip->ne[1], skip->ne[0]}))).tensor;
        }
        return x;
    }

    ggml_tensor * decode(ggml_tensor * x, const std::string & prefix) {
        x = dense(x, prefix + ".dense_block");
        x = conv(x, prefix + ".up_conv1.0.conv");
        const int64_t time = x->ne[1], bins = x->ne[0];
        auto value = core::reshape_tensor(ctx_, core::wrap_tensor(x, TensorShape::from_dims({1, 128, time, bins})),
            TensorShape::from_dims({1, 2, 64, time * bins}));
        value = core::ensure_backend_addressable_layout(ctx_, modules::TransposeModule({{0, 2, 3, 1}, 4}).build(ctx_, value));
        value = core::reshape_tensor(ctx_, value, TensorShape::from_dims({1, 64, time, bins * 2}));
        x = norm_prelu(value.tensor, prefix + ".up_conv1");
        return norm_prelu(conv(x, prefix + ".up_conv2.0.conv"), prefix + ".up_conv2");
    }

    ggml_tensor * mamba(ggml_tensor * input, const std::string & prefix, bool temporal) {
        const int64_t length = input->ne[1], batch = input->ne[2];
        const auto input_shape = TensorShape::from_dims({batch, length, 64});
        const auto sequence_shape = TensorShape::from_dims({batch, length, 256});
        const bool precise = ctx_.backend_type == core::BackendType::Vulkan;
        auto packed = modules::LinearModule({64, 512, false, GGML_PREC_DEFAULT, false, precise}).build(ctx_,
            core::wrap_tensor(input, input_shape), {weights_.at(prefix + ".in_proj.weight"), std::nullopt});
        auto gate = modules::SliceModule({2, 256, 256}).build(ctx_, packed);
        if (ctx_.backend_type == core::BackendType::Vulkan) gate = core::ensure_backend_addressable_layout(ctx_, gate);
        auto projected = modules::SliceModule({2, 0, 256}).build(ctx_, packed);
        projected = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx_, projected);
        // Metal opts into the existing compact scan and stateless causal convolution.
        const bool fused_mamba = ctx_.backend_type == core::BackendType::Vulkan ||
            ctx_.backend_type == core::BackendType::Metal;
        const bool compact_scan = fused_mamba && weights_.at(prefix + ".dt_proj.weight").tensor->type == GGML_TYPE_F32;
        const bool fused_conv = fused_mamba && !(streaming_ && temporal);
        if (streaming_ && temporal) {
            const auto shape = TensorShape::from_dims({batch, 256, 3});
            auto * history = state(prefix + ".conv_state", shape);
            projected = modules::ConcatModule({2}).build(ctx_, core::wrap_tensor(history, shape), projected);
            const auto tail = modules::SliceModule({2, projected.shape.dims[2] - 3, 3}).build(ctx_, projected);
            updates_.push_back(ggml_cpy(ctx_.ggml, tail.tensor, history));
        } else if (!fused_conv) {
            projected = core::wrap_tensor(ggml_pad_ext(ctx_.ggml, projected.tensor, 3, 0, 0, 0, 0, 0, 0, 0),
                TensorShape::from_dims({batch, 256, length + 3}));
        }
        TensorValue value;
        if (fused_conv) {
            value = core::wrap_tensor(ggml_ssm_conv_causal_silu(ctx_.ggml, projected.tensor,
                weights_.at(prefix + ".conv1d.weight").tensor, weights_.at(prefix + ".conv1d.bias").tensor), sequence_shape);
        } else {
            auto * convolved = ggml_ssm_conv(ctx_.ggml, projected.tensor, weights_.at(prefix + ".conv1d.weight").tensor);
            if (ctx_.backend_type == core::BackendType::Cuda && !(streaming_ && temporal))
                ggml_ssm_conv_set_fusion(convolved, GGML_SSM_CONV_FUSION_CAUSAL_PAD);
            const auto bias = core::reshape_tensor(ctx_, weights_.at(prefix + ".conv1d.bias"), TensorShape::from_dims({1, 1, 256}));
            value = modules::SiluModule().build(ctx_, modules::AddModule().build(ctx_,
                core::wrap_tensor(convolved, sequence_shape), modules::RepeatModule({sequence_shape}).build(ctx_, bias)));
        }
        auto parameters = modules::LinearModule({256, 36, false, GGML_PREC_DEFAULT, false, precise}).build(ctx_, value,
            {weights_.at(prefix + ".x_proj.weight"), std::nullopt});
        const auto state_shape = TensorShape::from_dims({batch, 256, 1, 16});
        auto * initial = streaming_ && temporal ? state(prefix + ".ssm_state", state_shape)
            : ggml_fill(ctx_.ggml, core::make_tensor(ctx_, GGML_TYPE_F32, state_shape).tensor, 0);
        auto * ids = ggml_new_tensor_1d(ctx_.ggml, GGML_TYPE_I32, batch);
        ggml_set_input(ids);
        ggml_set_output(ids);
        std::vector<int32_t> identity(batch);
        std::iota(identity.begin(), identity.end(), 0);
        indices_.emplace_back(ids, std::move(identity));
        ggml_tensor * scan;
        if (compact_scan) {
            scan = ggml_ssm_scan_mamba1(ctx_.ggml, initial,
                core::reshape_tensor(ctx_, value, TensorShape::from_dims({batch, length, 256, 1})).tensor,
                parameters.tensor, weights_.at(prefix + ".A_log").tensor, ids,
                weights_.at(prefix + ".dt_proj.weight").tensor, weights_.at(prefix + ".dt_proj.bias").tensor,
                weights_.at(prefix + ".D").tensor, gate.tensor);
        } else {
            auto delta = core::ensure_backend_addressable_layout(ctx_, modules::SliceModule({2, 0, 4}).build(ctx_, parameters));
            delta = modules::LinearModule({4, 256, true, GGML_PREC_DEFAULT, false, precise}).build(ctx_, delta,
                {weights_.at(prefix + ".dt_proj.weight"), weights_.at(prefix + ".dt_proj.bias")});
            auto * b = ggml_view_4d(ctx_.ggml, parameters.tensor, 16, 1, length, batch,
                16 * sizeof(float), parameters.tensor->nb[1], parameters.tensor->nb[2], 4 * sizeof(float));
            auto * c = ggml_view_4d(ctx_.ggml, parameters.tensor, 16, 1, length, batch,
                16 * sizeof(float), parameters.tensor->nb[1], parameters.tensor->nb[2], 20 * sizeof(float));
            scan = ggml_ssm_scan(ctx_.ggml, initial,
                core::reshape_tensor(ctx_, value, TensorShape::from_dims({batch, length, 256, 1})).tensor,
                delta.tensor, weights_.at(prefix + ".A_log").tensor, b, c, ids);
        }
        if (streaming_ && temporal) {
            auto * final = ggml_view_4d(ctx_.ggml, scan, 16, 1, 256, batch,
                initial->nb[1], initial->nb[2], initial->nb[3], batch * length * 256 * sizeof(float));
            updates_.push_back(ggml_cpy(ctx_.ggml, final, initial));
        }
        auto * y = ggml_view_3d(ctx_.ggml, scan, 256, length, batch,
            256 * sizeof(float), length * 256 * sizeof(float), 0);
        if (compact_scan) {
            return modules::LinearModule({256, 64, false, GGML_PREC_DEFAULT, false, precise}).build(ctx_,
                core::wrap_tensor(y, sequence_shape), {weights_.at(prefix + ".out_proj.weight"), std::nullopt}).tensor;
        }
        const bool fused_scan = ctx_.backend_type == core::BackendType::Cuda && !(streaming_ && temporal);
        if (fused_scan) ggml_ssm_scan_set_fusion(scan, GGML_SSM_SCAN_FUSION_GATE);
        const auto scale = core::reshape_tensor(ctx_, weights_.at(prefix + ".D"), TensorShape::from_dims({1, 1, 256}));
        auto result = modules::AddModule().build(ctx_, core::wrap_tensor(y, sequence_shape),
            modules::MulModule().build(ctx_, value, modules::RepeatModule({sequence_shape}).build(ctx_, scale)));
        result = core::wrap_tensor(ggml_swiglu_split(ctx_.ggml, gate.tensor, result.tensor), sequence_shape);
        if (fused_scan) result = core::wrap_tensor(ggml_cpy(ctx_.ggml, result.tensor, y), sequence_shape);
        return modules::LinearModule({256, 64, false, GGML_PREC_DEFAULT, false, precise}).build(ctx_, result,
            {weights_.at(prefix + ".out_proj.weight"), std::nullopt}).tensor;
    }

    ggml_tensor * bidirectional(ggml_tensor * input, const std::string & prefix) {
        const int64_t length = input->ne[1], batch = input->ne[2];
        const auto shape = TensorShape::from_dims({batch, length, 64});
        auto * reverse = ggml_new_tensor_2d(ctx_.ggml, GGML_TYPE_I32, length, batch);
        ggml_set_input(reverse);
        ggml_set_output(reverse);
        std::vector<int32_t> indices(length * batch);
        for (int64_t i = 0; i < batch; ++i)
            for (int64_t t = 0; t < length; ++t) indices[i * length + t] = length - t - 1;
        indices_.emplace_back(reverse, std::move(indices));
        auto forward = modules::AddModule().build(ctx_, core::wrap_tensor(mamba(input, prefix + ".forward_blocks", false), shape), core::wrap_tensor(input, shape));
        auto * reversed = ggml_get_rows(ctx_.ggml, input, reverse);
        auto backward = modules::AddModule().build(ctx_, core::wrap_tensor(mamba(reversed, prefix + ".backward_blocks", false), shape), core::wrap_tensor(reversed, shape));
        backward = core::wrap_tensor(ggml_get_rows(ctx_.ggml, backward.tensor, reverse), shape);
        auto joined = modules::ConcatModule({2}).build(ctx_, forward, backward);
        auto projected = modules::LinearModule({128, 64, true, GGML_PREC_DEFAULT}).build(ctx_, joined,
            {weights_.at(prefix + ".output_proj.weight"), weights_.at(prefix + ".output_proj.bias")});
        return modules::LayerNormModule({64}).build(ctx_, projected,
            {weights_.at(prefix + ".norm.weight"), weights_.at(prefix + ".norm.bias")}).tensor;
    }

    core::ExecutionContext & execution_;
    const TensorMap & weights_;
    ggml_context * state_context_;
    TensorMap & states_;
    std::vector<ggml_backend_buffer_t> & state_buffers_;
    std::string state_prefix_;
    Config config_;
    bool streaming_, first_;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context_;
    core::ModuleBuildContext ctx_;
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator_{nullptr, ggml_gallocr_free};
    ggml_cgraph * graph_ = nullptr;
    ggml_tensor * input_ = nullptr;
    std::array<ggml_tensor *, 3> outputs_;
    std::vector<ggml_tensor *> updates_;
    std::vector<std::pair<ggml_tensor *, std::vector<int32_t>>> indices_;
};
} // namespace

class Runtime::Impl {
public:
    Impl(std::shared_ptr<const Assets> assets, core::ExecutionContext & execution, assets::TensorStorageType storage, Config config)
        : execution(execution), config(config), store(execution.backend(), execution.backend_type(), "realtime_reuse.weights", 4 * 1024 * 1024),
          state_context(ggml_init({4 * 1024 * 1024, nullptr, true}), ggml_free) {
        if (config.exit_layer < 3 || config.exit_layer > 12 || config.lookahead_frames < 0 || config.lookahead_frames > 2)
            throw std::runtime_error("Real-time RE-USE exit_layer must be 3-12 and lookahead_frames 0-2");
        const auto & source = *assets->tensors;
        for (const auto & tensor : source.tensors()) {
            const auto & name = tensor.name;
            if (name.rfind("mask_decoder.", 0) == 0 || name.rfind("phase_decoder.", 0) == 0) continue;
            if (name.rfind("mask_decoder_list.", 0) == 0 || name.rfind("phase_decoder_list.", 0) == 0) {
                const auto start = name.find('.') + 1;
                if (std::stoi(name.substr(start, name.find('.', start) - start)) != config.exit_layer - 1) continue;
            }
            if (name.rfind("TSMamba.", 0) == 0 && std::stoi(name.substr(8, name.find('.', 8) - 8)) >= config.exit_layer) continue;
            if (name.rfind("dense_encoder.dense_conv_1_", 0) == 0 && name.rfind("dense_encoder.dense_conv_1_" + std::to_string(config.lookahead_frames + 1) + ".", 0) != 0) continue;
            if (name.size() >= 6 && name.compare(name.size() - 6, 6, ".A_log") == 0) {
                auto values = source.require_f32(name, tensor.shape);
                for (auto & value : values) value = -std::exp(value);
                weights.emplace(name, store.make_f32(TensorShape::from_dims({256, 16}), std::move(values)));
            } else if (tensor.shape.size() == 3) {
                weights.emplace(name, store.load_tensor_as_shape(source, name, assets::TensorStorageType::F32,
                    tensor.shape, TensorShape::from_dims({256, 4})));
            } else if (name.find(".up_conv2.0.conv.weight") != std::string::npos) {
                weights.emplace(name, store.load_tensor_as_shape(source, name, storage,
                    tensor.shape, TensorShape::from_dims({64, 64, 3, 1})));
            } else {
                weights.emplace(name, store.load_tensor(source, name,
                    tensor.shape.size() == 1 ? assets::TensorStorageType::F32 : storage, tensor.shape));
            }
        }
        store.upload();
        assets->tensors->release_storage();
        core::set_backend_threads(execution.backend(), execution.config().threads);
    }
    ~Impl() {
        graphs.clear();
        for (auto * buffer : state_buffers) ggml_backend_buffer_free(buffer);
    }
    core::ExecutionContext & execution;
    Config config;
    core::BackendWeightStore store;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> state_context;
    std::vector<ggml_backend_buffer_t> state_buffers;
    TensorMap weights, states;
    std::map<std::tuple<int64_t, int64_t, bool, bool>, std::unique_ptr<Graph>> graphs;
};

Runtime::Runtime(std::shared_ptr<const Assets> assets, core::ExecutionContext & execution, assets::TensorStorageType storage, Config config)
    : impl_(std::make_unique<Impl>(std::move(assets), execution, storage, config)) {}
Runtime::~Runtime() = default;
const Config & Runtime::config() const { return impl_->config; }
void Runtime::reset() {
    for (const auto & entry : impl_->states) ggml_backend_tensor_memset(entry.second.tensor, 0, 0, ggml_nbytes(entry.second.tensor));
}
std::array<std::vector<float>, 3> Runtime::infer(const std::vector<float> & features, int64_t frames, int64_t bins, bool streaming, bool first) {
    first = first && impl_->config.lookahead_frames != 0;
    auto & graph = impl_->graphs[{frames, bins, streaming, first}];
    if (!graph) graph = std::make_unique<Graph>(impl_->execution, impl_->weights, impl_->state_context.get(),
        impl_->states, impl_->state_buffers, impl_->config, frames, bins, streaming, first);
    return graph->run(features);
}
} // namespace engine::models::realtime_reuse
