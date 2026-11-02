#include "engine/framework/codecs/ltx_video_vae_runtime.h"

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/constant_tensor_cache.h"
#include "engine/framework/codecs/wan_video_vae_runtime.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <ggml-alloc.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>

namespace engine::codecs {
namespace {
using core::TensorShape;
using core::TensorValue;
using WeightValues = std::unordered_map<std::string, TensorValue>;

// One immutable topology per session shape. Only input contents change at run time.
class LtxVideoVaeGraph {
public:
    LtxVideoVaeGraph(core::ExecutionContext & execution, const assets::TensorSource & source,
        const WeightValues & weights, const LtxVideoVaeRuntimeConfig & runtime_config,
        bool encode, int64_t frames, int64_t height, int64_t width)
        : source(source), weights(weights), runtime_config(runtime_config), causal(encode),
          execution(execution), name(encode ? "ltx_video_vae.encode" : "ltx_video_vae.decode"),
          constants(execution.backend(), execution.config().threads, name + ".constants", 128 * 1024) {
        context.reset(ggml_init({32 * 1024 * 1024, nullptr, true}));
        if (!context) throw std::runtime_error("Cannot allocate LTX video VAE graph metadata");
        ctx = {context.get(), name.c_str(), execution.backend_type()};
        auto x = input("input", {encode ? 3 : 128, frames, height, width});
        if (encode) {
            x = patchify(x, 4);
            x = conv(x, "encoder.conv_in.conv");
            const int counts[] = {4, 0, 1, 3, 0, 1, 3, 0, 3, 4};
            for (int i = 0; i < 10; ++i) {
                const auto prefix = "encoder.down_blocks." + std::to_string(i);
                if (i == 1 || i == 4 || i == 7)
                    x = conv(x, prefix + ".conv", 2);
                else if (i == 2 || i == 5)
                    x = residual(x, prefix);
                else
                    for (int j = 0; j < counts[i]; ++j)
                        x = residual(x, prefix + ".res_blocks." + std::to_string(j));
            }
            x = conv(normalized_silu(x), "encoder.conv_out.conv");
            // Posterior sampling/statistics retain the original F32 graph and
            // caller-provided RNG noise, regardless of internal storage type.
            if (x.type != GGML_TYPE_F32)
                x = core::wrap_tensor(ggml_cast(ctx.ggml, x.tensor, GGML_TYPE_F32), x.shape, GGML_TYPE_F32);
            auto mean = modules::SliceModule({0, 0, 128}).build(ctx, x);
            auto log_variance = modules::SliceModule({0, 128, 1}).build(ctx, x);
            log_variance = core::wrap_tensor(ggml_clamp(ctx.ggml, log_variance.tensor, -30, 20), log_variance.shape);
            log_variance =
                modules::Bf16RoundModule().build(ctx, modules::ScaleBiasModule({0.5F, 0}).build(ctx, log_variance));
            auto stddev = modules::Bf16RoundModule().build(
                ctx, core::wrap_tensor(ggml_exp(ctx.ggml, log_variance.tensor), log_variance.shape));
            const auto noise =
                input("posterior_noise", {128, mean.shape.dims[1], mean.shape.dims[2], mean.shape.dims[3]});
            auto perturbation = modules::Bf16RoundModule().build(
                ctx, modules::MulModule().build(ctx, noise, modules::RepeatModule({noise.shape}).build(ctx, stddev)));
            mean = modules::Bf16RoundModule().build(ctx, modules::AddModule().build(ctx, mean, perturbation));
            auto stats = channel_stat("per_channel_statistics.mean-of-means");
            mean = modules::Bf16RoundModule().build(
                ctx, core::wrap_tensor(ggml_sub(ctx.ggml, mean.tensor, stats.tensor), mean.shape));
            stats = channel_stat("per_channel_statistics.std-of-means");
            mean = modules::Bf16RoundModule().build(
                ctx, core::wrap_tensor(ggml_div(ctx.ggml, mean.tensor, stats.tensor), mean.shape));
            finish(mean);
        } else {
            auto std = channel_stat("per_channel_statistics.std-of-means");
            auto mean = channel_stat("per_channel_statistics.mean-of-means");
            x = modules::Bf16RoundModule().build(
                ctx, modules::MulModule().build(ctx, x, modules::RepeatModule({x.shape}).build(ctx, std)));
            x = modules::Bf16RoundModule().build(ctx, modules::AddModule(false, true).build(ctx, x, mean));
            x = conv(x, "decoder.conv_in.conv");
            const int counts[] = {4, 3, 0, 3, 1, 0, 3, 1, 0, 4};
            for (int i = 0; i < 10; ++i) {
                const auto prefix = "decoder.up_blocks." + std::to_string(i);
                if (i == 2 || i == 5 || i == 8) {
                    x = conv(x, prefix + ".conv.conv");
                    x = depth_to_space(x, 2);
                    x = modules::SliceModule({1, 1, x.shape.dims[1] - 1}).build(ctx, x);
                } else if (i == 4 || i == 7)
                    x = residual(x, prefix);
                else
                    for (int j = 0; j < counts[i]; ++j)
                        x = residual(x, prefix + ".res_blocks." + std::to_string(j));
            }
            x = conv(normalized_silu(x), "decoder.conv_out.conv");
            x = unpatchify(x, 4);
            finish(core::ensure_backend_addressable_layout(ctx, x));
        }
    }

  private:
    TensorValue channel_stat(const std::string & name) {
        return modules::ReshapeModule({TensorShape::from_dims({128, 1, 1, 1})}).build(ctx, weights.at(name));
    }
    TensorValue normalized_silu(const TensorValue & x) {
        const auto channels = x.shape.dims[0];
        const auto gamma = constants.make_f32(TensorShape::from_dims({channels}),
                                             std::vector<float>(static_cast<size_t>(channels), 1.F));
        // Preserve F32 normalization/SiLU math, but store its fused result in
        // F16 for the native path instead of materializing F32 and casting it.
        WanVideoRMSNorm3dConfig config{channels, 1.e-8F,
            execution.backend_type() == core::BackendType::Cuda};
        config.fuse_silu = true;
        config.cuda_f16_output = runtime_config.cuda_f16_activations;
        auto value = WanVideoRMSNorm3dModule(config).build(ctx, x, {gamma});
        return runtime_config.cuda_f16_activations ? value : modules::Bf16RoundModule().build(ctx, value);
    }
    TensorValue conv(TensorValue x, const std::string & prefix, int stride = 1) {
        const auto source_shape = source.require_metadata(prefix + ".weight").shape;
        if (source_shape.size() != 4)
            throw std::runtime_error("LTX VAE requires normalized rank4 convolution weight: " + prefix);
        const int64_t k = source_shape[1];
        const bool pack_depth_padding = k > 1 && k % 2 != 0 && runtime_config.cuda_cudnn &&
            execution.backend_type() == core::BackendType::Cuda;
        if (!pack_depth_padding && k > 1 && x.type == GGML_TYPE_F16) {
            // Preserve HALF storage: gather would widen each padded frame to
            // F32 only for native convolution to convert it back to HALF.
            const auto original = core::ensure_backend_addressable_layout(ctx, x);
            x = original;
            const int64_t left = causal ? k - 1 : (k - 1) / 2;
            const int64_t right = causal ? 0 : (k - 1) / 2;
            if (left > 0) {
                auto first = modules::SliceModule({1, 0, 1}).build(ctx, original);
                if (left > 1) {
                    auto shape = first.shape;
                    shape.dims[1] = left;
                    first = modules::RepeatModule({shape}).build(ctx, first);
                }
                first = core::ensure_backend_addressable_layout(ctx, first);
                x = modules::ConcatModule({1, true}).build(ctx, first, x);
            }
            if (right > 0) {
                auto last = modules::SliceModule({1, original.shape.dims[1] - 1, 1}).build(ctx, original);
                if (right > 1) {
                    auto shape = last.shape;
                    shape.dims[1] = right;
                    last = modules::RepeatModule({shape}).build(ctx, last);
                }
                last = core::ensure_backend_addressable_layout(ctx, last);
                x = modules::ConcatModule({1, true}).build(ctx, x, last);
            }
        } else if (!pack_depth_padding && k > 1) {
            const auto channels = x.shape.dims[0], frames = x.shape.dims[1];
            const auto height = x.shape.dims[2], width = x.shape.dims[3];
            const int64_t left = causal ? k - 1 : (k - 1) / 2;
            const int64_t right = causal ? 0 : (k - 1) / 2;
            const int64_t padded_frames = frames + left + right;
            std::vector<int32_t> rows;
            rows.reserve(static_cast<size_t>(channels * padded_frames));
            for (int64_t channel = 0; channel < channels; ++channel)
                for (int64_t frame = 0; frame < padded_frames; ++frame)
                    rows.push_back(static_cast<int32_t>(channel * frames +
                        std::clamp<int64_t>(frame - left, 0, frames - 1)));
            const auto indices = constants.make_tensor(TensorShape::from_dims({channels * padded_frames}),
                                                       GGML_TYPE_I32, rows.data(), rows.size() * sizeof(int32_t));
            auto matrix = modules::ReshapeModule({TensorShape::from_dims({channels * frames, height * width})})
                .build(ctx, core::ensure_backend_addressable_layout(ctx, x));
            // get_rows materializes F32 even when its source matrix is F16;
            // preserve EmbeddingModule's actual result type through reshaping.
            x = modules::EmbeddingModule({channels * frames, height * width}).build(ctx, indices, matrix);
            x = modules::ReshapeModule({TensorShape::from_dims({channels, padded_frames, height, width})})
                .build(ctx, x);
        }
        modules::Conv3dConfig config;
        config.out_channels = weights.at(prefix + ".bias").shape.dims[0];
        if (source_shape[0] % config.out_channels != 0)
            throw std::runtime_error("LTX VAE flattened convolution channels mismatch: " + prefix);
        config.in_channels = source_shape[0] / config.out_channels;
        if (x.shape.dims[0] != config.in_channels)
            throw std::runtime_error("LTX VAE convolution input channels mismatch: " + prefix);
        config.kernel_depth = k;
        config.kernel_height = source_shape[2];
        config.kernel_width = source_shape[3];
        config.stride_depth = config.stride_height = config.stride_width = stride;
        if (pack_depth_padding) {
            config.padding_depth = (k - 1) / 2;
            config.cuda_replicate_depth_left = causal ? k - 1 : (k - 1) / 2;
        }
        config.padding_height = config.kernel_height / 2;
        config.padding_width = config.kernel_width / 2;
        modules::Conv3dWeights cw{weights.at(prefix + ".weight"), weights.at(prefix + ".bias")};
        config.cuda_cudnn = runtime_config.cuda_cudnn && execution.backend_type() == core::BackendType::Cuda;
        config.cuda_prepacked_weights = config.cuda_cudnn;
        config.cuda_f16_output = runtime_config.cuda_f16_activations;
        config.cuda_bf16_round = config.cuda_cudnn && !config.cuda_f16_output;
        auto output = modules::Conv3dModule(config).build(ctx, x, cw);
        return config.cuda_bf16_round || config.cuda_f16_output
            ? output : modules::Bf16RoundModule().build(ctx, output);
    }
    TensorValue residual(const TensorValue & x, const std::string & prefix) {
        auto y = conv(normalized_silu(x), prefix + ".conv1.conv");
        y = conv(normalized_silu(y), prefix + ".conv2.conv");
        auto skip = x;
        if (source.has_tensor(prefix + ".conv_shortcut.weight")) {
            skip = modules::TransposeModule({{1, 2, 3, 0}, 4}).build(ctx, skip);
            modules::NormWeights norm_weights{weights.at(prefix + ".norm3.norm.weight"),
                                              weights.at(prefix + ".norm3.norm.bias")};
            modules::NormConfig norm_config{skip.shape.last_dim(), 1.e-6F, true, true};
            norm_config.cuda_f16_output = runtime_config.cuda_f16_activations && skip.type == GGML_TYPE_F16;
            if (norm_config.cuda_f16_output)
                skip = core::ensure_backend_addressable_layout(ctx, skip);
            skip = modules::LayerNormModule(norm_config).build(ctx, skip, norm_weights);
            if (!runtime_config.cuda_f16_activations)
                skip = modules::Bf16RoundModule().build(ctx, skip);
            skip = modules::TransposeModule({{3, 0, 1, 2}, 4}).build(ctx, skip);
            skip = conv(skip, prefix + ".conv_shortcut");
        }
        const bool f16_add = runtime_config.cuda_f16_activations &&
            skip.type == GGML_TYPE_F16 && y.type == GGML_TYPE_F16;
        auto output = modules::AddModule(f16_add).build(ctx, skip, y);
        return runtime_config.cuda_f16_activations ? output : modules::Bf16RoundModule().build(ctx, output);
    }
    TensorValue patchify(TensorValue x, int64_t p) {
        const auto c = x.shape.dims[0], t = x.shape.dims[1], h = x.shape.dims[2], w = x.shape.dims[3];
        // Upstream causal_video_autoencoder.py:1268 with temporal patch size 1:
        // [c,f,h*q,w*r] -> [c*r*q,f,h,w], packed channel = (c*p + r)*p + q.
        // A = c*f*(H/p): [A,q,w,r] -> [r,q,A,w] -> [c,r*q,f*h,w].
        x = modules::ReshapeModule({TensorShape::from_dims({c * t * (h / p), p, w / p, p})})
                .build(ctx, x);
        x = modules::TransposeModule({{3, 1, 0, 2}, 4}).build(ctx, x);
        x = modules::ReshapeModule({TensorShape::from_dims({p * p, c, t * (h / p), w / p})})
                .build(ctx, core::ensure_backend_addressable_layout(ctx, x));
        x = modules::TransposeModule({{1, 0, 2, 3}, 4}).build(ctx, x);
        return modules::ReshapeModule({TensorShape::from_dims({c * p * p, t, h / p, w / p})})
            .build(ctx, core::ensure_backend_addressable_layout(ctx, x));
    }
    TensorValue unpatchify(TensorValue x, int64_t p) {
        const auto c = x.shape.dims[0] / (p * p), t = x.shape.dims[1];
        const auto h = x.shape.dims[2], w = x.shape.dims[3];
        // Exact inverse of upstream :1289: [c,r*q,f*h,w] -> [r,q,c*f*h,w]
        // -> [c*f*h,q,w,r], restoring output coordinates (h*p + q, w*p + r).
        x = modules::ReshapeModule({TensorShape::from_dims({c, p * p, t * h, w})})
                .build(ctx, x);
        x = modules::TransposeModule({{1, 0, 2, 3}, 4}).build(ctx, x);
        x = modules::ReshapeModule({TensorShape::from_dims({p, p, c * t * h, w})})
                .build(ctx, core::ensure_backend_addressable_layout(ctx, x));
        x = modules::TransposeModule({{2, 1, 3, 0}, 4}).build(ctx, x);
        return modules::ReshapeModule({TensorShape::from_dims({c, t, h * p, w * p})})
            .build(ctx, core::ensure_backend_addressable_layout(ctx, x));
    }
    TensorValue depth_to_space(TensorValue x, int64_t p) {
        auto c = x.shape.dims[0], t = x.shape.dims[1], h = x.shape.dims[2], w = x.shape.dims[3];
        x = modules::ReshapeModule({TensorShape::from_dims({c / p, p, t * h, w})})
                .build(ctx, core::ensure_backend_addressable_layout(ctx, x));
        x = modules::TransposeModule({{0, 2, 3, 1}, 4}).build(ctx, x);
        x = modules::ReshapeModule({TensorShape::from_dims({c / (p * p), p, t * h, w * p})})
                .build(ctx, core::ensure_backend_addressable_layout(ctx, x));
        x = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, x);
        x = modules::ReshapeModule({TensorShape::from_dims({c / (p * p * p), p, t, h * p * w * p})})
                .build(ctx, core::ensure_backend_addressable_layout(ctx, x));
        x = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, x);
        return modules::ReshapeModule({TensorShape::from_dims({c / (p * p * p), t * p, h * p, w * p})})
            .build(ctx, core::ensure_backend_addressable_layout(ctx, x));
    }
    const assets::TensorSource & source;
    const WeightValues & weights;
    LtxVideoVaeRuntimeConfig runtime_config;
    bool causal;

public:
    ~LtxVideoVaeGraph() {
        host_plan.reset();
        if (graph) core::release_backend_graph_resources(execution.backend(), graph, true);
    }
    LtxVideoVaeGraph(const LtxVideoVaeGraph &) = delete;
    LtxVideoVaeGraph & operator=(const LtxVideoVaeGraph &) = delete;

    void set_backend_input(const TensorValue & value) {
        core::validate_shape(value, inputs.at("input").shape, "LTX video VAE input");
        if (value.tensor->type != GGML_TYPE_F32 || !ggml_is_contiguous(value.tensor) ||
            !ggml_are_same_shape(value.tensor, inputs.at("input").tensor) || !value.tensor->buffer)
            throw std::runtime_error("LTX video VAE backend input must be allocated contiguous F32 with canonical layout");
        ggml_backend_tensor_copy(value.tensor, inputs.at("input").tensor);
    }

    TensorValue run() {
        const auto started = std::chrono::steady_clock::now();
        if (core::compute_graph(execution, graph, host_plan, name.c_str()) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("LTX video VAE execution failed: " + name);
        debug::timing_log_scalar(name + ".graph.compute_ms", debug::elapsed_ms(started));
        return output;
    }

    core::ExecutionContext & execution;
    std::string name;
    core::ConstantTensorCache constants;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context{nullptr, ggml_free};
    core::ModuleBuildContext ctx;
    ggml_cgraph * graph = nullptr;
    core::HostGraphPlan host_plan;
    std::map<std::string, TensorValue> inputs;
    TensorValue output;

private:
    TensorValue input(const std::string & key, std::initializer_list<int64_t> shape) {
        auto value = core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims(shape));
        ggml_set_name(value.tensor, (name + "." + key).c_str());
        ggml_set_input(value.tensor);
        // Keep input storage reserved throughout the combined allocation plan.
        ggml_set_output(value.tensor);
        inputs.emplace(key, value);
        return value;
    }
    void finish(const TensorValue & value) {
        // Keep decoder layout transforms in their internal type; cast only at
        // the public output boundary. Encoder posterior outputs are already F32.
        output = value.type == GGML_TYPE_F32 ? value : core::wrap_tensor(
            ggml_cast(ctx.ggml, value.tensor, GGML_TYPE_F32), value.shape, GGML_TYPE_F32);
        ggml_set_name(output.tensor, (name + ".output").c_str());
        // Retain the result through graph completion. Other cached graphs share
        // scratch storage, so the next encode/decode invalidates this borrowing.
        ggml_set_output(output.tensor);
        graph = ggml_new_graph_custom(context.get(), 32768, false);
        ggml_build_forward_expand(graph, output.tensor);
        constants.ensure_uploaded();
        core::validate_backend_graph_supported(execution.backend(), graph, name.c_str());
        debug::timing_log_context_reservation(name, context.get());
    }
};
} // namespace

struct LtxVideoVaeRuntime::Impl {
    core::ExecutionContext & execution;
    std::shared_ptr<const assets::TensorSource> source;
    LtxVideoVaeRuntimeConfig config;
    core::BackendWeightStore store;
    WeightValues weights;
    // Destruction order: graphs/plans, combined metadata, arena, weights, source.
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> arena{nullptr, ggml_gallocr_free};
    std::unique_ptr<ggml_context, decltype(&ggml_free)> allocation_context{nullptr, ggml_free};
    ggml_cgraph * allocation_graph = nullptr;
    using Key = std::tuple<bool, int64_t, int64_t, int64_t>; // encoder, T, pixel H, pixel W
    std::map<Key, std::unique_ptr<LtxVideoVaeGraph>> graphs;
    bool allocated = false;

    Impl(core::ExecutionContext & execution, std::shared_ptr<const assets::TensorSource> source,
        LtxVideoVaeRuntimeConfig config)
        : execution(execution), source(std::move(source)), config(config),
          store(execution.backend(), execution.backend_type(), "ltx_video_vae.weights", 4 * 1024 * 1024) {
        if (!this->source) throw std::runtime_error("LTX video VAE requires a tensor source");
        if (config.cuda_f16_activations &&
            (!config.cuda_cudnn || execution.backend_type() != core::BackendType::Cuda))
            throw std::runtime_error("LTX video VAE F16 activations require native cuDNN on CUDA");
        // Match the original bindings: normalized convolution kernels in F16,
        // matrix parameters in native storage, norms/statistics in F32.
        for (const auto & tensor : this->source->tensors()) {
            const auto & shape = tensor.shape;
            if (config.cuda_cudnn && execution.backend_type() == core::BackendType::Cuda && shape.size() == 4) {
                const int64_t kernel_elements = shape[1] * shape[2] * shape[3];
                if (kernel_elements > 1) {
                    const std::string suffix = ".weight";
                    if (tensor.name.size() < suffix.size() ||
                        tensor.name.compare(tensor.name.size() - suffix.size(), suffix.size(), suffix) != 0)
                        throw std::runtime_error("LTX video VAE rank4 convolution tensor must end in .weight: " + tensor.name);
                    const auto bias = this->source->require_metadata(
                        tensor.name.substr(0, tensor.name.size() - suffix.size()) + ".bias");
                    if (bias.shape.size() != 1 || bias.shape[0] <= 0 || shape[0] % bias.shape[0] != 0)
                        throw std::runtime_error("LTX video VAE prepacked convolution channels mismatch: " + tensor.name);
                    const int64_t out_channels = bias.shape[0];
                    const int64_t in_channels = shape[0] / out_channels;
                    const auto logical_shape = TensorShape::from_dims({shape[0], shape[1], shape[2], shape[3]});
                    std::vector<std::byte> packed;
                    {
                        const auto data = this->source->require_tensor(tensor.name, assets::TensorStorageType::F16, shape);
                        packed.resize(data.bytes.size());
                        // Private native format: [OC,IC,K] -> [OC,K,IC]. Keep
                        // CF metadata for module geometry; the explicit operator
                        // flag identifies physical ODHWI bytes to the backend.
                        for (int64_t oc = 0; oc < out_channels; ++oc)
                            for (int64_t k = 0; k < kernel_elements; ++k)
                                for (int64_t ic = 0; ic < in_channels; ++ic) {
                                    const size_t from = static_cast<size_t>((oc * in_channels + ic) * kernel_elements + k);
                                    const size_t to = static_cast<size_t>((oc * kernel_elements + k) * in_channels + ic);
                                    std::memcpy(packed.data() + to * sizeof(ggml_fp16_t),
                                        data.bytes.data() + from * sizeof(ggml_fp16_t), sizeof(ggml_fp16_t));
                                }
                    }
                    // Release converted CF bytes before make_tensor copies the
                    // packed payload into its pending upload. No raw GPU copy.
                    weights.emplace(tensor.name, store.make_tensor(logical_shape, GGML_TYPE_F16,
                        packed.data(), packed.size()));
                    continue;
                }
                // K==1 is already ODHWI; retain the original lazy F16 upload.
            }
            weights.emplace(tensor.name, store.load_tensor(*this->source, tensor.name,
                shape.size() == 2 ? assets::TensorStorageType::Native :
                (shape.size() == 4 ? assets::TensorStorageType::F16 : assets::TensorStorageType::F32), shape));
        }
        // make_tensor's pending packed bytes are freed by upload; only the
        // single packed backend weight buffer persists for session graphs.
        store.upload();
        arena.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend())));
        if (!arena) throw std::runtime_error("Cannot create LTX video VAE allocator");
    }

    void prepare(int64_t height, int64_t width, const std::vector<int64_t> & encoder_frames,
        const std::vector<int64_t> & decoder_frames) {
        if (height <= 0 || width <= 0 || height % 32 || width % 32)
            throw std::runtime_error("LTX video VAE pixel dimensions must be positive multiples of 32");
        for (const auto frames : encoder_frames)
            if (frames <= 0) throw std::runtime_error("LTX video VAE encoder frame count must be positive");
        for (const auto frames : decoder_frames)
            if (frames <= 0) throw std::runtime_error("LTX video VAE decoder latent frame count must be positive");

        // Construct new topologies without touching addresses of existing captures.
        // Keep each successful topology even if a subsequent shape fails to build.
        const auto add = [&](bool encoder, int64_t frames) {
            const Key key{encoder, frames, height, width};
            if (graphs.find(key) != graphs.end()) return;
            auto graph = std::make_unique<LtxVideoVaeGraph>(execution, *source, weights, config,
                encoder, frames, encoder ? height : height / 32, encoder ? width : width / 32);
            graphs.emplace(key, std::move(graph));
            allocated = false;
        };
        for (const auto frames : encoder_frames) add(true, frames);
        for (const auto frames : decoder_frames) add(false, frames);
        if (allocated || graphs.empty()) return;

        ggml_backend_synchronize(execution.backend());
        // No backend capture or host plan may survive a shared arena allocation.
        size_t nodes = 0;
        for (auto & entry : graphs) {
            auto & graph = *entry.second;
            graph.host_plan.reset();
            core::release_backend_graph_resources(execution.backend(), graph.graph, true);
            // gallocr treats bound tensors as externally allocated. Detach only
            // this graph's descriptors before replanning; source weights live in
            // BackendWeightStore's separate context and retain their bindings.
            for (auto * tensor = ggml_get_first_tensor(graph.context.get()); tensor;
                 tensor = ggml_get_next_tensor(graph.context.get(), tensor)) {
                tensor->data = nullptr;
                tensor->buffer = nullptr;
                tensor->extra = nullptr;
            }
            nodes += ggml_graph_size(graph.graph);
        }
        allocation_graph = nullptr;
        allocation_context.reset(ggml_init({ggml_graph_overhead_custom(nodes, false), nullptr, true}));
        if (!allocation_context) throw std::runtime_error("Cannot allocate LTX video VAE combined plan metadata");
        allocation_graph = ggml_new_graph_custom(allocation_context.get(), nodes, false);
        for (const auto & entry : graphs) ggml_build_forward_expand(allocation_graph, entry.second->output.tensor);
        if (!ggml_gallocr_reserve(arena.get(), allocation_graph) ||
            !ggml_gallocr_alloc_graph(arena.get(), allocation_graph))
            throw std::runtime_error("Cannot allocate LTX video VAE shared graphs");
        for (auto & entry : graphs)
            core::prepare_host_graph_plan(execution, entry.second->graph, entry.second->host_plan);
        debug::timing_log_context_reservation("ltx_video_vae.allocation_plan", allocation_context.get());
        debug::timing_log_scalar("ltx_video_vae.graph_buffer_mb",
            ggml_gallocr_get_buffer_size(arena.get(), 0) / 1048576.0);
        debug::timing_log_scalar("ltx_video_vae.cached_graphs", static_cast<int64_t>(graphs.size()));
        allocated = true;
    }

    template <typename Input>
    TensorValue run(bool encoder, const Input & input, const std::vector<float> * noise,
        int64_t frames, int64_t height, int64_t width) {
        const auto started = std::chrono::steady_clock::now();
        const auto found = graphs.find(Key{encoder, frames, height, width});
        if (!allocated || found == graphs.end())
            throw std::runtime_error("LTX video VAE shape has not been prepared");
        auto & graph = *found->second;
        if (noise && static_cast<int64_t>(noise->size()) != graph.inputs.at("posterior_noise").shape.num_elements())
            throw std::runtime_error("LTX video VAE posterior noise size does not match encoded latent shape");
        const auto upload_started = std::chrono::steady_clock::now();
        if constexpr (std::is_same_v<Input, TensorValue>) graph.set_backend_input(input);
        else core::write_tensor_f32(graph.inputs.at("input"), input);
        if (noise) core::write_tensor_f32(graph.inputs.at("posterior_noise"), *noise);
        debug::timing_log_scalar(graph.name + ".input_upload_ms", debug::elapsed_ms(upload_started));
        auto output = graph.run();
        debug::timing_log_scalar(graph.name + ".wall_ms", debug::elapsed_ms(started));
        debug::timing_log_scalar("ltx_video_vae.session.wall_ms", debug::elapsed_ms(started));
        return output;
    }
};

LtxVideoVaeRuntime::LtxVideoVaeRuntime(core::ExecutionContext & execution,
    std::shared_ptr<const assets::TensorSource> source, LtxVideoVaeRuntimeConfig config)
    : impl_(std::make_unique<Impl>(execution, std::move(source), config)) {}
LtxVideoVaeRuntime::~LtxVideoVaeRuntime() = default;
LtxVideoVaeRuntime::LtxVideoVaeRuntime(LtxVideoVaeRuntime &&) noexcept = default;
LtxVideoVaeRuntime & LtxVideoVaeRuntime::operator=(LtxVideoVaeRuntime &&) noexcept = default;

void LtxVideoVaeRuntime::prepare(int64_t height, int64_t width,
    const std::vector<int64_t> & encoder_frame_counts, const std::vector<int64_t> & decoder_latent_frame_counts) {
    impl_->prepare(height, width, encoder_frame_counts, decoder_latent_frame_counts);
}
core::TensorValue LtxVideoVaeRuntime::encode(const std::vector<float> & pixels,
    const std::vector<float> & posterior_noise, int64_t frames, int64_t height, int64_t width) {
    return impl_->run(true, pixels, &posterior_noise, frames, height, width);
}
core::TensorValue LtxVideoVaeRuntime::encode(const core::TensorValue & pixels,
    const std::vector<float> & posterior_noise, int64_t frames, int64_t height, int64_t width) {
    return impl_->run(true, pixels, &posterior_noise, frames, height, width);
}
core::TensorValue LtxVideoVaeRuntime::decode(const std::vector<float> & latents,
    int64_t latent_frames, int64_t height, int64_t width) {
    return impl_->run(false, latents, nullptr, latent_frames, height, width);
}
core::TensorValue LtxVideoVaeRuntime::decode(const core::TensorValue & latents,
    int64_t latent_frames, int64_t height, int64_t width) {
    return impl_->run(false, latents, nullptr, latent_frames, height, width);
}

} // namespace engine::codecs
