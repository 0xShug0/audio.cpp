#include "engine/framework/vision/runtimes/liveportrait_runtime.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/prepared_graph.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/feed_forward_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <map>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::modules {
using core::TensorShape;
using core::TensorValue;

class LivePortraitRuntime::Impl {
    public:
    core::ExecutionContext & execution;
    LivePortraitRuntimeConfig config;
    core::BackendWeightStore weights;
    std::map<std::string, TensorValue> tensors;
    struct Graph {
        std::unique_ptr<core::PreparedGraph> owner;
        TensorValue input, output;
    } appearance, decoder, landmarks;
    struct MotionGraph {
        std::unique_ptr<core::PreparedGraph> owner;
        TensorValue input;
        std::array<TensorValue, 7> output;
    } motion;
    struct WarpGraph {
        std::unique_ptr<core::PreparedGraph> owner;
        TensorValue input, source, driving, grid, background, output;
        core::PreparedGraph::GraphId source_graph = 0;
    } warp, renderer;
    struct StitchGraph {
        std::unique_ptr<core::PreparedGraph> owner;
        TensorValue source, driving, output;
    } stitch;
    bool source_ready = false;
    TensorValue one;

    Impl(const assets::TensorSource & source, core::ExecutionContext & e, LivePortraitRuntimeConfig c)
        : execution(e), config(c), weights(e.backend(), e.backend_type(), "liveportrait.weights", 16ull << 20) {
        if (config.f16_activations && !config.cudnn)
            throw std::invalid_argument("LivePortrait F16 activations require explicit cuDNN convolution");
        for (const auto & tensor : source.tensors()) {
            // Autocast applies to convolution/linear weights, not learned
            // pointwise GRN gamma/beta or normalization parameters.
            const bool keep_f32 = tensor.shape.size() < 2 || tensor.name.find(".norm.") != std::string::npos ||
                                  tensor.name.rfind("landmarks.", 0) == 0 || tensor.name.size() < 7 ||
                                  tensor.name.compare(tensor.name.size() - 7, 7, ".weight") != 0;
            tensors.emplace(tensor.name, binding::tensor_from_named_source(weights, source, tensor.name,
                                                                           keep_f32 ? assets::TensorStorageType::F32
                                                                                    : config.weight_type));
        }
        if (config.f16_activations)
            one = weights.make_from_f32(TensorShape::from_dims({1, 1, 1, 1}), assets::TensorStorageType::F16, {1.f});
        weights.upload();
    }

    TensorValue convolution(core::ModuleBuildContext & ctx, const TensorValue & input, const std::string & name,
                            int stride = 1, int padding = -1, bool volume = false, bool f16_output = true,
                            bool use_cudnn = true) const {
        const auto value = core::wrap_tensor(input.tensor, input.shape, input.tensor->type);
        const auto & w = tensors.at(name + ".weight");
        const auto bias = tensors.find(name + ".bias");
        const std::optional<TensorValue> b = bias == tensors.end() ? std::nullopt : std::make_optional(bias->second);
        if (volume) {
            // Converter packs O,I,D,H,W as (O*I),D,H,W, preserving storage order.
            const auto channels = input.shape.dims[0];
            Conv3dConfig c;
            c.in_channels = channels;
            c.out_channels = w.shape.dims[0] / channels;
            c.kernel_depth = w.shape.dims[1];
            c.kernel_height = w.shape.dims[2];
            c.kernel_width = w.shape.dims[3];
            c.stride_height = c.stride_width = stride;
            c.padding_depth = padding < 0 ? static_cast<int>(c.kernel_depth / 2) : padding;
            c.padding_height = padding < 0 ? static_cast<int>(c.kernel_height / 2) : padding;
            c.padding_width = padding < 0 ? static_cast<int>(c.kernel_width / 2) : padding;
            c.use_bias = b.has_value();
            c.cuda_cudnn = config.cudnn && use_cudnn;
            c.cuda_f16_output = config.f16_activations && f16_output && use_cudnn;
            return Conv3dModule(c).build(ctx, value, {w, b});
        }
        Conv2dConfig c;
        c.in_channels = w.shape.dims[1];
        c.out_channels = w.shape.dims[0];
        c.kernel_height = w.shape.dims[2];
        c.kernel_width = w.shape.dims[3];
        c.stride_height = c.stride_width = stride;
        c.padding_height = padding < 0 ? static_cast<int>(c.kernel_height / 2) : padding;
        c.padding_width = padding < 0 ? static_cast<int>(c.kernel_width / 2) : padding;
        c.use_bias = b.has_value();
        c.cuda_cudnn = config.cudnn && use_cudnn;
        c.cuda_f16_output = config.f16_activations && f16_output && use_cudnn;
        return Conv2dModule(c).build(ctx, value, {w, b});
    }

    TensorValue affine(core::ModuleBuildContext & ctx, const TensorValue & input, const std::string & name,
                       bool volume = false) const {
        const auto channels = input.shape.dims[volume ? 0 : 1];
        const auto shape =
            volume ? TensorShape::from_dims({channels, 1, 1, 1}) : TensorShape::from_dims({1, channels, 1, 1});
        auto value = core::wrap_tensor(ggml_cast(ctx.ggml, input.tensor, GGML_TYPE_F32), input.shape);
        auto x = MulModule(true).build(ctx, value, core::reshape_tensor(ctx, tensors.at(name + ".scale"), shape));
        x = AddModule(false, true).build(ctx, x, core::reshape_tensor(ctx, tensors.at(name + ".shift"), shape));
        if (config.f16_activations)
            x = core::wrap_tensor(ggml_cast(ctx.ggml, x.tensor, GGML_TYPE_F16), x.shape, GGML_TYPE_F16);
        return x;
    }

    TensorValue normalized_convolution(core::ModuleBuildContext & ctx, const TensorValue & input,
                                       const std::string & prefix, bool volume = false, float slope = 0) const {
        auto x = affine(ctx, convolution(ctx, input, prefix + ".conv", 1, -1, volume), prefix + ".norm", volume);
        return slope ? LeakyReluModule({slope}).build(ctx, x) : ReluModule().build(ctx, x);
    }

    TensorValue upsample(core::ModuleBuildContext & ctx, const TensorValue & input, int64_t height,
                         int64_t width) const {
        if (input.shape.dims[2] == height && input.shape.dims[3] == width)
            return input;
        auto shape = input.shape;
        shape.dims[2] = height;
        shape.dims[3] = width;
        auto * source =
            input.tensor->type == GGML_TYPE_F32 ? input.tensor : ggml_cast(ctx.ggml, input.tensor, GGML_TYPE_F32);
        auto * output = ggml_interpolate(ctx.ggml, source, width, height, input.tensor->ne[2], input.tensor->ne[3],
                                         GGML_SCALE_MODE_NEAREST);
        if (input.tensor->type == GGML_TYPE_F16)
            output = ggml_cast(ctx.ggml, output, GGML_TYPE_F16);
        return core::wrap_tensor(output, shape, output->type);
    }

    TensorValue spade(core::ModuleBuildContext & ctx, const TensorValue & input, const TensorValue & segmentation,
                      const std::string & prefix) const {
        // InstanceNorm2d is row LayerNorm over H*W, without affine parameters.
        const auto spatial = input.shape.dims[2] * input.shape.dims[3];
        const bool native_f16 = config.f16_activations && ctx.backend_type == core::BackendType::Cuda;
        auto * data = native_f16 ? input.tensor : ggml_cast(ctx.ggml, input.tensor, GGML_TYPE_F32);
        auto value = core::wrap_tensor(data, input.shape, data->type);
        auto flat = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, value),
                                         TensorShape::from_dims({input.shape.dims[1], spatial}));
        NormConfig normalization{spatial, 1e-5f, false, false};
        normalization.cuda_f16_output = native_f16;
        auto norm = LayerNormModule(normalization).build(ctx, flat, {});
        norm = core::reshape_tensor(ctx, norm, input.shape);
        if (config.f16_activations && norm.tensor->type != GGML_TYPE_F16)
            norm = core::wrap_tensor(ggml_cast(ctx.ggml, norm.tensor, GGML_TYPE_F16), norm.shape, GGML_TYPE_F16);
        const auto hidden = ReluModule().build(ctx, convolution(ctx, segmentation, prefix + ".mlp_shared.0"));
        const auto gamma = convolution(ctx, hidden, prefix + ".mlp_gamma");
        const auto scale = config.f16_activations ? AddModule(true, true).build(ctx, gamma, one)
                                                  : ScaleBiasModule({1, 1}).build(ctx, gamma);
        auto x = MulModule().build(ctx, norm, scale);
        x = core::wrap_tensor(x.tensor, x.shape, x.tensor->type);
        x = AddModule(config.f16_activations).build(ctx, x, convolution(ctx, hidden, prefix + ".mlp_beta"));
        return x;
    }

    TensorValue spade_block(core::ModuleBuildContext & ctx, const TensorValue & input, const TensorValue & segmentation,
                            const std::string & prefix) const {
        // Every SPADE branch uses the same segmentation at this resolution.
        const auto seg = upsample(ctx, segmentation, input.shape.dims[2], input.shape.dims[3]);
        auto shortcut = input;
        if (tensors.count(prefix + ".conv_s.weight"))
            shortcut = convolution(ctx, spade(ctx, input, seg, prefix + ".norm_s"), prefix + ".conv_s");
        auto x = convolution(ctx, LeakyReluModule({.2f}).build(ctx, spade(ctx, input, seg, prefix + ".norm_0")),
                             prefix + ".conv_0");
        x = convolution(ctx, LeakyReluModule({.2f}).build(ctx, spade(ctx, x, seg, prefix + ".norm_1")),
                        prefix + ".conv_1");
        x = AddModule(config.f16_activations).build(ctx, shortcut, x);
        return x;
    }

    TensorValue sample_volume(core::ModuleBuildContext & ctx, const TensorValue & feature,
                              const TensorValue & coordinates) {
        const auto channels = feature.shape.dims[0], depth = feature.shape.dims[1];
        const auto height = feature.shape.dims[2], width = feature.shape.dims[3];
        const auto count = coordinates.shape.dims[0] * coordinates.shape.dims[1];
        const int64_t sizes[] = {width, height, depth};
        std::array<TensorValue, 3> lower, fraction;
        for (int axis = 0; axis < 3; ++axis) {
            auto x = SliceModule({2, axis, 1}).build(ctx, coordinates);
            x = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, x),
                                     TensorShape::from_dims({count}));
            x = ScaleBiasModule({static_cast<float>(sizes[axis]) / 2, static_cast<float>(sizes[axis] - 1) / 2})
                    .build(ctx, x);
            lower[axis] = core::wrap_tensor(ggml_floor(ctx.ggml, x.tensor), x.shape);
            fraction[axis] = core::wrap_tensor(ggml_sub(ctx.ggml, x.tensor, lower[axis].tensor), x.shape);
        }
        auto table = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, feature),
                                          TensorShape::from_dims({channels, depth * height * width}));
        table = core::ensure_backend_addressable_layout(ctx, TransposeModule({{1, 0, 2, 3}, 2}).build(ctx, table));
        // Each axis has only two neighbors. Share their bounds/weights across
        // corners instead of rebuilding identical arithmetic eight times.
        std::array<std::array<TensorValue, 2>, 3> positions, axis_weights;
        const int64_t strides[] = {1, width, width * height};
        for (int axis = 0; axis < 3; ++axis)
            for (int high = 0; high < 2; ++high) {
                auto position = high ? ScaleBiasModule({1, 1}).build(ctx, lower[axis]) : lower[axis];
                auto weight = high ? fraction[axis] : ScaleBiasModule({-1, 1}).build(ctx, fraction[axis]);
                // Indices are integers. Half-integer comparisons express 0 <= i < size
                // without relying on a backend's STEP value at exactly zero.
                const auto low_valid = ScaleBiasModule({1, .5f}).build(ctx, position);
                const auto high_valid =
                    ScaleBiasModule({-1, static_cast<float>(sizes[axis]) - .5f}).build(ctx, position);
                const auto valid =
                    MulModule().build(ctx, core::wrap_tensor(ggml_step(ctx.ggml, low_valid.tensor), position.shape),
                                      core::wrap_tensor(ggml_step(ctx.ggml, high_valid.tensor), position.shape));
                axis_weights[axis][high] = MulModule().build(ctx, weight, valid);
                position = core::wrap_tensor(
                    ggml_clamp(ctx.ggml, ggml_dup(ctx.ggml, position.tensor), 0, static_cast<float>(sizes[axis] - 1)),
                    position.shape);
                positions[axis][high] = ScaleBiasModule({static_cast<float>(strides[axis]), 0}).build(ctx, position);
            }
        TensorValue output;
        for (int corner = 0; corner < 8; ++corner) {
            TensorValue index, weight;
            for (int axis = 0; axis < 3; ++axis) {
                const auto high = (corner >> axis) & 1;
                weight = weight.valid() ? MulModule().build(ctx, weight, axis_weights[axis][high])
                                        : axis_weights[axis][high];
                index = index.valid() ? AddModule().build(ctx, index, positions[axis][high]) : positions[axis][high];
            }
            const auto ids =
                core::wrap_tensor(ggml_cast(ctx.ggml, index.tensor, GGML_TYPE_I32), index.shape, GGML_TYPE_I32);
            const auto gathered = EmbeddingModule({depth * height * width, channels}).build(ctx, ids, table);
            weight = core::reshape_tensor(ctx, weight, TensorShape::from_dims({count, 1}));
            const auto weighted = MulModule(true).build(ctx, gathered, weight);
            output = output.valid() ? AddModule().build(ctx, output, weighted) : weighted;
        }
        output = core::reshape_tensor(
            ctx, output, TensorShape::from_dims({coordinates.shape.dims[0], coordinates.shape.dims[1], channels}));
        output = core::ensure_backend_addressable_layout(ctx, TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, output));
        return core::reshape_tensor(
            ctx, output, TensorShape::from_dims({coordinates.shape.dims[0] * channels, depth, height, width}));
    }

    TensorValue gaussian(core::ModuleBuildContext & ctx, const TensorValue & points, const TensorValue & grid) const {
        const auto delta = core::wrap_tensor(
            ggml_sub(ctx.ggml, RepeatModule({TensorShape::from_dims({21, 65536, 3})}).build(ctx, grid).tensor,
                     RepeatModule({TensorShape::from_dims({21, 65536, 3})})
                         .build(ctx, core::reshape_tensor(ctx, points, TensorShape::from_dims({21, 1, 3})))
                         .tensor),
            TensorShape::from_dims({21, 65536, 3}));
        auto distance = ReduceSumModule({2}).build(ctx, MulModule().build(ctx, delta, delta));
        distance = ScaleBiasModule({-50, 0}).build(ctx, distance);
        return core::wrap_tensor(ggml_exp(ctx.ggml, distance.tensor), distance.shape);
    }

    void prepare_warp(WarpGraph & warp, bool render = false) {
        if (warp.owner)
            return;
        warp.owner = std::make_unique<core::PreparedGraph>(
            execution, core::PreparedGraphConfig{render ? "liveportrait.render" : "liveportrait.warp", 96ull << 20,
                                                 49152, 2ull << 20});
        auto & ctx = warp.owner->build_context();
        auto & persistent = warp.owner->persistent_build_context();
        warp.grid = core::make_tensor(persistent, GGML_TYPE_F32, TensorShape::from_dims({1, 65536, 3}));
        warp.background = core::make_tensor(persistent, GGML_TYPE_F32, TensorShape::from_dims({1, 65536, 3}));
        warp.input =
            core::make_tensor(render ? persistent : ctx, GGML_TYPE_F32, TensorShape::from_dims({32, 16, 64, 64}));
        warp.source = core::make_tensor(render ? persistent : ctx, GGML_TYPE_F32, TensorShape::from_dims({21, 3}));
        warp.driving = core::make_tensor(ctx, GGML_TYPE_F32, warp.source.shape);
        for (auto value : {warp.input, warp.source, warp.driving})
            ggml_set_input(value.tensor);
        auto compressed = convolution(ctx, warp.input, "warp.dense_motion_network.compress", 1, 0, true);
        compressed = ReluModule().build(ctx, affine(ctx, compressed, "warp.dense_motion_network.norm", true));
        auto source_heatmap = gaussian(ctx, warp.source, warp.grid);
        if (render) {
            // Source-only results persist across frame executions. The source entry
            // runs once per set_source(), using the same math and shared scratch.
            warp.source_graph = warp.owner->add_graph("source");
            auto cached = core::make_tensor(persistent, compressed.tensor->type, compressed.shape);
            ggml_build_forward_expand(warp.owner->graph(warp.source_graph),
                                      ggml_cpy(ctx.ggml, compressed.tensor, cached.tensor));
            compressed = cached;
            cached = core::make_tensor(persistent, source_heatmap.tensor->type, source_heatmap.shape);
            ggml_build_forward_expand(warp.owner->graph(warp.source_graph),
                                      ggml_cpy(ctx.ggml, source_heatmap.tensor, cached.tensor));
            source_heatmap = cached;
        }
        const auto driving = render ? build_stitch(ctx, warp.source, warp.driving) : warp.driving;
        auto displacement =
            core::wrap_tensor(ggml_sub(ctx.ggml, warp.source.tensor, driving.tensor), warp.source.shape);
        displacement = RepeatModule({TensorShape::from_dims({21, 65536, 3})})
                           .build(ctx, core::reshape_tensor(ctx, displacement, TensorShape::from_dims({21, 1, 3})));
        auto sparse = AddModule().build(
            ctx, RepeatModule({TensorShape::from_dims({21, 65536, 3})}).build(ctx, warp.grid), displacement);
        sparse = ConcatModule({0}).build(ctx, warp.grid, sparse);
        auto deformed = sample_volume(ctx, compressed, sparse);
        auto heatmap =
            core::wrap_tensor(ggml_sub(ctx.ggml, gaussian(ctx, driving, warp.grid).tensor, source_heatmap.tensor),
                              TensorShape::from_dims({21, 65536, 1}));
        auto zero = SliceModule({2, 0, 1}).build(ctx, warp.background);
        heatmap = ConcatModule({0}).build(ctx, zero, heatmap);
        heatmap = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, heatmap),
                                       TensorShape::from_dims({22, 1, 65536}));
        deformed = core::reshape_tensor(ctx, deformed, TensorShape::from_dims({22, 4, 65536}));
        auto x = ConcatModule({1}).build(ctx, heatmap, deformed);
        x = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, x),
                                 TensorShape::from_dims({110, 16, 64, 64}));
        std::vector<TensorValue> skips{x};
        for (int layer = 0; layer < 5; ++layer) {
            x = normalized_convolution(
                ctx, x, "warp.dense_motion_network.hourglass.encoder.down_blocks." + std::to_string(layer), true);
            auto shape = x.shape;
            shape.dims[2] /= 2;
            shape.dims[3] /= 2;
            // Existing pooling consumes F32; restore the upstream F16 storage boundary afterwards.
            auto * pool_input =
                x.tensor->type == GGML_TYPE_F32 ? x.tensor : ggml_cast(ctx.ggml, x.tensor, GGML_TYPE_F32);
            x = core::wrap_tensor(ggml_pool_2d(ctx.ggml, pool_input, GGML_OP_POOL_AVG, 2, 2, 2, 2, 0, 0), shape);
            if (config.f16_activations)
                x = core::wrap_tensor(ggml_cast(ctx.ggml, x.tensor, GGML_TYPE_F16), shape, GGML_TYPE_F16);
            skips.push_back(x);
        }
        skips.pop_back();
        for (int layer = 0; layer < 5; ++layer) {
            x = upsample(ctx, x, x.shape.dims[2] * 2, x.shape.dims[3] * 2);
            x = normalized_convolution(
                ctx, x, "warp.dense_motion_network.hourglass.decoder.up_blocks." + std::to_string(layer), true);
            auto skip = skips.back();
            skips.pop_back();
            if (x.tensor->type != skip.tensor->type) {
                x = core::wrap_tensor(ggml_cast(ctx.ggml, x.tensor, GGML_TYPE_F32), x.shape);
                skip = core::wrap_tensor(ggml_cast(ctx.ggml, skip.tensor, GGML_TYPE_F32), skip.shape);
            }
            x = ConcatModule({0}).build(ctx, x, skip);
        }
        x = convolution(ctx, x, "warp.dense_motion_network.hourglass.decoder.conv", 1, -1, true);
        x = ReluModule().build(ctx, affine(ctx, x, "warp.dense_motion_network.hourglass.decoder.norm", true));
        auto mask = convolution(ctx, x, "warp.dense_motion_network.mask", 1, -1, true);
        mask = TransposeModule({{1, 0, 2, 3}, 2})
                   .build(ctx, core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, mask),
                                                    TensorShape::from_dims({22, 65536})));
        const auto mask_type = mask.tensor->type;
        mask = core::wrap_tensor(ggml_cast(ctx.ggml, mask.tensor, GGML_TYPE_F32), mask.shape);
        mask = SoftmaxModule().build(ctx, core::ensure_backend_addressable_layout(ctx, mask));
        if (mask_type == GGML_TYPE_F16) {
            auto * rounded = ggml_cast(ctx.ggml, mask.tensor, GGML_TYPE_F16);
            mask = core::wrap_tensor(ggml_cast(ctx.ggml, rounded, GGML_TYPE_F32), mask.shape);
        }
        mask = TransposeModule({{1, 0, 2, 3}, 2}).build(ctx, mask);
        mask = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, mask),
                                    TensorShape::from_dims({22, 65536, 1}));
        auto deformation = ReduceSumModule({0}).build(ctx, MulModule(true).build(ctx, sparse, mask));
        deformation = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, deformation),
                                           TensorShape::from_dims({1, 65536, 3}));
        auto output = sample_volume(ctx, warp.input, deformation);
        output = core::reshape_tensor(ctx, output, TensorShape::from_dims({1, 512, 64, 64}));
        output = normalized_convolution(ctx, output, "warp.third", false, .01f);
        output = convolution(ctx, output, "warp.fourth", 1, 0);
        auto occlusion = core::reshape_tensor(ctx, x, TensorShape::from_dims({1, x.shape.dims[0] * 16, 64, 64}));
        occlusion = SigmoidModule().build(ctx, convolution(ctx, occlusion, "warp.dense_motion_network.occlusion"));
        warp.output = MulModule(true).build(ctx, output, occlusion);
        if (render)
            warp.output = build_decoder(ctx, warp.output);
        else
            warp.output = core::wrap_tensor(ggml_cast(ctx.ggml, warp.output.tensor, GGML_TYPE_F32), warp.output.shape);
        ggml_set_output(warp.output.tensor);
        ggml_build_forward_expand(warp.owner->graph(), warp.output.tensor);
        warp.owner->allocate_persistent();
        std::vector<float> grid(65536 * 3);
        for (int z = 0; z < 16; ++z)
            for (int y = 0; y < 64; ++y)
                for (int x = 0; x < 64; ++x) {
                    const auto at = static_cast<size_t>(((z * 64 + y) * 64 + x) * 3);
                    grid[at] = 2.f * (static_cast<float>(x) / 63) - 1;
                    grid[at + 1] = 2.f * (static_cast<float>(y) / 63) - 1;
                    grid[at + 2] = 2.f * (static_cast<float>(z) / 15) - 1;
                }
        core::write_tensor_f32(warp.grid, grid);
        core::write_tensor_f32(warp.background, std::vector<float>(grid.size(), 0));
        warp.owner->allocate();
    }

    TensorValue build_stitch(core::ModuleBuildContext & ctx, const TensorValue & source,
                             const TensorValue & driving) const {
        auto x = ConcatModule({1}).build(ctx, core::reshape_tensor(ctx, source, TensorShape::from_dims({1, 63})),
                                         core::reshape_tensor(ctx, driving, TensorShape::from_dims({1, 63})));
        for (int layer = 0; layer <= 6; layer += 2) {
            const auto prefix = "stitch.mlp." + std::to_string(layer);
            const auto & weight = tensors.at(prefix + ".weight");
            x = LinearModule({weight.shape.dims[1], weight.shape.dims[0], true})
                    .build(ctx, x, {weight, tensors.at(prefix + ".bias")});
            if (layer < 6)
                x = ReluModule().build(ctx, x);
        }
        auto expression = core::reshape_tensor(
            ctx, core::ensure_backend_addressable_layout(ctx, SliceModule({1, 0, 63}).build(ctx, x)), driving.shape);
        auto translation = SliceModule({1, 63, 2}).build(ctx, x);
        auto xy = SliceModule({1, 0, 2}).build(ctx, expression);
        xy = AddModule(false, true).build(ctx, xy, translation);
        auto z = SliceModule({1, 2, 1}).build(ctx, expression);
        expression = ConcatModule({1}).build(ctx, xy, z);
        return AddModule().build(ctx, driving, expression);
    }

    void prepare_stitch() {
        if (stitch.owner)
            return;
        stitch.owner = std::make_unique<core::PreparedGraph>(
            execution, core::PreparedGraphConfig{"liveportrait.stitch", 2ull << 20, 512});
        auto & ctx = stitch.owner->build_context();
        stitch.source = core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({21, 3}));
        stitch.driving = core::make_tensor(ctx, GGML_TYPE_F32, stitch.source.shape);
        ggml_set_input(stitch.source.tensor);
        ggml_set_input(stitch.driving.tensor);
        stitch.output = build_stitch(ctx, stitch.source, stitch.driving);
        ggml_set_output(stitch.output.tensor);
        ggml_build_forward_expand(stitch.owner->graph(), stitch.output.tensor);
        stitch.owner->allocate();
    }

    enum class ConvNeXtVersion { V1, V2 };
    // ConvNeXt V2 uses GRN; the landmark V1 backbone uses layer scale.
    // Both use the same depthwise/norm/projection and spatial-pooling contracts.
    std::pair<TensorValue, TensorValue> build_convnext(core::ModuleBuildContext & ctx, const TensorValue & input,
                                                       const std::string & root, ConvNeXtVersion version,
                                                       bool use_cudnn, int tap_stage = -1) {
        auto norm = [&](const TensorValue & x, const std::string & name) {
            const auto channels = x.shape.dims[x.shape.rank - 1];
            return LayerNormModule({channels, 1e-6f, true, true})
                .build(ctx, x, {tensors.at(name + ".weight"), tensors.at(name + ".bias")});
        };
        auto linear = [&](const TensorValue & x, const std::string & name) {
            const auto & weight = tensors.at(name + ".weight");
            return LinearModule({weight.shape.dims[1], weight.shape.dims[0], true})
                .build(ctx, x, {weight, tensors.at(name + ".bias")});
        };
        TensorValue tapped;
        auto pool = [&](const TensorValue & value, const std::string & name) {
            auto nchw =
                core::ensure_backend_addressable_layout(ctx, TransposeModule({{0, 3, 1, 2}, 4}).build(ctx, value));
            auto flat = core::reshape_tensor(
                ctx, nchw, TensorShape::from_dims({nchw.shape.dims[1], nchw.shape.dims[2] * nchw.shape.dims[3]}));
            auto mean = ReduceMeanModule({1}).build(ctx, flat);
            return norm(core::reshape_tensor(ctx, mean, TensorShape::from_dims({1, mean.shape.dims[0]})), name);
        };
        auto x = input;
        for (int stage = 0; stage < 4; ++stage) {
            const auto down = root + "downsample_layers." + std::to_string(stage);
            if (stage) {
                x = core::ensure_backend_addressable_layout(ctx, TransposeModule({{0, 2, 3, 1}, 4}).build(ctx, x));
                x = norm(x, down + ".0");
                x = core::ensure_backend_addressable_layout(ctx, TransposeModule({{0, 3, 1, 2}, 4}).build(ctx, x));
            }
            x = convolution(ctx, x, down + (stage ? ".1" : ".0"), stage ? 2 : 4, 0, false, false, use_cudnn);
            x = core::ensure_backend_addressable_layout(ctx, TransposeModule({{0, 2, 3, 1}, 4}).build(ctx, x));
            if (!stage)
                x = norm(x, down + ".1");
            for (int block = 0;; ++block) {
                const auto prefix = root + "stages." + std::to_string(stage) + "." + std::to_string(block);
                if (!tensors.count(prefix + ".dwconv.weight"))
                    break;
                const auto residual = x;
                const auto channels = x.shape.dims[3];
                DepthwiseConv2dConfig dw;
                dw.channels = channels;
                dw.kernel_height = dw.kernel_width = 7;
                dw.padding_height = dw.padding_width = 3;
                x = DepthwiseConv2dModule(dw).build(
                    ctx, x, {tensors.at(prefix + ".dwconv.weight"), tensors.at(prefix + ".dwconv.bias")},
                    Conv2dLayout::ChannelsLast);
                if (version == ConvNeXtVersion::V2) {
                    x = linear(norm(x, prefix + ".norm"), prefix + ".pwconv1");
                    x = GeluModule().build(ctx, x);
                    // GRN reduces spatial energy per channel, then normalizes by channel mean.
                    auto flat = core::reshape_tensor(
                        ctx, core::ensure_backend_addressable_layout(ctx, x),
                        TensorShape::from_dims({x.shape.dims[1] * x.shape.dims[2], x.shape.dims[3]}));
                    auto energy =
                        SqrtModule().build(ctx, ReduceSumModule({0}).build(ctx, MulModule().build(ctx, flat, flat)));
                    auto denominator = ScaleBiasModule({1, 1e-6f}).build(ctx, ReduceMeanModule({1}).build(ctx, energy));
                    auto response = core::wrap_tensor(
                        ggml_div(ctx.ggml, energy.tensor, ggml_repeat(ctx.ggml, denominator.tensor, energy.tensor)),
                        energy.shape);
                    response = core::reshape_tensor(ctx, response, TensorShape::from_dims({1, 1, 1, x.shape.dims[3]}));
                    auto scaled = MulModule(true).build(ctx, x, response);
                    scaled = MulModule(true).build(ctx, scaled, tensors.at(prefix + ".grn.gamma"));
                    scaled = AddModule(false, true).build(ctx, scaled, tensors.at(prefix + ".grn.beta"));
                    x = AddModule().build(ctx, scaled, x);
                    x = AddModule().build(ctx, residual, linear(x, prefix + ".pwconv2"));
                } else {
                    FeedForwardConfig ffn;
                    ffn.hidden_size = channels;
                    ffn.intermediate_size = tensors.at(prefix + ".pwconv1.weight").shape.dims[0];
                    x = FeedForwardModule(ffn).build(
                        ctx, norm(x, prefix + ".norm"),
                        {tensors.at(prefix + ".pwconv1.weight"), tensors.at(prefix + ".pwconv1.bias"),
                         tensors.at(prefix + ".pwconv2.weight"), tensors.at(prefix + ".pwconv2.bias")});
                    x = MulModule(true).build(ctx, x, tensors.at(prefix + ".gamma"));
                    x = AddModule().build(ctx, residual, x);
                }
            }
            if (stage == tap_stage)
                tapped = pool(x, root + "norm_s3");
            if (stage != 3)
                x = core::ensure_backend_addressable_layout(ctx, TransposeModule({{0, 3, 1, 2}, 4}).build(ctx, x));
        }
        return {pool(x, root + "norm"), tapped};
    }

    void prepare_motion() {
        if (motion.owner)
            return;
        motion.owner = std::make_unique<core::PreparedGraph>(
            execution, core::PreparedGraphConfig{"liveportrait.motion", 32ull << 20, 16384});
        auto & ctx = motion.owner->build_context();
        motion.input = core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({1, 3, 256, 256}));
        ggml_set_input(motion.input.tensor);
        const std::string root = "motion.detector.";
        auto x = build_convnext(ctx, motion.input, root, ConvNeXtVersion::V2, true).first;
        auto linear = [&](const TensorValue & input, const std::string & name) {
            const auto & weight = tensors.at(name + ".weight");
            return LinearModule({weight.shape.dims[1], weight.shape.dims[0], true})
                .build(ctx, input, {weight, tensors.at(name + ".bias")});
        };
        const char * heads[] = {"pitch", "yaw", "roll", "t", "exp", "scale", "kp"};
        for (size_t i = 0; i < motion.output.size(); ++i) {
            motion.output[i] = linear(x, root + "fc_" + heads[i]);
            ggml_set_output(motion.output[i].tensor);
            ggml_build_forward_expand(motion.owner->graph(), motion.output[i].tensor);
        }
        motion.owner->allocate();
    }

    void prepare_landmarks() {
        if (landmarks.owner)
            return;
        landmarks.owner = std::make_unique<core::PreparedGraph>(
            execution, core::PreparedGraphConfig{"liveportrait.landmarks203", 32ull << 20, 16384});
        auto & ctx = landmarks.owner->build_context();
        landmarks.input = core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({1, 3, 224, 224}));
        ggml_set_input(landmarks.input.tensor);
        const auto features = build_convnext(ctx, landmarks.input, "landmarks.", ConvNeXtVersion::V1, false, 2);
        const auto joined = ConcatModule({1}).build(ctx, features.second, features.first);
        const auto & weight = tensors.at("landmarks.fc_pts.weight");
        landmarks.output = LinearModule({weight.shape.dims[1], weight.shape.dims[0], true})
                               .build(ctx, joined, {weight, tensors.at("landmarks.fc_pts.bias")});
        ggml_set_output(landmarks.output.tensor);
        ggml_build_forward_expand(landmarks.owner->graph(), landmarks.output.tensor);
        landmarks.owner->allocate();
    }

    void prepare_appearance() {
        if (appearance.owner)
            return;
        appearance.owner = std::make_unique<core::PreparedGraph>(
            execution, core::PreparedGraphConfig{"liveportrait.appearance", 16ull << 20, 8192});
        auto & ctx = appearance.owner->build_context();
        appearance.input = core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({1, 3, 256, 256}));
        ggml_set_input(appearance.input.tensor);
        auto x = normalized_convolution(ctx, appearance.input, "appearance.first");
        for (int layer = 0; tensors.count("appearance.down_blocks." + std::to_string(layer) + ".conv.weight");
             ++layer) {
            x = normalized_convolution(ctx, x, "appearance.down_blocks." + std::to_string(layer));
            auto shape = x.shape;
            shape.dims[2] /= 2;
            shape.dims[3] /= 2;
            // Existing pooling consumes F32; restore the upstream F16 storage boundary afterwards.
            auto * pool_input =
                x.tensor->type == GGML_TYPE_F32 ? x.tensor : ggml_cast(ctx.ggml, x.tensor, GGML_TYPE_F32);
            x = core::wrap_tensor(ggml_pool_2d(ctx.ggml, pool_input, GGML_OP_POOL_AVG, 2, 2, 2, 2, 0, 0), shape);
            if (config.f16_activations)
                x = core::wrap_tensor(ggml_cast(ctx.ggml, x.tensor, GGML_TYPE_F16), shape, GGML_TYPE_F16);
        }
        x = convolution(ctx, x, "appearance.second", 1, 0);
        x = core::reshape_tensor(ctx, x, TensorShape::from_dims({32, 16, 64, 64}));
        for (int layer = 0; tensors.count("appearance.resblocks_3d.3dr" + std::to_string(layer) + ".conv1.weight");
             ++layer) {
            const auto p = "appearance.resblocks_3d.3dr" + std::to_string(layer);
            auto y = convolution(ctx, ReluModule().build(ctx, affine(ctx, x, p + ".norm1", true)), p + ".conv1", 1, -1,
                                 true);
            y = convolution(ctx, ReluModule().build(ctx, affine(ctx, y, p + ".norm2", true)), p + ".conv2", 1, -1,
                            true);
            x = AddModule().build(ctx, x, y);
            if (config.f16_activations)
                x = core::wrap_tensor(ggml_cast(ctx.ggml, x.tensor, GGML_TYPE_F16), x.shape, GGML_TYPE_F16);
        }
        appearance.output = core::wrap_tensor(ggml_cast(ctx.ggml, x.tensor, GGML_TYPE_F32), x.shape);
        ggml_set_output(appearance.output.tensor);
        ggml_build_forward_expand(appearance.owner->graph(), appearance.output.tensor);
        appearance.owner->allocate();
    }

    TensorValue build_decoder(core::ModuleBuildContext & ctx, const TensorValue & input) const {
        auto x = convolution(ctx, input, "decoder.fc");
        for (int layer = 0; tensors.count("decoder.G_middle_" + std::to_string(layer) + ".conv_0.weight"); ++layer)
            x = spade_block(ctx, x, input, "decoder.G_middle_" + std::to_string(layer));
        for (int layer = 0; layer < 2; ++layer) {
            x = upsample(ctx, x, x.shape.dims[2] * 2, x.shape.dims[3] * 2);
            x = spade_block(ctx, x, input, "decoder.up_" + std::to_string(layer));
        }
        x = convolution(ctx, LeakyReluModule({.2f}).build(ctx, x), "decoder.conv_img.0");
        // PixelShuffle(2), N,C*r*r,H,W -> N,C,H,r,W,r -> N,C,H*r,W*r.
        auto * packed = ggml_reshape_4d(ctx.ggml, x.tensor, x.tensor->ne[0], x.tensor->ne[1], 2, 6);
        packed = ggml_cont(ctx.ggml, ggml_permute(ctx.ggml, packed, 1, 2, 0, 3));
        packed = ggml_reshape_4d(ctx.ggml, packed, 2 * x.tensor->ne[0], x.tensor->ne[1], 2, 3);
        packed = ggml_cont(ctx.ggml, ggml_permute(ctx.ggml, packed, 0, 2, 1, 3));
        x = core::wrap_tensor(ggml_reshape_4d(ctx.ggml, packed, 512, 512, 3, 1),
                              TensorShape::from_dims({1, 3, 512, 512}), x.type);
        x = SigmoidModule().build(ctx, x);
        return core::wrap_tensor(ggml_cast(ctx.ggml, x.tensor, GGML_TYPE_F32), x.shape);
    }

    void prepare_decoder() {
        if (decoder.owner)
            return;
        decoder.owner = std::make_unique<core::PreparedGraph>(
            execution, core::PreparedGraphConfig{"liveportrait.decoder", 32ull << 20, 16384});
        auto & ctx = decoder.owner->build_context();
        decoder.input = core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({1, 256, 64, 64}));
        ggml_set_input(decoder.input.tensor);
        decoder.output = build_decoder(ctx, decoder.input);
        ggml_set_output(decoder.output.tensor);
        ggml_build_forward_expand(decoder.owner->graph(), decoder.output.tensor);
        decoder.owner->allocate();
    }
};

LivePortraitRuntime::LivePortraitRuntime(const assets::TensorSource & source, core::ExecutionContext & execution,
                                         LivePortraitRuntimeConfig config)
    : impl_(std::make_unique<Impl>(source, execution, config)) {}
LivePortraitRuntime::~LivePortraitRuntime() = default;
std::vector<float> LivePortraitRuntime::stitch_keypoints(const std::vector<float> & source,
                                                         const std::vector<float> & driving) {
    impl_->prepare_stitch();
    core::write_tensor_f32(impl_->stitch.source, source);
    core::write_tensor_f32(impl_->stitch.driving, driving);
    impl_->stitch.owner->compute();
    return core::read_tensor_f32(impl_->stitch.output.tensor);
}
std::array<std::vector<float>, 7> LivePortraitRuntime::extract_motion(const std::vector<float> & rgb) {
    impl_->prepare_motion();
    core::write_tensor_f32(impl_->motion.input, rgb);
    impl_->motion.owner->compute();
    std::array<std::vector<float>, 7> result;
    for (size_t i = 0; i < result.size(); ++i)
        result[i] = core::read_tensor_f32(impl_->motion.output[i].tensor);
    return result;
}
std::vector<float> LivePortraitRuntime::extract_landmarks203(const std::vector<float> & rgb) {
    impl_->prepare_landmarks();
    core::write_tensor_f32(impl_->landmarks.input, rgb);
    impl_->landmarks.owner->compute();
    return core::read_tensor_f32(impl_->landmarks.output.tensor);
}
std::vector<float> LivePortraitRuntime::extract_appearance(const std::vector<float> & rgb) {
    impl_->prepare_appearance();
    core::write_tensor_f32(impl_->appearance.input, rgb);
    impl_->appearance.owner->compute();
    return core::read_tensor_f32(impl_->appearance.output.tensor);
}
std::vector<float> LivePortraitRuntime::decode_features(const std::vector<float> & features) {
    impl_->prepare_decoder();
    core::write_tensor_f32(impl_->decoder.input, features);
    impl_->decoder.owner->compute();
    return core::read_tensor_f32(impl_->decoder.output.tensor);
}
std::vector<float> LivePortraitRuntime::warp_features(const std::vector<float> & appearance,
                                                      const std::vector<float> & source_keypoints,
                                                      const std::vector<float> & driving_keypoints) {
    impl_->prepare_warp(impl_->warp);
    core::write_tensor_f32(impl_->warp.input, appearance);
    core::write_tensor_f32(impl_->warp.source, source_keypoints);
    core::write_tensor_f32(impl_->warp.driving, driving_keypoints);
    impl_->warp.owner->compute();
    return core::read_tensor_f32(impl_->warp.output.tensor);
}
void LivePortraitRuntime::set_source(const std::vector<float> & appearance,
                                     const std::vector<float> & source_keypoints) {
    auto & graph = impl_->renderer;
    impl_->prepare_warp(graph, true);
    impl_->source_ready = false;
    core::write_tensor_f32(graph.input, appearance);
    core::write_tensor_f32(graph.source, source_keypoints);
    graph.owner->compute(graph.source_graph);
    impl_->source_ready = true;
}
std::vector<float> LivePortraitRuntime::render_frame(const std::vector<float> & driving_keypoints) {
    if (!impl_->source_ready)
        throw std::logic_error("LivePortrait requires set_source before rendering");
    auto & graph = impl_->renderer;
    core::write_tensor_f32(graph.driving, driving_keypoints);
    graph.owner->compute();
    return core::read_tensor_f32(graph.output.tensor);
}
void LivePortraitRuntime::release_graphs() {
    impl_->source_ready = false;
    impl_->appearance = {};
    impl_->decoder = {};
    impl_->landmarks = {};
    impl_->warp = {};
    impl_->renderer = {};
    impl_->motion = {};
    impl_->stitch = {};
}
} // namespace engine::modules
