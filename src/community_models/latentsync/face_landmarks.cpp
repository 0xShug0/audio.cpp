#include "engine/community_models/latentsync/face_landmarks.h"

#include "engine/framework/io/json.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

namespace engine::models::latentsync {

LatentSyncLandmarkWeights load_landmark_weights(core::BackendWeightStore & store,
    const assets::TensorSource & source, const std::filesystem::path & config_path,
    assets::TensorStorageType type) {
    namespace binding = modules::binding;
    const auto config = io::json::parse_file(config_path);
    LatentSyncLandmarkWeights out;
    out.input_bias = store.make_from_f32(core::TensorShape::from_dims({1, 1, 1, 1}),
        assets::TensorStorageType::F32, {-static_cast<float>(config.require("input_offset").as_number())});
    out.input_scale = store.make_from_f32(core::TensorShape::from_dims({1, 1, 1, 1}),
        assets::TensorStorageType::F32, {static_cast<float>(config.require("input_scale").as_number())});
    for (const auto & item : config.require("layers").as_array()) {
        LatentSyncLandmarkLayer layer;
        auto & conv = layer.config;
        conv.in_channels = item.require("in_channels").as_i64();
        conv.out_channels = item.require("out_channels").as_i64();
        conv.kernel_height = item.require("kernel").as_array()[0].as_i64();
        conv.kernel_width = item.require("kernel").as_array()[1].as_i64();
        conv.stride_height = static_cast<int>(item.require("stride").as_array()[0].as_i64());
        conv.stride_width = static_cast<int>(item.require("stride").as_array()[1].as_i64());
        conv.padding_height = static_cast<int>(item.require("padding").as_array()[0].as_i64());
        conv.padding_width = static_cast<int>(item.require("padding").as_array()[1].as_i64());
        conv.dilation_height = static_cast<int>(item.require("dilation").as_array()[0].as_i64());
        conv.dilation_width = static_cast<int>(item.require("dilation").as_array()[1].as_i64());
        layer.depthwise = item.require("groups").as_i64() != 1;
        const auto prefix = "face.landmarks.layers." + std::to_string(out.layers.size());
        layer.convolution = binding::conv2d_from_source(store, source, prefix + ".conv", type,
            conv.out_channels, layer.depthwise ? 1 : conv.in_channels, conv.kernel_height, conv.kernel_width, true);
        layer.slope = store.load_tensor(source, prefix + ".slope", assets::TensorStorageType::F32,
            {1, conv.out_channels, 1, 1});
        layer.positive_scale = store.load_tensor(source, prefix + ".positive_scale", assets::TensorStorageType::F32,
            {1, conv.out_channels, 1, 1});
        out.layers.push_back(std::move(layer));
    }
    out.output = binding::linear_from_source(store, source, "face.landmarks.output", type, 212, 576, true);
    return out;
}

core::TensorValue LatentSyncDepthwiseCnnLandmarks::build(core::ModuleBuildContext & ctx,
    const core::TensorValue & image, const LatentSyncLandmarkWeights & weights) const {
    using namespace modules;
    core::validate_rank_between(image, 4, 4, "LatentSync landmark image");
    core::validate_shape(image, core::TensorShape::from_dims({image.shape.dims[0], 3, 192, 192}), "LatentSync landmark image");
    auto x = AddModule().build(ctx, image, RepeatModule({image.shape}).build(ctx, weights.input_bias));
    x = MulModule().build(ctx, x, RepeatModule({x.shape}).build(ctx, weights.input_scale));
    for (const auto & layer : weights.layers) {
        const auto & c = layer.config;
        if (layer.depthwise) {
            x = DepthwiseConv2dModule({c.in_channels, c.kernel_height, c.kernel_width,
                c.stride_height, c.stride_width, c.padding_height, c.padding_width,
                c.dilation_height, c.dilation_width, true}).build(ctx, x, layer.convolution);
        } else {
            x = Conv2dModule(c).build(ctx, x, layer.convolution);
        }
        // PReLU(x) = slope*x + (1-slope)*ReLU(x).
        const auto negative_scaled = MulModule().build(ctx, x, RepeatModule({x.shape}).build(ctx, layer.slope));
        const auto positive = MulModule().build(ctx, ReluModule().build(ctx, x),
            RepeatModule({x.shape}).build(ctx, layer.positive_scale));
        x = AddModule().build(ctx, negative_scaled, positive);
    }
    x = core::reshape_tensor(ctx, x, core::TensorShape::from_dims({image.shape.dims[0], 576}));
    return LinearModule({576, 212, true}).build(ctx, x, weights.output);
}

}  // namespace engine::models::latentsync
